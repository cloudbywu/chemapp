from __future__ import annotations

import io
import json
import sqlite3
import threading
import time
import zipfile

import numpy as np
import pytest
from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import app.api.store as store_module
from app.api.security import is_local_request
from app.ai import actions as ai_actions
from app.ai.experiment_report_agent import collect_data_context
from app.ai import llm_client
from app.analysis.models import AnalysisResult
from app.analysis.nmr_processing import build_processing_source
from app.api.routes import elucidate as elucidate_routes
from app.api.routes import ml as ml_routes
from app.api.routes.upload import _safe_extract_zip
from app.api.store import PersistentStore, RevisionConflict
from app.core.models import Peak, Spectrum, Technique
from app.main import app
from app.standards.database import StandardDatabase


def _spectrum() -> Spectrum:
    return Spectrum(
        technique=Technique.NMR,
        x_data=np.asarray([2.0, 1.0, 0.0]),
        y_data=np.asarray([0.0, 1.0, 0.0]),
        source_file="sample.jdf",
    )


def _result(summary: str = "result") -> AnalysisResult:
    return AnalysisResult(
        technique=Technique.NMR,
        peaks=[
            Peak(
                position=1.23,
                intensity=4.5,
                multiplicity="d",
                coupling_constant=7.2,
            )
        ],
        metrics={"ok": True},
        summary=summary,
    )


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.delenv("CHEMAPP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
    store_module._store = None
    return TestClient(app)


def test_access_token_protects_all_api_but_liveness(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.setenv("CHEMAPP_ACCESS_TOKEN", "access-secret")
    monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
    store_module._store = None
    with TestClient(app) as client:
        assert client.get("/api/live").status_code == 200
        assert client.get("/api/ready").status_code == 200
        assert client.get("/api/health").status_code == 401
        assert client.get(
            "/api/health",
            headers={"Authorization": "Bearer access-secret"},
        ).status_code == 200
        assert client.get(
            "/api/spectra",
            headers={"X-ChemApp-Access-Token": "wrong"},
        ).status_code == 401


def test_health_reports_asset_status(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.setenv("CHEMAPP_ACCESS_TOKEN", "access-secret")
    monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
    index_v2 = tmp_path / "nmr_spectral_index_v2.sqlite"
    index_v2.write_bytes(b"placeholder")
    monkeypatch.setenv("CHEMAPP_NMR_INDEX_V2", str(index_v2))
    store_module._store = None
    with TestClient(app) as client:
        response = client.get(
            "/api/health",
            headers={"Authorization": "Bearer access-secret"},
        )

    assert response.status_code == 200
    data = response.json()
    assert "assets" in data
    assert data["assets"]["csp5_weights"]["status"] == "ok"
    assert data["assets"]["calibration_policy"]["probability_claim_allowed"] is False
    assert data["assets"]["nmr_index_v2"]["path"] == str(index_v2)
    assert data["assets"]["nmr_index_v2"]["exists"] is True


def test_admin_token_is_distinct_from_access_token(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.setenv("CHEMAPP_ACCESS_TOKEN", "reader")
    monkeypatch.setenv("CHEMAPP_ADMIN_TOKEN", "administrator")
    store_module._store = None
    sid = store_module.get_store().add(_spectrum()).id
    with TestClient(app) as client:
        denied = client.delete(
            f"/api/spectra/{sid}",
            params={"expected_spectrum_revision": 1, "expected_result_revision": 0},
            headers={"X-ChemApp-Access-Token": "reader"},
        )
        assert denied.status_code == 401
        allowed = client.delete(
            f"/api/spectra/{sid}",
            params={"expected_spectrum_revision": 1, "expected_result_revision": 0},
            headers={"X-ChemApp-Admin-Token": "administrator"},
        )
        assert allowed.status_code == 200


def test_ranker_job_prune_keeps_running_and_new_terminal_jobs(monkeypatch):
    now = 1_000_000.0
    jobs = {
        "running": {"status": "started", "started_at": now - 10},
        "old-done": {"status": "done", "finished_at": now - 100_000},
        "new-error": {"status": "error", "finished_at": now - 10},
    }
    monkeypatch.setattr(elucidate_routes, "_ranker_jobs", jobs)
    monkeypatch.setattr(elucidate_routes.time, "time", lambda: now)
    monkeypatch.setattr(elucidate_routes, "_RANKER_JOB_MAX_AGE_SECONDS", 3600)

    elucidate_routes._prune_ranker_jobs()

    assert set(jobs) == {"running", "new-error"}


def test_stale_training_lock_detection(tmp_path, monkeypatch):
    from app.api.routes.ml import _stale_training_lock

    monkeypatch.setattr(
        ml_routes,
        "_process_alive",
        lambda pid: pid != 2_147_483_647,
    )
    lock = tmp_path / "training.lock"
    lock.write_text(
        json.dumps({"job_id": "j", "pid": 2_147_483_647}),
        encoding="utf-8",
    )
    assert _stale_training_lock(lock) is True

    lock.write_text(
        json.dumps({"job_id": "j", "pid": __import__("os").getpid()}),
        encoding="utf-8",
    )
    assert _stale_training_lock(lock) is False


def test_example_load_requires_admin_token(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.setenv("CHEMAPP_ACCESS_TOKEN", "reader")
    monkeypatch.setenv("CHEMAPP_ADMIN_TOKEN", "administrator")
    store_module._store = None
    payload = {"path": "紫外example/LA.txt"}
    with TestClient(app) as client:
        denied = client.post(
            "/api/spectra/examples/load",
            json=payload,
            headers={"X-ChemApp-Access-Token": "reader"},
        )
        assert denied.status_code == 401
        assert store_module.get_store().count() == 0

        allowed = client.post(
            "/api/spectra/examples/load",
            json=payload,
            headers={"X-ChemApp-Admin-Token": "administrator"},
        )
        assert allowed.status_code == 200, allowed.text
        assert store_module.get_store().count() == 1


def test_result_revision_conflict_and_atomic_version(tmp_path):
    store = PersistentStore(str(tmp_path / "store.db"))
    sid = store.add(_spectrum()).id
    first = store.set_result(sid, _result("first"), expected_revision=0)
    assert first is not None
    assert first.result_revision == 1

    with pytest.raises(RevisionConflict) as conflict:
        store.set_result(sid, _result("stale"), expected_revision=0)
    assert conflict.value.current_revision == 1

    saved = store.set_result_and_version(
        sid,
        _result("second"),
        note="manual",
        expected_revision=1,
    )
    assert saved is not None
    updated, version = saved
    assert updated.result_revision == 2

    with sqlite3.connect(store._db_path) as conn:
        current_json = conn.execute(
            "SELECT result_json FROM spectra WHERE id=?",
            (sid,),
        ).fetchone()[0]
        version_json = conn.execute(
            "SELECT result_json FROM result_versions WHERE spectrum_id=? AND version=?",
            (sid, version),
        ).fetchone()[0]
    assert current_json == version_json


def test_nonfinite_result_is_rejected_without_partial_write(tmp_path):
    store = PersistentStore(str(tmp_path / "store.db"))
    sid = store.add(_spectrum()).id
    first = store.set_result(sid, _result("safe"))
    assert first is not None
    invalid = _result("invalid")
    invalid.metrics["bad"] = float("nan")

    with pytest.raises(ValueError):
        store.set_result(sid, invalid, expected_revision=1)

    persisted = store.get(sid)
    assert persisted is not None
    assert persisted.result_revision == 1
    assert persisted.result is not None
    assert persisted.result.summary == "safe"


def test_peak_fields_roundtrip_and_delete_cascades(tmp_path):
    store = PersistentStore(str(tmp_path / "store.db"))
    sid = store.add(_spectrum()).id
    store.set_result_and_version(sid, _result(), note="version")
    store.record_ai_action(sid, "delete_peak", {}, 1, 2)

    loaded = store.get(sid)
    assert loaded is not None and loaded.result is not None
    assert loaded.result.peaks[0].multiplicity == "d"
    assert loaded.result.peaks[0].coupling_constant == pytest.approx(7.2)
    assert store.remove(sid)

    with sqlite3.connect(store._db_path) as conn:
        assert conn.execute(
            "SELECT COUNT(*) FROM result_versions WHERE spectrum_id=?",
            (sid,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM ai_action_history WHERE spectrum_id=?",
            (sid,),
        ).fetchone()[0] == 0


def test_processing_source_is_persisted_but_not_public(tmp_path):
    store = PersistentStore(str(tmp_path / "store.db"))
    spectrum = _spectrum()
    spectrum.parameters["processing_source"] = build_processing_source(
        spectrum.x_data,
        spectrum.y_data,
        source_kind="processed_spectrum",
        source_domain="frequency",
    )
    sid = store.add(spectrum).id

    with sqlite3.connect(store._db_path) as conn:
        persisted = json.loads(
            conn.execute(
                "SELECT spectrum_json FROM spectra WHERE id=?",
                (sid,),
            ).fetchone()[0]
        )
    assert "processing_source" in persisted["parameters"]
    public = store.get(sid).spectrum.to_dict(include_internal=False)
    assert "processing_source" not in public["parameters"]
    assert "processing_source_summary" in public["parameters"]


def test_manual_result_conflict_returns_current_revision(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        store = store_module.get_store()
        sid = store.add(_spectrum()).id
        stored = store.set_result(sid, _result())
        assert stored is not None

        response = client.put(
            f"/api/results/{sid}/manual",
            json={
                "peaks": [],
                "metrics": {},
                "summary": "stale edit",
                "expected_revision": 0,
            },
        )
        assert response.status_code == 409
        assert response.json()["detail"]["current_revision"] == 1
        fetched = client.get(f"/api/results/{sid}")
        assert fetched.status_code == 200
        assert fetched.json()["result_revision"] == 1


def test_analysis_writes_fail_closed_without_or_with_stale_revision(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch) as client:
        sid = store_module.get_store().add(_spectrum()).id

        missing = client.post(f"/api/analyze/{sid}", json={})
        assert missing.status_code == 428
        assert missing.json()["detail"]["code"] == "expected_revision_required"
        assert store_module.get_store().get(sid).result is None

        initial = client.post(
            f"/api/analyze/{sid}", json={"expected_revision": 0}
        )
        assert initial.status_code == 200, initial.text
        assert initial.json()["result_revision"] == 1

        stale = client.post(
            f"/api/analyze/{sid}", json={"expected_revision": 0}
        )
        assert stale.status_code == 409
        assert stale.json()["detail"]["current_revision"] == 1
        assert store_module.get_store().get(sid).result_revision == 1

        manual_missing = client.put(
            f"/api/results/{sid}/manual",
            json={"summary": "must not overwrite"},
        )
        assert manual_missing.status_code == 428
        assert store_module.get_store().get(sid).result_revision == 1


def test_batch_analysis_uses_per_spectrum_revision_preconditions(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch) as client:
        store = store_module.get_store()
        first = store.add(_spectrum()).id
        second = store.add(_spectrum()).id

        missing = client.post(
            "/api/analyze/batch",
            json={"ids": [first, second]},
        )
        assert missing.status_code == 200
        assert [item["error"] for item in missing.json()["errors"]] == [
            "expected_revision_required",
            "expected_revision_required",
        ]
        assert store.get(first).result is None
        assert store.get(second).result is None

        created = client.post(
            "/api/analyze/batch",
            json={
                "ids": [first, second],
                "expected_revisions": {first: 0, second: 0},
            },
        )
        assert created.status_code == 200, created.text
        assert created.json()["errors"] == []
        assert store.get(first).result_revision == 1
        assert store.get(second).result_revision == 1

        partial = client.post(
            "/api/analyze/batch",
            json={
                "ids": [first, second],
                "expected_revisions": {first: 0, second: 1},
            },
        )
        assert partial.status_code == 200, partial.text
        assert partial.json()["errors"] == [
            {
                "id": first,
                "error": "revision_conflict",
                "message": "The result changed after it was loaded",
                "current_revision": 1,
            }
        ]
        assert store.get(first).result_revision == 1
        assert store.get(second).result_revision == 2


def test_destructive_ai_actions_require_bound_preview_token(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch) as client:
        store = store_module.get_store()
        sid = store.add(_spectrum()).id
        stored = store.set_result(sid, _result(), expected_revision=0)
        assert stored is not None
        args = {
            "spectrum_id": sid,
            "index": 0,
            "expected_revision": stored.result_revision,
        }

        missing_revision = client.post(
            "/api/ai/actions/preview",
            json={
                "name": "delete_peak",
                "args": {"spectrum_id": sid, "index": 0},
            },
        )
        assert missing_revision.status_code == 400

        preview = client.post(
            "/api/ai/actions/preview",
            json={"name": "delete_peak", "args": args},
        )
        assert preview.status_code == 200, preview.text
        payload = preview.json()
        assert payload["expected_revision"] == 1
        assert payload["preview_token"]

        missing_token = client.post(
            "/api/ai/actions/execute",
            json={"name": "delete_peak", "args": args},
        )
        assert missing_token.status_code == 400

        tampered = client.post(
            "/api/ai/actions/execute",
            json={
                "name": "delete_peak",
                "args": {**args, "index": 1},
                "preview_token": payload["preview_token"],
            },
        )
        assert tampered.status_code == 400

        executed = client.post(
            "/api/ai/actions/execute",
            json={
                "name": "delete_peak",
                "args": args,
                "preview_token": payload["preview_token"],
            },
        )
        assert executed.status_code == 200, executed.text
        assert executed.json()["result"]["result_revision"] == 2

        replay = client.post(
            "/api/ai/actions/execute",
            json={
                "name": "delete_peak",
                "args": args,
                "preview_token": payload["preview_token"],
            },
        )
        assert replay.status_code == 409
        assert replay.json()["detail"]["current_revision"] == 2


def test_preview_token_expires(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        store = store_module.get_store()
        sid = store.add(_spectrum()).id
        store.set_result(sid, _result(), expected_revision=0)
        args = {"spectrum_id": sid, "index": 0, "expected_revision": 1}
        now = 1_700_000_000
        monkeypatch.setattr(ai_actions.time, "time", lambda: now)
        preview = client.post(
            "/api/ai/actions/preview",
            json={"name": "delete_peak", "args": args},
        )
        assert preview.status_code == 200, preview.text
        monkeypatch.setattr(ai_actions.time, "time", lambda: now + 301)
        expired = client.post(
            "/api/ai/actions/execute",
            json={
                "name": "delete_peak",
                "args": args,
                "preview_token": preview.json()["preview_token"],
            },
        )
        assert expired.status_code == 400
        assert "expired" in expired.text


def test_nmr_commits_and_delete_require_revision_preconditions(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch) as client:
        store = store_module.get_store()
        sid = store.add(_spectrum()).id
        store.set_result(sid, _result(), expected_revision=0)

        process_missing = client.post(
            f"/api/nmr/{sid}/process", json={"invert": True}
        )
        assert process_missing.status_code == 428
        reset_missing = client.post(f"/api/nmr/{sid}/reset", json={})
        assert reset_missing.status_code == 428

        delete_missing = client.delete(f"/api/spectra/{sid}")
        assert delete_missing.status_code == 422
        delete_stale = client.delete(
            f"/api/spectra/{sid}",
            params={"expected_spectrum_revision": 1, "expected_result_revision": 0},
        )
        assert delete_stale.status_code == 409
        assert delete_stale.json()["detail"]["current_result_revision"] == 1
        assert store.get(sid) is not None
        deleted = client.delete(
            f"/api/spectra/{sid}",
            params={"expected_spectrum_revision": 1, "expected_result_revision": 1},
        )
        assert deleted.status_code == 200, deleted.text
        assert store.get(sid) is None


def test_read_only_ai_helpers_do_not_persist_transient_analysis(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch):
        store = store_module.get_store()
        first = store.add(_spectrum()).id
        second = store.add(_spectrum()).id

        ai_actions.suggest_ai_actions([first])
        collect_data_context([second])

        assert store.get(first).result is None
        assert store.get(first).result_revision == 0
        assert store.get(second).result is None
        assert store.get(second).result_revision == 0


def test_model_import_rejects_client_filesystem_path(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/ml/elucidate/index/import",
            json={"source": "retrieval_db_v2", "path": r"C:\untrusted\payload.pt"},
        )
        assert response.status_code == 422


def test_llm_custom_url_never_receives_server_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "server-secret")
    with pytest.raises(ValueError, match="client-supplied API key"):
        llm_client._resolve(None, None, "https://8.8.8.8/v1")

    monkeypatch.setenv("CHEMAPP_LLM_ALLOWED_HOSTS", "127.0.0.1")
    with pytest.raises(ValueError, match="not allowed"):
        llm_client._resolve(None, "user-secret", "https://127.0.0.1/v1")

    captured = {}
    monkeypatch.setenv("CHEMAPP_LLM_ALLOWED_HOSTS", "provider.example")
    monkeypatch.setattr(
        llm_client.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [(2, 1, 6, "", ("8.8.8.8", 443))],
    )
    monkeypatch.setattr(
        llm_client,
        "_build_client",
        lambda key, url: captured.update(key=key, url=url) or object(),
    )
    llm_client._resolve("model", "user-secret", "https://provider.example/v1")
    assert captured == {
        "key": "user-secret",
        "url": "https://provider.example/v1",
    }


def test_zip_rejects_duplicate_members_and_zip_bomb(tmp_path):
    duplicate = io.BytesIO()
    with zipfile.ZipFile(duplicate, "w") as archive:
        archive.writestr("sample/acqu", "first")
        archive.writestr("SAMPLE/ACQU", "second")
    duplicate.seek(0)
    with zipfile.ZipFile(duplicate) as archive:
        with pytest.raises(HTTPException, match="Duplicate path"):
            _safe_extract_zip(archive, tmp_path / "duplicate")

    compressed = io.BytesIO()
    with zipfile.ZipFile(compressed, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("sample/fid", b"\0" * (1024 * 1024))
    compressed.seek(0)
    with zipfile.ZipFile(compressed) as archive:
        with pytest.raises(HTTPException, match="compression ratio"):
            _safe_extract_zip(archive, tmp_path / "compressed")


def test_user_standards_survive_database_recreation(tmp_path):
    path = str(tmp_path / "store.db")
    first = StandardDatabase(path)
    saved = first.add_record(
        {
            "name": "Local reference",
            "technique": "NMR",
            "formula": "C2H6O",
            "peaks": [{"shift": 1.2}],
        }
    )
    second = StandardDatabase(path)
    records = second.list_records(query="Local reference")
    assert [record["id"] for record in records] == [saved["id"]]


def test_training_lock_is_process_safe(tmp_path, monkeypatch):
    lock_path = tmp_path / "training.lock"
    monkeypatch.setenv("CHEMAPP_TRAINING_LOCK_PATH", str(lock_path))
    first = ml_routes._acquire_training_lock("job-one")
    assert first == lock_path
    assert ml_routes._acquire_training_lock("job-two") is None
    ml_routes._release_training_lock(lock_path, "job-one")
    assert ml_routes._acquire_training_lock("job-two") == lock_path
    ml_routes._release_training_lock(lock_path, "job-two")


def test_uncaught_exception_returns_sanitized_500(
    tmp_path, monkeypatch, caplog
):
    @app.get("/api/hardening-boom-probe")
    def _boom():
        raise RuntimeError("database password hunter2 must never leak")

    route = next(
        item
        for item in app.router.routes
        if isinstance(item, APIRoute) and item.path == "/api/hardening-boom-probe"
    )
    try:
        # ServerErrorMiddleware always re-raises after running the installed
        # 500 handler, so the client must not turn it back into a test error.
        monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
        monkeypatch.delenv("CHEMAPP_ACCESS_TOKEN", raising=False)
        monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
        store_module._store = None
        with TestClient(app, raise_server_exceptions=False) as client:
            with caplog.at_level("ERROR", logger="chemapp.main"):
                response = client.get("/api/hardening-boom-probe")
    finally:
        app.router.routes.remove(route)

    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"] == "Internal server error"
    assert payload["request_id"]
    assert "hunter2" not in response.text
    assert response.headers["X-Request-ID"] == payload["request_id"]
    assert payload["request_id"] in caplog.text


def test_security_headers_present_on_api_responses(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_ACCESS_TOKEN", "access-secret")
    monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
    store_module._store = None
    with TestClient(app) as client:
        allowed = client.get("/api/live")
        denied = client.get("/api/health")

    assert denied.status_code == 401
    for response in (allowed, denied):
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["X-Frame-Options"] == "DENY"
        assert response.headers["Referrer-Policy"] == "no-referrer"


def test_cors_preflight_allows_only_whitelisted_methods_and_headers(
    tmp_path, monkeypatch
):
    with _client(tmp_path, monkeypatch) as client:
        preflight = client.options(
            "/api/spectra",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert preflight.status_code == 200
        allowed_methods = {
            method.strip()
            for method in preflight.headers["access-control-allow-methods"].split(",")
        }
        assert allowed_methods == {
            "GET",
            "POST",
            "PUT",
            "PATCH",
            "DELETE",
            "OPTIONS",
        }
        allowed_headers = {
            header.strip()
            for header in preflight.headers["access-control-allow-headers"].split(",")
        }
        # Starlette automatically merges the CORS-safelisted headers into
        # every allow-headers reply; the app whitelist is the rest.
        assert allowed_headers == {
            "Accept",
            "Accept-Language",
            "Content-Language",
            "Authorization",
            "Content-Type",
            "X-Request-ID",
            "X-ChemApp-Access-Token",
            "X-ChemApp-Admin-Token",
            "X-ChemApp-Reviewer-Token",
        }

        whitelisted = client.options(
            "/api/spectra",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": (
                    "authorization, content-type, x-request-id, "
                    "x-chemapp-access-token, x-chemapp-admin-token, "
                    "x-chemapp-reviewer-token"
                ),
            },
        )
        assert whitelisted.status_code == 200

        for method in ("TRACE", "PURGE"):
            rejected_method = client.options(
                "/api/spectra",
                headers={
                    "Origin": "http://localhost:3000",
                    "Access-Control-Request-Method": method,
                },
            )
            assert rejected_method.status_code == 400

        rejected_headers = client.options(
            "/api/spectra",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "x-not-whitelisted",
            },
        )
        assert rejected_headers.status_code == 400


def _request_with(peer: str, host: str | None) -> Request:
    headers = [(b"host", host.encode("ascii"))] if host else []
    return Request(
        scope={
            "type": "http",
            "method": "GET",
            "path": "/api/live",
            "query_string": b"",
            "headers": headers,
            "client": (peer, 51_000),
        }
    )


def test_is_local_request_trusts_peer_ip_over_host_header():
    assert is_local_request(_request_with("127.0.0.1", "localhost")) is True
    assert is_local_request(_request_with("::1", "localhost")) is True
    # A same-host reverse proxy may rewrite Host to a public name; the
    # loopback peer still counts as local.
    assert is_local_request(_request_with("127.0.0.1", "public.example.com")) is True
    # A remote peer is never local, even with a spoofed loopback Host header.
    assert is_local_request(_request_with("192.0.2.10", "localhost")) is False
    assert is_local_request(_request_with("192.0.2.10", "public.example.com")) is False
    assert is_local_request(_request_with("192.0.2.10", None)) is False


def test_get_store_concurrent_init_returns_single_instance(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    store_module._store = None
    real_init = store_module.PersistentStore.__init__

    def slowed_init(self, *args, **kwargs):
        time.sleep(0.02)
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(store_module.PersistentStore, "__init__", slowed_init)
    results: list = [None] * 8

    def worker(index: int) -> None:
        results[index] = store_module.get_store()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert all(result is results[0] for result in results)
    assert store_module.get_store() is results[0]


def test_review_subject_overlap_fails_closed_for_reviewer_and_admin(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.setenv(
        "CHEMAPP_REVIEWER_TOKENS",
        json.dumps({"alice": "alice-review-token-0001"}),
    )
    monkeypatch.setenv("CHEMAPP_REVIEW_ADMIN_SUBJECT", "alice")
    monkeypatch.delenv("CHEMAPP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
    store_module._store = None

    reviewer_headers = {"X-ChemApp-Reviewer-Token": "alice-review-token-0001"}
    with TestClient(app, headers=reviewer_headers) as client:
        reviewer_blocked = client.get("/api/reviews/me")
        assert reviewer_blocked.status_code == 503
        assert "overlaps" in reviewer_blocked.json()["detail"]

    with TestClient(app) as client:
        admin_blocked = client.get("/api/reviews/gold-manifest")
        assert admin_blocked.status_code == 503
        assert "overlaps" in admin_blocked.json()["detail"]

    monkeypatch.setenv("CHEMAPP_REVIEW_ADMIN_SUBJECT", "curation-admin")
    with TestClient(app, headers=reviewer_headers) as client:
        assert client.get("/api/reviews/me").status_code == 200
