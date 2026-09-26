"""Route-layer fixes: manual-save metric whitelist, summary marker,
batch error sanitization, paged batch endpoints, request bounds, id typing.

All tests use tmp_path-isolated stores.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import app.api.store as store_module
from app.api.routes.nmr import NMRProcessRequest, _finite_float
from app.core.models import Spectrum, Technique
from app.main import app


def _spectrum() -> Spectrum:
    return Spectrum(
        technique=Technique.NMR,
        x_data=np.asarray([2.0, 1.0, 0.0]),
        y_data=np.asarray([0.0, 1.0, 0.0]),
        source_file="sample.jdf",
    )


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.delenv("CHEMAPP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
    store_module._store = None
    return TestClient(app)


def test_manual_save_drops_non_whitelisted_metric_keys(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        sid = store_module.get_store().add(_spectrum()).id
        initial = client.post(f"/api/analyze/{sid}", json={"expected_revision": 0})
        assert initial.status_code == 200, initial.text
        revision = initial.json()["result_revision"]

        saved = client.put(
            f"/api/results/{sid}/manual",
            json={
                "peaks": [],
                "metrics": {
                    # Provenance/server-authoritative keys: must be dropped.
                    "ai_modified": True,
                    "restored_from_version": 7,
                    "manual_version": 99,
                    "quality": {"status": "good", "score": 100},
                    # Unknown keys: dropped as well.
                    "custom_note": "tamper",
                    # Analyzer measurement keys stay client-writable.
                    "noise_level": 0.5,
                },
                "expected_revision": revision,
            },
        )
        assert saved.status_code == 200, saved.text
        metrics = saved.json()["metrics"]
        assert "ai_modified" not in metrics
        assert "restored_from_version" not in metrics
        assert "custom_note" not in metrics
        assert metrics["noise_level"] == 0.5
        # Server-authoritative values are computed, not taken from the client.
        assert metrics["manual_confirmed"] is True
        assert metrics["manual_version"] == 1
        assert metrics["n_peaks"] == 0
        assert metrics["quality"]["score"] != 100


def test_manual_save_appends_confirmation_marker_once(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        sid = store_module.get_store().add(_spectrum()).id
        analyzed = client.post(f"/api/analyze/{sid}", json={"expected_revision": 0})
        assert analyzed.status_code == 200, analyzed.text

        first = client.put(
            f"/api/results/{sid}/manual",
            json={"expected_revision": 1},
        )
        assert first.status_code == 200, first.text
        summary = first.json()["summary"]
        assert summary.count("Manual review confirmed.") == 1

        second = client.put(
            f"/api/results/{sid}/manual",
            json={"expected_revision": 2},
        )
        assert second.status_code == 200, second.text
        assert second.json()["summary"].count("Manual review confirmed.") == 1
        assert second.json()["summary"] == summary

        # An explicit client summary is used verbatim, without a marker.
        custom = client.put(
            f"/api/results/{sid}/manual",
            json={"summary": "Analyst override", "expected_revision": 3},
        )
        assert custom.status_code == 200, custom.text
        assert custom.json()["summary"] == "Analyst override"


def test_analyze_batch_error_is_sanitized(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        sid = store_module.get_store().add(_spectrum()).id

        def boom(stored, options=None):
            raise RuntimeError("internal secret detail")

        monkeypatch.setattr("app.api.routes.analysis._analyze_stored", boom)
        response = client.post(
            "/api/analyze/batch",
            json={"ids": [sid], "expected_revisions": {sid: 0}},
        )
        assert response.status_code == 200, response.text
        errors = response.json()["errors"]
        assert errors[0]["id"] == sid
        assert errors[0]["error"] == "Analysis failed for this spectrum"
        assert "secret" not in errors[0]["error"]


def test_batch_quality_pages_over_metadata(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        store = store_module.get_store()
        for _ in range(3):
            store.add(_spectrum())

        def forbidden_list_all(*args, **kwargs):
            pytest.fail("paged batch endpoints must not call store.list_all")

        monkeypatch.setattr(store, "list_all", forbidden_list_all)

        paged = client.post("/api/quality/batch?limit=2&offset=0", json={})
        assert paged.status_code == 200, paged.text
        data = paged.json()
        assert len(data["items"]) == 2
        assert set(data["counts"]) == {"good", "review", "poor"}
        page_ids = [item["id"] for item in data["items"]]
        assert page_ids == [m["id"] for m in store.list_metadata(limit=2, offset=0)]

        # Selecting the same ids explicitly yields the same rows.
        explicit = client.post("/api/quality/batch", json={"ids": page_ids})
        assert explicit.status_code == 200, explicit.text
        assert [item["id"] for item in explicit.json()["items"]] == page_ids

        too_large = client.post("/api/quality/batch?limit=5001", json={})
        assert too_large.status_code == 422


def test_batch_workbench_pages_over_metadata(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        store = store_module.get_store()
        for _ in range(3):
            store.add(_spectrum())

        def forbidden_list_all(*args, **kwargs):
            pytest.fail("paged batch endpoints must not call store.list_all")

        monkeypatch.setattr(store, "list_all", forbidden_list_all)

        paged = client.post("/api/batch/workbench?limit=2&offset=1", json={})
        assert paged.status_code == 200, paged.text
        body = paged.json()
        assert len(body["items"]) == 2
        assert body["summary"]["count"] == 2
        assert body["summary"]["technique_counts"] == {"NMR": 2}
        assert body["summary"]["total_points"] == 6
        page_ids = [item["id"] for item in body["items"]]
        assert page_ids == [m["id"] for m in store.list_metadata(limit=2, offset=1)]


def test_nmr_smoothness_request_bound_matches_runtime_cap():
    accepted = NMRProcessRequest(baseline_correct=True, baseline_smoothness=1e14)
    assert accepted.baseline_smoothness == 1e14
    with pytest.raises(ValueError):
        NMRProcessRequest(baseline_correct=True, baseline_smoothness=1e15)
    with pytest.raises(HTTPException) as exc_info:
        _finite_float(
            {"baseline_smoothness": 1e15},
            "baseline_smoothness",
            minimum=1.0,
            maximum=1e14,
        )
    assert exc_info.value.status_code == 422


def test_standards_match_requires_typed_spectrum_id(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        invalid = client.post("/api/standards/match/not..valid", json={})
        assert invalid.status_code == 422

        missing = client.post("/api/standards/match/doesnotexist", json={})
        assert missing.status_code == 404
