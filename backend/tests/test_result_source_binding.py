from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.store as store_module
from app.analysis.models import NMRAnalysisResult
from app.api.store import PersistentStore, ResultVersionSourceConflict, RevisionConflict
from app.core.models import Peak, Spectrum, Technique
from app.main import app


def _spectrum():
    return Spectrum(
        technique=Technique.NMR,
        x_data=np.linspace(10, 0, 32),
        y_data=np.linspace(0, 1, 32),
    )


def _result():
    return NMRAnalysisResult(
        technique=Technique.NMR,
        peaks=[Peak(position=7.0, intensity=1.0)],
        metrics={"manual_confirmed": True},
        summary="Reviewed source result",
    )


@pytest.fixture
def source_client(tmp_path, monkeypatch):
    store = PersistentStore(str(tmp_path / "source.db"))
    monkeypatch.setattr(store_module, "_store", store)
    sid = store.add(_spectrum()).id
    store.set_result_and_version(sid, _result(), expected_revision=0)
    with TestClient(app) as client:
        yield store, sid, client


def _change_spectrum(store, sid):
    stored = store.get(sid)
    stored.spectrum.x_data += 1
    return store.set_spectrum(sid, stored.spectrum, expected_revision=stored.spectrum_revision)


def test_same_source_restore_remains_supported(source_client):
    store, sid, client = source_client
    response = client.post(
        f"/api/results/{sid}/versions/1/restore", json={"expected_revision": 1}
    )
    assert response.status_code == 200, response.text
    assert response.json()["result_revision"] == 2
    history = store.list_result_versions(sid)
    assert len(history) == 2
    assert all(row["spectrum_revision"] == 1 and row["restorable"] for row in history)


def test_processed_spectrum_cannot_restore_older_source_result(source_client):
    store, sid, client = source_client
    _change_spectrum(store, sid)
    response = client.post(
        f"/api/results/{sid}/versions/1/restore", json={"expected_revision": 2}
    )
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "result_source_mismatch"
    assert detail["source_spectrum_revision"] == 1
    assert detail["current_spectrum_revision"] == 2
    assert store.get(sid).result is None
    assert store.get(sid).result_revision == 2
    assert store.list_result_versions(sid)[0]["restorable"] is False


def test_ai_undo_rejects_old_source_without_marking_action_undone(source_client):
    store, sid, client = source_client
    store.record_ai_action(sid, "delete_peak", {}, 1, 1)
    _change_spectrum(store, sid)
    replacement = _result()
    replacement.summary = "New spectrum analysis"
    store.set_result(sid, replacement, expected_revision=2)
    args = {"spectrum_id": sid, "expected_revision": 3}
    preview = client.post(
        "/api/ai/actions/preview", json={"name": "undo_last_ai_action", "args": args}
    )
    assert preview.status_code == 200, preview.text
    response = client.post(
        "/api/ai/actions/execute",
        json={"name": "undo_last_ai_action", "args": args, "preview_token": preview.json()["preview_token"]},
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "result_source_mismatch"
    assert store.get(sid).result.summary == "New spectrum analysis"
    assert store.get(sid).result_revision == 3
    assert store.get_last_ai_action(sid) is not None


def test_restore_rechecks_result_revision_after_loading_snapshot(source_client, monkeypatch):
    store, sid, client = source_client
    original = store.get_result_version

    def change_after_read(spectrum_id, version):
        saved = original(spectrum_id, version)
        _change_spectrum(store, sid)
        return saved

    monkeypatch.setattr(store, "get_result_version", change_after_read)
    response = client.post(
        f"/api/results/{sid}/versions/1/restore", json={"expected_revision": 1}
    )
    assert response.status_code == 409, response.text
    assert response.json()["detail"]["code"] == "revision_conflict"
    assert store.get(sid).result is None
    assert len(store.list_result_versions(sid)) == 1


def test_source_revision_guard_is_inside_write_transaction(source_client):
    store, sid, _ = source_client
    saved = store.get_result_version(sid, 1)
    current = _change_spectrum(store, sid)
    with pytest.raises(ResultVersionSourceConflict):
        store.set_result_and_version(
            sid, saved.result, expected_revision=current.result_revision, restore_from_version=1
        )
    assert store.get(sid).result is None
    assert len(store.list_result_versions(sid)) == 1


def test_stale_ai_backup_cannot_acquire_new_spectrum_provenance(source_client):
    store, sid, _ = source_client
    old = store.get(sid)
    _change_spectrum(store, sid)
    with pytest.raises(RevisionConflict):
        store.save_result_version(sid, old.result, expected_revision=old.result_revision)
    assert len(store.list_result_versions(sid)) == 1


@pytest.mark.parametrize("operation", ["process", "reset"])
def test_processing_cannot_clear_a_concurrently_saved_manual_result(source_client, monkeypatch, operation):
    store, sid, client = source_client
    before = store.get(sid)
    original = store.set_spectrum

    def save_manual_before_processing_commit(spectrum_id, spectrum, *args, **kwargs):
        edited = _result()
        edited.summary = "Concurrent manual review"
        current = store.get(spectrum_id)
        store.set_result_and_version(
            spectrum_id, edited, expected_revision=current.result_revision
        )
        return original(spectrum_id, spectrum, *args, **kwargs)

    monkeypatch.setattr(store, "set_spectrum", save_manual_before_processing_commit)
    payload = {
        "expected_revision": before.spectrum_revision,
        "expected_result_revision": before.result_revision,
    }
    if operation == "process":
        payload["invert"] = True
    response = client.post(f"/api/nmr/{sid}/{operation}", json=payload)
    assert response.status_code == 409, response.text
    detail = response.json()["detail"]
    assert detail["code"] == "revision_conflict"
    assert detail["current_revision"] == before.spectrum_revision
    assert detail["current_result_revision"] == before.result_revision + 1
    assert "analysis result changed" in detail["message"]
    current = store.get(sid)
    assert current.spectrum_revision == before.spectrum_revision
    assert current.result.summary == "Concurrent manual review"
    assert current.result_revision == before.result_revision + 1
    np.testing.assert_array_equal(current.spectrum.x_data, before.spectrum.x_data)
    np.testing.assert_array_equal(current.spectrum.y_data, before.spectrum.y_data)
    assert len(store.list_result_versions(sid)) == 2


@pytest.mark.parametrize("operation", ["preview", "process", "reset"])
@pytest.mark.parametrize("revision", [None, 1])
def test_processing_requires_client_revision_for_unseen_manual_save(source_client, operation, revision):
    store, sid, client = source_client
    manual = _result()
    manual.summary = "Manual review saved in another client"
    store.set_result_and_version(sid, manual, expected_revision=1)
    before = store.get(sid)
    history = store.list_result_versions(sid)
    payload = {"expected_revision": before.spectrum_revision}
    if revision is not None:
        payload["expected_result_revision"] = revision
    if operation != "reset":
        payload.update(invert=True, preview_only=operation == "preview")
    route = "process" if operation == "preview" else operation
    response = client.post(f"/api/nmr/{sid}/{route}", json=payload)
    assert response.status_code == (428 if revision is None else 409), response.text
    assert response.json()["detail"]["current_result_revision"] == 2
    current = store.get(sid)
    assert current.result.to_dict() == before.result.to_dict()
    assert current.result_revision == before.result_revision
    assert current.spectrum_revision == before.spectrum_revision
    np.testing.assert_array_equal(current.spectrum.x_data, before.spectrum.x_data)
    np.testing.assert_array_equal(current.spectrum.y_data, before.spectrum.y_data)
    assert store.list_result_versions(sid) == history


def test_manual_save_after_preview_blocks_apply_with_preview_revisions(source_client):
    store, sid, client = source_client
    payload = {"expected_revision": 1, "expected_result_revision": 1, "invert": True}
    preview = client.post(f"/api/nmr/{sid}/process", json={**payload, "preview_only": True})
    assert preview.status_code == 200, preview.text
    assert preview.json()["result_revision"] == 1
    manual = _result()
    manual.summary = "Reviewed after preview"
    store.set_result_and_version(sid, manual, expected_revision=1)
    response = client.post(f"/api/nmr/{sid}/process", json=payload)
    assert response.status_code == 409, response.text
    assert store.get(sid).result.summary == manual.summary
    assert store.get(sid).spectrum_revision == 1
    assert all(row["restorable"] for row in store.list_result_versions(sid))


@pytest.mark.parametrize("operation", ["process", "reset"])
def test_current_client_revisions_allow_confirmed_processing(source_client, operation):
    store, sid, client = source_client
    payload = {"expected_revision": 1, "expected_result_revision": 1}
    if operation == "process":
        payload["invert"] = True
    response = client.post(f"/api/nmr/{sid}/{operation}", json=payload)
    assert response.status_code == 200, response.text
    current = store.get(sid)
    assert current.result is None
    assert (current.spectrum_revision, current.result_revision) == (2, 2)
    assert len(store.list_result_versions(sid)) == 1


@pytest.mark.parametrize("operation", ["process", "reset"])
def test_legacy_empty_result_processing_still_guards_concurrent_first_save(source_client, monkeypatch, operation):
    store, _, client = source_client
    sid = store.add(_spectrum()).id
    original = store.set_spectrum

    def save_first_result_before_processing(*args, **kwargs):
        store.set_result_and_version(sid, _result(), expected_revision=0)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "set_spectrum", save_first_result_before_processing)
    payload = {"expected_revision": 1}
    if operation == "process":
        payload["invert"] = True
    response = client.post(f"/api/nmr/{sid}/{operation}", json=payload)
    assert response.status_code == 409, response.text
    assert store.get(sid).result.summary == "Reviewed source result"
    assert store.get(sid).spectrum_revision == 1


def test_legacy_migration_preserves_history_without_inventing_source(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    payload = json.dumps(_result().to_dict())
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE result_versions (
            id INTEGER PRIMARY KEY, spectrum_id TEXT NOT NULL,
            version INTEGER NOT NULL, result_json TEXT NOT NULL,
            note TEXT DEFAULT '', created_at TEXT DEFAULT (datetime('now')),
            UNIQUE(spectrum_id, version))""")
        conn.execute(
            "INSERT INTO result_versions (spectrum_id,version,result_json,note) VALUES (?,?,?,?)",
            ("legacy1", 1, payload, "Original review note"),
        )
    store = PersistentStore(str(path))
    # Populate the old parent row after schema initialization, using only this
    # isolated test database. The migration must not associate its version.
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO spectra (id,technique,spectrum_json,result_json,result_revision) VALUES (?,?,?,?,?)",
            ("legacy1", "NMR", json.dumps(_spectrum().to_dict()), payload, 1),
        )
    monkeypatch.setattr(store_module, "_store", store)
    with TestClient(app) as client:
        history = client.get("/api/results/legacy1/versions").json()["versions"]
        assert len(history) == 1
        assert history[0]["note"] == "Original review note"
        assert history[0]["spectrum_revision"] is None
        assert history[0]["restorable"] is False
        response = client.post(
            "/api/results/legacy1/versions/1/restore", json={"expected_revision": 1}
        )
        assert response.status_code == 409, response.text
        assert "no verifiable source" in response.json()["detail"]["message"]
    assert store.get("legacy1").result_revision == 1
    assert store.get_result_version("legacy1", 1).result.summary == "Reviewed source result"
    store.set_result_and_version("legacy1", _result(), expected_revision=1)
    assert store.list_result_versions("legacy1")[0]["restorable"] is True
    assert store.list_result_versions("legacy1")[1]["restorable"] is False
