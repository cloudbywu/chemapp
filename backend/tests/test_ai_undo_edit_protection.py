"""A whole-result AI undo must never cross a non-AI edit boundary."""
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import threading

import pytest
from fastapi.testclient import TestClient

import app.api.store as store_module
from app.api.store import AIUndoConflict, PersistentStore, RevisionConflict
from app.main import app
from tests.test_ai_action_transactions import _commit, _result, _state
from tests.test_ai_action_transactions import seeded as seeded


def _generation(store, sid):
    with sqlite3.connect(store._db_path) as conn:
        return conn.execute("SELECT non_ai_revision FROM spectra WHERE id=?", (sid,)).fetchone()[0]


@pytest.mark.parametrize("write", ["manual", "reanalysis", "restore", "same_result"])
def test_every_non_ai_result_replacement_blocks_undo_even_without_manual_metadata(seeded, write):
    store, sid = seeded
    saved = _commit(store, sid)
    generation = _generation(store, sid)
    current = store.get(sid)
    if write == "manual":
        store.set_result_and_version(sid, _result("manual"), expected_revision=2)
    elif write == "reanalysis":
        fresh = _result("fresh analysis")
        fresh.metrics = {}
        store.set_result(sid, fresh, expected_revision=2)
    elif write == "restore":
        # Restoring the very same AI post-image must still create an edit boundary.
        store.set_result_and_version(
            sid, store.get_result_version(sid, saved.version).result,
            expected_revision=2, restore_from_version=saved.version,
        )
    else:
        store.set_result_and_version(sid, current.result, expected_revision=2)
    assert _generation(store, sid) == generation + 1
    before = _state(store)
    # Reopen independently: no in-memory flag or caller-provided metadata suffices.
    with pytest.raises(AIUndoConflict) as exc:
        PersistentStore(store._db_path).undo_last_ai_action(sid, expected_revision=3)
    assert exc.value.reason == "later_result_edit"
    assert exc.value.current_revision == 3
    assert _state(store) == before


def test_undo_new_ai_actions_preserves_manual_result_and_stops_at_earlier_action(seeded):
    store, sid = seeded
    first = _commit(store, sid)
    manual = _result("manual review between AI actions")
    store.set_result_and_version(sid, manual, version_metric="manual_version", expected_revision=2)
    generation = _generation(store, sid)
    second = _commit(store, sid, "AI after manual review", expected_revision=3)
    third = _commit(store, sid, "another AI action", expected_revision=4)
    assert store.undo_last_ai_action(sid, expected_revision=5).action_id == third.action_id
    assert store.undo_last_ai_action(sid, expected_revision=6).action_id == second.action_id
    restored = store.get(sid).result
    assert restored.summary == manual.summary
    assert restored.metrics["manual_version"] == manual.metrics["manual_version"]
    assert _generation(store, sid) == generation
    before = _state(store)
    with pytest.raises(AIUndoConflict, match="later edits"):
        store.undo_last_ai_action(sid, expected_revision=7)
    assert _state(store) == before
    assert store.get_last_ai_action(sid)["id"] == first.action_id


def test_snapshot_only_history_does_not_block_undo(seeded):
    store, sid = seeded
    saved = _commit(store, sid)
    store.save_result_version(sid, _result("snapshot only"), expected_revision=2)
    assert store.undo_last_ai_action(sid, expected_revision=2).action_id == saved.action_id
    assert store.get(sid).result.summary == "original"


def test_legacy_non_atomic_action_record_cannot_claim_a_verified_generation(seeded):
    store, sid = seeded
    action_id = store.record_ai_action(sid, "delete_peak", {}, 1, 1)
    before = _state(store)
    with pytest.raises(AIUndoConflict) as exc:
        store.undo_last_ai_action(sid, expected_revision=1)
    assert exc.value.reason == "unverifiable_history"
    assert _state(store) == before
    assert store.get_last_ai_action(sid)["id"] == action_id


def test_legacy_migration_preserves_data_and_does_not_backfill_action_provenance(seeded):
    store, sid = seeded
    _commit(store, sid)
    store.set_result_and_version(sid, _result("legacy later manual review"), expected_revision=2)
    # A representative prior-generation DB, including current results, versions,
    # and an AI action that cannot prove the absence of a later manual save.
    with sqlite3.connect(store._db_path) as conn:
        conn.execute("ALTER TABLE spectra DROP COLUMN non_ai_revision")
        conn.execute("ALTER TABLE ai_action_history DROP COLUMN non_ai_revision")
        columns = {table: [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]
                   for table in ("spectra", "result_versions", "ai_action_history")}
        before = {table: conn.execute(f"SELECT {','.join(names)} FROM {table}").fetchall()
                  for table, names in columns.items()}
    # Starting two workers together must not duplicate ALTER TABLE operations.
    ready = threading.Barrier(2)

    def migrate(_):
        ready.wait(timeout=10)
        return PersistentStore(store._db_path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(migrate, range(2)))
    migrated = PersistentStore(store._db_path)
    with sqlite3.connect(store._db_path) as conn:
        after = {table: conn.execute(f"SELECT {','.join(names)} FROM {table}").fetchall()
                 for table, names in columns.items()}
        assert conn.execute("SELECT non_ai_revision FROM ai_action_history").fetchall() == [(None,)]
    assert after == before
    unchanged = _state(migrated)
    with pytest.raises(AIUndoConflict) as exc:
        migrated.undo_last_ai_action(sid, expected_revision=3)
    assert exc.value.reason == "unverifiable_history"
    assert _state(migrated) == unchanged
    assert migrated.get(sid).result.summary == "legacy later manual review"
    assert len(migrated.list_result_versions(sid)) == 4
    # Newly verified actions on that same migrated data still support undo.
    latest = _commit(migrated, sid, expected_revision=3)
    assert migrated.undo_last_ai_action(sid, expected_revision=4).action_id == latest.action_id
    assert migrated.get(sid).result.summary == "legacy later manual review"
    with pytest.raises(AIUndoConflict):
        migrated.undo_last_ai_action(sid, expected_revision=5)


@pytest.mark.parametrize("reason", ["later_result_edit", "unverifiable_history"])
def test_api_returns_structured_conflict_and_keeps_result_and_history(seeded, monkeypatch, reason):
    store, sid = seeded
    _commit(store, sid)
    monkeypatch.setattr(store_module, "_store", store)
    monkeypatch.setenv("CHEMAPP_ACCESS_TOKEN", "undo-test-access")
    monkeypatch.setenv("CHEMAPP_ADMIN_TOKEN", "undo-test-admin")
    with TestClient(app, headers={
        "X-ChemApp-Access-Token": "undo-test-access", "X-ChemApp-Admin-Token": "undo-test-admin",
    }) as client:
        revision = 2
        if reason == "later_result_edit":
            manual = client.put(f"/api/results/{sid}/manual", json={
                "expected_revision": 2, "summary": "Manual result must survive",
                "metrics": {"non_ai_revision": 0, "ai_undo": {}, "manual_version": 0},
            })
            assert manual.status_code == 200, manual.text
            revision = 3
        else:
            with sqlite3.connect(store._db_path) as conn:
                conn.execute("UPDATE ai_action_history SET non_ai_revision=NULL")
        args = {"spectrum_id": sid, "expected_revision": revision}
        preview = client.post("/api/ai/actions/preview", json={"name": "undo_last_ai_action", "args": args})
        assert preview.status_code == 200, preview.text
        before = _state(store)
        response = client.post("/api/ai/actions/execute", json={
            "name": "undo_last_ai_action", "args": args,
            "preview_token": preview.json()["preview_token"],
        })
        assert response.status_code == 409, response.text
        assert response.json()["detail"] == {
            "code": "ai_undo_blocked", "reason": reason, "current_revision": revision,
            "message": str(AIUndoConflict(reason, revision)),
        }
        assert _state(store) == before


@pytest.mark.parametrize("manual_first", [True, False])
def test_manual_save_and_undo_are_serialized_across_independent_connections(seeded, monkeypatch, manual_first):
    store, sid = seeded
    _commit(store, sid)
    manual_store, undo_store = PersistentStore(store._db_path), PersistentStore(store._db_path)
    first = manual_store if manual_first else undo_store
    acquired, release, second_attempted = threading.Event(), threading.Event(), threading.Event()

    class PausingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            result = super().execute(sql, parameters)
            if sql == "BEGIN IMMEDIATE":
                acquired.set()
                assert release.wait(timeout=10)
            return result

    monkeypatch.setattr(first, "_connect", lambda: sqlite3.connect(
        store._db_path, timeout=10, factory=PausingConnection,
    ))

    def manual():
        return manual_store.set_result_and_version(sid, _result("racing manual edit"), expected_revision=2)

    def undo():
        return undo_store.undo_last_ai_action(sid, expected_revision=2)

    def second():
        second_attempted.set()
        return undo() if manual_first else manual()

    with ThreadPoolExecutor(max_workers=2) as pool:
        winner = pool.submit(manual if manual_first else undo)
        assert acquired.wait(timeout=10)
        loser = pool.submit(second)
        assert second_attempted.wait(timeout=10)
        release.set()
        winner.result(timeout=15)
        with pytest.raises(RevisionConflict):
            loser.result(timeout=15)
    current = store.get(sid)
    assert current.result_revision == 3
    if manual_first:
        assert current.result.summary == "racing manual edit"
        assert store.get_last_ai_action(sid) is not None
        before = _state(store)
        with pytest.raises(AIUndoConflict):
            undo_store.undo_last_ai_action(sid, expected_revision=3)
        assert _state(store) == before
    else:
        assert current.result.summary == "original"
        assert store.get_last_ai_action(sid) is None
        # A manual save refreshed/retried after undo remains a normal safe edit.
        manual_store.set_result_and_version(sid, _result("retried manual edit"), expected_revision=3)
        assert store.get(sid).result.summary == "retried manual edit"
