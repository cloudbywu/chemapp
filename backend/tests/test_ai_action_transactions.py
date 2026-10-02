"""AI writes remain atomic across failures and independent SQLite workers."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import sqlite3
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.ai.actions as actions
import app.api.store as store_module
from app.analysis.models import NMRAnalysisResult
from app.api.store import AIUndoConflict, PersistentStore, ResultVersionSourceConflict, RevisionConflict
from app.core.models import Peak, Spectrum, Technique
from app.main import app


def _result(summary="original"):
    return NMRAnalysisResult(
        technique=Technique.NMR,
        peaks=[Peak(position=7.0, intensity=1.0), Peak(position=2.0, intensity=0.5)],
        metrics={"manual_confirmed": True}, summary=summary,
    )


@pytest.fixture
def seeded(tmp_path):
    store = PersistentStore(str(tmp_path / "actions.db"))
    sid = store.add(Spectrum(
        technique=Technique.NMR,
        x_data=np.linspace(10, 0, 32), y_data=np.linspace(0, 1, 32),
    )).id
    store.set_result_and_version(sid, _result(), expected_revision=0)
    return store, sid


def _commit(store, sid, summary="AI change", expected_revision=1):
    return store.commit_ai_action(
        sid, _result(summary), "delete_peak", {"spectrum_id": sid, "index": 0},
        note="AI test change", expected_revision=expected_revision,
        expected_spectrum_revision=1,
    )


def _state(store):
    with sqlite3.connect(store._db_path) as conn:
        return {
            table: conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
            for table in ("spectra", "result_versions", "ai_action_history")
        }


def _fail_at(store, operation, table, condition="1"):
    with sqlite3.connect(store._db_path) as conn:
        conn.execute(f"""CREATE TRIGGER injected_failure BEFORE {operation} ON {table}
            WHEN {condition}
            BEGIN SELECT RAISE(ABORT, 'injected write failure'); END""")


def test_ai_commit_captures_persisted_preimage_and_matching_audit(seeded):
    store, sid = seeded
    store.set_result(sid, _result("new persisted analysis"), expected_revision=1)
    saved = _commit(store, sid, expected_revision=2)
    assert (saved.previous_version, saved.version, saved.result_revision) == (2, 3, 3)
    assert store.get_result_version(sid, 2).result.to_dict() == _result("new persisted analysis").to_dict()
    assert store.get_result_version(sid, 3).result.to_dict() == _result("AI change").to_dict()
    assert store.get(sid).result.summary == "AI change"
    assert store.get(sid).result_revision == 3
    last = store.get_last_ai_action(sid)
    assert (last["id"], last["previous_version"], last["new_version"]) == (saved.action_id, 2, 3)
    assert last["action_name"] == "delete_peak"
    assert last["args"] == {"spectrum_id": sid, "index": 0}
    assert all(row["spectrum_revision"] == 1 for row in store.list_result_versions(sid))


@pytest.mark.parametrize(("operation", "table", "condition"), [
    ("INSERT", "result_versions", "NEW.note LIKE 'Before AI action%'"),
    ("UPDATE", "spectra", "1"),
    ("INSERT", "result_versions", "NEW.note = 'AI test change'"),
    ("INSERT", "ai_action_history", "1"),
])
def test_ai_commit_rolls_back_every_stage(seeded, operation, table, condition):
    store, sid = seeded
    before = _state(store)
    _fail_at(store, operation, table, condition)
    with pytest.raises(sqlite3.IntegrityError, match="injected write failure"):
        _commit(store, sid)
    assert _state(store) == before
    with sqlite3.connect(store._db_path) as conn:
        conn.execute("DROP TRIGGER injected_failure")
    assert _commit(PersistentStore(store._db_path), sid).result_revision == 2


@pytest.mark.parametrize(("operation", "table"), [
    ("UPDATE", "spectra"), ("INSERT", "result_versions"), ("UPDATE", "ai_action_history"),
])
def test_undo_rolls_back_restore_version_and_history_together(seeded, operation, table):
    store, sid = seeded
    saved = _commit(store, sid)
    before = _state(store)
    _fail_at(store, operation, table)
    with pytest.raises(sqlite3.IntegrityError, match="injected write failure"):
        store.undo_last_ai_action(sid, expected_revision=saved.result_revision)
    assert _state(store) == before
    with sqlite3.connect(store._db_path) as conn:
        conn.execute("DROP TRIGGER injected_failure")
    restored = PersistentStore(store._db_path).undo_last_ai_action(sid, expected_revision=2)
    assert restored.action_id == saved.action_id
    assert store.get(sid).result.summary == "original"
    assert store.get_last_ai_action(sid) is None


@pytest.mark.parametrize("bad_field", ["result", "args"])
def test_nonfinite_serialization_does_not_start_partial_history(seeded, bad_field):
    store, sid = seeded
    result, args = _result(), {"spectrum_id": sid}
    if bad_field == "result":
        result.metrics["bad"] = float("nan")
    else:
        args["bad"] = float("inf")
    before = _state(store)
    with pytest.raises(ValueError):
        store.commit_ai_action(
            sid, result, "delete_peak", args,
            expected_revision=1, expected_spectrum_revision=1,
        )
    assert _state(store) == before


def test_stale_commit_has_no_orphaned_preimage_or_audit(seeded):
    store, sid = seeded
    store.set_result_and_version(sid, _result("manual save"), expected_revision=1)
    before = _state(store)
    with pytest.raises(RevisionConflict) as exc:
        _commit(PersistentStore(store._db_path), sid)
    assert exc.value.current_revision == 2
    assert _state(store) == before


def test_commit_rejects_source_revision_even_when_result_revision_matches(seeded):
    store, sid = seeded
    current = store.get(sid)
    store.set_spectrum(sid, current.spectrum, clear_result=False, expected_revision=1)
    before = _state(store)
    with pytest.raises(ResultVersionSourceConflict):
        _commit(store, sid)
    assert _state(store) == before


@pytest.mark.parametrize("undo", [False, True])
def test_independent_connections_only_apply_same_revision_once(seeded, undo):
    store, sid = seeded
    if undo:
        _commit(store, sid)
    workers = [PersistentStore(store._db_path), PersistentStore(store._db_path)]
    ready = threading.Barrier(2)

    def run(index):
        ready.wait(timeout=10)
        try:
            if undo:
                return workers[index].undo_last_ai_action(sid, expected_revision=2)
            return _commit(workers[index], sid, f"worker {index}")
        except RevisionConflict as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, range(2)))
    assert sum(isinstance(item, RevisionConflict) for item in results) == 1
    assert store.get(sid).result_revision == (3 if undo else 2)
    assert len(store.list_result_versions(sid)) == (4 if undo else 3)
    assert len(_state(store)["ai_action_history"]) == 1
    assert store.get_result_version(sid, 2).result.summary == "original"
    if undo:
        assert store.get(sid).result.summary == "original"
        assert store.get_last_ai_action(sid) is None
    else:
        assert store.get_result_version(sid, 3).result.summary == store.get(sid).result.summary


def _process_write(path, sid, index, undo, ready, release, output):
    try:
        worker = PersistentStore(path)
        ready.put(index)
        if not release.wait(timeout=20):
            raise TimeoutError("worker release timed out")
        try:
            saved = (worker.undo_last_ai_action(sid, expected_revision=2) if undo
                     else _commit(worker, sid, f"process {index}"))
            output.put(("ok", saved.result_revision))
        except RevisionConflict as exc:
            output.put(("conflict", exc.current_revision))
    except Exception as exc:
        output.put(("error", str(exc)))


@pytest.mark.parametrize("undo", [False, True])
def test_separate_processes_cannot_both_apply_same_revision(seeded, undo):
    store, sid = seeded
    if undo:
        _commit(store, sid)
    context = multiprocessing.get_context("spawn")
    ready, output = context.Queue(), context.Queue()
    release = context.Event()
    workers = [context.Process(
        target=_process_write,
        args=(store._db_path, sid, index, undo, ready, release, output),
    ) for index in range(2)]
    try:
        for worker in workers:
            worker.start()
        assert sorted(ready.get(timeout=30) for _ in workers) == [0, 1]
        release.set()
        revision = 3 if undo else 2
        assert sorted(output.get(timeout=30) for _ in workers) == [
            ("conflict", revision), ("ok", revision),
        ]
        for worker in workers:
            worker.join(timeout=20)
            assert worker.exitcode == 0
    finally:
        release.set()
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.kill()
                    worker.join(timeout=5)
        ready.close()
        output.close()
    assert store.get(sid).result_revision == (3 if undo else 2)
    assert len(store.list_result_versions(sid)) == (4 if undo else 3)
    assert len(_state(store)["ai_action_history"]) == 1


def test_repeated_actions_undo_in_reverse_order_without_losing_history(seeded):
    store, sid = seeded
    first = _commit(store, sid, "first change")
    second = _commit(store, sid, "second change", expected_revision=2)
    assert (first.previous_version, first.version) == (2, 3)
    assert (second.previous_version, second.version) == (4, 5)
    assert store.get_result_version(sid, 4).result.summary == "first change"
    restored = store.undo_last_ai_action(sid, expected_revision=3)
    assert restored.action_id == second.action_id
    assert store.get(sid).result.summary == "first change"
    assert store.get_last_ai_action(sid)["id"] == first.action_id
    restored = store.undo_last_ai_action(sid, expected_revision=4)
    assert restored.action_id == first.action_id
    assert store.get(sid).result.summary == "original"
    assert store.get(sid).result_revision == 5
    assert store.get_last_ai_action(sid) is None
    assert store.undo_last_ai_action(sid, expected_revision=5) is None
    assert [row["version"] for row in store.list_result_versions(sid)] == list(range(7, 0, -1))


def test_action_api_uses_atomic_path_and_keeps_preview_replay_guard(seeded, monkeypatch):
    store, sid = seeded
    monkeypatch.setattr(store_module, "_store", store)
    monkeypatch.setenv("CHEMAPP_ACCESS_TOKEN", "transaction-test-access")
    monkeypatch.setenv("CHEMAPP_ADMIN_TOKEN", "transaction-test-admin")
    headers = {
        "X-ChemApp-Access-Token": "transaction-test-access",
        "X-ChemApp-Admin-Token": "transaction-test-admin",
    }

    def reject_old_path(*args, **kwargs):
        raise AssertionError("AI actions must not use separate persistence steps")

    for method in ("save_result_version", "set_result_and_version", "record_ai_action", "mark_ai_action_undone"):
        monkeypatch.setattr(store, method, reject_old_path)
    with TestClient(app, headers=headers) as client:
        args = {"spectrum_id": sid, "index": 0, "expected_revision": 1}
        preview = client.post("/api/ai/actions/preview", json={"name": "delete_peak", "args": args})
        assert preview.status_code == 200, preview.text
        payload = {"name": "delete_peak", "args": args, "preview_token": preview.json()["preview_token"]}
        response = client.post("/api/ai/actions/execute", json=payload)
        assert response.status_code == 200, response.text
        saved = response.json()["result"]
        assert (saved["previous_version"], saved["version"]) == (2, 3)
        assert saved["ai_action_id"] == store.get_last_ai_action(sid)["id"]
        assert len(store.get(sid).result.peaks) == 1
        before = _state(store)
        replay = client.post("/api/ai/actions/execute", json=payload)
        assert replay.status_code == 409, replay.text
        assert _state(store) == before
        args = {"spectrum_id": sid, "expected_revision": 2}
        preview = client.post("/api/ai/actions/preview", json={"name": "undo_last_ai_action", "args": args})
        response = client.post("/api/ai/actions/execute", json={
            "name": "undo_last_ai_action", "args": args,
            "preview_token": preview.json()["preview_token"],
        })
        assert response.status_code == 200, response.text
        assert response.json()["result"]["restored_from_version"] == 2
        assert len(store.get(sid).result.peaks) == 2
        assert store.get_last_ai_action(sid) is None


def test_computation_allows_other_worker_write_and_then_rejects_stale_save(seeded, monkeypatch):
    store, sid = seeded
    monkeypatch.setattr(store_module, "_store", store)
    worker = PersistentStore(store._db_path)
    seen = []

    def concurrent_review(*args):
        saved = worker.set_result_and_version(sid, _result("concurrent review"), expected_revision=1)
        seen.append(saved[0].result_revision)
        return {}

    monkeypatch.setattr(actions, "assess_quality", concurrent_review)
    args = {"spectrum_id": sid, "index": 0, "expected_revision": 1}
    preview = actions.preview_ai_action("delete_peak", args)
    with pytest.raises(RevisionConflict):
        actions.execute_ai_action("delete_peak", args, preview_token=preview["preview_token"])
    assert seen == [2]
    assert store.get(sid).result.summary == "concurrent review"
    assert len(store.list_result_versions(sid)) == 2
    assert store.get_last_ai_action(sid) is None


def test_undo_blocks_later_manual_save_without_changing_any_persisted_state(seeded):
    store, sid = seeded
    saved = _commit(store, sid)
    manual = store.set_result_and_version(
        sid, _result("later manual review"), note="Manual review",
        version_metric="manual_version", expected_revision=2,
    )
    before = _state(store)
    with pytest.raises(AIUndoConflict) as exc:
        store.undo_last_ai_action(sid, expected_revision=3)
    assert exc.value.reason == "later_result_edit"
    assert _state(store) == before
    assert store.get_last_ai_action(sid)["id"] == saved.action_id
    assert store.get(sid).result.summary == "later manual review"
    assert store.get_result_version(sid, manual[1]).result.metrics["manual_version"] == manual[1]


@pytest.mark.parametrize("undo", [False, True])
def test_final_commit_failure_rolls_back_all_ai_writes(seeded, monkeypatch, undo):
    store, sid = seeded
    if undo:
        _commit(store, sid)
    before = _state(store)

    class FailingCommitConnection(sqlite3.Connection):
        def commit(self):
            raise sqlite3.OperationalError("injected commit failure")

    with monkeypatch.context() as patch:
        patch.setattr(store, "_connect", lambda: sqlite3.connect(
            store._db_path, factory=FailingCommitConnection,
        ))
        with pytest.raises(sqlite3.OperationalError, match="injected commit failure"):
            if undo:
                store.undo_last_ai_action(sid, expected_revision=2)
            else:
                _commit(store, sid)
    assert _state(store) == before
    if undo:
        assert store.undo_last_ai_action(sid, expected_revision=2).result_revision == 3
    else:
        assert _commit(store, sid).result_revision == 2


@pytest.mark.parametrize("undo", [False, True])
def test_invariant_reads_happen_inside_immediate_write_transaction(seeded, monkeypatch, undo):
    store, sid = seeded
    if undo:
        _commit(store, sid)
    checked_reads = []

    class CheckingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            statement = sql.strip().upper()
            if statement.startswith("BEGIN"):
                assert statement == "BEGIN IMMEDIATE"
            if statement.startswith("SELECT"):
                assert self.in_transaction, "AI invariant read must hold the SQLite write reservation"
                checked_reads.append(statement)
            return super().execute(sql, parameters)

    with monkeypatch.context() as patch:
        patch.setattr(store, "_connect", lambda: sqlite3.connect(
            store._db_path, factory=CheckingConnection,
        ))
        if undo:
            store.undo_last_ai_action(sid, expected_revision=2)
        else:
            _commit(store, sid)
    assert len(checked_reads) >= (4 if undo else 2)
