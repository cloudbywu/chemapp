"""Elucidate route fixes: admin-gated job status, sanitized job errors,
prune-before-register, and guaranteed read-only index connection cleanup.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi.testclient import TestClient

import app.api.store as store_module
from app.api.routes import elucidate as route
from app.main import app


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.delenv("CHEMAPP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
    store_module._store = None
    return TestClient(app)


def _clear_jobs() -> None:
    with route._ranker_jobs_lock:
        route._ranker_jobs.clear()


def test_ranker_train_status_requires_admin(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        # Set after the client helper, which clears admin configuration.
        monkeypatch.setenv("CHEMAPP_ADMIN_TOKEN", "admin-token")
        denied = client.get("/api/ml/elucidate/ranker/train/status/whatever")
        assert denied.status_code == 401

        allowed = client.get(
            "/api/ml/elucidate/ranker/train/status/whatever",
            headers={"X-ChemApp-Admin-Token": "admin-token"},
        )
        # Past the admin gate; the job itself is unknown.
        assert allowed.status_code == 404


def test_run_ranker_job_stores_sanitized_error(tmp_path, monkeypatch):
    _clear_jobs()
    with route._ranker_jobs_lock:
        route._ranker_jobs["job-1"] = {
            "status": "started",
            "job_id": "job-1",
            "started_at": time.time(),
        }

    def boom(**kwargs):
        raise RuntimeError("secret internal detail")

    released: list[str] = []
    monkeypatch.setattr(route, "train_joint_ranker", boom)
    monkeypatch.setattr(
        "app.api.routes.ml._release_training_lock",
        lambda path, job_id: released.append(job_id),
    )
    route._run_ranker_job("job-1", route.TrainRankerRequest(), Path("lock"))
    try:
        job = route._ranker_jobs["job-1"]
        assert job["status"] == "error"
        assert job["error"] == "Ranker training failed"
        assert "secret" not in job["error"]
        assert released == ["job-1"]
    finally:
        _clear_jobs()


def test_train_ranker_prunes_jobs_before_registering(tmp_path, monkeypatch):
    _clear_jobs()
    with route._ranker_jobs_lock:
        for index in range(route._RANKER_JOB_MAX_ENTRIES + 10):
            route._ranker_jobs[f"old-{index}"] = {
                "status": "done",
                "started_at": float(index),
                "finished_at": float(index),
            }

    monkeypatch.setattr(
        "app.api.routes.ml._acquire_training_lock",
        lambda job_id: tmp_path / f"{job_id}.lock",
    )
    monkeypatch.setattr("app.api.routes.ml._release_training_lock", lambda path, job_id: None)
    monkeypatch.setattr(route, "train_joint_ranker", lambda **kwargs: {"ok": True})

    with _client(tmp_path, monkeypatch) as client:
        response = client.post("/api/ml/elucidate/ranker/train", json={})
    assert response.status_code == 202, response.text
    job_id = response.json()["job_id"]

    deadline = time.time() + 10
    while time.time() < deadline:
        with route._ranker_jobs_lock:
            job = route._ranker_jobs.get(job_id)
        if job is not None and job.get("status") == "done":
            break
        time.sleep(0.05)

    try:
        with route._ranker_jobs_lock:
            jobs = dict(route._ranker_jobs)
        assert job_id in jobs
        # Pruned to the cap (plus at most the freshly registered job).
        assert len(jobs) <= route._RANKER_JOB_MAX_ENTRIES + 1
        assert jobs[job_id]["status"] == "done"
    finally:
        _clear_jobs()


class _Rows(list):
    def fetchone(self):
        return self[0] if self else None


def test_v2_index_status_closes_connection_on_error(tmp_path, monkeypatch):
    closed: list[bool] = []

    class _Conn:
        def execute(self, *args, **kwargs):
            raise RuntimeError("query boom")

        def close(self):
            closed.append(True)

    index = tmp_path / "index.sqlite"
    index.write_bytes(b"placeholder")
    monkeypatch.setenv("CHEMAPP_NMR_INDEX_V2", str(index))
    monkeypatch.setattr(route, "connect_readonly", lambda path: _Conn())
    result = route._v2_index_status()
    assert result["available"] is False
    assert result["error"] == "RuntimeError"
    assert closed == [True]


def test_v2_index_status_closes_connection_on_success(tmp_path, monkeypatch):
    closed: list[bool] = []

    class _Conn:
        def execute(self, sql, params=None):
            normalized = " ".join(sql.split())
            if normalized.startswith("SELECT COUNT(*) FROM spectra"):
                return _Rows([(5,)])
            if normalized.startswith("SELECT COUNT(*) FROM peaks"):
                return _Rows([(7,)])
            if normalized.startswith("SELECT COUNT(*) FROM molecules"):
                return _Rows([(2,)])
            if normalized.startswith("SELECT measurement_kind"):
                return _Rows([("solution", 5)])
            if normalized.startswith("SELECT nucleus"):
                return _Rows([("13C", 5)])
            if normalized.startswith("SELECT source_name"):
                return _Rows([{
                    "source_name": "src",
                    "source_version": "v1",
                    "sha256": "abc",
                    "license_uri": "uri",
                    "validation_json": json.dumps({"build_options": {"x": 1}}),
                }])
            if normalized.startswith("SELECT COUNT(*) FROM import_rejections"):
                return _Rows([(0,)])
            raise AssertionError(f"unexpected SQL: {normalized}")

        def close(self):
            closed.append(True)

    index = tmp_path / "index.sqlite"
    index.write_bytes(b"placeholder")
    monkeypatch.setenv("CHEMAPP_NMR_INDEX_V2", str(index))
    monkeypatch.setattr(route, "connect_readonly", lambda path: _Conn())
    monkeypatch.setattr(route, "schema_version", lambda conn: 3)
    result = route._v2_index_status()
    assert result["available"] is True
    assert result["schema_version"] == 3
    assert result["spectra"] == 5
    assert result["snapshot"]["build_options"] == {"x": 1}
    assert closed == [True]
