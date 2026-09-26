"""Tests for ranker background job lifecycle management."""

from __future__ import annotations

import time

from app.api.routes import elucidate as route


def test_prune_removes_stale_jobs_and_keeps_fresh() -> None:
    with route._ranker_jobs_lock:
        route._ranker_jobs.clear()
        route._ranker_jobs["stale"] = {
            "status": "done",
            "started_at": time.time() - 90000,
        }
        route._ranker_jobs["fresh"] = {
            "status": "started",
            "started_at": time.time(),
        }
    route._prune_ranker_jobs()
    assert "stale" not in route._ranker_jobs
    assert "fresh" in route._ranker_jobs


def test_prune_caps_entry_count() -> None:
    with route._ranker_jobs_lock:
        route._ranker_jobs.clear()
        for index in range(route._RANKER_JOB_MAX_ENTRIES + 20):
            route._ranker_jobs[f"job-{index}"] = {
                "status": "done",
                "started_at": float(index),
            }
    route._prune_ranker_jobs()
    assert len(route._ranker_jobs) <= route._RANKER_JOB_MAX_ENTRIES
