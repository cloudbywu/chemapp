"""Security tests for the AI API routes (502 unification, execute admin gate)."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import app.api.store as store_module
from app.ai.actions import action_is_destructive
from app.analysis.models import AnalysisResult
from app.api.routes import ai as ai_routes
from app.core.models import Spectrum, Technique
from app.main import app


def _spectrum() -> Spectrum:
    return Spectrum(
        technique=Technique.NMR,
        x_data=np.asarray([2.0, 1.0, 0.0]),
        y_data=np.asarray([0.0, 1.0, 0.0]),
        source_file="sample.jdf",
    )


def _result() -> AnalysisResult:
    return AnalysisResult(technique=Technique.NMR, metrics={"ok": True}, summary="result")


def _seed(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.delenv("CHEMAPP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
    store_module._store = None
    sid = store_module.get_store().add(_spectrum()).id
    store_module.get_store().set_result(sid, _result(), expected_revision=0)
    return sid


def test_analyze_single_returns_502_with_sanitized_error(tmp_path, monkeypatch) -> None:
    sid = _seed(tmp_path, monkeypatch)

    def exploding(*args, **kwargs):
        raise RuntimeError("internal boom detail: api key leaked")

    monkeypatch.setattr(ai_routes, "analyze_single", exploding)
    with TestClient(app) as client:
        response = client.post("/api/ai/analyze", json={"ids": [sid]})
    assert response.status_code == 502
    assert response.json()["detail"] == "AI analysis failed; check the server log"
    assert "internal boom" not in response.text


def test_analyze_cross_returns_502_with_sanitized_error(tmp_path, monkeypatch) -> None:
    sid_one = _seed(tmp_path, monkeypatch)
    sid_two = store_module.get_store().add(_spectrum()).id
    store_module.get_store().set_result(sid_two, _result(), expected_revision=0)

    def exploding(*args, **kwargs):
        raise ConnectionError("cross backend exploded")

    monkeypatch.setattr(ai_routes, "analyze_cross", exploding)
    with TestClient(app) as client:
        response = client.post("/api/ai/analyze", json={"ids": [sid_one, sid_two]})
    assert response.status_code == 502
    assert response.json()["detail"] == "AI analysis failed; check the server log"
    assert "exploded" not in response.text


def test_execute_destructive_action_requires_admin_token(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.setenv("CHEMAPP_ACCESS_TOKEN", "access-1")
    monkeypatch.setenv("CHEMAPP_ADMIN_TOKEN", "admin-1")
    store_module._store = None

    assert action_is_destructive("delete_peak") is True

    with TestClient(app) as client:
        denied = client.post(
            "/api/ai/actions/execute",
            json={"name": "delete_peak", "args": {"spectrum_id": "x"}},
            headers={"X-ChemApp-Access-Token": "access-1"},
        )
        assert denied.status_code == 401

        allowed = client.post(
            "/api/ai/actions/execute",
            json={"name": "delete_peak", "args": {"spectrum_id": "x"}},
            headers={"X-ChemApp-Admin-Token": "admin-1"},
        )
        # Admin check passes; preview_token validation then fails closed (400).
        assert allowed.status_code == 400
        assert "preview_token" in allowed.text


def test_execute_destructive_action_rejects_wrong_admin_token(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.setenv("CHEMAPP_ACCESS_TOKEN", "access-1")
    monkeypatch.setenv("CHEMAPP_ADMIN_TOKEN", "admin-1")
    store_module._store = None

    with TestClient(app) as client:
        response = client.post(
            "/api/ai/actions/execute",
            json={"name": "update_nmr_integral_range", "args": {"spectrum_id": "x"}},
            headers={
                "X-ChemApp-Access-Token": "access-1",
                "X-ChemApp-Admin-Token": "wrong",
            },
        )
    assert response.status_code == 401


def test_execute_non_destructive_action_does_not_require_admin(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.setenv("CHEMAPP_ACCESS_TOKEN", "access-1")
    monkeypatch.setenv("CHEMAPP_ADMIN_TOKEN", "admin-1")
    store_module._store = None

    assert action_is_destructive("run_cross_inference") is False

    with TestClient(app) as client:
        response = client.post(
            "/api/ai/actions/execute",
            json={"name": "run_cross_inference", "args": {"ids": ["missing"]}},
            headers={"X-ChemApp-Access-Token": "access-1"},
        )
    # No admin gate for non-destructive actions; handler fails on its own (404/400).
    assert response.status_code in (400, 404, 409)
