"""Experiment-agent route fixes: ValueError mapped to 404 only for missing
spectra; every other input error is a 422, consistently across endpoints.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import app.api.store as store_module
from app.api.routes import experiment_agent as routes
from app.main import app

_HANDOUT = ("handout.md", b"# handout", "text/markdown")
_ENDPOINTS = (
    "/api/experiment-agent/report",
    "/api/experiment-agent/report/markdown",
    "/api/experiment-agent/report/docx",
)


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.delenv("CHEMAPP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
    store_module._store = None
    return TestClient(app)


def _post(client, path):
    return client.post(path, data={"ids": '["spec-x"]'}, files={"handout": _HANDOUT})


def test_missing_spectrum_maps_to_404_on_all_endpoints(tmp_path, monkeypatch):
    def boom(**kwargs):
        raise ValueError("Spectrum spec-x not found")

    monkeypatch.setattr(routes, "generate_experiment_report", boom)
    with _client(tmp_path, monkeypatch) as client:
        for path in _ENDPOINTS:
            response = _post(client, path)
            assert response.status_code == 404, path
            assert "spec-x" in response.text


def test_input_errors_map_to_422_on_all_endpoints(tmp_path, monkeypatch):
    def boom(**kwargs):
        raise ValueError("handout content is unreadable")

    monkeypatch.setattr(routes, "generate_experiment_report", boom)
    with _client(tmp_path, monkeypatch) as client:
        for path in _ENDPOINTS:
            response = _post(client, path)
            assert response.status_code == 422, path
            assert "handout content is unreadable" in response.text
