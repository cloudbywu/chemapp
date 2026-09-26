from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

import app.api.store as store_module
from app.analysis.models import AnalysisResult
from app.api.routes import ai as ai_routes
from app.core.models import Peak, Spectrum, Technique
from app.main import app


def _spectrum() -> Spectrum:
    return Spectrum(
        technique=Technique.NMR,
        x_data=np.asarray([2.0, 1.0, 0.0]),
        y_data=np.asarray([0.0, 1.0, 0.0]),
        source_file="sample.jdf",
    )


def _result() -> AnalysisResult:
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
        summary="result",
    )


def test_ai_stream_ends_without_error_event(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    store_module._store = None
    sid = store_module.get_store().add(_spectrum()).id
    store_module.get_store().set_result(sid, _result(), expected_revision=0)

    def fake_stream(*_args, **_kwargs):
        yield "hello world"

    monkeypatch.setattr(ai_routes, "stream_chat", fake_stream)

    with TestClient(app) as client:
        response = client.post("/api/ai/stream", json={"ids": [sid]})

    assert response.status_code == 200
    assert "data: hello world" in response.text
    assert '"type": "error"' not in response.text
