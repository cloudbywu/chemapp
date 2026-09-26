"""Event-loop hygiene checks: async routes must delegate blocking work.

Every async route that performs heavy synchronous CPU/IO work (report
generation, DOCX rendering, ZIP extraction, instrument parsing) must do so
through ``anyio.to_thread.run_sync`` so a single request cannot freeze the
event loop for all other clients.
"""
from __future__ import annotations

import inspect
import io
import zipfile

from fastapi.testclient import TestClient

import app.api.store as store_module
from app.api.routes import experiment_agent as experiment_agent_routes
from app.api.routes import upload as upload_routes
from app.main import app


class _RecordingToThread:
    """Drop-in for ``anyio.to_thread`` that records and inline-executes calls."""

    def __init__(self) -> None:
        self.calls: list = []

    async def run_sync(self, func, *args, **kwargs):
        self.calls.append(func)
        return func(*args)


class _FakeAnyio:
    """Minimal namespace exposing only the patched ``to_thread`` attribute."""

    def __init__(self, to_thread: _RecordingToThread) -> None:
        self.to_thread = to_thread


def _base_funcs(spy: _RecordingToThread) -> list:
    return [getattr(call, "func", call) for call in spy.calls]


def _patch_to_thread(monkeypatch, module, spy: _RecordingToThread) -> None:
    monkeypatch.setattr(module, "anyio", _FakeAnyio(spy))


def _client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    monkeypatch.delenv("CHEMAPP_ACCESS_TOKEN", raising=False)
    monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
    store_module._store = None
    return TestClient(app)


class _FakeReport:
    title = "实验报告"
    markdown = "# 实验报告"
    used_llm = False
    steps = [{"status": "done"}]

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "markdown": self.markdown,
            "used_llm": self.used_llm,
            "steps": self.steps,
        }


def _fake_generate(**kwargs) -> _FakeReport:
    assert kwargs["ids"] == ["spec-1"]
    assert kwargs["handout_content"] == b"# handout"
    return _FakeReport()


_HANDOUT = ("handout.md", b"# handout", "text/markdown")


def test_report_preview_stays_async_and_offloads_generation(tmp_path, monkeypatch):
    spy = _RecordingToThread()
    _patch_to_thread(monkeypatch, experiment_agent_routes, spy)
    monkeypatch.setattr(
        experiment_agent_routes, "generate_experiment_report", _fake_generate
    )

    assert inspect.iscoroutinefunction(
        experiment_agent_routes.generate_report_preview
    )
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/experiment-agent/report",
            data={"ids": '["spec-1"]', "use_llm": "false"},
            files={"handout": _HANDOUT},
        )

    assert response.status_code == 200, response.text
    assert response.json()["title"] == _FakeReport.title
    assert _fake_generate in _base_funcs(spy)


def test_report_markdown_stays_async_and_offloads_generation(tmp_path, monkeypatch):
    spy = _RecordingToThread()
    _patch_to_thread(monkeypatch, experiment_agent_routes, spy)
    monkeypatch.setattr(
        experiment_agent_routes, "generate_experiment_report", _fake_generate
    )

    assert inspect.iscoroutinefunction(
        experiment_agent_routes.download_report_markdown
    )
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/experiment-agent/report/markdown",
            data={"ids": '["spec-1"]', "use_llm": "false"},
            files={"handout": _HANDOUT},
        )

    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/markdown")
    assert "experiment-report.md" in response.headers["content-disposition"]
    assert response.text == _FakeReport.markdown
    assert _fake_generate in _base_funcs(spy)


def test_report_docx_stays_async_and_offloads_generation_and_rendering(
    tmp_path, monkeypatch
):
    spy = _RecordingToThread()
    _patch_to_thread(monkeypatch, experiment_agent_routes, spy)
    monkeypatch.setattr(
        experiment_agent_routes, "generate_experiment_report", _fake_generate
    )

    def fake_docx(markdown: str, title: str) -> bytes:
        assert markdown == _FakeReport.markdown
        assert title == _FakeReport.title
        return b"PK-fake-docx"

    monkeypatch.setattr(
        experiment_agent_routes, "markdown_to_docx_bytes", fake_docx
    )

    assert inspect.iscoroutinefunction(experiment_agent_routes.download_report_docx)
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/experiment-agent/report/docx",
            data={"ids": '["spec-1"]', "use_llm": "false"},
            files={"handout": _HANDOUT},
        )

    assert response.status_code == 200, response.text
    assert "wordprocessingml" in response.headers["content-type"]
    assert response.content == b"PK-fake-docx"
    base_funcs = _base_funcs(spy)
    assert _fake_generate in base_funcs
    assert fake_docx in base_funcs


def test_zip_upload_stays_async_and_offloads_extract_and_parse(
    tmp_path, monkeypatch
):
    spy = _RecordingToThread()
    _patch_to_thread(monkeypatch, upload_routes, spy)
    parse_calls: list[tuple[str, str | None]] = []

    def fake_parse(file_path, source_name=None) -> list[dict]:
        parse_calls.append((str(file_path), source_name))
        return [{"id": "fake-1", "technique": "NMR", "points": 3}]

    monkeypatch.setattr(upload_routes, "_parse_and_store", fake_parse)

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("sample/acqu", "params")
        zf.writestr("sample/fid", "data")

    assert inspect.iscoroutinefunction(upload_routes.upload_file)
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/upload",
            files={"file": ("bundle.zip", archive.getvalue(), "application/zip")},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["id"] == "fake-1"
    assert payload["name"] == "bundle.zip"
    base_funcs = _base_funcs(spy)
    assert upload_routes._safe_extract_zip in base_funcs
    assert fake_parse in base_funcs
    assert ("bundle.zip",) == tuple({src for _, src in parse_calls})


def test_single_file_upload_stays_async_and_offloads_parse(tmp_path, monkeypatch):
    spy = _RecordingToThread()
    _patch_to_thread(monkeypatch, upload_routes, spy)

    def fake_parse(file_path, source_name=None) -> list[dict]:
        assert source_name == "scan.jdx"
        return [{"id": "fake-2", "technique": "NMR", "points": 3}]

    monkeypatch.setattr(upload_routes, "_parse_and_store", fake_parse)

    assert inspect.iscoroutinefunction(upload_routes.upload_file)
    with _client(tmp_path, monkeypatch) as client:
        response = client.post(
            "/api/upload",
            files={"file": ("scan.jdx", b"##TITLE=demo", "application/octet-stream")},
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["id"] == "fake-2"
    assert payload["name"] == "scan.jdx"
    assert fake_parse in _base_funcs(spy)
