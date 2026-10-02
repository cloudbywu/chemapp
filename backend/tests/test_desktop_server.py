"""Desktop-only trust boundary tests; web server configuration is untouched."""
import asyncio
import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location("desktop_server", Path(__file__).parents[1] / "desktop_server.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

TOKEN = "a" * 64
ORIGIN = "http://127.0.0.1:54321"


def request(extra=None, *, headers=None, client="127.0.0.1"):
    async def application(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
    app = _MODULE.DesktopBoundary(application, token=TOKEN, origin=ORIGIN, instance="b" * 64)
    scope = {"type": "http", "client": (client, 50000), "headers": headers if headers is not None else [
        (b"host", b"127.0.0.1:54321"), (b"x-chemapp-desktop-token", TOKEN.encode()),
    ]}
    scope.update(extra or {})
    messages = []
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    async def send(message):
        messages.append(message)
    asyncio.run(app(scope, receive, send))
    return messages


def test_desktop_gate_authenticates_all_http_routes_and_adds_csp():
    result = request({"path": "/api/live"})
    assert result[0]["status"] == 200
    headers = dict(result[0]["headers"])
    assert headers[b"x-chemapp-desktop-instance"] == b"b" * 64
    assert b"script-src 'self'" in headers[b"content-security-policy"]
    assert b"unsafe-eval" not in headers[b"content-security-policy"]
    assert headers[b"cache-control"] == b"no-store"


@pytest.mark.parametrize("path", ["/", "/index.html", "/api/live", "/api/ready", "/api/reviews"])
def test_static_and_normally_exempt_api_routes_require_desktop_token(path):
    assert request({"path": path}, headers=[(b"host", b"127.0.0.1:54321")])[0]["status"] == 403


@pytest.mark.parametrize("headers", [
    [(b"host", b"localhost:54321"), (b"x-chemapp-desktop-token", TOKEN.encode())],
    [(b"host", b"127.0.0.1:54321"), (b"x-chemapp-desktop-token", b"wrong")],
    [(b"host", b"127.0.0.1:54321"), (b"x-chemapp-desktop-token", b"\xff")],
    [(b"host", b"127.0.0.1:54321"), (b"x-chemapp-desktop-token", TOKEN.encode()), (b"origin", b"https://evil.example")],
    [(b"host", b"127.0.0.1:54321"), (b"x-chemapp-desktop-token", TOKEN.encode()), (b"origin", b"null")],
    [(b"host", b"127.0.0.1:54321"), (b"x-chemapp-desktop-token", TOKEN.encode()), (b"x-chemapp-desktop-token", TOKEN.encode())],
])
def test_rejects_host_origin_and_token_ambiguity(headers):
    assert request(headers=headers)[0]["status"] == 403


def test_rejects_non_loopback_peers_and_websockets():
    assert request(client="192.0.2.10")[0]["status"] == 403
    assert request({"type": "websocket"}) == [{"type": "websocket.close", "code": 1008}]


def test_config_uses_writable_user_database_and_preserves_science_settings(monkeypatch, tmp_path):
    import os
    # configuration mutates multiple launch-only variables: isolate all of them.
    monkeypatch.setattr(os, "environ", os.environ.copy())
    frontend = tmp_path / "ui"
    frontend.mkdir()
    (frontend / "index.html").write_text("ok")
    data = tmp_path / "data"
    for key, char in [("CHEMAPP_DESKTOP_TOKEN", "a"), ("CHEMAPP_DESKTOP_INSTANCE", "b"), ("CHEMAPP_ACCESS_TOKEN", "c"), ("CHEMAPP_ADMIN_TOKEN", "d")]:
        monkeypatch.setenv(key, char * 64)
    monkeypatch.setenv("CHEMAPP_DESKTOP_FRONTEND", str(frontend))
    monkeypatch.setenv("CHEMAPP_DESKTOP_DATA_DIR", str(data))
    monkeypatch.setenv("CHEMAPP_CALIBRATION_MODE", "on")
    monkeypatch.chdir(tmp_path)
    assert _MODULE.configuration() == frontend
    assert os.environ["CHEMAPP_DB_PATH"] == str(data / "chemapp.db")
    assert os.environ["CHEMAPP_CALIBRATION_MODE"] == "on"
    assert os.environ["CHEMAPP_LOCAL_ADMIN_BYPASS"] == "0"
