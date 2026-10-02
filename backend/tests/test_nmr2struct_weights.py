"""Atomic official downloads: no network or large checkpoints needed here."""
from dataclasses import replace
import hashlib
from pathlib import Path
import threading
import time

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
import requests

from app.ml import nmr2struct_weights as weights
from app.api.routes import model_weights


DATA = b"official test checkpoint\n" * 100


class Response:
    def __init__(self, chunks=None, *, status=200, headers=None, gate=None, entered=None):
        self.status_code = status
        self.headers = headers or {}
        self.chunks = chunks if chunks is not None else [DATA]
        self.gate = gate
        self.entered = entered
        self.raw = self
        self._iterator = None

    def __enter__(self):
        self._iterator = None
        return self

    def __exit__(self, *args):
        pass

    def read1(self, chunk_size, decode_content=False):
        assert chunk_size == 64 * 1024
        assert decode_content is False
        if self._iterator is None:
            if self.entered:
                self.entered.set()
            if self.gate:
                assert self.gate.wait(5)
            self._iterator = iter(self.chunks)
        chunk = next(self._iterator, b"")
        if isinstance(chunk, Exception):
            raise chunk
        return chunk


@pytest.fixture
def setup(monkeypatch, tmp_path):
    directory = tmp_path / "writable"
    vendor = tmp_path / "vendor"
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_WEIGHTS_DIR", str(directory))
    monkeypatch.setattr(weights, "VENDOR_DIR", vendor)
    assets = {key: replace(asset, size_bytes=len(DATA), sha256=hashlib.sha256(DATA).hexdigest())
              for key, asset in weights.ASSETS.items()}
    monkeypatch.setattr(weights, "ASSETS", assets)
    weights._verified.clear()
    manager = weights.DownloadManager()
    return manager, directory, vendor, assets


def network(monkeypatch, response):
    calls = []

    def get(session, url, **kwargs):
        calls.append((url, kwargs))
        assert session.trust_env is False
        assert kwargs["allow_redirects"] is False
        assert kwargs["timeout"] == (10, 5)
        assert kwargs["verify"] is not False
        assert isinstance(kwargs["proxies"], dict)
        return response

    monkeypatch.setattr(requests.Session, "get", get)
    return calls


def finished(manager, job):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        current = manager.get(job["id"])
        if current["status"] not in weights._ACTIVE:
            # Terminal state includes cleanup completion (and released OS lock).
            return current
        time.sleep(0.005)
    pytest.fail("download worker did not terminate")


def test_success_is_atomic_verified_and_immediately_available(setup, monkeypatch):
    manager, directory, vendor, assets = setup
    calls = network(monkeypatch, Response([DATA[:500], DATA[500:]]))
    job = finished(manager, manager.start("cnmr_only"))
    target = directory / assets["cnmr_only"].filename
    assert job["status"] == "completed"
    assert job["downloaded_bytes"] == len(DATA)
    assert target.read_bytes() == DATA
    assert weights.checkpoint_path("cnmr_only") == target
    assert manager.inventory()["assets"][0]["status"] == "ready"
    assert not list(directory.glob("*.part"))
    assert calls[0][0] == assets["cnmr_only"].url
    assert weights.REVISION in calls[0][0]
    assert "cloudbywu" not in calls[0][0]
    # Repeated requests reuse verified bytes without any further network call.
    assert manager.start("cnmr_only")["status"] == "completed"
    assert len(calls) == 1
    assert finished(manager, manager.start("hnmr_only"))["status"] == "completed"


@pytest.mark.parametrize("response,error", [
    (Response([b"wrong" * (len(DATA) // 5)]), "checksum"),
    (Response([DATA[:5]]), "checksum"),
    (Response([DATA, b"extra"]), "exceeded"),
    (Response(status=302, headers={"Location": "http://evil.test"}), "HTTP"),
    (Response(status=404), "HTTP"),
    (Response(headers={"Content-Length": str(len(DATA) + 1)}), "size"),
    (Response(headers={"Content-Encoding": "gzip"}), "encoding"),
    (Response([DATA[:5], requests.ConnectionError("secret proxy credential")]), "connection"),
])
def test_failed_download_never_publishes_or_deletes_existing_file(setup, monkeypatch, response, error):
    manager, directory, _, assets = setup
    directory.mkdir()
    target = directory / assets["cnmr_only"].filename
    target.write_bytes(b"previous bytes")
    network(monkeypatch, response)
    job = finished(manager, manager.start("cnmr_only"))
    assert job["status"] == "error"
    assert error in job["error"]
    assert "secret" not in job["error"]
    assert target.read_bytes() == b"previous bytes"
    assert not list(directory.glob("*.part"))
    network(monkeypatch, Response())
    assert finished(manager, manager.start("cnmr_only"))["status"] == "completed"
    assert target.read_bytes() == DATA


def test_cancel_concurrency_and_retry_after_interruption(setup, monkeypatch):
    manager, directory, _, assets = setup
    gate, entered = threading.Event(), threading.Event()
    calls = network(monkeypatch, Response(gate=gate, entered=entered))
    first = manager.start("cnmr_only")
    assert entered.wait(5)
    assert manager.start("cnmr_only")["id"] == first["id"]
    with pytest.raises(weights.DownloadBusy):
        manager.start("hnmr_only")
    # An independent process/manager cannot race the same storage directory.
    second = weights.DownloadManager()
    with pytest.raises(weights.DownloadBusy):
        second.start("cnmr_only")
    assert not (directory / assets["cnmr_only"].filename).exists()
    assert manager.cancel(first["id"])["status"] == "cancelling"
    gate.set()
    assert finished(manager, first)["status"] == "cancelled"
    assert len(calls) == 1
    assert not list(directory.glob("*.part"))
    assert not (directory / assets["cnmr_only"].filename).exists()
    network(monkeypatch, Response())
    assert finished(second, second.start("cnmr_only"))["status"] == "completed"
    assert manager.cancel(first["id"])["status"] == "cancelled"


def test_cleanup_stale_partial_and_preserve_valid_bundled_file(setup, monkeypatch):
    manager, directory, vendor, assets = setup
    directory.mkdir()
    stale = directory / ".nmr2struct-abandoned.part"
    stale.write_bytes(b"incomplete")
    (directory / "unrelated.part").write_bytes(b"keep")
    network(monkeypatch, Response())
    assert finished(manager, manager.start("cnmr_only"))["status"] == "completed"
    assert not stale.exists()
    assert (directory / "unrelated.part").read_bytes() == b"keep"
    (vendor / "checkpoints").mkdir(parents=True)
    bundled = vendor / "checkpoints" / assets["hnmr_only"].filename
    bundled.write_bytes(DATA)
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: pytest.fail("must not redownload"))
    assert manager.start("hnmr_only")["status"] == "completed"
    assert weights.checkpoint_path("hnmr_only") == bundled


def test_disk_exhaustion_and_timeout_recover(setup, monkeypatch):
    manager, directory, _, _ = setup
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    with monkeypatch.context() as patch:
        patch.setattr(weights.shutil, "disk_usage", lambda _: usage(100, 99, 1))
        assert "disk space" in finished(manager, manager.start("cnmr_only"))["error"]
    with monkeypatch.context() as patch:
        patch.setattr(weights, "_MAX_SECONDS", -1)
        assert "timed out" in finished(manager, manager.start("cnmr_only"))["error"]
    network(monkeypatch, Response())
    assert finished(manager, manager.start("cnmr_only"))["status"] == "completed"


def test_invalid_downloaded_file_falls_back_and_hash_cache_detects_changes(setup):
    manager, directory, vendor, assets = setup
    directory.mkdir()
    target = directory / assets["cnmr_only"].filename
    target.write_bytes(DATA)
    assert weights.verified_file(target, assets["cnmr_only"])
    target.write_bytes(b"X" * len(DATA))
    assert not weights.verified_file(target, assets["cnmr_only"])
    assert weights.checkpoint_path("cnmr_only") is None
    assert manager.inventory()["assets"][0]["status"] == "invalid"
    (vendor / "checkpoints").mkdir(parents=True)
    bundled = vendor / "checkpoints" / assets["cnmr_only"].filename
    bundled.write_bytes(DATA)
    assert weights.checkpoint_path("cnmr_only") == bundled


def test_symlink_target_never_follows_or_overwrites_external_file(setup, monkeypatch, tmp_path):
    manager, directory, _, assets = setup
    directory.mkdir()
    outside = tmp_path / "external"
    outside.write_bytes(b"preserve")
    target = directory / assets["cnmr_only"].filename
    target.symlink_to(outside)
    network(monkeypatch, Response())
    assert finished(manager, manager.start("cnmr_only"))["status"] == "completed"
    assert not target.is_symlink()
    assert outside.read_bytes() == b"preserve"
    assert target.read_bytes() == DATA


def test_storage_configuration_is_per_user_and_desktop_safe(monkeypatch, tmp_path):
    monkeypatch.delenv("CHEMAPP_NMR2STRUCT_WEIGHTS_DIR", raising=False)
    monkeypatch.setenv("CHEMAPP_DESKTOP_DATA_DIR", str(tmp_path / "desktop"))
    assert weights.weights_directory() == tmp_path / "desktop/models/nmr2struct"
    monkeypatch.delenv("CHEMAPP_DESKTOP_DATA_DIR")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(weights.sys, "platform", "linux")
    assert weights.weights_directory() == tmp_path / "xdg/chemapp/models/nmr2struct"
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_WEIGHTS_DIR", str(tmp_path / "override"))
    assert weights.weights_directory() == tmp_path / "override"


def test_api_admin_auth_allowlist_and_polling(setup, monkeypatch):
    manager, _, _, _ = setup
    monkeypatch.setattr(model_weights, "manager", manager)
    monkeypatch.setenv("CHEMAPP_ADMIN_TOKEN", "admin")
    app = FastAPI()
    app.include_router(model_weights.router)
    client = TestClient(app)
    url = "/api/ml/nmr2struct/weights"
    assert client.get(url).status_code == 200
    assert client.post(url + "/downloads", json={"asset_id": "cnmr_only"}).status_code == 401
    headers = {"X-ChemApp-Admin-Token": "admin"}
    for payload in [{"asset_id": "../evil"}, {"asset_id": "cnmr_only", "url": "https://evil.test"},
                    {"asset_id": "multitask", "path": "/tmp/evil"}, {"asset_id": "transformer"}]:
        assert client.post(url + "/downloads", json=payload, headers=headers).status_code == 422
    network(monkeypatch, Response())
    response = client.post(url + "/downloads", json={"asset_id": "cnmr_only"}, headers=headers)
    assert response.status_code == 202
    job = finished(manager, response.json())
    assert client.get(url + "/downloads/" + job["id"]).json()["status"] == "completed"
    cancel = url + "/downloads/" + job["id"] + "/cancel"
    assert client.post(cancel).status_code == 401
    assert client.post(cancel, headers=headers).json()["status"] == "completed"
    assert client.get(url + "/downloads/no-such-job").status_code == 404
    assert client.post(url + "/downloads/no-such-job/cancel", headers=headers).status_code == 404


def test_readonly_storage_failure_is_sanitized(setup, monkeypatch):
    manager, _, _, _ = setup
    monkeypatch.setattr(weights.Path, "mkdir", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("private path")))
    with pytest.raises(weights.DownloadFailure, match="Cannot write"):
        manager.start("cnmr_only")


def test_manifest_paths_and_hashes_are_fixed():
    assert set(weights.ASSETS) == {"cnmr_only", "hnmr_only", "multitask"}
    for asset in weights.ASSETS.values():
        assert len(asset.sha256) == 64
        assert Path(asset.filename).name == asset.filename
        assert asset.url.startswith("https://raw.githubusercontent.com/MarklandGroup/NMR2Struct/" + weights.REVISION)
        assert asset.size_bytes > 100_000_000


def test_trickle_socket_honors_total_deadline_without_waiting_for_full_chunk(setup, monkeypatch):
    """Exercise real urllib3 read1, not a mock that always yields full chunks."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    manager, directory, _, assets = setup
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(DATA)))
            self.end_headers()
            try:
                for byte in DATA:
                    self.wfile.write(bytes([byte]))
                    self.wfile.flush()
                    time.sleep(0.01)
            except (BrokenPipeError, ConnectionResetError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original = requests.Session.get
    def local(session, url, **kwargs):
        assert url == assets["cnmr_only"].url
        kwargs["proxies"] = {}  # The test server is local, not the public fixed URL.
        return original(session, f"http://127.0.0.1:{server.server_port}/", **kwargs)
    monkeypatch.setattr(requests.Session, "get", local)
    monkeypatch.setattr(weights, "_MAX_SECONDS", 0.08)
    try:
        started = time.monotonic()
        result = finished(manager, manager.start("cnmr_only"))
        assert result["status"] == "error"
        assert "timed out" in result["error"]
        assert time.monotonic() - started < 1
        assert not (directory / assets["cnmr_only"].filename).exists()
    finally:
        server.shutdown()
        server.server_close()


def test_atime_only_change_does_not_poison_verification_cache(setup):
    import os
    _, directory, _, assets = setup
    directory.mkdir()
    target = directory / assets["cnmr_only"].filename
    target.write_bytes(DATA)
    os.utime(target, (1, time.time()))
    assert weights.verified_file(target, assets["cnmr_only"])
    assert weights.verified_file(target, assets["cnmr_only"])


def test_cleanup_failure_still_releases_cross_process_lock(setup, monkeypatch):
    manager, directory, _, _ = setup
    original = Path.unlink
    def fail_part(path, *args, **kwargs):
        if path.name.startswith(".nmr2struct-"):
            raise PermissionError("test staging cleanup failure")
        return original(path, *args, **kwargs)
    network(monkeypatch, Response([b"broken"]))
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_part)
        assert finished(manager, manager.start("cnmr_only"))["status"] == "error"
    network(monkeypatch, Response())
    another = weights.DownloadManager()
    assert finished(another, another.start("cnmr_only"))["status"] == "completed"
    assert not list(directory.glob("*.part"))


def test_configured_proxy_and_ca_are_honored_without_netrc(setup, monkeypatch):
    manager, _, _, _ = setup
    proxy = {"https": "http://proxy.example:8080"}
    monkeypatch.setattr(requests.utils, "get_environ_proxies", lambda url: proxy)
    monkeypatch.setenv("REQUESTS_CA_BUNDLE", "/operator-ca.pem")
    seen = network(monkeypatch, Response())
    assert finished(manager, manager.start("cnmr_only"))["status"] == "completed"
    assert seen[0][1]["proxies"] == proxy
    assert seen[0][1]["verify"] == "/operator-ca.pem"


def test_directory_fsync_failure_does_not_report_cancel_after_publication(setup, monkeypatch):
    import os
    import stat
    manager, directory, _, assets = setup
    original = os.fsync
    def fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            raise OSError("directory fsync unsupported")
        original(fd)
    monkeypatch.setattr(os, "fsync", fsync)
    network(monkeypatch, Response())
    job = finished(manager, manager.start("cnmr_only"))
    assert job["status"] == "completed"
    assert manager.cancel(job["id"])["status"] == "completed"
    assert (directory / assets["cnmr_only"].filename).read_bytes() == DATA
