"""Private desktop entry point. The ordinary web server remains app.main:app.

Electron owns this process, keeps its stdin pipe open, and authenticates every
request using a fresh, non-persistent token. Never launch this on a public host.
"""
from __future__ import annotations

import hmac
import json
import os
from pathlib import Path
import re
import socket
import signal
import sys
import threading

# -I isolates the bundled interpreter from user site packages/PYTHONPATH.
# Explicitly add only our sibling backend source directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

CSP = "; ".join([
    "default-src 'none'",
    "script-src 'self'",
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "font-src 'self' data:",
    "connect-src 'self'",
    "worker-src 'self' blob:",
    "object-src 'none'",
    "base-uri 'none'",
    "frame-ancestors 'none'",
    "form-action 'none'",
])


class DesktopBoundary:
    """An outer ASGI gate, including static files and exempt API endpoints."""

    def __init__(self, app, *, token: str, origin: str, instance: str):
        self.app = app
        self.token = token
        self.origin = origin
        self.authority = origin.removeprefix("http://")
        self.instance = instance

    async def __call__(self, scope, receive, send):
        if scope["type"] == "websocket":
            await send({"type": "websocket.close", "code": 1008})
            return
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {}
        for key, value in scope.get("headers", []):
            headers.setdefault(key.lower(), []).append(value.decode("latin-1"))
        token = headers.get(b"x-chemapp-desktop-token", [])
        hosts = headers.get(b"host", [])
        origins = headers.get(b"origin", [])
        allowed = (
            len(token) == 1
            and hmac.compare_digest(token[0].encode("latin-1"), self.token.encode("ascii"))
            and hosts == [self.authority]
            and (not origins or origins == [self.origin])
            and scope.get("client", ("",))[0] == "127.0.0.1"
        )
        if not allowed:
            await send({"type": "http.response.start", "status": 403, "headers": [
                (b"content-type", b"text/plain; charset=utf-8"),
                (b"cache-control", b"no-store"),
            ]})
            await send({"type": "http.response.body", "body": b"Desktop session required"})
            return

        async def secure_send(message):
            if message["type"] == "http.response.start":
                extra = [
                    (b"content-security-policy", CSP.encode()),
                    (b"x-chemapp-desktop-instance", self.instance.encode()),
                    (b"x-content-type-options", b"nosniff"),
                    (b"cache-control", b"no-store"),
                ]
                names = {name for name, _ in extra}
                message = {**message, "headers": [
                    (k, v) for k, v in message.get("headers", []) if k.lower() not in names
                ] + extra}
            await send(message)

        await self.app(scope, receive, secure_send)


def configuration():
    required = ["CHEMAPP_DESKTOP_TOKEN", "CHEMAPP_DESKTOP_INSTANCE", "CHEMAPP_ACCESS_TOKEN", "CHEMAPP_ADMIN_TOKEN"]
    for name in required:
        if not re.fullmatch(r"[a-f0-9]{64}", os.environ.get(name, "")):
            raise RuntimeError(f"{name} must be a launch-generated 256-bit value")
    if len({os.environ[name] for name in required}) != len(required):
        raise RuntimeError("Desktop launch values must be independent")
    frontend = Path(os.environ["CHEMAPP_DESKTOP_FRONTEND"]).resolve(strict=True)
    if not (frontend / "index.html").is_file():
        raise RuntimeError("Build the frontend before starting the desktop app")
    data = Path(os.environ["CHEMAPP_DESKTOP_DATA_DIR"]).resolve()
    data.mkdir(parents=True, exist_ok=True)
    os.environ.update({
        "CHEMAPP_DB_PATH": str(data / "chemapp.db"),
        "CHEMAPP_TRAINING_LOCK_PATH": str(data / "training.lock"),
        "CHEMAPP_LOCAL_ACCESS_BYPASS": "0",
        "CHEMAPP_LOCAL_ADMIN_BYPASS": "0",
        "CHEMAPP_TRUSTED_HOSTS": "127.0.0.1",
        "CHEMAPP_PROTECT_READY": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    # All relative writable paths belong to user data, never the install folder.
    os.chdir(data)
    return frontend


def protect_windows_process_tree():
    """Close-on-process-exit job kills descendants even if Python crashes."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("ProcessTime", ctypes.c_int64), ("JobTime", ctypes.c_int64),
            ("Flags", wintypes.DWORD), ("MinWorkingSet", ctypes.c_size_t),
            ("MaxWorkingSet", ctypes.c_size_t), ("ActiveProcesses", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t), ("Priority", wintypes.DWORD),
            ("Scheduling", wintypes.DWORD),
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("Basic", BasicLimits), ("IoCounters", ctypes.c_uint64 * 6),
            ("ProcessMemory", ctypes.c_size_t), ("JobMemory", ctypes.c_size_t),
            ("PeakProcessMemory", ctypes.c_size_t), ("PeakJobMemory", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    handle = kernel.CreateJobObjectW(None, None)
    limits = ExtendedLimits()
    limits.Basic.Flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not handle or not kernel.SetInformationJobObject(handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
        raise ctypes.WinError(ctypes.get_last_error())
    # Intentionally retained until OS process teardown; closing it kills this job.
    return handle


def hard_stop_owned_tree():
    if sys.platform != "win32" and os.getpgrp() == os.getpid():
        os.killpg(os.getpgrp(), signal.SIGKILL)
    os._exit(1)  # Windows job closes on process exit, including all descendants.


def run():
    frontend = configuration()
    windows_job = protect_windows_process_tree()
    owner_gone = threading.Event()
    server = None

    def watch_owner():
        try:
            while sys.stdin.buffer.read(1):
                pass
        finally:
            owner_gone.set()
            if server is not None:
                server.should_exit = True
            # Survives loss of Electron itself and covers blocked startup/imports
            # or long analysis requests after a best-effort graceful shutdown.
            timer = threading.Timer(7, hard_stop_owned_tree)
            timer.daemon = True
            timer.start()

    threading.Thread(target=watch_owner, name="desktop-owner", daemon=True).start()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Bind once and pass this exact socket to uvicorn: no free-port race.
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    origin = f"http://127.0.0.1:{port}"
    os.environ["CHEMAPP_CORS_ORIGINS"] = origin

    import uvicorn
    from starlette.staticfiles import StaticFiles
    from app.main import app

    # Existing UI/API URLs remain relative and work in both web and desktop modes.
    app.mount("/", StaticFiles(directory=str(frontend), html=True), name="desktop-ui")
    wrapped = DesktopBoundary(
        app, token=os.environ["CHEMAPP_DESKTOP_TOKEN"], origin=origin,
        instance=os.environ["CHEMAPP_DESKTOP_INSTANCE"],
    )
    server = uvicorn.Server(uvicorn.Config(
        wrapped, host="127.0.0.1", port=port, workers=1, proxy_headers=False,
        access_log=False, log_level="info", timeout_graceful_shutdown=5,
        ws="none",
    ))

    if owner_gone.is_set():
        server.should_exit = True
    # Keep the job handle alive until process teardown.
    _ = windows_job
    print("CHEMAPP_DESKTOP_PORT " + json.dumps({
        "port": port, "instance": os.environ["CHEMAPP_DESKTOP_INSTANCE"],
    }), flush=True)
    try:
        server.run(sockets=[listener])
    finally:
        listener.close()


if __name__ == "__main__":
    run()
