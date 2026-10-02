"""Pinned official NMR2Struct assets and crash-safe, bounded installation.

No user-supplied URL, filename or destination enters the download API. The
application only publishes an exact-size SHA-256-verified file with os.replace;
partial files never become checkpoints. No checkpoint is deserialized here.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
import threading
import time
from typing import Any
import uuid

import requests
from urllib3.exceptions import HTTPError as TransportError

logger = logging.getLogger(__name__)
SOURCE_REPO = "https://github.com/MarklandGroup/NMR2Struct"
REVISION = "2aee0a1e6c1a13ed89d8f1b774e9cf79af639bfe"
VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "nmr2struct"
_CHUNK_SIZE = 64 * 1024
# Cooperative transfer budget, checked during body reads. Connection/header
# waits use Requests connect/read timeouts; OS DNS resolution is platform-bound.
_MAX_SECONDS = 900
_ACTIVE = frozenset({"queued", "downloading", "verifying", "cancelling"})


@dataclass(frozen=True)
class Asset:
    id: str
    filename: str
    size_bytes: int
    sha256: str

    @property
    def url(self) -> str:
        return f"https://raw.githubusercontent.com/MarklandGroup/NMR2Struct/{REVISION}/checkpoints/{self.filename}"


ASSETS = {
    "cnmr_only": Asset("cnmr_only", "cnmr_only_checkpoint.pt", 101613138,
                       "1c14362a0b24c951641b14341c075457d9f1a2020ae0eabc0db2f51d358e3d40"),
    "hnmr_only": Asset("hnmr_only", "hnmr_only_checkpoint.pt", 102260771,
                       "4a9a0c6e114e296bd06b8fb3c722b7628f66c0c21109dc71b62db2ddfea121cf"),
    "multitask": Asset("multitask", "multitask_checkpoint.pt", 102344616,
                       "da28998d7993eded4d7298969a61a8b4952191ca7fdcf63b3f8182ae81bdf630"),
}


def weights_directory() -> Path:
    override = os.environ.get("CHEMAPP_NMR2STRUCT_WEIGHTS_DIR")
    if override:
        return Path(override).expanduser().absolute()
    desktop = os.environ.get("CHEMAPP_DESKTOP_DATA_DIR")
    if desktop:
        return Path(desktop).expanduser().absolute() / "models" / "nmr2struct"
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local" / "share")))
    return base.absolute() / "chemapp" / "models" / "nmr2struct"


_verify_lock = threading.RLock()
_verified: dict[tuple[Any, ...], bool] = {}


def verified_file(path: Path, asset: Asset) -> bool:
    """Cache verification by file identity, including nanosecond change time."""
    try:
        if path.is_symlink() or not path.is_file():
            return False
        stat = path.stat()
        if stat.st_size != asset.size_bytes:
            return False
        key = (str(path.absolute()), stat.st_dev, stat.st_ino, stat.st_size,
               stat.st_mtime_ns, stat.st_ctime_ns, asset.sha256)
        with _verify_lock:
            if key in _verified:
                return _verified[key]
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            after = path.stat()
            good = (digest.hexdigest() == asset.sha256 and
                    (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns))
            if len(_verified) >= 64:
                _verified.clear()
            _verified[key] = good
            return good
    except OSError:
        return False


def checkpoint_path(variant: str, vendor_dir: Path | None = None) -> Path | None:
    """Verified user assets take priority; preserve legacy bundled checkpoints.

    Bundled files retain the generator's historical presence-only preflight;
    they are still loaded with torch weights_only=True. The download inventory
    explicitly verifies both sources and never calls a corrupt file ready.
    """
    asset = ASSETS.get(variant)
    if asset is None:
        return None
    downloaded = weights_directory() / asset.filename
    if verified_file(downloaded, asset):
        return downloaded
    bundled = (vendor_dir if vendor_dir is not None else VENDOR_DIR) / "checkpoints" / asset.filename
    return bundled if bundled.is_file() else None


class DownloadBusy(Exception):
    pass


class DownloadFailure(Exception):
    """Only fixed, safe messages from this class are sent to the client."""


class _Cancelled(Exception):
    pass


class _DirectoryLock:
    """OS advisory lock released on process death; no stale-PID guessing."""
    def __init__(self, directory: Path):
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(directory / ".download.lock", flags, 0o600)
        self.handle = os.fdopen(fd, "r+b")
        try:
            if sys.platform == "win32":
                import msvcrt
                if self.handle.seek(0, os.SEEK_END) == 0:
                    self.handle.write(b"0")
                    self.handle.flush()
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise DownloadBusy("Another ChemApp process is downloading weights") from exc

    def close(self):
        if not self.handle.closed:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    self.handle.seek(0)
                    msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                # Closing the handle also releases the operating-system lock.
                pass
            finally:
                self.handle.close()


class DownloadManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._jobs: dict[str, dict[str, Any]] = {}
        self._latest: dict[str, str] = {}
        self._cancel: dict[str, threading.Event] = {}

    def get(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return dict(self._jobs[job_id])

    def inventory(self) -> dict[str, Any]:
        directory = weights_directory()
        rows = []
        with self._lock:
            jobs = {key: dict(self._jobs[value]) for key, value in self._latest.items()}
        for asset in ASSETS.values():
            paths = [directory / asset.filename, VENDOR_DIR / "checkpoints" / asset.filename]
            ready = any(verified_file(path, asset) for path in paths)
            size = 0
            exists = False
            for path in paths:
                try:
                    if path.is_file() or path.is_symlink():
                        exists = True
                        size = max(size, path.stat().st_size)
                except OSError:
                    exists = True
            rows.append({"id": asset.id, "filename": asset.filename,
                         "size_bytes": asset.size_bytes, "sha256": asset.sha256,
                         "status": "ready" if ready else "invalid" if exists else "missing",
                         "installed_bytes": asset.size_bytes if ready else size,
                         "job": jobs.get(asset.id)})
        return {"source_repo": SOURCE_REPO, "revision": REVISION,
                "storage_dir": str(directory), "assets": rows}

    def start(self, asset_id: str) -> dict[str, Any]:
        if asset_id not in ASSETS:
            raise KeyError(asset_id)
        asset = ASSETS[asset_id]
        with self._lock:
            for job in self._jobs.values():
                if job["status"] in _ACTIVE:
                    if job["asset_id"] == asset_id:
                        return dict(job)
                    raise DownloadBusy("Wait for the current download or cancel it first")
            # Limit memory use while keeping the last result for every asset.
            latest = set(self._latest.values())
            while len(self._jobs) >= 32:
                oldest = next((key for key in self._jobs if key not in latest), None)
                if oldest is None:
                    break
                self._jobs.pop(oldest)
                self._cancel.pop(oldest, None)
            directory = weights_directory()
            ready = any(verified_file(path, asset) for path in (
                directory / asset.filename, VENDOR_DIR / "checkpoints" / asset.filename))
            guard = None
            if not ready:
                try:
                    directory.mkdir(parents=True, exist_ok=True)
                    guard = _DirectoryLock(directory)
                    # Another process may have completed while we checked.
                    if verified_file(directory / asset.filename, asset):
                        guard.close()
                        guard = None
                        ready = True
                except DownloadBusy:
                    raise
                except OSError as exc:
                    raise DownloadFailure("Cannot write the model storage directory") from exc
            job_id = uuid.uuid4().hex
            job = {"id": job_id, "asset_id": asset_id,
                   "status": "completed" if ready else "queued",
                   "downloaded_bytes": asset.size_bytes if ready else 0,
                   "total_bytes": asset.size_bytes, "error": None}
            self._jobs[job_id] = job
            self._latest[asset_id] = job_id
            self._cancel[job_id] = threading.Event()
            if not ready:
                try:
                    threading.Thread(target=self._run, args=(job_id, asset, directory, guard),
                                     name="nmr2struct-download", daemon=True).start()
                except Exception:
                    guard.close()
                    job.update(status="error", error="Unable to start the download")
            return dict(job)

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self.get(job_id)
            if job["status"] in _ACTIVE:
                self._cancel[job_id].set()
                self._jobs[job_id]["status"] = "cancelling"
            return self.get(job_id)

    def _update(self, job_id: str, **values):
        with self._lock:
            # Cancellation stays visible until the worker acknowledges it.
            if self._cancel[job_id].is_set() and values.get("status") in _ACTIVE:
                values["status"] = "cancelling"
            self._jobs[job_id].update(values)

    def _run(self, job_id: str, asset: Asset, directory: Path, guard: _DirectoryLock):
        temporary = None
        cancel = self._cancel[job_id]
        started = time.monotonic()
        outcome: dict[str, Any] | None = None

        def check():
            if cancel.is_set():
                raise _Cancelled()
            if time.monotonic() - started > _MAX_SECONDS:
                raise DownloadFailure("Download timed out. Please retry")

        try:
            check()
            # Exclusive directory lock ensures these are abandoned by a previous
            # crashed process, never an active writer's staging files.
            for stale in directory.glob(".nmr2struct-*.part"):
                stale.unlink(missing_ok=True)
            if shutil.disk_usage(directory).free < asset.size_bytes + 1024 * 1024:
                raise DownloadFailure("Not enough free disk space for this model")
            self._update(job_id, status="downloading")
            fd, name = tempfile.mkstemp(prefix=".nmr2struct-", suffix=".part", dir=directory)
            temporary = Path(name)
            count = 0
            digest = hashlib.sha256()
            with os.fdopen(fd, "wb") as output, requests.Session() as session:
                # Never attach .netrc credentials to this public fetch. Keep
                # operator-configured proxy/CA routing, including enterprise
                # networks, without disabling TLS verification. Redirects fail closed.
                session.trust_env = False
                with session.get(asset.url, stream=True, timeout=(10, 5),
                                 allow_redirects=False,
                                 proxies=requests.utils.get_environ_proxies(asset.url),
                                 verify=(os.environ.get("REQUESTS_CA_BUNDLE") or
                                         os.environ.get("CURL_CA_BUNDLE") or True),
                                 headers={"Accept-Encoding": "identity", "User-Agent": "ChemApp-model-downloader/1"}) as response:
                    if response.status_code != 200:
                        raise DownloadFailure("Official download returned an unexpected HTTP response. Please retry")
                    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
                        raise DownloadFailure("Official download returned an unsupported encoding")
                    length = response.headers.get("Content-Length")
                    if length is not None and length != str(asset.size_bytes):
                        raise DownloadFailure("Official file size did not match the pinned model")
                    # read1 returns available data after at most one socket
                    # read: a trickling peer cannot hold a 64 KiB buffered read
                    # forever while bypassing cancel and the total deadline.
                    while True:
                        check()
                        chunk = response.raw.read1(_CHUNK_SIZE, decode_content=False)
                        check()
                        if not chunk:
                            break
                        count += len(chunk)
                        if count > asset.size_bytes:
                            raise DownloadFailure("Official file exceeded the pinned model size")
                        output.write(chunk)
                        digest.update(chunk)
                        self._update(job_id, downloaded_bytes=count)
                check()
                self._update(job_id, status="verifying")
                if count != asset.size_bytes or digest.hexdigest() != asset.sha256:
                    raise DownloadFailure("Model checksum verification failed. Please retry")
                output.flush()
                os.fsync(output.fileno())
            # Serialize final publication against cancellation: when cancellation
            # is acknowledged before this point no target can be replaced.
            with self._lock:
                check()
                os.replace(temporary, directory / asset.filename)
                temporary = None
                if sys.platform != "win32":
                    try:
                        directory_fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                        try:
                            os.fsync(directory_fd)
                        finally:
                            os.close(directory_fd)
                    except OSError:
                        # Publication already succeeded. Some filesystems do not
                        # support directory fsync; never misreport this as a
                        # cancelled/failed installation after replacing bytes.
                        logger.warning("Model installed; directory durability could not be confirmed")
                guard.close()
                self._jobs[job_id].update(status="completed", downloaded_bytes=count)
        except _Cancelled:
            outcome = {"status": "cancelled", "error": None}
        except DownloadFailure as exc:
            outcome = {"status": "error", "error": str(exc)}
        except (requests.RequestException, TransportError):
            outcome = {"status": "error", "error": "Cannot reach the official repository. Check your connection and retry"}
        except Exception:
            logger.exception("NMR2Struct download failed for %s", asset.id)
            outcome = {"status": "error", "error": "Could not save the model. Check disk space and permissions, then retry"}
        finally:
            try:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
            except OSError:
                logger.warning("Could not remove abandoned NMR2Struct staging file")
            finally:
                with self._lock:
                    guard.close()
                    if outcome is not None:
                        if cancel.is_set():
                            outcome = {"status": "cancelled", "error": None}
                        self._jobs[job_id].update(outcome)


manager = DownloadManager()
