from __future__ import annotations

import logging
import json
import os
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.ml.predictor import get_model_status

router = APIRouter(prefix="/api/ml", tags=["ml"])
logger = logging.getLogger("chemapp.ml")

_training_thread: threading.Thread | None = None
_training_progress: dict = {"status": "idle", "message": ""}
_training_state_lock = threading.RLock()


def _set_training_progress(**values):
    global _training_progress
    with _training_state_lock:
        _training_progress = values


def _update_training_progress(**values):
    with _training_state_lock:
        _training_progress.update(values)


def _training_progress_snapshot() -> dict:
    with _training_state_lock:
        return dict(_training_progress)


def _training_lock_path() -> Path:
    configured = os.environ.get("CHEMAPP_TRAINING_LOCK_PATH")
    if configured:
        return Path(configured)
    db_path = Path(os.environ.get("CHEMAPP_DB_PATH", str(Path("data") / "chemapp.db")))
    return db_path.with_suffix(db_path.suffix + ".training.lock")


def _acquire_training_lock(job_id: str) -> Path | None:
    path = _training_lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            if attempt == 0 and _stale_training_lock(path):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove stale training lock %s", path)
                    return None
                continue
            return None
        try:
            os.write(
                fd,
                json.dumps({"job_id": job_id, "pid": os.getpid()}).encode("utf-8"),
            )
        finally:
            os.close(fd)
        return path
    return None


def _process_alive(pid: int) -> bool:
    if pid == os.getpid():
        return True
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return int(exit_code.value) == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


def _stale_training_lock(path: Path) -> bool:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return True
    pid = raw.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return True
    return not _process_alive(pid)


def _release_training_lock(path: Path, job_id: str) -> None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("job_id") == job_id:
            path.unlink(missing_ok=True)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        logger.warning("Could not release ML training lock %s", path)


@router.get("/status")
def ml_status():
    return {**get_model_status(), "training": _training_progress_snapshot()}


@router.post("/predict/{sid}")
def ml_predict(sid: str):
    raise HTTPException(
        410,
        "Legacy /api/ml/predict is removed; use /api/ml/elucidate/predict "
        "with the hybrid forward pipeline.",
    )


@router.post("/predict/dual/{sid}")
def ml_predict_dual(sid: str):
    raise HTTPException(
        410,
        "Legacy /api/ml/predict/dual is removed; use /api/ml/elucidate/predict.",
    )


@router.post("/train")
def ml_train():
    raise HTTPException(
        410,
        "Legacy /api/ml/train is removed; training runs are executed by the "
        "forward_v1 / CSP5 pipelines with versioned run specs.",
    )


@router.get("/download")
def ml_download():
    raise HTTPException(
        410,
        "Legacy /api/ml/download is removed; dataset acquisition is managed "
        "by the versioned data-governance tooling.",
    )
