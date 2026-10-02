"""Admin-only installs of allowlisted official NMR2Struct checkpoints."""
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from app.api.deps import require_admin
from app.ml.nmr2struct_weights import DownloadBusy, DownloadFailure, manager

router = APIRouter(prefix="/api/ml/nmr2struct/weights", tags=["model-weights"])


class DownloadRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: Literal["cnmr_only", "hnmr_only", "multitask"]


@router.get("")
def inventory():
    return manager.inventory()


@router.post("/downloads", status_code=202, dependencies=[Depends(require_admin)])
def start_download(payload: DownloadRequest):
    try:
        return manager.start(payload.asset_id)
    except DownloadBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except DownloadFailure as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/downloads/{job_id}")
def download_status(job_id: str):
    try:
        return manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Download not found") from exc


@router.post("/downloads/{job_id}/cancel", dependencies=[Depends(require_admin)])
def cancel_download(job_id: str):
    try:
        return manager.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Download not found") from exc
