from __future__ import annotations

import math
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from app.api.deps import get_store, require_admin
from app.standards import get_standard_database

router = APIRouter(prefix="/api/standards", tags=["standards"])
SpectrumId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
ShortText = Annotated[str, StringConstraints(max_length=256)]


def _finite_tree(value: Any, depth: int = 0) -> Any:
    if depth > 8:
        raise ValueError("Standard payload is too deeply nested")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("NaN and Infinity are not allowed")
    if isinstance(value, str) and len(value) > 20000:
        raise ValueError("Standard text value is too long")
    if isinstance(value, list):
        if len(value) > 5000:
            raise ValueError("Standard list is too long")
        for item in value:
            _finite_tree(item, depth + 1)
    elif isinstance(value, dict):
        if len(value) > 1000:
            raise ValueError("Standard object has too many fields")
        for item in value.values():
            _finite_tree(item, depth + 1)
    return value


class StandardRecordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.-]{1,128}$")] | None = None
    name: Annotated[str, StringConstraints(min_length=1, max_length=256)]
    technique: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    formula: ShortText = ""
    source: ShortText = "User"
    tags: list[ShortText] = Field(default_factory=list, max_length=100)
    peaks: list[dict[str, Any]] = Field(default_factory=list, max_length=5000)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("peaks", "metadata")
    @classmethod
    def validate_nested_values(cls, value):
        return _finite_tree(value)


class StandardMatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    technique: Annotated[str, StringConstraints(min_length=1, max_length=32)]
    peaks: list[dict[str, Any]] = Field(min_length=1, max_length=5000)
    tolerance: float = Field(default=0.05, allow_inf_nan=False, gt=0, le=100)

    @field_validator("peaks")
    @classmethod
    def validate_peaks(cls, value):
        return _finite_tree(value)


class StandardToleranceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tolerance: float = Field(default=0.05, allow_inf_nan=False, gt=0, le=100)


@router.get("")
def list_standards(
    technique: Annotated[str | None, Query(max_length=32)] = None,
    query: Annotated[str | None, Query(max_length=256)] = None,
):
    return {"records": get_standard_database().list_records(technique=technique, query=query)}


@router.post("", dependencies=[Depends(require_admin)])
def add_standard(payload: StandardRecordRequest):
    try:
        return get_standard_database().add_record(payload.model_dump())
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@router.post("/match")
def match_standard(payload: StandardMatchRequest):
    return {
        "matches": get_standard_database().match_peaks(
            payload.technique,
            payload.peaks,
            tolerance=payload.tolerance,
        )
    }


@router.post("/match/{sid}")
def match_spectrum_to_standards(sid: SpectrumId, payload: StandardToleranceRequest | None = None):
    store = get_store()
    stored = store.get(sid)
    if stored is None:
        raise HTTPException(404, f"Spectrum {sid} not found")
    result = stored.result
    if result is None:
        raise HTTPException(404, "Analyze the spectrum before matching standards")
    peaks = [p.__dict__ for p in result.peaks]
    if hasattr(result, "integrals") and result.integrals:
        peaks.extend(result.integrals)
    tolerance = payload.tolerance if payload else 0.05
    matches = get_standard_database().match_peaks(stored.spectrum.technique.value, peaks, tolerance=tolerance)
    return {"spectrum_id": sid, "matches": matches}
