from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from app.api.deps import get_store
from app.analysis.helpers import analyze_if_missing
from app.integration import InferenceEngine, build_report

router = APIRouter(prefix="/api", tags=["inference"])
SpectrumId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[SpectrumId] = Field(min_length=1, max_length=10)

    @field_validator("ids")
    @classmethod
    def unique_ids(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("Spectrum IDs must be unique")
        return value


@router.post("/inference")
def run_inference(payload: InferenceRequest):
    ids = payload.ids
    store = get_store()
    results: dict[str, object] = {}
    fetched: list[tuple[str, Any]] = []

    for sid in ids:
        stored = store.get(sid)
        if stored is None:
            raise HTTPException(404, f"Spectrum {sid} not found")
        result = stored.result
        if result is None:
            # Inference is read-only. Missing analyses are calculated in
            # memory and are not persisted as a side effect.
            result = analyze_if_missing(stored.spectrum)
        if result is not None:
            results[sid] = result
            fetched.append((sid, stored))

    engine = InferenceEngine()
    inference = engine.analyze(results)

    sample_name = ", ".join(
        Path(s.spectrum.source_file).name if s.spectrum.source_file else sid
        for sid, s in fetched
    )

    techniques = list({s.spectrum.technique.value for _, s in fetched})

    report = build_report(sample_name, techniques, inference.technique_results, inference)

    return report.to_dict()


@router.get("/report/{sid}")
def get_report(sid: SpectrumId):
    raise HTTPException(501, "Report retrieval by ID not yet implemented. Use POST /api/inference.")
