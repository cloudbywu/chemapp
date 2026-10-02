from __future__ import annotations

import logging
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)
import numpy as np
import csv
import io

from app.analysis import AnalyzerRegistry
from app.analysis.models import AnalysisResult
from app.analysis.quality import assess_quality
from app.api.deps import get_store
from app.api.json_limits import validate_json_tree
from app.api.store import ResultVersionSourceConflict, RevisionConflict
from app.core.models import Peak

router = APIRouter(prefix="/api", tags=["analysis"])
logger = logging.getLogger("chemapp.analysis")
SpectrumId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]
NonNegativeRevision = Annotated[int, Field(ge=0)]


def _validate_json_tree(value: Any) -> Any:
    # Analysis payloads legitimately reach 10k peaks/integrals/multiplets
    # (each with several fields), so they keep the looser limits; see
    # app.api.json_limits for the shared fail-closed implementation and the
    # reasons both routes do not share one limit set.
    return validate_json_tree(
        value,
        max_nodes=20000,
        max_depth=12,
        max_string_length=100000,
        max_object_fields=5000,
        max_list_length=10000,
    )


# Metric keys a client may write when confirming a manual result. The review
# UI round-trips the measurement values produced by the analyzers, so those
# stay client-writable; server-authoritative and provenance keys — quality,
# manual_confirmed, manual_version, ai_modified, restored_from_version — are
# computed server-side below (or by restore/AI endpoints) and must never be
# accepted from the client. Anything outside this set is dropped and logged.
_CLIENT_METRIC_KEYS = frozenset({
    # NMR
    "n_peaks",
    "n_multiplets",
    "noise_level",
    "baseline",
    "baseline_method",
    "baseline_span",
    "total_integral",
    "solvent_reference",
    "reference_corrected",
    "phase_status",
    "signed_intensity",
    # UV-Vis
    "lambda_max",
    "absorbance_range",
    "baseline_corrected",
    "calibration_curve",
    # Fluorescence
    "excitation_peak_nm",
    "emission_peak_nm",
    "stokes_shift_nm",
    "stokes_shift_cm1",
    "sub_type",
    "normalized",
    # XRD
    "wavelength_a",
    "two_theta_range",
    "max_intensity",
    "crystallinity_percent",
    "dominant_phase",
    "matched_phases",
    "rietveld_rwp",
    # HPLC
    "time_range_min",
    "max_intensity_mau",
    "total_area",
    "n_channels",
    "channel_peaks",
    "integration_source",
    "instrument_result_source",
    "analysis_options",
    "integration_events",
    # Electrochemistry (CV/EIS)
    "scan_rate_v_s",
    "ep_anodic_v",
    "ep_cathodic_v",
    "ip_anodic_a",
    "ip_cathodic_a",
    "delta_ep_v",
    "e_formal_v",
    "ip_ratio",
    "q_forward_c",
    "q_reverse_c",
    "instrument_ep_v",
    "instrument_ip_a",
    "rs_ohm",
    "rct_ohm",
    "zd_max_ohm",
    "z_range_real",
    "z_range_imag",
    "error",
})


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    options: dict[str, Any] = Field(default_factory=dict)
    force_overwrite: bool = False
    expected_revision: int | None = Field(default=None, ge=0)

    @field_validator("options")
    @classmethod
    def validate_options(cls, value):
        return _validate_json_tree(value)


class AnalyzeBatchRequest(AnalyzeRequest):
    ids: list[SpectrumId] = Field(min_length=1, max_length=50)
    expected_revisions: dict[SpectrumId, NonNegativeRevision] = Field(
        default_factory=dict,
        max_length=50,
    )

    @model_validator(mode="after")
    def validate_revisions(self):
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Spectrum IDs must be unique")
        unknown = sorted(set(self.expected_revisions) - set(self.ids))
        if unknown:
            raise ValueError(
                "expected_revisions contains IDs outside ids: " + ", ".join(unknown)
            )
        if self.expected_revision is not None:
            if len(self.ids) != 1:
                raise ValueError(
                    "The legacy expected_revision field is only valid for a "
                    "single-ID batch; use expected_revisions for multiple IDs"
                )
            mapped = self.expected_revisions.get(self.ids[0])
            if mapped is not None and mapped != self.expected_revision:
                raise ValueError(
                    "expected_revision conflicts with expected_revisions for the ID"
                )
        return self


class IntegrationRangeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start: float = Field(allow_inf_nan=False, ge=-1e9, le=1e9)
    end: float = Field(allow_inf_nan=False, ge=-1e9, le=1e9)
    center: float | None = Field(default=None, allow_inf_nan=False, ge=-1e9, le=1e9)
    channel: Annotated[str, StringConstraints(max_length=256)] | None = None
    baseline: Literal["none", "linear"] | None = None


class IntegrationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ranges: list[IntegrationRangeRequest] = Field(min_length=1, max_length=5000)


class ManualPeakRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    position: float = Field(allow_inf_nan=False, ge=-1e9, le=1e9)
    intensity: float = Field(allow_inf_nan=False, ge=-1e15, le=1e15)
    area: float | None = Field(default=None, allow_inf_nan=False, ge=-1e30, le=1e30)
    width: float | None = Field(default=None, allow_inf_nan=False, ge=0, le=1e9)
    assignment: Annotated[str, StringConstraints(max_length=1000)] = ""
    multiplicity: Annotated[str, StringConstraints(max_length=64)] = ""
    coupling_constant: float | None = Field(default=None, allow_inf_nan=False, ge=0, le=1e6)


class ManualResultRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    peaks: list[ManualPeakRequest] | None = Field(default=None, max_length=10000)
    integrals: list[dict[str, Any]] | None = Field(default=None, max_length=10000)
    multiplets: list[dict[str, Any]] | None = Field(default=None, max_length=10000)
    metrics: dict[str, Any] | None = None
    channel_peaks: dict[str, Any] | None = None
    summary: Annotated[str, StringConstraints(max_length=100000)] | None = None
    note: Annotated[str, StringConstraints(max_length=1000)] | None = None
    expected_revision: int | None = Field(default=None, ge=0)

    @field_validator("integrals", "multiplets", "metrics", "channel_peaks")
    @classmethod
    def validate_nested_values(cls, value):
        return _validate_json_tree(value)


class RestoreResultRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int | None = Field(default=None, ge=0)


class BatchSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[SpectrumId] = Field(default_factory=list, max_length=1000)

    @field_validator("ids")
    @classmethod
    def unique_ids(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("Spectrum IDs must be unique")
        return value


class CompareSpectraRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id1: SpectrumId
    id2: SpectrumId

    @field_validator("id2")
    @classmethod
    def distinct_ids(cls, value, info):
        if value == info.data.get("id1"):
            raise ValueError("Choose two different spectra")
        return value


class HPLCComparisonRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ids: list[SpectrumId] = Field(min_length=2, max_length=100)
    channel: Annotated[str, StringConstraints(max_length=256)] | None = None
    rt_tolerance: float = Field(default=0.08, allow_inf_nan=False, gt=0, le=10)

    @field_validator("ids")
    @classmethod
    def unique_ids(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("HPLC spectrum IDs must be unique")
        return value


def _conflict_response(exc: RevisionConflict) -> HTTPException:
    return HTTPException(
        409,
        detail={
            "code": "revision_conflict",
            "message": "The result changed after it was loaded",
            "current_revision": exc.current_revision,
        },
    )


def _require_result_revision(value: int | None) -> int:
    if value is None:
        raise HTTPException(
            428,
            detail={
                "code": "expected_revision_required",
                "message": (
                    "expected_revision is required for analysis result writes; "
                    "use 0 when no result has been saved yet"
                ),
            },
        )
    return int(value)


def _result_response(stored) -> dict[str, Any]:
    data = stored.result.to_dict() if stored.result is not None else {}
    data["result_revision"] = stored.result_revision
    return data


def _csv_safe(value: Any) -> Any:
    if isinstance(value, str) and value.startswith(("\t", "\r", "\n", "=", "+", "-", "@")):
        return "'" + value
    return value


def _match_sample_peaks(
    reference_peaks: list[dict[str, Any]],
    sample_peaks: list[dict[str, Any]],
    rt_tolerance: float,
) -> list[dict[str, Any] | None]:
    """One-to-one RT matching: each sample peak is consumed at most once."""

    sample_sorted = sorted(
        sample_peaks, key=lambda peak: float(peak.get("position", 0))
    )
    used: set[int] = set()
    matched: list[dict[str, Any] | None] = []
    pointer = 0
    for ref_peak in reference_peaks:
        ref_rt = float(ref_peak.get("position", 0))
        while (
            pointer < len(sample_sorted)
            and float(sample_sorted[pointer].get("position", 0))
            < ref_rt - rt_tolerance
        ):
            pointer += 1
        best_index: int | None = None
        best_distance = rt_tolerance + 1.0
        scan = pointer
        while (
            scan < len(sample_sorted)
            and float(sample_sorted[scan].get("position", 0))
            <= ref_rt + rt_tolerance
        ):
            if scan not in used:
                distance = abs(
                    float(sample_sorted[scan].get("position", 0)) - ref_rt
                )
                if distance < best_distance:
                    best_distance = distance
                    best_index = scan
            scan += 1
        if best_index is None:
            matched.append(None)
            continue
        used.add(best_index)
        matched.append(sample_sorted[best_index])
    return matched


def _technique_key(value: str) -> str | None:
    technique = value.lower().replace("-", "")
    return {
        "nmr": "nmr",
        "uvvis": "uvvis",
        "uv-vis": "uvvis",
        "fluorescence": "fluorescence",
        "xrd": "xrd",
        "hplc": "hplc",
        "electrochem": "electrochem",
    }.get(technique)


def _analyze_stored(stored, options: dict | None = None):
    technique_key = _technique_key(stored.spectrum.technique.value)

    if technique_key is None:
        raise HTTPException(400, f"No analyzer for technique: {stored.spectrum.technique.value}")

    analyzer_cls = AnalyzerRegistry.get(technique_key)
    analyzer = analyzer_cls()
    result = analyzer.analyze(stored.spectrum, options=options)
    result.metrics["quality"] = assess_quality(stored.spectrum, result)
    return result


def _ensure_stored_result(stored) -> AnalysisResult:
    if stored.result is not None:
        return stored.result
    return _analyze_stored(stored)


def _baseline_corrected_segment(x: np.ndarray, y: np.ndarray, start: float, end: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = sorted((start, end))
    mask = (x >= lo) & (x <= hi)
    seg_x = x[mask]
    seg_y = y[mask]
    if len(seg_x) < 2:
        return seg_x, seg_y, np.zeros_like(seg_y)

    baseline = np.linspace(float(seg_y[0]), float(seg_y[-1]), len(seg_y))
    corrected = seg_y - baseline
    return seg_x, seg_y, corrected


def _integrate_range(
    x: np.ndarray,
    y: np.ndarray,
    item: dict,
    technique: str,
) -> dict:
    start = float(item["start"])
    end = float(item["end"])
    center = float(item.get("center", (start + end) / 2))
    baseline_mode = item.get("baseline", "linear" if technique == "HPLC" else "none")

    if baseline_mode == "linear":
        seg_x, seg_y, corrected = _baseline_corrected_segment(x, y, start, end)
        area_y = np.maximum(corrected, 0) if technique == "HPLC" else corrected
        area = float(np.trapezoid(area_y, seg_x)) if len(seg_x) > 1 else 0.0
        apex_idx = int(np.argmax(corrected)) if len(corrected) else 0
        height = float(corrected[apex_idx]) if len(corrected) else 0.0
        apex_time = float(seg_x[apex_idx]) if len(seg_x) else center
        baseline_start = float(seg_y[0]) if len(seg_y) else 0.0
        baseline_end = float(seg_y[-1]) if len(seg_y) else 0.0
    else:
        lo, hi = sorted((start, end))
        mask = (x >= lo) & (x <= hi)
        seg_x = x[mask]
        seg_y = y[mask]
        area = float(np.trapezoid(seg_y, seg_x)) if len(seg_x) > 1 else 0.0
        apex_idx = int(np.argmax(seg_y)) if len(seg_y) else 0
        height = float(seg_y[apex_idx]) if len(seg_y) else 0.0
        apex_time = float(seg_x[apex_idx]) if len(seg_x) else center
        baseline_start = 0.0
        baseline_end = 0.0

    if technique == "HPLC":
        area *= 60.0

    return {
        **item,
        "center": round(center, 4),
        "start": round(start, 4),
        "end": round(end, 4),
        "apex_time": round(apex_time, 4),
        "area": round(area, 4),
        "height": round(height, 4),
        "width": round(abs(end - start), 4),
        "baseline_start": round(baseline_start, 4),
        "baseline_end": round(baseline_end, 4),
        "baseline": baseline_mode,
    }


@router.post("/analyze/batch")
def analyze_batch(payload: AnalyzeBatchRequest, request: Request):
    ids = payload.ids
    options = payload.options

    store = get_store()
    rows = []
    errors = []
    for sid in ids:
        stored = store.get(sid)
        if stored is None:
            errors.append({"id": sid, "error": "Spectrum not found"})
            continue
        expected_revision = payload.expected_revisions.get(sid)
        if expected_revision is None and len(ids) == 1:
            expected_revision = payload.expected_revision
        if expected_revision is None:
            errors.append(
                {
                    "id": sid,
                    "error": "expected_revision_required",
                    "message": (
                        "Provide expected_revisions[id]; use 0 when no result "
                        "has been saved yet"
                    ),
                    "current_revision": stored.result_revision,
                }
            )
            continue
        if int(expected_revision) != stored.result_revision:
            errors.append(
                {
                    "id": sid,
                    "error": "revision_conflict",
                    "message": "The result changed after it was loaded",
                    "current_revision": stored.result_revision,
                }
            )
            continue
        if (
            stored.result is not None
            and stored.result.metrics.get("manual_confirmed")
            and not payload.force_overwrite
        ):
            errors.append({"id": sid, "error": "Manual result is protected; set force_overwrite=true"})
            continue
        try:
            result = _analyze_stored(stored, options=options)
            updated = store.set_result(
                sid,
                result,
                expected_revision=int(expected_revision),
            )
            if updated is None:
                errors.append({"id": sid, "error": "Spectrum not found during save"})
                continue
            rows.append({
                "id": sid,
                "technique": stored.spectrum.technique.value,
                "name": stored.spectrum.metadata.name or stored.spectrum.source_file,
                "n_peaks": len(result.peaks),
                "summary": result.summary,
                "quality": result.metrics.get("quality"),
                "metrics": result.metrics,
                "result_revision": updated.result_revision,
            })
        except RevisionConflict as exc:
            errors.append(
                {
                    "id": sid,
                    "error": "revision_conflict",
                    "message": "The result changed while it was being analyzed",
                    "current_revision": exc.current_revision,
                }
            )
        except Exception:
            # Never reflect internal exception text back to the client; the
            # details (with the request id) stay in the server log.
            request_id = getattr(request.state, "request_id", "")
            logger.exception(
                "Batch analysis failed for spectrum %s (request_id=%s)",
                sid,
                request_id,
            )
            errors.append({
                "id": sid,
                "error": "Analysis failed for this spectrum",
            })

    return {"results": rows, "errors": errors}


@router.post("/quality/batch")
def batch_quality(
    payload: BatchSelectionRequest,
    limit: Annotated[int | None, Query(ge=1, le=5000)] = 1000,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    ids = payload.ids
    store = get_store()
    if ids:
        stored_items = [store.get(sid) for sid in ids]
    else:
        # Page over lightweight metadata first and deserialize each spectrum
        # on demand; list_all would deserialize every spectrum/result payload
        # in the page up front. Response shape is unchanged.
        stored_items = [
            store.get(item["id"])
            for item in store.list_metadata(
                limit=limit if limit is not None else 1000,
                offset=offset,
            )
        ]
    rows = []
    for stored in stored_items:
        if stored is None:
            continue
        result = _ensure_stored_result(stored)
        quality = result.metrics.get("quality") or assess_quality(stored.spectrum, result)
        rows.append({
            "id": stored.id,
            "technique": stored.spectrum.technique.value,
            "name": stored.spectrum.metadata.name or stored.spectrum.source_file,
            "status": quality.get("status"),
            "score": quality.get("score"),
            "warnings": quality.get("warnings", []),
            "info": quality.get("info", []),
            "n_peaks": len(result.peaks),
            "manual_confirmed": bool(result.metrics.get("manual_confirmed")),
        })
    counts = {
        "good": sum(1 for row in rows if row["status"] == "good"),
        "review": sum(1 for row in rows if row["status"] == "review"),
        "poor": sum(1 for row in rows if row["status"] == "poor"),
    }
    return {"items": rows, "counts": counts}


@router.post("/batch/workbench")
def batch_workbench(
    payload: BatchSelectionRequest,
    limit: Annotated[int | None, Query(ge=1, le=5000)] = 1000,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    ids = payload.ids
    store = get_store()
    if ids:
        stored_items = [store.get(sid) for sid in ids]
    else:
        # Page over lightweight metadata first and deserialize each spectrum
        # on demand; list_all would deserialize every spectrum/result payload
        # in the page up front. Response shape is unchanged.
        stored_items = [
            store.get(item["id"])
            for item in store.list_metadata(
                limit=limit if limit is not None else 1000,
                offset=offset,
            )
        ]
    rows = []
    technique_counts: dict[str, int] = {}
    quality_counts = {"good": 0, "review": 0, "poor": 0, "unknown": 0}
    manual_count = 0
    total_points = 0
    for stored in stored_items:
        if stored is None:
            continue
        result = _ensure_stored_result(stored)
        quality = result.metrics.get("quality") or assess_quality(stored.spectrum, result)
        status = str(quality.get("status") or "unknown")
        quality_counts[status] = quality_counts.get(status, 0) + 1
        technique = stored.spectrum.technique.value
        technique_counts[technique] = technique_counts.get(technique, 0) + 1
        manual = bool(result.metrics.get("manual_confirmed"))
        manual_count += int(manual)
        total_points += stored.spectrum.num_points
        rows.append({
            "id": stored.id,
            "name": stored.spectrum.metadata.name or stored.spectrum.source_file,
            "technique": technique,
            "points": stored.spectrum.num_points,
            "n_peaks": len(result.peaks),
            "quality": quality,
            "manual_confirmed": manual,
            "ai_modified": bool(result.metrics.get("ai_modified")),
            "summary": result.summary,
            "export_ready": True,
        })
    return {
        "items": rows,
        "summary": {
            "count": len(rows),
            "technique_counts": technique_counts,
            "quality_counts": quality_counts,
            "manual_confirmed": manual_count,
            "total_points": total_points,
        },
    }


@router.post("/analyze/{sid}")
def analyze_spectrum(sid: SpectrumId, payload: AnalyzeRequest | None = None):
    store = get_store()
    stored = store.get(sid)
    if stored is None:
        raise HTTPException(404, f"Spectrum {sid} not found")

    request = payload or AnalyzeRequest()
    expected_revision = _require_result_revision(request.expected_revision)
    if expected_revision != stored.result_revision:
        raise _conflict_response(RevisionConflict(stored.result_revision))
    if (
        stored.result is not None
        and stored.result.metrics.get("manual_confirmed")
        and not request.force_overwrite
    ):
        raise HTTPException(
            409,
            "This spectrum has a manually confirmed result; set force_overwrite=true to replace it",
        )
    options = request.options

    try:
        result = _analyze_stored(stored, options=options)
        updated = store.set_result(
            sid,
            result,
            expected_revision=expected_revision,
        )
        if updated is None:
            raise HTTPException(404, f"Spectrum {sid} not found")
    except RevisionConflict as exc:
        raise _conflict_response(exc) from exc
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Spectrum analysis failed for %s", sid)
        raise HTTPException(500, "Spectrum analysis failed") from e

    return _result_response(updated)


@router.get("/results/{sid}")
def get_result(sid: SpectrumId):
    store = get_store()
    stored = store.get(sid)
    if stored is None:
        raise HTTPException(404, f"Spectrum {sid} not found")
    if stored.result is None:
        raise HTTPException(404, f"No analysis result for {sid}")
    return _result_response(stored)


@router.get("/results/{sid}/versions")
def list_result_versions(sid: SpectrumId):
    store = get_store()
    if store.get(sid) is None:
        raise HTTPException(404, f"Spectrum {sid} not found")
    return {"versions": store.list_result_versions(sid)}


@router.post("/results/{sid}/versions/{version}/restore")
def restore_result_version(
    sid: SpectrumId,
    version: Annotated[int, Field(ge=1)],
    request: RestoreResultRequest | None = None,
):
    store = get_store()
    stored = store.get(sid)
    if stored is None:
        raise HTTPException(404, f"Spectrum {sid} not found")
    expected_revision = _require_result_revision(
        request.expected_revision if request else None
    )
    if expected_revision != stored.result_revision:
        raise _conflict_response(RevisionConflict(stored.result_revision))
    saved = store.get_result_version(sid, version)
    if saved is None:
        raise HTTPException(404, f"Result version {version} not found")
    saved.result.metrics["restored_from_version"] = version
    try:
        restored = store.set_result_and_version(
            sid,
            saved.result,
            note=f"restored from version {version}",
            expected_revision=expected_revision,
            restore_from_version=version,
        )
    except ResultVersionSourceConflict as exc:
        raise HTTPException(409, detail=exc.to_detail()) from exc
    except RevisionConflict as exc:
        raise _conflict_response(exc) from exc
    if restored is None:
        raise HTTPException(404, f"Spectrum {sid} not found")
    updated, _ = restored
    return _result_response(updated)


def _peak_from_dict(data: dict) -> Peak:
    return Peak(
        position=float(data.get("position", 0)),
        intensity=float(data.get("intensity", 0)),
        area=data.get("area"),
        width=data.get("width"),
        assignment=data.get("assignment", ""),
        multiplicity=data.get("multiplicity", ""),
        coupling_constant=data.get("coupling_constant"),
    )


def _normalize_integrals(integrals: list[dict]) -> list[dict]:
    positive = [float(item.get("raw_area", 0)) for item in integrals if float(item.get("raw_area", 0)) > 0]
    min_area = min(positive) if positive else 0
    for item in integrals:
        raw_area = float(item.get("raw_area", 0))
        item["raw_area"] = round(raw_area, 4)
        item["relative_area"] = round(raw_area / min_area, 2) if min_area > 0 and raw_area > 0 else 0
    return integrals


@router.post("/results/{sid}/integrate")
def integrate_ranges(sid: SpectrumId, payload: IntegrationRequest):
    store = get_store()
    stored = store.get(sid)
    if stored is None:
        raise HTTPException(404, f"Spectrum {sid} not found")

    ranges = [item.model_dump(exclude_none=True) for item in payload.ranges]

    spectrum = stored.spectrum
    channels = spectrum.parameters.get("channels") or []
    results = []

    for item in ranges:
        channel_name = item.get("channel")
        x = np.asarray(spectrum.x_data, dtype=float)
        y = np.asarray(spectrum.y_data, dtype=float)
        if channel_name and channels:
            channel = next((ch for ch in channels if ch.get("name") == channel_name), None)
            if channel is None:
                raise HTTPException(404, f"Channel {channel_name} not found")
            y = np.asarray(channel["y_data"], dtype=float)

        results.append(_integrate_range(x, y, item, spectrum.technique.value))

    return {"integrals": results}


def _update_hplc_channel_peaks(result: AnalysisResult, payload: dict) -> None:
    channel_peaks = result.metrics.get("channel_peaks")
    updates = payload.get("channel_peaks")
    if not isinstance(channel_peaks, dict) or not isinstance(updates, dict):
        return

    for channel_name, channel_data in updates.items():
        if channel_name not in channel_peaks or not isinstance(channel_data, dict):
            continue
        peaks = [dict(p) for p in channel_data.get("peaks", [])]
        total_area = sum(float(p.get("area", 0) or 0) for p in peaks)
        for peak in peaks:
            area = float(peak.get("area", 0) or 0)
            peak["area_percent"] = round(area / total_area * 100, 2) if total_area > 0 else 0.0
        channel_peaks[channel_name]["peaks"] = peaks
        channel_peaks[channel_name]["total_area"] = round(total_area, 2)
        channel_peaks[channel_name]["source"] = "manual_confirmed"

    all_peaks = []
    for channel_name, channel_data in channel_peaks.items():
        for peak in channel_data.get("peaks", []):
            all_peaks.append(Peak(
                position=float(peak.get("position", peak.get("apex_time", 0))),
                intensity=float(peak.get("intensity", peak.get("height", 0))),
                area=peak.get("area"),
                width=peak.get("width"),
                assignment=channel_name,
            ))
    if all_peaks:
        result.peaks = all_peaks


@router.put("/results/{sid}/manual")
def save_manual_result(sid: SpectrumId, request: ManualResultRequest):
    payload = request.model_dump(exclude_unset=True)
    payload.pop("expected_revision", None)
    store = get_store()
    stored = store.get(sid)
    if stored is None:
        raise HTTPException(404, f"Spectrum {sid} not found")
    expected_revision = _require_result_revision(request.expected_revision)
    if expected_revision != stored.result_revision:
        raise _conflict_response(RevisionConflict(stored.result_revision))
    if stored.result is None:
        stored.result = _analyze_stored(stored)

    result = stored.result
    if "peaks" in payload:
        result.peaks = [_peak_from_dict(p) for p in payload.get("peaks", [])]

    if hasattr(result, "integrals") and "integrals" in payload:
        integrals = _normalize_integrals([dict(item) for item in payload.get("integrals", [])])
        result.integrals = integrals
        total = sum(float(item.get("raw_area", 0)) for item in integrals)
        result.total_integral = total

    if hasattr(result, "multiplets") and "multiplets" in payload:
        result.multiplets = [dict(item) for item in payload.get("multiplets", [])]
        result.metrics["n_multiplets"] = len(result.multiplets)

    if "metrics" in payload:
        client_metrics = payload.get("metrics") or {}
        allowed_metrics = {
            key: value
            for key, value in client_metrics.items()
            if key in _CLIENT_METRIC_KEYS
        }
        dropped_keys = sorted(set(client_metrics) - set(allowed_metrics))
        if dropped_keys:
            logger.warning(
                "Dropping non-whitelisted metric keys from manual save for "
                "%s: %s",
                sid,
                ", ".join(dropped_keys),
            )
        result.metrics.update(allowed_metrics)

    _update_hplc_channel_peaks(result, payload)

    provided_summary = payload.get("summary")
    if provided_summary:
        result.summary = str(provided_summary)

    result.metrics["n_peaks"] = len(result.peaks)
    result.metrics["manual_confirmed"] = True
    result.metrics["manual_version"] = int(result.metrics.get("manual_version", 0)) + 1
    result.metrics["quality"] = assess_quality(stored.spectrum, result)
    if not provided_summary:
        # Append the confirmation marker only once; repeated confirmations
        # used to stack duplicate markers onto the summary.
        marker = "Manual review confirmed."
        if marker not in (result.summary or ""):
            result.summary = f"{result.summary} {marker}".strip()

    try:
        saved = store.set_result_and_version(
            sid,
            result,
            note=str(payload.get("note") or "manual confirmation"),
            version_metric="manual_version",
            expected_revision=expected_revision,
        )
    except RevisionConflict as exc:
        raise _conflict_response(exc) from exc
    if saved is None:
        raise HTTPException(404, f"Spectrum {sid} not found")
    updated, _ = saved
    return _result_response(updated)


@router.post("/compare")
def compare_spectra(payload: CompareSpectraRequest):
    id1 = payload.id1
    id2 = payload.id2
    store = get_store()
    s1 = store.get(id1)
    s2 = store.get(id2)
    if s1 is None or s2 is None:
        raise HTTPException(404, "One or both spectra not found")

    t1 = s1.spectrum.technique.value
    t2 = s2.spectrum.technique.value

    comparison: dict = {
        "id1": id1,
        "id2": id2,
        "technique1": t1,
        "technique2": t2,
        "shared": {},
    }

    if t1 == t2 == "Fluorescence":
        sub1 = s1.spectrum.parameters.get("sub_type", "")
        sub2 = s2.spectrum.parameters.get("sub_type", "")
        if {sub1, sub2} == {"excitation", "emission"}:
            from app.analysis.fluorescence_analysis import FluorescenceAnalyzer

            fa = FluorescenceAnalyzer()
            if s1.result is None:
                s1.result = fa.analyze(s1.spectrum)
            if s2.result is None:
                s2.result = fa.analyze(s2.spectrum)

            ex_res = s1.result if sub1 == "excitation" else s2.result
            em_res = s2.result if sub2 == "emission" else s1.result
            comparison["stokes"] = fa.compute_stokes_shift(ex_res, em_res)

    comparison["shared"]["points1"] = s1.spectrum.num_points
    comparison["shared"]["points2"] = s2.spectrum.num_points
    comparison["shared"]["x_range1"] = list(s1.spectrum.x_range)
    comparison["shared"]["x_range2"] = list(s2.spectrum.x_range)

    return comparison


@router.post("/compare/hplc")
def compare_hplc_batch(payload: HPLCComparisonRequest):
    ids = payload.ids
    channel = payload.channel
    rt_tolerance = payload.rt_tolerance
    store = get_store()
    samples = []
    for sid in ids:
        stored = store.get(sid)
        if stored is None:
            raise HTTPException(404, f"Spectrum {sid} not found")
        if stored.spectrum.technique.value != "HPLC":
            raise HTTPException(400, f"Spectrum {sid} is not HPLC")
        result = _ensure_stored_result(stored)
        channel_peaks = result.metrics.get("channel_peaks") or {}
        selected_channel = channel or next(iter(channel_peaks.keys()), "")
        if selected_channel not in channel_peaks:
            raise HTTPException(404, f"Channel {selected_channel} not found in {sid}")
        samples.append({
            "id": sid,
            "name": stored.spectrum.metadata.name or stored.spectrum.source_file,
            "channel": selected_channel,
            "peaks": channel_peaks[selected_channel].get("peaks", []),
            "total_area": channel_peaks[selected_channel].get("total_area", 0),
        })

    reference = samples[0]
    reference_peaks = sorted(reference["peaks"], key=lambda p: float(p.get("position", 0)))
    # One-to-one consumed matching: each sample peak can be matched by at most
    # one reference peak, so drift statistics are not inflated by reuse.
    matches_by_sample: dict[str, list[dict[str, Any] | None]] = {}
    for sample in samples:
        matches_by_sample[sample["id"]] = _match_sample_peaks(
            reference_peaks,
            sample["peaks"],
            rt_tolerance,
        )

    rows = []
    for ref_index, ref_peak in enumerate(reference_peaks, start=1):
        ref_rt = float(ref_peak.get("position", 0))
        row = {
            "peak_index": ref_index,
            "reference_rt": round(ref_rt, 4),
            "reference_area": ref_peak.get("area", 0),
            "reference_type": ref_peak.get("type", ""),
            "name": ref_peak.get("name", ""),
            "samples": [],
        }
        for sample in samples:
            best = matches_by_sample[sample["id"]][ref_index - 1]
            if best is None or abs(float(best.get("position", 0)) - ref_rt) > rt_tolerance:
                row["samples"].append({
                    "id": sample["id"],
                    "name": sample["name"],
                    "matched": False,
                    "rt_shift": None,
                    "area": 0,
                    "area_percent": 0,
                })
                continue
            rt = float(best.get("position", 0))
            row["samples"].append({
                "id": sample["id"],
                "name": sample["name"],
                "matched": True,
                "rt": round(rt, 4),
                "rt_shift": round(rt - ref_rt, 4),
                "area": best.get("area", 0),
                "area_percent": best.get("area_percent", 0),
                "height": best.get("intensity", best.get("height", 0)),
            })
        rows.append(row)

    drift = []
    for sample in samples:
        shifts = []
        for row in rows:
            match = next((m for m in row["samples"] if m["id"] == sample["id"] and m["matched"]), None)
            if match and match["rt_shift"] is not None:
                shifts.append(float(match["rt_shift"]))
        drift.append({
            "id": sample["id"],
            "name": sample["name"],
            "channel": sample["channel"],
            "mean_rt_shift": round(float(np.mean(shifts)), 4) if shifts else None,
            "max_abs_rt_shift": round(float(np.max(np.abs(shifts))), 4) if shifts else None,
            "matched_peaks": len(shifts),
            "total_area": sample["total_area"],
        })

    return {
        "reference_id": reference["id"],
        "channel": reference["channel"],
        "rt_tolerance": rt_tolerance,
        "rows": rows,
        "drift": drift,
    }


@router.post("/compare/hplc.csv")
def export_hplc_batch_comparison_csv(payload: HPLCComparisonRequest):
    data = compare_hplc_batch(payload)
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(["HPLC RT Drift"])
    writer.writerow(["sample_id", "sample_name", "channel", "matched_peaks", "mean_rt_shift", "max_abs_rt_shift", "total_area"])
    for row in data["drift"]:
        writer.writerow([
            _csv_safe(row["id"]),
            _csv_safe(row["name"]),
            _csv_safe(row["channel"]),
            row["matched_peaks"],
            row["mean_rt_shift"],
            row["max_abs_rt_shift"],
            row["total_area"],
        ])

    writer.writerow([])
    sample_headers = []
    for drift_row in data["drift"]:
        label = drift_row["name"] or drift_row["id"]
        sample_headers.extend([f"{label} area", f"{label} area%", f"{label} rt_shift"])
    writer.writerow([
        "peak_index",
        "reference_rt",
        "reference_area",
        "type",
        "name",
        *[_csv_safe(value) for value in sample_headers],
    ])
    for row in data["rows"]:
        values = []
        for sample in row["samples"]:
            values.extend([
                sample.get("area", 0),
                sample.get("area_percent", 0),
                sample.get("rt_shift"),
            ])
        writer.writerow([
            row["peak_index"],
            row["reference_rt"],
            row["reference_area"],
            _csv_safe(row["reference_type"]),
            _csv_safe(row["name"]),
            *values,
        ])

    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=hplc-comparison.csv"},
    )
