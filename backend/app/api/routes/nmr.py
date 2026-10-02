from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from scipy import signal

from app.analysis.nmr_processing import (
    PROCESSING_SOURCE_KEY,
    NMRProcessingSourceError,
    apply_phase_correction,
    asymmetric_least_squares_baseline,
    automatic_phase_correction,
    build_processing_source,
    locate_reference_peak,
    processing_source_summary,
    read_processing_source,
    solvent_reference_ppm,
    spectrum_quality_metrics,
)
from app.api.deps import get_store
from app.api.store import RevisionConflict, SpectrumResultRevisionConflict, StoredSpectrum
from app.core.models import Technique

router = APIRouter(prefix="/api/nmr", tags=["nmr"])
SpectrumId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class NMRProcessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    replay_from_original: bool = True
    preview_only: bool = False
    phase_zero_deg: float | None = Field(default=None, allow_inf_nan=False, ge=-36000, le=36000)
    phase_first_deg: float | None = Field(default=None, allow_inf_nan=False, ge=-36000, le=36000)
    phase_pivot_ppm: float | None = Field(default=None, allow_inf_nan=False)
    auto_phase: bool = False
    auto_phase_first_order: bool = True
    auto_phase_max_first_deg: float = Field(
        default=720.0,
        allow_inf_nan=False,
        ge=0,
        le=5000,
    )
    crop_min_ppm: float | None = Field(default=None, allow_inf_nan=False)
    crop_max_ppm: float | None = Field(default=None, allow_inf_nan=False)
    smoothing_window: int = Field(default=0, ge=0, le=1001)
    smoothing_polynomial_order: int = Field(default=2, ge=0, le=5)
    baseline_correct: bool = False
    baseline_method: Literal["als", "asymmetric_least_squares", "percentile"] = "als"
    # Upper bound matches the processing path's _finite_float(maximum=1e14)
    # so request validation and runtime validation reject the same values.
    baseline_smoothness: float = Field(default=1e7, allow_inf_nan=False, ge=1, le=1e14)
    baseline_asymmetry: float = Field(default=0.001, allow_inf_nan=False, gt=0, lt=1)
    baseline_iterations: int = Field(default=8, ge=1, le=50)
    baseline_percentile: float = Field(default=10, allow_inf_nan=False, ge=0, le=100)
    normalize: bool = False
    normalize_ppm: float | None = Field(default=None, allow_inf_nan=False)
    normalize_window_points: int = Field(default=3, ge=1, le=1001)
    reference_current_ppm: float | None = Field(default=None, allow_inf_nan=False)
    reference_target_ppm: float | None = Field(default=None, allow_inf_nan=False)
    auto_reference: bool = False
    reference_solvent: str | None = Field(default=None, min_length=1, max_length=64)
    reference_window_ppm: float = Field(
        default=0.15,
        allow_inf_nan=False,
        gt=0,
        le=2,
    )
    reference_min_snr: float = Field(
        default=5.0,
        allow_inf_nan=False,
        gt=0,
        le=1e6,
    )
    invert: bool = False
    expected_revision: int | None = Field(default=None, ge=1)
    expected_result_revision: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def paired_ranges_and_references(self):
        if (self.crop_min_ppm is None) != (self.crop_max_ppm is None):
            raise ValueError("crop_min_ppm and crop_max_ppm must be provided together")
        if (
            self.reference_current_ppm is None
        ) != (self.reference_target_ppm is None):
            raise ValueError(
                "reference_current_ppm and reference_target_ppm must be provided together"
            )
        if self.auto_phase and (
            self.phase_zero_deg is not None or self.phase_first_deg is not None
        ):
            raise ValueError(
                "auto_phase cannot be combined with manual phase parameters"
            )
        if self.auto_reference and self.reference_current_ppm is not None:
            raise ValueError(
                "auto_reference cannot be combined with manual reference parameters"
            )
        if self.smoothing_window and self.smoothing_window % 2 == 0:
            raise ValueError("smoothing_window must be odd")
        if (
            self.smoothing_window
            and self.smoothing_polynomial_order >= self.smoothing_window
        ):
            raise ValueError(
                "smoothing_polynomial_order must be smaller than smoothing_window"
            )
        return self


class NMRResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    expected_revision: int | None = Field(default=None, ge=1)
    expected_result_revision: int | None = Field(default=None, ge=0, strict=True)


def _finite_float(
    payload: dict[str, Any],
    name: str,
    *,
    default: float | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    raw = payload.get(name)
    if raw in (None, ""):
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise HTTPException(422, detail=f"{name} must be a number")
    if not np.isfinite(value):
        raise HTTPException(422, detail=f"{name} must be finite")
    if minimum is not None and value < minimum:
        raise HTTPException(422, detail=f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        raise HTTPException(422, detail=f"{name} must be at most {maximum}")
    return value


def _processing_source(spectrum) -> tuple[dict[str, Any], list[str]]:
    source = spectrum.parameters.get(PROCESSING_SOURCE_KEY)
    warnings: list[str] = []
    if source is None:
        source = build_processing_source(
            spectrum.x_data,
            spectrum.y_data,
            source_kind="legacy_current_snapshot",
            source_quality="legacy_current_snapshot",
        )
        spectrum.parameters[PROCESSING_SOURCE_KEY] = source
        warnings.append(
            "This legacy record had no immutable acquisition snapshot; the current spectrum "
            "was preserved as the reset point."
        )
    try:
        read_processing_source(source, verify_checksum=True)
    except NMRProcessingSourceError as exc:
        raise HTTPException(
            409,
            detail={
                "code": "invalid_processing_source",
                "message": str(exc),
                "status": "rejected",
            },
        )
    return source, warnings


def _expected_revision(
    payload: dict[str, Any],
    *,
    required: bool = False,
) -> int | None:
    raw = payload.get("expected_revision")
    if raw in (None, ""):
        if required:
            raise HTTPException(
                428,
                detail={
                    "code": "expected_revision_required",
                    "message": (
                        "expected_revision is required when committing NMR "
                        "processing changes"
                    ),
                },
            )
        return None
    if isinstance(raw, bool):
        raise HTTPException(422, detail="expected_revision must be an integer")
    try:
        revision = int(raw)
    except (TypeError, ValueError):
        raise HTTPException(422, detail="expected_revision must be an integer")
    if revision < 1 or str(raw).strip() not in {str(revision), f"{revision}.0"}:
        raise HTTPException(422, detail="expected_revision must be a positive integer")
    return revision


def _revision_conflict(exc: RevisionConflict) -> HTTPException:
    if isinstance(exc, SpectrumResultRevisionConflict):
        return HTTPException(
            409,
            detail={
                "code": "revision_conflict",
                "message": "The analysis result changed while processing; reload before replacing it.",
                "current_revision": exc.current_revision,
                "current_result_revision": exc.current_result_revision,
            },
        )
    return HTTPException(
        409,
        detail={
            "code": "revision_conflict",
            "message": "The spectrum changed after it was loaded; reload before processing.",
            "current_revision": exc.current_revision,
        },
    )


def _expected_result_revision(payload: dict[str, Any], stored: StoredSpectrum) -> int:
    """Bind processing to the result the client actually reviewed.

    A legacy request without a result revision is safe only when there is no
    current result to discard. The transaction still guards that empty state
    against a result saved while processing is running.
    """
    expected = payload.get("expected_result_revision")
    if expected is None:
        if stored.result is not None:
            raise HTTPException(
                428,
                detail={
                    "code": "expected_result_revision_required",
                    "message": "Reload the analysis result before previewing or applying NMR processing; expected_result_revision is required.",
                    "current_result_revision": stored.result_revision,
                },
            )
        return stored.result_revision
    if expected != stored.result_revision:
        raise _revision_conflict(SpectrumResultRevisionConflict(
            stored.spectrum_revision, stored.result_revision,
        ))
    return int(expected)


def _next_revision(parameters: dict[str, Any]) -> int:
    revisions = parameters.get("processing_revisions") or []
    numbers = [
        int(item.get("revision") or 0)
        for item in revisions
        if isinstance(item, dict)
    ]
    return max(numbers, default=int(parameters.get("processing_revision") or 0)) + 1


def _append_revision(parameters: dict[str, Any], revision: dict[str, Any]) -> None:
    revisions = [
        dict(item)
        for item in (parameters.get("processing_revisions") or [])
        if isinstance(item, dict)
    ]
    revisions.append(revision)
    # Recipes are small but should not grow without bound in a frequently
    # edited spectrum. The immutable source remains available independently.
    parameters["processing_revisions"] = revisions[-100:]
    parameters["processing_revisions_truncated"] = len(revisions) > 100
    parameters["processing_revision"] = revision["revision"]


def _public_response(
    spectrum,
    sid: str,
    *,
    applied: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    warnings: list[str],
    state: str,
    committed: bool,
    spectrum_revision: int,
    result_revision: int,
    quality_before: dict[str, Any],
    quality_after: dict[str, Any],
) -> dict[str, Any]:
    data = spectrum.to_dict(include_internal=False)
    data["id"] = sid
    data["processing_applied"] = applied
    data["processing_skipped"] = skipped
    data["warnings"] = warnings
    data["spectrum_revision"] = int(spectrum_revision)
    data["result_revision"] = int(result_revision)
    data["quality_before"] = quality_before
    data["quality_after"] = quality_after
    phase_operation = next(
        (
            item
            for item in applied + skipped
            if item.get("type") == "phase_correct"
        ),
        None,
    )
    data["processing_status"] = {
        "state": state,
        "committed": committed,
        "phase": (
            str(phase_operation.get("status"))
            if phase_operation is not None
            else "not_requested"
        ),
        "revision": spectrum.parameters.get("processing_revision"),
        "available_views": ["original", "current", "preview"],
        "source": processing_source_summary(
            spectrum.parameters.get(PROCESSING_SOURCE_KEY)
        ),
    }
    return data


@router.get("/{sid}/view")
def get_nmr_spectrum_view(
    sid: SpectrumId,
    state: Literal["original", "current"] = Query(default="current"),
):
    """Return a real-only immutable-original or committed-current NMR view."""

    stored = get_store().get(sid)
    if stored is None:
        raise HTTPException(404, f"Spectrum {sid} not found")
    if stored.spectrum.technique != Technique.NMR:
        raise HTTPException(400, "NMR views can only be requested for NMR spectra")

    spectrum = stored.spectrum
    source, warnings = _processing_source(spectrum)
    if state == "original":
        x, y, _, _ = read_processing_source(source)
    else:
        x = np.asarray(spectrum.x_data, dtype=np.float64)
        y = np.asarray(spectrum.y_data, dtype=np.float64)

    data = spectrum.to_dict(include_internal=False)
    data["id"] = sid
    data["x_data"] = x.tolist()
    data["y_data"] = y.tolist()
    data["view_state"] = state
    data["quality_metrics"] = spectrum_quality_metrics(x, y)
    data["warnings"] = warnings
    data["spectrum_revision"] = stored.spectrum_revision
    data["result_revision"] = stored.result_revision
    data["processing_revision"] = spectrum.parameters.get("processing_revision")
    data["processing_status"] = {
        "state": state,
        "committed": state == "current",
        "available_views": ["original", "current", "preview"],
        "source": processing_source_summary(source),
    }
    return data


@router.post("/{sid}/process")
def process_nmr_spectrum(sid: SpectrumId, request: NMRProcessRequest):
    payload = request.model_dump()
    store = get_store()
    stored = store.get(sid)
    if stored is None:
        raise HTTPException(404, f"Spectrum {sid} not found")
    if stored.spectrum.technique != Technique.NMR:
        raise HTTPException(400, "NMR processing can only be applied to NMR spectra")

    spectrum = stored.spectrum
    source, warnings = _processing_source(spectrum)
    original_x, original_y, quadrature_real, quadrature_imaginary = (
        read_processing_source(source)
    )
    replay_from_original = bool(payload.get("replay_from_original", True))
    preview_only = bool(payload.get("preview_only", False))
    expected_revision = _expected_revision(payload, required=not preview_only)
    if (
        expected_revision is not None
        and expected_revision != stored.spectrum_revision
    ):
        raise _revision_conflict(RevisionConflict(stored.spectrum_revision))
    expected_result_revision = _expected_result_revision(payload, stored)

    manual_phase_requested = (
        payload.get("phase_zero_deg") not in (None, "")
        or payload.get("phase_first_deg") not in (None, "")
    )
    auto_phase_requested = bool(payload.get("auto_phase", False))
    if manual_phase_requested and (
        quadrature_real is None or quadrature_imaginary is None
    ):
        raise HTTPException(
            422,
            detail={
                "code": "phase_requires_quadrature",
                "message": (
                    "Manual phase correction requires a persisted real/imaginary "
                    "quadrature pair; this spectrum contains only a real channel."
                ),
                "status": "skipped_no_imaginary_channel",
                "processing_applied": [],
            },
        )
    if (
        (manual_phase_requested or auto_phase_requested)
        and quadrature_real is not None
        and quadrature_imaginary is not None
        and not replay_from_original
    ):
        raise HTTPException(
            422,
            detail={
                "code": "phase_requires_original_replay",
                "message": (
                    "Phase correction must be replayed from the immutable quadrature "
                    "source; set replay_from_original=true."
                ),
                "status": "rejected",
            },
        )

    if replay_from_original:
        x = original_x.copy()
        y = original_y.copy()
    else:
        x = np.asarray(spectrum.x_data, dtype=np.float64).copy()
        y = np.asarray(spectrum.y_data, dtype=np.float64).copy()
        warnings.append(
            "Cumulative processing was explicitly requested; reset remains available."
        )

    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    quality_before = spectrum_quality_metrics(x, y)

    # Phase on the full quadrature spectrum before crop/smoothing/baseline.
    phase_applied = False
    if auto_phase_requested and (
        quadrature_real is None or quadrature_imaginary is None
    ):
        skipped.append(
            {
                "type": "phase_correct",
                "mode": "automatic",
                "status": "skipped_no_imaginary_channel",
                "source": "persisted_quadrature",
            }
        )
        warnings.append(
            "Automatic phase correction was skipped because this spectrum has "
            "no persisted imaginary channel."
        )
    elif auto_phase_requested:
        pivot = _finite_float(
            payload,
            "phase_pivot_ppm",
            default=float(np.median(x)),
        )
        max_first = _finite_float(
            payload,
            "auto_phase_max_first_deg",
            default=720.0,
            minimum=0.0,
            maximum=5000.0,
        )
        y, _, phase_result = automatic_phase_correction(
            x,
            quadrature_real,
            quadrature_imaginary,
            pivot_ppm=float(pivot),
            optimize_first_order=bool(
                payload.get("auto_phase_first_order", True)
            ),
            max_first_deg=float(max_first),
        )
        applied.append(
            {
                "type": "phase_correct",
                "mode": "automatic",
                "status": "applied",
                "zero_deg": float(phase_result["zero_deg"]),
                "first_deg": float(phase_result["first_deg"]),
                "pivot_ppm": float(phase_result["pivot_ppm"]),
                "method": phase_result["method"],
                "objective_before": float(phase_result["objective_before"]),
                "objective_after": float(phase_result["objective_after"]),
                "converged": bool(phase_result["converged"]),
                "source": "persisted_quadrature",
            }
        )
        phase_applied = True
    elif manual_phase_requested:
        zero = _finite_float(
            payload, "phase_zero_deg", default=0.0, minimum=-36000, maximum=36000
        )
        first = _finite_float(
            payload, "phase_first_deg", default=0.0, minimum=-36000, maximum=36000
        )
        pivot = _finite_float(payload, "phase_pivot_ppm", default=float(np.median(x)))
        y, _ = apply_phase_correction(
            x,
            quadrature_real,
            quadrature_imaginary,
            zero_deg=float(zero),
            first_deg=float(first),
            pivot_ppm=float(pivot),
        )
        applied.append(
            {
                "type": "phase_correct",
                "mode": "manual",
                "status": "applied",
                "zero_deg": float(zero),
                "first_deg": float(first),
                "pivot_ppm": float(pivot),
                "source": "persisted_quadrature",
            }
        )
        phase_applied = True

    if phase_applied:
        default_baseline = dict(source.get("default_baseline") or {})
        if default_baseline.get("method") == "asymmetric_least_squares":
            baseline, y = asymmetric_least_squares_baseline(
                y,
                smoothness=float(default_baseline.get("smoothness", 1e7)),
                asymmetry=float(default_baseline.get("asymmetry", 0.001)),
                iterations=int(default_baseline.get("iterations", 8)),
                max_fit_points=int(default_baseline.get("max_fit_points", 8192)),
            )
            applied.append(
                {
                    "type": "restore_default_baseline",
                    "status": "applied",
                    "method": "asymmetric_least_squares",
                    "baseline_min": float(np.min(baseline)),
                    "baseline_max": float(np.max(baseline)),
                }
            )

    crop_min = _finite_float(payload, "crop_min_ppm")
    crop_max = _finite_float(payload, "crop_max_ppm")
    if (crop_min is None) != (crop_max is None):
        raise HTTPException(
            422, detail="crop_min_ppm and crop_max_ppm must be provided together"
        )
    if crop_min is not None and crop_max is not None:
        lo, hi = sorted((crop_min, crop_max))
        mask = (x >= lo) & (x <= hi)
        if int(mask.sum()) < 10:
            raise HTTPException(422, detail="Crop range leaves too few NMR data points")
        x = x[mask]
        y = y[mask]
        applied.append(
            {
                "type": "crop",
                "status": "applied",
                "min_ppm": lo,
                "max_ppm": hi,
                "points": int(mask.sum()),
            }
        )

    try:
        smoothing_window = int(payload.get("smoothing_window") or 0)
    except (TypeError, ValueError):
        raise HTTPException(422, detail="smoothing_window must be an integer")
    if smoothing_window < 0 or smoothing_window > 1001:
        raise HTTPException(
            422, detail="smoothing_window must be between 0 and 1001"
        )
    if smoothing_window > 1:
        requested_window = smoothing_window
        if smoothing_window % 2 == 0:
            smoothing_window += 1
            warnings.append(
                f"Smoothing window {requested_window} was made odd ({smoothing_window})."
            )
        max_window = len(y) if len(y) % 2 == 1 else len(y) - 1
        smoothing_window = min(smoothing_window, max_window)
        if smoothing_window < 3:
            raise HTTPException(422, detail="Spectrum is too short for smoothing")
        polynomial_order = min(
            int(payload.get("smoothing_polynomial_order") or 2),
            smoothing_window - 1,
        )
        if polynomial_order < 0 or polynomial_order > 5:
            raise HTTPException(
                422, detail="smoothing_polynomial_order must be between 0 and 5"
            )
        y = signal.savgol_filter(
            y,
            window_length=smoothing_window,
            polyorder=polynomial_order,
            mode="interp",
        )
        applied.append(
            {
                "type": "smooth",
                "status": "applied",
                "method": "savitzky_golay",
                "window": smoothing_window,
                "polynomial_order": polynomial_order,
            }
        )

    if payload.get("baseline_correct"):
        method = str(payload.get("baseline_method") or "asymmetric_least_squares")
        if method in {"asymmetric_least_squares", "als"}:
            smoothness = _finite_float(
                payload,
                "baseline_smoothness",
                default=1e7,
                minimum=1.0,
                maximum=1e14,
            )
            asymmetry = _finite_float(
                payload,
                "baseline_asymmetry",
                default=0.001,
                minimum=1e-8,
                maximum=1 - 1e-8,
            )
            try:
                iterations = int(payload.get("baseline_iterations") or 8)
            except (TypeError, ValueError):
                raise HTTPException(422, detail="baseline_iterations must be an integer")
            baseline, y = asymmetric_least_squares_baseline(
                y,
                smoothness=float(smoothness),
                asymmetry=float(asymmetry),
                iterations=iterations,
            )
            applied.append(
                {
                    "type": "baseline_correct",
                    "status": "applied",
                    "method": "asymmetric_least_squares",
                    "smoothness": float(smoothness),
                    "asymmetry": float(asymmetry),
                    "iterations": iterations,
                    "baseline_min": float(np.min(baseline)),
                    "baseline_max": float(np.max(baseline)),
                }
            )
        elif method == "percentile":
            percentile = _finite_float(
                payload,
                "baseline_percentile",
                default=10.0,
                minimum=0.0,
                maximum=100.0,
            )
            offset = float(np.percentile(y, percentile))
            y = y - offset
            applied.append(
                {
                    "type": "baseline_correct",
                    "status": "applied",
                    "method": "percentile",
                    "percentile": percentile,
                    "offset": offset,
                }
            )
        else:
            raise HTTPException(
                422,
                detail=(
                    "baseline_method must be 'asymmetric_least_squares' or 'percentile'"
                ),
            )

    if payload.get("invert"):
        y = -y
        applied.append({"type": "invert", "status": "applied"})

    if payload.get("normalize"):
        scale = 0.0
        normalize_ppm = _finite_float(payload, "normalize_ppm")
        if normalize_ppm is not None and len(x):
            index = int(np.argmin(np.abs(x - normalize_ppm)))
            try:
                half_window = int(payload.get("normalize_window_points") or 3)
            except (TypeError, ValueError):
                raise HTTPException(
                    422, detail="normalize_window_points must be an integer"
                )
            half_window = max(1, min(half_window, 1000))
            lo = max(0, index - half_window)
            hi = min(len(y), index + half_window + 1)
            scale = float(np.max(np.abs(y[lo:hi])))
        if scale <= np.finfo(float).eps:
            scale = float(np.max(np.abs(y))) if len(y) else 0.0
        if scale > np.finfo(float).eps:
            y = y / scale
            applied.append(
                {
                    "type": "normalize",
                    "status": "applied",
                    "scale": scale,
                    "target_ppm": normalize_ppm,
                }
            )
        else:
            skipped.append(
                {
                    "type": "normalize",
                    "status": "skipped_zero_signal",
                    "target_ppm": normalize_ppm,
                }
            )
            warnings.append("Normalization was skipped because the spectrum is zero.")

    current_ref = _finite_float(payload, "reference_current_ppm")
    target_ref = _finite_float(payload, "reference_target_ppm")
    if (current_ref is None) != (target_ref is None):
        raise HTTPException(
            422,
            detail=(
                "reference_current_ppm and reference_target_ppm must be provided together"
            ),
        )
    if current_ref is not None and target_ref is not None:
        x = x - (current_ref - target_ref)
        applied.append(
            {
                "type": "reference_shift",
                "status": "applied",
                "current_ppm": current_ref,
                "target_ppm": target_ref,
                "source": "manual",
            }
        )
    elif payload.get("auto_reference"):
        solvent = str(
            payload.get("reference_solvent")
            or spectrum.parameters.get("solvent")
            or spectrum.metadata.solvent
            or ""
        )
        nucleus = str(spectrum.parameters.get("nucleus") or "1H")
        expected_reference = solvent_reference_ppm(solvent, nucleus)
        if expected_reference is None:
            skipped.append(
                {
                    "type": "reference_shift",
                    "status": "skipped_unknown_solvent_reference",
                    "source": "automatic_solvent",
                    "solvent": solvent,
                    "nucleus": nucleus,
                }
            )
            warnings.append(
                "Automatic referencing was skipped because no solvent reference "
                f"is registered for {solvent or 'the unspecified solvent'} ({nucleus})."
            )
        else:
            located = locate_reference_peak(
                x,
                y,
                expected_ppm=expected_reference,
                window_ppm=float(payload.get("reference_window_ppm") or 0.15),
                min_snr=float(payload.get("reference_min_snr") or 5.0),
            )
            if located is None:
                skipped.append(
                    {
                        "type": "reference_shift",
                        "status": "skipped_reference_peak_not_found",
                        "source": "automatic_solvent",
                        "solvent": solvent,
                        "nucleus": nucleus,
                        "expected_ppm": expected_reference,
                        "window_ppm": float(
                            payload.get("reference_window_ppm") or 0.15
                        ),
                        "minimum_snr": float(
                            payload.get("reference_min_snr") or 5.0
                        ),
                    }
                )
                warnings.append(
                    "Automatic referencing was skipped because no sufficiently "
                    "strong solvent line was found in the requested window."
                )
            else:
                observed_reference = float(located["observed_ppm"])
                x = x - (observed_reference - expected_reference)
                applied.append(
                    {
                        "type": "reference_shift",
                        "status": "applied",
                        "source": "automatic_solvent",
                        "solvent": solvent,
                        "nucleus": nucleus,
                        "current_ppm": observed_reference,
                        "target_ppm": expected_reference,
                        "snr": float(located["snr"]),
                        "window_ppm": float(located["window_ppm"]),
                    }
                )

    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise HTTPException(
            422,
            detail={
                "code": "non_finite_processing_result",
                "message": "Processing produced NaN or infinite values; no data were saved.",
                "status": "rejected",
            },
        )

    quality_after = spectrum_quality_metrics(x, y)
    if not applied:
        return _public_response(
            spectrum,
            sid,
            applied=[],
            skipped=skipped,
            warnings=warnings,
            state="unchanged",
            committed=False,
            spectrum_revision=stored.spectrum_revision,
            result_revision=stored.result_revision,
            quality_before=quality_before,
            quality_after=quality_after,
        )

    revision_number = _next_revision(spectrum.parameters)
    revision = {
        "revision": revision_number,
        "created_at": datetime.now(UTC).isoformat(),
        "source": "immutable_original" if replay_from_original else "current_revision",
        "operations": applied + skipped,
        "preview_only": preview_only,
    }
    spectrum.x_data = np.asarray(x, dtype=np.float64)
    spectrum.y_data = np.asarray(y, dtype=np.float64)
    spectrum.parameters["processing_history"] = applied + skipped
    spectrum.parameters["processed"] = bool(applied)
    spectrum.parameters["processing_source_summary"] = processing_source_summary(source)
    spectrum.parameters["processing_quality"] = quality_after
    _append_revision(spectrum.parameters, revision)

    if not preview_only:
        try:
            updated = store.set_spectrum(
                sid,
                spectrum,
                clear_result=True,
                expected_revision=expected_revision,
                expected_result_revision=expected_result_revision,
            )
        except RevisionConflict as exc:
            raise _revision_conflict(exc)
        if updated is None:
            raise HTTPException(404, f"Spectrum {sid} not found")
        spectrum_revision = updated.spectrum_revision
        result_revision = updated.result_revision
    else:
        spectrum_revision = stored.spectrum_revision
        result_revision = stored.result_revision

    return _public_response(
        spectrum,
        sid,
        applied=applied,
        skipped=skipped,
        warnings=warnings,
        state="preview" if preview_only else "applied",
        committed=not preview_only,
        spectrum_revision=spectrum_revision,
        result_revision=result_revision,
        quality_before=quality_before,
        quality_after=quality_after,
    )


@router.post("/{sid}/reset")
def reset_nmr_processing(
    sid: SpectrumId,
    request: NMRResetRequest | None = None,
):
    payload = (request or NMRResetRequest()).model_dump()
    store = get_store()
    stored = store.get(sid)
    if stored is None:
        raise HTTPException(404, f"Spectrum {sid} not found")
    if stored.spectrum.technique != Technique.NMR:
        raise HTTPException(400, "NMR reset can only be applied to NMR spectra")

    expected_revision = _expected_revision(payload, required=True)
    if expected_revision != stored.spectrum_revision:
        raise _revision_conflict(RevisionConflict(stored.spectrum_revision))
    expected_result_revision = _expected_result_revision(payload, stored)

    spectrum = stored.spectrum
    source, warnings = _processing_source(spectrum)
    original_x, original_y, _, _ = read_processing_source(source)
    quality_before = spectrum_quality_metrics(
        spectrum.x_data,
        spectrum.y_data,
    )
    spectrum.x_data = original_x.copy()
    spectrum.y_data = original_y.copy()
    quality_after = spectrum_quality_metrics(original_x, original_y)
    revision_number = _next_revision(spectrum.parameters)
    operation = {
        "type": "reset",
        "status": "applied",
        "target": "immutable_original",
    }
    revision = {
        "revision": revision_number,
        "created_at": datetime.now(UTC).isoformat(),
        "source": "immutable_original",
        "operations": [operation],
        "preview_only": False,
    }
    spectrum.parameters["processing_history"] = []
    spectrum.parameters["processed"] = False
    spectrum.parameters["processing_source_summary"] = processing_source_summary(source)
    spectrum.parameters["processing_quality"] = quality_after
    _append_revision(spectrum.parameters, revision)

    try:
        updated = store.set_spectrum(
            sid,
            spectrum,
            clear_result=True,
            expected_revision=expected_revision,
            expected_result_revision=expected_result_revision,
        )
    except RevisionConflict as exc:
        raise _revision_conflict(exc)
    if updated is None:
        raise HTTPException(404, f"Spectrum {sid} not found")
    return _public_response(
        spectrum,
        sid,
        applied=[operation],
        skipped=[],
        warnings=warnings,
        state="reset",
        committed=True,
        spectrum_revision=updated.spectrum_revision,
        result_revision=updated.result_revision,
        quality_before=quality_before,
        quality_after=quality_after,
    )
