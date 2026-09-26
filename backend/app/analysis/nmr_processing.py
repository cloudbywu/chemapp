from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.sparse.linalg import spsolve

PROCESSING_SOURCE_KEY = "processing_source"
PROCESSING_SOURCE_SCHEMA_VERSION = 1
PROCESSING_PIPELINE_VERSION = "nmr-processing-v2"
QUALITY_METRICS_SCHEMA_VERSION = 1

_SOLVENT_REFERENCE_PPM: dict[str, dict[str, float]] = {
    "1H": {
        "ACETONE": 2.05,
        "ACETONE-D6": 2.05,
        "BENZENE-D6": 7.16,
        "C6D6": 7.16,
        "CD2CL2": 5.32,
        "CD3OD": 3.31,
        "CDCL3": 7.26,
        "CHLOROFORM-D": 7.26,
        "D2O": 4.79,
        "DEUTERIUMOXIDE": 4.79,
        "DICHLOROMETHANE-D2": 5.32,
        "DIMETHYLSULFOXIDE-D6": 2.50,
        "DMSO": 2.50,
        "DMSO-D6": 2.50,
        "DSS": 0.0,
        "METHANOL-D4": 3.31,
        "TMS": 0.0,
        "TSP": 0.0,
    },
    "13C": {
        "ACETONE": 29.84,
        "ACETONE-D6": 29.84,
        "BENZENE-D6": 128.06,
        "C6D6": 128.06,
        "CD2CL2": 53.84,
        "CD3OD": 49.00,
        "CDCL3": 77.16,
        "CHLOROFORM-D": 77.16,
        "DICHLOROMETHANE-D2": 53.84,
        "DIMETHYLSULFOXIDE-D6": 39.52,
        "DMSO": 39.52,
        "DMSO-D6": 39.52,
        "METHANOL-D4": 49.00,
        "TMS": 0.0,
    },
}


class NMRProcessingSourceError(ValueError):
    """Raised when persisted NMR source data are incomplete or corrupted."""


def _finite_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise NMRProcessingSourceError(f"{name} must be a one-dimensional array")
    if not np.isfinite(array).all():
        raise NMRProcessingSourceError(f"{name} contains NaN or infinite values")
    return array


def _source_checksum(
    x: np.ndarray,
    original_y: np.ndarray,
    quadrature_real: np.ndarray | None,
    quadrature_imaginary: np.ndarray | None,
) -> str:
    digest = hashlib.sha256()
    for name, array in (
        ("x", x),
        ("original_y", original_y),
        ("quadrature_real", quadrature_real),
        ("quadrature_imaginary", quadrature_imaginary),
    ):
        digest.update(name.encode("ascii"))
        if array is None:
            digest.update(b"\x00")
            continue
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(len(canonical).to_bytes(8, "little", signed=False))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _metadata_checksum(metadata: dict[str, Any]) -> str:
    encoded = json.dumps(
        metadata,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_processing_source(
    x_data: Any,
    original_y_data: Any,
    *,
    quadrature_real_data: Any | None = None,
    quadrature_imaginary_data: Any | None = None,
    source_kind: str,
    source_domain: str = "frequency",
    source_quality: str = "vendor_data",
    default_phase: dict[str, Any] | None = None,
    default_baseline: dict[str, Any] | None = None,
    reference_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable, JSON-serializable source used to replay processing.

    ``original_y_data`` is the exact spectrum initially presented to the user.
    The optional quadrature arrays are the phase-ready real and imaginary
    channels. They may differ from the displayed spectrum when an importer
    applied an automatic baseline correction.
    """

    x = _finite_vector(x_data, "x_data")
    original_y = _finite_vector(original_y_data, "original_y_data")
    if len(x) != len(original_y):
        raise NMRProcessingSourceError("x_data and original_y_data lengths differ")

    q_real = (
        _finite_vector(quadrature_real_data, "quadrature_real_data")
        if quadrature_real_data is not None
        else None
    )
    q_imag = (
        _finite_vector(quadrature_imaginary_data, "quadrature_imaginary_data")
        if quadrature_imaginary_data is not None
        else None
    )
    if (q_real is None) != (q_imag is None):
        raise NMRProcessingSourceError(
            "both quadrature real and imaginary channels are required"
        )
    if q_real is not None and (len(q_real) != len(x) or len(q_imag) != len(x)):
        raise NMRProcessingSourceError("quadrature channel lengths differ from x_data")

    q_real_is_original = q_real is not None and np.array_equal(q_real, original_y)
    arrays = [x, original_y]
    if q_real is not None:
        if not q_real_is_original:
            arrays.append(q_real)
        arrays.append(q_imag)
    estimated_bytes = int(sum(array.nbytes for array in arrays))
    checksum = _source_checksum(x, original_y, q_real, q_imag)
    original_x_list = x.tolist()
    original_y_list = original_y.tolist()
    q_real_list = (
        None if q_real_is_original else q_real.tolist() if q_real is not None else None
    )
    q_imaginary_list = q_imag.tolist() if q_imag is not None else None
    estimated_inline_json_bytes = len(
        json.dumps(
            {
                "original_x_data": original_x_list,
                "original_y_data": original_y_list,
                "quadrature_real_data": q_real_list,
                "quadrature_imaginary_data": q_imaginary_list,
            },
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    )

    processing_metadata = {
        "pipeline_version": PROCESSING_PIPELINE_VERSION,
        "source_kind": str(source_kind),
        "source_domain": str(source_domain),
        "source_quality": str(source_quality),
        "default_phase": dict(default_phase or {}),
        "default_baseline": dict(default_baseline or {}),
        "reference_metadata": dict(reference_metadata or {}),
    }

    return {
        "schema_version": PROCESSING_SOURCE_SCHEMA_VERSION,
        "pipeline_version": PROCESSING_PIPELINE_VERSION,
        "immutable": True,
        "storage": "inline_json_float64",
        "source_kind": str(source_kind),
        "source_domain": str(source_domain),
        "source_quality": str(source_quality),
        "point_count": len(x),
        "has_quadrature": q_real is not None,
        "estimated_binary_bytes": estimated_bytes,
        "estimated_inline_json_bytes": estimated_inline_json_bytes,
        "checksum_sha256": checksum,
        "metadata_checksum_sha256": _metadata_checksum(processing_metadata),
        "original_x_data": original_x_list,
        "original_y_data": original_y_list,
        "quadrature_real_source": (
            None
            if q_real is None
            else "original_y_data"
            if q_real_is_original
            else "inline"
        ),
        "quadrature_real_data": q_real_list,
        "quadrature_imaginary_data": q_imaginary_list,
        "default_phase": processing_metadata["default_phase"],
        "default_baseline": dict(default_baseline or {}),
        "reference_metadata": processing_metadata["reference_metadata"],
    }


def processing_source_summary(source: dict[str, Any] | None) -> dict[str, Any] | None:
    if not source:
        return None
    return {
        "schema_version": source.get("schema_version"),
        "pipeline_version": source.get("pipeline_version"),
        "immutable": bool(source.get("immutable")),
        "storage": source.get("storage"),
        "source_kind": source.get("source_kind"),
        "source_domain": source.get("source_domain"),
        "source_quality": source.get("source_quality"),
        "point_count": int(source.get("point_count") or 0),
        "has_quadrature": bool(source.get("has_quadrature")),
        "quadrature_real_source": source.get("quadrature_real_source"),
        "estimated_binary_bytes": int(source.get("estimated_binary_bytes") or 0),
        "estimated_inline_json_bytes": int(
            source.get("estimated_inline_json_bytes") or 0
        ),
        "checksum_sha256": source.get("checksum_sha256"),
        "metadata_checksum_sha256": source.get("metadata_checksum_sha256"),
        "default_phase": dict(source.get("default_phase") or {}),
        "default_baseline": dict(source.get("default_baseline") or {}),
        "reference_metadata": dict(source.get("reference_metadata") or {}),
    }


def read_processing_source(
    source: dict[str, Any],
    *,
    verify_checksum: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if int(source.get("schema_version") or 0) != PROCESSING_SOURCE_SCHEMA_VERSION:
        raise NMRProcessingSourceError("unsupported NMR processing source schema")
    if not source.get("immutable"):
        raise NMRProcessingSourceError("NMR processing source is not marked immutable")

    x = _finite_vector(source.get("original_x_data"), "original_x_data")
    original_y = _finite_vector(source.get("original_y_data"), "original_y_data")
    if len(x) != len(original_y):
        raise NMRProcessingSourceError("persisted original x/y lengths differ")

    q_real_raw = source.get("quadrature_real_data")
    q_imag_raw = source.get("quadrature_imaginary_data")
    real_source = source.get("quadrature_real_source")
    q_real = (
        original_y.copy()
        if real_source == "original_y_data"
        else _finite_vector(q_real_raw, "quadrature_real_data")
        if q_real_raw is not None
        else None
    )
    q_imag = (
        _finite_vector(q_imag_raw, "quadrature_imaginary_data")
        if q_imag_raw is not None
        else None
    )
    if (q_real is None) != (q_imag is None):
        raise NMRProcessingSourceError("persisted quadrature channel is incomplete")
    if q_real is not None and (len(q_real) != len(x) or len(q_imag) != len(x)):
        raise NMRProcessingSourceError("persisted quadrature channel lengths differ")

    expected_points = int(source.get("point_count") or 0)
    if expected_points != len(x):
        raise NMRProcessingSourceError("persisted NMR source point count is inconsistent")
    if verify_checksum:
        expected = str(source.get("checksum_sha256") or "")
        actual = _source_checksum(x, original_y, q_real, q_imag)
        if not expected or expected != actual:
            raise NMRProcessingSourceError("persisted NMR source checksum mismatch")
        expected_metadata = str(source.get("metadata_checksum_sha256") or "")
        if expected_metadata:
            processing_metadata = {
                "pipeline_version": source.get("pipeline_version"),
                "source_kind": str(source.get("source_kind") or ""),
                "source_domain": str(source.get("source_domain") or ""),
                "source_quality": str(source.get("source_quality") or ""),
                "default_phase": dict(source.get("default_phase") or {}),
                "default_baseline": dict(source.get("default_baseline") or {}),
                "reference_metadata": dict(source.get("reference_metadata") or {}),
            }
            if expected_metadata != _metadata_checksum(processing_metadata):
                raise NMRProcessingSourceError(
                    "persisted NMR source metadata checksum mismatch"
                )
    return x, original_y, q_real, q_imag


def apply_phase_correction(
    x_data: Any,
    real_data: Any,
    imaginary_data: Any,
    *,
    zero_deg: float = 0.0,
    first_deg: float = 0.0,
    pivot_ppm: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an incremental zero/first-order phase without clipping negatives."""

    x = _finite_vector(x_data, "x_data")
    real = _finite_vector(real_data, "real_data")
    imaginary = _finite_vector(imaginary_data, "imaginary_data")
    if len(x) != len(real) or len(real) != len(imaginary):
        raise ValueError("phase correction arrays must have the same length")
    if len(x) == 0:
        return real.copy(), imaginary.copy()

    zero = float(zero_deg)
    first = float(first_deg)
    pivot = float(np.median(x)) if pivot_ppm is None else float(pivot_ppm)
    if not np.isfinite([zero, first, pivot]).all():
        raise ValueError("phase parameters must be finite")
    span = max(float(np.max(x) - np.min(x)), np.finfo(float).eps)
    phase = np.deg2rad(zero + first * (x - pivot) / span)
    phased = (real + 1j * imaginary) * np.exp(1j * phase)
    return np.real(phased), np.imag(phased)


def automatic_phase_correction(
    x_data: Any,
    real_data: Any,
    imaginary_data: Any,
    *,
    pivot_ppm: float | None = None,
    optimize_first_order: bool = True,
    max_first_deg: float = 720.0,
    max_fit_points: int = 4096,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool | str]]:
    """Estimate deterministic zero/first-order phase on a complex 1D spectrum.

    The objective is evaluated only on high-magnitude points and combines
    negative absorptive energy with residual imaginary energy.  The returned
    phase is always applied to the full-resolution input.
    """

    x = _finite_vector(x_data, "x_data")
    real = _finite_vector(real_data, "real_data")
    imaginary = _finite_vector(imaginary_data, "imaginary_data")
    if len(x) != len(real) or len(real) != len(imaginary):
        raise ValueError("automatic phase correction arrays must have the same length")
    if len(x) == 0:
        return real.copy(), imaginary.copy(), {
            "method": "negative_and_imaginary_energy",
            "zero_deg": 0.0,
            "first_deg": 0.0,
            "pivot_ppm": 0.0,
            "objective_before": 0.0,
            "objective_after": 0.0,
            "converged": True,
        }

    pivot = float(np.median(x)) if pivot_ppm is None else float(pivot_ppm)
    max_first = float(max_first_deg)
    fit_points = int(max_fit_points)
    if not np.isfinite([pivot, max_first]).all() or max_first < 0:
        raise ValueError("automatic phase parameters must be finite and non-negative")
    if not 128 <= fit_points <= 65536:
        raise ValueError("automatic phase max_fit_points must be between 128 and 65536")

    if len(x) > fit_points:
        index = np.unique(np.linspace(0, len(x) - 1, fit_points).astype(int))
        fit_x = x[index]
        fit_complex = real[index] + 1j * imaginary[index]
    else:
        fit_x = x
        fit_complex = real + 1j * imaginary

    magnitude = np.abs(fit_complex)
    finite_magnitude = magnitude[np.isfinite(magnitude)]
    if not len(finite_magnitude) or float(np.max(finite_magnitude)) <= float(
        np.finfo(float).eps
    ):
        return real.copy(), imaginary.copy(), {
            "method": "negative_and_imaginary_energy",
            "zero_deg": 0.0,
            "first_deg": 0.0,
            "pivot_ppm": pivot,
            "objective_before": 0.0,
            "objective_after": 0.0,
            "converged": True,
        }
    threshold = (
        float(np.quantile(finite_magnitude, 0.80))
        if len(finite_magnitude)
        else 0.0
    )
    signal_mask = magnitude >= threshold
    if int(np.count_nonzero(signal_mask)) < min(32, len(fit_complex)):
        signal_mask = np.ones(len(fit_complex), dtype=bool)
    fit_x = fit_x[signal_mask]
    fit_complex = fit_complex[signal_mask]
    fit_scale = float(np.max(np.abs(fit_complex)))
    if fit_scale > 0:
        fit_complex = fit_complex / fit_scale
    span = max(float(np.max(x) - np.min(x)), np.finfo(float).eps)
    coordinate = (fit_x - pivot) / span
    eps = np.finfo(float).eps

    def objective(parameters: np.ndarray) -> float:
        zero, first = float(parameters[0]), float(parameters[1])
        phased = fit_complex * np.exp(
            1j * np.deg2rad(zero + first * coordinate)
        )
        absorptive = np.real(phased)
        dispersive = np.imag(phased)
        negative = float(np.sum(np.minimum(absorptive, 0.0) ** 2))
        positive = float(np.sum(np.maximum(absorptive, 0.0) ** 2))
        real_energy = float(np.sum(absorptive**2))
        imaginary_energy = float(np.sum(dispersive**2))
        return 4.0 * negative / max(positive, eps) + imaginary_energy / max(
            real_energy + imaginary_energy, eps
        )

    use_first_order = bool(
        optimize_first_order and max_first > np.finfo(float).eps
    )
    bounds = [(-180.0, 180.0), (-max_first, max_first)]
    starts = (
        (-120.0, 0.0),
        (-60.0, 0.0),
        (0.0, 0.0),
        (60.0, 0.0),
        (120.0, 0.0),
    )
    best = None
    for start in starts:
        if use_first_order:
            fitted = minimize(
                objective,
                np.asarray(start, dtype=np.float64),
                method="Powell",
                bounds=bounds,
                options={"xtol": 1e-5, "ftol": 1e-8, "maxiter": 300},
            )
        else:
            fitted = minimize(
                lambda parameter: objective(
                    np.asarray([float(parameter[0]), 0.0], dtype=np.float64)
                ),
                np.asarray([start[0]], dtype=np.float64),
                method="Powell",
                bounds=[(-180.0, 180.0)],
                options={"xtol": 1e-5, "ftol": 1e-8, "maxiter": 300},
            )
            fitted.x = np.asarray([float(fitted.x[0]), 0.0], dtype=np.float64)
        if best is None or float(fitted.fun) < float(best.fun):
            best = fitted

    assert best is not None
    zero = float(best.x[0])
    first = float(best.x[1]) if use_first_order else 0.0
    phased_real, phased_imaginary = apply_phase_correction(
        x,
        real,
        imaginary,
        zero_deg=zero,
        first_deg=first,
        pivot_ppm=pivot,
    )
    return phased_real, phased_imaginary, {
        "method": "negative_and_imaginary_energy",
        "zero_deg": zero,
        "first_deg": first,
        "pivot_ppm": pivot,
        "objective_before": float(objective(np.asarray([0.0, 0.0]))),
        "objective_after": float(best.fun),
        "converged": bool(best.success),
    }


def _robust_noise_sigma(y: np.ndarray) -> float:
    if len(y) < 3:
        return 0.0
    scale = float(np.max(np.abs(y)))
    if scale <= 0:
        return 0.0
    differences = np.diff(y / scale)
    center = float(np.median(differences))
    mad = float(np.median(np.abs(differences - center)))
    return float(scale * 1.4826 * mad / np.sqrt(2.0))


def spectrum_quality_metrics(
    x_data: Any,
    y_data: Any,
    *,
    imaginary_data: Any | None = None,
) -> dict[str, Any]:
    """Return compact, JSON-safe QC metrics without exposing internal channels."""

    x = _finite_vector(x_data, "x_data")
    y = _finite_vector(y_data, "y_data")
    if len(x) != len(y):
        raise ValueError("quality metric x/y arrays must have the same length")
    imaginary = (
        _finite_vector(imaginary_data, "imaginary_data")
        if imaginary_data is not None
        else None
    )
    if imaginary is not None and len(imaginary) != len(y):
        raise ValueError("quality metric imaginary array length differs from y_data")

    size = len(y)
    if size == 0:
        return {
            "schema_version": QUALITY_METRICS_SCHEMA_VERSION,
            "pipeline_version": PROCESSING_PIPELINE_VERSION,
            "point_count": 0,
            "quality_flags": ["empty_spectrum"],
        }

    noise = _robust_noise_sigma(y)
    absolute_max = float(np.max(np.abs(y)))
    scaled_y = y / absolute_max if absolute_max > 0 else y
    energy = float(np.sum(scaled_y**2))
    negative_energy = float(np.sum(np.minimum(scaled_y, 0.0) ** 2))
    spacing = np.abs(np.diff(x))
    median_spacing = float(np.median(spacing)) if len(spacing) else 0.0
    spacing_mad = (
        1.4826 * float(np.median(np.abs(spacing - median_spacing)))
        if len(spacing)
        else 0.0
    )

    bin_count = min(32, max(4, size // 128))
    chunks = np.array_split(np.arange(size), bin_count)
    centers: list[float] = []
    quiet_medians: list[float] = []
    activities: list[float] = []
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        chunk_y = y[chunk]
        cutoff = float(np.quantile(chunk_y, 0.60))
        quiet = chunk_y[chunk_y <= cutoff]
        if len(quiet) == 0:
            quiet = chunk_y
        centers.append(float(np.median(x[chunk])))
        quiet_medians.append(float(np.median(quiet)))
        activities.append(
            float(np.quantile(chunk_y, 0.90) - np.quantile(chunk_y, 0.10))
        )

    center_array = np.asarray(centers, dtype=np.float64)
    quiet_array = np.asarray(quiet_medians, dtype=np.float64)
    activity_array = np.asarray(activities, dtype=np.float64)
    if len(activity_array) >= 8:
        quiet_bin_mask = activity_array <= float(
            np.quantile(activity_array, 0.60)
        )
        if int(np.count_nonzero(quiet_bin_mask)) >= 4:
            center_array = center_array[quiet_bin_mask]
            quiet_array = quiet_array[quiet_bin_mask]
    baseline_span = (
        float(np.quantile(quiet_array, 0.90) - np.quantile(quiet_array, 0.10))
        if len(quiet_array)
        else 0.0
    )
    if len(center_array) >= 2 and float(np.ptp(center_array)) > np.finfo(
        float
    ).eps:
        baseline_slope = float(np.polyfit(center_array, quiet_array, 1)[0])
    else:
        baseline_slope = 0.0
    baseline_center = float(np.median(quiet_array)) if len(quiet_array) else 0.0
    if len(quiet_array):
        baseline_deviation = quiet_array - baseline_center
        deviation_scale = float(np.max(np.abs(baseline_deviation)))
        baseline_rms = (
            deviation_scale
            * float(
                np.sqrt(
                    np.mean((baseline_deviation / deviation_scale) ** 2)
                )
            )
            if deviation_scale > 0
            else 0.0
        )
    else:
        baseline_rms = 0.0

    imaginary_fraction = None
    if imaginary is not None:
        combined_scale = max(absolute_max, float(np.max(np.abs(imaginary))))
        if combined_scale > 0:
            real_energy = float(np.sum((y / combined_scale) ** 2))
            imaginary_energy = float(
                np.sum((imaginary / combined_scale) ** 2)
            )
            imaginary_fraction = imaginary_energy / max(
                real_energy + imaginary_energy,
                float(np.finfo(float).eps),
            )
        else:
            imaginary_fraction = 0.0

    flags: list[str] = []
    if size < 128:
        flags.append("low_point_count")
    if median_spacing > 0 and spacing_mad / median_spacing > 1e-3:
        flags.append("non_uniform_axis")
    drift_threshold = max(
        5.0 * noise,
        absolute_max * 1e-3,
        float(np.finfo(float).eps),
    )
    if baseline_span > drift_threshold:
        flags.append("baseline_drift")
    if energy > 0 and negative_energy / energy > 0.20:
        flags.append("high_negative_energy")
    if imaginary_fraction is not None and imaginary_fraction > 0.5:
        flags.append("high_imaginary_energy")
    if noise > 0 and absolute_max / noise < 5.0:
        flags.append("low_signal_to_noise")

    return {
        "schema_version": QUALITY_METRICS_SCHEMA_VERSION,
        "pipeline_version": PROCESSING_PIPELINE_VERSION,
        "point_count": size,
        "x_min": float(np.min(x)),
        "x_max": float(np.max(x)),
        "median_spacing": median_spacing,
        "spacing_mad": spacing_mad,
        "intensity_min": float(np.min(y)),
        "intensity_max": float(np.max(y)),
        "intensity_median": float(np.median(y)),
        "noise_sigma": noise,
        "max_abs_signal_to_noise": (
            absolute_max / noise if noise > np.finfo(float).eps else None
        ),
        "negative_energy_fraction": (
            negative_energy / energy if energy > np.finfo(float).eps else 0.0
        ),
        "baseline_offset": baseline_center,
        "baseline_span": baseline_span,
        "baseline_rms": baseline_rms,
        "baseline_slope_per_x": baseline_slope,
        "imaginary_energy_fraction": imaginary_fraction,
        "quality_flags": flags,
    }


def solvent_reference_ppm(solvent: str, nucleus: str) -> float | None:
    normalized_solvent = (
        str(solvent or "")
        .strip()
        .replace("<", "")
        .replace(">", "")
        .replace(" ", "")
        .upper()
    )
    normalized_nucleus = str(nucleus or "").strip().replace("^", "").upper()
    references = _SOLVENT_REFERENCE_PPM.get(normalized_nucleus, {})
    return references.get(normalized_solvent)


def locate_reference_peak(
    x_data: Any,
    y_data: Any,
    *,
    expected_ppm: float,
    window_ppm: float = 0.15,
    min_snr: float = 5.0,
) -> dict[str, float | str] | None:
    """Locate a traceable high-S/N solvent/reference line near an expected ppm."""

    x = _finite_vector(x_data, "x_data")
    y = _finite_vector(y_data, "y_data")
    if len(x) != len(y):
        raise ValueError("reference x/y arrays must have the same length")
    expected = float(expected_ppm)
    window = float(window_ppm)
    required_snr = float(min_snr)
    if not np.isfinite([expected, window, required_snr]).all():
        raise ValueError("reference parameters must be finite")
    if window <= 0 or required_snr <= 0:
        raise ValueError("reference window and minimum S/N must be positive")

    mask = np.abs(x - expected) <= window
    if not np.any(mask):
        return None
    indices = np.flatnonzero(mask)
    local = y[indices]
    relative_index = int(np.argmax(np.abs(local)))
    index = int(indices[relative_index])
    noise = _robust_noise_sigma(y)
    snr = float(abs(float(y[index])) / max(noise, float(np.finfo(float).eps)))
    if snr < required_snr:
        return None
    return {
        "status": "located",
        "observed_ppm": float(x[index]),
        "expected_ppm": expected,
        "intensity": float(y[index]),
        "snr": snr,
        "window_ppm": window,
    }


def asymmetric_least_squares_baseline(
    y_data: Any,
    *,
    smoothness: float = 1e7,
    asymmetry: float = 0.001,
    iterations: int = 8,
    max_fit_points: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate a smooth asymmetric baseline and retain signed residual noise."""

    y = _finite_vector(y_data, "y_data")
    size = len(y)
    if size == 0:
        return y.copy(), y.copy()
    if size < 5:
        baseline = np.full_like(y, float(np.median(y)))
        return baseline, y - baseline

    lam = float(smoothness)
    p = float(asymmetry)
    n_iter = int(iterations)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("baseline smoothness must be a positive finite number")
    if not np.isfinite(p) or not 0 < p < 1:
        raise ValueError("baseline asymmetry must be between 0 and 1")
    if not 1 <= n_iter <= 50:
        raise ValueError("baseline iterations must be between 1 and 50")

    if size > max_fit_points:
        sample_index = np.linspace(0, size - 1, max_fit_points).astype(int)
        sample_index = np.unique(sample_index)
        fit_y = y[sample_index]
        fit_baseline = _als_fit(fit_y, lam, p, n_iter)
        baseline = np.interp(np.arange(size), sample_index, fit_baseline)
    else:
        baseline = _als_fit(y, lam, p, n_iter)

    corrected = y - baseline
    corrected -= float(np.median(corrected))
    return np.asarray(baseline, dtype=np.float64), np.asarray(corrected, dtype=np.float64)


def _als_fit(y: np.ndarray, smoothness: float, asymmetry: float, iterations: int) -> np.ndarray:
    size = len(y)
    differences = sparse.diags(
        [np.ones(size - 2), -2.0 * np.ones(size - 2), np.ones(size - 2)],
        [0, 1, 2],
        shape=(size - 2, size),
        format="csc",
        dtype=np.float64,
    )
    penalty = smoothness * (differences.T @ differences)
    weights = np.ones(size, dtype=np.float64)
    baseline = np.zeros(size, dtype=np.float64)
    for _ in range(iterations):
        system = sparse.diags(weights, 0, shape=(size, size), format="csc") + penalty
        baseline = spsolve(system, weights * y)
        weights = asymmetry * (y > baseline) + (1.0 - asymmetry) * (y <= baseline)
    return np.asarray(baseline, dtype=np.float64)
