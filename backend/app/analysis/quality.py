from __future__ import annotations

from typing import Any

import numpy as np

from app.analysis.models import AnalysisResult
from app.core.models import Spectrum


def assess_quality(spectrum: Spectrum, result: AnalysisResult | None = None) -> dict[str, Any]:
    x = np.asarray(spectrum.x_data, dtype=float)
    y = np.asarray(spectrum.y_data, dtype=float)
    warnings: list[str] = []
    info: list[str] = []
    score = 1.0

    if len(x) != len(y):
        warnings.append("X/Y data length mismatch.")
        score -= 0.35

    if len(x) < 20:
        warnings.append("Very few data points; analysis may be unreliable.")
        score -= 0.25
    elif len(x) < 200:
        info.append("Low point count; peak widths and integration are approximate.")
        score -= 0.08

    finite_mask = np.isfinite(x) & np.isfinite(y)
    if len(x) and not finite_mask.all():
        invalid = int((~finite_mask).sum())
        warnings.append(f"{invalid} non-finite data point(s) detected.")
        score -= min(0.3, invalid / max(len(x), 1))

    if len(y):
        y_finite = y[np.isfinite(y)]
        if len(y_finite):
            y_min = float(np.min(y_finite))
            y_max = float(np.max(y_finite))
            dynamic_range = y_max - y_min
            if dynamic_range <= 1e-12:
                warnings.append("Signal is nearly flat.")
                score -= 0.35
            else:
                edge_n = max(min(len(y_finite) // 20, 200), 5)
                edge = np.concatenate([y_finite[:edge_n], y_finite[-edge_n:]])
                noise = float(np.std(edge))
                snr = dynamic_range / noise if noise > 1e-12 else float("inf")
                if snr < 5:
                    warnings.append("Low signal-to-noise ratio.")
                    score -= 0.2
                elif snr < 15:
                    info.append("Moderate signal-to-noise ratio.")
                    score -= 0.06

                edge_span = abs(float(np.mean(y_finite[-edge_n:]) - np.mean(y_finite[:edge_n])))
                if edge_span > dynamic_range * 0.25:
                    info.append("Baseline drift is visible across the trace.")
                    score -= 0.06

    if result is not None:
        n_peaks = len(result.peaks)
        if n_peaks == 0:
            warnings.append("No peaks detected with current analysis parameters.")
            score -= 0.2
        elif n_peaks > 200:
            info.append("Large number of detected peaks; consider raising thresholds.")
            score -= 0.08

        technique = spectrum.technique.value
        if technique == "HPLC":
            channel_peaks = result.metrics.get("channel_peaks") or {}
            for channel_name, channel_data in channel_peaks.items():
                peaks = channel_data.get("peaks", [])
                if not peaks:
                    warnings.append(f"{channel_name}: no integrated peaks.")
                    score -= 0.08
                    continue
                total_area = float(channel_data.get("total_area", 0) or 0)
                if total_area <= 0:
                    warnings.append(f"{channel_name}: total peak area is zero or negative.")
                    score -= 0.12
                sorted_peaks = sorted(peaks, key=lambda p: float(p.get("position", 0)))
                for left, right in zip(sorted_peaks, sorted_peaks[1:]):
                    left_end = float(left.get("end_time", left.get("position", 0)))
                    right_start = float(right.get("begin_time", right.get("position", 0)))
                    if right_start < left_end:
                        info.append(f"{channel_name}: adjacent peaks at {left.get('position', 0):.3f}/{right.get('position', 0):.3f} min overlap.")
                        score -= 0.03
                        break
                if channel_data.get("source") == "computed":
                    info.append(f"{channel_name}: peak table is computed by ChemApp; verify against instrument integration when available.")

        if technique == "NMR":
            integrals = getattr(result, "integrals", []) or []
            if not integrals:
                info.append("NMR integrals are missing; quantitative interpretation is limited.")
                score -= 0.08
            if result.metrics.get("solvent_reference") is None and not result.metrics.get("reference_corrected"):
                info.append("NMR chemical shift reference was not explicitly confirmed.")
                score -= 0.04
            multiplets = getattr(result, "multiplets", []) or []
            unresolved = [m for m in multiplets if not m.get("multiplicity")]
            if multiplets and len(unresolved) / max(len(multiplets), 1) > 0.5:
                info.append("Many NMR multiplets are unresolved; consider manual merge/split review.")
                score -= 0.05

        if technique == "XRD":
            phase_matches = getattr(result, "phase_matches", []) or []
            assignments = getattr(result, "peak_assignments", []) or []
            if not phase_matches:
                warnings.append("XRD phase matching found no candidate phase.")
                score -= 0.18
            elif phase_matches[0].get("match_score", 0) < 50:
                info.append("XRD top phase match score is low; verify background and reference database.")
                score -= 0.08
            if n_peaks and len(assignments) / max(n_peaks, 1) < 0.4:
                info.append("Less than 40% of XRD peaks were assigned to reference phases.")
                score -= 0.06

        if technique == "UV-Vis":
            if result.metrics.get("saturated_points", 0):
                warnings.append("UV-Vis spectrum may contain saturated absorbance points.")
                score -= 0.12
            if hasattr(result, "calibration") and result.calibration:
                r2 = float(result.calibration.get("r_squared", 1))
                if r2 < 0.99:
                    info.append("UV-Vis calibration R² is below 0.99.")
                    score -= 0.06

    score = max(0.0, min(1.0, score))
    if score >= 0.8:
        status = "good"
    elif score >= 0.55:
        status = "review"
    else:
        status = "poor"

    return {
        "status": status,
        "score": round(score, 3),
        "warnings": warnings,
        "info": info,
        "points": int(len(x)),
    }
