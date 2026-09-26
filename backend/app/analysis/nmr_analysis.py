from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal

from app.analysis.base import BaseAnalyzer
from app.analysis.models import NMRAnalysisResult
from app.analysis.nmr_processing import (
    PROCESSING_SOURCE_KEY,
    apply_phase_correction,
    asymmetric_least_squares_baseline,
    read_processing_source,
)
from app.analysis.utils import estimate_noise_region, trapezoidal_integrate
from app.core.models import Peak, Spectrum

_SOLVENT_REFERENCES = {
    "CDCl3": 7.26,
    "DMSO": 2.50,
    "D2O": 4.79,
    "CD3OD": 3.31,
    "C6D6": 7.15,
    "CD2Cl2": 5.32,
    "Acetone": 2.05,
}
_PROMINENCE_FACTOR = 0.03
_MIN_PEAK_DISTANCE = 5
_INTEGRAL_WING_FACTOR = 2.0
_MULTIPLET_J_WINDOW_PPM = 0.05
_MULTIPLET_LINE_REL_THRESHOLD = 0.03
_MULTIPLET_CLASS_REL_THRESHOLD = 0.08
_DEFAULT_SINGLET_MIN_SNR = 8.0


class NMRAnalyzer(BaseAnalyzer):
    technique = "nmr"

    def analyze(self, spectrum: Spectrum, options: dict[str, Any] | None = None) -> NMRAnalysisResult:
        options = options or {}
        x = np.asarray(spectrum.x_data, dtype=np.float64)
        y = np.asarray(spectrum.y_data, dtype=np.float64)
        if spectrum.parameters.get("signal_representation") == "magnitude":
            y = np.abs(y)

        baseline_percentile = float(options.get("baseline_percentile", 10))
        noise_factor = float(options.get("noise_factor", 3))
        default_prominence_factor = (
            5 if spectrum.parameters.get("phase_source") == "vendor_processed" else 8
        )
        prominence_factor = float(
            options.get("prominence_factor", default_prominence_factor)
        )
        min_peak_distance = int(options.get("min_peak_distance", _MIN_PEAK_DISTANCE))
        auto_reference = bool(options.get("auto_reference", False))
        hide_solvent_peaks = bool(options.get("hide_solvent_peaks", False))
        phase_zero_deg = float(options.get("phase_zero_deg", 0) or 0)
        phase_first_deg = float(options.get("phase_first_deg", 0) or 0)
        phase_requested = abs(phase_zero_deg) > 1e-9 or abs(phase_first_deg) > 1e-9
        phase_status = "not_requested"
        if phase_requested:
            source = spectrum.parameters.get(PROCESSING_SOURCE_KEY)
            if not source:
                raise ValueError(
                    "Phase correction requires a persisted real/imaginary quadrature source"
                )
            source_x, _, quadrature_real, quadrature_imaginary = read_processing_source(source)
            if quadrature_real is None or quadrature_imaginary is None:
                raise ValueError(
                    "Phase correction requires a persisted real/imaginary quadrature source"
                )
            x = source_x
            pivot = float(options.get("phase_pivot_ppm", np.median(x)))
            y, _ = apply_phase_correction(
                x,
                quadrature_real,
                quadrature_imaginary,
                zero_deg=phase_zero_deg,
                first_deg=phase_first_deg,
                pivot_ppm=pivot,
            )
            default_baseline = dict(source.get("default_baseline") or {})
            if default_baseline.get("method") == "asymmetric_least_squares":
                _, y = asymmetric_least_squares_baseline(
                    y,
                    smoothness=float(default_baseline.get("smoothness", 1e7)),
                    asymmetry=float(default_baseline.get("asymmetry", 0.001)),
                    iterations=int(default_baseline.get("iterations", 8)),
                    max_fit_points=int(default_baseline.get("max_fit_points", 8192)),
                )
            phase_status = "applied"

        input_min = float(np.min(y)) if len(y) else 0.0
        input_max = float(np.max(y)) if len(y) else 0.0
        negative_fraction = float(np.mean(y < 0)) if len(y) else 0.0

        baseline_method = str(options.get("baseline_method") or "percentile")
        if baseline_method in {"asymmetric_least_squares", "als"}:
            baseline_curve, y_signed = asymmetric_least_squares_baseline(
                y,
                smoothness=float(options.get("baseline_smoothness", 1e7)),
                asymmetry=float(options.get("baseline_asymmetry", 0.001)),
                iterations=int(options.get("baseline_iterations", 8)),
            )
            baseline = float(np.median(baseline_curve))
            baseline_span = float(np.max(baseline_curve) - np.min(baseline_curve))
            baseline_method = "asymmetric_least_squares"
        else:
            baseline = float(np.percentile(y, baseline_percentile))
            baseline_span = 0.0
            y_signed = y - baseline

        # Peak detection is positive-only, but the signed residual is retained
        # for noise estimation and diagnostics instead of being destroyed.
        y_corrected = np.maximum(y_signed, 0)
        edge_points = max(int(len(y_signed) * 0.05), 10)
        edge = np.concatenate((y_signed[:edge_points], y_signed[-edge_points:]))
        edge_median = float(np.median(edge)) if len(edge) else 0.0
        noise_level = (
            1.4826 * float(np.median(np.abs(edge - edge_median)))
            if len(edge)
            else 0.0
        )
        if noise_level <= np.finfo(float).eps:
            noise_level = estimate_noise_region(y_signed, fraction=0.05)
        if (
            spectrum.parameters.get("phase_corrected")
            and spectrum.parameters.get("phase_source") != "vendor_processed"
        ):
            # Autophased real spectra retain small signed baseline noise. A
            # relative floor prevents side lobes and residual ripple from
            # being promoted to thousands of peaks on high-dynamic-range FIDs.
            dynamic_range = float(np.max(y_corrected) - np.min(y_corrected))
            noise_floor_fraction = float(options.get("noise_floor_fraction", 0.002))
            noise_level = max(noise_level, dynamic_range * noise_floor_fraction)
        if noise_level <= np.finfo(float).eps:
            # A magnitude spectrum can become exactly zero in a quiet edge
            # region after baseline clipping (common for JEOL FIDs). Use the
            # lower quartile of positive residuals instead of accepting a
            # zero threshold, which would classify thousands of noise bins.
            positive = y_corrected[y_corrected > 0]
            if len(positive):
                noise_level = float(np.percentile(positive, 25))
        prominence = noise_level * prominence_factor

        peak_indices, peak_props = signal.find_peaks(
            y_corrected,
            height=noise_level * noise_factor,
            prominence=prominence,
            distance=min_peak_distance,
        )

        peaks = self._build_peaks(x, y_corrected, peak_indices, peak_props)
        detected_peak_count = len(peaks)
        solvent = spectrum.metadata.solvent or spectrum.parameters.get("solvent", "")
        solvent_ref = _SOLVENT_REFERENCES.get(solvent)
        solvent_peaks = []
        if solvent_ref is not None:
            for p in peaks:
                if abs(p.position - solvent_ref) <= float(options.get("solvent_tolerance_ppm", 0.04)):
                    p.assignment = "solvent"
                    solvent_peaks.append(round(p.position, 4))
            if hide_solvent_peaks:
                peaks = [p for p in peaks if p.assignment != "solvent"]

        integrals = self._compute_integrals(x, y_corrected, peaks)
        include_solvent_in_multiplets = bool(
            options.get("include_solvent_in_multiplets", False)
        )
        multiplet_peaks = (
            peaks
            if include_solvent_in_multiplets
            else [peak for peak in peaks if peak.assignment != "solvent"]
        )
        multiplets = self._group_multiplets(
            multiplet_peaks,
            spectrum,
            options,
            noise_level=noise_level,
        )

        total_integral = sum(integ["raw_area"] for integ in integrals)
        if integrals and total_integral > 0:
            areas = [integ["raw_area"] for integ in integrals if integ["raw_area"] > 0]
            min_area = min(areas) if areas else 1.0
            for integ in integrals:
                integ["relative_area"] = round(integ["raw_area"] / min_area, 2) if min_area > 0 else 0

        reference_corrected = False
        solvent_shift = None
        x = np.copy(x)  # Don't mutate original spectrum data
        if auto_reference and solvent in _SOLVENT_REFERENCES:
            ref_ppm = _SOLVENT_REFERENCES[solvent]
            detected = self._find_nearest_peak(peaks, ref_ppm, tolerance=0.3)
            if detected is not None:
                offset = detected.position - ref_ppm
                if abs(offset) > 0.001:
                    reference_corrected = True
                    solvent_shift = ref_ppm
                    x = x - offset
                    for p in peaks:
                        p.position -= offset
                    for integ in integrals:
                        integ["start_ppm"] -= offset
                        integ["end_ppm"] -= offset
                    for mp in multiplets:
                        mp["center_ppm"] = round(mp["center_ppm"] - offset, 4)
                        mp["range_ppm"] = (
                            round(mp["range_ppm"][0] - offset, 4),
                            round(mp["range_ppm"][1] - offset, 4),
                        )
                        mp["component_positions"] = [
                            round(pos - offset, 4) for pos in mp["component_positions"]
                        ]

        frequency_mhz = spectrum.parameters.get("frequency_mhz", 400.0)
        for mp in multiplets:
            j_values = mp.get("j_values_hz", [])
            if j_values:
                mp["estimated_j_hz"] = round(float(np.median(j_values)), 2)
            elif len(mp["component_positions"]) >= 2:
                diffs = np.diff(sorted(mp["component_positions"]))
                if len(diffs) > 0:
                    mp["estimated_j_hz"] = round(float(np.median(diffs)) * frequency_mhz, 2)
            mp["n_components"] = len(mp["component_positions"])

        metrics = {
            "n_peaks": len(peaks),
            "n_multiplets": len(multiplets),
            "noise_level": round(noise_level, 6),
            "baseline": round(baseline, 2),
            "baseline_method": baseline_method,
            "baseline_span": round(baseline_span, 6),
            "total_integral": round(total_integral, 2),
            "solvent_reference": solvent_shift,
            "reference_corrected": reference_corrected,
            "analysis_options": {
                "baseline_percentile": baseline_percentile,
                "noise_factor": noise_factor,
                "prominence_factor": prominence_factor,
                "min_peak_distance": min_peak_distance,
                "auto_reference": auto_reference,
                "hide_solvent_peaks": hide_solvent_peaks,
                "phase_zero_deg": phase_zero_deg,
                "phase_first_deg": phase_first_deg,
                "baseline_method": baseline_method,
            },
            "phase_status": phase_status,
            "signed_intensity": {
                "input_min": input_min,
                "input_max": input_max,
                "baseline_corrected_min": float(np.min(y_signed)) if len(y_signed) else 0.0,
                "baseline_corrected_max": float(np.max(y_signed)) if len(y_signed) else 0.0,
                "negative_point_fraction": negative_fraction,
                "contains_negative_values": bool(input_min < 0),
            },
            "solvent_peaks": solvent_peaks,
            "solvent_hidden": hide_solvent_peaks,
            "multiplet_grouping": {
                "source_peak_count": detected_peak_count,
                "solvent_lines_excluded": (
                    0
                    if include_solvent_in_multiplets
                    else len(solvent_peaks)
                ),
                "include_solvent_lines": include_solvent_in_multiplets,
                "include_isolated_singlets": bool(
                    options.get("include_isolated_singlets", True)
                ),
                "singlet_min_snr": self._singlet_min_snr(options),
                "isolated_singlets_retained": len(
                    [
                        multiplet
                        for multiplet in multiplets
                        if multiplet.get("is_isolated_singlet")
                    ]
                ),
            },
        }

        summary = self._build_summary(peaks, integrals, multiplets, metrics, spectrum)

        return NMRAnalysisResult(
            technique=spectrum.technique,
            peaks=peaks,
            metrics=metrics,
            summary=summary,
            integrals=integrals,
            multiplets=multiplets,
            noise_level=noise_level,
            total_integral=total_integral,
            solvent_shift=solvent_shift,
            reference_corrected=reference_corrected,
        )

    def _build_peaks(self, x, y, indices, props) -> list[Peak]:
        peaks = []
        for i in range(len(indices)):
            idx = int(indices[i])
            width = float(props.get("widths", [0])[i]) if "widths" in props and i < len(props["widths"]) else None
            height = props.get("peak_heights", [0])[i] if "peak_heights" in props else float(y[idx])
            peaks.append(Peak(position=float(x[idx]), intensity=float(height), width=width))
        return peaks

    def _compute_integrals(self, x, y, peaks) -> list[dict]:
        integrals = []
        for p in peaks:
            if p.width and p.width > 0:
                half_width = p.width * _INTEGRAL_WING_FACTOR
            else:
                half_width = 0.01
            start = p.position - half_width
            end = p.position + half_width
            area = trapezoidal_integrate(x, y, start, end)
            integrals.append({
                "center_ppm": round(p.position, 4),
                "start_ppm": round(start, 4),
                "end_ppm": round(end, 4),
                "raw_area": round(area, 4),
                "relative_area": 0.0,
                "intensity": round(p.intensity, 6),
            })
        return integrals

    def _group_multiplets(
        self,
        peaks,
        spectrum,
        options: dict[str, Any] | None = None,
        *,
        noise_level: float = 0.0,
    ) -> list[dict]:
        if not peaks:
            return []
        options = options or {}
        ranges = self._extract_multiplet_ranges(options)
        if ranges:
            multiplets = []
            for region in ranges:
                start, end = region["start"], region["end"]
                lo, hi = min(start, end), max(start, end)
                region_peaks = [p for p in peaks if lo <= p.position <= hi]
                if region_peaks:
                    multiplet = self._build_multiplet(
                        region_peaks,
                        spectrum,
                        bounds=(start, end),
                    )
                    self._annotate_grouping(
                        multiplet,
                        region_peaks,
                        noise_level=noise_level,
                        source="provided_range",
                    )
                    multiplets.append(multiplet)
            return multiplets

        sorted_peaks = sorted(peaks, key=lambda p: p.position, reverse=True)
        multiplets = []
        current_group = [sorted_peaks[0]]

        def append_group(group) -> None:
            is_isolated = len(group) == 1
            if is_isolated:
                if not bool(options.get("include_isolated_singlets", True)):
                    return
                threshold = self._singlet_min_snr(options)
                observed_snr = self._group_signal_to_noise(group, noise_level)
                if observed_snr < threshold:
                    return
            multiplet = self._build_multiplet(group, spectrum)
            self._annotate_grouping(
                multiplet,
                group,
                noise_level=noise_level,
                source="automatic_peak_grouping",
            )
            multiplets.append(multiplet)

        for p in sorted_peaks[1:]:
            last = current_group[-1]
            gap = abs(last.position - p.position)
            if gap < _MULTIPLET_J_WINDOW_PPM:
                current_group.append(p)
            else:
                append_group(current_group)
                current_group = [p]

        append_group(current_group)

        return multiplets

    def _singlet_min_snr(self, options: dict[str, Any]) -> float:
        try:
            value = float(options.get("singlet_min_snr", _DEFAULT_SINGLET_MIN_SNR))
        except (TypeError, ValueError):
            return _DEFAULT_SINGLET_MIN_SNR
        if not np.isfinite(value) or value < 0:
            return _DEFAULT_SINGLET_MIN_SNR
        return value

    def _group_signal_to_noise(self, group, noise_level: float) -> float:
        maximum = max((float(peak.intensity) for peak in group), default=0.0)
        if noise_level <= np.finfo(float).eps:
            return float("inf") if maximum > 0 else 0.0
        return maximum / noise_level

    def _annotate_grouping(
        self,
        multiplet: dict[str, Any],
        group,
        *,
        noise_level: float,
        source: str,
    ) -> None:
        multiplet["grouping_source"] = source
        multiplet["is_isolated_singlet"] = len(group) == 1
        signal_to_noise = self._group_signal_to_noise(group, noise_level)
        multiplet["signal_to_noise"] = (
            None if not np.isfinite(signal_to_noise) else round(signal_to_noise, 3)
        )

    def _extract_multiplet_ranges(self, options: dict[str, Any]) -> list[dict[str, float]]:
        raw_ranges = (
            options.get("multiplet_ranges")
            or options.get("integral_ranges")
            or options.get("integration_ranges")
            or []
        )
        ranges = []
        for item in raw_ranges:
            if isinstance(item, dict):
                start = item.get("start_ppm", item.get("start"))
                end = item.get("end_ppm", item.get("end"))
                if start is None and "range_ppm" in item:
                    range_ppm = item["range_ppm"]
                    start, end = range_ppm[0], range_ppm[1]
            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                start, end = item[0], item[1]
            else:
                continue
            try:
                ranges.append({"start": float(start), "end": float(end)})
            except (TypeError, ValueError):
                continue
        return sorted(ranges, key=lambda r: max(r["start"], r["end"]), reverse=True)

    def _build_multiplet(self, group, spectrum=None, bounds: tuple[float, float] | None = None) -> dict:
        lines = self._select_multiplet_lines(group, _MULTIPLET_LINE_REL_THRESHOLD)
        class_lines = self._select_multiplet_lines(group, _MULTIPLET_CLASS_REL_THRESHOLD)
        class_lines = self._drop_solvent_like_edge_line(class_lines)
        if not class_lines:
            class_lines = lines

        positions = [p.position for p in lines]
        class_positions = [p.position for p in class_lines]
        center = float(np.average(positions, weights=[max(p.intensity, 0.0) for p in lines])) if lines else float(np.mean([p.position for p in group]))
        range_min = min(bounds) if bounds else min(positions)
        range_max = max(bounds) if bounds else max(positions)
        frequency_mhz = 400.0
        if spectrum is not None:
            frequency_mhz = float(spectrum.parameters.get("frequency_mhz", 400.0))
        multiplicity, j_values = self._classify_multiplet(class_lines, range_min, range_max, frequency_mhz)

        return {
            "center_ppm": round(center, 4),
            "range_ppm": (round(range_min, 4), round(range_max, 4)),
            "component_positions": [round(p, 4) for p in sorted(positions, reverse=True)],
            "n_peaks": len(group),
            "n_lines_used": len(class_positions),
            "multiplicity": multiplicity,
            "j_values_hz": j_values,
            "estimated_j_hz": round(float(np.median(j_values)), 2) if j_values else None,
            "intensity_max": round(max(p.intensity for p in group), 6),
        }

    def _select_multiplet_lines(self, group, rel_threshold: float):
        if not group:
            return []
        max_intensity = max(p.intensity for p in group)
        if max_intensity <= 0:
            return sorted(group, key=lambda p: p.position, reverse=True)
        selected = [p for p in group if p.intensity >= max_intensity * rel_threshold]
        if not selected:
            selected = [max(group, key=lambda p: p.intensity)]
        return sorted(selected, key=lambda p: p.position, reverse=True)

    def _drop_solvent_like_edge_line(self, lines):
        if len(lines) < 3:
            return lines
        sorted_by_position = sorted(lines, key=lambda p: p.position, reverse=True)
        sorted_by_intensity = sorted(lines, key=lambda p: p.intensity, reverse=True)
        strongest = sorted_by_intensity[0]
        second = sorted_by_intensity[1]
        is_edge = strongest in (sorted_by_position[0], sorted_by_position[-1])
        if is_edge and second.intensity > 0 and strongest.intensity / second.intensity > 2.4:
            return [p for p in lines if p is not strongest]
        return lines

    def _classify_multiplet(self, lines, start_ppm: float, end_ppm: float, frequency_mhz: float) -> tuple[str, list[float]]:
        n_lines = len(lines)
        if n_lines <= 1:
            return "s", []

        positions = [p.position for p in sorted(lines, key=lambda p: p.position, reverse=True)]
        center = (start_ppm + end_ppm) / 2
        span = abs(end_ppm - start_ppm)
        couplings = self._infer_couplings(positions, frequency_mhz)

        if n_lines == 2:
            return "d", self._take_couplings(couplings, 1)
        if n_lines <= 4:
            if n_lines == 3:
                return "d", self._take_couplings(couplings, 1)
            return "dd", self._take_couplings(couplings, 2)

        in_heteroatom_region = 3.0 <= center <= 4.6
        if in_heteroatom_region and n_lines >= 7:
            small = self._small_coupling(couplings)
            mid = self._mid_coupling(couplings)
            large = self._large_coupling(couplings)
            if n_lines >= 9:
                dq_small = self._small_coupling(couplings, minimum=3.0)
                return "dq", [mid, mid, dq_small, large]
            if n_lines == 8:
                return "tt", [small, small, mid, mid]
            return "qd", [small, mid, mid, mid]

        if n_lines in (6, 7) and span <= 0.14:
            return "ddd", self._take_couplings(couplings, 3)

        return "m", self._take_couplings(couplings, 0)

    def _infer_couplings(self, positions: list[float], frequency_mhz: float) -> list[float]:
        diffs = []
        sorted_positions = sorted(positions, reverse=True)
        for i, a in enumerate(sorted_positions):
            for b in sorted_positions[i + 1:]:
                diff_hz = abs(a - b) * frequency_mhz
                if 1.0 <= diff_hz <= 30.0:
                    diffs.append(diff_hz)
        clusters: list[list[float]] = []
        for diff in sorted(diffs):
            for cluster in clusters:
                center = float(np.mean(cluster))
                tolerance = max(0.75, center * 0.08)
                if abs(center - diff) <= tolerance:
                    cluster.append(diff)
                    break
            else:
                clusters.append([diff])

        scored = []
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            value = float(np.mean(cluster))
            if any(abs(value - other) <= max(0.8, other * 0.12) for other in scored):
                continue
            scored.append(value)
        if not scored and diffs:
            scored.append(float(np.median(diffs)))
        scored.sort()
        return [round(value, 2) for value in scored]

    def _take_couplings(self, couplings: list[float], count: int) -> list[float]:
        if count <= 0 or not couplings:
            return []
        if len(couplings) <= count:
            return couplings[:count]
        if count == 1:
            return [couplings[0]]
        if count == 2:
            return [couplings[0], couplings[-1]]
        selected = [couplings[0]]
        for value in couplings[1:]:
            is_multiple = any(abs(value - base * 2) <= max(0.9, base * 0.18) for base in selected)
            if not is_multiple:
                selected.append(value)
            if len(selected) == count:
                break
        if len(selected) < count:
            for value in couplings:
                if value not in selected:
                    selected.append(value)
                if len(selected) == count:
                    break
        return selected[:count]

    def _small_coupling(self, couplings: list[float], minimum: float = 1.5) -> float:
        if not couplings:
            return 0.0
        for value in couplings:
            if value >= minimum:
                return value
        return couplings[0]

    def _mid_coupling(self, couplings: list[float]) -> float:
        if not couplings:
            return 0.0
        useful = [c for c in couplings if 5.0 <= c <= 8.5]
        if useful:
            return useful[0]
        useful = [c for c in couplings if c >= 3.0]
        if not useful:
            useful = couplings
        return useful[0]

    def _large_coupling(self, couplings: list[float]) -> float:
        if not couplings:
            return 0.0
        for value in couplings:
            if 9.0 <= value <= 13.5:
                return value
        return couplings[-1]

    def _find_nearest_peak(self, peaks, target_ppm, tolerance=0.3):
        for p in peaks:
            if abs(p.position - target_ppm) < tolerance:
                return p
        return None

    def _build_summary(self, peaks, integrals, multiplets, metrics, spectrum) -> str:
        nucleus = spectrum.parameters.get("nucleus", "1H")
        freq = spectrum.parameters.get("frequency_mhz", 400)
        solvent = spectrum.metadata.solvent or "unknown"

        parts = [
            f"{nucleus} NMR at {freq} MHz, {solvent}",
            f"Detected {metrics['n_peaks']} peaks, {metrics['n_multiplets']} multiplets",
        ]

        if integrals:
            ranges = [integ for integ in integrals if integ["center_ppm"] > 6.5]
            if ranges:
                parts.append(f"Aromatic region ({len(ranges)} peaks)")

            ranges = [integ for integ in integrals if 0.5 < integ["center_ppm"] < 4.5]
            if ranges:
                parts.append(f"Aliphatic region ({len(ranges)} peaks)")

        if metrics["reference_corrected"]:
            parts.append(f"Chemical shift calibrated to solvent peak ({metrics['solvent_reference']} ppm)")

        if multiplets:
            parts.append(f"Coupling analysis: {len(multiplets)} multiplet(s) resolved")

        return ". ".join(parts) + "."
