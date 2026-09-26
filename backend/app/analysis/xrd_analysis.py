from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import optimize, signal

from app.analysis.base import BaseAnalyzer
from app.analysis.models import AnalysisResult
from app.analysis.utils import estimate_noise_region
from app.core.models import Peak, Spectrum

_CU_KA1 = 1.54059
_SCHERRER_K = 0.9

_REFERENCE_PHASES = [
    {
        "name": "Cadmium Zinc Sulfide",
        "formula": "CdZnS",
        "database": "ICDD (PDF2.DAT)",
        "card_number": "synthetic-CdZnS",
        "crystal_system": "cubic",
        "space_group": "F-43m",
        "rir": 4.1,
        "peaks": [
            {"two_theta": 26.6, "hkl": "111", "rel_intensity": 100},
            {"two_theta": 44.0, "hkl": "220", "rel_intensity": 45},
            {"two_theta": 52.1, "hkl": "311", "rel_intensity": 28},
            {"two_theta": 70.9, "hkl": "331", "rel_intensity": 12},
        ],
    },
    {
        "name": "Cadmium Sulfide",
        "formula": "CdS",
        "database": "ICDD (PDF2.DAT)",
        "card_number": "41-1049",
        "crystal_system": "hexagonal",
        "space_group": "P63mc",
        "rir": 3.4,
        "peaks": [
            {"two_theta": 24.8, "hkl": "100", "rel_intensity": 65},
            {"two_theta": 26.5, "hkl": "002", "rel_intensity": 100},
            {"two_theta": 28.2, "hkl": "101", "rel_intensity": 78},
            {"two_theta": 36.6, "hkl": "102", "rel_intensity": 35},
            {"two_theta": 43.7, "hkl": "110", "rel_intensity": 45},
            {"two_theta": 47.8, "hkl": "103", "rel_intensity": 30},
            {"two_theta": 51.9, "hkl": "112", "rel_intensity": 28},
        ],
    },
    {
        "name": "Zinc Sulfide",
        "formula": "ZnS",
        "database": "ICDD (PDF2.DAT)",
        "card_number": "05-0566",
        "crystal_system": "cubic",
        "space_group": "F-43m",
        "rir": 5.0,
        "peaks": [
            {"two_theta": 28.5, "hkl": "111", "rel_intensity": 100},
            {"two_theta": 47.5, "hkl": "220", "rel_intensity": 55},
            {"two_theta": 56.3, "hkl": "311", "rel_intensity": 35},
            {"two_theta": 69.4, "hkl": "400", "rel_intensity": 16},
        ],
    },
    {
        "name": "Silicon",
        "formula": "Si",
        "database": "ICDD (PDF2.DAT)",
        "card_number": "27-1402",
        "crystal_system": "cubic",
        "space_group": "Fd-3m",
        "rir": 4.7,
        "peaks": [
            {"two_theta": 28.44, "hkl": "111", "rel_intensity": 100},
            {"two_theta": 47.30, "hkl": "220", "rel_intensity": 55},
            {"two_theta": 56.12, "hkl": "311", "rel_intensity": 32},
            {"two_theta": 69.13, "hkl": "400", "rel_intensity": 18},
            {"two_theta": 76.37, "hkl": "331", "rel_intensity": 12},
        ],
    },
    {
        "name": "Nickel",
        "formula": "Ni",
        "database": "ICDD (PDF2.DAT)",
        "card_number": "04-0850",
        "crystal_system": "cubic",
        "space_group": "Fm-3m",
        "rir": 8.3,
        "peaks": [
            {"two_theta": 44.5, "hkl": "111", "rel_intensity": 100},
            {"two_theta": 51.8, "hkl": "200", "rel_intensity": 45},
            {"two_theta": 76.4, "hkl": "220", "rel_intensity": 22},
        ],
    },
]


def _theta_to_d(theta_rad: float, wavelength: float = _CU_KA1) -> float:
    sin_theta = np.sin(theta_rad)
    if sin_theta < 1e-10:
        return 0.0
    return wavelength / (2.0 * sin_theta)


def _scherrer_size(fwhm_deg: float, theta_rad: float, wavelength: float = _CU_KA1) -> float:
    beta = np.radians(fwhm_deg)
    cos_theta = np.cos(theta_rad)
    if cos_theta < 1e-10 or beta < 1e-10:
        return 0.0
    return (_SCHERRER_K * wavelength) / (beta * cos_theta)


@dataclass
class XRDAnalysisResult(AnalysisResult):
    d_spacings: list[dict] = field(default_factory=list)
    crystallite_sizes: list[dict] = field(default_factory=list)
    peak_assignments: list[dict] = field(default_factory=list)
    phase_matches: list[dict] = field(default_factory=list)
    lattice_parameters: list[dict] = field(default_factory=list)
    crystallinity: dict = field(default_factory=dict)
    williamson_hall: dict = field(default_factory=dict)
    size_distribution: dict = field(default_factory=dict)
    quantitative_analysis: list[dict] = field(default_factory=list)
    crystal_structure: list[dict] = field(default_factory=list)
    rietveld_refinement: dict = field(default_factory=dict)
    wavelength_used: float = _CU_KA1

    def to_dict(self):
        d = super().to_dict()
        d["d_spacings"] = self.d_spacings
        d["crystallite_sizes"] = self.crystallite_sizes
        d["peak_assignments"] = self.peak_assignments
        d["phase_matches"] = self.phase_matches
        d["lattice_parameters"] = self.lattice_parameters
        d["crystallinity"] = self.crystallinity
        d["williamson_hall"] = self.williamson_hall
        d["size_distribution"] = self.size_distribution
        d["quantitative_analysis"] = self.quantitative_analysis
        d["crystal_structure"] = self.crystal_structure
        d["rietveld_refinement"] = self.rietveld_refinement
        d["wavelength_used"] = self.wavelength_used
        return d


class XRDAnalyzer(BaseAnalyzer):
    technique = "xrd"

    def analyze(self, spectrum: Spectrum, options: dict[str, Any] | None = None) -> XRDAnalysisResult:
        options = options or {}
        x = spectrum.x_data
        y = spectrum.y_data
        wavelength = spectrum.parameters.get("wavelength1_a", _CU_KA1)

        background = self._estimate_background(y)
        y_bg = np.maximum(y - background, 0)

        noise = estimate_noise_region(y_bg, fraction=0.05)
        noise_factor = float(options.get("noise_factor", 3))
        prominence_factor = float(options.get("prominence_factor", 5))
        relative_prominence = float(options.get("relative_prominence", 0.02))
        min_peak_distance = int(options.get("min_peak_distance", 10))
        prominence = max(noise * prominence_factor, relative_prominence * y_bg.max())

        peak_indices, peak_props = signal.find_peaks(
            y_bg,
            height=noise * noise_factor,
            prominence=prominence,
            distance=min_peak_distance,
            width=(1, 200),
        )

        peaks = []
        peak_indices_by_position = {}
        for i in range(len(peak_indices)):
            idx = int(peak_indices[i])
            width_pts = peak_props.get("widths", [0])[i] if "widths" in peak_props else 0
            fwhm_deg = float(width_pts * (x[1] - x[0])) if len(x) > 1 else 0.0
            height = peak_props.get("peak_heights", [0])[i] if "peak_heights" in peak_props else float(y_bg[idx])
            peaks.append(Peak(
                position=float(x[idx]),
                intensity=float(height),
                width=round(fwhm_deg, 4) if fwhm_deg > 0 else None,
            ))
            peak_indices_by_position[round(float(x[idx]), 6)] = idx

        peaks.sort(key=lambda p: p.intensity, reverse=True)

        d_spacings = []
        crystallite_sizes = []
        for p in peaks:
            theta_rad = np.radians(p.position / 2.0)
            d = _theta_to_d(theta_rad, wavelength)
            d_spacings.append({
                "two_theta": round(p.position, 3),
                "d_angstrom": round(d, 4),
                "intensity": round(p.intensity, 1),
            })

            if p.width and p.width > 0 and p.width < 5.0:
                size = _scherrer_size(p.width, theta_rad, wavelength)
                if size > 0:
                    crystallite_sizes.append({
                        "two_theta": round(p.position, 3),
                        "fwhm_deg": round(p.width, 4),
                        "size_a": round(size, 1),
                        "size_nm": round(size / 10.0, 2),
                    })

        phase_library = self._phase_library(options)
        phase_matches, peak_assignments = self._match_phases(peaks, wavelength, phase_library)
        lattice_parameters = self._estimate_lattice_parameters(peak_assignments, wavelength, phase_library)
        crystallinity = self._estimate_crystallinity(x, y, background, y_bg, peaks)
        williamson_hall = self._williamson_hall(peaks, wavelength)
        size_distribution = self._size_distribution(crystallite_sizes)
        quantitative_analysis = self._quantitative_rir(phase_matches, peak_assignments, phase_library)
        crystal_structure = self._crystal_structure(phase_matches, lattice_parameters)
        rietveld_refinement = self._rietveld_refinement(x, y, y_bg, phase_matches, phase_library, options)

        n_peaks = len(peaks)
        metrics = {
            "n_peaks": n_peaks,
            "wavelength_a": wavelength,
            "two_theta_range": (round(float(x[0]), 1), round(float(x[-1]), 1)),
            "max_intensity": round(float(y.max()), 1),
            "crystallinity_percent": crystallinity.get("crystallinity_percent"),
            "dominant_phase": phase_matches[0]["phase_name"] if phase_matches else None,
            "matched_phases": len([m for m in phase_matches if m["matched_peaks"] > 0]),
            "rietveld_rwp": rietveld_refinement.get("rwp"),
            "analysis_options": {
                "noise_factor": noise_factor,
                "prominence_factor": prominence_factor,
                "relative_prominence": relative_prominence,
                "min_peak_distance": min_peak_distance,
                "custom_phases": len(options.get("custom_phases") or []),
                "rietveld_enabled": bool(options.get("rietveld_enabled", True)),
            },
        }

        main_peaks = [p for p in d_spacings if peaks and p["intensity"] > 0.1 * peaks[0].intensity]
        top_d = ", ".join(
            f"{dp['two_theta']:.1f}° (d={dp['d_angstrom']:.3f}Å)"
            for dp in main_peaks[:6]
        )

        summary = f"XRD pattern: {n_peaks} peaks detected. Major peaks: {top_d}."
        if phase_matches:
            best = phase_matches[0]
            summary += f" Best phase match: {best['phase_name']} ({best['match_score']:.0f}% confidence)."
        if crystallinity:
            summary += f" Crystallinity: {crystallinity['crystallinity_percent']:.1f}%."
        if williamson_hall:
            summary += (
                f" Williamson-Hall size: {williamson_hall['crystallite_size_a']:.1f} Å, "
                f"strain: {williamson_hall['microstrain']:.4g}."
            )
        if crystallite_sizes:
            avg = np.mean([cs["size_a"] for cs in crystallite_sizes])
            summary += f" Avg. crystallite size: {avg:.1f} Å."
        if rietveld_refinement:
            summary += f" Rietveld-like fit Rwp: {rietveld_refinement['rwp']:.2f}%."

        return XRDAnalysisResult(
            technique=spectrum.technique,
            peaks=peaks,
            metrics=metrics,
            summary=summary,
            d_spacings=d_spacings,
            crystallite_sizes=crystallite_sizes,
            peak_assignments=peak_assignments,
            phase_matches=phase_matches,
            lattice_parameters=lattice_parameters,
            crystallinity=crystallinity,
            williamson_hall=williamson_hall,
            size_distribution=size_distribution,
            quantitative_analysis=quantitative_analysis,
            crystal_structure=crystal_structure,
            rietveld_refinement=rietveld_refinement,
            wavelength_used=wavelength,
        )

    def _subtract_background(self, y: np.ndarray) -> np.ndarray:
        bg = self._estimate_background(y)
        return np.maximum(y - bg, 0)

    def _estimate_background(self, y: np.ndarray) -> np.ndarray:
        window = max(len(y) // 20, 10)
        bg = np.zeros_like(y)
        for i in range(len(y)):
            lo = max(0, i - window)
            hi = min(len(y), i + window + 1)
            bg[i] = np.min(y[lo:hi])
        return np.convolve(bg, np.ones(2 * window + 1) / (2 * window + 1), mode="same")

    def _phase_library(self, options: dict[str, Any]) -> list[dict]:
        phases = [dict(phase) for phase in _REFERENCE_PHASES]
        for raw in options.get("custom_phases") or []:
            if not isinstance(raw, dict):
                continue
            peaks = raw.get("peaks") or []
            parsed_peaks = []
            for peak in peaks:
                if not isinstance(peak, dict):
                    continue
                try:
                    parsed_peaks.append({
                        "two_theta": float(peak["two_theta"]),
                        "hkl": str(peak.get("hkl", "")),
                        "rel_intensity": float(peak.get("rel_intensity", peak.get("intensity", 100))),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
            if not parsed_peaks:
                continue
            phases.append({
                "name": str(raw.get("name") or "Custom phase"),
                "formula": str(raw.get("formula") or ""),
                "database": str(raw.get("database") or "User"),
                "card_number": str(raw.get("card_number") or "custom"),
                "crystal_system": str(raw.get("crystal_system") or "unknown"),
                "space_group": str(raw.get("space_group") or ""),
                "rir": float(raw.get("rir", 1.0) or 1.0),
                "peaks": parsed_peaks,
            })
        return phases

    def _match_phases(self, peaks: list[Peak], wavelength: float, phase_library: list[dict]) -> tuple[list[dict], list[dict]]:
        if not peaks:
            return [], []
        max_intensity = max(p.intensity for p in peaks) or 1.0
        observed = peaks[:40]
        matches = []
        assignments = []

        for phase in phase_library:
            matched_refs = []
            score = 0.0
            possible = sum(ref["rel_intensity"] for ref in phase["peaks"])
            for ref in phase["peaks"]:
                nearest = min(observed, key=lambda p: abs(p.position - ref["two_theta"]))
                shift = nearest.position - ref["two_theta"]
                tolerance = 0.45 if ref["two_theta"] < 35 else 0.6
                if abs(shift) <= tolerance:
                    weight = ref["rel_intensity"]
                    intensity_weight = min(nearest.intensity / max_intensity, 1.0)
                    score += weight * (1.0 - abs(shift) / tolerance) * (0.65 + 0.35 * intensity_weight)
                    theta_rad = np.radians(nearest.position / 2.0)
                    matched_refs.append({
                        "phase_name": phase["name"],
                        "formula": phase["formula"],
                        "hkl": ref["hkl"],
                        "observed_two_theta": round(nearest.position, 4),
                        "reference_two_theta": ref["two_theta"],
                        "peak_shift": round(shift, 4),
                        "fwhm_deg": nearest.width,
                        "d_angstrom": round(float(_theta_to_d(theta_rad, wavelength)), 4),
                        "relative_intensity": round(100 * nearest.intensity / max_intensity, 1),
                    })
            match_score = 100.0 * score / possible if possible else 0.0
            matches.append({
                "phase_name": phase["name"],
                "formula": phase["formula"],
                "database": phase["database"],
                "card_number": phase["card_number"],
                "crystal_system": phase["crystal_system"],
                "space_group": phase["space_group"],
                "matched_peaks": len(matched_refs),
                "reference_peaks": len(phase["peaks"]),
                "match_score": round(match_score, 1),
            })
            assignments.extend(matched_refs)

        matches.sort(key=lambda item: (item["match_score"], item["matched_peaks"]), reverse=True)
        best_names = {m["phase_name"] for m in matches[:3] if m["matched_peaks"] > 0}
        assignments = [a for a in assignments if a["phase_name"] in best_names]
        assignments.sort(key=lambda item: item["observed_two_theta"])
        return matches, assignments

    def _estimate_lattice_parameters(self, assignments: list[dict], wavelength: float, phase_library: list[dict]) -> list[dict]:
        by_phase: dict[str, list[dict]] = {}
        for assignment in assignments:
            by_phase.setdefault(assignment["phase_name"], []).append(assignment)

        results = []
        for phase in phase_library:
            rows = by_phase.get(phase["name"], [])
            if len(rows) < 2:
                continue
            if phase["crystal_system"] == "cubic":
                a_values = []
                for row in rows:
                    hkl = self._parse_hkl(row["hkl"])
                    if hkl is None:
                        continue
                    h, k, ell = hkl
                    denom = h * h + k * k + ell * ell
                    if denom == 0:
                        continue
                    a_values.append(row["d_angstrom"] * np.sqrt(denom))
                if a_values:
                    results.append({
                        "phase_name": phase["name"],
                        "crystal_system": "cubic",
                        "a_angstrom": round(float(np.mean(a_values)), 4),
                        "a_std": round(float(np.std(a_values)), 5),
                        "indexed_peaks": len(a_values),
                        "angular_correction": "internal peak matching",
                    })
            elif phase["crystal_system"] == "hexagonal" and len(rows) >= 3:
                matrix = []
                target = []
                for row in rows:
                    hkl = self._parse_hkl(row["hkl"])
                    if hkl is None:
                        continue
                    h, k, ell = hkl
                    d = row["d_angstrom"]
                    matrix.append([(4.0 / 3.0) * (h * h + h * k + k * k), ell * ell])
                    target.append(1.0 / (d * d))
                if len(matrix) >= 2:
                    coeffs, *_ = np.linalg.lstsq(np.array(matrix), np.array(target), rcond=None)
                    inv_a2, inv_c2 = coeffs
                    if inv_a2 > 0 and inv_c2 > 0:
                        a = 1 / np.sqrt(inv_a2)
                        c = 1 / np.sqrt(inv_c2)
                        results.append({
                            "phase_name": phase["name"],
                            "crystal_system": "hexagonal",
                            "a_angstrom": round(float(a), 4),
                            "c_angstrom": round(float(c), 4),
                            "c_over_a": round(float(c / a), 4),
                            "indexed_peaks": len(matrix),
                            "angular_correction": "internal peak matching",
                        })
        return results

    def _estimate_crystallinity(
        self,
        x: np.ndarray,
        y: np.ndarray,
        background: np.ndarray,
        y_bg: np.ndarray,
        peaks: list[Peak],
    ) -> dict:
        total_area = abs(float(np.trapezoid(np.maximum(y, 0), x))) if len(x) > 1 else 0.0
        amorphous_area = abs(float(np.trapezoid(np.maximum(background, 0), x))) if len(x) > 1 else 0.0
        crystalline_area = abs(float(np.trapezoid(y_bg, x))) if len(x) > 1 else 0.0
        denom = crystalline_area + amorphous_area
        crystallinity = 100.0 * crystalline_area / denom if denom > 0 else 0.0
        return {
            "data_set_name": "XRD pattern",
            "crystallinity_percent": round(crystallinity, 1),
            "crystalline_area": round(crystalline_area, 2),
            "amorphous_area": round(amorphous_area, 2),
            "total_area": round(total_area, 2),
            "method": "background-separated integrated area",
            "peak_count_used": len(peaks),
        }

    def _williamson_hall(self, peaks: list[Peak], wavelength: float) -> dict:
        rows = []
        for p in peaks:
            if not p.width or p.width <= 0 or p.width >= 5:
                continue
            theta = np.radians(p.position / 2.0)
            beta = np.radians(p.width)
            rows.append({
                "two_theta": p.position,
                "x": 4.0 * np.sin(theta),
                "y": beta * np.cos(theta),
            })
        if len(rows) < 2:
            return {}
        xs = np.array([r["x"] for r in rows])
        ys = np.array([r["y"] for r in rows])
        slope, intercept = np.polyfit(xs, ys, 1)
        predicted = slope * xs + intercept
        ss_res = float(np.sum((ys - predicted) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        size = (_SCHERRER_K * wavelength) / intercept if intercept > 1e-10 else 0.0
        return {
            "method": "Williamson-Hall UDM",
            "crystallite_size_a": round(float(size), 1) if size > 0 else 0.0,
            "crystallite_size_nm": round(float(size / 10.0), 2) if size > 0 else 0.0,
            "microstrain": round(float(max(slope, 0.0)), 6),
            "intercept": round(float(intercept), 8),
            "r_squared": round(float(r_squared), 4),
            "points_used": len(rows),
        }

    def _size_distribution(self, crystallite_sizes: list[dict]) -> dict:
        values = [cs["size_a"] for cs in crystallite_sizes if cs.get("size_a", 0) > 0]
        if not values:
            return {}
        counts, edges = np.histogram(values, bins=min(6, max(2, len(values))))
        bins = []
        for i, count in enumerate(counts):
            bins.append({
                "min_a": round(float(edges[i]), 1),
                "max_a": round(float(edges[i + 1]), 1),
                "count": int(count),
            })
        return {
            "mean_a": round(float(np.mean(values)), 1),
            "median_a": round(float(np.median(values)), 1),
            "std_a": round(float(np.std(values)), 1),
            "min_a": round(float(np.min(values)), 1),
            "max_a": round(float(np.max(values)), 1),
            "bins": bins,
        }

    def _quantitative_rir(self, phase_matches: list[dict], assignments: list[dict], phase_library: list[dict]) -> list[dict]:
        if not phase_matches:
            return []
        intensity_by_phase: dict[str, float] = {}
        for row in assignments:
            intensity_by_phase[row["phase_name"]] = intensity_by_phase.get(row["phase_name"], 0.0) + row["relative_intensity"]
        weighted = []
        for phase in phase_matches:
            if phase["matched_peaks"] == 0:
                continue
            ref = next((p for p in phase_library if p["name"] == phase["phase_name"]), None)
            rir = float(ref.get("rir", 1.0)) if ref else 1.0
            value = intensity_by_phase.get(phase["phase_name"], 0.0) / max(rir, 1e-6)
            if value <= 0:
                continue
            weighted.append((phase, value, rir))
        total = sum(value for _, value, _ in weighted)
        if total <= 0:
            return []
        result = [
            {
                "phase_name": phase["phase_name"],
                "formula": phase["formula"],
                "rir": rir,
                "weight_percent": round(100.0 * value / total, 1),
                "method": "semi-quantitative RIR",
            }
            for phase, value, rir in weighted
        ]
        return sorted(result, key=lambda item: item["weight_percent"], reverse=True)

    def _crystal_structure(self, phase_matches: list[dict], lattice_parameters: list[dict]) -> list[dict]:
        lattice_by_phase = {row["phase_name"]: row for row in lattice_parameters}
        structures = []
        for phase in phase_matches:
            if phase["matched_peaks"] == 0:
                continue
            lattice = lattice_by_phase.get(phase["phase_name"], {})
            structures.append({
                "phase_name": phase["phase_name"],
                "formula": phase["formula"],
                "crystal_system": phase["crystal_system"],
                "space_group": phase["space_group"],
                "database": phase["database"],
                "card_number": phase["card_number"],
                "indexed_peaks": lattice.get("indexed_peaks", phase["matched_peaks"]),
                "lattice": lattice,
                "refinement": {
                    "measurement_range": "10.0000-80.0000 deg",
                    "refinement_range": "10.0000-80.0000 deg",
                    "refined_parameters": 0,
                    "status": "reference-pattern fit",
                },
            })
        return structures

    def _rietveld_refinement(
        self,
        x: np.ndarray,
        y_raw: np.ndarray,
        y_bg: np.ndarray,
        phase_matches: list[dict],
        phase_library: list[dict],
        options: dict[str, Any],
    ) -> dict:
        if not bool(options.get("rietveld_enabled", True)):
            return {}
        active = [m for m in phase_matches[:3] if m.get("matched_peaks", 0) > 0 and m.get("match_score", 0) > 5]
        if not active or len(x) < 20:
            return {}

        library_by_name = {phase["name"]: phase for phase in phase_library}
        background = self._estimate_background(y_raw)
        observed = np.maximum(y_raw - background, 0)
        max_obs = float(np.max(observed)) if len(observed) else 0.0
        if max_obs <= 0:
            return {}

        ref_sets = []
        for match in active:
            phase = library_by_name.get(match["phase_name"])
            if not phase:
                continue
            ref_sets.append({
                "phase": phase,
                "match": match,
                "scale0": max(float(match.get("match_score", 1)) / 100.0, 0.05),
            })
        if not ref_sets:
            return {}

        sigma0 = float(options.get("rietveld_sigma_deg", 0.18) or 0.18)
        bg0 = float(np.percentile(y_raw, 5))
        p0 = [item["scale0"] * max_obs for item in ref_sets] + [sigma0, bg0]
        lower = [0.0 for _ in ref_sets] + [0.03, 0.0]
        upper = [max_obs * 2.5 for _ in ref_sets] + [1.2, float(np.max(y_raw) if len(y_raw) else max_obs)]

        def model(params: np.ndarray) -> np.ndarray:
            sigma = max(float(params[-2]), 0.03)
            bg_const = max(float(params[-1]), 0.0)
            calc = np.full_like(x, bg_const, dtype=float)
            for scale, item in zip(params[:-2], ref_sets):
                phase = item["phase"]
                for ref in phase["peaks"]:
                    center = float(ref["two_theta"])
                    amp = float(scale) * float(ref.get("rel_intensity", 100)) / 100.0
                    calc += amp * np.exp(-0.5 * ((x - center) / sigma) ** 2)
            return calc

        def residual(params: np.ndarray) -> np.ndarray:
            weight = 1.0 / np.sqrt(np.maximum(y_raw, 1.0))
            return (model(params) - y_raw) * weight

        try:
            fit = optimize.least_squares(residual, p0, bounds=(lower, upper), max_nfev=400)
            params = fit.x
        except Exception:
            params = np.array(p0, dtype=float)
        calculated = model(params)
        diff = y_raw - calculated
        denom = float(np.sum(np.maximum(y_raw, 1.0) ** 2))
        rwp = float(np.sqrt(np.sum(diff * diff) / denom) * 100.0) if denom > 0 else 0.0
        rb = float(np.sum(np.abs(diff)) / max(float(np.sum(np.abs(y_raw))), 1e-9) * 100.0)
        scale_total = float(np.sum(params[:-2]))
        phase_rows = []
        for scale, item in zip(params[:-2], ref_sets):
            weight_percent = float(scale / scale_total * 100.0) if scale_total > 0 else 0.0
            phase_rows.append({
                "phase_name": item["phase"]["name"],
                "formula": item["phase"]["formula"],
                "scale": round(float(scale), 4),
                "weight_percent": round(weight_percent, 2),
                "matched_peaks": item["match"]["matched_peaks"],
            })
        return {
            "method": "reference-pattern constrained least-squares fit",
            "status": "converged" if "fit" in locals() and fit.success else "approximate",
            "rwp": round(rwp, 3),
            "rb": round(rb, 3),
            "sigma_deg": round(float(params[-2]), 4),
            "background_constant": round(float(params[-1]), 3),
            "n_observations": int(len(x)),
            "n_parameters": int(len(params)),
            "phases": sorted(phase_rows, key=lambda row: row["weight_percent"], reverse=True),
        }

    def _parse_hkl(self, value: str) -> tuple[int, int, int] | None:
        if len(value) != 3 or not value.isdigit():
            return None
        return int(value[0]), int(value[1]), int(value[2])
