from __future__ import annotations

from typing import Any

import numpy as np

from app.analysis.base import BaseAnalyzer
from app.analysis.models import FluorescenceAnalysisResult
from app.analysis.utils import local_maxima, normalize_minmax
from app.core.models import Peak, Spectrum


def _nm_to_cm1(nm: float) -> float:
    return 1e7 / nm if nm > 0 else 0.0


class FluorescenceAnalyzer(BaseAnalyzer):
    technique = "fluorescence"

    def analyze(self, spectrum: Spectrum, options: dict[str, Any] | None = None) -> FluorescenceAnalysisResult:
        options = options or {}
        x = spectrum.x_data
        y = spectrum.y_data
        y_norm = normalize_minmax(y)

        sub_type = spectrum.parameters.get("sub_type", "")
        min_peak_distance = int(options.get("min_peak_distance", 10))
        peaks_raw = local_maxima(x, y_norm, n=min_peak_distance)

        peaks = [Peak(position=p["position"], intensity=p["intensity"]) for p in peaks_raw[:10]]

        ex_peak = None
        em_peak = None
        ex_wl = spectrum.parameters.get("excitation_wavelength_nm")
        em_wl = spectrum.parameters.get("emission_wavelength_nm")

        if sub_type == "excitation":
            ex_peak = self._find_highest_peak_in_range(x, y_norm, 200, 600)
        elif sub_type == "emission":
            exc_wl = spectrum.parameters.get("excitation_wavelength_nm")
            search_start = max(350, (exc_wl or 0) + 30)
            em_peak = self._find_highest_peak_in_range(x, y_norm, search_start, 850)

        stokes_nm = None
        stokes_cm1 = None
        if ex_peak is not None and em_peak is not None:
            stokes_nm = em_peak - ex_peak
            stokes_cm1 = _nm_to_cm1(ex_peak) - _nm_to_cm1(em_peak)
        elif ex_wl and em_peak is not None:
            stokes_nm = em_peak - ex_wl
            stokes_cm1 = _nm_to_cm1(ex_wl) - _nm_to_cm1(em_peak)
        elif em_wl and ex_peak is not None:
            stokes_nm = em_wl - ex_peak
            stokes_cm1 = _nm_to_cm1(ex_peak) - _nm_to_cm1(em_wl)

        metrics = {
            "n_peaks": len(peaks),
            "excitation_peak_nm": round(ex_peak, 1) if ex_peak else None,
            "emission_peak_nm": round(em_peak, 1) if em_peak else None,
            "stokes_shift_nm": round(stokes_nm, 1) if stokes_nm else None,
            "stokes_shift_cm1": round(stokes_cm1, 0) if stokes_cm1 else None,
            "sub_type": sub_type,
            "normalized": True,
            "analysis_options": {"min_peak_distance": min_peak_distance},
        }

        summary = self._build_summary(spectrum, metrics)

        return FluorescenceAnalysisResult(
            technique=spectrum.technique,
            peaks=peaks,
            metrics=metrics,
            summary=summary,
            ex_peak=ex_peak,
            em_peak=em_peak,
            stokes_shift_nm=round(stokes_nm, 1) if stokes_nm else None,
            stokes_shift_cm1=round(stokes_cm1, 0) if stokes_cm1 else None,
            normalized=True,
        )

    def compute_stokes_shift(self, ex: FluorescenceAnalysisResult, em: FluorescenceAnalysisResult) -> dict:
        ex_wl = ex.ex_peak or em.metrics.get("excitation_wavelength_nm")
        em_wl = em.em_peak or ex.metrics.get("emission_wavelength_nm")

        if ex_wl is None or em_wl is None:
            return {"stokes_shift_nm": None, "stokes_shift_cm1": None}

        stokes_nm = em_wl - ex_wl
        stokes_cm1 = _nm_to_cm1(ex_wl) - _nm_to_cm1(em_wl)
        return {
            "stokes_shift_nm": round(stokes_nm, 1),
            "stokes_shift_cm1": round(stokes_cm1, 0),
            "excitation_peak_nm": ex_wl,
            "emission_peak_nm": em_wl,
        }

    def _find_highest_peak_in_range(self, x, y, x_min, x_max):
        mask = (x >= x_min) & (x <= x_max)
        if mask.sum() == 0:
            return None
        idx = np.argmax(y[mask])
        return float(x[mask][idx])

    def _build_summary(self, spectrum, metrics) -> str:
        parts = [f"Fluorescence {metrics['sub_type']} spectrum"]
        title = spectrum.metadata.name or spectrum.parameters.get("title", "")
        if title:
            parts.append(f"({title})")

        if metrics["excitation_peak_nm"]:
            parts.append(f"Ex peak: {metrics['excitation_peak_nm']} nm")
        if metrics["emission_peak_nm"]:
            parts.append(f"Em peak: {metrics['emission_peak_nm']} nm")
        if metrics["stokes_shift_nm"]:
            parts.append(f"Stokes shift: {metrics['stokes_shift_nm']} nm ({metrics['stokes_shift_cm1']} cm⁻¹)")

        return ", ".join(parts) + "."
