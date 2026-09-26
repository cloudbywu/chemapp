from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from app.analysis.base import BaseAnalyzer
from app.analysis.models import UVVisAnalysisResult
from app.analysis.utils import find_peaks, linear_fit
from app.core.models import Peak, Spectrum


class UVVisAnalyzer(BaseAnalyzer):
    technique = "uvvis"

    def analyze(self, spectrum: Spectrum, options: dict[str, Any] | None = None) -> UVVisAnalysisResult:
        options = options or {}

        if spectrum.parameters.get("type") == "calibration":
            return self._analyze_calibration(spectrum)
        else:
            return self._analyze_sample(spectrum, options)

    def _analyze_sample(self, spectrum: Spectrum, options: dict[str, Any]) -> UVVisAnalysisResult:
        x = spectrum.x_data
        y = spectrum.y_data

        baseline_correct = bool(options.get("baseline_correct", False))
        y_for_detection = self._correct_baseline(x, y) if baseline_correct else y
        peaks = self._detect_absorption_peaks(x, y_for_detection, options)

        lambda_max = [round(p.position, 1) for p in peaks if p.position > 250]

        metrics: dict = {
            "n_peaks": len(peaks),
            "lambda_max": lambda_max,
            "absorbance_range": (round(float(y.min()), 4), round(float(y.max()), 4)),
            "baseline_corrected": baseline_correct,
            "analysis_options": {
                "baseline_correct": baseline_correct,
                "height_fraction": float(options.get("height_fraction", 0.01)),
                "prominence_fraction": float(options.get("prominence_fraction", 0.02)),
                "min_peak_distance": int(options.get("min_peak_distance", 10)),
            },
        }

        concentration = None
        conc_unit = ""
        calibration = None

        summary = self._build_sample_summary(spectrum, lambda_max, metrics)

        return UVVisAnalysisResult(
            technique=spectrum.technique,
            peaks=peaks,
            metrics=metrics,
            summary=summary,
            lambda_max=lambda_max,
            calibration=calibration,
            sample_concentration=concentration,
            concentration_unit=conc_unit,
            baseline_corrected=baseline_correct,
        )

    def _analyze_calibration(self, spectrum: Spectrum) -> UVVisAnalysisResult:
        x = spectrum.x_data
        y = spectrum.y_data

        if len(x) < 2:
            return UVVisAnalysisResult(
                technique=spectrum.technique,
                summary="Insufficient calibration points.",
            )

        fit = linear_fit(x, y)

        calibration = {
            "slope": round(fit["slope"], 6),
            "intercept": round(fit["intercept"], 6),
            "r_squared": round(fit["r_squared"], 4),
            "n_points": len(x),
        }

        metrics = {
            "calibration_curve": calibration,
        }

        summary = (
            f"Calibration curve: Abs = {calibration['slope']:.4f} * C + {calibration['intercept']:.4f}, "
            f"R² = {calibration['r_squared']:.4f}, {calibration['n_points']} points"
        )

        return UVVisAnalysisResult(
            technique=spectrum.technique,
            metrics=metrics,
            summary=summary,
            calibration=calibration,
        )

    def compute_concentration(self, calibration_result: UVVisAnalysisResult, absorbance: float) -> float | None:
        if calibration_result.calibration is None:
            return None
        slope = calibration_result.calibration["slope"]
        intercept = calibration_result.calibration["intercept"]
        if slope == 0:
            return None
        return (absorbance - intercept) / slope

    def _detect_absorption_peaks(self, x: np.ndarray, y: np.ndarray, options: dict[str, Any]) -> list[Peak]:
        if len(x) < 3 or len(y) < 3:
            return []

        height_fraction = float(options.get("height_fraction", 0.01))
        prominence_fraction = float(options.get("prominence_fraction", 0.02))
        distance = int(options.get("min_peak_distance", 10))
        peak_indices, props = find_peaks(
            y,
            height=height_fraction * y.max(),
            prominence=prominence_fraction * (y.max() - y.min()),
            distance=distance,
        )
        peaks = []
        for i in range(len(peak_indices)):
            idx = int(peak_indices[i])
            peaks.append(Peak(
                position=float(x[idx]),
                intensity=float(y[idx]),
                width=float(props.get("widths", [0])[i] * (x[1] - x[0])) if "widths" in props else None,
            ))
        return peaks

    def _correct_baseline(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = max(len(x) // 20, 5)
        x_start = x[:n]
        y_start = y[:n]
        x_end = x[-n:]
        y_end = y[-n:]

        slope = (y_end.mean() - y_start.mean()) / (x_end.mean() - x_start.mean())
        intercept = y_start.mean() - slope * x_start.mean()
        baseline = slope * x + intercept
        return y - baseline

    def _build_sample_summary(self, spectrum: Spectrum, lambda_max: list[float], metrics: dict) -> str:
        parts = ["UV-Vis spectrum"]
        if lambda_max:
            parts.append(f"λmax: {', '.join(str(w) for w in lambda_max[:5])} nm")
        if spectrum.source_file:
            parts.append(f"({Path(spectrum.source_file).name})")
        return ". ".join(parts) + "."

    def _build_calibration_summary(self, calibration: dict) -> str:
        return (
            f"Calibration: A = {calibration['slope']:.4f}·C + {calibration['intercept']:.4f}, "
            f"R² = {calibration['r_squared']:.4f}"
        )
