from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.models import Peak, Technique


@dataclass
class AnalysisResult:
    technique: Technique
    peaks: list[Peak] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "technique": self.technique.value,
            "peaks": [
                {
                    "position": p.position,
                    "intensity": p.intensity,
                    "area": p.area,
                    "width": p.width,
                    "assignment": p.assignment,
                    "multiplicity": p.multiplicity,
                    "coupling_constant": p.coupling_constant,
                }
                for p in self.peaks
            ],
            "metrics": self.metrics,
            "summary": self.summary,
        }


@dataclass
class NMRAnalysisResult(AnalysisResult):
    integrals: list[dict[str, Any]] = field(default_factory=list)
    multiplets: list[dict[str, Any]] = field(default_factory=list)
    noise_level: float = 0.0
    total_integral: float = 0.0
    solvent_shift: float | None = None
    reference_corrected: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["integrals"] = self.integrals
        d["multiplets"] = self.multiplets
        d["noise_level"] = self.noise_level
        d["total_integral"] = self.total_integral
        d["solvent_shift"] = self.solvent_shift
        d["reference_corrected"] = self.reference_corrected
        return d


@dataclass
class UVVisAnalysisResult(AnalysisResult):
    lambda_max: list[float] = field(default_factory=list)
    calibration: dict[str, Any] | None = None
    sample_concentration: float | None = None
    concentration_unit: str = ""
    baseline_corrected: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["lambda_max"] = self.lambda_max
        d["calibration"] = self.calibration
        d["sample_concentration"] = self.sample_concentration
        d["concentration_unit"] = self.concentration_unit
        d["baseline_corrected"] = self.baseline_corrected
        return d


@dataclass
class FluorescenceAnalysisResult(AnalysisResult):
    ex_peak: float | None = None
    em_peak: float | None = None
    stokes_shift_nm: float | None = None
    stokes_shift_cm1: float | None = None
    quantum_yield_ref: dict[str, Any] | None = None
    normalized: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d["ex_peak"] = self.ex_peak
        d["em_peak"] = self.em_peak
        d["stokes_shift_nm"] = self.stokes_shift_nm
        d["stokes_shift_cm1"] = self.stokes_shift_cm1
        d["quantum_yield_ref"] = self.quantum_yield_ref
        d["normalized"] = self.normalized
        return d
