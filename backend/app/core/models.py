from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class Technique(Enum):
    NMR = "NMR"
    UVVIS = "UV-Vis"
    FLUORESCENCE = "Fluorescence"
    IR = "IR"
    XRD = "XRD"
    HPLC = "HPLC"
    ELECTROCHEM = "ElectroChem"


class NMRNucleus(Enum):
    H1 = "1H"
    C13 = "13C"
    F19 = "19F"
    P31 = "31P"
    N15 = "15N"
    SI29 = "29Si"


@dataclass
class SampleInfo:
    name: str = ""
    formula: str = ""
    solvent: str = ""
    concentration: float | None = None
    concentration_unit: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class Peak:
    position: float
    intensity: float
    area: float | None = None
    width: float | None = None
    assignment: str = ""
    multiplicity: str = ""
    coupling_constant: float | None = None


@dataclass
class Spectrum:
    technique: Technique
    x_data: np.ndarray
    y_data: np.ndarray
    x_label: str = ""
    y_label: str = ""
    x_unit: str = ""
    y_unit: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: SampleInfo = field(default_factory=SampleInfo)
    peaks: list[Peak] = field(default_factory=list)
    source_file: str = ""

    def to_dict(self, *, include_internal: bool = False) -> dict[str, Any]:
        params = {}
        for k, v in self.parameters.items():
            if k == "processing_source" and not include_internal:
                params["processing_source_summary"] = {
                    "schema_version": v.get("schema_version"),
                    "pipeline_version": v.get("pipeline_version"),
                    "immutable": bool(v.get("immutable")),
                    "storage": v.get("storage"),
                    "source_kind": v.get("source_kind"),
                    "source_domain": v.get("source_domain"),
                    "source_quality": v.get("source_quality"),
                    "point_count": int(v.get("point_count") or 0),
                    "has_quadrature": bool(v.get("has_quadrature")),
                    "quadrature_real_source": v.get("quadrature_real_source"),
                    "estimated_binary_bytes": int(v.get("estimated_binary_bytes") or 0),
                    "estimated_inline_json_bytes": int(
                        v.get("estimated_inline_json_bytes") or 0
                    ),
                    "checksum_sha256": v.get("checksum_sha256"),
                    "metadata_checksum_sha256": v.get(
                        "metadata_checksum_sha256"
                    ),
                    "default_phase": dict(v.get("default_phase") or {}),
                    "default_baseline": dict(v.get("default_baseline") or {}),
                    "reference_metadata": dict(
                        v.get("reference_metadata") or {}
                    ),
                }
                continue
            if k == "channels":
                params[k] = [
                    {
                        **{sk: sv for sk, sv in ch.items() if sk != "y_data"},
                        "y_data": ch["y_data"].tolist() if isinstance(ch["y_data"], np.ndarray) else ch["y_data"],
                    }
                    for ch in v
                ]
            else:
                params[k] = v

        return {
            "technique": self.technique.value,
            "x_data": self.x_data.tolist(),
            "y_data": self.y_data.tolist(),
            "x_label": self.x_label,
            "y_label": self.y_label,
            "x_unit": self.x_unit,
            "y_unit": self.y_unit,
            "parameters": params,
            "metadata": {
                "name": self.metadata.name,
                "formula": self.metadata.formula,
                "solvent": self.metadata.solvent,
                "concentration": self.metadata.concentration,
                "concentration_unit": self.metadata.concentration_unit,
                "extra": self.metadata.extra,
            },
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
            "source_file": self.source_file,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Spectrum:
        return cls(
            technique=Technique(data["technique"]),
            x_data=np.array(data["x_data"], dtype=np.float64),
            y_data=np.array(data["y_data"], dtype=np.float64),
            x_label=data.get("x_label", ""),
            y_label=data.get("y_label", ""),
            x_unit=data.get("x_unit", ""),
            y_unit=data.get("y_unit", ""),
            parameters=data.get("parameters", {}),
            metadata=SampleInfo(**data.get("metadata", {})),
            peaks=[Peak(**p) for p in data.get("peaks", [])],
            source_file=data.get("source_file", ""),
        )

    @property
    def x_range(self) -> tuple[float, float]:
        if len(self.x_data) == 0:
            return (0.0, 0.0)
        return (float(self.x_data.min()), float(self.x_data.max()))

    @property
    def num_points(self) -> int:
        return len(self.x_data)

    def __repr__(self) -> str:
        return (
            f"Spectrum(technique={self.technique.value}, "
            f"points={self.num_points}, "
            f"x_range={self.x_range}, "
            f"peaks={len(self.peaks)})"
        )
