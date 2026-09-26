"""CSP5 public API.

This module keeps top-level imports lazy so entrypoints like ``csp5``
avoid importing optional matching dependencies unless needed.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Dict, Tuple


if TYPE_CHECKING:
    from .api import (
        MoleculeRecord,
        PredictionResult,
        predict_molecule_file,
        predict_mols,
        predict_sdf,
        predict_smiles,
        predict_structures,
    )
    from .drawing import draw_prediction
    from .matching import MatchingResult, RankedAssignment, match_shifts


__all__ = [
    "PredictionResult",
    "MoleculeRecord",
    "draw_prediction",
    "MatchingResult",
    "RankedAssignment",
    "predict_molecule_file",
    "predict_smiles",
    "predict_mols",
    "predict_structures",
    "predict_sdf",
    "match_shifts",
]

_EXPORT_MAP: Dict[str, Tuple[str, str]] = {
    "PredictionResult": ("csp5.api", "PredictionResult"),
    "MoleculeRecord": ("csp5.api", "MoleculeRecord"),
    "draw_prediction": ("csp5.drawing", "draw_prediction"),
    "predict_molecule_file": ("csp5.api", "predict_molecule_file"),
    "predict_smiles": ("csp5.api", "predict_smiles"),
    "predict_mols": ("csp5.api", "predict_mols"),
    "predict_structures": ("csp5.api", "predict_structures"),
    "predict_sdf": ("csp5.api", "predict_sdf"),
    "MatchingResult": ("csp5.matching", "MatchingResult"),
    "RankedAssignment": ("csp5.matching", "RankedAssignment"),
    "match_shifts": ("csp5.matching", "match_shifts"),
}


def __getattr__(name: str):
    if name not in _EXPORT_MAP:
        raise AttributeError(f"module 'csp5' has no attribute {name!r}")
    module_name, attr_name = _EXPORT_MAP[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
