"""Shared non-ML analysis helpers used by API routes."""

from __future__ import annotations

from app.analysis import AnalyzerRegistry
from app.analysis.models import AnalysisResult
from app.core.models import Spectrum


def technique_key(value: str) -> str | None:
    """Map a spectrum technique value to its analyzer registry key."""

    key = value.lower().replace("-", "")
    return {
        "nmr": "nmr",
        "uvvis": "uvvis",
        "fluorescence": "fluorescence",
        "xrd": "xrd",
        "hplc": "hplc",
        "electrochem": "electrochem",
    }.get(key)


def analyze_if_missing(spectrum: Spectrum) -> AnalysisResult | None:
    """Run the registered analyzer for a spectrum without persisting."""

    key = technique_key(spectrum.technique.value)
    if key is None:
        return None
    analyzer = AnalyzerRegistry.get(key)()
    return analyzer.analyze(spectrum)
