from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from app.analysis.models import AnalysisResult
from app.core.models import Spectrum


class BaseAnalyzer(ABC):
    technique: str

    @abstractmethod
    def analyze(self, spectrum: Spectrum, options: dict[str, Any] | None = None) -> AnalysisResult:
        ...
