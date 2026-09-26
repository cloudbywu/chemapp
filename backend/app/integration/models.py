from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CrossValidationItem:
    pair: tuple[str, str]
    metric: str
    description: str
    score: float
    detail: str = ""


@dataclass
class InferenceResult:
    technique_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    cross_validations: list[CrossValidationItem] = field(default_factory=list)
    consistency_score: float = 0.0
    confidence: float = 0.0
    anomalies: list[str] = field(default_factory=list)
    conclusions: list[str] = field(default_factory=list)
    overall_assessment: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "technique_results": self.technique_results,
            "cross_validations": [
                {
                    "pair": list(cv.pair),
                    "metric": cv.metric,
                    "description": cv.description,
                    "score": cv.score,
                    "detail": cv.detail,
                }
                for cv in self.cross_validations
            ],
            "consistency_score": self.consistency_score,
            "confidence": self.confidence,
            "anomalies": self.anomalies,
            "conclusions": self.conclusions,
            "overall_assessment": self.overall_assessment,
        }


@dataclass
class ComprehensiveReport:
    sample_name: str
    techniques: list[str]
    inference: InferenceResult
    generated_at: str = ""
    report_markdown: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_name": self.sample_name,
            "techniques": self.techniques,
            "inference": self.inference.to_dict(),
            "generated_at": self.generated_at,
            "report_markdown": self.report_markdown,
        }
