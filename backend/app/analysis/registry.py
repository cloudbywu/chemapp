from __future__ import annotations

from typing import Type

from app.analysis.base import BaseAnalyzer


class AnalyzerRegistry:
    _analyzers: dict[str, Type[BaseAnalyzer]] = {}

    @classmethod
    def register(cls, name: str, analyzer_cls: Type[BaseAnalyzer]) -> None:
        cls._analyzers[name] = analyzer_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseAnalyzer]:
        if name not in cls._analyzers:
            raise KeyError(f"Unknown analyzer: {name}. Available: {list(cls._analyzers)}")
        return cls._analyzers[name]

    @classmethod
    def list_analyzers(cls) -> dict[str, Type[BaseAnalyzer]]:
        return dict(cls._analyzers)
