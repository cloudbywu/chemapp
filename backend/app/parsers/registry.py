from __future__ import annotations

from pathlib import Path
from typing import Type

from app.parsers.base import BaseParser


class ParserRegistry:
    _parsers: dict[str, Type[BaseParser]] = {}

    @classmethod
    def register(cls, name: str, parser_cls: Type[BaseParser]) -> None:
        cls._parsers[name] = parser_cls

    @classmethod
    def get(cls, name: str) -> Type[BaseParser]:
        if name not in cls._parsers:
            raise KeyError(f"Unknown parser: {name}. Available: {list(cls._parsers)}")
        return cls._parsers[name]

    @classmethod
    def detect(cls, file_path: str | Path) -> str | None:
        for name, parser_cls in cls._parsers.items():
            if parser_cls.can_parse(file_path):
                return name
        return None

    @classmethod
    def list_parsers(cls) -> dict[str, Type[BaseParser]]:
        return dict(cls._parsers)
