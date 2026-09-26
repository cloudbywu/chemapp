from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.core.models import Spectrum


class BaseParser(ABC):
    @abstractmethod
    def parse(self, file_path: str | Path) -> Spectrum:
        ...

    @classmethod
    @abstractmethod
    def can_parse(cls, file_path: str | Path) -> bool:
        ...
