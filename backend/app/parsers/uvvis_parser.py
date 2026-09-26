from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from app.core.models import SampleInfo, Spectrum, Technique
from app.parsers.base import BaseParser

_ENCODINGS = ("gbk", "gb2312", "utf-8-sig", "utf-8", "latin-1")
_SPECTRUM_KEYWORDS = ("波长", "Wavelength", "吸收", "Absorbance", "RawData")
_CAL_KEYWORDS = ("浓度", "Concentration", "样品", "Sample", "权重", "Weight")


def _read_with_fallback(file_path: Path) -> str:
    best = ""
    for enc in _ENCODINGS:
        try:
            text = file_path.read_text(encoding=enc, errors="replace")
            if any(kw in text for kw in _SPECTRUM_KEYWORDS + _CAL_KEYWORDS):
                return text
            if not best:
                best = text
        except (UnicodeDecodeError, LookupError):
            continue
    return best or file_path.read_text(encoding="utf-8", errors="replace")


class UVVisParser(BaseParser):
    @classmethod
    def can_parse(cls, file_path: str | Path) -> bool:
        p = Path(file_path)
        if not p.is_file():
            return False
        if p.suffix.lower() not in (".txt", ".csv", ".tsv"):
            return False
        try:
            text = _read_with_fallback(p)
            has_header = any(
                kw in text
                for kw in ["波长", "Wavelength", "吸收", "Absorbance", "浓度", "Concentration", "WL"]
            )
            has_numbers = bool(re.search(r"\d+\.\d+\t\d+\.\d+", text))
            return has_header or has_numbers
        except Exception:
            return False

    def parse(self, file_path: str | Path) -> Spectrum:
        p = Path(file_path)
        text = _read_with_fallback(p)

        if self._is_calibration_type(text):
            return self._parse_calibration(text, file_path)
        else:
            return self._parse_spectrum(text, file_path)

    @staticmethod
    def _is_calibration_type(text: str) -> bool:
        return "浓度" in text or "Concentration" in text.lower()

    @staticmethod
    def _is_spectrum_type(text: str) -> bool:
        return bool(re.search(r"波长.*nm|Wavelength", text))

    @staticmethod
    def _detect_delimiter(lines: list[str], start: int) -> str:
        # Vendor exports arrive as tab-, comma-, or semicolon-separated text;
        # decide from the first data rows instead of assuming tabs.
        counts = {"\t": 0, ",": 0, ";": 0}
        for line in lines[start:start + 20]:
            for delim in counts:
                counts[delim] += line.count(delim)
        best = max(counts, key=lambda d: counts[d])
        return best if counts[best] > 0 else "\t"

    def _parse_spectrum(self, text: str, file_path: str | Path) -> Spectrum:
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        data_start = 0
        for i, line in enumerate(lines):
            if "波长" in line or "Wavelength" in line.lower() or "RawData" in line:
                data_start = i + 1
                break

        x_vals: list[float] = []
        y_vals: list[float] = []
        delimiter = self._detect_delimiter(lines, data_start)

        for line in lines[data_start:]:
            parts = line.split(delimiter)
            if len(parts) < 2:
                continue
            try:
                x = float(parts[0].strip().strip('"'))
                y = float(parts[1].strip().strip('"'))
                x_vals.append(x)
                y_vals.append(y)
            except (ValueError, IndexError):
                continue

        if not x_vals:
            raise ValueError(
                "UV-Vis spectrum file contains no parseable numeric rows "
                f"(detected delimiter: {delimiter!r})."
            )

        x_data = np.array(x_vals, dtype=np.float64)
        y_data = np.array(y_vals, dtype=np.float64)

        return Spectrum(
            technique=Technique.UVVIS,
            x_data=x_data,
            y_data=y_data,
            x_label="Wavelength",
            y_label="Absorbance",
            x_unit="nm",
            y_unit="abs",
            parameters={},
            metadata=SampleInfo(),
            source_file=str(file_path),
        )

    def _parse_calibration(self, text: str, file_path: str | Path) -> Spectrum:
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        header_row = None
        for line in lines:
            if any(kw in line for kw in ["浓度", "Concentration", "样品", "Sample"]):
                header_row = line
                break

        if header_row is None:
            return self._parse_spectrum(text, file_path)

        delimiter = "," if "," in header_row else "\t"

        raw_header = [c.strip().strip('"') for c in header_row.split(delimiter)]
        conc_idx = None
        wl_idx = None
        for i, h in enumerate(raw_header):
            if "浓度" in h or "Concentration" in h.lower():
                conc_idx = i
            if h.upper().startswith("WL") or re.match(r"WL\d+", h):
                wl_idx = i

        if conc_idx is None or wl_idx is None:
            return self._parse_spectrum(text, file_path)

        x_vals: list[float] = []
        y_vals: list[float] = []

        header_found = False
        for line in lines:
            if not header_found:
                if any(kw in line for kw in ["浓度", "Concentration", "样品", "Sample"]):
                    header_found = True
                continue

            parts = [c.strip().strip('"') for c in line.split(delimiter)]
            try:
                if conc_idx < len(parts) and wl_idx < len(parts):
                    conc = float(parts[conc_idx])
                    abs_val = float(parts[wl_idx])
                    x_vals.append(conc)
                    y_vals.append(abs_val)
            except (ValueError, IndexError):
                continue

        if not x_vals:
            return self._parse_spectrum(text, file_path)

        return Spectrum(
            technique=Technique.UVVIS,
            x_data=np.array(x_vals, dtype=np.float64),
            y_data=np.array(y_vals, dtype=np.float64),
            x_label="Concentration",
            y_label="Absorbance",
            x_unit="mg/mL",
            y_unit="abs",
            parameters={"type": "calibration"},
            metadata=SampleInfo(),
            source_file=str(file_path),
        )
