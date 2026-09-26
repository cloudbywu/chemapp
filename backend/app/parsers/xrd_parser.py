from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from app.core.models import SampleInfo, Spectrum, Technique
from app.parsers.base import BaseParser

# Rigaku *.asc data rows are comma separated intensities. Accept plain
# integers as well as scientific notation and negative values (e.g. 1.5E+03).
_DATA_LINE_RE = re.compile(r"^[\d,\s.eE+\-]+$")


def _safe_header_float(
    value: str,
    default: float,
    key: str,
    warnings: list[str],
) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        warnings.append(f"XRD header *{key} has a non-numeric value; using default {default}.")
        return default
    if not np.isfinite(result):
        warnings.append(f"XRD header *{key} is not finite; using default {default}.")
        return default
    return result


def _safe_header_int(
    value: str,
    default: int,
    key: str,
    warnings: list[str],
) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        warnings.append(f"XRD header *{key} has a non-numeric value; using default {default}.")
        return default
    return result


def _optional_header_float(
    params: dict[str, str],
    key: str,
    warnings: list[str],
) -> float | None:
    raw = params.get(key)
    if raw is None:
        return None
    try:
        result = float(raw)
    except (TypeError, ValueError):
        warnings.append(f"XRD header *{key} has a non-numeric value; entry ignored.")
        return None
    if not np.isfinite(result):
        warnings.append(f"XRD header *{key} is not finite; entry ignored.")
        return None
    return result


class XRDSParser(BaseParser):
    @classmethod
    def can_parse(cls, file_path: str | Path) -> bool:
        p = Path(file_path)
        if not p.is_file():
            return False
        if p.suffix.lower() not in (".asc", ".ras"):
            return False
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            return ("*SCAN_AXIS" in text and "2theta" in text) or "*RAS_DATA_START" in text or ("Theta_2-Theta" in text)
        except Exception:
            return False

    def parse(self, file_path: str | Path) -> Spectrum:
        p = Path(file_path)
        if p.suffix.lower() == ".asc":
            return self._parse_asc(p)
        elif p.suffix.lower() == ".ras":
            return self._parse_ras(p)
        else:
            return self._parse_asc(p)

    def _parse_asc(self, p: Path) -> Spectrum:
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        params: dict[str, str] = {}
        parser_warnings: list[str] = []
        start_2theta = 10.0
        stop_2theta = 80.0
        step = 0.02
        wavelength1 = 1.54059
        wavelength2 = 1.54441

        i = 0
        intensities: list[float] = []
        in_data = False
        count = 0

        while i < len(lines):
            line = lines[i].strip()
            i += 1

            if not line:
                continue

            m = re.match(r"\*(\w+)\s*=\s*(.*)", line)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()
                params[key] = val

                if key == "START":
                    start_2theta = _safe_header_float(val, start_2theta, key, parser_warnings)
                elif key == "STOP":
                    stop_2theta = _safe_header_float(val, stop_2theta, key, parser_warnings)
                elif key == "STEP":
                    step = _safe_header_float(val, step, key, parser_warnings)
                elif key == "WAVE_LENGTH1":
                    wavelength1 = _safe_header_float(val, wavelength1, key, parser_warnings)
                elif key == "WAVE_LENGTH2":
                    wavelength2 = _safe_header_float(val, wavelength2, key, parser_warnings)
                elif key == "COUNT":
                    count = _safe_header_int(val, count, key, parser_warnings)
                    in_data = True
                continue

            if in_data and _DATA_LINE_RE.match(line):
                for token in line.split(","):
                    token = token.strip()
                    if token:
                        try:
                            intensities.append(float(token))
                        except ValueError:
                            parser_warnings.append(
                                f"XRD data row contains a non-numeric token that was skipped: {token!r}."
                            )

        n = len(intensities)
        if n == 0 and count > 0:
            raise ValueError(
                f"XRD ASC file declares *COUNT={count} but contains no parseable data rows."
            )
        if step <= 0:
            parser_warnings.append("XRD header *STEP is not positive; using default 0.02.")
            step = 0.02
        if n == 0:
            n = int((stop_2theta - start_2theta) / step) + 1
            intensities = [0.0] * n

        x_data = np.linspace(start_2theta, stop_2theta, n)

        parameters = {
            "wavelength1_a": wavelength1,
            "wavelength2_a": wavelength2,
            "target": params.get("TARGET", "Cu"),
            "scan_axis": params.get("SCAN_AXIS", "2theta/theta"),
            "scan_mode": params.get("SCAN_MODE", ""),
            "kv": _optional_header_float(params, "KV", parser_warnings),
            "ma": _optional_header_float(params, "MA", parser_warnings),
            "step": step,
            "parser_warnings": parser_warnings,
        }

        return Spectrum(
            technique=Technique.XRD,
            x_data=x_data,
            y_data=np.array(intensities, dtype=np.float64),
            x_label="2θ",
            y_label="Intensity",
            x_unit="deg",
            y_unit="counts",
            parameters=parameters,
            metadata=SampleInfo(name=p.stem),
            source_file=str(p),
        )

    def _parse_ras(self, p: Path) -> Spectrum:
        text = p.read_text(encoding="utf-8", errors="replace")

        x_vals: list[float] = []
        y_vals: list[float] = []
        start_2theta = 10.0
        stop_2theta = 80.0
        wavelength1 = 1.54059
        in_header = False
        in_data = False

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("*RAS_HEADER_START"):
                in_header = True
                in_data = False
                continue
            if line.startswith("*RAS_HEADER_END"):
                in_header = False
                continue
            if line.startswith("*RAS_DATA_START"):
                in_data = True
                continue
            if line.startswith("*RAS_DATA_END"):
                in_data = False
                continue

            if in_header:
                m = re.match(r"\*(\w+)\s+\"(.*)\"", line)
                if m:
                    key = m.group(1)
                    val = m.group(2)
                    if key == "MEAS_COND_WAVE_LENGTH1":
                        try:
                            wavelength1 = float(val)
                        except ValueError:
                            pass

            if in_data:
                parts = line.split()
                if len(parts) == 2:
                    try:
                        x_vals.append(float(parts[0]))
                        y_vals.append(float(parts[1]))
                    except ValueError:
                        continue

        if x_vals:
            x_data = np.array(x_vals, dtype=np.float64)
            y_data = np.array(y_vals, dtype=np.float64)
        else:
            x_data = np.linspace(start_2theta, stop_2theta, 3501)
            y_data = np.zeros(3501)

        return Spectrum(
            technique=Technique.XRD,
            x_data=x_data,
            y_data=y_data,
            x_label="2θ",
            y_label="Intensity",
            x_unit="deg",
            y_unit="counts",
            parameters={
                "wavelength1_a": wavelength1,
                "target": "Cu",
            },
            metadata=SampleInfo(name=p.stem),
            source_file=str(p),
        )
