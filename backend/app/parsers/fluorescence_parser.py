from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from app.core.models import SampleInfo, Spectrum, Technique
from app.parsers.base import BaseParser

# Same safety limit as the NMR JCAMP-DX reader: refuse oversized text files.
_MAX_DX_FILE_BYTES = 64 * 1024 * 1024

# JCAMP-DX pseudo-digit tables (mirrors the pinned nmrglue decoder).
_DIGITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]
_SQZ_DIGITS = {
    "@": "0",
    "A": "1", "B": "2", "C": "3", "D": "4", "E": "5",
    "F": "6", "G": "7", "H": "8", "I": "9",
    "a": "-1", "b": "-2", "c": "-3", "d": "-4", "e": "-5",
    "f": "-6", "g": "-7", "h": "-8", "i": "-9",
}
_DIF_DIGITS = {
    "%": "0",
    "J": "1", "K": "2", "L": "3", "M": "4", "N": "5",
    "O": "6", "P": "7", "Q": "8", "R": "9",
    "j": "-1", "k": "-2", "l": "-3", "m": "-4", "n": "-5",
    "o": "-6", "p": "-7", "q": "-8", "r": "-9",
}
_DUP_DIGITS = {
    "S": "1", "T": "2", "U": "3", "V": "4", "W": "5",
    "X": "6", "Y": "7", "Z": "8", "s": "9",
}

_EMISSION_KEYWORDS = {"emission", "em"}
_EXCITATION_KEYWORDS = {"excitation", "ex"}

_LEADING_NUMBER_RE = re.compile(r"(\s)*([+-]?\d+\.?\d*|[+-]?\.\d+)([eE][+-]?\d+)?")


def _finite_float_or_none(raw: object) -> float | None:
    try:
        value = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _float_param(raw: object, default: float) -> float:
    """Guarded float conversion in the style of nmr_parser._float_param."""
    value = _finite_float_or_none(raw)
    return float(default) if value is None else value


def _wavelength_or_none(raw: str | None) -> float | None:
    if not raw:
        return None
    cleaned = re.sub(r"\s*nm", "", raw).strip()
    return _finite_float_or_none(cleaned)


def _normalize_sub_type(scan_mode: str | None) -> str:
    if not scan_mode:
        return ""
    normalized = scan_mode.strip().lower()
    tokens = {t for t in re.split(r"[\s,;/]+", normalized) if t}
    if "emission" in normalized or tokens & _EMISSION_KEYWORDS:
        return "emission"
    if "excitation" in normalized or tokens & _EXCITATION_KEYWORDS:
        return "excitation"
    return ""


def _contains_pseudodigits(lines: list[str]) -> bool:
    for line in lines:
        match = _LEADING_NUMBER_RE.match(line.strip())
        if match is None:
            continue
        for char in line.strip()[match.end():]:
            if char.isalpha() and char not in ("e", "E"):
                return True
    return False


def _append_decoded(values: list[float], pending: tuple[float, bool]) -> None:
    value, is_dif = pending
    if is_dif:
        if not values:
            raise ValueError("JCAMP-DX DIF entry has no preceding value.")
        values.append(values[-1] + value)
    else:
        values.append(value)


def _finish_number(
    values: list[float],
    pending: tuple[float, bool] | None,
    number_str: str,
    mode: int,
) -> tuple[float, bool] | None:
    if mode == 0:
        return pending
    try:
        value = float(number_str)
    except ValueError as exc:
        raise ValueError(f"JCAMP-DX data row contains an invalid value: {number_str!r}.") from exc
    if mode == 1:  # SQZ: absolute value
        if pending is not None:
            _append_decoded(values, pending)
        return (value, False)
    if mode == 2:  # DIF: delta from the previous value
        if pending is not None:
            _append_decoded(values, pending)
        return (value, True)
    # mode == 3: DUP — repeat the preceding value/difference count times.
    if pending is None:
        raise ValueError("JCAMP-DX DUP entry has no preceding value.")
    for _ in range(int(value)):
        _append_decoded(values, pending)
    return None


def _decode_difdup_table(lines: list[str]) -> tuple[float, list[float]]:
    """Decode a full X++(Y..Y) DIF/DUP compressed table.

    Mirrors the pinned nmrglue decoder over the whole table: values are
    appended one token late because a DUP repeats the previous token, and
    when a row ended with a DIF the first token of the next row is a
    checkpoint repeat of that DIF and is skipped. Returns the first row's
    X anchor plus the flat decoded Y stream.
    """
    values: list[float] = []
    pending: tuple[float, bool] | None = None
    mode = 0
    number: list[str] = []
    skip_checkpoint = False
    first_x: float | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        match = _LEADING_NUMBER_RE.match(stripped)
        if match is None:
            continue
        if first_x is None:
            first_x = float(stripped[: match.end()])
        first_of_line = True
        for char in stripped[match.end():]:
            if char in _DIGITS:
                number.append(char)
                continue
            if char.isspace():
                continue
            if char in _SQZ_DIGITS:
                digit, new_mode = _SQZ_DIGITS[char], 1
            elif char in _DIF_DIGITS:
                digit, new_mode = _DIF_DIGITS[char], 2
            elif char in _DUP_DIGITS:
                digit, new_mode = _DUP_DIGITS[char], 3
            else:
                raise ValueError(
                    f"Unsupported JCAMP-DX pseudo-digit {char!r} in row: {stripped!r}."
                )
            previous_is_dif = mode == 2 or (mode == 3 and pending is not None and pending[1])
            if not skip_checkpoint:
                pending = _finish_number(values, pending, "".join(number), mode)
            skip_checkpoint = first_of_line and previous_is_dif
            mode = new_mode
            number = [digit]
            first_of_line = False
    if not skip_checkpoint:
        pending = _finish_number(values, pending, "".join(number), mode)
    if pending is not None:
        _append_decoded(values, pending)
    if first_x is None:
        raise ValueError("JCAMP-DX DIF/DUP table contains no data rows.")
    return first_x, values


def _parse_xydata_table(lines: list[str], delta_x: float | None) -> tuple[list[float], list[float]]:
    """Decode an ##XYDATA= (X++(Y..Y)) table (AFFN or DIF/DUP compressed)."""
    if _contains_pseudodigits(lines):
        if delta_x is None:
            raise ValueError("JCAMP-DX DIF/DUP table requires a valid ##DELTAX.")
        first_x, y_vals = _decode_difdup_table(lines)
        x_vals = [first_x + index * delta_x for index in range(len(y_vals))]
        return x_vals, y_vals

    rows: list[tuple[float, list[float]]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        tokens = stripped.replace(",", " ").split()
        try:
            x_value = float(tokens[0])
            y_values = [float(token) for token in tokens[1:]]
        except (ValueError, IndexError):
            continue
        if y_values:
            rows.append((x_value, y_values))

    x_vals: list[float] = []
    y_vals: list[float] = []
    for x_value, y_values in rows:
        if len(y_values) > 1 and delta_x is None:
            raise ValueError(
                "JCAMP-DX X++(Y..Y) table holds multiple Y values per row "
                "but ##DELTAX is missing or invalid."
            )
        step = delta_x if delta_x is not None else 0.0
        for index, y_value in enumerate(y_values):
            x_vals.append(x_value + index * step)
            y_vals.append(y_value)
    return x_vals, y_vals


def _parse_peak_table(lines: list[str]) -> tuple[list[float], list[float]]:
    x_vals: list[float] = []
    y_vals: list[float] = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            float(parts[0])
            x_vals.append(float(parts[0]))
            y_vals.append(float(parts[1]))
        except ValueError:
            continue
    return x_vals, y_vals


class FluorescenceParser(BaseParser):
    @classmethod
    def can_parse(cls, file_path: str | Path) -> bool:
        p = Path(file_path)
        if not p.is_file():
            return False
        if p.suffix.lower() != ".dx":
            return False
        if p.stat().st_size > _MAX_DX_FILE_BYTES:
            return False
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            return "##JCAMP-DX" in text and ("FL SPECTRUM" in text or "FLUORESCENCE" in text.upper())
        except Exception:
            return False

    def parse(self, file_path: str | Path) -> Spectrum:
        p = Path(file_path)
        if p.stat().st_size > _MAX_DX_FILE_BYTES:
            raise ValueError("Fluorescence JCAMP-DX file exceeds the 64 MiB safety limit.")

        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        metadata: dict[str, str] = {}
        xydata_lines: list[str] = []
        xydata_format = ""
        peak_table_lines: list[str] = []
        data_mode = ""  # "", "xydata", "peak"

        key_pattern = re.compile(r"##(\$?[\w\s/]+)=\s*(.*)")
        multi_line_key = ""
        multi_line_val = ""

        for line in lines:
            stripped = line.strip()

            if data_mode:
                if stripped.startswith("##"):
                    data_mode = ""
                else:
                    if stripped:
                        if data_mode == "xydata":
                            xydata_lines.append(line)
                        else:
                            peak_table_lines.append(line)
                    continue

            if not stripped:
                continue

            m = key_pattern.match(stripped)
            if m:
                key = m.group(1).strip()
                val = m.group(2).strip()

                if val.endswith(";"):
                    multi_line_key = key
                    multi_line_val = val
                else:
                    metadata[key] = val
            elif multi_line_key:
                if stripped.startswith("##"):
                    metadata[multi_line_key] = multi_line_val.rstrip(";")
                    multi_line_key = ""
                    multi_line_val = ""
                    m2 = key_pattern.match(stripped)
                    if m2:
                        metadata[m2.group(1).strip()] = m2.group(2).strip()
                else:
                    multi_line_val += " " + stripped

            if stripped.startswith("##XYDATA="):
                format_match = re.match(r"##XYDATA=\s*(\S+)", stripped)
                xydata_format = format_match.group(1) if format_match else ""
                data_mode = "xydata"
                continue
            if "##PEAK TABLE" in stripped:
                data_mode = "peak"
                continue

        if multi_line_key:
            metadata[multi_line_key] = multi_line_val.rstrip(";")

        scan_mode = self._extract_param(metadata, "INSTRUMENT PARAMETERS", "Scan mode")
        ex_wl_raw = self._extract_param(metadata, "INSTRUMENT PARAMETERS", "EX WL")
        em_wl_raw = self._extract_param(metadata, "INSTRUMENT PARAMETERS", "EM WL")
        title = metadata.get("TITLE", p.stem)

        delta_x = _finite_float_or_none(metadata.get("DELTAX"))
        ex_wl = _wavelength_or_none(ex_wl_raw)
        em_wl = _wavelength_or_none(em_wl_raw)

        sub_type = _normalize_sub_type(scan_mode)

        x_vals: list[float] = []
        y_vals: list[float] = []
        if xydata_lines:
            normalized_format = xydata_format.upper().replace(" ", "")
            if "X++(Y..Y)" in normalized_format or not xydata_format:
                x_vals, y_vals = _parse_xydata_table(xydata_lines, delta_x)
            else:
                raise ValueError(
                    f"Unsupported fluorescence ##XYDATA= format: {xydata_format}."
                )
            if not x_vals:
                raise ValueError(
                    "Fluorescence ##XYDATA= table declared no parseable data rows."
                )
        elif peak_table_lines:
            x_vals, y_vals = _parse_peak_table(peak_table_lines)

        parameters = {
            "scan_mode": scan_mode,
            "excitation_wavelength_nm": ex_wl,
            "emission_wavelength_nm": em_wl,
            "delta_x_nm": delta_x if delta_x is not None else 0.2,
            "title": title,
            "sub_type": sub_type,
            "raw_metadata": metadata,
        }

        x_data = np.array(x_vals, dtype=np.float64)
        y_data = np.array(y_vals, dtype=np.float64)

        return Spectrum(
            technique=Technique.FLUORESCENCE,
            x_data=x_data,
            y_data=y_data,
            x_label="Wavelength",
            y_label="Intensity",
            x_unit="nm",
            y_unit="arb. units",
            parameters=parameters,
            metadata=SampleInfo(name=title),
            source_file=str(p),
        )

    @staticmethod
    def _extract_param(metadata: dict[str, str], section: str, key: str) -> str | None:
        val = metadata.get(section, "")
        if not val:
            return None
        m = re.search(rf"{key}=\s*([^;]+)", val)
        return m.group(1).strip() if m else None
