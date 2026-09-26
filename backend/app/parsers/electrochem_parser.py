from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from app.core.models import SampleInfo, Spectrum, Technique
from app.parsers.base import BaseParser


def _float_param(params: dict[str, str], name: str, default: float) -> float:
    try:
        value = float(params.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


def _int_param(params: dict[str, str], name: str, default: int) -> int:
    try:
        return int(float(params.get(name, default)))
    except (TypeError, ValueError):
        return int(default)


class ElectrochemParser(BaseParser):
    @classmethod
    def can_parse(cls, file_path: str | Path) -> bool:
        p = Path(file_path)
        if not p.is_file():
            return False
        if p.suffix.lower() not in (".txt", ".csv"):
            return False
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            return bool(re.search(r"(Cyclic Voltammetry|A\.C\. Impedance)", text))
        except Exception:
            return False

    def parse(self, file_path: str | Path) -> Spectrum:
        p = Path(file_path)
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()

        params: dict[str, str] = {}
        segments_meta: list[dict] = []
        data_lines: list[str] = []
        eis_header = ""
        technique_str = "CV"
        in_data = False
        in_segment_meta = False

        for line in lines:
            line = line.strip()

            if not in_data:
                if "A.C. Impedance" in line:
                    technique_str = "EIS"
                elif "Cyclic Voltammetry" in line:
                    technique_str = "CV"

                m = re.match(r"^(.+?)\s*=\s*(.+)$", line)
                if m:
                    params[m.group(1).strip()] = m.group(2).strip()

                if re.match(r"^Segment \d+:$", line):
                    in_segment_meta = True
                    segments_meta.append({})
                    continue
                if in_segment_meta:
                    m2 = re.match(r"^(\w+)\s*=\s*(.+)$", line)
                    if m2:
                        seg_name = m2.group(1).strip()
                        seg_val_str = m2.group(2).strip()
                        try:
                            if seg_name in ("ip",):
                                segments_meta[-1][f"{seg_name}_a"] = float(seg_val_str.replace("A", ""))
                            else:
                                segments_meta[-1][seg_name] = float(seg_val_str.rstrip("VCA"))
                        except ValueError:
                            pass
                    if re.match(r"Potential/V|Freq/Hz", line):
                        if "Freq/Hz" in line:
                            eis_header = line
                        in_segment_meta = False
                        in_data = True
                    continue
                # EIS: no Segment markers, detect data header directly
                if re.match(r"Freq/Hz", line) and not in_segment_meta:
                    eis_header = line
                    in_data = True
                    continue
            else:
                if re.match(r"Segment \d+:", line):
                    continue
                if re.match(r"^[\d.\-+eE]+,\s*[\d.\-+eE]", line):
                    data_lines.append(line)

        # Parse data
        pot_vals: list[float] = []
        curr_vals: list[float] = []
        charge_vals: list[float] = []
        time_vals: list[float] = []
        z_prime_vals: list[float] = []
        z_double_vals: list[float] = []

        is_eis = technique_str == "EIS"
        eis_cols = self._eis_columns(eis_header) if is_eis else None

        freq_vals: list[float] = []
        for dl in data_lines:
            parts = [p.strip() for p in dl.split(",")]
            try:
                if is_eis:
                    if eis_cols is not None:
                        freq_idx, z_prime_idx, z_double_idx = eis_cols
                        if max(freq_idx, z_prime_idx, z_double_idx) < len(parts):
                            freq_vals.append(float(parts[freq_idx]))
                            z_prime_vals.append(float(parts[z_prime_idx]))
                            z_double_vals.append(float(parts[z_double_idx]))
                    elif len(parts) >= 5:
                        freq_vals.append(float(parts[0]))
                        z_prime_vals.append(float(parts[1]))
                        z_double_vals.append(float(parts[2]))
                    continue
                if len(parts) >= 2:
                    pot_vals.append(float(parts[0]))
                    curr_vals.append(float(parts[1]))
                if len(parts) >= 3:
                    charge_vals.append(float(parts[2]))
                if len(parts) >= 4:
                    time_vals.append(float(parts[3]))
            except (ValueError, IndexError):
                continue

        scan_rate = _float_param(params, "Scan Rate (V/s)", 0.05)
        quiet_time = _float_param(params, "Quiet Time (sec)", 2)
        n_segments = _int_param(params, "Segment", 2)
        init_e = _float_param(params, "Init E (V)", 0.6)
        high_e = _float_param(params, "High E (V)", 0.6)
        low_e = _float_param(params, "Low E (V)", 0)

        parameters = {
            "scan_rate_v_s": scan_rate,
            "quiet_time_sec": quiet_time,
            "n_segments": n_segments,
            "init_e_v": init_e,
            "high_e_v": high_e,
            "low_e_v": low_e,
            "segments_meta": segments_meta,
            "sub_type": technique_str,
        }

        if is_eis:
            # The y axis is labelled -Z'' (positive for a capacitive
            # semicircle). CHI files store the raw Z'' column (negative
            # values), while some vendors export an already negated -Z''
            # column. Decide from the column header and record the source.
            if eis_cols is not None:
                imag_label = eis_header.split(",")[eis_cols[2]].strip()
                imag_already_negated = (
                    imag_label.upper().startswith("-") or "-Z" in imag_label.upper().replace(" ", "")
                )
                if imag_already_negated:
                    z_imag_sign_source = f"file_header:{imag_label} (already -Z'')"
                else:
                    z_double_vals = [-value for value in z_double_vals]
                    z_imag_sign_source = (
                        f"file_header:{imag_label} (raw Z'' negated to -Z'')"
                    )
            else:
                imag_already_negated = None
                z_imag_sign_source = "unknown:no column header; values kept as stored"

            params_meta = {
                "amplitude_v": _float_param(params, "Amplitude (V)", 0.001),
                "high_freq_hz": _float_param(params, "High Frequency (Hz)", 1e5),
                "low_freq_hz": _float_param(params, "Low Frequency (Hz)", 0.1),
                "freq_hz": [float(value) for value in freq_vals],
                "z_imag_sign_source": z_imag_sign_source,
                "z_imag_already_negated": imag_already_negated,
            }
            parameters.update(params_meta)

            return Spectrum(
                technique=Technique.ELECTROCHEM,
                x_data=np.array(z_prime_vals, dtype=np.float64),
                y_data=np.array(z_double_vals, dtype=np.float64),
                x_label="Z' (Real)",
                y_label="-Z\" (Imaginary)",
                x_unit="Ω",
                y_unit="Ω",
                parameters=parameters,
                metadata=SampleInfo(name=p.stem),
                source_file=str(p),
            )

        return Spectrum(
            technique=Technique.ELECTROCHEM,
            x_data=np.array(pot_vals, dtype=np.float64),
            y_data=np.array(curr_vals, dtype=np.float64),
            x_label="Potential",
            y_label="Current",
            x_unit="V",
            y_unit="A",
            parameters=parameters,
            metadata=SampleInfo(name=p.stem),
            source_file=str(p),
        )

    @staticmethod
    def _eis_columns(header: str) -> tuple[int, int, int] | None:
        """Locate (freq, Z', Z'') column indices in an EIS data header.

        Returns None when the header does not name both impedance columns;
        callers then fall back to the historical fixed positions.
        """
        if not header:
            return None
        cells = [cell.strip() for cell in header.split(",")]
        z_prime_idx = None
        z_double_idx = None
        for index, cell in enumerate(cells):
            upper = cell.upper()
            if 'Z"' in upper or "Z''" in upper:
                z_double_idx = index
            elif "Z'" in upper:
                z_prime_idx = index
        if z_prime_idx is None or z_double_idx is None:
            return None
        return 0, z_prime_idx, z_double_idx
