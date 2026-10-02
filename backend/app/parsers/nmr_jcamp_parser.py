from __future__ import annotations

import hashlib
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module=r"nmrglue(\..*)?",
    )
    import nmrglue as ng

from app.analysis.nmr_processing import (
    PROCESSING_PIPELINE_VERSION,
    asymmetric_least_squares_baseline,
    automatic_phase_correction,
    build_processing_source,
    processing_source_summary,
    solvent_reference_ppm,
    spectrum_quality_metrics,
)
from app.core.models import NMRNucleus, SampleInfo, Spectrum, Technique
from app.parsers.base import BaseParser
from app.parsers.nmr_parser import _remove_digital_filter

_NUCLEI = {
    "1H": NMRNucleus.H1,
    "13C": NMRNucleus.C13,
    "15N": NMRNucleus.N15,
    "19F": NMRNucleus.F19,
    "29SI": NMRNucleus.SI29,
    "31P": NMRNucleus.P31,
}
_MAX_JCAMP_FILE_BYTES = 64 * 1024 * 1024
_MAX_JCAMP_NUMERIC_VALUES = 16_000_000


def _first(dictionary: dict[str, Any], key: str, default: Any = None) -> Any:
    value = dictionary.get(key)
    if isinstance(value, list):
        return value[0] if value else default
    return default if value is None else value


def _clean_text(value: Any) -> str:
    return str(value or "").strip().strip('"').strip("<>").strip()


def _finite_float(dictionary: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        raw = _first(dictionary, key)
        if raw in (None, ""):
            continue
        try:
            value = float(str(raw).split(",", 1)[0].strip())
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            return value
    return None


def _first_component(dictionary: dict[str, Any], *keys: str) -> float | None:
    return _finite_float(dictionary, *keys)


def _x_unit(dictionary: dict[str, Any]) -> str:
    raw = _first(dictionary, "XUNITS")
    if raw in (None, ""):
        raw = _first(dictionary, "UNITS", "")
    return str(raw).split(",", 1)[0].strip().upper()


def _nucleus(dictionary: dict[str, Any]) -> NMRNucleus:
    raw = _clean_text(
        _first(
            dictionary,
            ".OBSERVENUCLEUS",
            _first(dictionary, "$NUC1", "1H"),
        )
    )
    normalized = raw.replace("^", "").replace(" ", "").upper()
    return _NUCLEI.get(normalized, NMRNucleus.H1)


def _frequency_mhz(dictionary: dict[str, Any]) -> float:
    value = _finite_float(
        dictionary,
        ".OBSERVEFREQUENCY",
        "$SFO1",
        "$SF",
        "$BF1",
    )
    if value is None or value <= 0:
        raise ValueError("NMR JCAMP-DX is missing a positive observe frequency")
    return float(value)


def _frequency_axis_ppm(
    dictionary: dict[str, Any],
    size: int,
    *,
    frequency_mhz: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    first_x = _first_component(dictionary, "FIRSTX", "FIRST")
    last_x = _first_component(dictionary, "LASTX", "LAST")
    if first_x is None or last_x is None:
        raise ValueError("NMR JCAMP-DX is missing FIRST/LAST axis values")
    unit = _x_unit(dictionary)
    raw_axis = np.linspace(first_x, last_x, size, dtype=np.float64)

    if "PPM" in unit:
        return raw_axis, {
            "input_unit": unit,
            "conversion": "identity_ppm",
            "first_input": first_x,
            "last_input": last_x,
        }
    if "HZ" not in unit:
        raise ValueError(
            f"Unsupported NMR JCAMP-DX frequency-axis unit: {unit or 'missing'}"
        )

    sf = _finite_float(dictionary, "$SF", "$SFO1") or frequency_mhz
    offset = _finite_float(dictionary, "$OFFSET")
    if offset is not None:
        ppm_axis = offset + (raw_axis - first_x) / sf
        conversion = "bruker_offset_and_sf"
    else:
        carrier_hz = _finite_float(dictionary, "$O1")
        carrier_ppm = carrier_hz / sf if carrier_hz is not None else 0.0
        midpoint_hz = 0.5 * (first_x + last_x)
        ppm_axis = carrier_ppm + (raw_axis - midpoint_hz) / sf
        conversion = "carrier_centered_hz"
    return ppm_axis, {
        "input_unit": unit,
        "conversion": conversion,
        "first_input": first_x,
        "last_input": last_x,
        "spectrometer_frequency_mhz": sf,
        "offset_ppm": offset,
    }


def _fid_axis_ppm(
    dictionary: dict[str, Any],
    size: int,
    *,
    frequency_mhz: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    sweep_hz = _finite_float(dictionary, "$SWH", "$SW_H", "$SWP")
    if sweep_hz is None or sweep_hz <= 0:
        first_x = _first_component(dictionary, "FIRSTX", "FIRST")
        last_x = _first_component(dictionary, "LASTX", "LAST")
        duration = (
            abs(last_x - first_x)
            if first_x is not None and last_x is not None
            else 0.0
        )
        sweep_hz = (size - 1) / duration if duration > 0 else None
    if sweep_hz is None or sweep_hz <= 0:
        raise ValueError("NMR JCAMP-DX FID is missing a usable sweep width")

    sfo1 = _finite_float(dictionary, "$SFO1") or frequency_mhz
    carrier_hz = _finite_float(dictionary, "$O1")
    carrier_ppm = carrier_hz / sfo1 if carrier_hz is not None else 0.0
    sweep_ppm = sweep_hz / frequency_mhz
    axis = np.linspace(
        carrier_ppm + sweep_ppm / 2.0,
        carrier_ppm - sweep_ppm / 2.0,
        size,
        dtype=np.float64,
    )
    return axis, {
        "input_unit": _x_unit(dictionary),
        "conversion": "fid_fft_carrier_centered",
        "spectrometer_frequency_mhz": sfo1,
        "carrier_ppm": carrier_ppm,
        "sweep_width_hz": sweep_hz,
        "sweep_width_ppm": sweep_ppm,
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _numeric_digest(channels: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(channels):
        array = np.asarray(channels[name], dtype=np.float64)
        canonical = np.ascontiguousarray(array, dtype="<f8")
        digest.update(name.encode("ascii"))
        digest.update(str(tuple(canonical.shape)).encode("ascii"))
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def _parse_number_list(raw: Any) -> list[float]:
    values: list[float] = []
    for item in str(raw or "").replace("\n", " ").split(","):
        for token in item.split():
            try:
                value = float(token)
            except ValueError:
                continue
            if np.isfinite(value):
                values.append(value)
    return values


def _select_raw_nmr_block(raw_dictionary: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    for datatype in ("NMRSPECTRUM", "NMRFID", "NDNMRSPECTRUM"):
        blocks = raw_dictionary.get(f"_datatype_{datatype}") or []
        if blocks:
            return datatype, blocks[0]
    raise ValueError("JCAMP-DX has no supported NMR data block")


def _check_numeric_budget(count: int) -> None:
    if count > _MAX_JCAMP_NUMERIC_VALUES:
        raise ValueError("NMR JCAMP-DX numeric payload exceeds the safety limit")


def _table_numeric_count(table: str) -> int:
    """Count nmrglue 0.11 table output without expanding any DUP tokens.

    Keep the pending-token/checkpoint semantics of the pinned decoder: a DUP
    replaces the pending token with N copies, rather than adding N extra copies.
    No global decoder hooks are changed, so concurrent uploads stay isolated.
    """

    lines = table.split("\n")[1:]
    if not lines or not lines[0].strip():
        raise ValueError("NMR JCAMP-DX table contains no data rows")
    try:
        mode = ng.fileio.jcampdx._detect_format(lines[0])
    except (AttributeError, IndexError) as exc:
        raise ValueError("NMR JCAMP-DX table contains an invalid data row") from exc
    count = 0
    if mode == 0:
        numbers = re.compile(r"(\s|,)*([+-]?\d+\.?\d*|[+-]?\.\d+)([eE][+-]?\d+)?")
        for line in lines:
            count += max(sum(1 for _ in numbers.finditer(line)) - 1, 0)
            _check_numeric_budget(count)
        return count
    if mode != 1:
        raise ValueError("NMR JCAMP-DX table contains an unsupported numeric format")

    decoder = ng.fileio.jcampdx
    anchor = re.compile(r"(\s)*([+-]?\d+\.?\d*|[+-]?\.\d+)")
    pending: bool | None = None
    current_mode = 0
    digits: list[str] = []
    skip_checkpoint = False

    def finish() -> None:
        nonlocal count, pending
        if current_mode in (1, 2):
            float("".join(digits))  # Match the decoder's numeric syntax.
            count += int(pending is not None)
            pending = current_mode == 2
        elif current_mode == 3:
            if pending is None:
                raise ValueError("NMR JCAMP-DX DUP entry has no preceding value")
            token = "".join(digits).lstrip("0") or "0"
            if not token.isdecimal():
                raise ValueError("NMR JCAMP-DX DUP count must be a positive integer")
            if len(token) > len(str(_MAX_JCAMP_NUMERIC_VALUES)):
                raise ValueError("NMR JCAMP-DX numeric payload exceeds the safety limit")
            repeats = int(token)
            if repeats < 1:
                raise ValueError("NMR JCAMP-DX DUP count must be a positive integer")
            count += repeats
            pending = None
        _check_numeric_budget(count)

    for line in lines:
        if not line:
            continue
        match = anchor.match(line)
        if match is None:
            raise ValueError("NMR JCAMP-DX table contains an invalid X anchor")
        first_of_line = True
        for char in line[match.end():].strip():
            if char in decoder._DIGITS:
                digits.append(char)
                continue
            if char in decoder._SQZ_DIGITS:
                digit, next_mode = decoder._SQZ_DIGITS[char], 1
            elif char in decoder._DIF_DIGITS:
                digit, next_mode = decoder._DIF_DIGITS[char], 2
            elif char in decoder._DUP_DIGITS:
                digit, next_mode = decoder._DUP_DIGITS[char], 3
            else:
                raise ValueError("NMR JCAMP-DX table contains an invalid pseudo-digit")
            previous_is_dif = current_mode == 2 or (current_mode == 3 and pending is True)
            if not skip_checkpoint:
                finish()
            skip_checkpoint = first_of_line and previous_is_dif
            current_mode = next_mode
            digits = [digit]
            first_of_line = False
    if not skip_checkpoint:
        finish()
    count += int(pending is not None)
    _check_numeric_budget(count)
    return count


def _validate_raw_numeric_budget(raw_dictionary: dict[str, Any]) -> None:
    # read() may fall back to later spectrum/FID/untyped blocks, and NTUPLES
    # parses every real/imaginary page even if only its first array is returned.
    total = 0
    for blocks in raw_dictionary.values():
        for block in blocks:
            for key in ("XYDATA", "DATATABLE"):
                for table in block.get(key, []):
                    total += _table_numeric_count(table)
                    _check_numeric_budget(total)


def _decode_nd_numeric(
    block: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[int]]:
    """Decode nD profile pages without collapsing them into a 1D Spectrum."""

    tables = block.get("DATATABLE") or []
    if not tables:
        raise ValueError("nD NMR JCAMP-DX has no DATA TABLE pages")
    real_rows: list[np.ndarray] = []
    imaginary_rows: list[np.ndarray] = []
    numeric_values = 0
    for table in tables:
        parsed = ng.fileio.jcampdx._parse_data(table)  # pinned nmrglue 0.11
        if parsed is None:
            raise ValueError("nD NMR JCAMP-DX contains an undecodable data page")
        row, channel = parsed
        row_array = np.asarray(row, dtype=np.float64)
        if row_array.ndim != 1 or len(row_array) == 0:
            raise ValueError("nD NMR JCAMP-DX contains an invalid data page")
        numeric_values += len(row_array)
        if numeric_values > _MAX_JCAMP_NUMERIC_VALUES:
            raise ValueError("NMR JCAMP-DX numeric payload exceeds the safety limit")
        if channel == "I":
            imaginary_rows.append(row_array)
        else:
            real_rows.append(row_array)

    factors = _parse_number_list(_first(block, "FACTOR", "1"))
    y_factor = factors[-1] if factors else 1.0

    channels: dict[str, np.ndarray] = {}
    for name, rows in (("real", real_rows), ("imaginary", imaginary_rows)):
        if not rows:
            continue
        row_lengths = {len(row) for row in rows}
        if len(row_lengths) != 1:
            raise ValueError("nD NMR JCAMP-DX pages have inconsistent lengths")
        channels[name] = np.vstack(rows) * y_factor
    if not channels:
        raise ValueError("nD NMR JCAMP-DX contains no numeric spectrum channel")
    declared = [
        int(value)
        for value in _parse_number_list(_first(block, "VARDIM", ""))
        if value >= 0 and float(value).is_integer()
    ]
    return channels, declared


def _decode_jcamp_numeric(
    path: Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    if path.stat().st_size > _MAX_JCAMP_FILE_BYTES:
        raise ValueError("NMR JCAMP-DX file exceeds the 64 MiB safety limit")
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        raw_dictionary = ng.fileio.jcampdx._readrawdic(  # pinned nmrglue 0.11
            str(path)
        )
        _validate_raw_numeric_budget(raw_dictionary)
        raw_datatype, raw_block = _select_raw_nmr_block(raw_dictionary)
        if raw_datatype == "NDNMRSPECTRUM":
            dictionary = raw_block
            channels, declared_dimensions = _decode_nd_numeric(raw_block)
            decoder_api = "nmrglue.fileio.jcampdx._parse_data"
            is_fid = False
        else:
            dictionary, decoded = ng.jcampdx.read(str(path))
            if decoded is None:
                raise ValueError(
                    "NMR JCAMP-DX contains no decodable XYDATA/NTUPLES data"
                )
            if isinstance(decoded, list):
                real = (
                    np.asarray(decoded[0], dtype=np.float64)
                    if decoded and decoded[0] is not None
                    else None
                )
                imaginary = (
                    np.asarray(decoded[1], dtype=np.float64)
                    if len(decoded) > 1 and decoded[1] is not None
                    else None
                )
            else:
                real = np.asarray(decoded, dtype=np.float64)
                imaginary = None
            if real is None:
                raise ValueError("NMR JCAMP-DX has no real spectrum channel")
            channels = {"real": real}
            if imaginary is not None:
                channels["imaginary"] = imaginary
            declared_dimensions = [
                int(value)
                for value in _parse_number_list(_first(dictionary, "VARDIM", ""))
                if value >= 0 and float(value).is_integer()
            ]
            decoder_api = "nmrglue.fileio.jcampdx.read"
            is_fid = raw_datatype == "NMRFID"

    numeric_value_count = int(sum(array.size for array in channels.values()))
    if numeric_value_count > _MAX_JCAMP_NUMERIC_VALUES:
        raise ValueError("NMR JCAMP-DX numeric payload exceeds the safety limit")
    shapes = {name: list(array.shape) for name, array in channels.items()}
    ranks = {array.ndim for array in channels.values()}
    if len(ranks) != 1:
        raise ValueError("NMR JCAMP-DX channels have inconsistent dimensions")
    dimension = next(iter(ranks))
    finite = all(bool(np.isfinite(array).all()) for array in channels.values())
    if not finite:
        raise ValueError("NMR JCAMP-DX numeric payload contains non-finite values")
    point_count = max(int(array.size) for array in channels.values())
    inspection = {
        "numeric_payload_decoded": True,
        "unsupported_reason": None,
        "decoder": "nmrglue",
        "decoder_version": str(getattr(ng, "__version__", "0.11")),
        "decoder_api": decoder_api,
        "data_type": _clean_text(_first(dictionary, "DATATYPE", raw_datatype)),
        "is_fid": is_fid,
        "dimension": dimension,
        "channel_shapes": shapes,
        "point_count": point_count,
        "numeric_value_count": numeric_value_count,
        "finite": finite,
        "declared_var_dim": declared_dimensions,
        "numeric_sha256": _numeric_digest(channels),
        "source_sha256": _sha256_file(path),
        "warnings": sorted({str(item.message) for item in captured}),
    }
    return dictionary, channels, inspection


def inspect_jcamp_numeric(file_path: str | Path) -> dict[str, Any]:
    """Decode and describe JCAMP numeric data without constructing a Spectrum.

    Multi-dimensional input remains multi-dimensional.  Unsupported content
    returns a structured reason instead of being silently flattened.
    """

    path = Path(file_path)
    try:
        _, _, inspection = _decode_jcamp_numeric(path)
        return inspection
    except (OSError, TypeError, ValueError) as exc:
        return {
            "numeric_payload_decoded": False,
            "unsupported_reason": str(exc),
            "decoder": "nmrglue",
            "decoder_version": str(getattr(ng, "__version__", "0.11")),
            "source_sha256": _sha256_file(path) if path.is_file() else None,
        }


class NMRJCAMPParser(BaseParser):
    """Read 1D NMR JCAMP-DX, including Bruker NTUPLES/DIFDUP files."""

    @classmethod
    def can_parse(cls, file_path: str | Path) -> bool:
        path = Path(file_path)
        if not path.is_file() or path.suffix.lower() not in {
            ".dx",
            ".jcamp",
            ".jdx",
        }:
            return False
        try:
            header = path.read_text(
                encoding="utf-8", errors="replace"
            )[:131072].upper()
        except OSError:
            return False
        return (
            "##JCAMP" in header
            and (
                "##DATA TYPE= NMR" in header
                or "##DATATYPE= NMR" in header
                or "##NTUPLES=NMR" in header
            )
        )

    def parse(self, file_path: str | Path) -> Spectrum:
        path = Path(file_path)
        dictionary, channels, inspection = _decode_jcamp_numeric(path)
        if inspection["dimension"] != 1:
            raise ValueError(
                "Only one-dimensional NMR JCAMP-DX can be represented as a "
                f"Spectrum; decoded dimension={inspection['dimension']}"
            )
        parser_warnings = list(inspection["warnings"])
        real = channels["real"]
        imaginary = channels.get("imaginary")
        if real.ndim != 1 or len(real) < 2 or not np.isfinite(real).all():
            raise ValueError("NMR JCAMP-DX real channel is invalid")
        if imaginary is not None and (
            imaginary.shape != real.shape or not np.isfinite(imaginary).all()
        ):
            raise ValueError("NMR JCAMP-DX imaginary channel is invalid")

        nucleus = _nucleus(dictionary)
        frequency_mhz = _frequency_mhz(dictionary)
        datatype = _clean_text(_first(dictionary, "DATATYPE", "NMR Spectrum"))
        is_fid = "FID" in datatype.upper()
        phase_result: dict[str, Any]
        baseline_method = "vendor_processed"
        digital_filter_applied = 0.0

        if is_fid:
            if imaginary is None:
                raise ValueError(
                    "NMR JCAMP-DX FID requires both real and imaginary channels"
                )
            fid = real + 1j * imaginary
            group_delay = _finite_float(dictionary, "$GRPDLY") or 0.0
            corrected_fid, removed_points, fractional_delay = _remove_digital_filter(
                fid,
                group_delay,
            )
            digital_filter_applied = float(removed_points + fractional_delay)
            requested_size = int(
                _finite_float(dictionary, "$SI") or len(corrected_fid)
            )
            fft_size = max(len(corrected_fid), min(requested_size, 1_048_576))
            complex_spectrum = np.fft.fftshift(
                np.fft.fft(corrected_fid, n=fft_size)
            )[::-1]
            x_data, axis_trace = _fid_axis_ppm(
                dictionary,
                len(complex_spectrum),
                frequency_mhz=frequency_mhz,
            )
            phased_real, phased_imaginary, phase_result = (
                automatic_phase_correction(
                    x_data,
                    np.real(complex_spectrum),
                    np.imag(complex_spectrum),
                    optimize_first_order=True,
                    max_first_deg=1440.0,
                )
            )
            _, y_data = asymmetric_least_squares_baseline(phased_real)
            quadrature_real = phased_real
            quadrature_imaginary = phased_imaginary
            baseline_method = "asymmetric_least_squares"
            phase_source = "automatic_zero_and_first_order"
            source_kind = "jcamp_nmr_fid_fft"
        else:
            x_data, axis_trace = _frequency_axis_ppm(
                dictionary,
                len(real),
                frequency_mhz=frequency_mhz,
            )
            y_data = real
            quadrature_real = real if imaginary is not None else None
            quadrature_imaginary = imaginary
            phase_result = {
                "method": "vendor_processed",
                "zero_deg": _finite_float(dictionary, "$PHC0") or 0.0,
                "first_deg": _finite_float(dictionary, "$PHC1") or 0.0,
                "pivot_ppm": float(np.median(x_data)),
                "converged": True,
            }
            phase_source = "vendor_processed"
            source_kind = (
                "jcamp_nmr_spectrum_real_imaginary"
                if imaginary is not None
                else "jcamp_nmr_spectrum_real"
            )
        phase_result["applied_to_quadrature"] = bool(
            is_fid or quadrature_imaginary is not None
        )
        phase_result["manual_adjustments_are_incremental"] = True

        solvent = _clean_text(_first(dictionary, "$SOLVENT", ""))
        reference_ppm = solvent_reference_ppm(solvent, nucleus.value)
        reference_metadata = {
            "solvent": solvent,
            "nucleus": nucleus.value,
            "expected_solvent_reference_ppm": reference_ppm,
            "status": "vendor_metadata_unverified" if reference_ppm is not None else "unavailable",
        }
        default_baseline = (
            {
                "method": "asymmetric_least_squares",
                "smoothness": 1e7,
                "asymmetry": 0.001,
                "iterations": 8,
                "max_fit_points": 8192,
            }
            if baseline_method == "asymmetric_least_squares"
            else {}
        )
        processing_source = build_processing_source(
            x_data,
            y_data,
            quadrature_real_data=quadrature_real,
            quadrature_imaginary_data=quadrature_imaginary,
            source_kind=source_kind,
            source_domain="frequency",
            default_phase=phase_result,
            default_baseline=default_baseline,
            reference_metadata=reference_metadata,
        )
        quality = spectrum_quality_metrics(
            x_data,
            y_data,
            imaginary_data=quadrature_imaginary,
        )

        scans = int(_finite_float(dictionary, ".AVERAGES", "$NS") or 1)
        pulse_program = _clean_text(_first(dictionary, "$PULPROG", ""))
        title = _clean_text(_first(dictionary, "TITLE", path.stem)) or path.stem
        parameters = {
            "vendor": _clean_text(_first(dictionary, "ORIGIN", "JCAMP-DX"))
            or "JCAMP-DX",
            "format": "JCAMP-DX",
            "jcamp_data_type": datatype,
            "nucleus": nucleus.value,
            "frequency_mhz": round(frequency_mhz, 9),
            "solvent": solvent,
            "scans": scans,
            "pulse_program": pulse_program,
            "data_points": len(y_data),
            "source_domain": "time" if is_fid else "frequency",
            "signal_representation": "real",
            "phase_corrected": bool(is_fid or quadrature_imaginary is not None),
            "phase_source": phase_source,
            "auto_phase_zero_deg": round(float(phase_result["zero_deg"]), 6),
            "auto_phase_first_deg": round(float(phase_result["first_deg"]), 6),
            "baseline_corrected": baseline_method != "vendor_processed",
            "baseline_method": baseline_method,
            "quadrature_available": quadrature_imaginary is not None,
            "original_data_preserved": True,
            "processing_pipeline_version": PROCESSING_PIPELINE_VERSION,
            "processing_source": processing_source,
            "processing_source_summary": processing_source_summary(processing_source),
            "processing_quality": quality,
            "processing_history": [],
            "processing_revisions": [],
            "axis_trace": axis_trace,
            "reference_metadata": reference_metadata,
            "digital_filter_applied_points": round(digital_filter_applied, 6),
            "parser_warnings": parser_warnings,
            "jcamp_numeric": inspection,
        }
        return Spectrum(
            technique=Technique.NMR,
            x_data=np.asarray(x_data, dtype=np.float64),
            y_data=np.asarray(y_data, dtype=np.float64),
            x_label=f"Chemical Shift ({nucleus.value})",
            y_label="Intensity",
            x_unit="ppm",
            y_unit="arb. units",
            parameters=parameters,
            metadata=SampleInfo(
                name=title,
                solvent=solvent,
                extra={
                    "vendor": parameters["vendor"],
                    "jcamp_data_type": datatype,
                },
            ),
            source_file=str(path),
        )
