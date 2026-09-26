"""Reader for one-dimensional JEOL Delta JDF NMR data.

The binary layout follows the JEOL reader in NMRGlue (BSD-3-Clause):
https://github.com/jjhelmus/nmrglue/blob/master/nmrglue/fileio/jeol.py
Only the one-dimensional path needed by ChemApp is implemented here.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import numpy as np

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

_ENDIANNESS = {0: "big", 1: "little"}
_DATA_TYPES = {0: "float64", 1: "float32"}
_AXIS_TYPES = {0: None, 1: "real", 2: "tppi", 3: "complex", 4: "real_complex", 5: "envelope"}
_VALUE_TYPES = {0: "string", 1: "integer", 2: "float", 3: "complex", 4: "infinity"}
_BASE_UNITS = {
    0: None, 1: "abundance", 6: "degree", 13: "hertz", 26: "ppm", 28: "second",
    31: "tesla", 38: "generic", 42: "index",
}
_NUCLEUS_NAMES = {
    "proton": NMRNucleus.H1,
    "1h": NMRNucleus.H1,
    "carbon13": NMRNucleus.C13,
    "13c": NMRNucleus.C13,
    "fluorine19": NMRNucleus.F19,
    "19f": NMRNucleus.F19,
    "phosphorus31": NMRNucleus.P31,
    "31p": NMRNucleus.P31,
    "nitrogen15": NMRNucleus.N15,
    "15n": NMRNucleus.N15,
    "silicon29": NMRNucleus.SI29,
    "29si": NMRNucleus.SI29,
}
_MAX_JDF_FILE_BYTES = 512 * 1024 * 1024

_SOLVENT_NAMES = {
    "CHLOROFORM-D": "CDCl3",
    "DMSO-D6": "DMSO",
    "DIMETHYL SULFOXIDE-D6": "DMSO",
    "METHANOL-D4": "CD3OD",
    "BENZENE-D6": "C6D6",
    "DICHLOROMETHANE-D2": "CD2Cl2",
    "ACETONE-D6": "Acetone",
    "DEUTERIUM OXIDE": "D2O",
}


class _Buffer:
    def __init__(self, data: bytes):
        self.data = data
        self.position = 0
        self.prefix = ">"

    def set_endian(self, endian: str) -> None:
        self.prefix = "<" if endian == "little" else ">"

    def _read(self, fmt: str) -> Any:
        size = struct.calcsize(fmt)
        if self.position + size > len(self.data):
            raise ValueError("Unexpected end of JEOL JDF file")
        value = struct.unpack_from(self.prefix + fmt, self.data, self.position)[0]
        self.position += size
        return value

    def u8(self) -> int: return int(self._read("B"))
    def i8(self) -> int: return int(self._read("b"))
    def u16(self) -> int: return int(self._read("H"))
    def i16(self) -> int: return int(self._read("h"))
    def u32(self) -> int: return int(self._read("I"))
    def i32(self) -> int: return int(self._read("i"))
    def f64(self) -> float: return float(self._read("d"))

    def chars(self, count: int) -> str:
        if self.position + count > len(self.data):
            raise ValueError("Unexpected end of JEOL JDF file")
        raw = self.data[self.position:self.position + count]
        self.position += count
        return raw.decode("utf-8", errors="replace").replace("\x00", "").strip()

    def skip(self, count: int) -> None:
        self.position += count

    def array(self, reader: str, count: int) -> list[Any]:
        return [getattr(self, reader)() for _ in range(count)]

    def unit(self, count: int) -> list[tuple[int, int, str | None]]:
        result = []
        for _ in range(count):
            descriptor = self.u8()
            power = descriptor & 0x0F
            base = _BASE_UNITS.get(self.i8())
            result.append((descriptor >> 4, power, base))
        return result


def _parse_header(data: bytes) -> dict[str, Any]:
    b = _Buffer(data)
    identifier = b.chars(8)
    if identifier not in {"JEOL.NMR", "RMN.LOEJ"}:
        raise ValueError("Not a JEOL Delta JDF NMR file")
    endian_code = b.i8()
    if endian_code not in _ENDIANNESS:
        raise ValueError(f"Unsupported JEOL byte-order flag: {endian_code}")

    header: dict[str, Any] = {
        "file_identifier": identifier,
        "endian": _ENDIANNESS[endian_code],
        "major_version": b.u8(),
        "minor_version": b.u16(),
        "dimensions": b.u8(),
    }
    exists = b.u8()
    header["dimension_exists"] = [bool(int(bit)) for bit in format(exists, "08b")]
    info = b.u8()
    data_type_code = info >> 6
    header["data_type"] = _DATA_TYPES.get(data_type_code)
    header["data_format"] = info & 0x3F
    if header["data_type"] is None:
        raise ValueError(f"Unsupported JEOL numeric data type: {data_type_code}")
    header["instrument_code"] = b.i8()
    header["translate"] = b.array("i8", 8)
    header["axis_types"] = [_AXIS_TYPES.get(value) for value in b.array("i8", 8)]
    header["units"] = b.unit(8)
    header["title"] = b.chars(124)
    b.skip(4)  # ranged/listed flags
    header["data_points"] = b.array("u32", 8)
    header["offset_start"] = b.array("u32", 8)
    header["offset_stop"] = b.array("u32", 8)
    header["axis_start"] = b.array("f64", 8)
    header["axis_stop"] = b.array("f64", 8)
    creation = b.array("u8", 4)
    header["creation_date"] = {
        "year": 1990 + (creation[0] >> 1),
        "month": ((creation[0] << 3) & 0x08) + (creation[1] >> 5),
        "day": creation[2] & 0x1F,
    }
    b.skip(4)  # revision date
    header["node_name"] = b.chars(16)
    header["site"] = b.chars(128)
    header["author"] = b.chars(128)
    header["comment"] = b.chars(128)
    header["axis_titles"] = [b.chars(32) for _ in range(8)]
    header["base_freq"] = b.array("f64", 8)
    header["zero_point"] = b.array("f64", 8)
    header["reversed"] = [bool(value) for value in b.array("u8", 8)]
    b.skip(4)
    header["history_used"] = b.u32()
    header["history_length"] = b.u32()
    header["param_start"] = b.u32()
    header["param_length"] = b.u32()
    header["list_start"] = b.array("u32", 8)
    header["list_length"] = b.array("u32", 8)
    header["data_start"] = b.u32()
    header["data_length"] = (b.u32() << 32) | b.u32()
    return header


def _parse_parameters(data: bytes, header: dict[str, Any]) -> dict[str, Any]:
    b = _Buffer(data)
    b.set_endian(header["endian"])
    b.position = header["param_start"]
    _parameter_size = b.u32()
    _low_index = b.u32()
    high_index = b.u32()
    _total_size = b.u32()
    params: dict[str, Any] = {}

    for _ in range(high_index):
        b.skip(4)  # parameter class
        unit_scaler = b.i16()
        b.unit(5)
        b.skip(16)
        value_type_code = b.i32()
        b.position -= 20
        value_type = _VALUE_TYPES.get(value_type_code)
        value: Any = None
        if value_type == "string":
            value = b.chars(16).replace(" ", "")
        elif value_type == "integer":
            value = b.i32()
            b.skip(12)
        elif value_type == "float":
            value = b.f64()
            b.skip(8)
        elif value_type == "complex":
            value = complex(b.f64(), b.f64())
        else:
            b.skip(16)
        b.skip(4)
        name = b.chars(28).lower().replace(" ", "")
        if name:
            params[name] = value * (10 ** unit_scaler) if isinstance(value, (int, float, complex)) else value
    return params


def _read_1d_data(data: bytes, header: dict[str, Any]) -> np.ndarray:
    if header["dimensions"] != 1:
        raise ValueError(f"Only one-dimensional JEOL JDF spectra are supported; found {header['dimensions']}D")
    axis_type = header["axis_types"][0]
    sections = 2 if axis_type in {"complex", "real_complex"} else 1
    count = int(np.prod(header["data_points"])) * sections
    dtype_code = "f8" if header["data_type"] == "float64" else "f4"
    endian = "<" if header["endian"] == "little" else ">"
    start = int(header["data_start"])
    raw = np.frombuffer(data, dtype=np.dtype(endian + dtype_code), count=count, offset=start).astype(np.float64)
    if sections == 2:
        sectioned = raw.reshape(2, -1)
        values = sectioned[0] - 1j * sectioned[1]
    else:
        values = raw
    first = int(header["offset_start"][0])
    last = int(header["offset_stop"][0]) + 1
    return values[first:last]


def _first(params: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        value = params.get(name)
        if value not in (None, ""):
            return value
    return default


def _digital_filter_points(params: dict[str, Any]) -> float:
    """Calculate the fractional JEOL digital-filter group delay."""
    orders = str(params.get("orders") or "")
    factors = str(params.get("factors") or "")
    if not orders or not factors or not orders[0].isdigit():
        return 0.0
    count = int(orders[0])
    if count <= 0 or len(factors) < count or (len(orders) - 1) % count:
        return 0.0
    width = (len(orders) - 1) // count
    factor_values = [int(value) for value in factors[:count] if value.isdigit()]
    if len(factor_values) != count:
        return 0.0
    cursor = 1
    delay = 0.0
    for index in range(count):
        denominator = float(np.prod(factor_values[index:]))
        order = int(orders[cursor:cursor + width])
        cursor += width
        delay += (order - 1) / denominator
    delay /= 2.0
    sweep = float(params.get("x_sweep") or 0.0)
    duration = float(params.get("x_acq_time") or params.get("x_acq_duration") or 0.0)
    points = int(params.get("x_points") or 0)
    if sweep > 0 and duration > 0 and points > 1:
        delay *= (points - 1) / (sweep * duration)
    return float(delay)


def _auto_phase(spectrum: np.ndarray, digital_filter_points: float) -> tuple[np.ndarray, float, float]:
    """Apply JEOL group delay, then use the shared deterministic phase optimizer."""
    size = len(spectrum)
    coordinate = np.arange(size, dtype=np.float64) / max(size, 1)
    first_deg = 360.0 * digital_filter_points
    first_corrected = spectrum * np.exp(1j * np.deg2rad(first_deg * coordinate))
    phased_real, phased_imaginary, phase_result = automatic_phase_correction(
        coordinate,
        np.real(first_corrected),
        np.imag(first_corrected),
        pivot_ppm=0.5,
        optimize_first_order=False,
    )
    phased = phased_real + 1j * phased_imaginary
    return np.real(phased), float(phase_result["zero_deg"]), first_deg


class JEOLJDFParser(BaseParser):
    @classmethod
    def can_parse(cls, file_path: str | Path) -> bool:
        path = Path(file_path)
        if not path.is_file() or path.suffix.lower() != ".jdf":
            return False
        try:
            signature = path.read_bytes()[:8]
        except OSError:
            return False
        return signature in {b"JEOL.NMR", b"RMN.LOEJ"}

    def parse(self, file_path: str | Path) -> Spectrum:
        path = Path(file_path)
        if path.stat().st_size > _MAX_JDF_FILE_BYTES:
            raise ValueError("JEOL JDF file exceeds the 512 MiB safety limit.")
        data = path.read_bytes()
        header = _parse_header(data)
        params = _parse_parameters(data, header)
        values = _read_1d_data(data, header)

        domain = str(_first(params, "x_domain", "x_nucleus", default=header["axis_titles"][0])).lower()
        nucleus = next((value for key, value in _NUCLEUS_NAMES.items() if key in domain), NMRNucleus.H1)
        obs_hz = float(_first(params, "x_freq", "x_frequency", default=header["base_freq"][0] or 0.0))
        frequency_mhz = obs_hz / 1e6 if abs(obs_hz) > 1e5 else obs_hz
        sweep_hz = float(_first(params, "x_sweep", "x_spectral_width", default=0.0))
        offset_ppm = float(_first(params, "x_offset", default=0.0))
        unit = header["units"][0][2]
        phase_zero_deg = 0.0
        phase_first_deg = 0.0
        digital_filter_points = 0.0
        baseline_span = 0.0
        quadrature_real = None
        quadrature_imaginary = None

        if unit == "second" or np.iscomplexobj(values):
            fid = np.asarray(values, dtype=np.complex128)
            if len(fid) > 4:
                fid = fid - np.mean(fid[-max(4, len(fid) // 20):])
            complex_spectrum = np.fft.fftshift(np.fft.fft(fid))
            digital_filter_points = _digital_filter_points(params)
            phased, phase_zero_deg, phase_first_deg = _auto_phase(complex_spectrum, digital_filter_points)
            coordinate = np.arange(len(complex_spectrum), dtype=np.float64) / max(len(complex_spectrum), 1)
            autophased_complex = complex_spectrum * np.exp(
                1j
                * np.deg2rad(
                    phase_first_deg * coordinate + phase_zero_deg
                )
            )
            quadrature_real = np.real(autophased_complex)
            quadrature_imaginary = np.imag(autophased_complex)
            baseline, y_data = asymmetric_least_squares_baseline(
                phased,
                max_fit_points=len(phased),
            )
            baseline_span = float(np.max(baseline) - np.min(baseline))
            if sweep_hz <= 0 and len(fid) > 1:
                dwell = float(header["axis_stop"][0] - header["axis_start"][0]) / (len(fid) - 1)
                sweep_hz = 1.0 / dwell if dwell > 0 else 0.0
            sweep_ppm = sweep_hz / frequency_mhz if frequency_mhz > 0 else 0.0
            # fftshift orders frequency bins from negative to positive. JEOL's
            # x_offset is the carrier position, so ppm increases with the bin
            # index; the viewer applies the conventional reversed NMR display.
            x_data = np.linspace(offset_ppm - sweep_ppm / 2, offset_ppm + sweep_ppm / 2, len(y_data))
            source_domain = "time"
        else:
            # Preserve signed frequency-domain intensity. Taking abs() masks
            # phase/baseline defects and makes negative lobes impossible to
            # diagnose in the workbench.
            y_data = np.real(values).astype(np.float64)
            x_data = np.linspace(float(header["axis_start"][0]), float(header["axis_stop"][0]), len(y_data))
            if header["reversed"][0]:
                x_data = x_data[::-1]
                y_data = y_data[::-1]
            sweep_ppm = abs(float(x_data[-1] - x_data[0])) if len(x_data) > 1 else 0.0
            sweep_hz = sweep_ppm * frequency_mhz
            source_domain = "frequency"

        vendor_solvent = str(_first(params, "solvent", "x_solvent", default=""))
        solvent = _SOLVENT_NAMES.get(vendor_solvent.upper(), vendor_solvent)
        scans = int(_first(params, "scans", "total_scans", "x_scans", default=1))
        pulse_program = str(_first(params, "experiment", "pulse_sequence", default=""))
        sample_name = str(
            _first(
                params,
                "sample_name",
                "sample_id",
                "sampleid",
                default=header["title"],
            )
        )
        reference_ppm = solvent_reference_ppm(solvent, nucleus.value)
        reference_metadata = {
            "solvent": solvent,
            "nucleus": nucleus.value,
            "expected_solvent_reference_ppm": reference_ppm,
            "status": (
                "vendor_metadata_unverified"
                if reference_ppm is not None
                else "unavailable"
            ),
        }
        default_phase = {
            "method": "jeol_group_delay_and_shared_optimizer",
            "zero_deg": phase_zero_deg,
            "first_deg": phase_first_deg,
            "pivot_ppm": float(np.median(x_data)) if len(x_data) else 0.0,
            "applied_to_quadrature": source_domain == "time",
            "manual_adjustments_are_incremental": True,
        }
        processing_source = build_processing_source(
            x_data,
            y_data,
            quadrature_real_data=quadrature_real,
            quadrature_imaginary_data=quadrature_imaginary,
            source_kind=(
                "jeol_time_domain_fft"
                if source_domain == "time"
                else "jeol_frequency_domain"
            ),
            source_domain="frequency",
            default_phase=default_phase,
            default_baseline=(
                {
                    "method": "asymmetric_least_squares",
                    "smoothness": 1e7,
                    "asymmetry": 0.001,
                    "iterations": 8,
                    "max_fit_points": len(y_data),
                }
                if source_domain == "time"
                else {}
            ),
            reference_metadata=reference_metadata,
        )
        quality = spectrum_quality_metrics(
            x_data,
            y_data,
            imaginary_data=quadrature_imaginary,
        )

        parameters = {
            "vendor": "JEOL",
            "format": "JDF",
            "nucleus": nucleus.value,
            "frequency_mhz": round(frequency_mhz, 6),
            "solvent": solvent,
            "scans": scans,
            "pulse_program": pulse_program,
            "spectral_width_ppm": round(sweep_ppm, 6),
            "spectral_width_hz": round(sweep_hz, 3),
            "data_points": len(y_data),
            "source_domain": source_domain,
            "jdf_version": f"{header['major_version']}.{header['minor_version']}",
            "phase_corrected": source_domain == "time",
            "auto_phase_zero_deg": round(phase_zero_deg, 6),
            "auto_phase_first_deg": round(phase_first_deg, 6),
            "digital_filter_points": round(digital_filter_points, 6),
            "baseline_corrected": source_domain == "time",
            "baseline_method": "asymmetric_least_squares" if source_domain == "time" else "none",
            "baseline_span": round(baseline_span, 6),
            "signal_representation": "real",
            "quadrature_available": quadrature_imaginary is not None,
            "original_data_preserved": True,
            "processing_pipeline_version": PROCESSING_PIPELINE_VERSION,
            "processing_source": processing_source,
            "processing_source_summary": processing_source_summary(processing_source),
            "processing_quality": quality,
            "reference_metadata": reference_metadata,
            "processing_history": [],
            "processing_revisions": [],
        }
        metadata = SampleInfo(
            name=sample_name or path.stem,
            solvent=solvent,
            extra={
                "vendor": "JEOL",
                "author": header["author"],
                "site": header["site"],
                "comment": header["comment"],
                "vendor_solvent": vendor_solvent,
                "creation_date": header["creation_date"],
            },
        )
        return Spectrum(
            technique=Technique.NMR,
            x_data=np.asarray(x_data, dtype=np.float64),
            y_data=np.asarray(y_data, dtype=np.float64),
            x_label=f"Chemical Shift ({nucleus.value})",
            y_label="Intensity",
            x_unit="ppm",
            y_unit="arb. units",
            parameters=parameters,
            metadata=metadata,
            source_file=str(path),
        )
