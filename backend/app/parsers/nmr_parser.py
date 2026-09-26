from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

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
from app.core.models import NMRNucleus, Peak, SampleInfo, Spectrum, Technique
from app.parsers.base import BaseParser

NUCLEUS_MAP = {
    "1H": NMRNucleus.H1,
    "13C": NMRNucleus.C13,
    "19F": NMRNucleus.F19,
    "31P": NMRNucleus.P31,
    "15N": NMRNucleus.N15,
    "29Si": NMRNucleus.SI29,
}


def _parse_jcamp_params(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("$$"):
            continue
        match = re.match(r"##\$(\w+)=\s*(.*)", line)
        if match:
            key = match.group(1)
            value = match.group(2).strip()
            result[key] = re.sub(r"<|>", "", value)
    return result


def _read_parameter_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    return _parse_jcamp_params(path.read_text(encoding="utf-8", errors="replace"))


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


def _bruker_dtype(byte_order: int, data_type: int) -> np.dtype:
    endian = ">" if int(byte_order) == 1 else "<"
    if int(data_type) == 0:
        return np.dtype(f"{endian}i4")
    if int(data_type) == 1:
        return np.dtype(f"{endian}f4")
    if int(data_type) == 2:
        return np.dtype(f"{endian}f8")
    raise ValueError(f"Unsupported Bruker data type code: {data_type}")


_MAX_BINARY_BYTES = 512 * 1024 * 1024


def _read_bruker_binary(
    path: Path,
    *,
    byte_order: int,
    data_type: int,
    count: int = -1,
) -> np.ndarray:
    if path.stat().st_size > _MAX_BINARY_BYTES:
        raise ValueError("Bruker binary file exceeds the 512 MiB safety limit.")
    dtype = _bruker_dtype(byte_order, data_type)
    values = np.fromfile(str(path), dtype=dtype, count=count)
    return values.astype(np.float64, copy=False)


def _decode_bruker_fid(
    raw_values: np.ndarray,
    *,
    td: int,
    scale_exponent: int = 0,
) -> np.ndarray:
    """Decode interleaved R/I acquisition values into complex samples."""

    raw = np.asarray(raw_values)
    valid_values = min(max(int(td), 0), len(raw))
    valid_values -= valid_values % 2
    if valid_values < 2:
        raise ValueError("Bruker FID contains fewer than one complex point")
    raw = raw[:valid_values].astype(np.float64, copy=False)
    if scale_exponent:
        raw = raw * float(2.0 ** int(scale_exponent))
    return raw[0::2] + 1j * raw[1::2]


def _remove_digital_filter(fid: np.ndarray, group_delay: float) -> tuple[np.ndarray, int, float]:
    """Approximate Bruker's digital-filter removal from GRPDLY.

    This follows the open NMRGlue/NMRPipe-compatible procedure: apply the
    frequency shift, fold the filter tail into the start of the FID, then
    discard the affected tail. The group delay is intentionally truncated,
    which generally produces a cleaner 1D spectrum than a fractional shift.
    """

    values = np.asarray(fid, dtype=np.complex128)
    delay = float(group_delay)
    if not np.isfinite(delay) or delay <= 0:
        return values.copy(), 0, 0.0
    phase = min(int(np.floor(delay)), max(len(values) - 3, 0))
    fractional = delay - phase
    size = len(values)

    # Positive-exponential inverse/forward transforms in NMR ordering.
    inverse = np.fft.fft(np.fft.ifftshift(values)) / max(size, 1)
    shifted = inverse * np.exp(
        2j * np.pi * phase * np.arange(size, dtype=np.float64) / max(size, 1)
    )
    corrected = np.fft.fftshift(np.fft.ifft(shifted)) * size

    skip = min(int(np.floor(phase + 2.0)), max(size - 1, 0))
    add = int(max(skip - 6, 0))
    if add:
        corrected[:add] += corrected[: -(add + 1) : -1]
    if skip:
        corrected = corrected[:-skip]
    return np.asarray(corrected, dtype=np.complex128), phase, fractional


def _auto_phase_zero_order(spectrum: np.ndarray) -> tuple[np.ndarray, float]:
    """Choose zero-order phase through the shared processing-v2 optimizer."""

    complex_spectrum = np.asarray(spectrum, dtype=np.complex128)
    if len(complex_spectrum) == 0:
        return complex_spectrum.copy(), 0.0
    coordinate = np.arange(len(complex_spectrum), dtype=np.float64)
    phased_real, phased_imaginary, result = automatic_phase_correction(
        coordinate,
        np.real(complex_spectrum),
        np.imag(complex_spectrum),
        optimize_first_order=False,
    )
    return (
        phased_real + 1j * phased_imaginary,
        float(result["zero_deg"]),
    )


class NMRSpectrumParser(BaseParser):
    @classmethod
    def can_parse(cls, file_path: str | Path) -> bool:
        path = Path(file_path)
        return path.is_dir() and (path / "acqu").exists() and (path / "fid").exists()

    def parse(self, file_path: str | Path) -> Spectrum:
        base_dir = Path(file_path)

        # acqus/procs are status files and describe the bytes actually written.
        # Merge them over the editable acquisition parameters when available.
        acqu_params = _read_parameter_file(base_dir / "acqu")
        acqu_params.update(_read_parameter_file(base_dir / "acqus"))
        procs_params = _read_parameter_file(base_dir / "pdata" / "1" / "procs")

        bf1 = _float_param(acqu_params, "BF1", 400.15)
        sfo1 = _float_param(acqu_params, "SFO1", bf1)
        sw_ppm = _float_param(acqu_params, "SW", 20.48)
        sw_h = _float_param(acqu_params, "SW_h", sw_ppm * max(sfo1, 1.0))
        td = max(2, _int_param(acqu_params, "TD", 65536))
        ns = max(1, _int_param(acqu_params, "NS", 1))
        nucleus_str = acqu_params.get("NUC1", "1H").strip()
        solvent = acqu_params.get("SOLVENT", "")
        pulse_program = acqu_params.get("PULPROG", "")
        byte_order_acq = _int_param(acqu_params, "BYTORDA", 0)
        data_type_acq = _int_param(acqu_params, "DTYPA", 0)
        nc_acq = _int_param(acqu_params, "NC", 0)
        aq_mod = _int_param(acqu_params, "AQ_mod", 3)
        group_delay = _float_param(acqu_params, "GRPDLY", -1.0)
        decimation = _int_param(acqu_params, "DECIM", 0)
        dsp_firmware = _int_param(acqu_params, "DSPFVS", 0)

        nucleus = NUCLEUS_MAP.get(nucleus_str, NMRNucleus.H1)
        sf = _float_param(procs_params, "SF", bf1)
        sw_p = _float_param(procs_params, "SW_p", sw_h)
        si = max(2, _int_param(procs_params, "SI", td // 2))
        if "O1P" in acqu_params:
            carrier_ppm = _float_param(acqu_params, "O1P", 0.0)
        else:
            # Older acqus files often store only O1 in Hz.
            carrier_ppm = _float_param(acqu_params, "O1", 0.0) / max(
                sfo1, np.finfo(float).eps
            )
        offset_default = carrier_ppm + (sw_p / max(sf, np.finfo(float).eps)) / 2.0
        offset_ppm = _float_param(procs_params, "OFFSET", offset_default)
        byte_order_proc = _int_param(procs_params, "BYTORDP", 0)
        data_type_proc = _int_param(procs_params, "DTYPP", 0)
        nc_proc = _int_param(procs_params, "NC_proc", 0)
        phase_mode = _int_param(procs_params, "PH_mod", -1)
        baseline_mode = _int_param(procs_params, "BC_mod", -1)

        x_data = offset_ppm - np.arange(si, dtype=np.float64) * (sw_p / sf) / si
        processed_path = base_dir / "pdata" / "1" / "1r"
        imaginary_path = base_dir / "pdata" / "1" / "1i"
        parser_warnings: list[str] = []

        if processed_path.exists():
            real = _read_bruker_binary(
                processed_path,
                byte_order=byte_order_proc,
                data_type=data_type_proc,
            )
            if data_type_proc == 0 and nc_proc:
                real = real * float(2.0 ** nc_proc)
            real = self._fit_length(real, si)

            imaginary = None
            if imaginary_path.exists():
                imaginary = _read_bruker_binary(
                    imaginary_path,
                    byte_order=byte_order_proc,
                    data_type=data_type_proc,
                )
                if data_type_proc == 0 and nc_proc:
                    imaginary = imaginary * float(2.0 ** nc_proc)
                imaginary = self._fit_length(imaginary, si)
            else:
                parser_warnings.append(
                    "Processed imaginary channel 1i is absent; manual phase correction is unavailable."
                )

            y_data = real
            quadrature_real = real if imaginary is not None else None
            quadrature_imaginary = imaginary
            source_kind = "bruker_processed_1r_1i" if imaginary is not None else "bruker_processed_1r"
            phase_zero = _float_param(procs_params, "PHC0", 0.0)
            phase_first = _float_param(procs_params, "PHC1", 0.0)
            group_delay_applied = 0.0
            if baseline_mode > 0:
                baseline_method = f"vendor_processed_mode_{baseline_mode}"
            elif baseline_mode == 0:
                baseline_method = "none"
            else:
                baseline_method = "unknown_vendor_processed"
            phase_corrected = phase_mode > 0
        else:
            raw_values = _read_bruker_binary(
                base_dir / "fid",
                byte_order=byte_order_acq,
                data_type=data_type_acq,
            )
            fid = _decode_bruker_fid(
                raw_values,
                td=td,
                scale_exponent=nc_acq if data_type_acq == 0 else 0,
            )
            corrected_fid, removed_points, fractional_delay = _remove_digital_filter(
                fid,
                group_delay,
            )
            # Bruker processed spectra are stored from high to low ppm.
            complex_spectrum = np.fft.fftshift(
                np.fft.fft(corrected_fid, n=si)
            )[::-1]
            phased_complex, phase_zero = _auto_phase_zero_order(complex_spectrum)
            _, y_data = asymmetric_least_squares_baseline(np.real(phased_complex))
            quadrature_real = np.real(phased_complex)
            quadrature_imaginary = np.imag(phased_complex)
            source_kind = "bruker_raw_fid_fft"
            phase_first = 0.0
            group_delay_applied = float(removed_points + fractional_delay)
            baseline_method = "asymmetric_least_squares"
            phase_corrected = True
            if group_delay < 0:
                parser_warnings.append(
                    "GRPDLY is unavailable; raw FID was transformed without digital-filter correction."
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
            "method": (
                "vendor_processed"
                if processed_path.exists()
                else "shared_automatic_zero_order"
            ),
            "zero_deg": float(phase_zero),
            "first_deg": float(phase_first),
            "pivot_ppm": float(np.median(x_data)) if len(x_data) else 0.0,
            "applied_to_quadrature": bool(
                not processed_path.exists() or phase_corrected
            ),
            "manual_adjustments_are_incremental": True,
        }
        processing_source = build_processing_source(
            x_data,
            y_data,
            quadrature_real_data=quadrature_real,
            quadrature_imaginary_data=quadrature_imaginary,
            source_kind=source_kind,
            source_domain="frequency",
            default_phase=default_phase,
            default_baseline=(
                {
                    "method": "asymmetric_least_squares",
                    "smoothness": 1e7,
                    "asymmetry": 0.001,
                    "iterations": 8,
                    "max_fit_points": 8192,
                }
                if baseline_method == "asymmetric_least_squares"
                else {}
            ),
            reference_metadata=reference_metadata,
        )
        processing_quality = spectrum_quality_metrics(
            x_data,
            y_data,
            imaginary_data=quadrature_imaginary,
        )

        sw_ppm_calc = sw_p / sf
        peaks = self._parse_peaklist(base_dir)
        parameters = {
            "vendor": "Bruker",
            "format": "TopSpin",
            "nucleus": nucleus.value,
            "frequency_mhz": round(bf1, 2),
            "solvent": solvent,
            "scans": ns,
            "pulse_program": pulse_program,
            "spectral_width_ppm": round(sw_ppm_calc, 6),
            "spectral_width_hz": round(sw_p, 6),
            "transmitter_frequency_mhz": round(sfo1, 9),
            "data_points": si,
            "digital_resolution_hz_per_pt": round(sw_p / si, 6),
            "source_domain": "frequency" if processed_path.exists() else "time",
            "signal_representation": "real",
            "phase_corrected": phase_corrected,
            "phase_source": "vendor_processed" if processed_path.exists() else "automatic_zero_order",
            "auto_phase_zero_deg": round(float(phase_zero), 6),
            "auto_phase_first_deg": round(float(phase_first), 6),
            "baseline_corrected": baseline_mode > 0 if processed_path.exists() else True,
            "baseline_method": baseline_method,
            "quadrature_available": quadrature_imaginary is not None,
            "original_data_preserved": True,
            "processing_pipeline_version": PROCESSING_PIPELINE_VERSION,
            "processing_source": processing_source,
            "processing_source_summary": processing_source_summary(processing_source),
            "processing_quality": processing_quality,
            "reference_metadata": reference_metadata,
            "processing_history": [],
            "processing_revisions": [],
            "bruker_parameters": {
                "AQ_mod": aq_mod,
                "BYTORDA": byte_order_acq,
                "DTYPA": data_type_acq,
                "BYTORDP": byte_order_proc,
                "DTYPP": data_type_proc,
                "NC": nc_acq,
                "NC_proc": nc_proc,
                "PH_mod": phase_mode,
                "BC_mod": baseline_mode,
                "GRPDLY": group_delay,
                "group_delay_applied_points": round(group_delay_applied, 6),
                "DECIM": decimation,
                "DSPFVS": dsp_firmware,
            },
            "parser_warnings": parser_warnings,
        }

        metadata = SampleInfo(
            solvent=solvent,
            extra={"nucleus": nucleus.value, "frequency_mhz": bf1, "vendor": "Bruker"},
        )
        return Spectrum(
            technique=Technique.NMR,
            x_data=x_data,
            y_data=np.asarray(y_data, dtype=np.float64),
            x_label=f"Chemical Shift ({nucleus.value})",
            y_label="Intensity",
            x_unit="ppm",
            y_unit="arb. units",
            parameters=parameters,
            metadata=metadata,
            peaks=peaks,
            source_file=str(base_dir),
        )

    @staticmethod
    def _fit_length(values: np.ndarray, size: int) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if len(array) < size:
            return np.pad(array, (0, size - len(array)))
        return array[:size]

    def _parse_peaklist(self, base_dir: Path) -> list[Peak]:
        xml_path = base_dir / "pdata" / "1" / "peaklist.xml"
        if not xml_path.exists():
            return []

        peaks: list[Peak] = []
        try:
            root = ET.parse(str(xml_path)).getroot()
            for elem in root.iter("Peak1D"):
                position = elem.get("F1", "")
                intensity = elem.get("intensity", "")
                if position and intensity:
                    peaks.append(
                        Peak(position=float(position), intensity=float(intensity))
                    )
        except (ET.ParseError, OSError, ValueError):
            return []
        return peaks
