from __future__ import annotations

import os
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from app.core.models import SampleInfo, Spectrum, Technique
from app.parsers.base import BaseParser

ACMD_NS = {"a": "urn:schemas-agilent-com:acmd20"}
CHANNEL_COLORS = ["#06b6d4", "#f59e0b", "#10b981", "#ef4444", "#8b5cf6"]
_MAX_XML_BYTES = int(os.environ.get("CHEMAPP_HPLC_MAX_XML_BYTES", str(20 * 1024 * 1024)))
_MAX_TRACE_BYTES = int(os.environ.get("CHEMAPP_HPLC_MAX_TRACE_BYTES", str(128 * 1024 * 1024)))
_MAX_COMPRESSION_RATIO = float(os.environ.get("CHEMAPP_MAX_ZIP_COMPRESSION_RATIO", "200"))
_MAX_HPLC_CHANNELS = 64
_MAX_HPLC_TOTAL_POINTS = 16_000_000


class HPLCParser(BaseParser):
    @staticmethod
    def _safe_zip_read(zf: zipfile.ZipFile, name: str, max_bytes: int) -> bytes:
        info = zf.getinfo(name)
        if info.file_size > max_bytes:
            raise ValueError(f"HPLC archive member is too large: {name}")
        if info.file_size and (
            info.compress_size <= 0
            or info.file_size / max(info.compress_size, 1) > _MAX_COMPRESSION_RATIO
        ):
            raise ValueError(f"Suspicious HPLC archive compression ratio: {name}")
        output = bytearray()
        with zf.open(info, "r") as source:
            while True:
                chunk = source.read(min(1024 * 1024, max_bytes + 1 - len(output)))
                if not chunk:
                    break
                output.extend(chunk)
                if len(output) > max_bytes:
                    raise ValueError(f"HPLC archive member expanded beyond its limit: {name}")
        return bytes(output)

    @staticmethod
    def _parse_xml(xml_text: str) -> ET.Element:
        lowered = xml_text.lower()
        if "<!doctype" in lowered or "<!entity" in lowered:
            raise ValueError("DTD and entity declarations are not allowed in HPLC XML")
        return ET.fromstring(xml_text)

    @classmethod
    def can_parse(cls, file_path: str | Path) -> bool:
        p = Path(file_path)
        if not p.is_file():
            return False
        if p.suffix.lower() != ".dx":
            return False
        try:
            if not zipfile.is_zipfile(str(p)):
                return False
            with zipfile.ZipFile(str(p)) as zf:
                names = zf.namelist()
                has_ch = any(n.endswith(".CH") for n in names)
                has_acmd = "injection.acmd" in names
                has_uv = any(n.endswith(".UV") for n in names)
                return has_ch and has_acmd and has_uv
        except Exception:
            return False

    def parse(self, file_path: str | Path) -> Spectrum:
        p = Path(file_path)

        with zipfile.ZipFile(str(p)) as zf:
            signals = self._parse_acmd(zf)
            chrom_signals = [s for s in signals if s.get("IsIntegrable", False)]
            # Per-member ZIP limits cannot bound repeated references to one
            # trace. Budget the retained arrays across the whole sample before
            # reading any chromatograms, including duplicate trace references.
            if len(chrom_signals) > _MAX_HPLC_CHANNELS:
                raise ValueError("HPLC integrable channel count exceeds the safety limit")
            total_points = 0
            for sig in chrom_signals:
                points = sig.get("NumberOfValues", 0)
                if points < 1:
                    raise ValueError("HPLC channel point count must be positive")
                total_points += points
                if total_points > _MAX_HPLC_TOTAL_POINTS:
                    raise ValueError("HPLC aggregate decoded points exceed the safety limit")
            signal_results = self._parse_instrument_results(p)

            channels_data = []
            channel_signals: list[dict] = []
            for i, sig in enumerate(chrom_signals):
                trace_id = sig.get("TraceId", "")
                ch_data = self._read_chromatogram(zf, trace_id, sig)
                if ch_data is None:
                    continue

                desc = sig.get("Description", "")
                wl = self._extract_wavelength(desc)
                ch_name = sig.get("ChannelName", "")

                channels_data.append({
                    "name": ch_name,
                    "wavelength_nm": wl,
                    "description": desc,
                    "y_data": ch_data["intensity_mau"],
                    "color": CHANNEL_COLORS[i % len(CHANNEL_COLORS)],
                    "instrument_peaks": signal_results.get(ch_name, []),
                })
                channel_signals.append(sig)

            if not channels_data:
                raise ValueError("No integrable channels found in HPLC data")

            # channels_data[0] may belong to a later signal when earlier
            # signals have no readable trace; the time axis and header
            # parameters must come from that primary channel's own signal.
            primary = channels_data[0]
            primary_signal = channel_signals[0]
            time_data = self._compute_time_axis(primary_signal)

            parameters = {
                "channels": channels_data,
                "n_channels": len(channels_data),
                "time_start_ms": float(primary_signal.get("TimeStart", 0)),
                "time_end_ms": float(primary_signal.get("TimeEnd", 0)),
                "units": primary_signal.get("Units", "mAU"),
                "device_name": primary_signal.get("DeviceName", ""),
                "instrument_result_source": str(self._sibling_result_file(p)) if signal_results else "",
            }

            wl_labels = [f"{ch['name']} ({ch['wavelength_nm']}nm)" for ch in channels_data]
            name = " / ".join(wl_labels)

            return Spectrum(
                technique=Technique.HPLC,
                x_data=time_data,
                y_data=primary["y_data"],
                x_label="Time",
                y_label="Absorbance",
                x_unit="min",
                y_unit="mAU",
                parameters=parameters,
                metadata=SampleInfo(name=name),
                source_file=str(p),
            )

    @staticmethod
    def _parse_acmd(zf: zipfile.ZipFile) -> list[dict]:
        xml_text = HPLCParser._safe_zip_read(
            zf,
            "injection.acmd",
            _MAX_XML_BYTES,
        ).decode("utf-8-sig", errors="replace")
        root = HPLCParser._parse_xml(xml_text)

        signals: list[dict] = []
        for sig_elem in root.findall(".//a:Signal", ACMD_NS):
            sig: dict = {}
            for tag in [
                "Encoding", "TraceId", "DeviceName", "DeviceNumber",
                "ChannelName", "Description", "TimeStart", "TimeEnd",
                "Minimum", "Maximum", "Slope", "NumberOfValues",
                "DetectorType", "ScaleFactor", "Units",
                "NumberOfRecords", "IsIntegrable",
            ]:
                el = sig_elem.find(f"a:{tag}", ACMD_NS)
                if el is not None and el.text:
                    sig[tag] = el.text

            try:
                sig["TimeStart"] = float(sig.get("TimeStart", 0))
                sig["TimeEnd"] = float(sig.get("TimeEnd", 0))
                sig["NumberOfValues"] = int(float(sig.get("NumberOfValues", 0)))
                sig["NumberOfRecords"] = int(float(sig.get("NumberOfRecords", 0)))
                sig["Slope"] = float(sig.get("Slope", 1))
                sig["Maximum"] = float(sig.get("Maximum", 0))
                sig["Minimum"] = float(sig.get("Minimum", 0))
                sig["ScaleFactor"] = float(sig.get("ScaleFactor", 0))
                sig["DetectorType"] = int(float(sig.get("DetectorType", 0)))
            except (TypeError, ValueError):
                # One malformed signal must not take down the whole run.
                continue
            sig["IsIntegrable"] = sig.get("IsIntegrable", "false").lower() == "true"
            signals.append(sig)
        return signals

    @classmethod
    def _parse_instrument_results(cls, dx_path: Path) -> dict[str, list[dict[str, Any]]]:
        rx_path = cls._sibling_result_file(dx_path)
        if not rx_path or not rx_path.exists() or not zipfile.is_zipfile(str(rx_path)):
            return {}

        acaml_path = cls._matching_sidecar(dx_path, ".acaml")
        signal_map = cls._parse_signal_map(acaml_path)
        try:
            with zipfile.ZipFile(str(rx_path)) as zf:
                xml_text = cls._safe_zip_read(
                    zf,
                    "Base/InjectionACAML",
                    _MAX_XML_BYTES,
                ).decode("utf-8-sig", errors="replace")
        except Exception:
            return {}

        try:
            root = cls._parse_xml(xml_text)
        except (ET.ParseError, ValueError):
            return {}

        compound_names = cls._parse_peak_compound_names(root)
        channel_peaks: dict[str, list[dict[str, Any]]] = {}
        for signal_result in root.iter():
            if cls._local_name(signal_result.tag) != "SignalResult":
                continue

            signal_id = ""
            for child in signal_result:
                if cls._local_name(child.tag) == "Signal_ID":
                    signal_id = child.attrib.get("id", "")
                    break

            signal_meta = signal_map.get(signal_id)
            if not signal_meta:
                continue
            channel_name = signal_meta.get("name", "")
            if not channel_name:
                continue

            peaks = [
                cls._parse_peak_node(peak, channel_name, compound_names)
                for peak in signal_result
                if cls._local_name(peak.tag) == "Peak"
            ]
            peaks = [peak for peak in peaks if peak]
            if peaks:
                peaks.sort(key=lambda peak: peak["position"])
                channel_peaks[channel_name] = peaks

        return channel_peaks

    @staticmethod
    def _matching_sidecar(dx_path: Path, suffix: str) -> Path | None:
        exact = dx_path.with_suffix(suffix)
        if exact.is_file():
            return exact
        matches = [
            path for path in dx_path.parent.iterdir()
            if path.is_file()
            and path.stem.casefold() == dx_path.stem.casefold()
            and path.suffix.casefold() == suffix.casefold()
        ]
        # Never associate another run's results, or choose among ambiguous
        # case variants on a case-sensitive filesystem.
        return matches[0] if len(matches) == 1 else None

    @classmethod
    def _sibling_result_file(cls, dx_path: Path) -> Path | None:
        return cls._matching_sidecar(dx_path, ".rx")

    @classmethod
    def _parse_signal_map(cls, acaml_path: Path | None) -> dict[str, dict[str, str]]:
        if acaml_path is None or not acaml_path.exists():
            return {}
        try:
            if acaml_path.stat().st_size > _MAX_XML_BYTES:
                return {}
            xml_text = acaml_path.read_text(encoding="utf-8-sig", errors="replace")
            root = cls._parse_xml(xml_text)
        except Exception:
            return {}

        signals: dict[str, dict[str, str]] = {}
        for sig_elem in root.iter():
            if cls._local_name(sig_elem.tag) != "Signal":
                continue
            sig_id = sig_elem.attrib.get("id", "")
            name = cls._child_text(sig_elem, "Name")
            signal_type = cls._child_text(sig_elem, "Type")
            if not sig_id or signal_type != "Chromatogram" or not name:
                continue
            signals[sig_id] = {
                "name": name,
                "description": cls._child_text(sig_elem, "Description"),
                "trace_id": cls._child_text(sig_elem, "TraceID"),
            }
        return signals

    @classmethod
    def _parse_peak_compound_names(cls, root: ET.Element) -> dict[str, str]:
        names: dict[str, str] = {}
        for compound in root.iter():
            if cls._local_name(compound.tag) != "InjectionCompound":
                continue
            compound_name = cls._child_text(compound, "CompoundName").strip()
            if not compound_name:
                continue
            for child in compound.iter():
                if cls._local_name(child.tag) == "Peak_ID":
                    peak_id = child.attrib.get("id", "")
                    if peak_id:
                        names[peak_id] = compound_name
        return names

    @classmethod
    def _parse_peak_node(
        cls,
        peak: ET.Element,
        channel_name: str,
        compound_names: dict[str, str],
    ) -> dict[str, Any]:
        peak_id = peak.attrib.get("id", "")
        baseline_code = cls._child_text(peak, "BaselineCode").strip()
        begin = cls._child_val(peak, "BeginTime")
        end = cls._child_val(peak, "EndTime")
        return {
            "id": peak_id,
            "channel": channel_name,
            "position": round(cls._child_val(peak, "RetentionTime"), 4),
            "retention_time": cls._child_val(peak, "RetentionTime"),
            "intensity": round(cls._child_val(peak, "Height"), 4),
            "height": cls._child_val(peak, "Height"),
            "width": round(cls._child_val(peak, "WidthBase"), 4),
            "area": round(cls._child_val(peak, "Area"), 2),
            "area_percent": round(cls._child_val(peak, "AreaPercent"), 2),
            "height_percent": round(cls._child_val(peak, "HeightPercent"), 2),
            "type": baseline_code,
            "peak_type": cls._child_text(peak, "Type"),
            "name": compound_names.get(peak_id, ""),
            "begin_time": round(begin, 4),
            "end_time": round(end, 4),
            "baseline_start": round(cls._child_val(peak, "BaselineStart"), 4),
            "baseline_end": round(cls._child_val(peak, "BaselineEnd"), 4),
            "level_start": round(cls._child_val(peak, "LevelStart"), 4),
            "level_end": round(cls._child_val(peak, "LevelEnd"), 4),
        }

    @classmethod
    def _child_text(cls, elem: ET.Element, child_name: str) -> str:
        for child in elem:
            if cls._local_name(child.tag) == child_name:
                return child.text or ""
        return ""

    @classmethod
    def _child_val(cls, elem: ET.Element, child_name: str) -> float:
        for child in elem:
            if cls._local_name(child.tag) == child_name:
                try:
                    return float(child.attrib.get("val", "0"))
                except ValueError:
                    return 0.0
        return 0.0

    @staticmethod
    def _local_name(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    @staticmethod
    def _read_chromatogram(zf: zipfile.ZipFile, trace_id: str, signal_meta: dict) -> dict | None:
        entry_name = f"{trace_id}.CH"
        try:
            data = HPLCParser._safe_zip_read(zf, entry_name, _MAX_TRACE_BYTES)
        except (KeyError, ValueError):
            return None

        n_vals = signal_meta.get("NumberOfValues", 0)
        slope = signal_meta.get("Slope", 1.0)
        data_offset = 6144

        raw = np.frombuffer(data[data_offset:data_offset + n_vals * 8], dtype=np.float64)
        if len(raw) != n_vals:
            return None

        return {
            "intensity_mau": raw * slope,
            "n_points": n_vals,
        }

    @staticmethod
    def _compute_time_axis(signal_meta: dict) -> np.ndarray:
        t_start_ms = signal_meta.get("TimeStart", 0)
        t_end_ms = signal_meta.get("TimeEnd", 360000)
        n_vals = signal_meta.get("NumberOfValues", 900)
        return np.linspace(t_start_ms, t_end_ms, n_vals) / 60000.0

    @staticmethod
    def _extract_wavelength(description: str) -> float | None:
        import re
        m = re.search(r"Sig=(\d+\.?\d*)", description)
        if m:
            return float(m.group(1))
        return None
