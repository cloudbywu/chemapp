from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from app.core.models import Technique
from app.parsers.hplc_parser import HPLCParser
from app.analysis.hplc_analysis import HPLCAnalyzer

DATA_DIR = Path(__file__).parent.parent.parent / "dataexample" / "液相色谱example"

W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


def _docx_paragraphs(path: Path) -> list[str]:
    with ZipFile(path) as zf:
        root = ET.fromstring(zf.read("word/document.xml"))
    paras = []
    for para in root.findall(".//w:p", W_NS):
        text = "".join(t.text or "" for t in para.findall(".//w:t", W_NS)).strip()
        if text:
            paras.append(text)
    return paras


def _parse_report_peak_tables(path: Path) -> dict[str, list[dict]]:
    paras = _docx_paragraphs(path)
    tables: dict[str, list[dict]] = {}
    current_channel: str | None = None
    i = 0
    while i < len(paras):
        line = paras[i]
        if line.startswith("DAD1"):
            current_channel = line.split(",", 1)[0]
            tables.setdefault(current_channel, [])
            i += 1
            continue
        if not current_channel:
            i += 1
            continue

        if line.replace(".", "", 1).isdigit() and i + 5 < len(paras):
            parts = paras[i:i + 7]
            if not parts[1].isalpha():
                i += 1
                continue
            name = parts[6].strip() if len(parts) > 6 else ""
            if name == "总和" or name.replace(".", "", 1).isdigit():
                name = ""
            tables[current_channel].append({
                "position": float(parts[0]),
                "type": parts[1].strip(),
                "width": float(parts[2]),
                "area": float(parts[3]),
                "height": float(parts[4]),
                "area_percent": float(parts[5]),
                "name": name,
            })
            i += 7 if tables[current_channel][-1]["name"] else 6
            continue

        i += 1
    return tables


def test_can_parse():
    assert HPLCParser.can_parse(DATA_DIR / "-S-001.sirslt" / "-S-001.dx") is True
    assert HPLCParser.can_parse(DATA_DIR / "-S-001.sirslt" / "-S-001.acaml") is False


def test_parse():
    parser = HPLCParser()
    spectrum = parser.parse(DATA_DIR / "-S-001.sirslt" / "-S-001.dx")

    assert spectrum.technique == Technique.HPLC
    assert spectrum.x_label == "Time"
    assert spectrum.y_label == "Absorbance"
    assert spectrum.x_unit == "min"
    assert spectrum.y_unit == "mAU"
    assert spectrum.num_points == 900

    channels = spectrum.parameters.get("channels", [])
    assert len(channels) >= 2
    ch_names = {ch["name"] for ch in channels}
    assert "DAD1A" in ch_names
    assert "DAD1B" in ch_names
    wls = {ch["wavelength_nm"] for ch in channels}
    assert 220.0 in wls
    assert 254.0 in wls
    assert "y_data" in channels[0]
    assert any(ch.get("instrument_peaks") for ch in channels)


def test_parse_multiple():
    parser = HPLCParser()
    for sample_num in ["001", "002", "003", "004", "005"]:
        path = DATA_DIR / f"-S-{sample_num}.sirslt" / f"-S-{sample_num}.dx"
        if path.exists():
            spectrum = parser.parse(path)
            assert spectrum.technique == Technique.HPLC
            assert len(spectrum.parameters.get("channels", [])) >= 2


def test_analyze():
    parser = HPLCParser()
    spectrum = parser.parse(DATA_DIR / "-S-001.sirslt" / "-S-001.dx")
    analyzer = HPLCAnalyzer()
    result = analyzer.analyze(spectrum)

    assert result.technique == Technique.HPLC
    assert len(result.peaks) > 0
    assert len(result.summary) > 0
    assert "DAD1A" in result.summary
    assert "DAD1B" in result.summary
    assert result.metrics["integration_source"] == "instrument_record"


def test_peak_detection():
    parser = HPLCParser()
    spectrum = parser.parse(DATA_DIR / "-S-001.sirslt" / "-S-001.dx")
    analyzer = HPLCAnalyzer()
    result = analyzer.analyze(spectrum)

    if len(result.peaks) > 0:
        main_peak = result.peaks[0]
        assert 1.0 < main_peak.position < 5.0
        assert main_peak.intensity > 0
        assert main_peak.area is not None
        assert main_peak.area > 0


def test_hplc_instrument_results_match_docx_report():
    parser = HPLCParser()
    spectrum = parser.parse(DATA_DIR / "-S-001.sirslt" / "-S-001.dx")
    result = HPLCAnalyzer().analyze(spectrum)
    report_tables = _parse_report_peak_tables(DATA_DIR / "-S-001.sirslt" / "-S-001_1.docx")
    channel_peaks = result.metrics["channel_peaks"]

    assert sorted(report_tables) == ["DAD1A", "DAD1B"]
    for channel_name, expected_peaks in report_tables.items():
        actual_peaks = channel_peaks[channel_name]["peaks"]
        assert len(actual_peaks) == len(expected_peaks)
        for actual, expected in zip(actual_peaks, expected_peaks):
            assert actual["position"] == pytest.approx(expected["position"], abs=0.001)
            assert actual["type"] == expected["type"]
            assert actual["width"] == pytest.approx(expected["width"], abs=0.01)
            assert actual["area"] == pytest.approx(expected["area"], abs=0.01)
            assert actual["intensity"] == pytest.approx(expected["height"], abs=0.01)
            assert actual["area_percent"] == pytest.approx(expected["area_percent"], abs=0.01)
            assert actual["name"] == expected["name"]


_ACMD_XML = (
    '<?xml version="1.0" encoding="utf-8"?>'
    '<AcqResult xmlns="urn:schemas-agilent-com:acmd20">'
    "{signals}"
    "</AcqResult>"
)


def _acmd_signal(trace_id, channel, time_start, time_end, n_values):
    return f"""
    <Signal>
      <Encoding>0</Encoding>
      <TraceId>{trace_id}</TraceId>
      <DeviceName>DAD1</DeviceName>
      <ChannelName>{channel}</ChannelName>
      <Description>Sig=220,4</Description>
      <TimeStart>{time_start}</TimeStart>
      <TimeEnd>{time_end}</TimeEnd>
      <Minimum>0</Minimum>
      <Maximum>1</Maximum>
      <Slope>1</Slope>
      <NumberOfValues>{n_values}</NumberOfValues>
      <DetectorType>1</DetectorType>
      <ScaleFactor>0</ScaleFactor>
      <Units>mAU</Units>
      <NumberOfRecords>{n_values}</NumberOfRecords>
      <IsIntegrable>true</IsIntegrable>
    </Signal>"""


def _build_dx(tmp_path, signal_xml_blocks, traces):
    # traces: {trace_id: [float]}; the .CH payload is a 6144-byte binary
    # header followed by float64 intensity values.
    path = tmp_path / "sample.dx"
    rng = np.random.default_rng(0)
    with ZipFile(path, "w") as zf:
        zf.writestr("injection.acmd", _ACMD_XML.format(signals="".join(signal_xml_blocks)))
        for trace_id, values in traces.items():
            header = rng.standard_normal(6144 // 8).tobytes()
            payload = header + np.asarray(values, dtype=np.float64).tobytes()
            zf.writestr(f"{trace_id}.CH", payload)
    return path


def test_parse_skips_malformed_signal_and_keeps_valid_ones(tmp_path):
    bad = _acmd_signal("TRACEBAD", "DAD1B", time_start="abc", time_end="600000", n_values=4)
    good = _acmd_signal("TRACEGOOD", "DAD1A", time_start="0", time_end="600000", n_values=4)
    path = _build_dx(tmp_path, [bad, good], {"TRACEGOOD": [1.0, 2.0, 3.0, 4.0]})

    spectrum = HPLCParser().parse(path)

    assert spectrum.num_points == 4
    channels = spectrum.parameters["channels"]
    assert [ch["name"] for ch in channels] == ["DAD1A"]
    assert spectrum.x_range == (0.0, 10.0)  # linspace(0, 600000, 4) / 60000


def test_time_axis_follows_primary_channel_signal(tmp_path):
    # The first integrable signal has no readable trace, so the primary
    # channel is the second one and its own meta must drive the time axis.
    first = _acmd_signal("TRACEMISSING", "DAD1A", time_start="0", time_end="600000", n_values=4)
    second = _acmd_signal("TRACEGOOD", "DAD1B", time_start="120000", time_end="480000", n_values=4)
    path = _build_dx(tmp_path, [first, second], {"TRACEGOOD": [5.0, 6.0, 7.0, 8.0]})

    spectrum = HPLCParser().parse(path)

    assert spectrum.parameters["channels"][0]["name"] == "DAD1B"
    assert spectrum.x_range == (2.0, 8.0)  # linspace(120000, 480000, 4) / 60000
    assert spectrum.parameters["time_start_ms"] == 120000.0
    assert spectrum.parameters["time_end_ms"] == 480000.0


def test_all_signals_malformed_raises(tmp_path):
    bad = _acmd_signal("TRACEBAD", "DAD1A", time_start="oops", time_end="600000", n_values=4)
    path = _build_dx(tmp_path, [bad], {})
    with pytest.raises(ValueError, match="No integrable channels"):
        HPLCParser().parse(path)


def test_channels_in_dict():
    parser = HPLCParser()
    spectrum = parser.parse(DATA_DIR / "-S-001.sirslt" / "-S-001.dx")
    d = spectrum.to_dict()

    assert d["technique"] == "HPLC"
    params = d["parameters"]
    assert "channels" in params
    assert len(params["channels"]) >= 2
    assert isinstance(params["channels"][0]["y_data"], list)
