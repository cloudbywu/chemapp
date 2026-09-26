from pathlib import Path

import numpy as np

from app.analysis.electrochem_analysis import ElectrochemAnalyzer
from app.core.models import Spectrum, Technique
from app.parsers.electrochem_parser import ElectrochemParser

DATA_DIR = Path(__file__).parent.parent.parent / "dataexample" / "伏安法example"


def test_can_parse_cv():
    assert ElectrochemParser.can_parse(DATA_DIR / "0_1mL.txt") is True
    assert ElectrochemParser.can_parse(DATA_DIR / "100mV_s.txt") is True


def test_can_parse_eis():
    assert ElectrochemParser.can_parse(DATA_DIR / "1.txt") is True


def test_cannot_parse():
    assert ElectrochemParser.can_parse(DATA_DIR / "0_1mL.bin") is False


def test_parse_cv():
    parser = ElectrochemParser()
    spectrum = parser.parse(DATA_DIR / "0_1mL.txt")

    assert spectrum.technique == Technique.ELECTROCHEM
    assert spectrum.num_points == 1200
    assert spectrum.x_label == "Potential"
    assert spectrum.y_label == "Current"
    assert spectrum.x_unit == "V"
    assert spectrum.y_unit == "A"
    assert spectrum.parameters["sub_type"] == "CV"
    assert spectrum.parameters["scan_rate_v_s"] == 0.05


def test_parse_eis():
    parser = ElectrochemParser()
    spectrum = parser.parse(DATA_DIR / "1.txt")

    assert spectrum.technique == Technique.ELECTROCHEM
    assert spectrum.num_points == 72
    assert spectrum.x_label == "Z' (Real)"
    assert spectrum.parameters["sub_type"] == "EIS"


def test_parse_multiple_cv():
    parser = ElectrochemParser()
    for fname in ["0_1mL.txt", "0_2mL.txt", "0_4mL.txt", "0_6mL.txt",
                  "5mV_s.txt", "50mV_s.txt", "200mV_s.txt", "empty.txt"]:
        spectrum = parser.parse(DATA_DIR / fname)
        assert spectrum.technique == Technique.ELECTROCHEM
        assert spectrum.num_points > 100


def test_analyze_cv():
    parser = ElectrochemParser()
    spectrum = parser.parse(DATA_DIR / "0_1mL.txt")
    analyzer = ElectrochemAnalyzer()
    result = analyzer.analyze(spectrum)

    assert result.technique == Technique.ELECTROCHEM
    assert result.metrics["sub_type"] == "CV"
    assert result.metrics["ep_anodic_v"] is not None
    assert result.metrics["ep_cathodic_v"] is not None
    assert result.metrics["ip_anodic_a"] is not None
    assert result.metrics["ip_cathodic_a"] is not None
    assert len(result.peaks) >= 1
    assert len(result.summary) > 0


def test_cv_peak_values():
    parser = ElectrochemParser()
    spectrum = parser.parse(DATA_DIR / "0_1mL.txt")
    analyzer = ElectrochemAnalyzer()
    result = analyzer.analyze(spectrum)

    ep_a = result.metrics["ep_anodic_v"]
    ep_c = result.metrics["ep_cathodic_v"]
    delta_ep = result.metrics["delta_ep_v"]

    assert 0.15 < ep_a < 0.25
    assert 0.15 < ep_c < 0.30
    assert delta_ep > 0.05
    assert delta_ep < 0.15


def test_instrument_vs_parser():
    parser = ElectrochemParser()
    spectrum = parser.parse(DATA_DIR / "0_1mL.txt")
    analyzer = ElectrochemAnalyzer()
    result = analyzer.analyze(spectrum)

    inst_ep = result.metrics["instrument_ep_v"]
    our_ep = result.metrics["ep_anodic_v"]
    assert inst_ep is not None
    assert our_ep is not None
    assert abs(inst_ep - our_ep) < 0.05


def test_analyze_eis():
    parser = ElectrochemParser()
    spectrum = parser.parse(DATA_DIR / "1.txt")
    analyzer = ElectrochemAnalyzer()
    result = analyzer.analyze(spectrum)

    assert result.technique == Technique.ELECTROCHEM
    assert result.metrics["sub_type"] == "EIS"
    assert result.metrics["rs_ohm"] > 0
    assert result.metrics["rct_ohm"] > 0
    assert len(result.summary) > 0


def test_scan_rate_series():
    parser = ElectrochemParser()
    analyzer = ElectrochemAnalyzer()

    prev_ip = None
    for v_s in [5, 10, 20, 50, 100, 200]:
        spectrum = parser.parse(DATA_DIR / f"{v_s}mV_s.txt")
        result = analyzer.analyze(spectrum)
        ip_a = result.metrics["ip_anodic_a"]
        assert ip_a is not None
        # Peak current should increase with scan rate (Randles-Sevcik)
        if prev_ip is not None:
            assert abs(ip_a) >= abs(prev_ip) * 0.5  # rough trend check
        prev_ip = ip_a


def _spectrum(x: np.ndarray, y: np.ndarray, sub_type: str) -> Spectrum:
    return Spectrum(
        technique=Technique.ELECTROCHEM,
        x_data=x,
        y_data=y,
        parameters={"sub_type": sub_type, "scan_rate_v_s": 0.05},
    )


def test_cv_split_uses_potential_vertex_not_midpoint():
    # Sweep starts at 0 V, reverses at 1.2 V, then returns with more points on
    # the reverse leg than the forward leg; the vertex is at index 40 of 100.
    x_forward = np.linspace(0.0, 1.2, 41)
    x_reverse = np.linspace(1.2, 0.0, 60)[1:]
    x = np.concatenate([x_forward, x_reverse])
    y = np.sin(x * 3.0) + 0.1 * np.cos(x * 7.0)
    result = ElectrochemAnalyzer().analyze(_spectrum(x, y, "CV"))
    assert result.metrics["sub_type"] == "CV"
    assert result.metrics["ep_anodic_v"] is not None
    assert result.metrics["ep_cathodic_v"] is not None
    # The vertex potential itself must be preserved inside the scan split:
    # reverse-leg peak search must see values beyond 0.6 V (midpoint).
    assert result.metrics["ep_anodic_v"] > 0.8 or result.metrics["ep_cathodic_v"] > 0.8


def test_cv_split_handles_descending_start():
    # Sweep starts high (1.2 V) and reverses at 0.0 V.
    x_forward = np.linspace(1.2, 0.0, 41)
    x_reverse = np.linspace(0.0, 1.2, 60)[1:]
    x = np.concatenate([x_forward, x_reverse])
    y = np.cos(x * 4.0)
    result = ElectrochemAnalyzer().analyze(_spectrum(x, y, "CV"))
    assert result.metrics["ep_anodic_v"] is not None
    assert result.metrics["ep_cathodic_v"] is not None


def test_eis_rs_and_rct_from_semicircle():
    # Semicircle from Rs=10 to Rs+Rct=60 with -Z'' maximum at the top.
    rs_true = 10.0
    rct_true = 50.0
    angles = np.linspace(0.0, np.pi, 72)
    z_prime = rs_true + rct_true / 2 * (1 - np.cos(angles))
    z_double = rct_true / 2 * np.sin(angles)
    result = ElectrochemAnalyzer().analyze(_spectrum(z_prime, z_double, "EIS"))
    assert result.metrics["sub_type"] == "EIS"
    assert abs(result.metrics["rs_ohm"] - rs_true) < 1e-6
    assert abs(result.metrics["rct_ohm"] - rct_true) < 1e-6


def test_cv_garbage_params_fall_back_to_defaults(tmp_path):
    path = tmp_path / "garbage.txt"
    path.write_text(
        "\n".join([
            "Cyclic Voltammetry",
            "Init E (V) = oops",
            "High E (V) = 0.8",
            "Low E (V) = 0",
            "Scan Rate (V/s) = fast",
            "Segment = many",
            "",
            "Segment 1:",
            "Ep = 0.2V",
            "",
            "Potential/V, Current/A",
            "0.0, 1.0",
            "0.1, 1.1",
            "0.2, 1.2",
        ]) + "\n",
        encoding="utf-8",
    )

    spectrum = ElectrochemParser().parse(path)
    assert spectrum.parameters["sub_type"] == "CV"
    assert spectrum.parameters["scan_rate_v_s"] == 0.05
    assert spectrum.parameters["n_segments"] == 2
    assert spectrum.parameters["init_e_v"] == 0.6
    assert spectrum.num_points == 3


def _write_eis(tmp_path, header, rows, name="eis.txt"):
    path = tmp_path / name
    path.write_text(
        "\n".join([
            "A.C. Impedance",
            "Amplitude (V) = 0.005",
            "High Frequency (Hz) = 1e5",
            "Low Frequency (Hz) = 1",
            "",
            header,
            "",
            *rows,
        ]) + "\n",
        encoding="utf-8",
    )
    return path


def test_eis_raw_z_double_prime_is_negated_per_header(tmp_path):
    path = _write_eis(
        tmp_path,
        "Freq/Hz, Z'/ohm, Z\"/ohm, Z/ohm, Phase/deg",
        [
            "1.0e+3, 10.0, -5.0, 11.2, -26.5",
            "1.0e+2, 20.0, -10.0, 22.4, -26.5",
        ],
    )
    spectrum = ElectrochemParser().parse(path)

    assert list(spectrum.x_data) == [10.0, 20.0]
    assert list(spectrum.y_data) == [5.0, 10.0]  # negated to -Z''
    assert spectrum.parameters["freq_hz"] == [1000.0, 100.0]
    assert spectrum.parameters["z_imag_already_negated"] is False
    assert "negated" in spectrum.parameters["z_imag_sign_source"]


def test_eis_negated_column_is_kept_per_header(tmp_path):
    path = _write_eis(
        tmp_path,
        "Freq/Hz, Z'/ohm, -Z\"/ohm, Z/ohm, Phase/deg",
        ["1.0e+3, 10.0, 5.0, 11.2, -26.5"],
    )
    spectrum = ElectrochemParser().parse(path)

    assert list(spectrum.y_data) == [5.0]  # already -Z'', left untouched
    assert spectrum.parameters["z_imag_already_negated"] is True


def test_eis_without_recognized_header_keeps_values_and_marks_unknown(tmp_path):
    path = _write_eis(
        tmp_path,
        "Freq/Hz, Re, Im, Mag, Phase",
        ["1.0e+3, 10.0, -5.0, 11.2, -26.5"],
    )
    spectrum = ElectrochemParser().parse(path)

    assert list(spectrum.y_data) == [-5.0]
    assert "unknown" in spectrum.parameters["z_imag_sign_source"]


def test_eis_handles_empty_data():
    result = ElectrochemAnalyzer().analyze(
        _spectrum(np.array([]), np.array([]), "EIS")
    )
    assert result.metrics["error"] == "insufficient_points"
