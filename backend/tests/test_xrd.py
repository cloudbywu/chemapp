from pathlib import Path

import pytest

from app.core.models import Technique
from app.parsers.xrd_parser import XRDSParser
from app.analysis.xrd_analysis import XRDAnalyzer

DATA_DIR = Path(__file__).parent.parent.parent / "dataexample" / "XRDexample"


def test_can_parse_asc():
    assert XRDSParser.can_parse(DATA_DIR / "CdS-1_Theta_2-Theta.asc") is True
    assert XRDSParser.can_parse(DATA_DIR / "CdS-1.doc") is False


def test_can_parse_ras():
    assert XRDSParser.can_parse(DATA_DIR / "CdS-1.ras") is True


def test_parse_asc():
    parser = XRDSParser()
    spectrum = parser.parse(DATA_DIR / "CdS-1_Theta_2-Theta.asc")

    assert spectrum.technique == Technique.XRD
    assert spectrum.num_points == 3501
    assert spectrum.x_label == "2θ"
    assert spectrum.y_label == "Intensity"
    assert spectrum.x_unit == "deg"
    assert spectrum.x_range == (10.0, 80.0)
    assert spectrum.parameters["wavelength1_a"] == 1.54059
    assert spectrum.parameters["target"] == "29"


def test_parse_multiple_asc():
    parser = XRDSParser()
    for fname in ["Cu-ZnCdS_Theta_2-Theta.asc", "Fe-ZnCdS_Theta_2-Theta.asc",
                  "blanksi_Theta_2-Theta.asc", "ni_Theta_2-Theta.asc"]:
        spectrum = parser.parse(DATA_DIR / fname)
        assert spectrum.technique == Technique.XRD
        assert spectrum.num_points > 0


def test_analyze_xrd():
    parser = XRDSParser()
    spectrum = parser.parse(DATA_DIR / "CdS-1_Theta_2-Theta.asc")
    analyzer = XRDAnalyzer()
    result = analyzer.analyze(spectrum)

    assert result.technique == Technique.XRD
    assert len(result.peaks) > 0
    assert len(result.d_spacings) > 0
    assert len(result.d_spacings) == len(result.peaks)
    assert len(result.summary) > 0


def test_d_spacing():
    parser = XRDSParser()
    spectrum = parser.parse(DATA_DIR / "CdS-1_Theta_2-Theta.asc")
    analyzer = XRDAnalyzer()
    result = analyzer.analyze(spectrum)

    for ds in result.d_spacings[:5]:
        assert "two_theta" in ds
        assert "d_angstrom" in ds
        assert ds["d_angstrom"] > 0.5
        assert ds["d_angstrom"] < 20.0


def test_crystallite_size():
    parser = XRDSParser()
    spectrum = parser.parse(DATA_DIR / "CdS-1_Theta_2-Theta.asc")
    analyzer = XRDAnalyzer()
    result = analyzer.analyze(spectrum)

    # May or may not have crystallite sizes depending on peak widths
    for cs in result.crystallite_sizes:
        assert cs["size_a"] > 0
        assert cs["fwhm_deg"] > 0


def test_compare_samples():
    parser = XRDSParser()
    analyzer = XRDAnalyzer()

    spectrum_cds = parser.parse(DATA_DIR / "CdS-1_Theta_2-Theta.asc")
    spectrum_cuzn = parser.parse(DATA_DIR / "Cu-ZnCdS_Theta_2-Theta.asc")

    result_cds = analyzer.analyze(spectrum_cds)
    result_cuzn = analyzer.analyze(spectrum_cuzn)

    assert len(result_cds.peaks) > 0
    assert len(result_cuzn.peaks) > 0
    # Doping with Cu should shift peak positions
    cds_main = result_cds.peaks[0].position
    cuzn_main = result_cuzn.peaks[0].position
    assert abs(cds_main - cuzn_main) < 5.0  # should be in similar range


def _write_asc(tmp_path, lines, name="sample.asc"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_asc_count_without_data_rows_raises(tmp_path):
    path = _write_asc(tmp_path, [
        "*SCAN_AXIS = 2theta/theta",
        "*START = 10",
        "*STOP = 80",
        "*STEP = 0.02",
        "*COUNT = 10",
        "header junk line",
        "more junk",
    ])
    with pytest.raises(ValueError, match="COUNT"):
        XRDSParser().parse(path)


def test_asc_scientific_and_negative_data_rows(tmp_path):
    path = _write_asc(tmp_path, [
        "*SCAN_AXIS = 2theta/theta",
        "*START = 10",
        "*STOP = 12",
        "*STEP = 1",
        "*COUNT = 3",
        "1.5E+03, -2, 3.0",
    ])
    spectrum = XRDSParser().parse(path)
    assert spectrum.num_points == 3
    assert list(spectrum.y_data) == [1500.0, -2.0, 3.0]


def test_asc_bad_header_values_warn_and_default(tmp_path):
    path = _write_asc(tmp_path, [
        "*SCAN_AXIS = 2theta/theta",
        "*START = abc",
        "*STOP = 80",
        "*STEP = 0",
        "*WAVE_LENGTH1 = nope",
        "*KV = xx",
        "*MA = 150",
        "*COUNT = 2",
        "5, 6",
    ])
    spectrum = XRDSParser().parse(path)
    warnings = spectrum.parameters["parser_warnings"]
    assert warnings
    assert spectrum.parameters["kv"] is None
    assert spectrum.parameters["ma"] == 150.0
    assert spectrum.parameters["wavelength1_a"] == 1.54059
    assert spectrum.parameters["step"] == 0.02  # non-positive *STEP guarded
    assert spectrum.x_range == (10.0, 80.0)  # *START fell back to default


def test_asc_count_with_data_rows_parses(tmp_path):
    path = _write_asc(tmp_path, [
        "*SCAN_AXIS = 2theta/theta",
        "*START = 10",
        "*STOP = 12",
        "*STEP = 1",
        "*COUNT = 3",
        "7, 8, 9",
    ])
    spectrum = XRDSParser().parse(path)
    assert spectrum.num_points == 3
    assert list(spectrum.y_data) == [7.0, 8.0, 9.0]


def test_xrd_report_sections_from_pdxl_docs():
    parser = XRDSParser()
    spectrum = parser.parse(DATA_DIR / "CdS-1_Theta_2-Theta.asc")
    analyzer = XRDAnalyzer()
    result = analyzer.analyze(spectrum)
    data = result.to_dict()

    assert data["phase_matches"]
    assert data["phase_matches"][0]["phase_name"] in {"Cadmium Zinc Sulfide", "Cadmium Sulfide"}
    assert data["peak_assignments"]
    assert {"observed_two_theta", "reference_two_theta", "peak_shift", "hkl"} <= set(data["peak_assignments"][0])
    assert data["lattice_parameters"]
    assert data["crystallinity"]["crystallinity_percent"] > 0
    assert data["williamson_hall"]["points_used"] >= 2
    assert data["size_distribution"]["bins"]
    assert data["quantitative_analysis"]
    assert data["crystal_structure"]
