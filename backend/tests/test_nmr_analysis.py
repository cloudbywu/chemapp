from pathlib import Path

import numpy as np

from app.analysis.nmr_analysis import NMRAnalyzer
from app.core.models import Peak, SampleInfo, Spectrum, Technique
from app.parsers.nmr_parser import NMRSpectrumParser

DATA_DIR = Path(__file__).parent.parent.parent / "dataexample" / "HNMRexample"


def test_analyze_nmr():
    parser = NMRSpectrumParser()
    spectrum = parser.parse(DATA_DIR)
    analyzer = NMRAnalyzer()
    result = analyzer.analyze(spectrum)

    assert result.technique.value == "NMR"
    assert result.noise_level > 0
    assert result.total_integral > 0
    assert len(result.peaks) > 5
    assert len(result.metrics) > 0
    assert len(result.summary) > 0


def test_nmr_peak_detection():
    parser = NMRSpectrumParser()
    spectrum = parser.parse(DATA_DIR)
    analyzer = NMRAnalyzer()
    result = analyzer.analyze(spectrum)

    aromatic_peaks = [p for p in result.peaks if 7.0 < p.position < 9.0]
    aliphatic_peaks = [p for p in result.peaks if 0.5 < p.position < 5.0]
    assert len(aromatic_peaks) > 0
    assert len(aliphatic_peaks) > 0


def test_nmr_integrals():
    parser = NMRSpectrumParser()
    spectrum = parser.parse(DATA_DIR)
    analyzer = NMRAnalyzer()
    result = analyzer.analyze(spectrum)

    assert len(result.integrals) > 0
    for integ in result.integrals:
        assert "center_ppm" in integ
        assert "raw_area" in integ
        assert "relative_area" in integ
        assert integ["raw_area"] >= 0


def test_nmr_multiplets():
    parser = NMRSpectrumParser()
    spectrum = parser.parse(DATA_DIR)
    analyzer = NMRAnalyzer()
    default_result = analyzer.analyze(spectrum)
    assert default_result.reference_corrected is False

    result = analyzer.analyze(spectrum, {"auto_reference": True})

    assert len(result.multiplets) > 0
    for mp in result.multiplets:
        assert "center_ppm" in mp
        assert "component_positions" in mp
        assert "multiplicity" in mp
        assert len(mp["component_positions"]) >= 1


def test_mnova_style_multiplet_ranges():
    parser = NMRSpectrumParser()
    spectrum = parser.parse(DATA_DIR)
    analyzer = NMRAnalyzer()

    mnova_ranges = [
        (8.42, 8.37),
        (8.18, 8.06),
        (8.06, 8.00),
        (7.73, 7.66),
        (7.60, 7.51),
        (7.46, 7.37),
        (7.20, 7.13),
        (6.60, 6.49),
        (6.47, 6.39),
        (6.38, 6.31),
        (4.46, 4.41),
        (4.26, 4.17),
        (4.10, 3.96),
        (3.55, 3.45),
        (2.96, 2.91),
        (2.00, 1.71),
        (1.61, 1.42),
        (1.27, 1.10),
    ]
    expected_types = [
        "d", "ddd", "d", "dd", "ddd", "m", "d", "dd", "m",
        "d", "s", "tt", "dq", "qd", "s", "m", "m", "m",
    ]

    result = analyzer.analyze(
        spectrum,
        {"auto_reference": False, "multiplet_ranges": mnova_ranges},
    )

    assert len(result.multiplets) == 18
    assert [mp["multiplicity"] for mp in result.multiplets] == expected_types
    assert result.multiplets[0]["j_values_hz"] == [2.5]
    assert result.multiplets[3]["j_values_hz"] == [2.56, 8.44]
    assert result.multiplets[7]["j_values_hz"] == [8.82, 18.7]
    assert result.multiplets[11]["j_values_hz"][2:] == [6.08, 6.08]
    assert result.multiplets[12]["j_values_hz"][:2] == [6.13, 6.13]
    assert result.multiplets[13]["j_values_hz"][1:] == [6.3, 6.3, 6.3]


def test_nmr_solvent_reference():
    parser = NMRSpectrumParser()
    spectrum = parser.parse(DATA_DIR)

    assert spectrum.metadata.solvent == "CDCl3"

    analyzer = NMRAnalyzer()
    result = analyzer.analyze(spectrum)

    chloroform_peaks = [p for p in result.peaks if abs(p.position - 7.26) < 0.1]
    assert len(chloroform_peaks) > 0


def test_to_dict_roundtrip():
    parser = NMRSpectrumParser()
    spectrum = parser.parse(DATA_DIR)
    analyzer = NMRAnalyzer()
    result = analyzer.analyze(spectrum)

    d = result.to_dict()
    assert d["technique"] == "NMR"
    assert len(d["peaks"]) == len(result.peaks)
    assert len(d["integrals"]) == len(result.integrals)
    assert "noise_level" in d


def test_automatic_grouping_keeps_strong_singlets_and_filters_weak_ones():
    analyzer = NMRAnalyzer()
    spectrum = Spectrum(
        technique=Technique.NMR,
        x_data=np.array([0.0, 1.0]),
        y_data=np.array([0.0, 0.0]),
        metadata=SampleInfo(solvent="CDCl3"),
        parameters={"frequency_mhz": 600.0},
    )
    multiplets = analyzer._group_multiplets(
        [
            Peak(position=8.04, intensity=12.0),
            Peak(position=6.50, intensity=7.9),
        ],
        spectrum,
        noise_level=1.0,
    )

    assert [item["center_ppm"] for item in multiplets] == [8.04]
    assert multiplets[0]["multiplicity"] == "s"
    assert multiplets[0]["is_isolated_singlet"] is True
    assert multiplets[0]["signal_to_noise"] == 12.0


def test_automatic_grouping_does_not_merge_a_solvent_line_with_analyte_lines():
    analyzer = NMRAnalyzer()
    spectrum = Spectrum(
        technique=Technique.NMR,
        x_data=np.array([0.0, 1.0]),
        y_data=np.array([0.0, 0.0]),
        metadata=SampleInfo(solvent="CDCl3"),
        parameters={"frequency_mhz": 600.0},
    )
    peaks = [
        Peak(position=7.26, intensity=100.0, assignment="solvent"),
        Peak(position=7.331, intensity=30.0),
        Peak(position=7.3806, intensity=40.0),
    ]
    analyte_peaks = [peak for peak in peaks if peak.assignment != "solvent"]
    multiplets = analyzer._group_multiplets(
        analyte_peaks,
        spectrum,
        noise_level=1.0,
    )

    assert len(multiplets) == 1
    assert multiplets[0]["component_positions"] == [7.3806, 7.331]
    assert 7.34 < multiplets[0]["center_ppm"] < 7.39
