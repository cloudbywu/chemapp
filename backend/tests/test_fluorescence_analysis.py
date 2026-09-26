from pathlib import Path

from app.analysis.fluorescence_analysis import FluorescenceAnalyzer
from app.parsers.fluorescence_parser import FluorescenceParser

DATA_DIR = Path(__file__).parent.parent.parent / "dataexample" / "荧光example"


def test_analyze_emission_lao():
    parser = FluorescenceParser()
    spectrum = parser.parse(DATA_DIR / "em-lao(FDS).DX")
    analyzer = FluorescenceAnalyzer()
    result = analyzer.analyze(spectrum)

    assert result.technique.value == "Fluorescence"
    assert result.em_peak is not None
    assert 400 < result.em_peak < 700
    assert len(result.peaks) > 0
    assert len(result.summary) > 0


def test_analyze_excitation_lao():
    parser = FluorescenceParser()
    spectrum = parser.parse(DATA_DIR / "ex-lao.DX")
    analyzer = FluorescenceAnalyzer()
    result = analyzer.analyze(spectrum)

    assert result.technique.value == "Fluorescence"
    assert result.ex_peak is not None
    assert 200 < result.ex_peak < 500
    assert result.metrics["sub_type"] == "excitation"


def test_analyze_emission_se():
    parser = FluorescenceParser()
    spectrum = parser.parse(DATA_DIR / "em-se(FDS).DX")
    analyzer = FluorescenceAnalyzer()
    result = analyzer.analyze(spectrum)

    assert result.em_peak is not None
    assert 300 < result.em_peak < 750


def test_analyze_excitation_se():
    parser = FluorescenceParser()
    spectrum = parser.parse(DATA_DIR / "ex-se(FDS).DX")
    analyzer = FluorescenceAnalyzer()
    result = analyzer.analyze(spectrum)

    assert result.ex_peak is not None
    assert 200 < result.ex_peak < 500


def test_stokes_shift_computation():
    parser = FluorescenceParser()
    ex_spectrum = parser.parse(DATA_DIR / "ex-lao.DX")
    em_spectrum = parser.parse(DATA_DIR / "em-lao(FDS).DX")
    analyzer = FluorescenceAnalyzer()

    ex_result = analyzer.analyze(ex_spectrum)
    em_result = analyzer.analyze(em_spectrum)

    stokes = analyzer.compute_stokes_shift(ex_result, em_result)
    assert stokes["stokes_shift_nm"] is not None
    assert stokes["stokes_shift_nm"] > 0
    assert stokes["stokes_shift_cm1"] is not None


def test_to_dict_roundtrip():
    parser = FluorescenceParser()
    spectrum = parser.parse(DATA_DIR / "em-lao(FDS).DX")
    analyzer = FluorescenceAnalyzer()
    result = analyzer.analyze(spectrum)

    d = result.to_dict()
    assert d["technique"] == "Fluorescence"
    assert d["em_peak"] == result.em_peak
    assert "stokes_shift_nm" in d


def test_compare_lao_vs_se_emission():
    parser = FluorescenceParser()
    analyzer = FluorescenceAnalyzer()

    result_lao = analyzer.analyze(parser.parse(DATA_DIR / "em-lao(FDS).DX"))
    result_se = analyzer.analyze(parser.parse(DATA_DIR / "em-se(FDS).DX"))

    assert result_lao.em_peak is not None
    assert result_se.em_peak is not None
    assert 300 < result_lao.em_peak < 750
    assert 300 < result_se.em_peak < 750
