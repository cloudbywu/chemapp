from pathlib import Path

from app.analysis.uvvis_analysis import UVVisAnalyzer
from app.parsers.uvvis_parser import UVVisParser

DATA_DIR = Path(__file__).parent.parent.parent / "dataexample" / "紫外example"


def test_analyze_la_spectrum():
    parser = UVVisParser()
    spectrum = parser.parse(DATA_DIR / "LA.txt")
    analyzer = UVVisAnalyzer()
    result = analyzer.analyze(spectrum)

    assert result.technique.value == "UV-Vis"
    assert len(result.lambda_max) >= 1
    assert len(result.peaks) >= 1
    assert len(result.summary) > 0


def test_analyze_se_spectrum():
    parser = UVVisParser()
    spectrum = parser.parse(DATA_DIR / "SE.txt")
    analyzer = UVVisAnalyzer()
    result = analyzer.analyze(spectrum)

    assert result.technique.value == "UV-Vis"
    assert len(result.lambda_max) > 0


def test_lambda_max_detection():
    parser = UVVisParser()
    spectrum = parser.parse(DATA_DIR / "LA.txt")
    analyzer = UVVisAnalyzer()
    result = analyzer.analyze(spectrum)

    # LA should show absorption in UV range
    uv_peaks = [wl for wl in result.lambda_max if 200 < wl < 400]
    assert len(uv_peaks) > 0 or len(result.peaks) > 0


def test_analyze_calibration():
    parser = UVVisParser()
    spectrum = parser.parse(DATA_DIR / "标准.txt")
    analyzer = UVVisAnalyzer()
    result = analyzer.analyze(spectrum)

    assert result.calibration is not None
    assert "slope" in result.calibration
    assert "intercept" in result.calibration
    assert "r_squared" in result.calibration
    assert result.calibration["r_squared"] > 0.95
    assert result.calibration["n_points"] == 4


def test_compute_concentration():
    parser = UVVisParser()
    cal_spectrum = parser.parse(DATA_DIR / "标准.txt")
    analyzer = UVVisAnalyzer()
    cal_result = analyzer.analyze(cal_spectrum)

    conc = analyzer.compute_concentration(cal_result, absorbance=0.557)
    assert conc is not None
    assert 18 < conc < 24
