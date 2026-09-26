from pathlib import Path

import pytest

from app.core.models import Technique
from app.parsers.uvvis_parser import UVVisParser

DATA_DIR = Path(__file__).parent.parent.parent / "dataexample" / "紫外example"


def test_can_parse_spectrum():
    assert UVVisParser.can_parse(DATA_DIR / "LA.txt") is True
    assert UVVisParser.can_parse(DATA_DIR / "SE.txt") is True


def test_can_parse_calibration():
    assert UVVisParser.can_parse(DATA_DIR / "标准.txt") is True


def test_parse_la_spectrum():
    parser = UVVisParser()
    spectrum = parser.parse(DATA_DIR / "LA.txt")

    assert spectrum.technique == Technique.UVVIS
    assert spectrum.num_points == 601
    assert spectrum.x_label == "Wavelength"
    assert spectrum.y_label == "Absorbance"
    assert spectrum.x_unit == "nm"
    assert spectrum.y_unit == "abs"

    x_min, x_max = spectrum.x_range
    assert x_min == 200.0
    assert x_max == 800.0


def test_parse_se_spectrum():
    parser = UVVisParser()
    spectrum = parser.parse(DATA_DIR / "SE.txt")

    assert spectrum.technique == Technique.UVVIS
    assert spectrum.num_points == 601
    assert spectrum.x_range == (200.0, 800.0)


def test_parse_comma_csv_spectrum(tmp_path):
    path = tmp_path / "comma.csv"
    path.write_text(
        "Wavelength,Absorbance\n"
        "200.0,0.10\n"
        "300.0,0.20\n"
        "400.0,0.30\n",
        encoding="utf-8",
    )

    assert UVVisParser.can_parse(path) is True
    spectrum = UVVisParser().parse(path)
    assert spectrum.num_points == 3
    assert list(spectrum.x_data) == [200.0, 300.0, 400.0]
    assert list(spectrum.y_data) == [0.10, 0.20, 0.30]


def test_parse_semicolon_csv_spectrum(tmp_path):
    path = tmp_path / "semi.csv"
    path.write_text(
        "Wavelength;Absorbance\n"
        "200.0;0.10\n"
        "300.0;0.20\n",
        encoding="utf-8",
    )

    spectrum = UVVisParser().parse(path)
    assert spectrum.num_points == 2
    assert list(spectrum.x_data) == [200.0, 300.0]


def test_parse_spectrum_without_rows_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text(
        "Wavelength,Absorbance\n"
        "not,numeric\n"
        "bad,rows\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no parseable numeric rows"):
        UVVisParser().parse(path)


def test_parse_calibration():
    parser = UVVisParser()
    spectrum = parser.parse(DATA_DIR / "标准.txt")

    assert spectrum.technique == Technique.UVVIS
    assert spectrum.x_label == "Concentration"
    assert spectrum.y_label == "Absorbance"
    assert spectrum.parameters.get("type") == "calibration"
    assert spectrum.num_points == 4
