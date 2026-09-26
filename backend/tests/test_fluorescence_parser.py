from pathlib import Path

import pytest

from app.core.models import Technique
from app.parsers.fluorescence_parser import FluorescenceParser

DATA_DIR = Path(__file__).parent.parent.parent / "dataexample" / "荧光example"


def test_can_parse():
    assert FluorescenceParser.can_parse(DATA_DIR / "em-lao(FDS).DX") is True
    assert FluorescenceParser.can_parse(DATA_DIR / "ex-lao.DX") is True
    assert FluorescenceParser.can_parse(Path("nonexistent.dx")) is False


def test_parse_emission():
    parser = FluorescenceParser()
    spectrum = parser.parse(DATA_DIR / "em-lao(FDS).DX")

    assert spectrum.technique == Technique.FLUORESCENCE
    assert spectrum.num_points == 3501
    assert spectrum.x_label == "Wavelength"
    assert spectrum.y_label == "Intensity"
    assert spectrum.x_unit == "nm"
    assert spectrum.y_unit == "arb. units"
    assert spectrum.x_range == (200.0, 900.0)
    assert spectrum.parameters["sub_type"] == "emission"
    assert spectrum.parameters["excitation_wavelength_nm"] == 280.0
    assert spectrum.parameters["delta_x_nm"] == 0.2
    assert spectrum.metadata.name == "em-lao"


def test_parse_excitation():
    parser = FluorescenceParser()
    spectrum = parser.parse(DATA_DIR / "ex-lao.DX")

    assert spectrum.technique == Technique.FLUORESCENCE
    assert spectrum.num_points == 3501
    assert spectrum.parameters["sub_type"] == "excitation"
    assert spectrum.parameters["emission_wavelength_nm"] == 600.0


def test_parse_emission_se():
    parser = FluorescenceParser()
    spectrum = parser.parse(DATA_DIR / "em-se(FDS).DX")

    assert spectrum.technique == Technique.FLUORESCENCE
    assert spectrum.num_points == 3501
    assert spectrum.parameters["sub_type"] == "emission"
    assert spectrum.parameters["excitation_wavelength_nm"] == 260.0


def test_parse_excitation_se():
    parser = FluorescenceParser()
    spectrum = parser.parse(DATA_DIR / "ex-se(FDS).DX")

    assert spectrum.technique == Technique.FLUORESCENCE
    assert spectrum.num_points == 3501


def _write_dx(tmp_path, lines, name="sample.DX"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _dx_lines(data_rows, extra_labels=()):
    lines = [
        "##TITLE= synth",
        "##JCAMP-DX= 4.24",
        "##DATA TYPE= FL SPECTRUM",
    ]
    lines.extend(extra_labels)
    lines.append("##XYDATA= (X++(Y..Y))")
    lines.extend(data_rows)
    lines.append("##END=")
    return lines


def test_parse_xydata_difdup_compressed(tmp_path):
    # Packed DIF/DUP rows; the first token of the second/third row repeats
    # the previous DIF as a checkpoint and must be skipped on decode.
    # SQZ: A=1..I=9,a=-1..i=-9; DIF: J=1..R=9,j=-1..r=-9; DUP: S=1..Z=8,s=9.
    path = _write_dx(tmp_path, _dx_lines(
        ["400.0A0KK", "403.0OO%T", "406.0%n"],
        extra_labels=[
            "##INSTRUMENT PARAMETERS= Scan mode= Emission;",
            "EX WL= 280.0 nm;",
            "##DELTAX= 1.0",
            "##NPOINTS= 7",
        ],
    ))
    spectrum = FluorescenceParser().parse(path)

    assert spectrum.num_points == 7
    assert list(spectrum.y_data) == [10.0, 12.0, 14.0, 20.0, 20.0, 20.0, 15.0]
    assert list(spectrum.x_data) == [400.0, 401.0, 402.0, 403.0, 404.0, 405.0, 406.0]
    assert spectrum.parameters["sub_type"] == "emission"
    assert spectrum.parameters["excitation_wavelength_nm"] == 280.0


def test_parse_xydata_affn_multi_y_rows(tmp_path):
    path = _write_dx(tmp_path, _dx_lines(
        ["400.0 1.0 2.0 3.0", "401.5 4.0 5.0"],
        extra_labels=["##DELTAX= 0.5", "##NPOINTS= 5"],
    ))
    spectrum = FluorescenceParser().parse(path)

    assert list(spectrum.x_data) == [400.0, 400.5, 401.0, 401.5, 402.0]
    assert list(spectrum.y_data) == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_parse_non_numeric_wavelengths_become_none(tmp_path):
    path = _write_dx(tmp_path, _dx_lines(
        ["400.0 1.0", "401.0 2.0"],
        extra_labels=[
            "##INSTRUMENT PARAMETERS= Scan mode= Emission;",
            "EX WL= bright;",
            "EM WL= 600.0 nm;",
            "##DELTAX= 1.0",
            "##NPOINTS= 2",
        ],
    ))
    spectrum = FluorescenceParser().parse(path)

    assert spectrum.parameters["excitation_wavelength_nm"] is None
    assert spectrum.parameters["emission_wavelength_nm"] == 600.0
    assert spectrum.num_points == 2


def test_unknown_scan_mode_leaves_sub_type_empty(tmp_path):
    path = _write_dx(tmp_path, _dx_lines(
        ["400.0 1.0", "401.0 2.0"],
        extra_labels=[
            "##INSTRUMENT PARAMETERS= Scan mode= Kinetics;",
            "##DELTAX= 1.0",
            "##NPOINTS= 2",
        ],
    ))
    spectrum = FluorescenceParser().parse(path)

    assert spectrum.parameters["sub_type"] == ""


def test_xydata_multi_y_without_valid_deltax_raises(tmp_path):
    path = _write_dx(tmp_path, _dx_lines(
        ["400.0 1.0 2.0", "401.0 3.0 4.0"],
        extra_labels=["##DELTAX= junk", "##NPOINTS= 4"],
    ))
    with pytest.raises(ValueError, match="DELTAX"):
        FluorescenceParser().parse(path)


def test_parse_rejects_oversized_file(tmp_path, monkeypatch):
    monkeypatch.setattr("app.parsers.fluorescence_parser._MAX_DX_FILE_BYTES", 32)
    big = tmp_path / "big.DX"
    big.write_bytes(b"##JCAMP-DX= 4.24 " + b"x" * 64)
    with pytest.raises(ValueError, match="safety limit"):
        FluorescenceParser().parse(big)


def test_to_dict_roundtrip():
    parser = FluorescenceParser()
    spectrum = parser.parse(DATA_DIR / "em-lao(FDS).DX")
    d = spectrum.to_dict()
    assert d["technique"] == "Fluorescence"
    assert len(d["x_data"]) == 3501
    assert d["parameters"]["sub_type"] == "emission"
