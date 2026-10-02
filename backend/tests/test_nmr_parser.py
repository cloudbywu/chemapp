from pathlib import Path

import pytest

from app.core.models import Technique
from app.parsers.nmr_parser import NMRSpectrumParser

DATA_DIR = Path(__file__).parent.parent.parent / "dataexample" / "HNMRexample"


def test_can_parse():
    assert NMRSpectrumParser.can_parse(DATA_DIR) is True
    assert NMRSpectrumParser.can_parse(str(DATA_DIR)) is True
    assert NMRSpectrumParser.can_parse(DATA_DIR / "acqu") is False


def test_parse_spectrum():
    parser = NMRSpectrumParser()
    spectrum = parser.parse(DATA_DIR)

    assert spectrum.technique == Technique.NMR
    assert spectrum.num_points == 65536
    assert spectrum.x_label == "Chemical Shift (1H)"
    assert spectrum.y_label == "Intensity"
    assert spectrum.x_unit == "ppm"
    assert spectrum.metadata.solvent == "CDCl3"
    assert spectrum.parameters["nucleus"] == "1H"
    assert spectrum.parameters["frequency_mhz"] == 400.15
    assert spectrum.parameters["scans"] == 16
    assert spectrum.parameters["pulse_program"] == "zg30"

    x_min, x_max = spectrum.x_range
    assert x_min < -2.0
    assert x_max > 15.0


def test_parse_peaks():
    parser = NMRSpectrumParser()
    spectrum = parser.parse(DATA_DIR)

    assert len(spectrum.peaks) > 0
    x_min, x_max = spectrum.x_range
    for peak in spectrum.peaks:
        assert peak.position > x_min
        assert peak.position < x_max
        assert peak.intensity > 0


def test_parse_rejects_oversized_bruker_binary(tmp_path, monkeypatch):
    monkeypatch.setattr("app.parsers.nmr_parser._MAX_BINARY_BYTES", 16)
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "acqu").write_text("##$BF1= 400.15\n", encoding="utf-8")
    (exp_dir / "fid").write_bytes(b"\x00" * 64)

    with pytest.raises(ValueError, match="512 MiB"):
        NMRSpectrumParser().parse(exp_dir)


@pytest.mark.parametrize(
    ("td", "si", "field"),
    [
        ("65536", "1000000000000", "SI"),
        ("1000000000000", None, "TD"),
        ("1000000000000", "8", "TD"),
        ("65536", "inf", "SI"),
        ("nan", "8", "TD"),
        ("65536", "8.5", "SI"),
        ("-2", "8", "TD"),
    ],
)
def test_parse_rejects_unsafe_dimensions_before_allocation(tmp_path, monkeypatch, td, si, field):
    exp_dir = tmp_path / "exp"
    exp_dir.mkdir()
    (exp_dir / "acqu").write_text(f"##$TD= {td}\n", encoding="utf-8")
    (exp_dir / "fid").write_bytes(b"\x00" * 8)
    if si is not None:
        processed = exp_dir / "pdata" / "1"
        processed.mkdir(parents=True)
        (processed / "procs").write_text(f"##$SI= {si}\n", encoding="utf-8")

    def unexpected_allocation(*args, **kwargs):
        pytest.fail("Dimension validation must happen before allocating the spectrum axis")

    monkeypatch.setattr("app.parsers.nmr_parser.np.arange", unexpected_allocation)
    with pytest.raises(ValueError, match=f"Bruker {field}"):
        NMRSpectrumParser().parse(exp_dir)


def test_bruker_point_count_accepts_limit_and_scientific_notation():
    from app.parsers.nmr_parser import _point_count

    assert _point_count({"SI": "1.6e7"}, "SI", 65536, 16_000_000) == 16_000_000
    assert _point_count({}, "SI", 65536, 16_000_000) == 65536


def test_to_dict_roundtrip():
    parser = NMRSpectrumParser()
    spectrum = parser.parse(DATA_DIR)
    d = spectrum.to_dict()
    assert d["technique"] == "NMR"
    assert len(d["x_data"]) == 65536
    assert len(d["peaks"]) == len(spectrum.peaks)
