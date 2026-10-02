from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.store as store_module
from app.main import app
from app.parsers import NMRJCAMPParser, ParserRegistry, inspect_jcamp_numeric

EXTERNAL_ROOT = (
    Path(__file__).parent.parent
    / "data"
    / "external"
    / "zenodo-16881130"
)

_ONE_DIMENSIONAL = """\
##TITLE= synthetic 1H
##JCAMP-DX= 5.00
##DATA TYPE= NMR SPECTRUM
##.OBSERVE FREQUENCY= 400.13
##.OBSERVE NUCLEUS= ^1H
##$SOLVENT= <CDCl3>
##XUNITS= PPM
##YUNITS= ARBITRARY UNITS
##FIRSTX= 10
##LASTX= 0
##NPOINTS= 8
##XFACTOR= 1
##YFACTOR= 1
##XYDATA= (X++(Y..Y))
10 0 1 0 -1 3 0 1 0
##END=
"""

_TWO_DIMENSIONAL = """\
##TITLE= synthetic 2D
##JCAMP-DX= 5.00
##DATA TYPE= nD NMR SPECTRUM
##NTUPLES= nD NMR SPECTRUM
##VAR_NAME= F1,F2,SPECTRUM
##VAR_DIM= 2,3,3
##FACTOR= 1,1,2
##PAGE= F1=1
##DATA TABLE= (F2++(Y..Y)), PROFILE
3 1 2 3
##PAGE= F1=0
##DATA TABLE= (F2++(Y..Y)), PROFILE
3 4 5 6
##END=
"""


def test_parses_one_dimensional_nmr_jcamp_and_preserves_source(tmp_path):
    path = tmp_path / "spectrum.jdx"
    path.write_text(_ONE_DIMENSIONAL, encoding="utf-8")

    assert NMRJCAMPParser.can_parse(path)
    assert ParserRegistry.detect(path) == "nmr_jcamp"
    spectrum = NMRJCAMPParser().parse(path)

    assert spectrum.parameters["format"] == "JCAMP-DX"
    assert spectrum.parameters["nucleus"] == "1H"
    assert spectrum.parameters["frequency_mhz"] == pytest.approx(400.13)
    assert spectrum.parameters["axis_trace"]["conversion"] == "identity_ppm"
    assert spectrum.parameters["processing_pipeline_version"] == "nmr-processing-v2"
    assert spectrum.parameters["jcamp_numeric"]["numeric_payload_decoded"] is True
    assert spectrum.parameters["jcamp_numeric"]["dimension"] == 1
    assert spectrum.parameters["processing_source"]["immutable"] is True
    assert np.allclose(spectrum.x_data, np.linspace(10, 0, 8))
    assert np.array_equal(spectrum.y_data, [0, 1, 0, -1, 3, 0, 1, 0])


def test_upload_accepts_nmr_jcamp(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    store_module._store = None
    with TestClient(app) as client:
        response = client.post(
            "/api/upload",
            files={
                "file": (
                    "spectrum.jdx",
                    _ONE_DIMENSIONAL.encode("utf-8"),
                    "chemical/x-jcamp-dx",
                )
            },
        )

    assert response.status_code == 200, response.text
    assert response.json()["technique"] == "NMR"
    assert response.json()["points"] == 8


def test_numeric_inspection_preserves_2d_shape_and_parser_refuses_flattening(
    tmp_path,
):
    path = tmp_path / "spectrum-2d.jdx"
    path.write_text(_TWO_DIMENSIONAL, encoding="utf-8")

    inspection = inspect_jcamp_numeric(path)

    assert inspection["numeric_payload_decoded"] is True
    assert inspection["dimension"] == 2
    assert inspection["channel_shapes"] == {"real": [2, 3]}
    assert inspection["point_count"] == 6
    assert inspection["finite"] is True
    assert len(inspection["numeric_sha256"]) == 64
    with pytest.raises(ValueError, match="dimension=2"):
        NMRJCAMPParser().parse(path)


@pytest.mark.parametrize("datatype", ["NMR SPECTRUM", "NMR FID"])
def test_rejects_huge_dup_before_nmrglue_decode(tmp_path, monkeypatch, datatype):
    path = tmp_path / "bomb.jdx"
    path.write_text(
        _ONE_DIMENSIONAL.replace("NMR SPECTRUM", datatype).replace(
            "10 0 1 0 -1 3 0 1 0", "10A0S000000000"
        ),
        encoding="utf-8",
    )

    def unexpected_decode(*args, **kwargs):
        pytest.fail("Compressed point limits must be checked before numeric decoding")

    monkeypatch.setattr("app.parsers.nmr_jcamp_parser.ng.jcampdx.read", unexpected_decode)
    with pytest.raises(ValueError, match="safety limit"):
        NMRJCAMPParser().parse(path)


def test_rejects_cumulative_nd_page_budget_before_decoding(tmp_path, monkeypatch):
    monkeypatch.setattr("app.parsers.nmr_jcamp_parser._MAX_JCAMP_NUMERIC_VALUES", 5)
    path = tmp_path / "pages.jdx"
    path.write_text(_TWO_DIMENSIONAL, encoding="utf-8")

    def unexpected_decode(*args, **kwargs):
        pytest.fail("All pages must fit the budget before the first page is decoded")

    monkeypatch.setattr("app.parsers.nmr_jcamp_parser.ng.fileio.jcampdx._parse_data", unexpected_decode)
    inspection = inspect_jcamp_numeric(path)
    assert inspection["numeric_payload_decoded"] is False
    assert "safety limit" in inspection["unsupported_reason"]


@pytest.mark.parametrize("rows", [
    "400A0U",
    "400A0KK\n403OO%T\n406%n",
    "400 1 2 3\n403 4 5",
    "400+1+2-3\n403+4+5",
])
def test_preflight_counts_match_pinned_decoder_at_exact_limit(monkeypatch, rows):
    from app.parsers.nmr_jcamp_parser import _table_numeric_count, ng

    table = "(X++(Y..Y))\n" + rows
    decoded, _ = ng.fileio.jcampdx._parse_data(table)
    monkeypatch.setattr("app.parsers.nmr_jcamp_parser._MAX_JCAMP_NUMERIC_VALUES", len(decoded))
    assert _table_numeric_count(table) == len(decoded)


def test_one_dimensional_point_budget_is_checked_before_decode(tmp_path, monkeypatch):
    monkeypatch.setattr("app.parsers.nmr_jcamp_parser._MAX_JCAMP_NUMERIC_VALUES", 7)
    path = tmp_path / "spectrum.jdx"
    path.write_text(_ONE_DIMENSIONAL, encoding="utf-8")
    with pytest.raises(ValueError, match="safety limit"):
        NMRJCAMPParser().parse(path)


@pytest.mark.skipif(
    not (EXTERNAL_ROOT / "4.zip").exists(),
    reason="fixed external smoke dataset has not been fetched",
)
def test_real_zenodo_nmr_jcamp_axis_and_all_processed_payloads(tmp_path):
    rows: list[dict] = []
    for archive in sorted(EXTERNAL_ROOT.glob("*.zip")):
        with zipfile.ZipFile(archive) as bundle:
            members = [
                name
                for name in bundle.namelist()
                if "/spectra/nmr/" in name
                and name.lower().endswith(".jdx")
                and ".fid.jdx" not in name.lower()
            ]
            for index, member in enumerate(members):
                target = tmp_path / f"{archive.stem}-{index}.jdx"
                target.write_bytes(bundle.read(member))
                inspection = inspect_jcamp_numeric(target)
                rows.append(inspection)
                if archive.name == "4.zip" and member.endswith("/28.jdx"):
                    spectrum = NMRJCAMPParser().parse(target)
                    assert spectrum.num_points == 16384
                    assert spectrum.x_data[0] == pytest.approx(11.00659)
                    assert spectrum.x_data[-1] == pytest.approx(
                        -1.0093363894736704
                    )
                    assert spectrum.parameters["quadrature_available"] is True

    assert len(rows) == 16
    assert all(row["numeric_payload_decoded"] for row in rows)
    assert sum(row["dimension"] == 1 for row in rows) == 10
    assert sum(row["dimension"] == 2 for row in rows) == 6


@pytest.mark.skipif(
    not (EXTERNAL_ROOT / "3.zip").exists(),
    reason="fixed external smoke dataset has not been fetched",
)
def test_real_zenodo_jcamp_fid_fft_aligns_with_processed_companion(tmp_path):
    with zipfile.ZipFile(EXTERNAL_ROOT / "3.zip") as bundle:
        names = [
            name
            for name in bundle.namelist()
            if "/spectra/nmr/" in name and name.lower().endswith(".jdx")
        ]
        fid_name = next(name for name in names if name.lower().endswith(".fid.jdx"))
        spectrum_name = next(
            name for name in names if not name.lower().endswith(".fid.jdx")
        )
        fid_path = tmp_path / "source.fid.jdx"
        spectrum_path = tmp_path / "processed.jdx"
        fid_path.write_bytes(bundle.read(fid_name))
        spectrum_path.write_bytes(bundle.read(spectrum_name))

    fid = NMRJCAMPParser().parse(fid_path)
    processed = NMRJCAMPParser().parse(spectrum_path)
    region_fid = (fid.x_data >= -1) & (fid.x_data <= 14)
    region_processed = (processed.x_data >= -1) & (processed.x_data <= 14)
    fid_peak = fid.x_data[
        np.flatnonzero(region_fid)[np.argmax(fid.y_data[region_fid])]
    ]
    processed_peak = processed.x_data[
        np.flatnonzero(region_processed)[
            np.argmax(processed.y_data[region_processed])
        ]
    ]

    assert fid.parameters["source_domain"] == "time"
    assert fid.parameters["phase_source"] == "automatic_zero_and_first_order"
    assert fid.parameters["digital_filter_applied_points"] == pytest.approx(76)
    assert fid_peak == pytest.approx(processed_peak, abs=0.01)
