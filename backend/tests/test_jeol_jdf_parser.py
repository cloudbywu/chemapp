from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.store as store_module
from app.analysis.nmr_analysis import NMRAnalyzer
from app.analysis.nmr_processing import read_processing_source
from app.core.models import Technique
from app.main import app
from app.parsers import ParserRegistry
from app.parsers.jeol_jdf_parser import JEOLJDFParser

JDF_FILE = Path(__file__).parent.parent.parent / "dataexample" / "26-7-11-cyj-sja_proton-1-1.jdf"


def test_detects_real_jeol_jdf_sample():
    assert JDF_FILE.exists()
    assert JEOLJDFParser.can_parse(JDF_FILE)
    assert ParserRegistry.detect(JDF_FILE) == "jeol_jdf"


def test_parses_real_jeol_1h_fid_to_spectrum():
    spectrum = JEOLJDFParser().parse(JDF_FILE)

    assert spectrum.technique == Technique.NMR
    assert spectrum.num_points == 30000
    assert spectrum.parameters["vendor"] == "JEOL"
    assert spectrum.parameters["format"] == "JDF"
    assert spectrum.parameters["nucleus"] == "1H"
    assert spectrum.parameters["frequency_mhz"] == 600.172305
    assert spectrum.parameters["scans"] == 16
    assert spectrum.parameters["pulse_program"] == "proton.jxp"
    assert spectrum.parameters["source_domain"] == "time"
    assert spectrum.parameters["phase_corrected"] is True
    assert spectrum.parameters["baseline_corrected"] is True
    assert spectrum.parameters["baseline_method"] == "asymmetric_least_squares"
    assert spectrum.parameters["quadrature_available"] is True
    assert spectrum.parameters["original_data_preserved"] is True
    assert 19.8 < spectrum.parameters["digital_filter_points"] < 20.0
    assert 7000 < spectrum.parameters["auto_phase_first_deg"] < 7300
    assert spectrum.metadata.name == "26-7-11-cyj-sja"
    assert spectrum.metadata.solvent == "CDCl3"
    assert np.isfinite(spectrum.x_data).all()
    assert np.isfinite(spectrum.y_data).all()
    assert np.max(spectrum.y_data) > 1000

    strongest_ppm = float(spectrum.x_data[int(np.argmax(spectrum.y_data))])
    assert 7.1 < strongest_ppm < 7.4

    # Peak-free regions should remain close to a horizontal zero baseline.
    quiet = ((spectrum.x_data >= 4.0) & (spectrum.x_data <= 6.0)) | (
        (spectrum.x_data >= 9.0) & (spectrum.x_data <= 15.0)
    )
    quiet_values = spectrum.y_data[quiet]
    assert abs(float(np.median(quiet_values))) < 1.0
    assert float(np.percentile(quiet_values, 95) - np.percentile(quiet_values, 5)) < 15.0

    source = spectrum.parameters["processing_source"]
    source_x, source_y, source_real, source_imaginary = read_processing_source(source)
    assert np.array_equal(source_x, spectrum.x_data)
    assert np.array_equal(source_y, spectrum.y_data)
    assert source_real is not None
    assert source_imaginary is not None
    assert np.isfinite(source_real).all()
    assert np.isfinite(source_imaginary).all()

    public = spectrum.to_dict(include_internal=False)
    assert "processing_source" not in public["parameters"]
    assert public["parameters"]["processing_source_summary"]["has_quadrature"] is True
    assert (
        public["parameters"]["processing_source_summary"]["pipeline_version"]
        == "nmr-processing-v2"
    )
    assert public["parameters"]["processing_source_summary"]["default_phase"][
        "manual_adjustments_are_incremental"
    ] is True


def test_parse_rejects_oversized_jdf(tmp_path, monkeypatch):
    monkeypatch.setattr("app.parsers.jeol_jdf_parser._MAX_JDF_FILE_BYTES", 16)
    fake = tmp_path / "huge.jdf"
    fake.write_bytes(b"JEOL.NMR" + b"\x00" * 64)

    with pytest.raises(ValueError, match="512 MiB"):
        JEOLJDFParser().parse(fake)


def test_rejects_non_jeol_jdf(tmp_path):
    fake = tmp_path / "printing-job.jdf"
    fake.write_text("<JDF />", encoding="utf-8")
    assert not JEOLJDFParser.can_parse(fake)


def test_real_jeol_query_groups_keep_singlets_without_solvent_contamination():
    spectrum = JEOLJDFParser().parse(JDF_FILE)
    result = NMRAnalyzer().analyze(spectrum)

    assert any(
        abs(item["center_ppm"] - 8.0403) < 0.001
        and item["is_isolated_singlet"]
        for item in result.multiplets
    )
    assert all(abs(item["center_ppm"] - 7.26) > 0.04 for item in result.multiplets)
    assert any(
        7.34 < item["center_ppm"] < 7.39
        and len(item["component_positions"]) == 2
        for item in result.multiplets
    )
    grouping = result.metrics["multiplet_grouping"]
    assert grouping["solvent_lines_excluded"] == 2
    assert grouping["isolated_singlets_retained"] >= 1


def test_uploads_jdf_and_preserves_original_filename(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    store_module._store = None
    with TestClient(app) as client, JDF_FILE.open("rb") as handle:
        response = client.post(
            "/api/upload",
            files={"file": (JDF_FILE.name, handle, "application/octet-stream")},
        )
        assert response.status_code == 200, response.text
        uploaded = response.json()
        assert uploaded["technique"] == "NMR"
        assert uploaded["points"] == 30000
        analyzed = client.post(
            f"/api/analyze/{uploaded['id']}", json={"expected_revision": 0}
        )
        assert analyzed.status_code == 200, analyzed.text
        assert analyzed.json()["technique"] == "NMR"
        spectra = client.get("/api/spectra").json()
        assert spectra[0]["name"] == JDF_FILE.name
