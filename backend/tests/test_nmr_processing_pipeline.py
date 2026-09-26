from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.api.store as store_module
from app.analysis.nmr_processing import (
    NMRProcessingSourceError,
    apply_phase_correction,
    asymmetric_least_squares_baseline,
    automatic_phase_correction,
    build_processing_source,
    locate_reference_peak,
    read_processing_source,
    spectrum_quality_metrics,
)
from app.core.models import Spectrum, Technique
from app.main import app
from app.parsers.nmr_parser import (
    NMRSpectrumParser,
    _decode_bruker_fid,
    _read_bruker_binary,
    _remove_digital_filter,
)

DATA_DIR = Path(__file__).parent.parent.parent / "dataexample" / "HNMRexample"


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    store_module._store = None
    return TestClient(app)


def test_processing_source_round_trip_and_checksum_guard():
    x = np.linspace(10, 0, 16)
    real = np.linspace(-2, 3, 16)
    imaginary = np.linspace(1, -1, 16)
    source = build_processing_source(
        x,
        real,
        quadrature_real_data=real,
        quadrature_imaginary_data=imaginary,
        source_kind="synthetic",
    )

    assert source["has_quadrature"] is True
    assert source["quadrature_real_source"] == "original_y_data"
    assert source["estimated_inline_json_bytes"] > source["estimated_binary_bytes"]
    restored_x, restored_y, restored_real, restored_imaginary = read_processing_source(
        source
    )
    assert np.array_equal(restored_x, x)
    assert np.array_equal(restored_y, real)
    assert np.array_equal(restored_real, real)
    assert np.array_equal(restored_imaginary, imaginary)

    source["original_y_data"][3] += 0.25
    with pytest.raises(NMRProcessingSourceError, match="checksum mismatch"):
        read_processing_source(source)


def test_processing_source_guards_replay_metadata():
    x = np.linspace(10, 0, 16)
    y = np.linspace(-1, 1, 16)
    source = build_processing_source(
        x,
        y,
        source_kind="synthetic",
        default_phase={"zero_deg": 12.5, "first_deg": -4.0},
        default_baseline={"method": "asymmetric_least_squares"},
        reference_metadata={"solvent": "CDCl3", "nucleus": "1H"},
    )

    assert source["pipeline_version"] == "nmr-processing-v2"
    read_processing_source(source)
    source["default_phase"]["zero_deg"] = 13.0
    with pytest.raises(NMRProcessingSourceError, match="metadata checksum mismatch"):
        read_processing_source(source)


def test_phase_correction_preserves_signed_information():
    x = np.linspace(1, -1, 5)
    real = np.array([2.0, 1.0, 0.0, -1.0, -2.0])
    imaginary = np.array([1.0, -3.0, 2.0, 4.0, -5.0])

    phased_real, phased_imaginary = apply_phase_correction(
        x,
        real,
        imaginary,
        zero_deg=90,
    )

    assert np.allclose(phased_real, -imaginary, atol=1e-12)
    assert np.allclose(phased_imaginary, real, atol=1e-12)
    assert np.min(phased_real) < 0


def test_automatic_zero_and_first_order_phase_recovers_absorptive_signal():
    x = np.linspace(10, 0, 4096)
    absorptive = np.exp(-((x - 7.2) / 0.03) ** 2)
    absorptive += 0.6 * np.exp(-((x - 2.1) / 0.05) ** 2)
    misphased_real, misphased_imaginary = apply_phase_correction(
        x,
        absorptive,
        np.zeros_like(absorptive),
        zero_deg=55,
        first_deg=-130,
        pivot_ppm=5,
    )

    corrected_real, corrected_imaginary, trace = automatic_phase_correction(
        x,
        misphased_real,
        misphased_imaginary,
        pivot_ppm=5,
        max_first_deg=360,
    )

    assert trace["objective_after"] < trace["objective_before"] * 1e-6
    assert trace["zero_deg"] == pytest.approx(-55, abs=1e-3)
    assert trace["first_deg"] == pytest.approx(130, abs=1e-3)
    assert np.allclose(corrected_real, absorptive, atol=1e-9)
    assert np.max(np.abs(corrected_imaginary)) < 1e-9


def test_asymmetric_baseline_flattens_quiet_regions_without_clipping():
    x = np.linspace(0, 1, 3000)
    drift = 7.0 * x**2 - 2.0 * x + 4.0
    peak = 80.0 * np.exp(-((x - 0.52) / 0.012) ** 2)
    ripple = 0.08 * np.sin(2 * np.pi * 37 * x)
    y = drift + peak + ripple

    baseline, corrected = asymmetric_least_squares_baseline(
        y,
        smoothness=1e7,
        asymmetry=0.001,
        iterations=10,
    )

    quiet = (x < 0.4) | (x > 0.65)
    assert np.ptp(corrected[quiet]) < np.ptp(y[quiet]) * 0.1
    assert np.max(corrected) > 60
    assert np.min(corrected) < 0
    assert np.isfinite(baseline).all()


def test_quality_metrics_and_reference_peak_are_json_safe_and_traceable():
    x = np.linspace(8, 6, 2001)
    drift = 0.4 * (x - 7)
    signal = 12 * np.exp(-((x - 7.30) / 0.006) ** 2)
    noise = 0.02 * np.sin(np.arange(len(x), dtype=float) * 1.618)
    y = drift + signal + noise

    quality = spectrum_quality_metrics(x, y)
    located = locate_reference_peak(
        x,
        y,
        expected_ppm=7.26,
        window_ppm=0.08,
        min_snr=5,
    )

    assert "baseline_drift" in quality["quality_flags"]
    assert quality["noise_sigma"] > 0
    assert quality["baseline_span"] > quality["noise_sigma"]
    assert located is not None
    assert located["observed_ppm"] == pytest.approx(7.30, abs=0.002)
    assert isinstance(located["snr"], float)


def test_high_imaginary_energy_is_flagged():
    x = np.linspace(8, 6, 2001)
    real = 12 * np.exp(-((x - 7.30) / 0.006) ** 2)
    imaginary = 8.0 * real

    quality = spectrum_quality_metrics(x, real, imaginary_data=imaginary)

    assert "high_imaginary_energy" in quality["quality_flags"]
    assert quality["imaginary_energy_fraction"] > 0.9


def test_bruker_binary_endianness_types_and_interleaving(tmp_path):
    int_path = tmp_path / "int32-big"
    np.array([1, -2, 3, -4], dtype=">i4").tofile(int_path)
    int_values = _read_bruker_binary(
        int_path, byte_order=1, data_type=0
    )
    assert np.array_equal(int_values, [1, -2, 3, -4])

    double_path = tmp_path / "float64-little"
    np.array([1.25, -2.5, 3.75, -4.0], dtype="<f8").tofile(double_path)
    double_values = _read_bruker_binary(
        double_path, byte_order=0, data_type=2
    )
    decoded = _decode_bruker_fid(double_values, td=4)
    assert np.allclose(decoded, [1.25 - 2.5j, 3.75 - 4.0j])


def test_bruker_group_delay_removes_integer_and_fractional_delay():
    fid = np.arange(10, dtype=float) + 1j * np.arange(10, dtype=float)
    corrected, whole, fractional = _remove_digital_filter(fid, 2.0)
    assert whole == 2
    assert fractional == 0
    assert len(corrected) == len(fid) - 4
    assert np.isfinite(corrected).all()

    fractional_corrected, whole, fractional = _remove_digital_filter(fid, 2.5)
    assert whole == 2
    assert fractional == pytest.approx(0.5)
    assert len(fractional_corrected) == len(fid) - 4
    assert np.isfinite(fractional_corrected).all()


def test_real_bruker_parser_preserves_quadrature_and_scaling():
    spectrum = NMRSpectrumParser().parse(DATA_DIR)
    source = spectrum.parameters["processing_source"]
    x, original_y, quadrature_real, quadrature_imaginary = read_processing_source(
        source
    )

    raw_first = np.fromfile(DATA_DIR / "pdata" / "1" / "1r", dtype="<i4", count=1)[0]
    assert original_y[0] == pytest.approx(float(raw_first) * 2**10)
    assert np.array_equal(x, spectrum.x_data)
    assert np.array_equal(original_y, spectrum.y_data)
    assert np.array_equal(quadrature_real, spectrum.y_data)
    assert quadrature_imaginary is not None
    assert spectrum.parameters["bruker_parameters"]["DTYPA"] == 2
    assert spectrum.parameters["bruker_parameters"]["GRPDLY"] == 76
    assert spectrum.parameters["quadrature_available"] is True


def test_bruker_raw_fid_fallback_is_finite_and_phase_ready(tmp_path):
    raw_dir = tmp_path / "raw-only"
    raw_dir.mkdir()
    for filename in ("acqu", "acqus", "fid"):
        shutil.copyfile(DATA_DIR / filename, raw_dir / filename)

    spectrum = NMRSpectrumParser().parse(raw_dir)

    assert spectrum.num_points == 32768
    assert spectrum.parameters["source_domain"] == "time"
    assert spectrum.parameters["phase_corrected"] is True
    assert spectrum.parameters["baseline_method"] == "asymmetric_least_squares"
    assert spectrum.parameters["bruker_parameters"]["group_delay_applied_points"] == 76
    assert np.isfinite(spectrum.y_data).all()
    assert np.max(spectrum.y_data) > 0
    strongest_ppm = float(spectrum.x_data[int(np.argmax(spectrum.y_data))])
    assert 2.8 < strongest_ppm < 3.2
    solvent_region = (spectrum.x_data > 7.1) & (spectrum.x_data < 7.4)
    assert float(np.max(spectrum.y_data[solvent_region])) > 0
    _, original_y, quadrature_real, quadrature_imaginary = read_processing_source(
        spectrum.parameters["processing_source"]
    )
    assert np.array_equal(original_y, spectrum.y_data)
    assert quadrature_real is not None
    assert quadrature_imaginary is not None


def test_public_spectrum_hides_processing_arrays(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        loaded = client.post(
            "/api/spectra/examples/load", json={"path": "HNMRexample.zip"}
        )
        assert loaded.status_code == 200, loaded.text
        spectrum = client.get(f"/api/spectra/{loaded.json()['id']}")
        assert spectrum.status_code == 200, spectrum.text
        parameters = spectrum.json()["parameters"]
        assert "processing_source" not in parameters
        assert parameters["processing_source_summary"]["has_quadrature"] is True
        assert parameters["processing_source_summary"]["point_count"] == 65536


def test_process_replays_original_applies_phase_and_resets(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        loaded = client.post(
            "/api/spectra/examples/load", json={"path": "HNMRexample.zip"}
        )
        assert loaded.status_code == 200, loaded.text
        sid = loaded.json()["id"]
        original = client.get(f"/api/spectra/{sid}").json()

        processed = client.post(
            f"/api/nmr/{sid}/process",
            json={
                "expected_revision": original["spectrum_revision"],
                "phase_zero_deg": 5,
                "phase_first_deg": -10,
                "phase_pivot_ppm": 7.26,
                "baseline_correct": True,
                "baseline_method": "asymmetric_least_squares",
                "crop_min_ppm": 0,
                "crop_max_ppm": 10,
                "smoothing_window": 5,
            },
        )
        assert processed.status_code == 200, processed.text
        body = processed.json()
        assert body["processing_status"]["phase"] == "applied"
        assert body["processing_status"]["committed"] is True
        assert body["spectrum_revision"] == original["spectrum_revision"] + 1
        assert len(body["x_data"]) < len(original["x_data"])
        assert "processing_source" not in body["parameters"]

        stale_reset = client.post(
            f"/api/nmr/{sid}/reset",
            json={"expected_revision": original["spectrum_revision"]},
        )
        assert stale_reset.status_code == 409
        assert stale_reset.json()["detail"]["code"] == "revision_conflict"

        reset = client.post(
            f"/api/nmr/{sid}/reset",
            json={"expected_revision": body["spectrum_revision"]},
        )
        assert reset.status_code == 200, reset.text
        reset_body = reset.json()
        assert reset_body["processing_status"]["state"] == "reset"
        assert reset_body["x_data"] == original["x_data"]
        assert reset_body["y_data"] == original["y_data"]


def test_jdf_manual_phase_uses_persisted_quadrature(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        loaded = client.post(
            "/api/spectra/examples/load",
            json={"path": "26-7-11-cyj-sja_proton-1-1.jdf"},
        )
        assert loaded.status_code == 200, loaded.text
        sid = loaded.json()["id"]
        original = client.get(f"/api/spectra/{sid}").json()

        # An explicit zero-degree phase is a useful replay invariant: it must
        # exercise the quadrature path while reproducing the imported display.
        phased = client.post(
            f"/api/nmr/{sid}/process",
            json={
                "phase_zero_deg": 0,
                "phase_first_deg": 0,
                "expected_revision": original["spectrum_revision"],
            },
        )
        assert phased.status_code == 200, phased.text
        body = phased.json()
        assert body["processing_status"]["phase"] == "applied"
        assert any(
            operation["type"] == "restore_default_baseline"
            for operation in body["processing_applied"]
        )
        assert np.allclose(body["x_data"], original["x_data"], rtol=0, atol=0)
        assert np.allclose(body["y_data"], original["y_data"], rtol=0, atol=1e-5)


def test_phase_request_without_quadrature_is_rejected(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        spectrum = Spectrum(
            technique=Technique.NMR,
            x_data=np.linspace(10, 0, 64),
            y_data=np.sin(np.linspace(0, 4 * np.pi, 64)),
            parameters={
                "processing_source": build_processing_source(
                    np.linspace(10, 0, 64),
                    np.sin(np.linspace(0, 4 * np.pi, 64)),
                    source_kind="real_only",
                )
            },
        )
        sid = store_module.get_store().add(spectrum).id
        response = client.post(
            f"/api/nmr/{sid}/process",
            json={"phase_zero_deg": 30, "expected_revision": 1},
        )
        assert response.status_code == 422
        assert response.json()["detail"]["status"] == "skipped_no_imaginary_channel"


def test_auto_phase_preview_reports_qc_and_preserves_committed_view(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch) as client:
        x = np.linspace(10, 0, 1024)
        absorptive = np.exp(-((x - 7.2) / 0.025) ** 2)
        absorptive += 0.5 * np.exp(-((x - 2.2) / 0.04) ** 2)
        real, imaginary = apply_phase_correction(
            x,
            absorptive,
            np.zeros_like(absorptive),
            zero_deg=35,
            first_deg=-80,
            pivot_ppm=5,
        )
        spectrum = Spectrum(
            technique=Technique.NMR,
            x_data=x,
            y_data=real,
            parameters={
                "nucleus": "1H",
                "solvent": "CDCl3",
                "processing_source": build_processing_source(
                    x,
                    real,
                    quadrature_real_data=real,
                    quadrature_imaginary_data=imaginary,
                    source_kind="synthetic_phase_ready",
                ),
            },
        )
        stored = store_module.get_store().add(spectrum)

        preview = client.post(
            f"/api/nmr/{stored.id}/process",
            json={
                "preview_only": True,
                "auto_phase": True,
                "phase_pivot_ppm": 5,
                "auto_phase_max_first_deg": 180,
            },
        )

        assert preview.status_code == 200, preview.text
        body = preview.json()
        operation = next(
            item
            for item in body["processing_applied"]
            if item["type"] == "phase_correct"
        )
        assert operation["mode"] == "automatic"
        assert operation["objective_after"] < operation["objective_before"]
        assert body["processing_status"]["state"] == "preview"
        assert body["processing_status"]["committed"] is False
        assert body["processing_status"]["available_views"] == [
            "original",
            "current",
            "preview",
        ]
        assert body["quality_before"]["pipeline_version"] == "nmr-processing-v2"
        assert body["quality_after"]["pipeline_version"] == "nmr-processing-v2"

        current = client.get(f"/api/nmr/{stored.id}/view?state=current")
        original = client.get(f"/api/nmr/{stored.id}/view?state=original")
        assert current.status_code == 200, current.text
        assert original.status_code == 200, original.text
        assert current.json()["y_data"] == original.json()["y_data"]
        assert current.json()["spectrum_revision"] == stored.spectrum_revision
        assert "processing_source" not in current.json()["parameters"]
        assert "quadrature_imaginary_data" not in current.text


def test_auto_phase_without_quadrature_is_skipped_without_mutation(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch) as client:
        x = np.linspace(10, 0, 128)
        y = np.sin(np.linspace(0, 4 * np.pi, 128))
        spectrum = Spectrum(
            technique=Technique.NMR,
            x_data=x,
            y_data=y,
            parameters={
                "processing_source": build_processing_source(
                    x,
                    y,
                    source_kind="real_only",
                )
            },
        )
        stored = store_module.get_store().add(spectrum)

        response = client.post(
            f"/api/nmr/{stored.id}/process",
            json={
                "auto_phase": True,
                "expected_revision": stored.spectrum_revision,
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()
        assert body["processing_status"]["state"] == "unchanged"
        assert body["processing_status"]["committed"] is False
        assert body["processing_status"]["phase"] == "skipped_no_imaginary_channel"
        assert body["processing_skipped"][0]["mode"] == "automatic"
        assert body["spectrum_revision"] == stored.spectrum_revision


def test_nmr_view_validates_identifier_and_technique(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch) as client:
        spectrum = Spectrum(
            technique=Technique.UVVIS,
            x_data=np.linspace(200, 800, 32),
            y_data=np.ones(32),
        )
        sid = store_module.get_store().add(spectrum).id

        wrong_technique = client.get(f"/api/nmr/{sid}/view")
        invalid_identifier = client.get("/api/nmr/not!safe/view")

        assert wrong_technique.status_code == 400
        assert invalid_identifier.status_code == 422


def test_auto_solvent_reference_is_previewable_and_traceable(
    tmp_path,
    monkeypatch,
):
    with _client(tmp_path, monkeypatch) as client:
        x = np.linspace(8, 6, 4001)
        y = 10 * np.exp(-((x - 7.30) / 0.004) ** 2)
        y += 0.01 * np.sin(np.arange(len(x), dtype=float))
        spectrum = Spectrum(
            technique=Technique.NMR,
            x_data=x,
            y_data=y,
            parameters={
                "nucleus": "1H",
                "solvent": "CDCl3",
                "processing_source": build_processing_source(
                    x,
                    y,
                    source_kind="reference_fixture",
                ),
            },
        )
        sid = store_module.get_store().add(spectrum).id

        response = client.post(
            f"/api/nmr/{sid}/process",
            json={
                "preview_only": True,
                "auto_reference": True,
                "reference_window_ppm": 0.08,
            },
        )

        assert response.status_code == 200, response.text
        body = response.json()
        operation = next(
            item
            for item in body["processing_applied"]
            if item["type"] == "reference_shift"
        )
        assert operation["source"] == "automatic_solvent"
        assert operation["current_ppm"] == pytest.approx(7.30, abs=0.001)
        assert operation["target_ppm"] == pytest.approx(7.26)
        strongest = int(np.argmax(np.abs(body["y_data"])))
        assert body["x_data"][strongest] == pytest.approx(7.26, abs=0.001)
