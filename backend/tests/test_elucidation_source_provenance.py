"""Source binding and provenance contracts; no trained checkpoint is required."""
from copy import deepcopy
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np
import pytest

from app.api import deps
from app.api.routes import elucidate as route
from app.core.models import Peak, SampleInfo, Spectrum, Technique
from app.ml import nmr2struct_generator as generator
from app.ml.nmr_evidence import validate_generated_smiles


@pytest.fixture
def source_client(monkeypatch):
    rows = {
        "proton": SimpleNamespace(
            spectrum=Spectrum(Technique.NMR, np.array([0., 1., 2.]), np.array([0., 1., 0.]),
                              x_unit="ppm", parameters={"nucleus": "1H"}, metadata=SampleInfo(solvent="DMSO")),
            result=SimpleNamespace(peaks=[Peak(1.2, 1.)], multiplets=[])),
        "carbon": SimpleNamespace(
            spectrum=Spectrum(Technique.NMR, np.array([0., 10., 20.]), np.array([0., 1., 0.]),
                              x_unit="ppm", parameters={"nucleus": "13C"}),
            result=SimpleNamespace(peaks=[Peak(18.3, 1.), Peak(58.1, 1.)], multiplets=[])),
    }
    monkeypatch.setattr(deps, "get_store", lambda: SimpleNamespace(get=rows.get))
    calls = []

    def generate(p13, p1h, formula, top_k, num_beams, spectrum_1h=None):
        calls.append({"p13": p13, "p1h": p1h, "spectrum_1h": spectrum_1h})
        return {"status": "completed", "candidates": []}

    monkeypatch.setattr(route, "_dispatch_generation", generate)
    monkeypatch.setattr(route, "_cached_forward_scorer", lambda: None)
    monkeypatch.setattr(route, "_cached_candidate_providers", lambda: ())
    monkeypatch.setattr(route, "hybrid_rank_candidates", lambda **kw: {
        "schema_version": "test", "pipeline_version": "test", "decision": {},
        "evidence_state": "unverified", "uncertainty_kind": "test", "calibrated_probability": False,
        "stages": [], "reference_context": {}, "forward_context": {},
    })
    monkeypatch.setattr(route, "hybrid_to_legacy_rank_result", lambda result: {
        "candidates": [], "index": {}, "query": {}, "candidate_pool_status": "empty", "ranker": {},
    })
    monkeypatch.setattr(route, "_run_forward_shadow", lambda *a: {})
    monkeypatch.setattr(route, "analyze_mixture", lambda *a, **kw: {})
    app = FastAPI()
    app.include_router(route.router)
    with TestClient(app) as client:
        yield client, rows, calls


def direct_payload(**kw):
    return {"peaks_13c": [{"shift": 18.3}], "spectrum_1h_id": "proton", "generate_experimental": True, **kw}


@pytest.mark.parametrize("ids", [("carbon", "proton"), ("proton", "carbon")])
def test_combined_forwards_selected_continuous_proton_in_both_orders(source_client, ids):
    client, rows, calls = source_client
    response = client.post("/api/ml/elucidate/predict/combined", json={
        "id1": ids[0], "id2": ids[1], "generate_experimental": True,
    })
    assert response.status_code == 200, response.text
    assert response.json()["query"]["spectrum_1h_id"] == "proton"
    assert calls[0]["p13"] and calls[0]["p1h"]
    np.testing.assert_array_equal(calls[0]["spectrum_1h"][0], rows["proton"].spectrum.x_data)
    np.testing.assert_array_equal(calls[0]["spectrum_1h"][1], rows["proton"].spectrum.y_data)


@pytest.mark.parametrize("nucleus", ["", "19F", "1H-decoupled", None])
def test_combined_rejects_unknown_or_unrelated_nuclei(source_client, nucleus):
    client, rows, calls = source_client
    rows["proton"].spectrum.parameters["nucleus"] = nucleus
    response = client.post("/api/ml/elucidate/predict/combined", json={"id1": "carbon", "id2": "proton"})
    assert response.status_code == 422
    assert not calls


@pytest.mark.parametrize("source,expected", [("missing", 404), ("carbon", 422)])
def test_direct_rejects_missing_or_wrong_nucleus_proton_source(source_client, source, expected):
    client, _, calls = source_client
    response = client.post("/api/ml/elucidate/predict", json=direct_payload(spectrum_1h_id=source))
    assert response.status_code == expected
    assert not calls


@pytest.mark.parametrize("unit", ["", "s", "Hz", "seconds"])
@pytest.mark.parametrize("combined", [False, True])
def test_non_ppm_sources_fail_closed(source_client, unit, combined):
    client, rows, calls = source_client
    rows["proton"].spectrum.x_unit = unit
    endpoint = "/api/ml/elucidate/predict"
    payload = direct_payload()
    if combined:
        endpoint += "/combined"
        payload = {"id1": "carbon", "id2": "proton", "generate_experimental": True}
    response = client.post(endpoint, json=payload)
    assert response.status_code == 422
    assert not calls


@pytest.mark.parametrize("x,y", [
    ([1.], [1.]), ([1., 2.], [1.]), ([[1., 2.]], [[1., 2.]]),
    ([1., 1.], [1., 2.]), ([1., 3., 2.], [1., 2., 3.]),
    ([1., float("nan")], [1., 2.]), ([1., 2.], [1., float("inf")]),
    ([1., 2.], [0., 0.]), ([40., 41.], [1., 2.]),
])
def test_invalid_continuous_sources_fail_before_inference(source_client, x, y):
    client, rows, calls = source_client
    rows["proton"].spectrum.x_data = np.asarray(x)
    rows["proton"].spectrum.y_data = np.asarray(y)
    response = client.post("/api/ml/elucidate/predict", json=direct_payload())
    assert response.status_code == 422
    assert not calls


def test_processed_fid_with_current_ppm_axis_remains_valid(source_client):
    client, rows, calls = source_client
    rows["proton"].spectrum.parameters["source_domain"] = "time"
    response = client.post("/api/ml/elucidate/predict", json=direct_payload())
    assert response.status_code == 200
    assert calls[0]["spectrum_1h"] is not None


@pytest.mark.parametrize("key,shift", [("peaks_1h", -6.), ("peaks_1h", 100.), ("peaks_13c", 301.)])
def test_peak_domains_are_not_silently_filtered(source_client, key, shift):
    client, _, calls = source_client
    response = client.post("/api/ml/elucidate/predict", json={key: [{"shift": shift}, {"shift": 1.2}]})
    assert response.status_code == 422
    assert not calls


def test_validator_preserves_origin_without_trusting_scores():
    provenance = {"model": {"name": "NMR2Struct", "variant": "hnmr_only"},
                  "input_mode": "1h_spectrum", "prompt_schema": "nmr2struct-1h-spectrum-v1"}
    candidates, validation = validate_generated_smiles([
        {"smiles": "CCO", "origin": "nmr2struct_generated", "score": 0.99,
         "calibrated_probability": True, "evidence_level": "verified"},
        {"smiles": "OCC"}, {"smiles": "CC"},
    ], formula="C2H6O", generator="nmr2struct", provenance=provenance)
    assert len(candidates) == 1
    assert candidates[0]["source"] == "nmr2struct-generation-experimental"
    assert candidates[0]["origin"] == "nmr2struct_generated"
    assert candidates[0]["model"] == provenance["model"]
    assert candidates[0]["input_mode"] == "1h_spectrum"
    assert candidates[0]["evidence_level"] == "unverified"
    assert candidates[0]["calibrated_probability"] is False
    assert "score" not in candidates[0]
    assert validation["accepted"] == 1


@pytest.mark.parametrize("variant,input_mode,schema", [
    ("cnmr_only", "13c_peaks", "nmr2struct-13c-peaks-v1"),
    ("hnmr_only", "1h_spectrum", "nmr2struct-1h-spectrum-v1"),
    ("multitask", "13c_peaks+1h_spectrum", "nmr2struct-13c-peaks+1h-spectrum-v1"),
])
def test_dispatch_retains_actual_model_and_input_provenance(monkeypatch, variant, input_mode, schema):
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "nmr2struct")
    payload = {"status": "ok", "candidates": [{"smiles": "CCO", "origin": "nmr2struct_generated"}],
               "model": {"name": "NMR2Struct", "variant": variant, "checkpoint": f"{variant}_checkpoint.pt"},
               "input_mode": input_mode, "prompt_schema": schema,
               "provided_modalities": input_mode.split("+"), "used_modalities": input_mode.split("+"),
               "ignored_modalities": [], "inference_time_ms": 1,
               "input_warnings": ["13c_shifts_use_boundary_bins"], "observed_carbon_lower_bound": 2}
    monkeypatch.setattr(generator, "generate_candidates", lambda *a, **kw: deepcopy(payload))
    result = route._dispatch_generation([{"shift": 18.3}], [], "C2H6O", 5, 5)
    for key in ("model", "input_mode", "prompt_schema", "used_modalities", "input_warnings", "observed_carbon_lower_bound"):
        assert result[key] == payload[key]
        assert result["candidates"][0][key] == payload[key]
    assert result["status"] == "completed"
    assert result["candidates"][0]["source"] == "nmr2struct-generation-experimental"
    assert result["calibrated_probability"] is False


def test_auto_availability_receives_actual_evidence(monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "auto")
    received = {}
    def available(peaks, **kw):
        received.update(peaks=peaks, **kw)
        return True
    monkeypatch.setattr(generator, "available", available)
    monkeypatch.setattr(generator, "generate_candidates", lambda *a, **kw: {"status": "ok", "candidates": []})
    spectrum = ([0., 1., 2.], [0., 1., 0.])
    route._dispatch_generation([], [{"shift": 1.2}], "C2H6O", 5, 5, spectrum_1h=spectrum)
    assert received == {"peaks": [], "formula": "C2H6O", "spectrum_1h": spectrum}


def test_combined_uses_one_atomic_proton_snapshot_even_if_source_changes(source_client, monkeypatch):
    client, rows, calls = source_client
    rows["proton"].spectrum_revision = 3
    updated = deepcopy(rows["proton"])
    updated.spectrum_revision = 4
    updated.spectrum.y_data = np.array([3., 4., 5.])
    reads = []
    def get(sid):
        reads.append(sid)
        if sid == "proton" and reads.count("proton") > 1:
            return updated
        return rows.get(sid)
    monkeypatch.setattr(deps, "get_store", lambda: SimpleNamespace(get=get))
    response = client.post("/api/ml/elucidate/predict/combined", json={
        "id1": "proton", "id2": "carbon", "generate_experimental": True,
    })
    assert response.status_code == 200
    assert reads.count("proton") == 1
    assert response.json()["query"]["spectrum_1h_revision"] == 3
    np.testing.assert_array_equal(calls[0]["spectrum_1h"][1], [0., 1., 0.])


def test_combined_invalid_stored_peak_returns_client_error(source_client):
    client, rows, calls = source_client
    rows["proton"].result.peaks[0].position = 90.
    response = client.post("/api/ml/elucidate/predict/combined", json={"id1": "proton", "id2": "carbon"})
    assert response.status_code == 422
    assert not calls


@pytest.mark.parametrize("reason", ["no_1h_spectrum", "unsupported_variant", "invalid_1h_spectrum", "out_of_domain_heavy_atoms"])
def test_auto_dispatch_does_not_conceal_invalid_input_with_t5(monkeypatch, reason):
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "auto")
    monkeypatch.setattr(generator, "available", lambda *a, **kw: False)
    monkeypatch.setattr(generator, "resolve_input_mode", lambda *a, **kw: {"status": "unavailable", "reason": reason})
    monkeypatch.setattr(route, "_safe_generate", lambda *a, **kw: pytest.fail("must fail closed"))
    result = route._dispatch_generation([{"shift": 18.3}], [], "C2H6O", 5, 5)
    assert result["status"] == "generation_unavailable"
    assert result["error_code"] == reason
    assert result["generator"] == "nmr2struct"


def test_status_reports_nmr2struct_assets_without_claiming_runtime_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(route, "MODEL_DIR", tmp_path / "missing-t5")
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "nmr2struct")
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", "hnmr_only")
    monkeypatch.setattr(generator, "checkpoint_status", lambda: {"cnmr_only": False, "hnmr_only": True, "multitask": False})
    monkeypatch.setattr(route, "index_status", lambda: {})
    monkeypatch.setattr(route, "_v2_index_status", lambda: {})
    monkeypatch.setattr(route, "_external_smoke_status", lambda: {})
    status = route.get_status()["experimental_generation"]
    assert status["available"] is True
    assert status["t5_checkpoint_available"] is False
    assert status["availability_scope"] == "checkpoint_assets_only"
    assert status["input_compatibility_required"] is True
    assert status["runtime_verified"] is False


def test_generation_input_warnings_reach_top_level_response(source_client, monkeypatch):
    client, _, _ = source_client
    monkeypatch.setattr(route, "_dispatch_generation", lambda *a, **kw: {
        "status": "completed", "candidates": [], "input_warnings": ["13c_shifts_use_boundary_bins"],
    })
    response = client.post("/api/ml/elucidate/predict", json=direct_payload())
    assert response.status_code == 200
    assert "13c_shifts_use_boundary_bins" in response.json()["warnings"]
    assert response.json()["generation"]["input_warnings"] == ["13c_shifts_use_boundary_bins"]


def test_fully_auto_keeps_valid_t5_proton_peak_only_path(monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "auto")
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", "auto")
    monkeypatch.setattr(generator, "_available_variant", lambda variant: False)
    monkeypatch.setattr(route, "_safe_generate", lambda *a, **kw: {
        "status": "completed", "candidates": [{"smiles": "CCO"}],
    })
    result = route._dispatch_generation([], [{"shift": 1.2}], "C2H6O", 5, 5)
    assert result["generator"] == "t5"
    assert result["status"] == "completed"
    assert result["used_modalities"] == ["1h_peaks"]
    assert result["ignored_modalities"] == []
    assert result["fallback_from"]["reason"] == "no_13c_peaks"
    assert result["candidates"][0]["source"] == "t5-generation-experimental"


@pytest.mark.parametrize("variant", ["cnmr_only", "multitask", "hnmr_only"])
def test_explicit_variant_does_not_fallback_for_peak_only_proton_request(monkeypatch, variant):
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "auto")
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", variant)
    monkeypatch.setattr(route, "_safe_generate", lambda *a, **kw: pytest.fail("explicit variant cannot silently fall back"))
    result = route._dispatch_generation([], [{"shift": 1.2}], "C2H6O", 5, 5)
    assert result["generator"] == "nmr2struct"
    assert result["status"] == "generation_unavailable"
    assert result["error_code"] in {"no_13c_peaks", "no_1h_spectrum"}


@pytest.mark.parametrize("choice", ["t5", "auto"])
def test_failed_t5_generation_does_not_claim_consumed_modalities(monkeypatch, choice):
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", choice)
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", "auto")
    monkeypatch.setattr(generator, "_available_variant", lambda variant: False)
    monkeypatch.setattr(route, "_safe_generate", lambda *args: {
        "status": "generation_unavailable", "candidates": [], "error_code": "generation_failed",
    })
    result = route._dispatch_generation([{ "shift": 18.3}], [{"shift": 1.2}], None, 5, 5)
    assert result["status"] == "generation_unavailable"
    assert result["generator"] == "t5"
    assert result["provided_modalities"] == ["13c_peaks", "1h_peaks"]
    assert result["used_modalities"] == []
    assert result["candidates"] == []
    assert result["calibrated_probability"] is False
    if choice == "auto":
        assert result["fallback_from"]["generator"] == "nmr2struct"
