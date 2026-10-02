from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import app.ml.forward_v1.csp5_scorer as csp5_scorer_module
import app.ml.forward_v1.factory as forward_factory
import app.ml.nmr_candidate_generation_v2 as candidate_generation
from fastapi.testclient import TestClient

from app.main import app
from app.api.routes import elucidate as route
from app.ml import nmr_structure_elucidation as engine
from app.ml.nmr_forward import NMRForwardTimeoutError


def _reference_index(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR_INDEX", str(tmp_path / "nmr-index.sqlite"))
    monkeypatch.setenv("CHEMAPP_NMR_RANKER", str(tmp_path / "missing-ranker.joblib"))
    engine._load_records_cached.cache_clear()
    conn = engine._connect()
    conn.execute(
        """
        INSERT INTO nmr_records
        (source, source_id, name, smiles, formula, mw, peaks_13c, peaks_1h, metadata)
        VALUES ('test', 'target', 'candidate',
                'CC(C)c1cnc(C(C)(C)Br)cn1', 'C10H15BrN2', 243.15,
                '[]', ?, '{}')
        """,
        (
            json.dumps(
                [
                    {"shift": 1.24},
                    {"shift": 1.55},
                    {"shift": 7.8},
                    {"shift": 8.0},
                ]
            ),
        ),
    )
    conn.commit()
    conn.close()


def test_predict_enforces_formula_and_reports_peak_preprocessing(tmp_path, monkeypatch):
    _reference_index(tmp_path, monkeypatch)
    monkeypatch.setattr(
        route,
        "_safe_generate",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "inference_time_ms": 0,
            "candidates": [],
            "validation": {"accepted": 0, "rejected": []},
        },
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/elucidate/predict",
            json={
                "formula": "C10H15N2Br",
                "solvent": "CDCl3",
                "peaks_1h": [
                    {"shift": -0.0024, "intensity": 1.0},
                    {"shift": 7.2483, "intensity": 0.3},
                    {"shift": 7.2821, "intensity": 0.2},
                    {"shift": 1.2389, "intensity": 0.7},
                    {"shift": 1.2524, "intensity": 1.0},
                    {"shift": 1.5477, "intensity": 0.8},
                    {"shift": 7.8, "intensity": 0.4},
                    {"shift": 8.0, "intensity": 0.3},
                ],
            },
        )
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["query"]["formula"] == "C10H15BrN2"
    assert data["candidate_pool_status"] == "formula_match"
    assert data["result_type"] == "candidate_ranking_not_identification"
    assert data["schema_version"] == "chemapp.nmr.hybrid-prediction.v1"
    assert data["pipeline_version"] == "1.0.0"
    assert data["decision"]["action"] == "abstain"
    assert data["decision"]["selected_candidate_id"] is None
    assert data["calibrated_probability"] is False
    assert [stage["stage"] for stage in data["pipeline"]["stages"]] == [
        "input_validation",
        "candidate_generation",
        "reference_scoring",
        "forward_scoring",
        "assignment",
        "ranking",
        "decision",
    ]
    assert data["candidates"][0]["source_id"] == "target"
    assert data["candidates"][0]["evidence"]["calibrated_probability"] is False
    assert data["forward_model"]["status"] == "unsupported_modality"
    assert data["forward_model"]["model_called"] is False
    excluded = data["query"]["preprocessing"]["1h"]["excluded"]
    assert {item["reason"] for item in excluded} == {
        "reference_peak",
        "solvent_peak",
    }


def test_predict_applies_candidate_substructure_constraints(tmp_path, monkeypatch):
    _reference_index(tmp_path, monkeypatch)
    with TestClient(app) as client:
        accepted = client.post(
            "/api/ml/elucidate/predict",
            json={
                "formula": "C10H15BrN2",
                "peaks_1h": [{"shift": 1.24}],
                "required_smarts": ["Br"],
            },
        )
        rejected = client.post(
            "/api/ml/elucidate/predict",
            json={
                "formula": "C10H15BrN2",
                "peaks_1h": [{"shift": 1.24}],
                "forbidden_smarts": ["Br"],
            },
        )

    assert accepted.status_code == 200, accepted.text
    assert len(accepted.json()["candidates"]) == 1
    assert rejected.status_code == 200, rejected.text
    assert rejected.json()["candidates"] == []
    assert rejected.json()["decision"]["reason_code"] == "no_candidates"


def test_predict_rejects_ambiguous_formula_before_inference(monkeypatch):
    called = False

    def fail_if_called(*_args, **_kwargs):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(route, "_safe_generate", fail_if_called)
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/elucidate/predict",
            json={"formula": "C6H5(OH)", "peaks_1h": [{"shift": 7.2}]},
        )
    assert response.status_code == 422
    assert called is False


def test_predict_does_not_load_experimental_generator_by_default(tmp_path, monkeypatch):
    _reference_index(tmp_path, monkeypatch)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("experimental generator must be opt-in")

    monkeypatch.setattr(route, "_safe_generate", fail_if_called)
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/elucidate/predict",
            json={"formula": "C10H15BrN2", "peaks_1h": [{"shift": 1.24}]},
        )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["generated_candidates"] == []
    assert data["generation"]["status"] == "not_requested"


def test_predict_runs_experimental_generator_only_when_requested(tmp_path, monkeypatch):
    _reference_index(tmp_path, monkeypatch)
    # 该测试钉住 legacy T5 生成路径（_safe_generate mock）；NMR2Struct 分派
    # 路径由 test_nmr2struct_generator.py 覆盖。
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "t5")
    called = False

    def generate(*_args, **_kwargs):
        nonlocal called
        called = True
        return {
            "status": "completed",
            "inference_time_ms": 1,
            "candidates": [{"rank": 1, "smiles": "CC"}],
        }

    monkeypatch.setattr(route, "_safe_generate", generate)
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/elucidate/predict",
            json={
                "formula": "C10H15BrN2",
                "peaks_1h": [{"shift": 1.24}],
                "generate_experimental": True,
            },
        )

    assert response.status_code == 200, response.text
    assert called is True
    assert response.json()["generation"]["status"] == "completed"


def test_predict_includes_nmrshiftdb2_license_notice_when_source_is_present(
    tmp_path,
    monkeypatch,
):
    _reference_index(tmp_path, monkeypatch)
    conn = engine._connect()
    conn.execute(
        """
        INSERT INTO nmr_records
        (source, source_id, name, smiles, formula, mw, peaks_13c, peaks_1h, metadata)
        VALUES ('nmrshiftdb2', 'licensed', '', 'CC', 'C2H6', 30.07,
                '[]', '[{"shift": 1.2}]', '{}')
        """
    )
    conn.commit()
    conn.close()

    with TestClient(app) as client:
        response = client.post(
            "/api/ml/elucidate/predict",
            json={"formula": "C10H15BrN2", "peaks_1h": [{"shift": 1.24}]},
        )

    assert response.status_code == 200
    notice = response.json()["data_attribution"][0]
    assert notice["source"] == "nmrshiftdb2"
    assert notice["license_uri"].endswith("nmrshiftdb2datalicense.txt")


def _shadow_ranked_result() -> dict:
    return {
        "candidates": [
            {
                "rank": 1,
                "smiles": "CC",
                "compound_name": "ethane",
                "molecular_formula": "C2H6",
                "ranking_score": 0.42,
                "evidence_level": "weak",
                "matched_13c": 1,
                "matched_1h": 0,
            }
        ],
        "candidate_pool_status": "formula_match",
        "query": {
            "formula": "C2H6",
            "modality": "13c",
            "n_13c": 1,
            "n_1h": 0,
        },
        "warnings": [],
        "index": {"by_source": {}},
        "ranker": {
            "loaded": False,
            "reason": "model_unavailable",
            "calibrated_probability": False,
        },
    }


def _configure_shadow(monkeypatch):
    monkeypatch.setenv("CHEMAPP_DP5Q_MODE", "shadow")
    monkeypatch.setenv("CHEMAPP_DP5Q_PYTHON", "fake-python")
    monkeypatch.setenv("CHEMAPP_DP5Q_REPO", "fake-repository")
    monkeypatch.setenv("CHEMAPP_DP5Q_BUSY_TIMEOUT_SECONDS", "0")


def test_forward_shadow_adds_diagnostics_without_changing_ranking(monkeypatch):
    _configure_shadow(monkeypatch)
    # 该测试钉住 legacy T5 生成路径（_safe_generate mock）。
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "t5")
    ranked = _shadow_ranked_result()
    original_candidate = deepcopy(ranked["candidates"][0])
    generated_smiles = []
    received = {}

    monkeypatch.setattr(
        route,
        "rank_candidates",
        lambda *_args, generated_smiles=None, **_kwargs: (
            generated_smiles is not None
            and generated_smiles.extend([])
            or deepcopy(ranked)
        ),
    )
    monkeypatch.setattr(
        route,
        "analyze_mixture",
        lambda *_args, **_kwargs: {
            "status": "not_evaluated",
            "is_mixture": None,
            "components": [],
        },
    )
    monkeypatch.setattr(
        route,
        "_safe_generate",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "inference_time_ms": 0,
            "candidates": [{"rank": 1, "smiles": "CCC"}],
        },
    )

    class FakeAdapter:
        def score_candidates(self, observed_13c, candidates, *, formula=None):
            received.update(
                {
                    "observed_13c": observed_13c,
                    "candidates": candidates,
                    "formula": formula,
                }
            )
            return {
                "status": "ok",
                "model": {
                    "name": "DP5q-CASCADE-mean",
                    "repository_commit": "pinned",
                    "mean_model_sha256": "mean",
                    "preprocessor_sha256": "preprocessor",
                },
                "candidates": [
                    {
                        "candidate_id": "ranked-1",
                        "relative_rank": 1,
                        "assignment_mode": "unassigned_hungarian_atom_level",
                        "matched_count": 1,
                        "observed_count": 1,
                        "predicted_atom_count": 2,
                        "unmatched_observed_count": 0,
                        "unmatched_predicted_atom_count": 1,
                        "observed_coverage": 1.0,
                        "predicted_atom_coverage": 0.5,
                        "bidirectional_coverage": 0.5,
                        "assignment_complete": False,
                        "mae_ppm": 0.8,
                        "rmse_ppm": 0.8,
                        "max_abs_error_ppm": 0.8,
                        "prediction": {
                            "canonical_smiles": "CC",
                            "conformer_count": 1,
                            "atom_predictions": [
                                {"atom_index": 0, "shift_ppm": 7.2},
                                {"atom_index": 1, "shift_ppm": 7.2},
                            ],
                            "warnings": [],
                        },
                    }
                ],
                "rank_basis": "coverage_then_error",
                "assignment_limitation": "test limitation",
            }

    monkeypatch.setattr(route, "get_nmr_forward_adapter", lambda: FakeAdapter())
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/elucidate/predict",
            json={
                "formula": "C2H6",
                "peaks_13c": [{"shift": 8.0}],
                "generate_experimental": True,
            },
        )

    assert response.status_code == 200, response.text
    data = response.json()
    candidate = data["candidates"][0]
    assert {
        key: candidate[key]
        for key in (
            "rank",
            "smiles",
            "ranking_score",
            "evidence_level",
        )
    } == {
        key: original_candidate[key]
        for key in (
            "rank",
            "smiles",
            "ranking_score",
            "evidence_level",
        )
    }
    assert candidate["forward_evidence"]["mae_ppm"] == 0.8
    assert candidate["forward_evidence"]["used_for_ranking"] is False
    assert data["forward_model"]["status"] == "completed"
    assert data["forward_model"]["calibrated_probability"] is False
    assert data["generated_candidates"][0]["smiles"] == "CCC"
    assert received["candidates"] == [{"candidate_id": "ranked-1", "smiles": "CC"}]
    assert received["formula"] == "C2H6"
    assert generated_smiles == []


def test_forward_timeout_is_fail_open_and_does_not_expose_exception(
    monkeypatch,
):
    _configure_shadow(monkeypatch)
    ranked = _shadow_ranked_result()
    monkeypatch.setattr(route, "rank_candidates", lambda *_args, **_kwargs: deepcopy(ranked))
    monkeypatch.setattr(
        route,
        "analyze_mixture",
        lambda *_args, **_kwargs: {
            "status": "not_evaluated",
            "is_mixture": None,
            "components": [],
        },
    )

    class TimeoutAdapter:
        def score_candidates(self, *_args, **_kwargs):
            raise NMRForwardTimeoutError(
                r"secret path C:\models\dp5q exceeded timeout"
            )

    monkeypatch.setattr(route, "get_nmr_forward_adapter", lambda: TimeoutAdapter())
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/elucidate/predict",
            json={"formula": "C2H6", "peaks_13c": [{"shift": 8.0}]},
        )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["candidates"][0]["rank"] == 1
    assert data["candidates"][0]["ranking_score"] == 0.42
    assert data["forward_model"]["status"] == "timed_out"
    assert data["forward_model"]["reason_code"] == "sidecar_timeout"
    assert "secret" not in response.text
    assert "C:\\models" not in response.text


def test_forward_status_does_not_start_sidecar(monkeypatch):
    _configure_shadow(monkeypatch)

    def fail_if_called():
        raise AssertionError("status endpoint must not start the sidecar")

    monkeypatch.setattr(route, "get_nmr_forward_adapter", fail_if_called)
    with TestClient(app) as client:
        response = client.get("/api/ml/elucidate/status")

    assert response.status_code == 200
    status = response.json()["forward_model"]
    assert status["enabled"] is True
    assert status["configured"] is True
    assert status["used_for_ranking"] is False
    assert status["calibrated_probability"] is False
    assert status["quantile_enabled"] is False
    external = response.json()["external_smoke_benchmark"]
    if not route.EXTERNAL_SMOKE_STATUS_PATH.is_file():
        # 精简发行版（如公开仓库）不附带 docs/ 下的 smoke 清单工件；
        # 端点必须 fail-closed 报 unavailable 而不是编造状态。
        assert external["status"] == "unavailable"
        assert external["accuracy_eligibility"]["headline_accuracy_allowed"] is False
        return
    assert external["status"] == "verified_smoke_only"
    assert external["scope"]["numeric_payloads_decoded"] == 16
    assert external["scope"]["one_dimensional_spectrum_model_supported"] == 10
    assert external["scope"]["two_dimensional_matrix_only"] == 6
    assert external["accuracy_eligibility"]["headline_accuracy_allowed"] is False


def test_forward_shadow_gate_priority_never_starts_sidecar(monkeypatch):
    def fail_if_called():
        raise AssertionError("forward adapter must not be acquired")

    monkeypatch.setattr(route, "get_nmr_forward_adapter", fail_if_called)
    _configure_shadow(monkeypatch)
    candidate = [{"rank": 1, "smiles": "CC"}]

    unsupported = route._run_forward_shadow([], deepcopy(candidate), "C2H6")
    no_candidates = route._run_forward_shadow(
        [{"shift": 8.0}],
        [],
        "C2H6",
    )
    monkeypatch.setenv("CHEMAPP_DP5Q_MODE", "off")
    disabled = route._run_forward_shadow(
        [{"shift": 8.0}],
        deepcopy(candidate),
        "C2H6",
    )
    monkeypatch.setenv("CHEMAPP_DP5Q_MODE", "shadow")
    monkeypatch.delenv("CHEMAPP_DP5Q_PYTHON")
    not_configured = route._run_forward_shadow(
        [{"shift": 8.0}],
        deepcopy(candidate),
        "C2H6",
    )

    assert unsupported["status"] == "unsupported_modality"
    assert no_candidates["status"] == "no_candidates"
    assert disabled["status"] == "disabled"
    assert not_configured["status"] == "not_configured"
    assert all(
        result["model_called"] is False
        for result in (
            unsupported,
            no_candidates,
            disabled,
            not_configured,
        )
    )


def test_forward_shadow_busy_gate_does_not_queue_or_start_model(monkeypatch):
    _configure_shadow(monkeypatch)

    def fail_if_called():
        raise AssertionError("busy gate must not acquire the adapter")

    monkeypatch.setattr(route, "get_nmr_forward_adapter", fail_if_called)
    assert route._FORWARD_SEMAPHORE.acquire(blocking=False)
    try:
        result = route._run_forward_shadow(
            [{"shift": 8.0}],
            [{"rank": 1, "smiles": "CC"}],
            "C2H6",
        )
    finally:
        route._FORWARD_SEMAPHORE.release()

    assert result["status"] == "busy"
    assert result["reason_code"] == "concurrency_limit"
    assert result["model_called"] is False


class _FakeCsp5Scorer:
    pass


def test_cached_forward_scorer_off_returns_none(monkeypatch) -> None:
    route._forward_scorer_cache.clear()
    monkeypatch.setenv("CHEMAPP_CSP5_MODE", "off")
    assert route._cached_forward_scorer() is None


def test_cached_forward_scorer_auto_allows_local_fallback(monkeypatch) -> None:
    route._forward_scorer_cache.clear()
    monkeypatch.setenv("CHEMAPP_CSP5_MODE", "auto")
    local = object()
    monkeypatch.setattr(
        forward_factory,
        "get_forward_scorer",
        lambda prefer: local,
    )
    assert route._cached_forward_scorer() is local


def test_cached_forward_scorer_on_rejects_local_fallback(monkeypatch) -> None:
    route._forward_scorer_cache.clear()
    monkeypatch.setenv("CHEMAPP_CSP5_MODE", "on")
    local = object()
    monkeypatch.setattr(
        forward_factory,
        "get_forward_scorer",
        lambda prefer: local,
    )
    assert route._cached_forward_scorer() is None


def test_cached_forward_scorer_on_accepts_csp5(monkeypatch) -> None:
    route._forward_scorer_cache.clear()
    monkeypatch.setenv("CHEMAPP_CSP5_MODE", "on")
    fake = _FakeCsp5Scorer()
    monkeypatch.setattr(
        csp5_scorer_module,
        "Csp5ForwardScorer",
        _FakeCsp5Scorer,
    )
    monkeypatch.setattr(
        forward_factory,
        "get_forward_scorer",
        lambda prefer: fake,
    )
    assert route._cached_forward_scorer() is fake


def test_candidate_provider_path_points_to_backend_data(monkeypatch) -> None:
    monkeypatch.setenv("CHEMAPP_HYBRID_GENERATION", "on")
    route._candidate_providers_cache.clear()
    captured: dict[str, str] = {}

    class _FakePubChem:
        def __init__(self, path, *, offline):
            captured["pubchem"] = str(path)
            self.offline = offline

    class _FakeProvider:
        def __init__(self, index_path, pubchem):
            captured["index"] = str(index_path)
            captured["provider"] = str(pubchem)

    monkeypatch.setattr(
        candidate_generation,
        "PubChemFormulaSource",
        _FakePubChem,
    )
    monkeypatch.setattr(
        candidate_generation,
        "HybridOpenWorldProvider",
        _FakeProvider,
    )

    providers = route._cached_candidate_providers()

    assert providers
    # 跨平台断言：不依赖 Windows 反斜杠分隔符（Linux 上路径为 POSIX 形式）。
    pubchem_parts = Path(captured["pubchem"]).parts
    assert pubchem_parts[-4:-1] == ("data", "derived", "nmr-candidate-generation-v2")
    index_parts = Path(captured["index"]).parts
    assert index_parts[-2:] == ("data", "nmr_spectral_index_v2.sqlite")
    assert "app" not in index_parts
