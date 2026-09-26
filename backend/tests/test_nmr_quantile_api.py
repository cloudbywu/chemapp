from __future__ import annotations

from fastapi.testclient import TestClient

from app.api.routes import elucidate as route
from app.main import app
from app.ml.nmr_forward import NMRForwardTimeoutError
from app.ml.nmr_quantile_forward import DP5Q_UNASSIGNED_SHADOW_SEMANTICS


def _configure_quantile_shadow(monkeypatch, *, maximum: int = 3) -> None:
    monkeypatch.setenv("CHEMAPP_DP5Q_QUANTILE_MODE", "shadow")
    monkeypatch.setenv("CHEMAPP_DP5Q_QUANTILE_MAX_CANDIDATES", str(maximum))
    monkeypatch.setenv("CHEMAPP_DP5Q_QUANTILE_BUSY_TIMEOUT_SECONDS", "0")
    monkeypatch.setenv("CHEMAPP_DP5Q_PYTHON", "fake-python")
    monkeypatch.setenv("CHEMAPP_DP5Q_REPO", "fake-repository")
    monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("CHEMAPP_LOCAL_ADMIN_BYPASS", "1")


def _quantile_result() -> dict:
    return {
        "status": "ok",
        "nucleus": "13C",
        "evidence_kind": DP5Q_UNASSIGNED_SHADOW_SEMANTICS,
        "diagnostic_only": True,
        "used_for_ranking": False,
        "rank_basis": (
            "higher_bidirectional_coverage_then_fewer_unmatched_then_"
            "higher_dp5q_shadow_score_then_lower_q50_assignment_mae"
        ),
        "assignment_limitation": (
            "Official DP5q requires atom-assigned 13C shifts; this result uses "
            "diagnostic Hungarian assignment."
        ),
        "official_assignment_semantics": False,
        "official_equation_parity": True,
        "official_workflow_parity": False,
        "conformer_protocol": "chemapp.dp5q-conformer.v2",
        "calibrated_probability": False,
        "quantile_enabled": True,
        "model": {
            "name": "DP5q-CASCADE-99quantiles",
            "repository_commit": "b79968cf63cb282e8871d5595ea6cef5b4dc0d49",
        },
        "runtime": {
            "protocol_version": "chemapp.dp5q-forward.v1",
            "tensorflow_version": "2.14.0",
            "keras_version": "2.14.0",
        },
        "candidates": [
            {
                "candidate_id": "quantile-1",
                "relative_rank": 1,
                "score_semantics": DP5Q_UNASSIGNED_SHADOW_SEMANTICS,
                "assignment_mode": "unassigned_hungarian_q50_atom_level",
                "official_assignment_semantics": False,
                "official_equation_parity": True,
                "official_workflow_parity": False,
                "conformer_protocol": "chemapp.dp5q-conformer.v2",
                "diagnostic_only": True,
                "used_for_ranking": False,
                "matched_count": 2,
                "observed_count": 2,
                "predicted_atom_count": 2,
                "unmatched_observed_count": 0,
                "unmatched_predicted_atom_count": 0,
                "observed_coverage": 1.0,
                "predicted_atom_coverage": 1.0,
                "bidirectional_coverage": 1.0,
                "assignment_complete": True,
                "q50_assignment_mae_ppm": 0.4,
                "dp5q_shadow_score": 0.81,
                "calibrated_probability": False,
                "quantile_enabled": True,
                "equation_details": {
                    "official_equation_parity": True,
                    "official_workflow_parity": False,
                    "official_assignment_semantics": False,
                    "conformer_protocol": "chemapp.dp5q-conformer.v2",
                },
                "prediction": {
                    "candidate_id": "quantile-1",
                    "canonical_smiles": "CCO",
                    "conformer_count": 1,
                    "conformer_populations": [1.0],
                    "atom_predictions": [
                        {
                            "atom_index": 0,
                            "quantiles_ppm": [21.0] * 99,
                            "conformer_mu_ppm": [21.0],
                            "conformer_sigma_ppm": [1.5],
                            "conformer_quantile_crossing_counts": [2],
                            "conformer_max_quantile_crossing_ppm": [0.05],
                        },
                        {
                            "atom_index": 1,
                            "quantiles_ppm": [60.0] * 99,
                            "conformer_mu_ppm": [60.0],
                            "conformer_sigma_ppm": [1.5],
                            "conformer_quantile_crossing_counts": [0],
                            "conformer_max_quantile_crossing_ppm": [0.0],
                        },
                    ],
                    "warnings": ["minor_raw_quantile_crossing"],
                },
            }
        ],
    }


def test_quantile_shadow_is_off_by_default_and_does_not_change_mean_contract(
    monkeypatch,
):
    monkeypatch.delenv("CHEMAPP_DP5Q_QUANTILE_MODE", raising=False)
    with TestClient(app) as client:
        response = client.get("/api/ml/elucidate/status")

    assert response.status_code == 200
    forward = response.json()["forward_model"]
    assert forward["quantile_enabled"] is False
    assert forward["used_for_ranking"] is False
    quantile = forward["quantile_shadow"]
    assert quantile["mode"] == "off"
    assert quantile["enabled"] is False
    assert quantile["used_for_ranking"] is False
    assert quantile["official_assignment_semantics"] is False
    assert quantile["calibrated_probability"] is False
    assert quantile["quantile_enabled"] is True


def test_quantile_shadow_success_is_compact_and_never_claims_probability(
    monkeypatch,
):
    _configure_quantile_shadow(monkeypatch)

    class FakeAdapter:
        def score_candidates(self, observed, candidates, *, formula):
            assert observed == [21.0, 60.0]
            assert candidates == [
                {"candidate_id": "quantile-1", "smiles": "CCO"}
            ]
            assert formula == "C2H6O"
            return _quantile_result()

    monkeypatch.setattr(
        route,
        "get_nmr_quantile_forward_adapter",
        lambda: FakeAdapter(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/elucidate/dp5q/quantile-shadow",
            json={
                "observed_shifts_ppm": [21.0, 60.0],
                "candidate_smiles": ["CCO"],
                "formula": "C2H6O",
            },
        )

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["schema_version"] == "dp5q-quantile-shadow-v1"
    assert result["used_for_ranking"] is False
    assert result["diagnostic_only"] is True
    assert result["official_assignment_semantics"] is False
    assert result["official_equation_parity"] is True
    assert result["official_workflow_parity"] is False
    assert result["calibrated_probability"] is False
    assert result["quantile_enabled"] is True
    candidate = result["candidates"][0]
    assert candidate["score_semantics"] == DP5Q_UNASSIGNED_SHADOW_SEMANTICS
    assert candidate["official_equation_parity"] is True
    assert candidate["official_workflow_parity"] is False
    assert candidate["used_for_ranking"] is False
    assert candidate["dp5q_shadow_score"] == 0.81
    assert candidate["q50_assignment_mae_ppm"] == 0.4
    assert candidate["prediction_summary"]["quantile_levels"] == 99
    assert candidate["prediction_summary"]["quantile_crossing_count"] == 2
    assert candidate["prediction_summary"]["max_quantile_crossing_ppm"] == 0.05
    assert "prediction" not in candidate
    assert "quantiles_ppm" not in str(candidate)


def test_quantile_shadow_disabled_does_not_construct_adapter(monkeypatch):
    monkeypatch.setenv("CHEMAPP_DP5Q_QUANTILE_MODE", "off")
    monkeypatch.delenv("CHEMAPP_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("CHEMAPP_LOCAL_ADMIN_BYPASS", "1")
    monkeypatch.setattr(
        route,
        "get_nmr_quantile_forward_adapter",
        lambda: (_ for _ in ()).throw(
            AssertionError("disabled endpoint must not construct adapter")
        ),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/elucidate/dp5q/quantile-shadow",
            json={
                "observed_shifts_ppm": [21.0],
                "candidate_smiles": ["CCO"],
            },
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "DP5q quantile shadow is disabled"


def test_quantile_shadow_requires_admin_not_only_access_token(monkeypatch):
    _configure_quantile_shadow(monkeypatch)
    monkeypatch.setenv("CHEMAPP_ACCESS_TOKEN", "access-token-quantile-test")
    monkeypatch.setenv("CHEMAPP_ADMIN_TOKEN", "admin-token-quantile-test")
    monkeypatch.setattr(
        route,
        "get_nmr_quantile_forward_adapter",
        lambda: (_ for _ in ()).throw(
            AssertionError("unauthorized request must not construct adapter")
        ),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/elucidate/dp5q/quantile-shadow",
            headers={"X-ChemApp-Access-Token": "access-token-quantile-test"},
            json={
                "observed_shifts_ppm": [21.0],
                "candidate_smiles": ["CCO"],
            },
        )

    assert response.status_code == 401
    assert "admin token" in response.json()["detail"]


def test_quantile_status_rejects_invalid_shared_runtime_configuration(
    monkeypatch,
):
    _configure_quantile_shadow(monkeypatch)
    monkeypatch.setenv("CHEMAPP_DP5Q_TIMEOUT_SECONDS", "not-a-number")
    monkeypatch.setattr(
        route,
        "get_nmr_quantile_forward_adapter",
        lambda: (_ for _ in ()).throw(
            AssertionError("invalid configuration must not construct adapter")
        ),
    )
    with TestClient(app) as client:
        status_response = client.get("/api/ml/elucidate/status")
        endpoint_response = client.post(
            "/api/ml/elucidate/dp5q/quantile-shadow",
            json={
                "observed_shifts_ppm": [21.0],
                "candidate_smiles": ["CCO"],
            },
        )

    quantile = status_response.json()["forward_model"]["quantile_shadow"]
    assert quantile["enabled"] is False
    assert quantile["configuration_valid"] is False
    assert quantile["asset_verification"] == "deferred_until_sidecar_start"
    assert quantile["runtime_health"] == "not_probed_by_status_endpoint"
    assert endpoint_response.status_code == 503
    assert "invalid server configuration" in endpoint_response.json()["detail"]


def test_quantile_shadow_enforces_server_candidate_limit(monkeypatch):
    _configure_quantile_shadow(monkeypatch, maximum=1)
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/elucidate/dp5q/quantile-shadow",
            json={
                "observed_shifts_ppm": [21.0],
                "candidate_smiles": ["CCO", "CCC"],
            },
        )

    assert response.status_code == 422
    assert "configured quantile limit" in response.json()["detail"]


def test_quantile_shadow_timeout_releases_shared_capacity(monkeypatch):
    _configure_quantile_shadow(monkeypatch)

    class TimeoutAdapter:
        def score_candidates(self, *_args, **_kwargs):
            raise NMRForwardTimeoutError("sensitive internal timeout")

    monkeypatch.setattr(
        route,
        "get_nmr_quantile_forward_adapter",
        lambda: TimeoutAdapter(),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/elucidate/dp5q/quantile-shadow",
            json={
                "observed_shifts_ppm": [21.0],
                "candidate_smiles": ["CCO"],
            },
        )

    assert response.status_code == 504
    assert "sensitive" not in response.text
    assert route._FORWARD_SEMAPHORE.acquire(blocking=False) is True
    route._FORWARD_SEMAPHORE.release()


def test_quantile_shadow_rejects_unsupported_candidate_before_model(
    monkeypatch,
):
    _configure_quantile_shadow(monkeypatch)
    monkeypatch.setattr(
        route,
        "get_nmr_quantile_forward_adapter",
        lambda: (_ for _ in ()).throw(
            AssertionError("unsafe candidate must not reach the adapter")
        ),
    )
    with TestClient(app) as client:
        response = client.post(
            "/api/ml/elucidate/dp5q/quantile-shadow",
            json={
                "observed_shifts_ppm": [21.0],
                "candidate_smiles": ["C[Na]"],
            },
        )

    assert response.status_code == 422
    assert "supported model contract" in response.json()["detail"]
