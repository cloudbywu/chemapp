from __future__ import annotations

import json
from typing import Any

import pytest

from app.ml.nmr_hybrid_predictor import (
    CandidateConstraints,
    HybridPredictorConfig,
    HybridPredictorInputError,
    HybridPredictorV1,
    SCHEMA_VERSION,
    hybrid_to_legacy_rank_result,
)


def _breakdown(c13_score: float, h1_score: float) -> dict[str, Any]:
    return {
        "c13": {
            "score": c13_score,
            "matched": 2,
            "mae": 0.2,
            "query_recall": 1.0,
            "reference_coverage": 1.0,
            "reference_count": 2,
            "assignments": [
                {"query_shift": 20.0, "reference_shift": 20.1, "error": 0.1}
            ],
        },
        "h1": {
            "score": h1_score,
            "matched": 2,
            "mae": 0.02,
            "query_recall": 1.0,
            "reference_coverage": 1.0,
            "reference_count": 2,
            "assignments": [
                {"query_shift": 1.0, "reference_shift": 1.01, "error": 0.01}
            ],
        },
        "formula_match": True,
    }


def _reference_ranker(**_: Any) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "rank": 1,
                "smiles": "CCO",
                "compound_name": "ethanol",
                "molecular_formula": "C2H6O",
                "molecular_weight": 46.069,
                "source": "test-library",
                "source_id": "ethanol",
                "ranking_score": 0.9,
                "evidence_level": "strong",
                "score_breakdown": _breakdown(0.9, 0.8),
            },
            {
                "rank": 2,
                "smiles": "COC",
                "compound_name": "dimethyl ether",
                "molecular_formula": "C2H6O",
                "molecular_weight": 46.069,
                "source": "test-library",
                "source_id": "ether",
                "ranking_score": 0.3,
                "evidence_level": "moderate",
                "score_breakdown": _breakdown(0.3, 0.4),
            },
        ],
        "candidate_pool_status": "formula_match",
        "query": {},
        "warnings": ["reference warning"],
        "index": {"total_records": 2},
        "ranker": {
            "loaded": False,
            "calibrated_probability": False,
        },
    }


class _ForwardScorer:
    def __init__(self, *, calibrated_probability: bool = False) -> None:
        self.called = 0
        self.calibrated_probability = calibrated_probability

    def score_candidates(
        self,
        observed_13c,
        candidates,
        *,
        formula=None,
    ):
        self.called += 1
        assert observed_13c == [20.0, 30.0]
        assert formula == "C2H6O"
        ordered = sorted(
            candidates,
            key=lambda candidate: 0 if candidate["smiles"] == "COC" else 1,
        )
        return {
            "status": "ok",
            "calibrated_probability": self.calibrated_probability,
            "quantile_enabled": False,
            "model": {"name": "fake-13c-point-model"},
            "candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "relative_rank": rank,
                    "assignment_mode": "test_hungarian",
                    "matched_count": 2,
                    "observed_count": 2,
                    "predicted_atom_count": 2,
                    "unmatched_observed_count": 0,
                    "unmatched_predicted_atom_count": 0,
                    "bidirectional_coverage": 1.0,
                    "mae_ppm": 0.2 if candidate["smiles"] == "COC" else 1.2,
                    "rmse_ppm": 0.3 if candidate["smiles"] == "COC" else 1.4,
                    "max_abs_error_ppm": 0.4 if candidate["smiles"] == "COC" else 1.8,
                }
                for rank, candidate in enumerate(ordered, start=1)
            ],
        }


class _FailingForwardScorer:
    def score_candidates(self, *args, **kwargs):
        raise RuntimeError("model process is unavailable")


class _PartialFailureForwardScorer:
    def score_candidates(
        self,
        observed_13c,
        candidates,
        *,
        formula=None,
    ):
        ordered = sorted(
            candidates,
            key=lambda candidate: 0 if candidate["smiles"] == "COC" else 1,
        )
        return {
            "status": "partial_failure",
            "calibrated_probability": False,
            "quantile_enabled": True,
            "model": {
                "provider": "csp5_forward_13c_v1",
                "model_name": "CSP5q-13C",
                "sha256": "0123456789abcdef",
            },
            "candidates": [
                {
                    "candidate_id": candidate["candidate_id"],
                    "relative_rank": rank,
                    "assignment_mode": "test_hungarian",
                    "matched_count": 2,
                    "observed_count": 2,
                    "predicted_atom_count": 2,
                    "unmatched_observed_count": 0,
                    "unmatched_predicted_atom_count": 0,
                    "bidirectional_coverage": 1.0,
                    "mae_ppm": 0.2,
                    "rmse_ppm": 0.3,
                    "max_abs_error_ppm": 0.4,
                }
                if rank == 1
                else {
                    "candidate_id": candidate["candidate_id"],
                    "relative_rank": rank,
                    "prediction_failed": True,
                    "assignment_mode": "none_prediction_failed",
                    "matched_count": 0,
                    "observed_count": 2,
                    "predicted_atom_count": 0,
                    "unmatched_observed_count": 2,
                    "unmatched_predicted_atom_count": 0,
                    "bidirectional_coverage": 0.0,
                    "mae_ppm": None,
                    "rmse_ppm": None,
                    "max_abs_error_ppm": None,
                }
                for rank, candidate in enumerate(ordered, start=1)
            ],
        }


class _Provider:
    provider_id = "enumerator-test"

    def generate(self, *, formula, constraints, limit):
        assert formula == "C2H6O"
        assert constraints["required_smarts"] == ["O"]
        assert limit == 100
        return [
            {
                "smiles": "CCO",
                "source": "enumerator",
                "source_id": "same-ethanol",
            }
        ]


def _query() -> dict[str, Any]:
    return {
        "peaks_13c": [{"shift": 20.0}, {"shift": 30.0}],
        "peaks_1h": [{"shift": 1.0}, {"shift": 3.5}],
        "formula": "C2H6O",
    }


def test_schema_provenance_and_conservative_decision_are_stable(monkeypatch):
    monkeypatch.setenv("CHEMAPP_CALIBRATION_MODE", "off")
    scorer = _ForwardScorer()
    predictor = HybridPredictorV1(
        reference_ranker=_reference_ranker,
        candidate_providers=[_Provider()],
        forward_scorer=scorer,
    )

    result = predictor.predict(
        **_query(),
        constraints=CandidateConstraints(required_smarts=("O",)),
        generated_smiles=["CCO"],
        candidate_smiles=["CCO"],
    )

    assert result["schema_version"] == SCHEMA_VERSION
    assert result["pipeline_version"] == "1.0.0"
    assert [stage["stage"] for stage in result["stages"]] == [
        "input_validation",
        "candidate_generation",
        "reference_scoring",
        "forward_scoring",
        "assignment",
        "ranking",
        "decision",
    ]
    assert result["candidates"][0]["smiles"] == "COC"
    ethanol = next(
        candidate
        for candidate in result["candidates"]
        if candidate["smiles"] == "CCO"
    )
    assert {item["kind"] for item in ethanol["provenance"]} == {
        "provided",
        "generated",
        "provider",
        "spectral_library",
    }
    assert result["decision"] == {
        "action": "abstain",
        "reason_code": "correctness_probability_not_calibrated",
        "selected_candidate_id": None,
        "leading_hypothesis_candidate_id": result["candidates"][0][
            "candidate_id"
        ],
        "evidence_state": "weak",
        "uncertainty_kind": "point_prediction_only",
        "calibrated_probability": False,
        "top1_probability": None,
    }
    assert result["evidence_state"] == "weak"
    assert result["calibrated_probability"] is False
    assert all(
        candidate["evidence_state"] == "weak"
        and candidate["calibrated_probability"] is False
        for candidate in result["candidates"]
    )
    assert result["forward_context"]["used_for_ranking"] is True
    json.dumps(result)


# NOTE(public-trim): removed test_machine_readable_schema_matches_runtime_versions — depends on docs/ research manifests not shipped in this repository.

def test_calibrated_probability_stays_closed_by_policy_and_guard():
    scorer = _ForwardScorer()
    predictor = HybridPredictorV1(
        reference_ranker=_reference_ranker,
        candidate_providers=[_Provider()],
        forward_scorer=scorer,
    )

    result = predictor.predict(
        **_query(),
        constraints=CandidateConstraints(required_smarts=("O",)),
        generated_smiles=["CCO"],
        candidate_smiles=["CCO"],
    )

    assert result["calibrated_probability"] is False
    assert result["top1_calibrated_probability"] is None
    assert result["candidates"][0]["calibrated_probability"] is False
    assert result["decision"]["reason_code"] == (
        "correctness_probability_not_calibrated"
    )
    assert result["decision"]["top1_probability"] is None
    assert result["decision"]["calibrated_probability"] is False
    assert result["forward_context"]["calibrated_probability"] is False
    assert result["forward_context"]["calibration"] is None
    assert any(
        "no calibrated structure-correctness probability" in warning
        for warning in result["warnings"]
    )


def test_partial_forward_coverage_is_diagnostic_and_does_not_reorder():
    scorer = _ForwardScorer()
    predictor = HybridPredictorV1(
        reference_ranker=_reference_ranker,
        forward_scorer=scorer,
        config=HybridPredictorConfig(forward_candidate_limit=1),
    )

    result = predictor.predict(**_query())
    forward_stage = next(
        stage for stage in result["stages"] if stage["stage"] == "forward_scoring"
    )

    assert forward_stage["status"] == "partial"
    assert forward_stage["used_for_ranking"] is False
    assert result["forward_context"]["used_for_ranking"] is False
    assert result["candidates"][0]["smiles"] == "CCO"
    assert result["candidates"][0]["ranking_evidence"]["basis"] == (
        "13c_reference_then_1h_reference"
    )


def test_calibrated_probability_claim_from_plugin_is_rejected():
    scorer = _ForwardScorer(calibrated_probability=True)
    predictor = HybridPredictorV1(
        reference_ranker=_reference_ranker,
        forward_scorer=scorer,
    )

    result = predictor.predict(**_query())
    forward_stage = next(
        stage for stage in result["stages"] if stage["stage"] == "forward_scoring"
    )

    assert forward_stage["status"] == "failed"
    assert (
        forward_stage["reason_code"]
        == "unsupported_calibrated_probability_claim"
    )
    assert result["candidates"][0]["smiles"] == "CCO"
    assert result["calibrated_probability"] is False


def test_forward_failure_falls_back_to_reference_and_abstains():
    predictor = HybridPredictorV1(
        reference_ranker=_reference_ranker,
        forward_scorer=_FailingForwardScorer(),
    )

    result = predictor.predict(**_query())
    forward_stage = next(
        stage for stage in result["stages"] if stage["stage"] == "forward_scoring"
    )

    assert forward_stage["status"] == "failed"
    assert forward_stage["reason_code"] == "forward_scorer_error"
    assert result["candidates"][0]["smiles"] == "CCO"
    assert result["decision"]["action"] == "abstain"


def test_partial_forward_failure_never_uses_or_calibrates_failed_pool():
    predictor = HybridPredictorV1(
        reference_ranker=_reference_ranker,
        forward_scorer=_PartialFailureForwardScorer(),
    )

    result = predictor.predict(**_query())
    forward_stage = next(
        stage for stage in result["stages"] if stage["stage"] == "forward_scoring"
    )

    assert forward_stage["status"] == "failed"
    assert forward_stage["reason_code"] == "partial_failure"
    assert forward_stage["used_for_ranking"] is False
    assert result["forward_context"]["used_for_ranking"] is False
    assert result["calibrated_probability"] is False
    assert result["decision"]["reason_code"] == (
        "correctness_probability_not_calibrated"
    )


def test_1h_only_query_never_calls_13c_forward_scorer():
    scorer = _ForwardScorer()
    predictor = HybridPredictorV1(
        reference_ranker=_reference_ranker,
        forward_scorer=scorer,
    )

    result = predictor.predict(
        peaks_1h=[{"shift": 1.0}, {"shift": 3.5}],
        formula="C2H6O",
    )

    assert scorer.called == 0
    assert result["query"]["modality"] == "1h"
    assert result["candidates"][0]["smiles"] == "CCO"
    assert result["uncertainty_kind"] == "fixed_tolerance_only"


def test_1h_query_with_only_13c_library_data_is_no_reference():
    def carbon_only_ranker(**_: Any) -> dict[str, Any]:
        breakdown = _breakdown(0.9, 0.0)
        breakdown["h1"] = {
            "score": 0.0,
            "matched": 0,
            "mae": None,
            "query_recall": 0.0,
            "reference_coverage": None,
            "reference_count": 0,
            "assignments": [],
        }
        return {
            "candidates": [
                {
                    "rank": 1,
                    "smiles": "CCO",
                    "source": "carbon-only",
                    "source_id": "ethanol",
                    "ranking_score": 0.0,
                    "evidence_level": "no_reference",
                    "score_breakdown": breakdown,
                }
            ],
            "candidate_pool_status": "formula_match",
            "warnings": [],
        }

    result = HybridPredictorV1(
        reference_ranker=carbon_only_ranker
    ).predict(
        peaks_1h=[{"shift": 1.0}],
        formula="C2H6O",
    )

    assert result["candidates"][0]["evidence_state"] == "no_reference"
    assert result["candidates"][0]["uncertainty_kind"] == "unavailable"
    assert result["decision"]["reason_code"] == "no_spectral_reference"


def test_formula_and_substructure_constraints_fail_closed():
    def no_references(**_: Any) -> dict[str, Any]:
        return {
            "candidates": [],
            "candidate_pool_status": "no_formula_match",
            "warnings": [],
        }

    predictor = HybridPredictorV1(reference_ranker=no_references)
    result = predictor.predict(
        peaks_1h=[{"shift": 1.0}],
        formula="C2H6O",
        candidate_smiles=["CCO", "CC", "COC"],
        constraints=CandidateConstraints(
            required_smarts=("O",),
            forbidden_smarts=("COC",),
        ),
    )

    assert [candidate["smiles"] for candidate in result["candidates"]] == ["CCO"]
    generation = next(
        stage
        for stage in result["stages"]
        if stage["stage"] == "candidate_generation"
    )
    assert generation["rejection_counts"] == {
        "formula_mismatch": 1,
        "forbidden_substructure_present": 1,
    }
    assert result["evidence_state"] == "no_reference"
    assert result["decision"]["reason_code"] == "no_spectral_reference"
    assert result["candidate_pool_status"] == "formula_match"


def test_empty_candidate_pool_returns_versioned_abstention_result():
    def no_references(**_: Any) -> dict[str, Any]:
        return {
            "candidates": [],
            "candidate_pool_status": "no_formula_match",
            "warnings": [],
        }

    result = HybridPredictorV1(
        reference_ranker=no_references
    ).predict(
        peaks_13c=[{"shift": 20.0}],
        formula="C2H6O",
        candidate_smiles=["CC"],
    )

    assert result["status"] == "no_candidates"
    assert result["candidates"] == []
    assert result["decision"]["action"] == "abstain"
    assert result["decision"]["reason_code"] == "no_candidates"
    assert result["evidence_state"] == "no_reference"
    assert result["uncertainty_kind"] == "unavailable"
    assert result["calibrated_probability"] is False
    assert len(result["stages"]) == 7
    json.dumps(result)


def test_invalid_smarts_and_empty_spectrum_are_rejected():
    predictor = HybridPredictorV1(reference_ranker=_reference_ranker)

    with pytest.raises(HybridPredictorInputError, match="required_smarts"):
        predictor.predict(
            peaks_1h=[{"shift": 1.0}],
            formula="C2H6O",
            constraints=CandidateConstraints(required_smarts=("[",)),
        )
    with pytest.raises(HybridPredictorInputError, match="No finite"):
        predictor.predict(
            peaks_1h=[{"shift": float("nan")}],
            formula="C2H6O",
        )


def test_legacy_adapter_preserves_reference_score_without_fake_probability():
    result = HybridPredictorV1(
        reference_ranker=_reference_ranker,
    ).predict(**_query())

    legacy = hybrid_to_legacy_rank_result(result)

    assert legacy["hybrid_schema_version"] == SCHEMA_VERSION
    assert legacy["decision"]["action"] == "abstain"
    assert legacy["calibrated_probability"] is False
    assert legacy["candidates"][0]["ranking_score"] == 0.9
    assert legacy["candidates"][0]["ranking_score_semantics"] == (
        "legacy_reference_heuristic_not_hybrid_probability"
    )
    assert legacy["candidates"][0]["evidence_level"] == "weak"
    assert legacy["candidates"][0]["calibrated_probability"] is False
