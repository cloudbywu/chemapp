from __future__ import annotations

import copy

import pytest

from app.ml.nmr_candidate_scorer_v4 import rank_candidate_evidence
from app.ml.nmr_candidate_scorer_v5 import (
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    NMRCandidateScorerV5InputError,
    PROTOCOL_VERSION,
    extract_feature_rows,
    finalize_model_artifact,
    rank_candidate_evidence_v5,
)


def _v4_ranking():
    return rank_candidate_evidence(
        [12.0, 58.0],
        [
            {
                "candidate_id": "candidate-aaaaaaaaaaaaaaaaaaaaaaaa",
                "smiles": "CCO",
                "atom_predictions": [
                    {"atom_index": 0, "shift_ppm": 12.5},
                    {"atom_index": 1, "shift_ppm": 57.0},
                ],
            },
            {
                "candidate_id": "candidate-bbbbbbbbbbbbbbbbbbbbbbbb",
                "smiles": "COC",
                "atom_predictions": [
                    {"atom_index": 0, "shift_ppm": 52.0},
                    {"atom_index": 2, "shift_ppm": 52.0},
                ],
            },
            {
                "candidate_id": "candidate-cccccccccccccccccccccccc",
                "smiles": "CCC",
                "atom_predictions": [
                    {"atom_index": 0, "shift_ppm": 20.0},
                    {"atom_index": 1, "shift_ppm": 35.0},
                    {"atom_index": 2, "shift_ppm": 20.0},
                ],
            },
        ],
    )


def _model():
    coefficients = [0.0] * len(FEATURE_NAMES)
    coefficients[0] = 2.0
    coefficients[1] = -0.2
    coefficients[14] = 1.0
    return finalize_model_artifact(
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "model_kind": "symmetric_pairwise_l2_logistic_v1",
            "normalization": {
                "means": [0.0] * len(FEATURE_NAMES),
                "scales": [1.0] * len(FEATURE_NAMES),
            },
            "coefficients": coefficients,
            "fit_intercept": False,
            "intercept": 0.0,
            "development_binding": {"fixture": True},
            "calibrated_probability": False,
            "conditional_probability": False,
            "runtime": {"fixture": True},
        }
    )


def test_candidate_input_order_does_not_change_utilities_or_v5_order():
    ranking = _v4_ranking()
    model = _model()
    expected_hash = model["artifact_sha256"]
    first = rank_candidate_evidence_v5(
        ranking, model, expected_model_sha256=expected_hash
    )
    permuted = copy.deepcopy(ranking)
    permuted["candidates"].reverse()
    second = rank_candidate_evidence_v5(
        permuted, model, expected_model_sha256=expected_hash
    )

    assert first["v5_order"] == second["v5_order"]
    assert {row["candidate_id"]: row["v5_utility"] for row in first["candidates"]} == {
        row["candidate_id"]: row["v5_utility"] for row in second["candidates"]
    }
    assert first["calibrated_probability"] is False
    assert first["conditional_probability"] is False


@pytest.mark.parametrize(
    "key",
    [
        "truth_candidate_id",
        "gold_role",
        "outcome",
        "split_group",
        "source_id",
        "reviewer_identity",
        "top1_correct",
        "training_label",
    ],
)
def test_role_outcome_and_provenance_fields_are_recursively_rejected(key):
    ranking = _v4_ranking()
    ranking["candidates"][0][key] = "forbidden"
    with pytest.raises(
        NMRCandidateScorerV5InputError,
        match="prohibited role/outcome metadata",
    ):
        extract_feature_rows(ranking)


def test_outer_cached_row_with_split_is_not_a_valid_scorer_input():
    with pytest.raises(NMRCandidateScorerV5InputError, match="prohibited"):
        extract_feature_rows({"ranking": _v4_ranking(), "split": "dev"})


def test_non_finite_evidence_fails_closed():
    ranking = _v4_ranking()
    ranking["candidates"][0]["v4"]["set_similarity"] = float("nan")
    with pytest.raises(NMRCandidateScorerV5InputError, match="finite"):
        extract_feature_rows(ranking)


def test_model_tampering_and_unpinned_hash_are_rejected():
    ranking = _v4_ranking()
    model = _model()
    expected_hash = model["artifact_sha256"]
    tampered = copy.deepcopy(model)
    tampered["coefficients"][0] += 0.001
    with pytest.raises(NMRCandidateScorerV5InputError, match="hash mismatch"):
        rank_candidate_evidence_v5(
            ranking, tampered, expected_model_sha256=expected_hash
        )
    with pytest.raises(NMRCandidateScorerV5InputError, match="hash mismatch"):
        rank_candidate_evidence_v5(ranking, model, expected_model_sha256="0" * 64)


def test_strict_v4_and_model_schemas_reject_unknown_fields():
    ranking = _v4_ranking()
    ranking["unexpected"] = 1
    with pytest.raises(NMRCandidateScorerV5InputError, match="strict allowlist"):
        extract_feature_rows(ranking)

    model = _model()
    model["unexpected"] = 1
    with pytest.raises(NMRCandidateScorerV5InputError, match="strict allowlist"):
        rank_candidate_evidence_v5(
            _v4_ranking(),
            model,
            expected_model_sha256=model["artifact_sha256"],
        )


def test_fixed_feature_schema_covers_required_evidence_families():
    assert FEATURE_NAMES == (
        "v4_set_similarity",
        "v4_matched_mae_ppm",
        "v4_matched_rmse_ppm",
        "v4_matched_max_abs_error_ppm",
        "v4_bidirectional_coverage",
        "v4_count_agreement",
        "v3_matched_mae_ppm",
        "v3_matched_rmse_ppm",
        "v3_matched_max_abs_error_ppm",
        "v3_bidirectional_coverage",
        "v3_count_agreement",
        "v4_signal_count_log_ratio",
        "v3_v4_rank_consensus",
        "v3_v4_rank_disagreement",
        "v4_competitor_margin",
        "v3_competitor_margin",
        "rank_consensus_competitor_margin",
        "v4_minus_v3_coverage",
    )
