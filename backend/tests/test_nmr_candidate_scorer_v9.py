from __future__ import annotations

import copy
import hashlib

import pytest

from app.ml.nmr_candidate_scorer_v8 import FEATURE_NAMES
from app.ml.nmr_candidate_scorer_v9 import (
    FEATURE_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    NMRCandidateScorerV9InputError,
    PROTOCOL_VERSION,
    ROLELESS_POOL_SCHEMA_VERSION,
    TIE_BREAK_SEMANTICS,
    canonical_sha256,
    finalize_model_artifact,
    rank_candidate_evidence_v9,
    runtime_binding,
)


def _missing(kind: str):
    return copy.deepcopy(
        {
            "proton1h": {
                "status": "missing",
                "matched_mae_ppm": None,
                "bidirectional_coverage": None,
                "count_agreement": None,
            },
            "dept": {"status": "missing", "class_agreement": None, "coverage": None},
            "two_d": {"status": "missing", "edge_precision": None, "edge_recall": None},
        }[kind]
    )


def _candidate(candidate_id: str, smiles: str, mae: float):
    return {
        "candidate_id": candidate_id,
        "canonical_smiles": smiles,
        "evidence": {
            "carbon13": {
                "status": "ok",
                "matched_mae_ppm": mae,
                "matched_rmse_ppm": mae + 0.5,
                "matched_max_abs_error_ppm": mae + 1.0,
                "observed_coverage": 1.0,
                "predicted_coverage": 1.0,
                "count_agreement": 1.0,
                "interval_available": False,
                "interval_coverage": None,
                "interval_mean_width_ppm": None,
                "observed_signal_count": 2,
                "predicted_signal_count": 2,
            },
            "formula": {"status": "ok", "exact_match": True, "element_l1_distance": 0.0},
            "retrieval": {
                "status": "missing",
                "prior_score": None,
                "score_semantics": "normalized_structure_retrieval_similarity_v1",
                "provenance_version": "chemapp.retrieval-prior.v1",
                "query_independent_of_candidate_order": True,
            },
            "quality": {"status": "missing", "score": None},
            "applicability": {"status": "ok", "in_domain": True, "distance": 0.1},
            "proton1h": _missing("proton1h"),
            "dept": _missing("dept"),
            "two_d": _missing("two_d"),
            "failures": {
                "prediction_failed": False,
                "parsing_failed": False,
                "formula_failed": False,
                "retrieval_failed": False,
            },
        },
    }


def _bundle():
    return {
        "schema_version": ROLELESS_POOL_SCHEMA_VERSION,
        "candidates": [
            _candidate("legacy-position-0", "CCO", 1.0),
            _candidate("legacy-position-1", "CCC", 4.0),
            _candidate("legacy-position-2", "CCN", 7.0),
        ],
    }


def _model():
    coefficients = [0.0] * len(FEATURE_NAMES)
    coefficients[FEATURE_NAMES.index("c13_matched_mae_ppm")] = -1.0
    return finalize_model_artifact(
        {
            "schema_version": MODEL_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "evidence_schema_version": ROLELESS_POOL_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "model_kind": "query_conditional_logit_l2_linear_v1",
            "model_parameters": {
                "means": [0.0] * len(FEATURE_NAMES),
                "scales": [1.0] * len(FEATURE_NAMES),
                "coefficients": coefficients,
            },
            "calibrated_probability": False,
            "conditional_probability": False,
            "probability": False,
            "automatic_selection": False,
            "calibration_or_sealed_read": False,
            "scientific_training_complete": False,
            "production_eligible": False,
            "promotion_status": "framework_smoke_only",
            "tie_break_semantics": TIE_BREAK_SEMANTICS,
            "development_binding": {
                "status": "synthetic_framework_smoke",
                "case_count": 10,
                "rankable_case_count": 8,
                "training_case_count": 6,
                "generator_empty_or_failed_count": 1,
                "singleton_count": 1,
                "exact_truth_miss_count": 4,
                "no_valid_candidate_count": 0,
                "component_count": 5,
                "group_dimensions": ["source", "scaffold"],
                "seed": 1,
                "candidate_pool_sha256": "1" * 64,
                "evidence_manifest_sha256": "2" * 64,
                "development_gold_sha256": "3" * 64,
                "groups_sha256": "4" * 64,
                "scorer_sha256": "5" * 64,
                "trainer_sha256": "6" * 64,
                "nested_cv": {
                    "outer_folds": 3,
                    "inner_folds_max": 2,
                    "family_grid_sha256": "7" * 64,
                    "selected_family": "query_conditional_logit_l2_linear_v1",
                    "selected_hyperparameters": {"l2": 0.1},
                    "fold_semantics": "synthetic_connected_components_v1",
                },
                "limitations": ["synthetic smoke only"],
            },
            "runtime": runtime_binding(),
        }
    )


def _smiles_order(result):
    by_id = {row["candidate_id"]: row["canonical_smiles"] for row in result["candidates"]}
    return [by_id[candidate_id] for candidate_id in result["v9_order"]]


def test_candidate_shuffle_and_opaque_id_remap_are_structure_invariant():
    model = _model()
    bundle = _bundle()
    first = rank_candidate_evidence_v9(
        bundle,
        model,
        expected_evidence_sha256=canonical_sha256(bundle),
        expected_model_sha256=model["artifact_sha256"],
        allow_research_artifact=True,
    )
    remapped = copy.deepcopy(bundle)
    remapped["candidates"].reverse()
    for index, candidate in enumerate(remapped["candidates"]):
        candidate["candidate_id"] = f"opaque-remap-{index}"
    second = rank_candidate_evidence_v9(
        remapped,
        model,
        expected_evidence_sha256=canonical_sha256(remapped),
        expected_model_sha256=model["artifact_sha256"],
        allow_research_artifact=True,
    )
    assert _smiles_order(first) == _smiles_order(second) == ["CCO", "CCC", "CCN"]
    assert first["probability"] is False
    assert first["automatic_selection"] is False
    assert first["production_eligible"] is False


def test_evidence_and_model_hashes_fail_closed():
    model = _model()
    bundle = _bundle()
    with pytest.raises(NMRCandidateScorerV9InputError, match="evidence bundle hash"):
        rank_candidate_evidence_v9(
            bundle,
            model,
            expected_evidence_sha256="0" * 64,
            expected_model_sha256=model["artifact_sha256"],
            allow_research_artifact=True,
        )
    tampered = copy.deepcopy(model)
    tampered["model_parameters"]["coefficients"][0] = 1.0
    with pytest.raises(NMRCandidateScorerV9InputError, match="model artifact hash"):
        rank_candidate_evidence_v9(
            bundle,
            tampered,
            expected_evidence_sha256=canonical_sha256(bundle),
            expected_model_sha256=model["artifact_sha256"],
            allow_research_artifact=True,
        )


def test_runtime_requires_stage4_pool_schema_and_research_opt_in():
    model = _model()
    bundle = _bundle()
    with pytest.raises(NMRCandidateScorerV9InputError, match="research-only"):
        rank_candidate_evidence_v9(
            bundle,
            model,
            expected_evidence_sha256=canonical_sha256(bundle),
            expected_model_sha256=model["artifact_sha256"],
        )
    old = copy.deepcopy(bundle)
    old["schema_version"] = "chemapp.nmr.roleless-evidence.v8"
    with pytest.raises(NMRCandidateScorerV9InputError, match="Stage-4"):
        rank_candidate_evidence_v9(
            old,
            model,
            expected_evidence_sha256=canonical_sha256(old),
            expected_model_sha256=model["artifact_sha256"],
            allow_research_artifact=True,
        )


def test_recursive_v8_leakage_guard_is_preserved():
    model = _model()
    bundle = _bundle()
    bundle["candidates"][0]["evidence"]["quality"]["source_position"] = 0
    with pytest.raises(NMRCandidateScorerV9InputError, match="roleless evidence"):
        rank_candidate_evidence_v9(
            bundle,
            model,
            expected_evidence_sha256=canonical_sha256(bundle),
            expected_model_sha256=model["artifact_sha256"],
            allow_research_artifact=True,
        )


def test_exact_utility_tie_uses_structure_hash_not_failure_count_or_id():
    model = _model()
    model.pop("artifact_sha256")
    model["model_parameters"]["coefficients"] = [0.0] * len(FEATURE_NAMES)
    model = finalize_model_artifact(model)
    bundle = _bundle()
    # Formula failure is not a hard validity failure; it must not precede the
    # declared canonical-structure tie break.
    candidate = bundle["candidates"][0]
    candidate["evidence"]["formula"] = {
        "status": "failed",
        "exact_match": None,
        "element_l1_distance": None,
    }
    candidate["evidence"]["failures"]["formula_failed"] = True
    result = rank_candidate_evidence_v9(
        bundle,
        model,
        expected_evidence_sha256=canonical_sha256(bundle),
        expected_model_sha256=model["artifact_sha256"],
        allow_research_artifact=True,
    )
    expected = sorted(
        bundle["candidates"],
        key=lambda row: hashlib.sha256(row["canonical_smiles"].encode()).hexdigest(),
    )
    assert _smiles_order(result) == [row["canonical_smiles"] for row in expected]
