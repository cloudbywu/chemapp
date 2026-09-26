from __future__ import annotations

import copy
import platform

import numpy as np
import pytest
from rdkit import rdBase
import sklearn

from app.ml.nmr_candidate_scorer_v8 import (
    EVIDENCE_SCHEMA_VERSION,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    MODEL_SCHEMA_VERSION,
    NMRCandidateScorerV8InputError,
    PROTOCOL_VERSION,
    RETRIEVAL_PROVENANCE_VERSION,
    RETRIEVAL_SCORE_SEMANTICS,
    TIE_BREAK_SEMANTICS,
    extract_feature_rows,
    finalize_model_artifact,
    rank_candidate_evidence_v8,
)


def _missing(kind: str):
    values = {
        "formula": {
            "status": "missing",
            "exact_match": None,
            "element_l1_distance": None,
        },
        "quality": {"status": "missing", "score": None},
        "applicability": {
            "status": "missing",
            "in_domain": None,
            "distance": None,
        },
        "proton1h": {
            "status": "missing",
            "matched_mae_ppm": None,
            "bidirectional_coverage": None,
            "count_agreement": None,
        },
        "dept": {
            "status": "missing",
            "class_agreement": None,
            "coverage": None,
        },
        "two_d": {
            "status": "missing",
            "edge_precision": None,
            "edge_recall": None,
        },
    }
    return copy.deepcopy(values[kind])


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
                "interval_available": True,
                "interval_coverage": 0.8,
                "interval_mean_width_ppm": 5.0,
                "observed_signal_count": 2,
                "predicted_signal_count": 2,
            },
            "formula": {
                "status": "ok",
                "exact_match": True,
                "element_l1_distance": 0.0,
            },
            "retrieval": {
                "status": "ok",
                "prior_score": 0.7,
                "score_semantics": RETRIEVAL_SCORE_SEMANTICS,
                "provenance_version": RETRIEVAL_PROVENANCE_VERSION,
                "query_independent_of_candidate_order": True,
            },
            "quality": {"status": "ok", "score": 0.9},
            "applicability": {
                "status": "ok",
                "in_domain": True,
                "distance": 0.1,
            },
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
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "candidates": [
            _candidate("opaque-a", "CCO", 1.0),
            _candidate("opaque-b", "CCC", 4.0),
            _candidate("opaque-c", "CCN", 7.0),
        ],
    }


def _model():
    coefficients = [0.0] * len(FEATURE_NAMES)
    coefficients[FEATURE_NAMES.index("c13_matched_mae_ppm")] = -1.0
    coefficients[FEATURE_NAMES.index("retrieval_prior_score")] = 0.2
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
            "calibrated_probability": False,
            "conditional_probability": False,
            "probability": False,
            "automatic_selection": False,
            "calibration_or_test_gold_read": False,
            "scientific_training_complete": False,
            "production_eligible": False,
            "promotion_status": "research_smoke_only",
            "tie_break_semantics": TIE_BREAK_SEMANTICS,
            "retrieval_prior_contract": {
                "score_semantics": RETRIEVAL_SCORE_SEMANTICS,
                "provenance_version": RETRIEVAL_PROVENANCE_VERSION,
                "candidate_order_independent": True,
            },
            "development_binding": {
                "status": "smoke_only",
                "scientific_training_complete": False,
                "scorer_sha256": "7" * 64,
                "trainer_sha256": "8" * 64,
                "data_protocol_sha256": "9" * 64,
                "source_snapshot_sha256": "a" * 64,
                "manifest_sha256": "1" * 64,
                "cases_sha256": "2" * 64,
                "case_count": 3,
                "component_count": 3,
                "group_dimensions": ["doi", "scaffold", "connectivity"],
                "legacy_group_proxy": False,
                "seed": 1,
                "cv": {
                    "folds": 3,
                    "c_grid": [0.1],
                    "selected_c": 0.1,
                    "fold_assignment": (
                        "connected_components_of_doi_scaffold_connectivity_and_"
                        "ecfp4_tanimoto_0.70_then_balanced_hash_v1"
                    ),
                },
                "input_provenance": {
                    "approval_status": "frozen_default_inputs_approved",
                    "adapter": "synthetic_smoke_v1",
                    "development_gold_sha256": "3" * 64,
                    "frozen_input_contract_sha256": "b" * 64,
                    "rankings_sha256": "4" * 64,
                    "release_id": "fixture-release",
                    "split_manifest_sha256": "5" * 64,
                    "v5_model_sha256": "6" * 64,
                },
            },
            "runtime": {
                "canonicalizer": (
                    "rdkit_molfromsmiles_then_canonical_isomeric_smiles_v1"
                ),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "sklearn_version": sklearn.__version__,
                "rdkit_version": rdBase.rdkitVersion,
            },
        }
    )


def test_candidate_id_and_input_order_do_not_change_structure_ranking():
    bundle = _bundle()
    model = _model()
    first = rank_candidate_evidence_v8(
        bundle,
        model,
        expected_model_sha256=model["artifact_sha256"],
        allow_research_artifact=True,
    )

    permuted = copy.deepcopy(bundle)
    permuted["candidates"].reverse()
    for index, candidate in enumerate(permuted["candidates"]):
        candidate["candidate_id"] = f"renamed-{index}"
    second = rank_candidate_evidence_v8(
        permuted,
        model,
        expected_model_sha256=model["artifact_sha256"],
        allow_research_artifact=True,
    )

    def smiles_order(result):
        by_id = {
            row["candidate_id"]: row["canonical_smiles"]
            for row in result["candidates"]
        }
        return [by_id[candidate_id] for candidate_id in result["v8_order"]]

    assert smiles_order(first) == smiles_order(second) == ["CCO", "CCC", "CCN"]
    assert first["calibrated_probability"] is False
    assert first["probability"] is False
    assert first["automatic_selection"] is False


@pytest.mark.parametrize(
    "field",
    [
        "source_order",
        "candidate_rank",
        "truth_role",
        "doi_group",
        "scaffold_id",
        "connectivity_group",
    ],
)
def test_evidence_recursively_rejects_identity_order_and_group_leakage(field):
    bundle = _bundle()
    bundle["candidates"][0]["evidence"]["quality"][field] = "leak"
    with pytest.raises(NMRCandidateScorerV8InputError, match="prohibited"):
        extract_feature_rows(bundle)


def test_retrieval_contract_is_versioned_and_order_independent():
    for key, value in (
        ("score_semantics", "unknown"),
        ("provenance_version", "unversioned"),
        ("query_independent_of_candidate_order", False),
    ):
        bundle = _bundle()
        bundle["candidates"][0]["evidence"]["retrieval"][key] = value
        with pytest.raises(NMRCandidateScorerV8InputError, match="retrieval"):
            extract_feature_rows(bundle)


def test_noncanonical_structure_and_nonfinite_evidence_fail_closed():
    bundle = _bundle()
    bundle["candidates"][0]["canonical_smiles"] = "OCC"
    with pytest.raises(NMRCandidateScorerV8InputError, match="not RDKit canonical"):
        extract_feature_rows(bundle)

    bundle = _bundle()
    bundle["candidates"][0]["evidence"]["carbon13"]["matched_mae_ppm"] = float(
        "nan"
    )
    with pytest.raises(NMRCandidateScorerV8InputError, match="finite"):
        extract_feature_rows(bundle)


def test_missing_optional_modality_is_distinct_from_failure():
    rows = extract_feature_rows(_bundle())
    first = rows[0]
    assert first["evidence_availability"]["proton1h"] == "missing"
    assert first["feature_values"][FEATURE_NAMES.index("h1_available")] == 0.0
    assert first["feature_values"][FEATURE_NAMES.index("h1_failed")] == 0.0


def test_unknown_applicability_recommends_abstention_but_never_auto_selects():
    bundle = _bundle()
    for candidate in bundle["candidates"]:
        candidate["evidence"]["applicability"] = _missing("applicability")
    model = _model()
    result = rank_candidate_evidence_v8(
        bundle,
        model,
        expected_model_sha256=model["artifact_sha256"],
        allow_research_artifact=True,
    )
    assert result["decision_status"] == "abstain_recommended"
    assert result["abstention_recommended"] is True
    assert "top_candidate_applicability_unknown" in result["abstention_reasons"]
    assert result["automatic_selection"] is False


def test_all_required_predictions_failed_is_insufficient_evidence():
    bundle = _bundle()
    for candidate in bundle["candidates"]:
        carbon = candidate["evidence"]["carbon13"]
        for key in (
            "matched_mae_ppm",
            "matched_rmse_ppm",
            "matched_max_abs_error_ppm",
            "observed_coverage",
            "predicted_coverage",
            "count_agreement",
            "interval_coverage",
            "interval_mean_width_ppm",
            "observed_signal_count",
            "predicted_signal_count",
        ):
            carbon[key] = None
        carbon["status"] = "failed"
        carbon["interval_available"] = False
        candidate["evidence"]["failures"]["prediction_failed"] = True
    model = _model()
    result = rank_candidate_evidence_v8(
        bundle,
        model,
        expected_model_sha256=model["artifact_sha256"],
        allow_research_artifact=True,
    )
    assert result["invalid_candidate_count"] == 3
    assert result["decision_status"] == "insufficient_evidence"
    assert "required_carbon13_unavailable_or_failed" in result["abstention_reasons"]


def test_model_hash_tampering_and_automatic_selection_claim_are_rejected():
    model = _model()
    tampered = copy.deepcopy(model)
    tampered["coefficients"][0] += 0.01
    with pytest.raises(NMRCandidateScorerV8InputError, match="hash mismatch"):
        rank_candidate_evidence_v8(
            _bundle(),
            tampered,
            expected_model_sha256=model["artifact_sha256"],
            allow_research_artifact=True,
        )

    unsafe = copy.deepcopy(model)
    unsafe["automatic_selection"] = True
    unsafe.pop("artifact_sha256")
    unsafe = finalize_model_artifact(unsafe)
    with pytest.raises(NMRCandidateScorerV8InputError, match="unsupported"):
        rank_candidate_evidence_v8(
            _bundle(),
            unsafe,
            expected_model_sha256=unsafe["artifact_sha256"],
            allow_research_artifact=True,
        )


def test_research_artifact_is_rejected_by_default():
    model = _model()
    with pytest.raises(NMRCandidateScorerV8InputError, match="allow_research"):
        rank_candidate_evidence_v8(
            _bundle(), model, expected_model_sha256=model["artifact_sha256"]
        )


def test_self_declared_production_artifact_is_not_accepted_without_verifier():
    model = _model()
    model.pop("artifact_sha256")
    model["promotion_status"] = "production_approved"
    model["production_eligible"] = True
    model = finalize_model_artifact(model)
    with pytest.raises(NMRCandidateScorerV8InputError, match="promotion"):
        rank_candidate_evidence_v8(
            _bundle(),
            model,
            expected_model_sha256=model["artifact_sha256"],
            allow_research_artifact=True,
        )
