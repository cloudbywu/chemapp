"""Strict runtime for the Stage-4 NMR candidate-ranker research models.

The scorer consumes only the roleless ``v8`` evidence bundle.  Development
labels, source/group metadata, candidate positions and split roles are not
accepted by this module.  ``candidate_id`` is an opaque join key only: model
features, utilities and exact-tie resolution depend on the canonical structure
and evidence, never on the identifier or input order.

Utilities are ranking scores, not probabilities.  This module deliberately
has no production-approval path; every v9 artifact requires an explicit
``allow_research_artifact=True`` opt-in.

Status: research (explicit opt-in only).  Canonical hashing is imported from
the frozen v8 scorer; strict key validation and the finite numeric guards
live in :mod:`app.ml._core`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import math
import platform
from typing import Any

import numpy as np
from rdkit import rdBase
import sklearn

from app.ml._core.numeric import finite_float, sha256_text
from app.ml._core.schema import strict_fields_mismatch
from app.ml.nmr_candidate_scorer_v8 import (
    EVIDENCE_SCHEMA_VERSION as V8_EVIDENCE_SCHEMA_VERSION,
    FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    canonical_json_bytes,
    extract_feature_rows,
)


PROTOCOL_VERSION = "chemapp.nmr.candidate-scorer.v9"
MODEL_SCHEMA_VERSION = "chemapp.nmr.candidate-ranker-model.v9"
OUTPUT_SCHEMA_VERSION = "chemapp.nmr.roleless-ranking.v9"
ROLELESS_POOL_SCHEMA_VERSION = "chemapp.nmr.roleless-evidence-pool.v9"
TIE_BREAK_SEMANTICS = "sha256(canonical_smiles)_ascending_on_exact_utility_tie_v1"

LINEAR_MODEL_KINDS = frozenset(
    {
        "symmetric_pairwise_l2_logistic_v2",
        "query_conditional_logit_l2_linear_v1",
    }
)
TREE_MODEL_KIND = "symmetric_pairwise_gradient_boosted_trees_v1"
MODEL_KINDS = LINEAR_MODEL_KINDS | {TREE_MODEL_KIND}

_MODEL_KEYS = frozenset(
    {
        "artifact_sha256",
        "schema_version",
        "protocol_version",
        "evidence_schema_version",
        "feature_schema_version",
        "feature_names",
        "model_kind",
        "model_parameters",
        "calibrated_probability",
        "conditional_probability",
        "probability",
        "automatic_selection",
        "calibration_or_sealed_read",
        "scientific_training_complete",
        "production_eligible",
        "promotion_status",
        "tie_break_semantics",
        "development_binding",
        "runtime",
    }
)
_LINEAR_PARAMETER_KEYS = frozenset({"means", "scales", "coefficients"})
_TREE_PARAMETER_KEYS = frozenset(
    {
        "means",
        "scales",
        "learning_rate",
        "initial_raw_score",
        "pairwise_antisymmetric",
        "trees",
    }
)
_TREE_KEYS = frozenset(
    {"children_left", "children_right", "features", "thresholds", "values"}
)
_DEVELOPMENT_BINDING_KEYS = frozenset(
    {
        "status",
        "case_count",
        "rankable_case_count",
        "training_case_count",
        "generator_empty_or_failed_count",
        "singleton_count",
        "exact_truth_miss_count",
        "no_valid_candidate_count",
        "component_count",
        "group_dimensions",
        "seed",
        "candidate_pool_sha256",
        "evidence_manifest_sha256",
        "development_gold_sha256",
        "groups_sha256",
        "scorer_sha256",
        "trainer_sha256",
        "nested_cv",
        "limitations",
    }
)
_NESTED_CV_KEYS = frozenset(
    {
        "outer_folds",
        "inner_folds_max",
        "family_grid_sha256",
        "selected_family",
        "selected_hyperparameters",
        "fold_semantics",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "canonicalizer",
        "python_version",
        "numpy_version",
        "sklearn_version",
        "rdkit_version",
    }
)


class NMRCandidateScorerV9InputError(ValueError):
    """Raised when evidence or a v9 research artifact fails closed."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, context: str
) -> None:
    strict_fields_mismatch(
        value, expected, context=context, error=NMRCandidateScorerV9InputError
    )


def _sha256(value: Any, *, context: str) -> str:
    return sha256_text(
        value,
        context=context,
        error=NMRCandidateScorerV9InputError,
        description="SHA-256",
    )


def _finite(value: Any, *, context: str) -> float:
    return finite_float(value, context=context, error=NMRCandidateScorerV9InputError)


def _vector(value: Any, *, context: str, positive: bool = False) -> list[float]:
    if not isinstance(value, list) or len(value) != len(FEATURE_NAMES):
        raise NMRCandidateScorerV9InputError(
            f"{context} must have {len(FEATURE_NAMES)} entries"
        )
    output = [_finite(item, context=f"{context}[{index}]") for index, item in enumerate(value)]
    if positive and any(item <= 0.0 for item in output):
        raise NMRCandidateScorerV9InputError(f"{context} must be strictly positive")
    return output


def model_artifact_sha256(model: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in model.items() if key != "artifact_sha256"}
    )


def finalize_model_artifact(model_core: Mapping[str, Any]) -> dict[str, Any]:
    if "artifact_sha256" in model_core:
        raise NMRCandidateScorerV9InputError(
            "model core must not self-declare artifact_sha256"
        )
    artifact = dict(model_core)
    artifact["artifact_sha256"] = model_artifact_sha256(artifact)
    return artifact


def runtime_binding() -> dict[str, str]:
    """Return the strict runtime block used by trainers and test fixtures."""

    return {
        "canonicalizer": "rdkit_molfromsmiles_then_canonical_isomeric_smiles_v1",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "rdkit_version": rdBase.rdkitVersion,
    }


def _validate_tree(tree: Any, *, tree_index: int) -> dict[str, list[Any]]:
    context = f"model_parameters.trees[{tree_index}]"
    if not isinstance(tree, Mapping):
        raise NMRCandidateScorerV9InputError(f"{context} must be an object")
    _strict_keys(tree, _TREE_KEYS, context=context)
    lengths = []
    cleaned: dict[str, list[Any]] = {}
    for key in ("children_left", "children_right", "features"):
        raw = tree[key]
        if not isinstance(raw, list) or not raw:
            raise NMRCandidateScorerV9InputError(f"{context}.{key} must be non-empty")
        values = []
        for index, value in enumerate(raw):
            if isinstance(value, bool) or not isinstance(value, int):
                raise NMRCandidateScorerV9InputError(
                    f"{context}.{key}[{index}] must be an integer"
                )
            values.append(value)
        cleaned[key] = values
        lengths.append(len(values))
    for key in ("thresholds", "values"):
        raw = tree[key]
        if not isinstance(raw, list) or not raw:
            raise NMRCandidateScorerV9InputError(f"{context}.{key} must be non-empty")
        cleaned[key] = [
            _finite(value, context=f"{context}.{key}[{index}]")
            for index, value in enumerate(raw)
        ]
        lengths.append(len(raw))
    if len(set(lengths)) != 1:
        raise NMRCandidateScorerV9InputError(f"{context} arrays have unequal lengths")
    node_count = lengths[0]
    for index, (left, right, feature) in enumerate(
        zip(
            cleaned["children_left"],
            cleaned["children_right"],
            cleaned["features"],
            strict=True,
        )
    ):
        leaf = left == right == -1
        if leaf:
            if feature != -2:
                raise NMRCandidateScorerV9InputError(
                    f"{context} leaf {index} must use feature=-2"
                )
        elif (
            left <= index
            or right <= index
            or left >= node_count
            or right >= node_count
            or not 0 <= feature < len(FEATURE_NAMES)
        ):
            raise NMRCandidateScorerV9InputError(
                f"{context} node {index} has invalid children/feature"
            )
    return cleaned


def validate_model_artifact(
    model: Mapping[str, Any],
    *,
    expected_model_sha256: str,
    allow_research_artifact: bool = False,
) -> dict[str, Any]:
    """Validate a v9 model and its development-only provenance binding."""

    if not isinstance(model, Mapping):
        raise NMRCandidateScorerV9InputError("model must be an object")
    _strict_keys(model, _MODEL_KEYS, context="v9 model")
    expected_hash = _sha256(expected_model_sha256, context="expected model hash")
    actual_hash = model_artifact_sha256(model)
    if model["artifact_sha256"] != actual_hash or expected_hash != actual_hash:
        raise NMRCandidateScorerV9InputError("v9 model artifact hash mismatch")
    if (
        model["schema_version"] != MODEL_SCHEMA_VERSION
        or model["protocol_version"] != PROTOCOL_VERSION
        or model["evidence_schema_version"] != ROLELESS_POOL_SCHEMA_VERSION
        or model["feature_schema_version"] != FEATURE_SCHEMA_VERSION
        or model["feature_names"] != list(FEATURE_NAMES)
        or model["model_kind"] not in MODEL_KINDS
        or model["tie_break_semantics"] != TIE_BREAK_SEMANTICS
    ):
        raise NMRCandidateScorerV9InputError("unsupported v9 model contract")
    for key in (
        "calibrated_probability",
        "conditional_probability",
        "probability",
        "automatic_selection",
        "calibration_or_sealed_read",
        "production_eligible",
    ):
        if model[key] is not False:
            raise NMRCandidateScorerV9InputError(f"v9 requires {key}=false")
    if not isinstance(model["scientific_training_complete"], bool):
        raise NMRCandidateScorerV9InputError(
            "scientific_training_complete must be boolean"
        )
    if model["promotion_status"] not in {
        "framework_smoke_only",
        "blocked_insufficient_natural_truth_coverage",
        "blocked_missing_frozen_v5_same_pool_comparator",
        "rejected_development_noninferiority",
        "eligible_for_external_evaluation_only",
    }:
        raise NMRCandidateScorerV9InputError("unsupported promotion status")
    if not allow_research_artifact:
        raise NMRCandidateScorerV9InputError(
            "v9 is research-only and requires allow_research_artifact=True"
        )

    parameters = model["model_parameters"]
    if not isinstance(parameters, Mapping):
        raise NMRCandidateScorerV9InputError("model_parameters must be an object")
    if model["model_kind"] in LINEAR_MODEL_KINDS:
        _strict_keys(parameters, _LINEAR_PARAMETER_KEYS, context="model_parameters")
        clean_parameters: dict[str, Any] = {
            "means": _vector(parameters["means"], context="model_parameters.means"),
            "scales": _vector(
                parameters["scales"],
                context="model_parameters.scales",
                positive=True,
            ),
            "coefficients": _vector(
                parameters["coefficients"], context="model_parameters.coefficients"
            ),
        }
    else:
        _strict_keys(parameters, _TREE_PARAMETER_KEYS, context="model_parameters")
        trees = parameters["trees"]
        if not isinstance(trees, list) or not trees:
            raise NMRCandidateScorerV9InputError(
                "model_parameters.trees must be non-empty"
            )
        learning_rate = _finite(
            parameters["learning_rate"], context="model_parameters.learning_rate"
        )
        if learning_rate <= 0.0:
            raise NMRCandidateScorerV9InputError("learning_rate must be positive")
        if parameters["pairwise_antisymmetric"] is not True:
            raise NMRCandidateScorerV9InputError(
                "tree scorer requires antisymmetric pairwise evaluation"
            )
        clean_parameters = {
            "means": _vector(parameters["means"], context="model_parameters.means"),
            "scales": _vector(
                parameters["scales"],
                context="model_parameters.scales",
                positive=True,
            ),
            "learning_rate": learning_rate,
            "initial_raw_score": _finite(
                parameters["initial_raw_score"],
                context="model_parameters.initial_raw_score",
            ),
            "pairwise_antisymmetric": True,
            "trees": [
                _validate_tree(tree, tree_index=index)
                for index, tree in enumerate(trees)
            ],
        }

    binding = model["development_binding"]
    runtime = model["runtime"]
    if not isinstance(binding, Mapping) or not isinstance(runtime, Mapping):
        raise NMRCandidateScorerV9InputError(
            "development_binding/runtime must be objects"
        )
    _strict_keys(binding, _DEVELOPMENT_BINDING_KEYS, context="development_binding")
    _strict_keys(runtime, _RUNTIME_KEYS, context="runtime")
    if binding["status"] not in {
        "synthetic_framework_smoke",
        "retrospective_in_domain_development",
    }:
        raise NMRCandidateScorerV9InputError("invalid development binding status")
    for key in (
        "case_count",
        "rankable_case_count",
        "training_case_count",
        "generator_empty_or_failed_count",
        "singleton_count",
        "exact_truth_miss_count",
        "no_valid_candidate_count",
        "component_count",
        "seed",
    ):
        value = binding[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise NMRCandidateScorerV9InputError(
                f"development_binding.{key} must be a nonnegative integer"
            )
    if binding["component_count"] < 2:
        raise NMRCandidateScorerV9InputError("at least two components are required")
    if binding["rankable_case_count"] > binding["case_count"]:
        raise NMRCandidateScorerV9InputError("rankable count exceeds full denominator")
    if binding["training_case_count"] > binding["rankable_case_count"]:
        raise NMRCandidateScorerV9InputError("training count exceeds rankable count")
    if not isinstance(binding["group_dimensions"], list) or not binding[
        "group_dimensions"
    ]:
        raise NMRCandidateScorerV9InputError("group dimensions are missing")
    for key in (
        "candidate_pool_sha256",
        "evidence_manifest_sha256",
        "development_gold_sha256",
        "groups_sha256",
        "scorer_sha256",
        "trainer_sha256",
    ):
        _sha256(binding[key], context=f"development_binding.{key}")
    nested = binding["nested_cv"]
    if not isinstance(nested, Mapping):
        raise NMRCandidateScorerV9InputError("nested_cv must be an object")
    _strict_keys(nested, _NESTED_CV_KEYS, context="development_binding.nested_cv")
    for key in ("outer_folds", "inner_folds_max"):
        value = nested[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise NMRCandidateScorerV9InputError(f"nested_cv.{key} is invalid")
    _sha256(nested["family_grid_sha256"], context="nested_cv.family_grid_sha256")
    if nested["selected_family"] != model["model_kind"]:
        raise NMRCandidateScorerV9InputError("selected family/model kind mismatch")
    if not isinstance(nested["selected_hyperparameters"], Mapping):
        raise NMRCandidateScorerV9InputError("selected hyperparameters must be an object")
    if not isinstance(nested["fold_semantics"], str) or not nested["fold_semantics"]:
        raise NMRCandidateScorerV9InputError("fold semantics are missing")
    if not isinstance(binding["limitations"], list) or not all(
        isinstance(item, str) and item for item in binding["limitations"]
    ):
        raise NMRCandidateScorerV9InputError("development limitations are invalid")
    if model["scientific_training_complete"] is not False:
        raise NMRCandidateScorerV9InputError(
            "v9 research artifacts must keep scientific_training_complete=false"
        )
    if (
        runtime["canonicalizer"]
        != "rdkit_molfromsmiles_then_canonical_isomeric_smiles_v1"
        or runtime != runtime_binding()
    ):
        raise NMRCandidateScorerV9InputError("RDKit runtime binding mismatch")
    return {**dict(model), "model_parameters": clean_parameters}


def _tree_value(tree: Mapping[str, Sequence[Any]], features: np.ndarray) -> float:
    node = 0
    while tree["children_left"][node] != -1:
        feature = tree["features"][node]
        node = (
            tree["children_left"][node]
            if features[feature] <= tree["thresholds"][node]
            else tree["children_right"][node]
        )
    return float(tree["values"][node])


def _tree_raw(parameters: Mapping[str, Any], difference: np.ndarray) -> float:
    return float(parameters["initial_raw_score"]) + float(
        parameters["learning_rate"]
    ) * sum(_tree_value(tree, difference) for tree in parameters["trees"])


def _utilities(
    rows: Sequence[Mapping[str, Any]], model: Mapping[str, Any]
) -> dict[str, float]:
    parameters = model["model_parameters"]
    means = np.asarray(parameters["means"], dtype=np.float64)
    scales = np.asarray(parameters["scales"], dtype=np.float64)
    valid = [row for row in rows if row["valid"]]
    normalized = {
        row["structure_sha256"]: (
            np.asarray(row["feature_values"], dtype=np.float64) - means
        )
        / scales
        for row in valid
    }
    if model["model_kind"] in LINEAR_MODEL_KINDS:
        coefficients = np.asarray(parameters["coefficients"], dtype=np.float64)
        return {
            structure_hash: float(np.dot(features, coefficients))
            for structure_hash, features in normalized.items()
        }
    output = {structure_hash: 0.0 for structure_hash in normalized}
    keys = sorted(normalized)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            difference = normalized[left] - normalized[right]
            # This removes any orientation bias from a nonlinear pairwise learner.
            preference = 0.5 * (
                _tree_raw(parameters, difference)
                - _tree_raw(parameters, -difference)
            )
            output[left] += preference
            output[right] -= preference
    denominator = max(len(keys) - 1, 1)
    return {key: value / denominator for key, value in output.items()}


def rank_candidate_evidence_v9(
    evidence_bundle: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    expected_evidence_sha256: str,
    expected_model_sha256: str,
    allow_research_artifact: bool = False,
) -> dict[str, Any]:
    """Rank a roleless pool for human review without emitting probabilities."""

    expected_evidence_hash = _sha256(
        expected_evidence_sha256, context="expected evidence hash"
    )
    if canonical_sha256(evidence_bundle) != expected_evidence_hash:
        raise NMRCandidateScorerV9InputError("evidence bundle hash mismatch")
    clean_model = validate_model_artifact(
        model,
        expected_model_sha256=expected_model_sha256,
        allow_research_artifact=allow_research_artifact,
    )
    if (
        not isinstance(evidence_bundle, Mapping)
        or set(evidence_bundle) != {"schema_version", "candidates"}
        or evidence_bundle.get("schema_version") != ROLELESS_POOL_SCHEMA_VERSION
    ):
        raise NMRCandidateScorerV9InputError(
            "unsupported Stage-4 roleless evidence-pool schema"
        )
    candidates = evidence_bundle.get("candidates")
    if (
        isinstance(candidates, (str, bytes, Mapping))
        or not isinstance(candidates, Sequence)
        or not 2 <= len(candidates) <= 500
    ):
        raise NMRCandidateScorerV9InputError(
            "runtime scoring requires 2..500 Stage-4 candidates"
        )
    v8_bundle = {
        "schema_version": V8_EVIDENCE_SCHEMA_VERSION,
        "candidates": list(candidates),
    }
    try:
        rows = extract_feature_rows(v8_bundle)
    except ValueError as exc:
        raise NMRCandidateScorerV9InputError("roleless evidence validation failed") from exc
    utilities = _utilities(rows, clean_model)
    scored = []
    for row in rows:
        utility = utilities.get(row["structure_sha256"])
        if utility is not None and not math.isfinite(utility):
            raise NMRCandidateScorerV9InputError("model produced non-finite utility")
        scored.append(
            {
                "candidate_id": row["candidate_id"],
                "canonical_smiles": row["canonical_smiles"],
                "structure_sha256": row["structure_sha256"],
                "valid": row["valid"],
                "failure_count": row["failure_count"],
                "evidence_availability": row["evidence_availability"],
                "safety": row["safety"],
                "v9_utility": utility,
            }
        )
    ordered = sorted(
        scored,
        key=lambda row: (
            not row["valid"],
            -float(row["v9_utility"] or 0.0),
            row["structure_sha256"],
        ),
    )
    valid_ordered = [row for row in ordered if row["valid"]]
    reasons: list[str] = []
    if not valid_ordered:
        reasons.extend(("no_valid_candidate", "required_carbon13_unavailable_or_failed"))
    else:
        top = valid_ordered[0]
        if top["safety"]["minimum_carbon13_coverage"] < 0.8:
            reasons.append("top_candidate_low_carbon13_coverage")
        applicability = top["safety"]["applicability_in_domain"]
        if applicability is False:
            reasons.append("top_candidate_out_of_applicability_domain")
        elif applicability is None:
            reasons.append("top_candidate_applicability_unknown")
    rank_by_structure = {
        row["structure_sha256"]: index
        for index, row in enumerate(ordered, start=1)
    }
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evidence_schema_version": ROLELESS_POOL_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "evidence_bundle_sha256": expected_evidence_hash,
        "model_artifact_sha256": expected_model_sha256,
        "model_kind": clean_model["model_kind"],
        "calibrated_probability": False,
        "conditional_probability": False,
        "probability": False,
        "automatic_selection": False,
        "production_eligible": False,
        "promotion_status": clean_model["promotion_status"],
        "decision_status": (
            "insufficient_evidence"
            if not valid_ordered
            else "abstain_recommended"
            if reasons
            else "ranked_for_human_review_only"
        ),
        "abstention_recommended": bool(reasons),
        "abstention_reasons": reasons,
        "tie_break_semantics": TIE_BREAK_SEMANTICS,
        "candidate_count": len(scored),
        "invalid_candidate_count": sum(not row["valid"] for row in scored),
        "v9_order": [row["candidate_id"] for row in ordered],
        "candidates": [
            {**row, "v9_rank": rank_by_structure[row["structure_sha256"]]}
            for row in sorted(scored, key=lambda item: item["structure_sha256"])
        ],
    }


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "LINEAR_MODEL_KINDS",
    "MODEL_KINDS",
    "MODEL_SCHEMA_VERSION",
    "NMRCandidateScorerV9InputError",
    "OUTPUT_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "ROLELESS_POOL_SCHEMA_VERSION",
    "TIE_BREAK_SEMANTICS",
    "TREE_MODEL_KIND",
    "canonical_sha256",
    "finalize_model_artifact",
    "model_artifact_sha256",
    "rank_candidate_evidence_v9",
    "runtime_binding",
    "validate_model_artifact",
]
