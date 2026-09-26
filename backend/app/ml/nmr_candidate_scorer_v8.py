"""Strict, roleless multi-evidence candidate ranking for NMR structure review.

V8 deliberately separates identity from evidence.  ``candidate_id`` is only an
opaque join key in the returned result: it is never included in a feature,
normalisation statistic, utility, or tie break.  Exact utility ties are broken
with a SHA-256 of the canonical structure.  Evaluation labels and grouping
metadata belong to the offline development pipeline and are rejected here.

The returned utility is *not* a probability and the scorer never authorises an
automatic structure selection.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
from typing import Any

import numpy as np
from rdkit import Chem, rdBase


PROTOCOL_VERSION = "chemapp.nmr.candidate-scorer.v8"
EVIDENCE_SCHEMA_VERSION = "chemapp.nmr.roleless-evidence.v8"
FEATURE_SCHEMA_VERSION = "chemapp.nmr.candidate-features.v8"
MODEL_SCHEMA_VERSION = "chemapp.nmr.candidate-ranker-model.v8"
OUTPUT_SCHEMA_VERSION = "chemapp.nmr.roleless-ranking.v8"
TIE_BREAK_SEMANTICS = "sha256(canonical_smiles)_ascending_on_exact_utility_tie_v1"

RETRIEVAL_SCORE_SEMANTICS = "normalized_structure_retrieval_similarity_v1"
RETRIEVAL_PROVENANCE_VERSION = "chemapp.retrieval-prior.v1"

FEATURE_NAMES = (
    # 13C error, bidirectional coverage, interval and peak/carbon consistency.
    "c13_matched_mae_ppm",
    "c13_matched_rmse_ppm",
    "c13_matched_max_abs_error_ppm",
    "c13_observed_coverage",
    "c13_predicted_coverage",
    "c13_count_agreement",
    "c13_interval_coverage",
    "c13_interval_mean_width_ppm",
    "c13_signal_count_log_ratio_abs",
    "c13_available",
    "c13_interval_available",
    "c13_failed",
    # Molecular formula.
    "formula_exact_match",
    "formula_element_l1_distance",
    "formula_available",
    "formula_failed",
    # Retrieval prior: fixed, order-independent semantics.
    "retrieval_prior_score",
    "retrieval_available",
    "retrieval_failed",
    # Spectrum/candidate quality and applicability domain.
    "quality_score",
    "quality_available",
    "quality_failed",
    "applicability_in_domain",
    "applicability_distance",
    "applicability_available",
    "applicability_failed",
    # Optional 1H, DEPT, and 2D evidence with explicit availability/failure.
    "h1_matched_mae_ppm",
    "h1_bidirectional_coverage",
    "h1_count_agreement",
    "h1_available",
    "h1_failed",
    "dept_class_agreement",
    "dept_coverage",
    "dept_available",
    "dept_failed",
    "two_d_edge_precision",
    "two_d_edge_recall",
    "two_d_available",
    "two_d_failed",
    # Explicit pipeline failures; these also drive fail-closed validity.
    "prediction_failed",
    "parsing_failed",
    "formula_pipeline_failed",
    "retrieval_pipeline_failed",
)

_BUNDLE_KEYS = frozenset({"schema_version", "candidates"})
_CANDIDATE_KEYS = frozenset({"candidate_id", "canonical_smiles", "evidence"})
_EVIDENCE_KEYS = frozenset(
    {
        "carbon13",
        "formula",
        "retrieval",
        "quality",
        "applicability",
        "proton1h",
        "dept",
        "two_d",
        "failures",
    }
)
_C13_KEYS = frozenset(
    {
        "status",
        "matched_mae_ppm",
        "matched_rmse_ppm",
        "matched_max_abs_error_ppm",
        "observed_coverage",
        "predicted_coverage",
        "count_agreement",
        "interval_available",
        "interval_coverage",
        "interval_mean_width_ppm",
        "observed_signal_count",
        "predicted_signal_count",
    }
)
_FORMULA_KEYS = frozenset({"status", "exact_match", "element_l1_distance"})
_RETRIEVAL_KEYS = frozenset(
    {
        "status",
        "prior_score",
        "score_semantics",
        "provenance_version",
        "query_independent_of_candidate_order",
    }
)
_QUALITY_KEYS = frozenset({"status", "score"})
_APPLICABILITY_KEYS = frozenset({"status", "in_domain", "distance"})
_H1_KEYS = frozenset(
    {"status", "matched_mae_ppm", "bidirectional_coverage", "count_agreement"}
)
_DEPT_KEYS = frozenset({"status", "class_agreement", "coverage"})
_TWO_D_KEYS = frozenset({"status", "edge_precision", "edge_recall"})
_FAILURE_KEYS = frozenset(
    {
        "prediction_failed",
        "parsing_failed",
        "formula_failed",
        "retrieval_failed",
    }
)
_NORMALIZATION_KEYS = frozenset({"means", "scales"})
_MODEL_KEYS = frozenset(
    {
        "artifact_sha256",
        "automatic_selection",
        "calibration_or_test_gold_read",
        "calibrated_probability",
        "coefficients",
        "conditional_probability",
        "development_binding",
        "feature_names",
        "feature_schema_version",
        "fit_intercept",
        "intercept",
        "model_kind",
        "normalization",
        "probability",
        "production_eligible",
        "promotion_status",
        "protocol_version",
        "retrieval_prior_contract",
        "runtime",
        "schema_version",
        "scientific_training_complete",
        "tie_break_semantics",
    }
)
_DEVELOPMENT_BINDING_KEYS = frozenset(
    {
        "case_count",
        "cases_sha256",
        "component_count",
        "cv",
        "group_dimensions",
        "input_provenance",
        "legacy_group_proxy",
        "manifest_sha256",
        "scientific_training_complete",
        "scorer_sha256",
        "seed",
        "source_snapshot_sha256",
        "status",
        "trainer_sha256",
        "data_protocol_sha256",
    }
)
_RUNTIME_KEYS = frozenset(
    {
        "canonicalizer",
        "numpy_version",
        "python_version",
        "rdkit_version",
        "sklearn_version",
    }
)
_CV_BINDING_KEYS = frozenset(
    {"c_grid", "fold_assignment", "folds", "selected_c"}
)
_INPUT_PROVENANCE_KEYS = frozenset(
    {
        "approval_status",
        "adapter",
        "development_gold_sha256",
        "frozen_input_contract_sha256",
        "rankings_sha256",
        "release_id",
        "split_manifest_sha256",
        "v5_model_sha256",
    }
)

# These tokens are forbidden recursively in roleless evidence.  Candidate ID is
# an explicitly allowed root field, but no spelling of identity/order is
# accepted inside evidence, where it could become a learned shortcut.
_BANNED_EVIDENCE_KEY_FRAGMENTS = (
    "candidate_id",
    "source",
    "order",
    "position",
    "row_id",
    "rank",
    "truth",
    "gold",
    "label",
    "outcome",
    "correct",
    "split",
    "fold",
    "doi",
    "scaffold",
    "connectivity",
    "reviewer",
)
_CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class NMRCandidateScorerV8InputError(ValueError):
    """Raised when evidence or a model violates the frozen v8 contract."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NMRCandidateScorerV8InputError(
            "value is not finite canonical JSON"
        ) from exc
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_keys(
    value: Mapping[str, Any], expected: frozenset[str], *, context: str
) -> None:
    actual = set(value)
    if actual != expected:
        raise NMRCandidateScorerV8InputError(
            f"{context} fields mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _reject_evidence_leakage(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise NMRCandidateScorerV8InputError(
                    f"{path} contains a non-string key"
                )
            lowered = key.casefold()
            if key != "query_independent_of_candidate_order" and any(
                token in lowered for token in _BANNED_EVIDENCE_KEY_FRAGMENTS
            ):
                raise NMRCandidateScorerV8InputError(
                    f"{path}.{key} contains prohibited identity/order/outcome metadata"
                )
            _reject_evidence_leakage(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_evidence_leakage(nested, path=f"{path}[{index}]")


def _finite(value: Any, *, context: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NMRCandidateScorerV8InputError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise NMRCandidateScorerV8InputError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise NMRCandidateScorerV8InputError(
            f"{context} must be at least {minimum}"
        )
    return result


def _unit(value: Any, *, context: str) -> float:
    result = _finite(value, context=context)
    if not 0.0 <= result <= 1.0:
        raise NMRCandidateScorerV8InputError(f"{context} must be within 0..1")
    return result


def _boolean(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise NMRCandidateScorerV8InputError(f"{context} must be boolean")
    return value


def _status(value: Any, *, context: str) -> str:
    if value not in {"ok", "missing", "failed"}:
        raise NMRCandidateScorerV8InputError(
            f"{context} must be one of ok/missing/failed"
        )
    return str(value)


def _nullable_metric(
    value: Any,
    *,
    present: bool,
    context: str,
    unit: bool = False,
    minimum: float | None = 0.0,
) -> float:
    if not present:
        if value is not None:
            raise NMRCandidateScorerV8InputError(
                f"{context} must be null when evidence is unavailable"
            )
        return 0.0
    if value is None:
        raise NMRCandidateScorerV8InputError(
            f"{context} is required when evidence status is ok"
        )
    if unit:
        return _unit(value, context=context)
    return _finite(value, context=context, minimum=minimum)


def _normalise_status_block(
    block: Any,
    expected_keys: frozenset[str],
    *,
    context: str,
) -> tuple[Mapping[str, Any], str, bool]:
    if not isinstance(block, Mapping):
        raise NMRCandidateScorerV8InputError(f"{context} must be an object")
    _strict_keys(block, expected_keys, context=context)
    status = _status(block["status"], context=f"{context}.status")
    return block, status, status == "ok"


def _extract_candidate(candidate: Any, *, position: int) -> dict[str, Any]:
    context = f"candidates[{position}]"
    if not isinstance(candidate, Mapping):
        raise NMRCandidateScorerV8InputError(f"{context} must be an object")
    _strict_keys(candidate, _CANDIDATE_KEYS, context=context)
    candidate_id = candidate["candidate_id"]
    if (
        not isinstance(candidate_id, str)
        or _CANDIDATE_ID_RE.fullmatch(candidate_id) is None
    ):
        raise NMRCandidateScorerV8InputError(
            f"{context}.candidate_id violates the opaque join-key contract"
        )
    smiles = candidate["canonical_smiles"]
    if not isinstance(smiles, str) or not smiles or smiles != smiles.strip():
        raise NMRCandidateScorerV8InputError(
            f"{context}.canonical_smiles must be non-empty and stripped"
        )
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise NMRCandidateScorerV8InputError(
            f"{context}.canonical_smiles must describe one valid molecule"
        )
    canonical_smiles = Chem.MolToSmiles(
        molecule, canonical=True, isomericSmiles=True
    )
    if canonical_smiles != smiles:
        raise NMRCandidateScorerV8InputError(
            f"{context}.canonical_smiles is not RDKit canonical isomeric SMILES"
        )
    evidence = candidate["evidence"]
    if not isinstance(evidence, Mapping):
        raise NMRCandidateScorerV8InputError(f"{context}.evidence must be an object")
    _reject_evidence_leakage(evidence, path=f"{context}.evidence")
    _strict_keys(evidence, _EVIDENCE_KEYS, context=f"{context}.evidence")

    c13, c13_status, c13_ok = _normalise_status_block(
        evidence["carbon13"], _C13_KEYS, context=f"{context}.evidence.carbon13"
    )
    c13_mae = _nullable_metric(
        c13["matched_mae_ppm"], present=c13_ok, context=f"{context}.c13.mae"
    )
    c13_rmse = _nullable_metric(
        c13["matched_rmse_ppm"], present=c13_ok, context=f"{context}.c13.rmse"
    )
    c13_max = _nullable_metric(
        c13["matched_max_abs_error_ppm"],
        present=c13_ok,
        context=f"{context}.c13.max_error",
    )
    if c13_ok and not c13_mae <= c13_rmse + 1e-12 <= c13_max + 1e-12:
        raise NMRCandidateScorerV8InputError(
            f"{context}.carbon13 error metrics are inconsistent"
        )
    c13_observed_coverage = _nullable_metric(
        c13["observed_coverage"],
        present=c13_ok,
        context=f"{context}.c13.observed_coverage",
        unit=True,
    )
    c13_predicted_coverage = _nullable_metric(
        c13["predicted_coverage"],
        present=c13_ok,
        context=f"{context}.c13.predicted_coverage",
        unit=True,
    )
    c13_count_agreement = _nullable_metric(
        c13["count_agreement"],
        present=c13_ok,
        context=f"{context}.c13.count_agreement",
        unit=True,
    )
    interval_available = _boolean(
        c13["interval_available"], context=f"{context}.c13.interval_available"
    )
    if not c13_ok and interval_available:
        raise NMRCandidateScorerV8InputError(
            f"{context}.carbon13 interval cannot exist without 13C evidence"
        )
    interval_coverage = _nullable_metric(
        c13["interval_coverage"],
        present=interval_available,
        context=f"{context}.c13.interval_coverage",
        unit=True,
    )
    interval_width = _nullable_metric(
        c13["interval_mean_width_ppm"],
        present=interval_available,
        context=f"{context}.c13.interval_mean_width_ppm",
    )
    observed_count = c13["observed_signal_count"]
    predicted_count = c13["predicted_signal_count"]
    if c13_ok:
        if (
            isinstance(observed_count, bool)
            or not isinstance(observed_count, int)
            or observed_count <= 0
            or isinstance(predicted_count, bool)
            or not isinstance(predicted_count, int)
            or predicted_count <= 0
        ):
            raise NMRCandidateScorerV8InputError(
                f"{context}.carbon13 signal counts must be positive integers"
            )
        expected_count_agreement = min(observed_count, predicted_count) / max(
            observed_count, predicted_count
        )
        if not math.isclose(
            c13_count_agreement, expected_count_agreement, abs_tol=1e-12
        ):
            raise NMRCandidateScorerV8InputError(
                f"{context}.carbon13 count agreement is inconsistent"
            )
        signal_log_ratio = abs(math.log(observed_count / predicted_count))
    else:
        if observed_count is not None or predicted_count is not None:
            raise NMRCandidateScorerV8InputError(
                f"{context}.carbon13 counts must be null when unavailable"
            )
        signal_log_ratio = 0.0

    formula, formula_status, formula_ok = _normalise_status_block(
        evidence["formula"], _FORMULA_KEYS, context=f"{context}.evidence.formula"
    )
    if formula_ok:
        formula_exact = _boolean(
            formula["exact_match"], context=f"{context}.formula.exact_match"
        )
    else:
        if formula["exact_match"] is not None:
            raise NMRCandidateScorerV8InputError(
                f"{context}.formula.exact_match must be null when unavailable"
            )
        formula_exact = False
    formula_distance = _nullable_metric(
        formula["element_l1_distance"],
        present=formula_ok,
        context=f"{context}.formula.element_l1_distance",
    )
    if formula_ok and formula_exact != math.isclose(formula_distance, 0.0):
        raise NMRCandidateScorerV8InputError(
            f"{context}.formula exact flag and distance disagree"
        )

    retrieval, retrieval_status, retrieval_ok = _normalise_status_block(
        evidence["retrieval"],
        _RETRIEVAL_KEYS,
        context=f"{context}.evidence.retrieval",
    )
    if retrieval["score_semantics"] != RETRIEVAL_SCORE_SEMANTICS:
        raise NMRCandidateScorerV8InputError(
            f"{context}.retrieval score semantics are not the frozen v1 semantics"
        )
    if retrieval["provenance_version"] != RETRIEVAL_PROVENANCE_VERSION:
        raise NMRCandidateScorerV8InputError(
            f"{context}.retrieval provenance version is unsupported"
        )
    if retrieval["query_independent_of_candidate_order"] is not True:
        raise NMRCandidateScorerV8InputError(
            f"{context}.retrieval must attest candidate-order independence"
        )
    retrieval_score = _nullable_metric(
        retrieval["prior_score"],
        present=retrieval_ok,
        context=f"{context}.retrieval.prior_score",
        unit=True,
    )

    quality, quality_status, quality_ok = _normalise_status_block(
        evidence["quality"], _QUALITY_KEYS, context=f"{context}.evidence.quality"
    )
    quality_score = _nullable_metric(
        quality["score"],
        present=quality_ok,
        context=f"{context}.quality.score",
        unit=True,
    )

    applicability, applicability_status, applicability_ok = _normalise_status_block(
        evidence["applicability"],
        _APPLICABILITY_KEYS,
        context=f"{context}.evidence.applicability",
    )
    if applicability_ok:
        applicability_in_domain = _boolean(
            applicability["in_domain"],
            context=f"{context}.applicability.in_domain",
        )
    else:
        if applicability["in_domain"] is not None:
            raise NMRCandidateScorerV8InputError(
                f"{context}.applicability.in_domain must be null when unavailable"
            )
        applicability_in_domain = False
    applicability_distance = _nullable_metric(
        applicability["distance"],
        present=applicability_ok,
        context=f"{context}.applicability.distance",
    )

    h1, h1_status, h1_ok = _normalise_status_block(
        evidence["proton1h"], _H1_KEYS, context=f"{context}.evidence.proton1h"
    )
    h1_mae = _nullable_metric(
        h1["matched_mae_ppm"], present=h1_ok, context=f"{context}.h1.mae"
    )
    h1_coverage = _nullable_metric(
        h1["bidirectional_coverage"],
        present=h1_ok,
        context=f"{context}.h1.bidirectional_coverage",
        unit=True,
    )
    h1_count = _nullable_metric(
        h1["count_agreement"],
        present=h1_ok,
        context=f"{context}.h1.count_agreement",
        unit=True,
    )

    dept, dept_status, dept_ok = _normalise_status_block(
        evidence["dept"], _DEPT_KEYS, context=f"{context}.evidence.dept"
    )
    dept_agreement = _nullable_metric(
        dept["class_agreement"],
        present=dept_ok,
        context=f"{context}.dept.class_agreement",
        unit=True,
    )
    dept_coverage = _nullable_metric(
        dept["coverage"],
        present=dept_ok,
        context=f"{context}.dept.coverage",
        unit=True,
    )

    two_d, two_d_status, two_d_ok = _normalise_status_block(
        evidence["two_d"], _TWO_D_KEYS, context=f"{context}.evidence.two_d"
    )
    two_d_precision = _nullable_metric(
        two_d["edge_precision"],
        present=two_d_ok,
        context=f"{context}.two_d.edge_precision",
        unit=True,
    )
    two_d_recall = _nullable_metric(
        two_d["edge_recall"],
        present=two_d_ok,
        context=f"{context}.two_d.edge_recall",
        unit=True,
    )

    failures = evidence["failures"]
    if not isinstance(failures, Mapping):
        raise NMRCandidateScorerV8InputError(f"{context}.failures must be an object")
    _strict_keys(failures, _FAILURE_KEYS, context=f"{context}.failures")
    failure_flags = {
        key: _boolean(value, context=f"{context}.failures.{key}")
        for key, value in failures.items()
    }
    if failure_flags["prediction_failed"] != (c13_status == "failed"):
        raise NMRCandidateScorerV8InputError(
            f"{context}.prediction failure flag disagrees with carbon13 status"
        )
    if failure_flags["formula_failed"] != (formula_status == "failed"):
        raise NMRCandidateScorerV8InputError(
            f"{context}.formula failure flag disagrees with formula status"
        )
    if failure_flags["retrieval_failed"] != (retrieval_status == "failed"):
        raise NMRCandidateScorerV8InputError(
            f"{context}.retrieval failure flag disagrees with retrieval status"
        )

    values = (
        c13_mae,
        c13_rmse,
        c13_max,
        c13_observed_coverage,
        c13_predicted_coverage,
        c13_count_agreement,
        interval_coverage,
        interval_width,
        signal_log_ratio,
        float(c13_ok),
        float(interval_available),
        float(c13_status == "failed"),
        float(formula_exact),
        formula_distance,
        float(formula_ok),
        float(formula_status == "failed"),
        retrieval_score,
        float(retrieval_ok),
        float(retrieval_status == "failed"),
        quality_score,
        float(quality_ok),
        float(quality_status == "failed"),
        float(applicability_in_domain),
        applicability_distance,
        float(applicability_ok),
        float(applicability_status == "failed"),
        h1_mae,
        h1_coverage,
        h1_count,
        float(h1_ok),
        float(h1_status == "failed"),
        dept_agreement,
        dept_coverage,
        float(dept_ok),
        float(dept_status == "failed"),
        two_d_precision,
        two_d_recall,
        float(two_d_ok),
        float(two_d_status == "failed"),
        float(failure_flags["prediction_failed"]),
        float(failure_flags["parsing_failed"]),
        float(failure_flags["formula_failed"]),
        float(failure_flags["retrieval_failed"]),
    )
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(v) for v in values):
        raise AssertionError("v8 feature construction violated its frozen schema")
    hard_failures = (
        failure_flags["prediction_failed"],
        failure_flags["parsing_failed"],
        not c13_ok,
    )
    return {
        "candidate_id": candidate_id,
        "canonical_smiles": smiles,
        "structure_sha256": hashlib.sha256(smiles.encode("utf-8")).hexdigest(),
        "feature_names": list(FEATURE_NAMES),
        "feature_values": list(values),
        "valid": not any(hard_failures),
        "failure_count": sum(bool(value) for value in failure_flags.values()),
        "safety": {
            "required_carbon13_ok": c13_ok,
            "minimum_carbon13_coverage": min(
                c13_observed_coverage, c13_predicted_coverage
            )
            if c13_ok
            else 0.0,
            "applicability_in_domain": applicability_in_domain
            if applicability_ok
            else None,
        },
        "evidence_availability": {
            "carbon13": c13_status,
            "formula": formula_status,
            "retrieval": retrieval_status,
            "quality": quality_status,
            "applicability": applicability_status,
            "proton1h": h1_status,
            "dept": dept_status,
            "two_d": two_d_status,
        },
    }


def extract_feature_rows(evidence_bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate roleless evidence and return input-order-neutral feature rows."""

    if not isinstance(evidence_bundle, Mapping):
        raise NMRCandidateScorerV8InputError("evidence bundle must be an object")
    _strict_keys(evidence_bundle, _BUNDLE_KEYS, context="evidence bundle")
    if evidence_bundle["schema_version"] != EVIDENCE_SCHEMA_VERSION:
        raise NMRCandidateScorerV8InputError("unsupported evidence schema version")
    candidates = evidence_bundle["candidates"]
    if (
        isinstance(candidates, (str, bytes, Mapping))
        or not isinstance(candidates, Sequence)
        or not 2 <= len(candidates) <= 500
    ):
        raise NMRCandidateScorerV8InputError(
            "candidates must contain between 2 and 500 entries"
        )
    rows = [_extract_candidate(candidate, position=i) for i, candidate in enumerate(candidates)]
    ids = [row["candidate_id"] for row in rows]
    structures = [row["structure_sha256"] for row in rows]
    if len(set(ids)) != len(ids):
        raise NMRCandidateScorerV8InputError("candidate IDs must be unique")
    if len(set(structures)) != len(structures):
        raise NMRCandidateScorerV8InputError(
            "canonical structures must be unique within a candidate pool"
        )
    # Stable serialisation is useful for hashing but cannot affect utilities.
    return sorted(rows, key=lambda row: row["structure_sha256"])


def model_artifact_sha256(model: Mapping[str, Any]) -> str:
    return canonical_sha256({key: value for key, value in model.items() if key != "artifact_sha256"})


def finalize_model_artifact(model_core: Mapping[str, Any]) -> dict[str, Any]:
    if "artifact_sha256" in model_core:
        raise NMRCandidateScorerV8InputError(
            "model core must not self-declare artifact_sha256"
        )
    artifact = dict(model_core)
    artifact["artifact_sha256"] = model_artifact_sha256(artifact)
    return artifact


def validate_model_artifact(
    model: Mapping[str, Any],
    *,
    expected_model_sha256: str,
    allow_research_artifact: bool = False,
) -> dict[str, Any]:
    if not isinstance(model, Mapping):
        raise NMRCandidateScorerV8InputError("model must be an object")
    _strict_keys(model, _MODEL_KEYS, context="v8 model")
    if not isinstance(expected_model_sha256, str) or _SHA256_RE.fullmatch(
        expected_model_sha256
    ) is None:
        raise NMRCandidateScorerV8InputError("expected model hash must be SHA-256")
    actual = model_artifact_sha256(model)
    if model["artifact_sha256"] != actual or expected_model_sha256 != actual:
        raise NMRCandidateScorerV8InputError("v8 model artifact hash mismatch")
    if (
        model["schema_version"] != MODEL_SCHEMA_VERSION
        or model["protocol_version"] != PROTOCOL_VERSION
        or model["feature_schema_version"] != FEATURE_SCHEMA_VERSION
        or model["feature_names"] != list(FEATURE_NAMES)
        or model["model_kind"] != "symmetric_pairwise_l2_logistic_v1"
        or model["fit_intercept"] is not False
        or model["intercept"] != 0.0
        or model["calibrated_probability"] is not False
        or model["conditional_probability"] is not False
        or model["probability"] is not False
        or model["automatic_selection"] is not False
        or model["scientific_training_complete"] is not False
        or model["tie_break_semantics"] != TIE_BREAK_SEMANTICS
        or model["retrieval_prior_contract"]
        != {
            "score_semantics": RETRIEVAL_SCORE_SEMANTICS,
            "provenance_version": RETRIEVAL_PROVENANCE_VERSION,
            "candidate_order_independent": True,
        }
    ):
        raise NMRCandidateScorerV8InputError("unsupported v8 model contract")
    if model["promotion_status"] not in {
        "research_smoke_only",
        "rejected_development_noninferiority",
        "eligible_for_independent_evaluation",
    } or not isinstance(model["production_eligible"], bool):
        raise NMRCandidateScorerV8InputError("invalid model promotion contract")
    if model["production_eligible"] is not False:
        raise NMRCandidateScorerV8InputError(
            "v8 has no independent-release verifier and cannot be production eligible"
        )
    if not allow_research_artifact:
        raise NMRCandidateScorerV8InputError(
            "non-production v8 artifact requires allow_research_artifact=True"
        )
    normalization = model["normalization"]
    if not isinstance(normalization, Mapping):
        raise NMRCandidateScorerV8InputError("normalization must be an object")
    _strict_keys(normalization, _NORMALIZATION_KEYS, context="normalization")
    vectors: list[list[float]] = []
    for name, raw in (
        ("means", normalization["means"]),
        ("scales", normalization["scales"]),
        ("coefficients", model["coefficients"]),
    ):
        if not isinstance(raw, list) or len(raw) != len(FEATURE_NAMES):
            raise NMRCandidateScorerV8InputError(
                f"model {name} has the wrong feature dimension"
            )
        vector = [_finite(value, context=f"model.{name}[{i}]") for i, value in enumerate(raw)]
        vectors.append(vector)
    if any(value <= 0.0 for value in vectors[1]):
        raise NMRCandidateScorerV8InputError("normalization scales must be positive")
    development_binding = model["development_binding"]
    runtime = model["runtime"]
    if not isinstance(development_binding, Mapping) or not isinstance(runtime, Mapping):
        raise NMRCandidateScorerV8InputError(
            "model development/runtime bindings must be objects"
        )
    _strict_keys(
        development_binding,
        _DEVELOPMENT_BINDING_KEYS,
        context="model.development_binding",
    )
    _strict_keys(runtime, _RUNTIME_KEYS, context="model.runtime")
    if (
        development_binding["status"]
        not in {
            "smoke_only",
            "retrospective_non_independent",
            "development_grouped_cv",
        }
        or not isinstance(development_binding["scientific_training_complete"], bool)
        or development_binding["group_dimensions"]
        != ["doi", "scaffold", "connectivity"]
        or not isinstance(development_binding["legacy_group_proxy"], bool)
        or not isinstance(development_binding["cv"], Mapping)
        or not isinstance(development_binding["input_provenance"], Mapping)
        or isinstance(development_binding["seed"], bool)
        or not isinstance(development_binding["seed"], int)
        or isinstance(development_binding["case_count"], bool)
        or not isinstance(development_binding["case_count"], int)
        or development_binding["case_count"] < 1
        or isinstance(development_binding["component_count"], bool)
        or not isinstance(development_binding["component_count"], int)
        or development_binding["component_count"] < 2
    ):
        raise NMRCandidateScorerV8InputError(
            "model development binding violates the frozen development-only contract"
        )
    if (
        development_binding["scientific_training_complete"]
        and development_binding["status"] != "development_grouped_cv"
    ):
        raise NMRCandidateScorerV8InputError(
            "only a populated development grouped-CV artifact may mark training complete"
        )
    _strict_keys(
        development_binding["cv"], _CV_BINDING_KEYS, context="model.development_binding.cv"
    )
    _strict_keys(
        development_binding["input_provenance"],
        _INPUT_PROVENANCE_KEYS,
        context="model.development_binding.input_provenance",
    )
    if (
        isinstance(development_binding["cv"]["folds"], bool)
        or not isinstance(development_binding["cv"]["folds"], int)
        or development_binding["cv"]["folds"] < 2
        or not isinstance(development_binding["cv"]["c_grid"], list)
        or not development_binding["cv"]["c_grid"]
        or not isinstance(development_binding["cv"]["selected_c"], (int, float))
        or isinstance(development_binding["cv"]["selected_c"], bool)
        or development_binding["cv"]["fold_assignment"]
        != (
            "connected_components_of_doi_scaffold_connectivity_and_"
            "ecfp4_tanimoto_0.70_then_balanced_hash_v1"
        )
    ):
        raise NMRCandidateScorerV8InputError("model CV binding is invalid")
    if development_binding["input_provenance"]["adapter"] not in {
        "v8_manifest_v1",
        "v4_retrospective_development_only_v1",
        "synthetic_smoke_v1",
    }:
        raise NMRCandidateScorerV8InputError("unsupported development adapter")
    approval_status = development_binding["input_provenance"]["approval_status"]
    if approval_status not in {
        "frozen_default_inputs_approved",
        "unsafe_custom_research_inputs",
    }:
        raise NMRCandidateScorerV8InputError("unsupported input approval status")
    if (
        approval_status == "frozen_default_inputs_approved"
        and model["calibration_or_test_gold_read"] is not False
    ) or (
        approval_status == "unsafe_custom_research_inputs"
        and model["calibration_or_test_gold_read"] is not None
    ):
        raise NMRCandidateScorerV8InputError(
            "Gold-access claim disagrees with the input approval contract"
        )
    if not isinstance(
        development_binding["input_provenance"]["release_id"], str
    ) or not development_binding["input_provenance"]["release_id"]:
        raise NMRCandidateScorerV8InputError("input release_id must be non-empty")
    for key in (
        "development_gold_sha256",
        "frozen_input_contract_sha256",
        "rankings_sha256",
        "split_manifest_sha256",
        "v5_model_sha256",
    ):
        if not isinstance(
            development_binding["input_provenance"][key], str
        ) or _SHA256_RE.fullmatch(development_binding["input_provenance"][key]) is None:
            raise NMRCandidateScorerV8InputError(
                f"model input provenance {key} must be SHA-256"
            )
    for key in (
        "manifest_sha256",
        "cases_sha256",
        "scorer_sha256",
        "trainer_sha256",
        "data_protocol_sha256",
        "source_snapshot_sha256",
    ):
        if not isinstance(development_binding[key], str) or _SHA256_RE.fullmatch(
            development_binding[key]
        ) is None:
            raise NMRCandidateScorerV8InputError(
                f"model.development_binding.{key} must be SHA-256"
            )
    if (
        runtime["canonicalizer"]
        != "rdkit_molfromsmiles_then_canonical_isomeric_smiles_v1"
        or runtime["rdkit_version"] != rdBase.rdkitVersion
    ):
        raise NMRCandidateScorerV8InputError(
            "model RDKit canonicalizer/runtime does not match this process"
        )
    for key in ("numpy_version", "python_version", "sklearn_version"):
        if not isinstance(runtime[key], str) or not runtime[key]:
            raise NMRCandidateScorerV8InputError(
                f"model.runtime.{key} must be non-empty"
            )
    return {
        **dict(model),
        "normalization": {"means": vectors[0], "scales": vectors[1]},
        "coefficients": vectors[2],
    }


def rank_candidate_evidence_v8(
    evidence_bundle: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    expected_model_sha256: str,
    allow_research_artifact: bool = False,
) -> dict[str, Any]:
    """Rank candidates without emitting probabilities or selecting a structure."""

    clean_model = validate_model_artifact(
        model,
        expected_model_sha256=expected_model_sha256,
        allow_research_artifact=allow_research_artifact,
    )
    rows = extract_feature_rows(evidence_bundle)
    means = np.asarray(clean_model["normalization"]["means"], dtype=np.float64)
    scales = np.asarray(clean_model["normalization"]["scales"], dtype=np.float64)
    coefficients = np.asarray(clean_model["coefficients"], dtype=np.float64)
    scored: list[dict[str, Any]] = []
    for row in rows:
        utility = float(
            np.dot(
                (np.asarray(row["feature_values"], dtype=np.float64) - means) / scales,
                coefficients,
            )
        )
        if not math.isfinite(utility):
            raise NMRCandidateScorerV8InputError("model produced non-finite utility")
        scored.append(
            {
                "candidate_id": row["candidate_id"],
                "canonical_smiles": row["canonical_smiles"],
                "structure_sha256": row["structure_sha256"],
                "valid": row["valid"],
                "failure_count": row["failure_count"],
                "evidence_availability": row["evidence_availability"],
                "safety": row["safety"],
                "v8_utility": utility if row["valid"] else None,
            }
        )
    ordered = sorted(
        scored,
        key=lambda row: (
            not row["valid"],
            -(row["v8_utility"] if row["v8_utility"] is not None else 0.0),
            row["failure_count"],
            row["structure_sha256"],
        ),
    )
    rank_by_structure = {
        row["structure_sha256"]: rank for rank, row in enumerate(ordered, start=1)
    }
    valid_ordered = [row for row in ordered if row["valid"]]
    abstention_reasons: list[str] = []
    if not valid_ordered:
        abstention_reasons.extend(
            ["no_valid_candidate", "required_carbon13_unavailable_or_failed"]
        )
    else:
        top = valid_ordered[0]
        if top["safety"]["minimum_carbon13_coverage"] < 0.8:
            abstention_reasons.append("top_candidate_low_carbon13_coverage")
        if top["safety"]["applicability_in_domain"] is False:
            abstention_reasons.append("top_candidate_out_of_applicability_domain")
        if top["safety"]["applicability_in_domain"] is None:
            abstention_reasons.append("top_candidate_applicability_unknown")
    return {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "model_artifact_sha256": expected_model_sha256,
        "calibrated_probability": False,
        "conditional_probability": False,
        "probability": False,
        "automatic_selection": False,
        "calibration_or_test_gold_read": clean_model[
            "calibration_or_test_gold_read"
        ],
        "production_eligible": clean_model["production_eligible"],
        "promotion_status": clean_model["promotion_status"],
        "decision_status": (
            "insufficient_evidence"
            if not valid_ordered
            else (
                "abstain_recommended"
                if abstention_reasons
                else "ranked_for_human_review_only"
            )
        ),
        "abstention_recommended": bool(abstention_reasons),
        "abstention_reasons": abstention_reasons,
        "tie_break_semantics": TIE_BREAK_SEMANTICS,
        "candidate_count": len(scored),
        "invalid_candidate_count": sum(not row["valid"] for row in scored),
        "v8_order": [row["candidate_id"] for row in ordered],
        "candidates": [
            {**row, "v8_rank": rank_by_structure[row["structure_sha256"]]}
            for row in sorted(scored, key=lambda item: item["structure_sha256"])
        ],
    }


__all__ = [
    "EVIDENCE_SCHEMA_VERSION",
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "MODEL_SCHEMA_VERSION",
    "NMRCandidateScorerV8InputError",
    "OUTPUT_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "RETRIEVAL_PROVENANCE_VERSION",
    "RETRIEVAL_SCORE_SEMANTICS",
    "TIE_BREAK_SEMANTICS",
    "canonical_json_bytes",
    "canonical_sha256",
    "extract_feature_rows",
    "finalize_model_artifact",
    "model_artifact_sha256",
    "rank_candidate_evidence_v8",
    "validate_model_artifact",
]
