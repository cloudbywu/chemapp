"""Fail-closed probability calibration primitives for NMR candidate ranking v5.

The module calibrates one narrowly defined quantity::

    P(top-1 exact structure is correct |
      truth was retrieved, QC passed, input is in-domain,
      candidate generator and ranker releases are fixed)

It does *not* turn candidate-pool recall or end-to-end structure elucidation
into a probability.  Feature construction is outcome-free, all statistical
splits are group isolated, and unsafe sample sizes stop the workflow before a
model can be fitted.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import re
from typing import Any

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.special import expit, logsumexp
from scipy.stats import binom


FEATURE_SCHEMA_VERSION = "chemapp.nmr.probability-features.v5"
OUTCOME_SCHEMA_VERSION = "chemapp.nmr.probability-outcomes.v5"
SELECTION_SCHEMA_VERSION = "chemapp.nmr.ridge-selection.v5"
CALIBRATOR_SCHEMA_VERSION = "chemapp.nmr.top1-calibrator.v5"
PREDICTION_SCHEMA_VERSION = "chemapp.nmr.top1-probabilities.v5"
POLICY_SCHEMA_VERSION = "chemapp.nmr.abstention-policy.v5"
PROTOCOL_VERSION = 1

ROLES = ("dev", "prob_cal", "risk_cal", "external_test")
FEATURE_NAMES = (
    "pool_log_odds",
    "log_candidate_count",
    "normalized_pool_entropy",
    "log1p_top1_matched_mae_ppm",
    "log1p_top1_max_abs_error_ppm",
    "top1_bidirectional_coverage",
    "v3_v5_agreement",
)
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
RISK_THRESHOLDS = tuple(round(0.50 + 0.01 * index, 2) for index in range(50))

MIN_ERROR_GROUPS = 50
MIN_CORRECT_GROUPS = 100
CV_FOLDS = 5
MIN_ERROR_GROUPS_PER_FOLD = 5
MIN_RISK_GROUPS = 100
MIN_ACCEPTED_RISK_GROUPS = 60
TARGET_SELECTIVE_ERROR = 0.05
RISK_FAILURE_PROBABILITY = 0.05
MIN_RISK_COVERAGE = 0.30

CONDITIONAL_TARGET = (
    "P(top1_exact_structure_correct | truth_retrieved, qc_pass, in_domain, "
    "fixed_candidate_generator_and_ranker_release)"
)
RISK_TARGET = "end_to_end_incorrect_auto_accept"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_FEATURE_INPUT_FIELDS = frozenset(
    {
        "record_id",
        "role",
        "split_group",
        "spectrum_fingerprint_sha256",
        "model_release_id",
        "candidate_generator_release_id",
        "ranking_logit_semantics",
        "reference_top_candidate_id",
        "qc_pass",
        "in_domain",
        "applicability_reasons",
        "candidates",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "rank",
        "ranking_logit",
        "matched_mae_ppm",
        "matched_max_abs_error_ppm",
        "bidirectional_coverage",
    }
)
_OUTCOME_INPUT_FIELDS = frozenset(
    {
        "record_id",
        "role",
        "split_group",
        "truth_candidate_id",
        "truth_retrieved",
    }
)
_FEATURE_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "outcome_free",
        "target_semantics",
        "feature_names",
        "model_release_id",
        "candidate_generator_release_id",
        "rows",
        "artifact_sha256",
    }
)
_FEATURE_ROW_FIELDS = frozenset(
    {
        "schema_version",
        "record_id",
        "role",
        "split_group",
        "spectrum_fingerprint_sha256",
        "model_release_id",
        "candidate_generator_release_id",
        "ranking_logit_semantics",
        "candidate_pool_sha256",
        "candidate_ids",
        "top_candidate_id",
        "qc_pass",
        "in_domain",
        "applicability_reasons",
        "features",
    }
)
_OUTCOME_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "feature_artifact_sha256",
        "model_release_id",
        "candidate_generator_release_id",
        "role",
        "conditional_target_semantics",
        "retrieval_and_end_to_end_are_separate",
        "rows",
        "artifact_sha256",
    }
)
_OUTCOME_ROW_FIELDS = frozenset(
    {
        "record_id",
        "role",
        "split_group",
        "top_candidate_id",
        "truth_candidate_id",
        "truth_retrieved",
        "conditional_top1_correct",
        "end_to_end_correct",
        "qc_pass",
        "in_domain",
        "features",
    }
)
_SELECTION_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "method",
        "target_semantics",
        "split_roles_consumed",
        "feature_names",
        "lambda_grid",
        "selection_rule",
        "selected_lambda",
        "one_standard_error_limit",
        "candidates",
        "normalization",
        "group_weighting",
        "n_splits",
        "seed",
        "fold_audit",
        "group_assignments_sha256",
        "dev_gate",
        "model_release_id",
        "candidate_generator_release_id",
        "outcome_artifact_sha256",
        "artifact_sha256",
    }
)
_CALIBRATOR_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "method",
        "conditional_target_semantics",
        "not_retrieval_probability",
        "not_end_to_end_probability",
        "model_release_id",
        "candidate_generator_release_id",
        "feature_names",
        "normalization",
        "parameters",
        "group_weighting",
        "fit_role",
        "prob_cal_gate",
        "selection_artifact_sha256",
        "outcome_artifact_sha256",
        "research_only",
        "probability_claim_allowed",
        "artifact_sha256",
    }
)
_PREDICTION_ARTIFACT_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_version",
        "conditional_target_semantics",
        "roles",
        "calibrator_sha256",
        "feature_artifact_sha256",
        "research_only",
        "probability_claim_allowed",
        "predictions",
        "artifact_sha256",
    }
)
_PREDICTION_ROW_FIELDS = frozenset(
    {
        "record_id",
        "role",
        "split_group",
        "top_candidate_id",
        "conditional_top1_probability",
        "applicable",
        "applicability_reasons",
        "not_retrieval_probability",
        "not_end_to_end_probability",
    }
)


class NMRCalibrationV5Error(ValueError):
    """Raised when a v5 artifact violates its frozen contract."""


class NMRCalibrationV5Blocked(NMRCalibrationV5Error):
    """Raised when a statistical or protocol gate fails closed."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(str(reason) for reason in reasons)
        super().__init__("calibration blocked: " + "; ".join(self.reasons))


def canonical_json_dumps(value: Any) -> str:
    """Return deterministic strict JSON."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Hash a value's canonical JSON representation."""

    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def _artifact(core: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(core)
    return {**value, "artifact_sha256": canonical_sha256(value)}


def _validate_artifact_hash(value: Mapping[str, Any], context: str) -> None:
    artifact_hash = value.get("artifact_sha256")
    if not isinstance(artifact_hash, str) or not _HASH_RE.fullmatch(artifact_hash):
        raise NMRCalibrationV5Error(f"{context}.artifact_sha256 is invalid")
    core = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if canonical_sha256(core) != artifact_hash:
        raise NMRCalibrationV5Error(f"{context} artifact hash mismatch")


def _strict_fields(
    value: Mapping[str, Any], expected: frozenset[str], context: str
) -> None:
    if set(value) != expected:
        difference = sorted(set(value) ^ expected)
        raise NMRCalibrationV5Error(
            f"{context} field allowlist mismatch: {difference}"
        )


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NMRCalibrationV5Error(f"{field} must be a non-empty string")
    return value


def _finite(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not math.isfinite(float(value))
    ):
        raise NMRCalibrationV5Error(f"{field} must be finite")
    return float(value)


def _nonnegative(value: Any, field: str) -> float:
    result = _finite(value, field)
    if result < 0.0:
        raise NMRCalibrationV5Error(f"{field} must be non-negative")
    return result


def _unit_interval(value: Any, field: str) -> float:
    result = _finite(value, field)
    if not 0.0 <= result <= 1.0:
        raise NMRCalibrationV5Error(f"{field} must be in [0, 1]")
    return result


def _role(value: Any, field: str) -> str:
    result = _text(value, field)
    if result not in ROLES:
        raise NMRCalibrationV5Error(f"{field} must be one of {ROLES}")
    return result


def _boolean(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise NMRCalibrationV5Error(f"{field} must be boolean")
    return value


def _validate_feature_artifact(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    _validate_artifact_hash(value, "feature artifact")
    _strict_fields(value, _FEATURE_ARTIFACT_FIELDS, "feature artifact")
    if value.get("schema_version") != FEATURE_SCHEMA_VERSION:
        raise NMRCalibrationV5Error("unsupported feature artifact schema")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise NMRCalibrationV5Error("feature artifact protocol changed")
    if value.get("outcome_free") is not True:
        raise NMRCalibrationV5Error("feature artifact must be outcome-free")
    if value.get("target_semantics") != CONDITIONAL_TARGET:
        raise NMRCalibrationV5Error("feature target semantics changed")
    if value.get("feature_names") != list(FEATURE_NAMES):
        raise NMRCalibrationV5Error("feature specification changed")
    _text(value.get("model_release_id"), "feature model_release_id")
    _text(
        value.get("candidate_generator_release_id"),
        "feature candidate_generator_release_id",
    )
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise NMRCalibrationV5Error("feature artifact rows must be non-empty")
    seen_records: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise NMRCalibrationV5Error(f"feature rows[{index}] must be an object")
        _strict_fields(row, _FEATURE_ROW_FIELDS, f"feature rows[{index}]")
        if row.get("schema_version") != FEATURE_SCHEMA_VERSION:
            raise NMRCalibrationV5Error(f"feature rows[{index}] schema changed")
        record_id = _text(row.get("record_id"), f"feature rows[{index}].record_id")
        if record_id in seen_records:
            raise NMRCalibrationV5Error(f"duplicate feature record_id: {record_id}")
        seen_records.add(record_id)
        _role(row.get("role"), f"feature rows[{index}].role")
        _text(row.get("split_group"), f"feature rows[{index}].split_group")
        fingerprint = row.get("spectrum_fingerprint_sha256")
        if not isinstance(fingerprint, str) or not _HASH_RE.fullmatch(fingerprint):
            raise NMRCalibrationV5Error(
                f"feature rows[{index}].spectrum_fingerprint_sha256 is invalid"
            )
        if row.get("model_release_id") != value.get("model_release_id"):
            raise NMRCalibrationV5Error(
                f"feature rows[{index}] model release binding mismatch"
            )
        if row.get("candidate_generator_release_id") != value.get(
            "candidate_generator_release_id"
        ):
            raise NMRCalibrationV5Error(
                f"feature rows[{index}] candidate-generator binding mismatch"
            )
        _text(
            row.get("ranking_logit_semantics"),
            f"feature rows[{index}].ranking_logit_semantics",
        )
        pool_hash = row.get("candidate_pool_sha256")
        if not isinstance(pool_hash, str) or not _HASH_RE.fullmatch(pool_hash):
            raise NMRCalibrationV5Error(
                f"feature rows[{index}].candidate_pool_sha256 is invalid"
            )
        candidate_ids = row.get("candidate_ids")
        if (
            not isinstance(candidate_ids, list)
            or len(candidate_ids) < 2
            or any(not isinstance(item, str) or not item for item in candidate_ids)
            or candidate_ids != sorted(set(candidate_ids))
            or row.get("top_candidate_id") not in candidate_ids
        ):
            raise NMRCalibrationV5Error(
                f"feature rows[{index}] candidate identities are invalid"
            )
        qc_pass = _boolean(row.get("qc_pass"), f"feature rows[{index}].qc_pass")
        in_domain = _boolean(
            row.get("in_domain"), f"feature rows[{index}].in_domain"
        )
        reasons = row.get("applicability_reasons")
        if (
            not isinstance(reasons, list)
            or any(not isinstance(item, str) or not item for item in reasons)
            or reasons != sorted(set(reasons))
            or bool(qc_pass and in_domain) == bool(reasons)
        ):
            raise NMRCalibrationV5Error(
                f"feature rows[{index}] applicability state is invalid"
            )
        features = np.asarray(row.get("features"), dtype=float)
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise NMRCalibrationV5Error(
                f"feature rows[{index}] feature vector is invalid"
            )
    return rows


def build_probability_features_v5(
    rankings: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the seven outcome-free features from the minimal v5 contract.

    Candidate list order is ignored.  The explicit unique ``rank`` values
    define the frozen ranker order, while pool aggregates are symmetric in all
    candidate logits.
    """

    if not rankings:
        raise NMRCalibrationV5Error("rankings must be non-empty")
    built_rows: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    for row_index, source in enumerate(rankings):
        if not isinstance(source, Mapping):
            raise NMRCalibrationV5Error(f"rankings[{row_index}] must be an object")
        _strict_fields(source, _FEATURE_INPUT_FIELDS, f"rankings[{row_index}]")
        record_id = _text(source["record_id"], f"rankings[{row_index}].record_id")
        if record_id in seen_records:
            raise NMRCalibrationV5Error(f"duplicate record_id: {record_id}")
        seen_records.add(record_id)
        role = _role(source["role"], f"rankings[{row_index}].role")
        split_group = _text(
            source["split_group"], f"rankings[{row_index}].split_group"
        )
        fingerprint = source["spectrum_fingerprint_sha256"]
        if not isinstance(fingerprint, str) or not _HASH_RE.fullmatch(fingerprint):
            raise NMRCalibrationV5Error(
                f"rankings[{row_index}].spectrum_fingerprint_sha256 is invalid"
            )
        model_release_id = _text(
            source["model_release_id"], f"rankings[{row_index}].model_release_id"
        )
        candidate_generator_release_id = _text(
            source["candidate_generator_release_id"],
            f"rankings[{row_index}].candidate_generator_release_id",
        )
        logit_semantics = _text(
            source["ranking_logit_semantics"],
            f"rankings[{row_index}].ranking_logit_semantics",
        )
        reference_top = _text(
            source["reference_top_candidate_id"],
            f"rankings[{row_index}].reference_top_candidate_id",
        )
        qc_pass = _boolean(source["qc_pass"], f"rankings[{row_index}].qc_pass")
        in_domain = _boolean(
            source["in_domain"], f"rankings[{row_index}].in_domain"
        )
        raw_reasons = source["applicability_reasons"]
        if not isinstance(raw_reasons, list) or any(
            not isinstance(item, str) or not item for item in raw_reasons
        ):
            raise NMRCalibrationV5Error(
                f"rankings[{row_index}].applicability_reasons must be strings"
            )
        reasons = sorted(set(raw_reasons))
        if bool(qc_pass and in_domain) == bool(reasons):
            raise NMRCalibrationV5Error(
                f"rankings[{row_index}] applicability reasons/status disagree"
            )

        raw_candidates = source["candidates"]
        if not isinstance(raw_candidates, list) or len(raw_candidates) < 2:
            raise NMRCalibrationV5Error(
                f"rankings[{row_index}].candidates requires at least two candidates"
            )
        candidates: list[dict[str, Any]] = []
        candidate_ids: set[str] = set()
        ranks: set[int] = set()
        for candidate_index, raw_candidate in enumerate(raw_candidates):
            context = f"rankings[{row_index}].candidates[{candidate_index}]"
            if not isinstance(raw_candidate, Mapping):
                raise NMRCalibrationV5Error(f"{context} must be an object")
            _strict_fields(raw_candidate, _CANDIDATE_FIELDS, context)
            candidate_id = _text(raw_candidate["candidate_id"], f"{context}.candidate_id")
            rank_value = raw_candidate["rank"]
            if isinstance(rank_value, bool) or not isinstance(rank_value, int):
                raise NMRCalibrationV5Error(f"{context}.rank must be an integer")
            if candidate_id in candidate_ids or rank_value in ranks:
                raise NMRCalibrationV5Error(
                    f"{context} has a duplicate candidate_id or rank"
                )
            candidate_ids.add(candidate_id)
            ranks.add(rank_value)
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "rank": rank_value,
                    "ranking_logit": _finite(
                        raw_candidate["ranking_logit"], f"{context}.ranking_logit"
                    ),
                    "matched_mae_ppm": _nonnegative(
                        raw_candidate["matched_mae_ppm"],
                        f"{context}.matched_mae_ppm",
                    ),
                    "matched_max_abs_error_ppm": _nonnegative(
                        raw_candidate["matched_max_abs_error_ppm"],
                        f"{context}.matched_max_abs_error_ppm",
                    ),
                    "bidirectional_coverage": _unit_interval(
                        raw_candidate["bidirectional_coverage"],
                        f"{context}.bidirectional_coverage",
                    ),
                }
            )
        expected_ranks = set(range(1, len(candidates) + 1))
        if ranks != expected_ranks:
            raise NMRCalibrationV5Error(
                f"rankings[{row_index}] candidate ranks must be 1..K"
            )
        if reference_top not in candidate_ids:
            raise NMRCalibrationV5Error(
                f"rankings[{row_index}].reference_top_candidate_id is absent"
            )
        ranked = sorted(candidates, key=lambda item: int(item["rank"]))
        if any(
            float(right["ranking_logit"]) > float(left["ranking_logit"]) + 1e-12
            for left, right in zip(ranked, ranked[1:])
        ):
            raise NMRCalibrationV5Error(
                f"rankings[{row_index}] ranks disagree with ranking logits"
            )
        top = ranked[0]
        logits = np.asarray(
            [float(candidate["ranking_logit"]) for candidate in candidates],
            dtype=float,
        )
        top_logit = float(top["ranking_logit"])
        other_logits = np.asarray(
            [
                float(candidate["ranking_logit"])
                for candidate in candidates
                if int(candidate["rank"]) != 1
            ],
            dtype=float,
        )
        pool_log_odds = top_logit - float(logsumexp(other_logits))
        shifted = logits - float(np.max(logits))
        probabilities = np.exp(shifted)
        probabilities /= float(np.sum(probabilities))
        entropy = -float(
            np.sum(probabilities * np.log(np.clip(probabilities, 1e-15, 1.0)))
        )
        normalized_entropy = entropy / math.log(len(candidates))
        features = [
            pool_log_odds,
            math.log(len(candidates)),
            normalized_entropy,
            math.log1p(float(top["matched_mae_ppm"])),
            math.log1p(float(top["matched_max_abs_error_ppm"])),
            float(top["bidirectional_coverage"]),
            float(top["candidate_id"] == reference_top),
        ]
        if not all(math.isfinite(value) for value in features):
            raise NMRCalibrationV5Error(
                f"rankings[{row_index}] produced non-finite features"
            )
        pool_payload = sorted(
            (
                {
                    "candidate_id": candidate["candidate_id"],
                    "rank": candidate["rank"],
                    "ranking_logit": candidate["ranking_logit"],
                }
                for candidate in candidates
            ),
            key=lambda item: str(item["candidate_id"]),
        )
        built_rows.append(
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "record_id": record_id,
                "role": role,
                "split_group": split_group,
                "spectrum_fingerprint_sha256": fingerprint,
                "model_release_id": model_release_id,
                "candidate_generator_release_id": candidate_generator_release_id,
                "ranking_logit_semantics": logit_semantics,
                "candidate_pool_sha256": canonical_sha256(pool_payload),
                "candidate_ids": sorted(candidate_ids),
                "top_candidate_id": top["candidate_id"],
                "qc_pass": qc_pass,
                "in_domain": in_domain,
                "applicability_reasons": reasons,
                "features": features,
            }
        )
    built_rows.sort(key=lambda item: str(item["record_id"]))
    model_releases = sorted({str(row["model_release_id"]) for row in built_rows})
    if len(model_releases) != 1:
        raise NMRCalibrationV5Error("one feature artifact must bind one model release")
    generator_releases = sorted(
        {str(row["candidate_generator_release_id"]) for row in built_rows}
    )
    if len(generator_releases) != 1:
        raise NMRCalibrationV5Error(
            "one feature artifact must bind one candidate-generator release"
        )
    logit_semantics = {str(row["ranking_logit_semantics"]) for row in built_rows}
    if len(logit_semantics) != 1:
        raise NMRCalibrationV5Error(
            "one feature artifact must bind one ranking-logit semantics"
        )
    return _artifact(
        {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "outcome_free": True,
            "target_semantics": CONDITIONAL_TARGET,
            "feature_names": list(FEATURE_NAMES),
            "model_release_id": model_releases[0],
            "candidate_generator_release_id": generator_releases[0],
            "rows": built_rows,
        }
    )


def join_probability_outcomes_v5(
    feature_artifact: Mapping[str, Any],
    outcomes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Join sealed outcomes after outcome-free feature materialisation."""

    feature_rows = _validate_feature_artifact(feature_artifact)
    by_record = {str(row["record_id"]): row for row in feature_rows}
    if not outcomes:
        raise NMRCalibrationV5Error("outcomes must be non-empty")
    outcome_roles = {
        _role(outcome.get("role"), f"outcomes[{index}].role")
        for index, outcome in enumerate(outcomes)
        if isinstance(outcome, Mapping)
    }
    if len(outcome_roles) != 1 or len(outcome_roles) != len(
        {outcome.get("role") for outcome in outcomes if isinstance(outcome, Mapping)}
    ):
        raise NMRCalibrationV5Error("one outcome artifact must contain one role")
    artifact_role = next(iter(outcome_roles))
    role_feature_rows = [row for row in feature_rows if row["role"] == artifact_role]
    if len(outcomes) != len(role_feature_rows):
        raise NMRCalibrationV5Error(
            "outcomes must cover every feature row for exactly one role"
        )
    joined: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, Mapping):
            raise NMRCalibrationV5Error(f"outcomes[{index}] must be an object")
        _strict_fields(outcome, _OUTCOME_INPUT_FIELDS, f"outcomes[{index}]")
        record_id = _text(outcome["record_id"], f"outcomes[{index}].record_id")
        if record_id in seen or record_id not in by_record:
            raise NMRCalibrationV5Error(
                f"outcomes[{index}] is duplicate or has no feature row"
            )
        seen.add(record_id)
        feature = by_record[record_id]
        role = _role(outcome["role"], f"outcomes[{index}].role")
        group = _text(
            outcome["split_group"], f"outcomes[{index}].split_group"
        )
        if role != feature["role"] or group != feature["split_group"]:
            raise NMRCalibrationV5Error(
                f"outcomes[{index}] role/group binding mismatch"
            )
        truth = _text(
            outcome["truth_candidate_id"],
            f"outcomes[{index}].truth_candidate_id",
        )
        retrieved = _boolean(
            outcome["truth_retrieved"], f"outcomes[{index}].truth_retrieved"
        )
        actual_retrieved = truth in feature["candidate_ids"]
        if retrieved != actual_retrieved:
            raise NMRCalibrationV5Error(
                f"outcomes[{index}].truth_retrieved disagrees with candidate pool"
            )
        conditional = truth == feature["top_candidate_id"] if retrieved else None
        joined.append(
            {
                "record_id": record_id,
                "role": role,
                "split_group": group,
                "top_candidate_id": feature["top_candidate_id"],
                "truth_candidate_id": truth,
                "truth_retrieved": retrieved,
                "conditional_top1_correct": conditional,
                "end_to_end_correct": bool(retrieved and conditional),
                "qc_pass": feature["qc_pass"],
                "in_domain": feature["in_domain"],
                "features": list(feature["features"]),
            }
        )
    joined.sort(key=lambda item: str(item["record_id"]))
    return _artifact(
        {
            "schema_version": OUTCOME_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "feature_artifact_sha256": feature_artifact["artifact_sha256"],
            "model_release_id": feature_artifact["model_release_id"],
            "candidate_generator_release_id": feature_artifact[
                "candidate_generator_release_id"
            ],
            "role": artifact_role,
            "conditional_target_semantics": CONDITIONAL_TARGET,
            "retrieval_and_end_to_end_are_separate": True,
            "rows": joined,
        }
    )


def _validate_outcome_artifact(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    _validate_artifact_hash(value, "outcome artifact")
    _strict_fields(value, _OUTCOME_ARTIFACT_FIELDS, "outcome artifact")
    if value.get("schema_version") != OUTCOME_SCHEMA_VERSION:
        raise NMRCalibrationV5Error("unsupported outcome artifact schema")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise NMRCalibrationV5Error("outcome artifact protocol changed")
    if value.get("conditional_target_semantics") != CONDITIONAL_TARGET:
        raise NMRCalibrationV5Error("conditional target semantics changed")
    if value.get("retrieval_and_end_to_end_are_separate") is not True:
        raise NMRCalibrationV5Error("outcome target separation changed")
    feature_hash = value.get("feature_artifact_sha256")
    if not isinstance(feature_hash, str) or not _HASH_RE.fullmatch(feature_hash):
        raise NMRCalibrationV5Error("outcome feature artifact binding is invalid")
    _text(value.get("model_release_id"), "outcome model_release_id")
    _text(
        value.get("candidate_generator_release_id"),
        "outcome candidate_generator_release_id",
    )
    artifact_role = _role(value.get("role"), "outcome artifact role")
    rows = value.get("rows")
    if not isinstance(rows, list) or not rows:
        raise NMRCalibrationV5Error("outcome rows must be non-empty")
    seen_records: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise NMRCalibrationV5Error(f"outcome rows[{index}] must be an object")
        _strict_fields(row, _OUTCOME_ROW_FIELDS, f"outcome rows[{index}]")
        if _role(row.get("role"), f"outcome rows[{index}].role") != artifact_role:
            raise NMRCalibrationV5Error(
                f"outcome rows[{index}] role binding mismatch"
            )
        record_id = _text(row.get("record_id"), f"outcome rows[{index}].record_id")
        if record_id in seen_records:
            raise NMRCalibrationV5Error(f"duplicate outcome record_id: {record_id}")
        seen_records.add(record_id)
        _text(row.get("split_group"), f"outcome rows[{index}].split_group")
        top_candidate_id = _text(
            row.get("top_candidate_id"),
            f"outcome rows[{index}].top_candidate_id",
        )
        truth_candidate_id = _text(
            row.get("truth_candidate_id"),
            f"outcome rows[{index}].truth_candidate_id",
        )
        retrieved = _boolean(
            row.get("truth_retrieved"),
            f"outcome rows[{index}].truth_retrieved",
        )
        conditional = row.get("conditional_top1_correct")
        expected_conditional = truth_candidate_id == top_candidate_id
        if not retrieved and conditional is not None:
            raise NMRCalibrationV5Error(
                f"outcome rows[{index}] unretrieved truth must have null conditional"
            )
        if retrieved and (
            type(conditional) is not bool or conditional != expected_conditional
        ):
            raise NMRCalibrationV5Error(
                f"outcome rows[{index}] conditional correctness is inconsistent"
            )
        for field in (
            "truth_retrieved",
            "end_to_end_correct",
            "qc_pass",
            "in_domain",
        ):
            _boolean(row.get(field), f"outcome rows[{index}].{field}")
        expected_end_to_end = bool(retrieved and conditional)
        if row.get("end_to_end_correct") != expected_end_to_end:
            raise NMRCalibrationV5Error(
                f"outcome rows[{index}] end-to-end correctness is inconsistent"
            )
        features = np.asarray(row.get("features"), dtype=float)
        if features.shape != (len(FEATURE_NAMES),) or not np.isfinite(features).all():
            raise NMRCalibrationV5Error(
                f"outcome rows[{index}] feature vector is invalid"
            )
    return rows


def role_data_gate_v5(
    outcome_artifact: Mapping[str, Any] | None, role: str
) -> dict[str, Any]:
    """Describe and enforce independent-group sample-size gates."""

    requested_role = _role(role, "role")
    if outcome_artifact is None:
        rows: list[dict[str, Any]] = []
    else:
        rows = _validate_outcome_artifact(outcome_artifact)
        if outcome_artifact.get("role") != requested_role:
            raise NMRCalibrationV5Error(
                f"{requested_role} gate received {outcome_artifact.get('role')} outcomes"
            )
    groups: dict[str, list[bool]] = defaultdict(list)
    unretrieved_rows = 0
    excluded_applicability_rows = 0
    for row in rows:
        conditional = row["conditional_top1_correct"]
        if conditional is None:
            unretrieved_rows += 1
            continue
        if requested_role != "risk_cal" and not (
            row["qc_pass"] and row["in_domain"]
        ):
            excluded_applicability_rows += 1
            continue
        groups[str(row["split_group"])].append(bool(conditional))
    error_groups = sum(any(not label for label in labels) for labels in groups.values())
    correct_groups = sum(all(labels) for labels in groups.values())
    mixed_groups = sum(
        any(labels) and any(not label for label in labels) for labels in groups.values()
    )
    all_role_groups = {str(row["split_group"]) for row in rows}
    reasons: list[str] = []
    if not rows:
        reasons.append(f"missing_{requested_role}_rows")
    if requested_role in {"dev", "prob_cal"}:
        if error_groups < MIN_ERROR_GROUPS:
            reasons.append(
                f"{requested_role}_error_groups_{error_groups}_below_{MIN_ERROR_GROUPS}"
            )
        if correct_groups < MIN_CORRECT_GROUPS:
            reasons.append(
                f"{requested_role}_correct_groups_{correct_groups}_below_"
                f"{MIN_CORRECT_GROUPS}"
            )
    if requested_role == "risk_cal" and len(all_role_groups) < MIN_RISK_GROUPS:
        reasons.append(
            f"risk_cal_groups_{len(all_role_groups)}_below_{MIN_RISK_GROUPS}"
        )
    return {
        "role": requested_role,
        "status": "passed" if not reasons else "blocked",
        "row_count": len(rows),
        "group_count": len(all_role_groups),
        "conditional_group_count": len(groups),
        "error_groups": int(error_groups),
        "correct_groups": int(correct_groups),
        "mixed_groups": int(mixed_groups),
        "unretrieved_rows": unretrieved_rows,
        "excluded_applicability_rows": excluded_applicability_rows,
        "minimums": {
            "error_groups": (
                MIN_ERROR_GROUPS
                if requested_role in {"dev", "prob_cal"}
                else None
            ),
            "correct_groups": (
                MIN_CORRECT_GROUPS
                if requested_role in {"dev", "prob_cal"}
                else None
            ),
            "groups": MIN_RISK_GROUPS if requested_role == "risk_cal" else None,
        },
        "blocking_reasons": reasons,
    }


def external_validation_gate_v5(
    outcome_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the probability-validation event gate to external test groups."""

    gate = role_data_gate_v5(outcome_artifact, "external_test")
    reasons = list(gate["blocking_reasons"])
    if gate["error_groups"] < MIN_ERROR_GROUPS:
        reasons.append(
            "external_test_error_groups_"
            f"{gate['error_groups']}_below_{MIN_ERROR_GROUPS}"
        )
    if gate["correct_groups"] < MIN_CORRECT_GROUPS:
        reasons.append(
            "external_test_correct_groups_"
            f"{gate['correct_groups']}_below_{MIN_CORRECT_GROUPS}"
        )
    return {
        **gate,
        "status": "passed" if not reasons else "blocked",
        "minimums": {
            "error_groups": MIN_ERROR_GROUPS,
            "correct_groups": MIN_CORRECT_GROUPS,
            "groups": None,
        },
        "blocking_reasons": reasons,
    }


def _conditional_rows(
    outcome_artifact: Mapping[str, Any], role: str
) -> list[dict[str, Any]]:
    if outcome_artifact.get("role") != role:
        raise NMRCalibrationV5Error(f"{role} operation received another role")
    rows = [
        row
        for row in _validate_outcome_artifact(outcome_artifact)
        if row["conditional_top1_correct"] is not None
        and row["qc_pass"]
        and row["in_domain"]
    ]
    if not rows:
        raise NMRCalibrationV5Blocked([f"no_applicable_{role}_conditional_rows"])
    return rows


def group_weights_v5(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Give every group equal total weight and every row within it equal weight."""

    if not rows:
        raise NMRCalibrationV5Error("group weights require rows")
    counts: dict[str, int] = defaultdict(int)
    for index, row in enumerate(rows):
        group = _text(row.get("split_group"), f"rows[{index}].split_group")
        counts[group] += 1
    group_count = len(counts)
    weights = np.asarray(
        [1.0 / (group_count * counts[str(row["split_group"])]) for row in rows],
        dtype=float,
    )
    return weights


def _matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray([row["features"] for row in rows], dtype=float)
    labels = np.asarray(
        [int(bool(row["conditional_top1_correct"])) for row in rows], dtype=int
    )
    if matrix.ndim != 2 or matrix.shape[1] != len(FEATURE_NAMES):
        raise NMRCalibrationV5Error("feature matrix shape changed")
    if not np.isfinite(matrix).all() or set(labels.tolist()) != {0, 1}:
        raise NMRCalibrationV5Blocked(["non_finite_features_or_single_class"])
    return matrix, labels


def _normalization(
    matrix: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    center = np.average(matrix, axis=0, weights=weights)
    variance = np.average((matrix - center) ** 2, axis=0, weights=weights)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    return center, scale


def _linearly_separable(matrix: np.ndarray, labels: np.ndarray) -> bool:
    signed = np.where(labels == 1, 1.0, -1.0)
    design = np.column_stack((np.ones(len(matrix)), matrix))
    result = linprog(
        np.zeros(design.shape[1]),
        A_ub=-(signed[:, None] * design),
        b_ub=-np.ones(len(matrix)),
        bounds=[(None, None)] * design.shape[1],
        method="highs",
    )
    return bool(result.success)


def _fit_ridge(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    regularization_lambda: float,
) -> tuple[float, np.ndarray]:
    if regularization_lambda <= 0.0 or not math.isfinite(regularization_lambda):
        raise NMRCalibrationV5Error("regularization lambda must be positive")
    prevalence = float(np.dot(weights, labels) / np.sum(weights))
    prevalence = min(max(prevalence, 1e-9), 1.0 - 1e-9)
    initial = np.zeros(matrix.shape[1] + 1, dtype=float)
    initial[0] = math.log(prevalence / (1.0 - prevalence))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = parameters[0] + matrix @ parameters[1:]
        loss = np.logaddexp(0.0, linear) - labels * linear
        value = float(np.dot(weights, loss))
        value += 0.5 * regularization_lambda * float(
            np.dot(parameters[1:], parameters[1:])
        )
        residual = weights * (expit(linear) - labels)
        gradient = np.empty_like(parameters)
        gradient[0] = float(np.sum(residual))
        gradient[1:] = matrix.T @ residual + regularization_lambda * parameters[1:]
        return value, gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise NMRCalibrationV5Blocked(["ridge_optimizer_failed"])
    return float(result.x[0]), np.asarray(result.x[1:], dtype=float)


def _group_fold_assignments(
    rows: Sequence[Mapping[str, Any]], *, n_splits: int, seed: int
) -> dict[str, int]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split_group"])].append(
            bool(row["conditional_top1_correct"])
        )
    strata = {
        group: "error" if any(not value for value in values) else "correct"
        for group, values in grouped.items()
    }
    assignments: dict[str, int] = {}
    fold_strata = [{"error": 0, "correct": 0} for _ in range(n_splits)]
    fold_rows = [0] * n_splits
    for stratum in ("error", "correct"):
        groups = [group for group, value in strata.items() if value == stratum]
        groups.sort(
            key=lambda group: (
                -len(grouped[group]),
                hashlib.sha256(f"{seed}:{group}".encode()).hexdigest(),
            )
        )
        for group in groups:
            fold = min(
                range(n_splits),
                key=lambda index: (
                    fold_strata[index][stratum],
                    fold_rows[index],
                    index,
                ),
            )
            assignments[group] = fold
            fold_strata[fold][stratum] += 1
            fold_rows[fold] += len(grouped[group])
    return assignments


def _weighted_log_loss(
    probabilities: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float:
    clipped = np.clip(probabilities, 1e-15, 1.0 - 1e-15)
    losses = -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    return float(np.dot(weights, losses) / np.sum(weights))


def select_ridge_lambda_dev(
    outcome_artifact: Mapping[str, Any],
    *,
    lambda_grid: Sequence[float] = LAMBDA_GRID,
    n_splits: int = CV_FOLDS,
    seed: int = 20260801,
) -> dict[str, Any]:
    """Select only ridge strength on dev using deterministic group CV + 1SE."""

    if n_splits != CV_FOLDS:
        raise NMRCalibrationV5Error(f"n_splits is frozen at {CV_FOLDS}")
    values = tuple(float(value) for value in lambda_grid)
    if values != LAMBDA_GRID:
        raise NMRCalibrationV5Error("lambda grid changed from preregistration")
    gate = role_data_gate_v5(outcome_artifact, "dev")
    if gate["status"] != "passed":
        raise NMRCalibrationV5Blocked(gate["blocking_reasons"])
    rows = _conditional_rows(outcome_artifact, "dev")
    matrix, labels = _matrix(rows)
    full_weights = group_weights_v5(rows)
    full_center, full_scale = _normalization(matrix, full_weights)
    standardized = (matrix - full_center) / full_scale
    if _linearly_separable(standardized, labels):
        raise NMRCalibrationV5Blocked(["dev_complete_linear_separation"])
    assignments = _group_fold_assignments(rows, n_splits=n_splits, seed=seed)
    fold_audit: list[dict[str, Any]] = []
    for fold in range(n_splits):
        validation = [
            row for row in rows if assignments[str(row["split_group"])] == fold
        ]
        validation_groups: dict[str, list[bool]] = defaultdict(list)
        for row in validation:
            validation_groups[str(row["split_group"])].append(
                bool(row["conditional_top1_correct"])
            )
        error_groups = sum(
            any(not label for label in values)
            for values in validation_groups.values()
        )
        correct_groups = sum(all(values) for values in validation_groups.values())
        reasons: list[str] = []
        if error_groups < MIN_ERROR_GROUPS_PER_FOLD:
            reasons.append(
                f"fold_{fold}_error_groups_{error_groups}_below_"
                f"{MIN_ERROR_GROUPS_PER_FOLD}"
            )
        if correct_groups < 1:
            reasons.append(f"fold_{fold}_has_no_correct_group")
        if len({bool(row["conditional_top1_correct"]) for row in validation}) != 2:
            reasons.append(f"fold_{fold}_is_single_class")
        fold_audit.append(
            {
                "fold": fold,
                "groups": len(validation_groups),
                "rows": len(validation),
                "error_groups": int(error_groups),
                "correct_groups": int(correct_groups),
                "blocking_reasons": reasons,
            }
        )
    fold_reasons = [
        reason for audit in fold_audit for reason in audit["blocking_reasons"]
    ]
    if fold_reasons:
        raise NMRCalibrationV5Blocked(fold_reasons)

    losses: dict[float, list[float]] = {value: [] for value in values}
    for fold in range(n_splits):
        train_rows = [
            row for row in rows if assignments[str(row["split_group"])] != fold
        ]
        validation_rows = [
            row for row in rows if assignments[str(row["split_group"])] == fold
        ]
        train_matrix, train_labels = _matrix(train_rows)
        validation_matrix, validation_labels = _matrix(validation_rows)
        train_weights = group_weights_v5(train_rows)
        validation_weights = group_weights_v5(validation_rows)
        center, scale = _normalization(train_matrix, train_weights)
        train_standardized = (train_matrix - center) / scale
        validation_standardized = (validation_matrix - center) / scale
        if _linearly_separable(train_standardized, train_labels):
            raise NMRCalibrationV5Blocked(
                [f"dev_fold_{fold}_training_complete_linear_separation"]
            )
        for regularization_lambda in values:
            intercept, coefficients = _fit_ridge(
                train_standardized,
                train_labels,
                train_weights,
                regularization_lambda,
            )
            predicted = expit(intercept + validation_standardized @ coefficients)
            losses[regularization_lambda].append(
                _weighted_log_loss(
                    predicted,
                    validation_labels,
                    validation_weights,
                )
            )
    candidates: list[dict[str, Any]] = []
    for regularization_lambda in values:
        fold_losses = np.asarray(losses[regularization_lambda], dtype=float)
        candidates.append(
            {
                "lambda": regularization_lambda,
                "fold_group_weighted_log_loss": fold_losses.tolist(),
                "mean_group_weighted_log_loss": float(np.mean(fold_losses)),
                "standard_error": float(
                    np.std(fold_losses, ddof=1) / math.sqrt(len(fold_losses))
                ),
            }
        )
    best = min(candidates, key=lambda item: float(item["mean_group_weighted_log_loss"]))
    one_se_limit = float(best["mean_group_weighted_log_loss"]) + float(
        best["standard_error"]
    )
    eligible = [
        item
        for item in candidates
        if float(item["mean_group_weighted_log_loss"]) <= one_se_limit
    ]
    selected = max(eligible, key=lambda item: float(item["lambda"]))
    return _artifact(
        {
            "schema_version": SELECTION_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "method": "group_weighted_ridge_top1_context_logistic",
            "target_semantics": CONDITIONAL_TARGET,
            "split_roles_consumed": ["dev"],
            "feature_names": list(FEATURE_NAMES),
            "lambda_grid": list(values),
            "selection_rule": "largest_lambda_within_one_standard_error_of_minimum",
            "selected_lambda": float(selected["lambda"]),
            "one_standard_error_limit": one_se_limit,
            "candidates": candidates,
            "normalization": {
                "source_role": "dev",
                "center": full_center.tolist(),
                "scale": full_scale.tolist(),
            },
            "group_weighting": "each_group_total_weight_one_then_normalized",
            "n_splits": n_splits,
            "seed": seed,
            "fold_audit": fold_audit,
            "group_assignments_sha256": canonical_sha256(assignments),
            "dev_gate": gate,
            "model_release_id": outcome_artifact["model_release_id"],
            "candidate_generator_release_id": outcome_artifact[
                "candidate_generator_release_id"
            ],
            "outcome_artifact_sha256": outcome_artifact["artifact_sha256"],
        }
    )


def _validate_selection_artifact(value: Mapping[str, Any]) -> None:
    _validate_artifact_hash(value, "selection artifact")
    _strict_fields(value, _SELECTION_ARTIFACT_FIELDS, "selection artifact")
    if value.get("schema_version") != SELECTION_SCHEMA_VERSION:
        raise NMRCalibrationV5Error("unsupported selection artifact schema")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise NMRCalibrationV5Error("selection protocol changed")
    if value.get("method") != "group_weighted_ridge_top1_context_logistic":
        raise NMRCalibrationV5Error("selection method changed")
    if value.get("target_semantics") != CONDITIONAL_TARGET:
        raise NMRCalibrationV5Error("selection target semantics changed")
    if value.get("split_roles_consumed") != ["dev"]:
        raise NMRCalibrationV5Error("selection was not dev-only")
    if value.get("feature_names") != list(FEATURE_NAMES):
        raise NMRCalibrationV5Error("selected feature specification changed")
    if value.get("lambda_grid") != list(LAMBDA_GRID):
        raise NMRCalibrationV5Error("selected lambda grid changed")
    if value.get("selection_rule") != (
        "largest_lambda_within_one_standard_error_of_minimum"
    ):
        raise NMRCalibrationV5Error("selection rule changed")
    selected = _finite(value.get("selected_lambda"), "selected_lambda")
    if selected not in LAMBDA_GRID:
        raise NMRCalibrationV5Error("selected lambda is outside the frozen grid")
    if value.get("n_splits") != CV_FOLDS:
        raise NMRCalibrationV5Error("selection fold count changed")
    _text(value.get("model_release_id"), "selection model_release_id")
    _text(
        value.get("candidate_generator_release_id"),
        "selection candidate_generator_release_id",
    )
    normalization = value.get("normalization")
    if not isinstance(normalization, Mapping):
        raise NMRCalibrationV5Error("selection normalization is missing")
    if normalization.get("source_role") != "dev":
        raise NMRCalibrationV5Error("selection normalization is not dev-only")
    center = np.asarray(normalization.get("center"), dtype=float)
    scale = np.asarray(normalization.get("scale"), dtype=float)
    expected = (len(FEATURE_NAMES),)
    if (
        center.shape != expected
        or scale.shape != expected
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise NMRCalibrationV5Error("selection normalization is invalid")


def fit_top1_calibrator_v5(
    outcome_artifact: Mapping[str, Any],
    selection_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit the sole preregistered model on ``prob_cal`` only."""

    _validate_selection_artifact(selection_artifact)
    if selection_artifact.get("model_release_id") != outcome_artifact.get(
        "model_release_id"
    ):
        raise NMRCalibrationV5Error("selection/prob_cal model release mismatch")
    if selection_artifact.get("candidate_generator_release_id") != (
        outcome_artifact.get("candidate_generator_release_id")
    ):
        raise NMRCalibrationV5Error(
            "selection/prob_cal candidate-generator release mismatch"
        )
    gate = role_data_gate_v5(outcome_artifact, "prob_cal")
    if gate["status"] != "passed":
        raise NMRCalibrationV5Blocked(gate["blocking_reasons"])
    rows = _conditional_rows(outcome_artifact, "prob_cal")
    matrix, labels = _matrix(rows)
    normalization = selection_artifact.get("normalization")
    if not isinstance(normalization, Mapping):
        raise NMRCalibrationV5Error("selection normalization is missing")
    center = np.asarray(normalization.get("center"), dtype=float)
    scale = np.asarray(normalization.get("scale"), dtype=float)
    if center.shape != (len(FEATURE_NAMES),) or scale.shape != center.shape:
        raise NMRCalibrationV5Error("selection normalization shape changed")
    standardized = (matrix - center) / scale
    if _linearly_separable(standardized, labels):
        raise NMRCalibrationV5Blocked(["prob_cal_complete_linear_separation"])
    weights = group_weights_v5(rows)
    regularization_lambda = _finite(
        selection_artifact.get("selected_lambda"), "selected_lambda"
    )
    intercept, coefficients = _fit_ridge(
        standardized,
        labels,
        weights,
        regularization_lambda,
    )
    model_release_id = _text(
        outcome_artifact.get("model_release_id"), "outcome model_release_id"
    )
    return _artifact(
        {
            "schema_version": CALIBRATOR_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "method": "group_weighted_ridge_top1_context_logistic",
            "conditional_target_semantics": CONDITIONAL_TARGET,
            "not_retrieval_probability": True,
            "not_end_to_end_probability": True,
            "model_release_id": model_release_id,
            "candidate_generator_release_id": outcome_artifact[
                "candidate_generator_release_id"
            ],
            "feature_names": list(FEATURE_NAMES),
            "normalization": {
                "source_role": "dev",
                "center": center.tolist(),
                "scale": scale.tolist(),
            },
            "parameters": {
                "intercept": intercept,
                "coefficients": coefficients.tolist(),
                "lambda": regularization_lambda,
            },
            "group_weighting": "each_group_total_weight_one_then_normalized",
            "fit_role": "prob_cal",
            "prob_cal_gate": gate,
            "selection_artifact_sha256": selection_artifact["artifact_sha256"],
            "outcome_artifact_sha256": outcome_artifact["artifact_sha256"],
            "research_only": True,
            "probability_claim_allowed": False,
        }
    )


def _validate_calibrator_artifact(value: Mapping[str, Any]) -> None:
    _validate_artifact_hash(value, "calibrator")
    _strict_fields(value, _CALIBRATOR_ARTIFACT_FIELDS, "calibrator")
    if value.get("schema_version") != CALIBRATOR_SCHEMA_VERSION:
        raise NMRCalibrationV5Error("unsupported calibrator schema")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise NMRCalibrationV5Error("calibrator protocol changed")
    if value.get("method") != "group_weighted_ridge_top1_context_logistic":
        raise NMRCalibrationV5Error("calibrator method changed")
    if value.get("conditional_target_semantics") != CONDITIONAL_TARGET:
        raise NMRCalibrationV5Error("calibrator target semantics changed")
    _text(value.get("model_release_id"), "calibrator model_release_id")
    _text(
        value.get("candidate_generator_release_id"),
        "calibrator candidate_generator_release_id",
    )
    if (
        value.get("not_retrieval_probability") is not True
        or value.get("not_end_to_end_probability") is not True
        or value.get("fit_role") != "prob_cal"
        or value.get("research_only") is not True
        or value.get("probability_claim_allowed") is not False
        or value.get("feature_names") != list(FEATURE_NAMES)
    ):
        raise NMRCalibrationV5Error("calibrator claim/role boundary changed")


def predict_top1_probability_v5(
    calibrator: Mapping[str, Any],
    feature_artifact: Mapping[str, Any],
    *,
    roles: Sequence[str] = ("risk_cal",),
) -> dict[str, Any]:
    """Emit conditional Top-1 probabilities, never retrieval probabilities."""

    _validate_calibrator_artifact(calibrator)
    requested_roles = tuple(_role(value, "roles") for value in roles)
    if not requested_roles or any(
        value not in {"risk_cal", "external_test"} for value in requested_roles
    ):
        raise NMRCalibrationV5Error(
            "probability emission is limited to risk_cal/external_test"
        )
    feature_rows = _validate_feature_artifact(feature_artifact)
    if calibrator.get("model_release_id") != feature_artifact.get("model_release_id"):
        raise NMRCalibrationV5Error("calibrator/model release binding mismatch")
    if calibrator.get("candidate_generator_release_id") != feature_artifact.get(
        "candidate_generator_release_id"
    ):
        raise NMRCalibrationV5Error(
            "calibrator/candidate-generator release binding mismatch"
        )
    normalization = calibrator.get("normalization")
    parameters = calibrator.get("parameters")
    if not isinstance(normalization, Mapping) or not isinstance(parameters, Mapping):
        raise NMRCalibrationV5Error("calibrator parameters are missing")
    center = np.asarray(normalization.get("center"), dtype=float)
    scale = np.asarray(normalization.get("scale"), dtype=float)
    coefficients = np.asarray(parameters.get("coefficients"), dtype=float)
    intercept = _finite(parameters.get("intercept"), "calibrator intercept")
    expected_shape = (len(FEATURE_NAMES),)
    if (
        center.shape != expected_shape
        or scale.shape != expected_shape
        or coefficients.shape != expected_shape
        or not np.isfinite(center).all()
        or not np.isfinite(scale).all()
        or not np.isfinite(coefficients).all()
        or np.any(scale <= 0.0)
    ):
        raise NMRCalibrationV5Error("calibrator parameter shape/value changed")
    predictions: list[dict[str, Any]] = []
    for row in feature_rows:
        if row["role"] not in requested_roles:
            continue
        applicable = bool(row["qc_pass"] and row["in_domain"])
        probability: float | None = None
        if applicable:
            features = np.asarray(row["features"], dtype=float)
            probability = float(
                expit(intercept + ((features - center) / scale) @ coefficients)
            )
        predictions.append(
            {
                "record_id": row["record_id"],
                "role": row["role"],
                "split_group": row["split_group"],
                "top_candidate_id": row["top_candidate_id"],
                "conditional_top1_probability": probability,
                "applicable": applicable,
                "applicability_reasons": row["applicability_reasons"],
                "not_retrieval_probability": True,
                "not_end_to_end_probability": True,
            }
        )
    if not predictions:
        raise NMRCalibrationV5Blocked(["requested_prediction_roles_are_empty"])
    return _artifact(
        {
            "schema_version": PREDICTION_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "conditional_target_semantics": CONDITIONAL_TARGET,
            "roles": list(requested_roles),
            "calibrator_sha256": calibrator["artifact_sha256"],
            "feature_artifact_sha256": feature_artifact["artifact_sha256"],
            "research_only": True,
            "probability_claim_allowed": False,
            "predictions": predictions,
        }
    )


def _validate_prediction_artifact(value: Mapping[str, Any]) -> list[dict[str, Any]]:
    _validate_artifact_hash(value, "prediction artifact")
    _strict_fields(value, _PREDICTION_ARTIFACT_FIELDS, "prediction artifact")
    if value.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        raise NMRCalibrationV5Error("unsupported prediction artifact schema")
    if value.get("protocol_version") != PROTOCOL_VERSION:
        raise NMRCalibrationV5Error("prediction protocol changed")
    if value.get("conditional_target_semantics") != CONDITIONAL_TARGET:
        raise NMRCalibrationV5Error("prediction target semantics changed")
    if (
        value.get("research_only") is not True
        or value.get("probability_claim_allowed") is not False
    ):
        raise NMRCalibrationV5Error("prediction claim boundary changed")
    roles = value.get("roles")
    if not isinstance(roles, list) or not roles or any(
        role not in {"risk_cal", "external_test"} for role in roles
    ) or len(roles) != len(set(roles)):
        raise NMRCalibrationV5Error("prediction roles are invalid")
    for field in ("calibrator_sha256", "feature_artifact_sha256"):
        binding = value.get(field)
        if not isinstance(binding, str) or not _HASH_RE.fullmatch(binding):
            raise NMRCalibrationV5Error(f"prediction {field} is invalid")
    predictions = value.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise NMRCalibrationV5Error("prediction rows are missing")
    seen_records: set[str] = set()
    for index, row in enumerate(predictions):
        if not isinstance(row, Mapping):
            raise NMRCalibrationV5Error(f"prediction rows[{index}] must be an object")
        _strict_fields(row, _PREDICTION_ROW_FIELDS, f"prediction rows[{index}]")
        if row.get("role") not in roles:
            raise NMRCalibrationV5Error(f"prediction rows[{index}] role is invalid")
        record_id = _text(
            row.get("record_id"), f"prediction rows[{index}].record_id"
        )
        if record_id in seen_records:
            raise NMRCalibrationV5Error(f"duplicate prediction record_id: {record_id}")
        seen_records.add(record_id)
        _text(row.get("split_group"), f"prediction rows[{index}].split_group")
        _text(
            row.get("top_candidate_id"),
            f"prediction rows[{index}].top_candidate_id",
        )
        applicable = _boolean(
            row.get("applicable"), f"prediction rows[{index}].applicable"
        )
        reasons = row.get("applicability_reasons")
        if (
            not isinstance(reasons, list)
            or any(not isinstance(item, str) or not item for item in reasons)
            or reasons != sorted(set(reasons))
            or applicable == bool(reasons)
        ):
            raise NMRCalibrationV5Error(
                f"prediction rows[{index}] applicability state is invalid"
            )
        probability = row.get("conditional_top1_probability")
        if applicable:
            if probability is None:
                raise NMRCalibrationV5Error(
                    f"prediction rows[{index}] applicable row lacks probability"
                )
            _unit_interval(
                probability,
                f"prediction rows[{index}].conditional_top1_probability",
            )
        elif probability is not None:
            raise NMRCalibrationV5Error(
                f"prediction rows[{index}] inapplicable row must have null probability"
            )
        if (
            row.get("not_retrieval_probability") is not True
            or row.get("not_end_to_end_probability") is not True
        ):
            raise NMRCalibrationV5Error(
                f"prediction rows[{index}] probability semantics changed"
            )
    return predictions


def _representative_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> dict[str, Mapping[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split_group"])].append(row)
    representatives: dict[str, Mapping[str, Any]] = {}
    for group, group_rows in grouped.items():
        representatives[group] = min(
            group_rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['record_id']}".encode()
            ).hexdigest(),
        )
    return representatives


def select_abstention_policy_v5(
    prediction_artifact: Mapping[str, Any],
    outcome_artifact: Mapping[str, Any],
    *,
    seed: int = 20260801,
) -> dict[str, Any]:
    """Apply fixed-sequence LTT to independent group representatives.

    Thresholds are tested from most to least selective after outcome-free
    coverage/minimum-count filtering.  The loss is end-to-end error, so a
    retrieval failure remains an error even though the emitted probability is
    explicitly conditional on retrieval.
    """

    predictions = _validate_prediction_artifact(prediction_artifact)
    if prediction_artifact.get("roles") != ["risk_cal"]:
        raise NMRCalibrationV5Error("risk policy requires risk_cal predictions only")
    gate = role_data_gate_v5(outcome_artifact, "risk_cal")
    common = {
        "schema_version": POLICY_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "method": "fixed_sequence_learn_then_test_exact_binomial",
        "risk_target": RISK_TARGET,
        "probability_semantics": CONDITIONAL_TARGET,
        "representative_unit": "one_outcome_free_hash_selected_record_per_group",
        "representative_seed": seed,
        "target_selective_error": TARGET_SELECTIVE_ERROR,
        "failure_probability": RISK_FAILURE_PROBABILITY,
        "minimum_coverage": MIN_RISK_COVERAGE,
        "minimum_accepted_groups": MIN_ACCEPTED_RISK_GROUPS,
        "threshold_grid": list(RISK_THRESHOLDS),
        "risk_cal_gate": gate,
        "prediction_artifact_sha256": prediction_artifact["artifact_sha256"],
        "outcome_artifact_sha256": outcome_artifact["artifact_sha256"],
        "probability_claim_allowed": False,
    }
    if gate["status"] != "passed":
        return _artifact(
            {
                **common,
                "status": "blocked",
                "decision": "manual_review_all",
                "selected_threshold": None,
                "tested_thresholds": [],
                "blocking_reasons": gate["blocking_reasons"],
            }
        )
    outcomes = [
        row
        for row in _validate_outcome_artifact(outcome_artifact)
        if row["role"] == "risk_cal"
    ]
    prediction_by_record = {str(row["record_id"]): row for row in predictions}
    if set(prediction_by_record) != {str(row["record_id"]) for row in outcomes}:
        raise NMRCalibrationV5Error("risk predictions/outcomes do not align")
    representatives = _representative_rows(outcomes, seed=seed)
    representative_records = [
        (outcome, prediction_by_record[str(outcome["record_id"])])
        for outcome in representatives.values()
    ]
    total = len(representative_records)
    eligible_thresholds: list[float] = []
    threshold_counts: dict[float, tuple[int, int]] = {}
    for threshold in sorted(RISK_THRESHOLDS, reverse=True):
        accepted = [
            (outcome, prediction)
            for outcome, prediction in representative_records
            if prediction["conditional_top1_probability"] is not None
            and float(prediction["conditional_top1_probability"]) >= threshold
        ]
        accepted_count = len(accepted)
        errors = sum(not bool(outcome["end_to_end_correct"]) for outcome, _ in accepted)
        threshold_counts[threshold] = (accepted_count, int(errors))
        if (
            accepted_count >= MIN_ACCEPTED_RISK_GROUPS
            and accepted_count / total >= MIN_RISK_COVERAGE
        ):
            eligible_thresholds.append(threshold)
    if not eligible_thresholds:
        return _artifact(
            {
                **common,
                "status": "blocked",
                "decision": "manual_review_all",
                "selected_threshold": None,
                "tested_thresholds": [],
                "blocking_reasons": ["no_threshold_meets_coverage_and_count_gates"],
            }
        )
    tested: list[dict[str, Any]] = []
    selected_threshold: float | None = None
    for threshold in eligible_thresholds:
        accepted_count, errors = threshold_counts[threshold]
        p_value = float(
            binom.cdf(errors, accepted_count, TARGET_SELECTIVE_ERROR)
        )
        passed = p_value <= RISK_FAILURE_PROBABILITY
        tested.append(
            {
                "threshold": threshold,
                "accepted_groups": accepted_count,
                "coverage": accepted_count / total,
                "end_to_end_errors": errors,
                "empirical_risk": errors / accepted_count,
                "least_favourable_binomial_p_value": p_value,
                "passed": passed,
            }
        )
        if not passed:
            break
        selected_threshold = threshold
    if selected_threshold is None:
        return _artifact(
            {
                **common,
                "status": "blocked",
                "decision": "manual_review_all",
                "selected_threshold": None,
                "tested_thresholds": tested,
                "blocking_reasons": ["no_threshold_passed_fixed_sequence_risk_test"],
            }
        )
    return _artifact(
        {
            **common,
            "status": "passed_research_gate",
            "decision": "thresholded_auto_accept_else_manual_review",
            "selected_threshold": selected_threshold,
            "tested_thresholds": tested,
            "blocking_reasons": [],
        }
    )
