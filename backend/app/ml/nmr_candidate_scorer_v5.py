"""Outcome-blind multi-evidence ranking over frozen v4 ranking evidence.

The scorer intentionally accepts only the ``ranking`` object emitted by the
frozen v4 scorer.  Record identifiers, split/group metadata, candidate origin,
review metadata, and outcome labels belong to the offline experiment layer and
are rejected recursively here.

The fitted model is a small JSON artifact.  Callers must provide its expected
canonical SHA-256; an internal self-declared hash alone is not trusted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import platform
import re
from typing import Any

import numpy as np


PROTOCOL_VERSION = "chemapp.nmr.candidate-scorer.v5"
V4_PROTOCOL_VERSION = "chemapp.nmr.candidate-scorer.v4"
MODEL_SCHEMA_VERSION = "chemapp.nmr.candidate-ranker-model.v5"
FEATURE_SCHEMA_VERSION = "chemapp.nmr.candidate-features.v5"
OUTPUT_SCHEMA_VERSION = "chemapp.nmr.roleless-ranking.v5"

FEATURE_NAMES = (
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

_BANNED_KEY_FRAGMENTS = (
    "truth",
    "gold",
    "outcome",
    "split",
    "source",
    "reviewer",
    "correct",
    "label",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_ID_RE = re.compile(r"^candidate-[0-9a-f]{24}$")

_V3_SCORE_SEMANTICS = "atom_level_13c_hungarian_mae_v1"
_V4_SCORE_SEMANTICS = "symmetry_collapsed_13c_gaussian_set_similarity_v1"
_V4_SYMMETRY_METHOD = "rdkit_addhs_canonical_rank_atoms_break_ties_false_v1"
_V3_RANK_BASIS = (
    "higher_atom_bidirectional_coverage_then_fewer_unmatched_then_"
    "lower_atom_hungarian_mae_rmse_then_candidate_id"
)
_V4_RANK_BASIS = (
    "higher_symmetry_gaussian_set_similarity_then_lower_matched_max_"
    "error_then_lower_matched_mae_then_candidate_id"
)

_RANKING_KEYS = frozenset(
    {
        "calibrated_probability",
        "candidate_count",
        "candidates",
        "observed_13c",
        "observed_13c_sha256",
        "protocol_version",
        "runtime",
        "v3_order",
        "v3_rank_basis",
        "v4_order",
        "v4_rank_basis",
    }
)
_CANDIDATE_KEYS = frozenset(
    {
        "calibrated_probability",
        "candidate_id",
        "canonical_smiles",
        "protocol_version",
        "v3",
        "v3_rank",
        "v4",
        "v4_rank",
    }
)
_V3_KEYS = frozenset(
    {
        "bidirectional_coverage",
        "candidate_score",
        "count_agreement",
        "matched_count",
        "matched_mae_ppm",
        "matched_max_abs_error_ppm",
        "matched_rmse_ppm",
        "matching_sha256",
        "observed_count",
        "observed_coverage",
        "predicted_count",
        "predicted_coverage",
        "score_semantics",
        "unmatched_observed_count",
        "unmatched_predicted_count",
    }
)
_V4_KEYS = frozenset(
    {
        "bidirectional_coverage",
        "candidate_score",
        "count_agreement",
        "matched_count",
        "matched_kernel_sum",
        "matched_mae_ppm",
        "matched_max_abs_error_ppm",
        "matched_rmse_ppm",
        "matching_sha256",
        "observed_count",
        "observed_coverage",
        "predicted_carbon_atom_count",
        "predicted_count",
        "predicted_coverage",
        "predicted_symmetry_signal_count",
        "score_semantics",
        "set_similarity",
        "sigma_13c_ppm",
        "symmetry_classes_sha256",
        "symmetry_method",
        "unmatched_observed_count",
        "unmatched_predicted_count",
    }
)
_RUNTIME_KEYS = frozenset({"numpy_version", "rdkit_version", "scipy_version"})
_MODEL_KEYS = frozenset(
    {
        "artifact_sha256",
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
        "protocol_version",
        "runtime",
        "schema_version",
    }
)
_NORMALIZATION_KEYS = frozenset({"means", "scales"})


class NMRCandidateScorerV5InputError(ValueError):
    """Raised when roleless evidence or a v5 model violates its contract."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical JSON encoding used by every v5 hash."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NMRCandidateScorerV5InputError(
            "value is not finite canonical JSON"
        ) from exc
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _strict_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        raise NMRCandidateScorerV5InputError(
            f"{context} fields do not match the strict allowlist; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _reject_role_metadata(value: Any, *, path: str = "ranking") -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise NMRCandidateScorerV5InputError(
                    f"{path} contains a non-string object key"
                )
            lowered = key.casefold()
            if any(fragment in lowered for fragment in _BANNED_KEY_FRAGMENTS):
                raise NMRCandidateScorerV5InputError(
                    f"{path}.{key} contains prohibited role/outcome metadata"
                )
            _reject_role_metadata(nested, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, nested in enumerate(value):
            _reject_role_metadata(nested, path=f"{path}[{index}]")


def _finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NMRCandidateScorerV5InputError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise NMRCandidateScorerV5InputError(f"{context} must be finite")
    return result


def _nonnegative_int(value: Any, *, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise NMRCandidateScorerV5InputError(
            f"{context} must be a non-negative integer"
        )
    return value


def _positive_rank(value: Any, *, context: str, maximum: int) -> int:
    rank = _nonnegative_int(value, context=context)
    if not 1 <= rank <= maximum:
        raise NMRCandidateScorerV5InputError(f"{context} must be within 1..{maximum}")
    return rank


def _sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise NMRCandidateScorerV5InputError(f"{context} must be a lowercase SHA-256")
    return value


def _validate_metrics(
    metrics: Mapping[str, Any],
    *,
    expected_keys: frozenset[str],
    context: str,
    expected_observed_count: int,
) -> dict[str, Any]:
    if not isinstance(metrics, Mapping):
        raise NMRCandidateScorerV5InputError(f"{context} must be an object")
    _strict_keys(metrics, expected_keys, context=context)
    clean = dict(metrics)
    for key in (
        "matched_count",
        "observed_count",
        "predicted_count",
        "unmatched_observed_count",
        "unmatched_predicted_count",
    ):
        clean[key] = _nonnegative_int(metrics[key], context=f"{context}.{key}")
    if clean["observed_count"] != expected_observed_count:
        raise NMRCandidateScorerV5InputError(
            f"{context}.observed_count disagrees with observed_13c"
        )
    if clean["predicted_count"] == 0:
        raise NMRCandidateScorerV5InputError(
            f"{context}.predicted_count must be positive"
        )
    if clean["matched_count"] > min(clean["observed_count"], clean["predicted_count"]):
        raise NMRCandidateScorerV5InputError(
            f"{context}.matched_count exceeds an assignment dimension"
        )
    if clean["unmatched_observed_count"] != (
        clean["observed_count"] - clean["matched_count"]
    ) or clean["unmatched_predicted_count"] != (
        clean["predicted_count"] - clean["matched_count"]
    ):
        raise NMRCandidateScorerV5InputError(
            f"{context} unmatched counts are inconsistent"
        )
    for key in (
        "candidate_score",
        "matched_mae_ppm",
        "matched_max_abs_error_ppm",
        "matched_rmse_ppm",
        "observed_coverage",
        "predicted_coverage",
        "bidirectional_coverage",
        "count_agreement",
    ):
        clean[key] = _finite_float(metrics[key], context=f"{context}.{key}")
    for key in (
        "observed_coverage",
        "predicted_coverage",
        "bidirectional_coverage",
        "count_agreement",
    ):
        if not 0.0 <= clean[key] <= 1.0:
            raise NMRCandidateScorerV5InputError(f"{context}.{key} must be within 0..1")
    for key in (
        "matched_mae_ppm",
        "matched_max_abs_error_ppm",
        "matched_rmse_ppm",
    ):
        if clean[key] < 0.0:
            raise NMRCandidateScorerV5InputError(
                f"{context}.{key} must be non-negative"
            )
    if not (
        clean["matched_mae_ppm"]
        <= clean["matched_rmse_ppm"] + 1e-12
        <= clean["matched_max_abs_error_ppm"] + 1e-12
    ):
        raise NMRCandidateScorerV5InputError(
            f"{context} assignment error metrics are inconsistent"
        )
    _sha256(metrics["matching_sha256"], context=f"{context}.matching_sha256")
    if (
        not isinstance(metrics["score_semantics"], str)
        or not metrics["score_semantics"]
    ):
        raise NMRCandidateScorerV5InputError(
            f"{context}.score_semantics must be non-empty"
        )
    expected_observed_coverage = clean["matched_count"] / clean["observed_count"]
    expected_predicted_coverage = clean["matched_count"] / clean["predicted_count"]
    expected_count_agreement = min(
        clean["observed_count"], clean["predicted_count"]
    ) / max(clean["observed_count"], clean["predicted_count"])
    relationships = (
        ("observed_coverage", expected_observed_coverage),
        ("predicted_coverage", expected_predicted_coverage),
        (
            "bidirectional_coverage",
            min(expected_observed_coverage, expected_predicted_coverage),
        ),
        ("count_agreement", expected_count_agreement),
    )
    for key, expected in relationships:
        if not math.isclose(clean[key], expected, rel_tol=0.0, abs_tol=1e-12):
            raise NMRCandidateScorerV5InputError(
                f"{context}.{key} is mathematically inconsistent"
            )
    return clean


def _validate_ranking(v4_ranking: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(v4_ranking, Mapping):
        raise NMRCandidateScorerV5InputError("v4 ranking must be an object")
    _reject_role_metadata(v4_ranking)
    _strict_keys(v4_ranking, _RANKING_KEYS, context="v4 ranking")
    if v4_ranking["protocol_version"] != V4_PROTOCOL_VERSION:
        raise NMRCandidateScorerV5InputError("unsupported v4 protocol version")
    if v4_ranking["calibrated_probability"] is not False:
        raise NMRCandidateScorerV5InputError(
            "v4 evidence must not claim a calibrated probability"
        )
    observed = v4_ranking["observed_13c"]
    if not isinstance(observed, list) or not observed:
        raise NMRCandidateScorerV5InputError("observed_13c must be a non-empty list")
    clean_observed = [
        _finite_float(value, context=f"observed_13c[{index}]")
        for index, value in enumerate(observed)
    ]
    if clean_observed != sorted(clean_observed):
        raise NMRCandidateScorerV5InputError("observed_13c must be sorted")
    if _sha256(
        v4_ranking["observed_13c_sha256"], context="observed_13c_sha256"
    ) != canonical_sha256(clean_observed):
        raise NMRCandidateScorerV5InputError("observed_13c hash mismatch")

    raw_candidates = v4_ranking["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise NMRCandidateScorerV5InputError("candidates must be a non-empty list")
    candidate_count = _nonnegative_int(
        v4_ranking["candidate_count"], context="candidate_count"
    )
    if candidate_count != len(raw_candidates):
        raise NMRCandidateScorerV5InputError("candidate_count mismatch")

    clean_candidates: list[dict[str, Any]] = []
    for position, candidate in enumerate(raw_candidates):
        context = f"candidates[{position}]"
        if not isinstance(candidate, Mapping):
            raise NMRCandidateScorerV5InputError(f"{context} must be an object")
        _strict_keys(candidate, _CANDIDATE_KEYS, context=context)
        if candidate["protocol_version"] != V4_PROTOCOL_VERSION:
            raise NMRCandidateScorerV5InputError(
                f"{context} has an unsupported protocol version"
            )
        if candidate["calibrated_probability"] is not False:
            raise NMRCandidateScorerV5InputError(
                f"{context} claims a calibrated probability"
            )
        candidate_id = candidate["candidate_id"]
        smiles = candidate["canonical_smiles"]
        if (
            not isinstance(candidate_id, str)
            or _CANDIDATE_ID_RE.fullmatch(candidate_id) is None
        ):
            raise NMRCandidateScorerV5InputError(
                f"{context}.candidate_id must satisfy the opaque-ID contract"
            )
        if not isinstance(smiles, str) or not smiles:
            raise NMRCandidateScorerV5InputError(
                f"{context}.canonical_smiles must be non-empty"
            )
        v3 = _validate_metrics(
            candidate["v3"],
            expected_keys=_V3_KEYS,
            context=f"{context}.v3",
            expected_observed_count=len(clean_observed),
        )
        v4 = _validate_metrics(
            candidate["v4"],
            expected_keys=_V4_KEYS,
            context=f"{context}.v4",
            expected_observed_count=len(clean_observed),
        )
        for key in (
            "matched_kernel_sum",
            "set_similarity",
            "sigma_13c_ppm",
        ):
            v4[key] = _finite_float(candidate["v4"][key], context=f"{context}.v4.{key}")
        for key in ("predicted_carbon_atom_count", "predicted_symmetry_signal_count"):
            v4[key] = _nonnegative_int(
                candidate["v4"][key], context=f"{context}.v4.{key}"
            )
        if v4["predicted_count"] != v4["predicted_symmetry_signal_count"]:
            raise NMRCandidateScorerV5InputError(
                f"{context}.v4 symmetry signal count mismatch"
            )
        if (
            v4["predicted_symmetry_signal_count"] == 0
            or v4["predicted_carbon_atom_count"] < v4["predicted_symmetry_signal_count"]
        ):
            raise NMRCandidateScorerV5InputError(
                f"{context}.v4 carbon/symmetry counts are inconsistent"
            )
        if not 0.0 <= v4["set_similarity"] <= 1.0:
            raise NMRCandidateScorerV5InputError(
                f"{context}.v4.set_similarity must be within 0..1"
            )
        expected_set_similarity = v4["matched_kernel_sum"] / math.sqrt(
            v4["observed_count"] * v4["predicted_count"]
        )
        if (
            not 0.0 <= v4["matched_kernel_sum"] <= v4["matched_count"]
            or not math.isclose(
                v4["set_similarity"],
                expected_set_similarity,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(v4["sigma_13c_ppm"], 10.0, rel_tol=0.0, abs_tol=1e-12)
            or v3["score_semantics"] != _V3_SCORE_SEMANTICS
            or v4["score_semantics"] != _V4_SCORE_SEMANTICS
            or candidate["v4"]["symmetry_method"] != _V4_SYMMETRY_METHOD
        ):
            raise NMRCandidateScorerV5InputError(
                f"{context} frozen v3/v4 evidence semantics are inconsistent"
            )
        if not math.isclose(
            v4["candidate_score"],
            v4["set_similarity"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            v3["candidate_score"],
            -v3["matched_mae_ppm"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise NMRCandidateScorerV5InputError(
                f"{context} candidate score semantics are inconsistent"
            )
        _sha256(
            candidate["v4"]["symmetry_classes_sha256"],
            context=f"{context}.v4.symmetry_classes_sha256",
        )
        v3_rank = _positive_rank(
            candidate["v3_rank"],
            context=f"{context}.v3_rank",
            maximum=candidate_count,
        )
        v4_rank = _positive_rank(
            candidate["v4_rank"],
            context=f"{context}.v4_rank",
            maximum=candidate_count,
        )
        clean_candidates.append(
            {
                "candidate_id": candidate_id,
                "canonical_smiles": smiles,
                "v3": v3,
                "v4": v4,
                "v3_rank": v3_rank,
                "v4_rank": v4_rank,
            }
        )

    ids = [candidate["candidate_id"] for candidate in clean_candidates]
    smiles_values = [candidate["canonical_smiles"] for candidate in clean_candidates]
    if len(set(ids)) != candidate_count or len(set(smiles_values)) != candidate_count:
        raise NMRCandidateScorerV5InputError(
            "candidate IDs and canonical structures must be unique"
        )
    for ranker in ("v3", "v4"):
        order = v4_ranking[f"{ranker}_order"]
        if (
            not isinstance(order, list)
            or len(order) != candidate_count
            or set(order) != set(ids)
        ):
            raise NMRCandidateScorerV5InputError(f"{ranker}_order is invalid")
        rank_map = {
            candidate["candidate_id"]: candidate[f"{ranker}_rank"]
            for candidate in clean_candidates
        }
        if set(rank_map.values()) != set(range(1, candidate_count + 1)):
            raise NMRCandidateScorerV5InputError(
                f"{ranker} ranks must be a complete permutation"
            )
        if order != sorted(ids, key=lambda candidate_id: rank_map[candidate_id]):
            raise NMRCandidateScorerV5InputError(
                f"{ranker}_order disagrees with candidate ranks"
            )
        basis = v4_ranking[f"{ranker}_rank_basis"]
        expected_basis = _V3_RANK_BASIS if ranker == "v3" else _V4_RANK_BASIS
        if basis != expected_basis:
            raise NMRCandidateScorerV5InputError(
                f"{ranker}_rank_basis changed from the frozen v4 contract"
            )
    runtime = v4_ranking["runtime"]
    if not isinstance(runtime, Mapping):
        raise NMRCandidateScorerV5InputError("runtime must be an object")
    _strict_keys(runtime, _RUNTIME_KEYS, context="runtime")
    if any(not isinstance(runtime[key], str) or not runtime[key] for key in runtime):
        raise NMRCandidateScorerV5InputError(
            "runtime versions must be non-empty strings"
        )
    return {
        "observed_13c": clean_observed,
        "candidates": clean_candidates,
        "candidate_count": candidate_count,
    }


def extract_feature_rows(v4_ranking: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate v4 evidence and extract the fixed, ordered v5 feature rows."""

    clean = _validate_ranking(v4_ranking)
    candidates = clean["candidates"]
    candidate_count = clean["candidate_count"]
    v4_scores = {
        candidate["candidate_id"]: float(candidate["v4"]["candidate_score"])
        for candidate in candidates
    }
    v3_scores = {
        candidate["candidate_id"]: float(candidate["v3"]["candidate_score"])
        for candidate in candidates
    }
    consensus = {
        candidate["candidate_id"]: 0.5
        * (1.0 / candidate["v3_rank"] + 1.0 / candidate["v4_rank"])
        for candidate in candidates
    }

    def competitor_margin(candidate_id: str, values: Mapping[str, float]) -> float:
        competitors = [value for key, value in values.items() if key != candidate_id]
        return 0.0 if not competitors else values[candidate_id] - max(competitors)

    output: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        v3 = candidate["v3"]
        v4 = candidate["v4"]
        rank_denominator = max(candidate_count - 1, 1)
        values = (
            float(v4["set_similarity"]),
            float(v4["matched_mae_ppm"]),
            float(v4["matched_rmse_ppm"]),
            float(v4["matched_max_abs_error_ppm"]),
            float(v4["bidirectional_coverage"]),
            float(v4["count_agreement"]),
            float(v3["matched_mae_ppm"]),
            float(v3["matched_rmse_ppm"]),
            float(v3["matched_max_abs_error_ppm"]),
            float(v3["bidirectional_coverage"]),
            float(v3["count_agreement"]),
            math.log(float(v4["predicted_count"]) / float(v4["observed_count"])),
            consensus[candidate_id],
            abs(candidate["v3_rank"] - candidate["v4_rank"]) / rank_denominator,
            competitor_margin(candidate_id, v4_scores),
            competitor_margin(candidate_id, v3_scores),
            competitor_margin(candidate_id, consensus),
            float(v4["bidirectional_coverage"]) - float(v3["bidirectional_coverage"]),
        )
        if len(values) != len(FEATURE_NAMES) or not all(map(math.isfinite, values)):
            raise NMRCandidateScorerV5InputError(
                f"candidate {candidate_id} produced an invalid feature vector"
            )
        output.append(
            {
                "candidate_id": candidate_id,
                "canonical_smiles": candidate["canonical_smiles"],
                "feature_values": list(values),
            }
        )
    return sorted(output, key=lambda row: str(row["candidate_id"]))


def model_artifact_sha256(model: Mapping[str, Any]) -> str:
    core = {key: value for key, value in model.items() if key != "artifact_sha256"}
    return canonical_sha256(core)


def finalize_model_artifact(model_core: Mapping[str, Any]) -> dict[str, Any]:
    if "artifact_sha256" in model_core:
        raise NMRCandidateScorerV5InputError(
            "model core must not contain artifact_sha256"
        )
    model = dict(model_core)
    model["artifact_sha256"] = model_artifact_sha256(model)
    return model


def validate_model_artifact(
    model: Mapping[str, Any], *, expected_model_sha256: str
) -> dict[str, Any]:
    if not isinstance(model, Mapping):
        raise NMRCandidateScorerV5InputError("v5 model must be an object")
    _strict_keys(model, _MODEL_KEYS, context="v5 model")
    expected_model_sha256 = _sha256(
        expected_model_sha256, context="expected_model_sha256"
    )
    declared = _sha256(model["artifact_sha256"], context="artifact_sha256")
    computed = model_artifact_sha256(model)
    if declared != computed or declared != expected_model_sha256:
        raise NMRCandidateScorerV5InputError("v5 model hash mismatch")
    if model["schema_version"] != MODEL_SCHEMA_VERSION:
        raise NMRCandidateScorerV5InputError("unsupported v5 model schema")
    if model["protocol_version"] != PROTOCOL_VERSION:
        raise NMRCandidateScorerV5InputError("v5 model protocol mismatch")
    if model["feature_schema_version"] != FEATURE_SCHEMA_VERSION:
        raise NMRCandidateScorerV5InputError("v5 feature schema mismatch")
    if list(model["feature_names"]) != list(FEATURE_NAMES):
        raise NMRCandidateScorerV5InputError("v5 feature order mismatch")
    if model["model_kind"] != "symmetric_pairwise_l2_logistic_v1":
        raise NMRCandidateScorerV5InputError("unsupported v5 model kind")
    if model["fit_intercept"] is not False:
        raise NMRCandidateScorerV5InputError("v5 model must not fit an intercept")
    intercept = _finite_float(model["intercept"], context="intercept")
    if intercept != 0.0:
        raise NMRCandidateScorerV5InputError("v5 model intercept must be zero")
    if (
        model["calibrated_probability"] is not False
        or model["conditional_probability"] is not False
    ):
        raise NMRCandidateScorerV5InputError(
            "v5 model must not claim calibrated or conditional probability"
        )
    normalization = model["normalization"]
    if not isinstance(normalization, Mapping):
        raise NMRCandidateScorerV5InputError("normalization must be an object")
    _strict_keys(normalization, _NORMALIZATION_KEYS, context="normalization")
    means = normalization["means"]
    scales = normalization["scales"]
    coefficients = model["coefficients"]
    if not all(isinstance(value, list) for value in (means, scales, coefficients)):
        raise NMRCandidateScorerV5InputError(
            "normalization and coefficients must be arrays"
        )
    if not (len(means) == len(scales) == len(coefficients) == len(FEATURE_NAMES)):
        raise NMRCandidateScorerV5InputError("v5 model vector length mismatch")
    clean_means = [
        _finite_float(value, context=f"normalization.means[{index}]")
        for index, value in enumerate(means)
    ]
    clean_scales = [
        _finite_float(value, context=f"normalization.scales[{index}]")
        for index, value in enumerate(scales)
    ]
    clean_coefficients = [
        _finite_float(value, context=f"coefficients[{index}]")
        for index, value in enumerate(coefficients)
    ]
    if any(scale <= 0.0 for scale in clean_scales):
        raise NMRCandidateScorerV5InputError("normalization scales must be positive")
    if not isinstance(model["development_binding"], Mapping):
        raise NMRCandidateScorerV5InputError("development_binding must be an object")
    if not isinstance(model["runtime"], Mapping):
        raise NMRCandidateScorerV5InputError("model runtime must be an object")
    return {
        "artifact_sha256": declared,
        "means": clean_means,
        "scales": clean_scales,
        "coefficients": clean_coefficients,
    }


def rank_candidate_evidence_v5(
    v4_ranking: Mapping[str, Any],
    model: Mapping[str, Any],
    *,
    expected_model_sha256: str,
) -> dict[str, Any]:
    """Rank one candidate pool without accepting any outcome-bearing fields."""

    model_values = validate_model_artifact(
        model, expected_model_sha256=expected_model_sha256
    )
    feature_rows = extract_feature_rows(v4_ranking)
    means = np.asarray(model_values["means"], dtype=np.float64)
    scales = np.asarray(model_values["scales"], dtype=np.float64)
    coefficients = np.asarray(model_values["coefficients"], dtype=np.float64)
    candidates: list[dict[str, Any]] = []
    for row in feature_rows:
        raw = np.asarray(row["feature_values"], dtype=np.float64)
        normalised = (raw - means) / scales
        contributions = normalised * coefficients
        utility = float(np.sum(contributions))
        if not math.isfinite(utility):
            raise NMRCandidateScorerV5InputError(
                f"candidate {row['candidate_id']} produced a non-finite utility"
            )
        candidates.append(
            {
                "candidate_id": row["candidate_id"],
                "canonical_smiles": row["canonical_smiles"],
                "feature_values": raw.tolist(),
                "normalised_feature_values": normalised.tolist(),
                "feature_contributions": contributions.tolist(),
                "v5_utility": utility,
                "calibrated_probability": False,
                "conditional_probability": False,
            }
        )
    ordered = sorted(
        candidates,
        key=lambda row: (-float(row["v5_utility"]), str(row["candidate_id"])),
    )
    ranks = {
        str(candidate["candidate_id"]): rank
        for rank, candidate in enumerate(ordered, start=1)
    }
    stable_candidates = [
        {**candidate, "v5_rank": ranks[str(candidate["candidate_id"])]}
        for candidate in sorted(candidates, key=lambda row: str(row["candidate_id"]))
    ]
    output_core = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "model_artifact_sha256": model_values["artifact_sha256"],
        "input_v4_ranking_sha256": canonical_sha256(v4_ranking),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "candidate_count": len(stable_candidates),
        "v5_order": [str(candidate["candidate_id"]) for candidate in ordered],
        "v5_rank_basis": "higher_pairwise_linear_utility_then_candidate_id",
        "candidates": stable_candidates,
        "calibrated_probability": False,
        "conditional_probability": False,
        "runtime": {
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
        },
    }
    return {**output_core, "ranking_sha256": canonical_sha256(output_core)}


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_SCHEMA_VERSION",
    "MODEL_SCHEMA_VERSION",
    "NMRCandidateScorerV5InputError",
    "OUTPUT_SCHEMA_VERSION",
    "PROTOCOL_VERSION",
    "canonical_json_bytes",
    "canonical_sha256",
    "extract_feature_rows",
    "finalize_model_artifact",
    "model_artifact_sha256",
    "rank_candidate_evidence_v5",
    "validate_model_artifact",
]
