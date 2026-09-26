"""Leakage-resistant calibration primitives for the fourth NMR experiment.

This module is deliberately independent from :mod:`nmr_calibration_training`.
The v3 pipeline and its consumed frozen-test evidence remain immutable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score


TEST_COHORT_BINDING_SCHEMA_VERSION = "chemapp.nmr.test-cohort-binding.v1"
TEST_CONSUMPTION_SCHEMA_VERSION = "chemapp.nmr.frozen-test-consumption.v2"
TEST_CONSUMPTION_LEDGER_VERSION = 2
CALIBRATOR_SCHEMA_VERSION = "chemapp.nmr.correctness-calibrator.v2"
METHOD_SELECTION_SCHEMA_VERSION = "chemapp.nmr.calibration-method-selection.v1"
METHOD_SELECTION_PROTOCOL_VERSION = 2

SUPPORTED_METHODS = ("regularized_sigmoid", "isotonic", "beta")
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_LEDGER_ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "ledger_version",
        "event",
        "run_id",
        "source_release_id",
        "run_spec_file_sha256",
        "test_cohort_sha256",
        "reserved_at",
    }
)
_CALIBRATOR_ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "method",
        "target",
        "input_domain",
        "score_direction",
        "fit_config",
        "parameters",
        "artifact_sha256",
    }
)
_DEV_ROW_ALLOWED_FIELDS = frozenset(
    {"record_id", "split", "group_id", "raw_score", "outcome"}
)


def canonical_json_dumps(value: Any) -> str:
    """Render deterministic JSON and reject non-finite JSON numbers."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    """Hash the canonical JSON representation of ``value``."""

    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HASH_RE.fullmatch(value) is not None


def _nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _parse_utc(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an explicit UTC timestamp") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
    ):
        raise ValueError(f"{field_name} must be an explicit UTC timestamp")
    return parsed


def test_cohort_sha256(
    release_id: str,
    records: Sequence[Mapping[str, Any]],
) -> str:
    """Return a model-independent identity for one external test cohort.

    Only the stable sorted spectrum-fingerprint set enters the digest.  Source
    release, record, group, split, model, score, outcome, and calibration names
    are intentionally excluded so renaming/repartitioning cannot mint a new
    cohort identity.  ``release_id`` remains a validated compatibility
    argument; release-level single-use is enforced separately by the ledger.
    """

    _nonempty_text(release_id, "release_id")
    fingerprints: list[str] = []
    seen_fingerprints: set[str] = set()
    for index, row in enumerate(records):
        if not isinstance(row, Mapping):
            raise ValueError(f"records[{index}] must be an object")
        fingerprint = row.get("spectrum_fingerprint_sha256")
        if not _is_sha256(fingerprint):
            raise ValueError(
                f"records[{index}].spectrum_fingerprint_sha256 "
                "must be lowercase SHA-256"
            )
        fingerprint_text = str(fingerprint)
        if fingerprint_text in seen_fingerprints:
            raise ValueError(
                f"duplicate test cohort spectrum fingerprint: {fingerprint_text}"
            )
        seen_fingerprints.add(fingerprint_text)
        fingerprints.append(fingerprint_text)
    if not fingerprints:
        raise ValueError("test cohort must contain at least one record")
    fingerprints.sort()
    return canonical_sha256(
        {
            "schema_version": TEST_COHORT_BINDING_SCHEMA_VERSION,
            "spectrum_fingerprint_sha256": fingerprints,
        }
    )


# Prevent pytest from collecting this public API when a test module imports it.
test_cohort_sha256.__test__ = False


def _validate_ledger_entry(
    value: Any,
    *,
    location: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location}: ledger entry must be an object")
    fields = set(value)
    if fields != _LEDGER_ALLOWED_FIELDS:
        difference = sorted(fields ^ _LEDGER_ALLOWED_FIELDS)
        raise ValueError(f"{location}: ledger field allowlist mismatch: {difference}")
    if value["schema_version"] != TEST_CONSUMPTION_SCHEMA_VERSION:
        raise ValueError(f"{location}: ledger schema changed")
    if value["ledger_version"] != TEST_CONSUMPTION_LEDGER_VERSION:
        raise ValueError(f"{location}: ledger version changed")
    if value["event"] != "reserved_before_external_test_outcome_read":
        raise ValueError(f"{location}: ledger event is invalid")
    _nonempty_text(value["run_id"], f"{location}.run_id")
    for field_name in (
        "source_release_id",
        "run_spec_file_sha256",
        "test_cohort_sha256",
    ):
        if not _is_sha256(value[field_name]):
            raise ValueError(f"{location}.{field_name} must be lowercase SHA-256")
    _parse_utc(value["reserved_at"], f"{location}.reserved_at")
    return value


def reserve_test_cohort_consumption(
    ledger_path: str | Path,
    *,
    run_id: str,
    source_release_id: str,
    run_spec_file_sha256: str,
    test_cohort_sha256: str,
    _clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Atomically reserve a cohort once within the canonical v2 ledger.

    Rebuilding predictions, changing the ranker, or changing the calibrator
    cannot bypass the check because those model-dependent values are not part
    of the permanent consumption key.
    """

    _nonempty_text(run_id, "run_id")
    if not _is_sha256(source_release_id):
        raise ValueError("source_release_id must be lowercase SHA-256")
    if not _is_sha256(run_spec_file_sha256):
        raise ValueError("run_spec_file_sha256 must be lowercase SHA-256")
    if not _is_sha256(test_cohort_sha256):
        raise ValueError("test_cohort_sha256 must be lowercase SHA-256")
    now = (_clock or (lambda: datetime.now(timezone.utc)))()
    if not isinstance(now, datetime):
        raise ValueError("ledger clock must return datetime")
    reserved_at = now.astimezone(timezone.utc).isoformat()
    _parse_utc(reserved_at, "reserved_at")

    path = Path(ledger_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f"{path.name}.lock")
    try:
        lock_path.mkdir()
    except FileExistsError as exc:
        raise ValueError(
            f"test-consumption ledger is locked; inspect stale lock {lock_path}"
        ) from exc

    temporary: Path | None = None
    try:
        entries: list[dict[str, Any]] = []
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()
            for line_number, line in enumerate(lines, start=1):
                if not line:
                    raise ValueError(f"{path}:{line_number}: blank ledger line")
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: corrupt ledger JSON"
                    ) from exc
                entries.append(
                    _validate_ledger_entry(
                        parsed,
                        location=f"{path}:{line_number}",
                    )
                )

        cohort_ids = [str(entry["test_cohort_sha256"]) for entry in entries]
        release_ids = [str(entry["source_release_id"]) for entry in entries]
        if len(cohort_ids) != len(set(cohort_ids)):
            raise ValueError(f"{path}: ledger contains duplicate cohort reservations")
        if len(release_ids) != len(set(release_ids)):
            raise ValueError(f"{path}: ledger contains duplicate release reservations")
        if test_cohort_sha256 in cohort_ids:
            raise ValueError(
                "external test cohort was already reserved/consumed and "
                "cannot be reused"
            )
        if source_release_id in release_ids:
            raise ValueError(
                "external source release was already reserved/consumed and "
                "cannot be reused with a changed cohort or split"
            )

        entry = {
            "schema_version": TEST_CONSUMPTION_SCHEMA_VERSION,
            "ledger_version": TEST_CONSUMPTION_LEDGER_VERSION,
            "event": "reserved_before_external_test_outcome_read",
            "run_id": run_id,
            "source_release_id": source_release_id,
            "run_spec_file_sha256": run_spec_file_sha256,
            "test_cohort_sha256": test_cohort_sha256,
            "reserved_at": reserved_at,
        }
        _validate_ledger_entry(entry, location="new ledger entry")
        entries.append(entry)
        rendered = "".join(canonical_json_dumps(item) + "\n" for item in entries)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".part",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        return entry
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock_path.rmdir()


def reserve_frozen_test_consumption_v2(
    ledger_path: str | Path,
    *,
    run_id: str,
    source_release_id: str,
    run_spec_file_sha256: str,
    test_cohort_sha256: str,
    _clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Compatibility name for the explicit v2 cohort reservation primitive."""

    return reserve_test_cohort_consumption(
        ledger_path,
        run_id=run_id,
        source_release_id=source_release_id,
        run_spec_file_sha256=run_spec_file_sha256,
        test_cohort_sha256=test_cohort_sha256,
        _clock=_clock,
    )


def _validated_vectors(
    raw_scores: Sequence[float],
    outcomes: Sequence[int | bool],
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(raw_scores, dtype=float)
    labels = np.asarray(outcomes)
    if (
        scores.ndim != 1
        or labels.ndim != 1
        or len(scores) != len(labels)
        or not len(scores)
    ):
        raise ValueError("calibrator requires aligned, non-empty vectors")
    if not np.isfinite(scores).all():
        raise ValueError("calibrator scores must be finite")
    if any(type(value) not in (bool, int, np.bool_, np.int64) for value in labels):
        raise ValueError("calibrator outcomes must be binary integers")
    integer_labels = labels.astype(int)
    if not set(integer_labels.tolist()).issubset({0, 1}):
        raise ValueError("calibrator outcomes must be binary integers")
    if len(set(integer_labels.tolist())) != 2:
        raise ValueError("calibrator requires both outcome classes")
    return scores, integer_labels


def _positive_finite(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{field_name} must be finite and positive")
    return float(value)


def _beta_features(probabilities: np.ndarray, epsilon: float) -> np.ndarray:
    if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
        raise ValueError("beta calibration requires unit-interval inputs")
    clipped = np.clip(probabilities, epsilon, 1.0 - epsilon)
    return np.column_stack((np.log(clipped), -np.log1p(-clipped)))


def _artifact_with_hash(core: Mapping[str, Any]) -> dict[str, Any]:
    materialized = dict(core)
    return {**materialized, "artifact_sha256": canonical_sha256(materialized)}


def fit_calibrator(
    raw_scores: Sequence[float],
    outcomes: Sequence[int | bool],
    *,
    method: str,
    regularization_c: float = 1.0,
    beta_epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Fit one deterministic JSON calibrator.

    ``regularized_sigmoid`` and ``isotonic`` accept finite real scores. ``beta``
    accepts probabilities in the closed unit interval and records the clipping
    epsilon used for its log features.
    """

    scores, labels = _validated_vectors(raw_scores, outcomes)
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"unsupported calibration method: {method}")

    if method == "regularized_sigmoid":
        c_value = _positive_finite(regularization_c, "regularization_c")
        center = float(np.mean(scores))
        scale = float(np.std(scores))
        if not math.isfinite(scale) or scale <= 0.0:
            raise ValueError("regularized sigmoid requires non-constant scores")
        normalized = ((scores - center) / scale).reshape(-1, 1)
        estimator = LogisticRegression(
            C=c_value,
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
        )
        estimator.fit(normalized, labels)
        coefficient = float(estimator.coef_[0, 0])
        intercept = float(estimator.intercept_[0])
        if (
            not math.isfinite(coefficient)
            or not math.isfinite(intercept)
            or coefficient <= 0.0
        ):
            raise ValueError(
                "regularized sigmoid violates higher-is-more-confident direction"
            )
        return _artifact_with_hash(
            {
                "schema_version": CALIBRATOR_SCHEMA_VERSION,
                "method": method,
                "target": "top1_exact_structure_correct",
                "input_domain": "finite_real",
                "score_direction": "higher_is_more_confident",
                "fit_config": {
                    "regularization_c": c_value,
                    "solver": "lbfgs",
                    "max_iter": 2000,
                    "random_state": 0,
                },
                "parameters": {
                    "score_center": center,
                    "score_scale": scale,
                    "coefficient": coefficient,
                    "intercept": intercept,
                },
            }
        )

    if method == "isotonic":
        if len(set(scores.tolist())) < 2:
            raise ValueError("isotonic calibration requires non-constant scores")
        estimator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        estimator.fit(scores, labels)
        x_thresholds = [float(value) for value in estimator.X_thresholds_]
        y_thresholds = [float(value) for value in estimator.y_thresholds_]
        if (
            len(x_thresholds) < 2
            or any(not math.isfinite(value) for value in x_thresholds + y_thresholds)
            or any(right <= left for left, right in zip(x_thresholds, x_thresholds[1:]))
            or any(right < left for left, right in zip(y_thresholds, y_thresholds[1:]))
            or any(not 0.0 <= value <= 1.0 for value in y_thresholds)
        ):
            raise ValueError("fitted isotonic calibrator is not monotonic in [0,1]")
        return _artifact_with_hash(
            {
                "schema_version": CALIBRATOR_SCHEMA_VERSION,
                "method": method,
                "target": "top1_exact_structure_correct",
                "input_domain": "finite_real",
                "score_direction": "higher_is_more_confident",
                "fit_config": {
                    "increasing": True,
                    "out_of_bounds": "clip",
                },
                "parameters": {
                    "x_thresholds": x_thresholds,
                    "y_thresholds": y_thresholds,
                },
            }
        )

    c_value = _positive_finite(regularization_c, "regularization_c")
    epsilon = _positive_finite(beta_epsilon, "beta_epsilon")
    if epsilon >= 0.5:
        raise ValueError("beta_epsilon must be smaller than 0.5")
    features = _beta_features(scores, epsilon)
    estimator = LogisticRegression(
        C=c_value,
        solver="lbfgs",
        max_iter=2000,
        random_state=0,
    )
    estimator.fit(features, labels)
    coefficient_log_p = float(estimator.coef_[0, 0])
    coefficient_log_one_minus_p = float(estimator.coef_[0, 1])
    intercept = float(estimator.intercept_[0])
    parameters = (
        coefficient_log_p,
        coefficient_log_one_minus_p,
        intercept,
    )
    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("beta calibration produced non-finite coefficients")
    if coefficient_log_p < 0.0 or coefficient_log_one_minus_p < 0.0:
        raise ValueError("beta calibrator is not monotonic increasing")
    return _artifact_with_hash(
        {
            "schema_version": CALIBRATOR_SCHEMA_VERSION,
            "method": method,
            "target": "top1_exact_structure_correct",
            "input_domain": "unit_interval_probability",
            "score_direction": "higher_is_more_confident",
            "fit_config": {
                "regularization_c": c_value,
                "solver": "lbfgs",
                "max_iter": 2000,
                "random_state": 0,
                "beta_epsilon": epsilon,
                "feature_transform": ["log(p)", "-log1p(-p)"],
            },
            "parameters": {
                "coefficient_log_p": coefficient_log_p,
                "coefficient_log_one_minus_p": coefficient_log_one_minus_p,
                "intercept": intercept,
            },
        }
    )


def _validate_calibrator_artifact(calibrator: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(calibrator)
    if set(value) != _CALIBRATOR_ALLOWED_FIELDS:
        difference = sorted(set(value) ^ _CALIBRATOR_ALLOWED_FIELDS)
        raise ValueError(f"calibrator field allowlist mismatch: {difference}")
    if value["schema_version"] != CALIBRATOR_SCHEMA_VERSION:
        raise ValueError("unsupported calibrator schema")
    method = value["method"]
    if method not in SUPPORTED_METHODS:
        raise ValueError("unsupported calibrator method")
    if value["target"] != "top1_exact_structure_correct":
        raise ValueError("unsupported calibrator target")
    if value["score_direction"] != "higher_is_more_confident":
        raise ValueError("unsupported calibrator score direction")
    claimed_hash = value["artifact_sha256"]
    core = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if not _is_sha256(claimed_hash) or claimed_hash != canonical_sha256(core):
        raise ValueError("calibrator artifact hash mismatch")
    if not isinstance(value["fit_config"], Mapping) or not isinstance(
        value["parameters"],
        Mapping,
    ):
        raise ValueError("calibrator config and parameters must be objects")
    return value


def _finite_parameter(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{field_name} must be finite")
    return float(value)


def predict_calibrated(
    calibrator: Mapping[str, Any],
    raw_scores: Sequence[float],
) -> list[float]:
    """Apply a validated v2 calibrator."""

    artifact = _validate_calibrator_artifact(calibrator)
    scores = np.asarray(raw_scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("raw scores must be a finite one-dimensional vector")
    method = str(artifact["method"])
    parameters = artifact["parameters"]
    fit_config = artifact["fit_config"]

    if method == "regularized_sigmoid":
        if artifact["input_domain"] != "finite_real":
            raise ValueError("regularized sigmoid input domain changed")
        if set(fit_config) != {
            "regularization_c",
            "solver",
            "max_iter",
            "random_state",
        }:
            raise ValueError("regularized sigmoid fit-config schema changed")
        _positive_finite(fit_config["regularization_c"], "regularization_c")
        if (
            fit_config["solver"] != "lbfgs"
            or fit_config["max_iter"] != 2000
            or fit_config["random_state"] != 0
        ):
            raise ValueError("regularized sigmoid fit configuration changed")
        if set(parameters) != {
            "score_center",
            "score_scale",
            "coefficient",
            "intercept",
        }:
            raise ValueError("regularized sigmoid parameter schema changed")
        center = _finite_parameter(parameters["score_center"], "score_center")
        scale = _positive_finite(parameters["score_scale"], "score_scale")
        coefficient = _positive_finite(parameters["coefficient"], "coefficient")
        intercept = _finite_parameter(parameters["intercept"], "intercept")
        logits = coefficient * ((scores - center) / scale) + intercept
    elif method == "isotonic":
        if artifact["input_domain"] != "finite_real":
            raise ValueError("isotonic input domain changed")
        if dict(fit_config) != {
            "increasing": True,
            "out_of_bounds": "clip",
        }:
            raise ValueError("isotonic fit configuration changed")
        if set(parameters) != {"x_thresholds", "y_thresholds"}:
            raise ValueError("isotonic parameter schema changed")
        thresholds = np.asarray(parameters["x_thresholds"], dtype=float)
        values = np.asarray(parameters["y_thresholds"], dtype=float)
        if (
            thresholds.ndim != 1
            or values.ndim != 1
            or len(thresholds) != len(values)
            or len(thresholds) < 2
            or not np.isfinite(thresholds).all()
            or not np.isfinite(values).all()
            or np.any(np.diff(thresholds) <= 0.0)
            or np.any(np.diff(values) < 0.0)
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError("invalid isotonic calibrator thresholds")
        probabilities = np.interp(scores, thresholds, values)
        return [float(value) for value in probabilities]
    else:
        if artifact["input_domain"] != "unit_interval_probability":
            raise ValueError("beta calibrator input domain changed")
        if set(parameters) != {
            "coefficient_log_p",
            "coefficient_log_one_minus_p",
            "intercept",
        }:
            raise ValueError("beta parameter schema changed")
        if set(fit_config) != {
            "regularization_c",
            "solver",
            "max_iter",
            "random_state",
            "beta_epsilon",
            "feature_transform",
        }:
            raise ValueError("beta fit-config schema changed")
        _positive_finite(fit_config["regularization_c"], "regularization_c")
        if (
            fit_config["solver"] != "lbfgs"
            or fit_config["max_iter"] != 2000
            or fit_config["random_state"] != 0
            or fit_config["feature_transform"] != ["log(p)", "-log1p(-p)"]
        ):
            raise ValueError("beta fit configuration changed")
        epsilon = _positive_finite(fit_config["beta_epsilon"], "beta_epsilon")
        if epsilon >= 0.5:
            raise ValueError("beta_epsilon must be smaller than 0.5")
        features = _beta_features(scores, epsilon)
        coefficient_log_p = _finite_parameter(
            parameters["coefficient_log_p"],
            "coefficient_log_p",
        )
        coefficient_log_one_minus_p = _finite_parameter(
            parameters["coefficient_log_one_minus_p"],
            "coefficient_log_one_minus_p",
        )
        if coefficient_log_p < 0.0 or coefficient_log_one_minus_p < 0.0:
            raise ValueError("beta calibrator is not monotonic increasing")
        intercept = _finite_parameter(parameters["intercept"], "intercept")
        logits = (
            coefficient_log_p * features[:, 0]
            + coefficient_log_one_minus_p * features[:, 1]
            + intercept
        )

    probabilities = np.empty_like(logits, dtype=float)
    positive = logits >= 0.0
    probabilities[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_logits = np.exp(logits[~positive])
    probabilities[~positive] = exp_logits / (1.0 + exp_logits)
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError("calibrator produced values outside finite [0,1]")
    return [float(value) for value in probabilities]


def _validated_probabilities_and_outcomes(
    probabilities: Sequence[float],
    outcomes: Sequence[int | bool],
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(probabilities, dtype=float)
    labels = np.asarray(outcomes)
    if (
        values.ndim != 1
        or labels.ndim != 1
        or len(values) != len(labels)
        or not len(values)
    ):
        raise ValueError("metrics require aligned, non-empty vectors")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("metric probabilities must be finite and in [0,1]")
    if any(type(value) not in (bool, int, np.bool_, np.int64) for value in labels):
        raise ValueError("metric outcomes must be binary integers")
    integer_labels = labels.astype(int)
    if not set(integer_labels.tolist()).issubset({0, 1}):
        raise ValueError("metric outcomes must be binary integers")
    return values, integer_labels


def risk_coverage_metrics(
    probabilities: Sequence[float],
    outcomes: Sequence[int | bool],
) -> dict[str, Any]:
    """Compute tie-invariant correctness risk as confidence coverage grows.

    The published curve contains only thresholds that can actually be applied:
    every row sharing the threshold confidence is accepted together.  AURC
    retains the previous rank-wise mean-risk definition, but replaces arbitrary
    input ordering inside a confidence tie with its exact expectation under a
    uniform random ordering of that tie.
    """

    values, labels = _validated_probabilities_and_outcomes(probabilities, outcomes)
    confidence_groups: dict[float, dict[str, int]] = {}
    for probability, outcome in zip(values, labels, strict=True):
        confidence = float(probability)
        group = confidence_groups.setdefault(
            confidence,
            {"count": 0, "errors": 0},
        )
        group["count"] += 1
        group["errors"] += 1 - int(outcome)

    accepted = 0
    cumulative_errors = 0
    expected_rankwise_risk_sum = 0.0
    curve: list[dict[str, Any]] = []
    for confidence in sorted(confidence_groups, reverse=True):
        group = confidence_groups[confidence]
        group_count = group["count"]
        group_errors = group["errors"]
        expected_error_fraction = group_errors / group_count
        for within_group_rank in range(1, group_count + 1):
            expected_errors = (
                cumulative_errors + within_group_rank * expected_error_fraction
            )
            expected_rankwise_risk_sum += expected_errors / (
                accepted + within_group_rank
            )
        accepted += group_count
        cumulative_errors += group_errors
        curve.append(
            {
                "accepted": accepted,
                "coverage": accepted / len(values),
                "risk": cumulative_errors / accepted,
                "minimum_confidence": confidence,
            }
        )

    return {
        "aurc": float(expected_rankwise_risk_sum / len(values)),
        "aurc_semantics": (
            "expected_rankwise_mean_cumulative_risk_under_uniform_random_"
            "ordering_within_confidence_ties"
        ),
        "curve_semantics": ("unique_confidence_thresholds_accept_all_rows_in_each_tie"),
        "curve": curve,
    }


def calibration_metrics(
    probabilities: Sequence[float],
    outcomes: Sequence[int | bool],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Return basic held-out discrimination, calibration, and coverage metrics."""

    values, labels = _validated_probabilities_and_outcomes(probabilities, outcomes)
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    epsilon = 1e-15
    clipped = np.clip(values, epsilon, 1.0 - epsilon)
    brier = float(np.mean((values - labels) ** 2))
    log_loss = float(
        -np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
    )

    indices = np.minimum((values * bins).astype(int), bins - 1)
    reliability_bins: list[dict[str, Any]] = []
    ece = 0.0
    for index in range(bins):
        mask = indices == index
        count = int(np.sum(mask))
        lower = index / bins
        upper = (index + 1) / bins
        if not count:
            reliability_bins.append(
                {
                    "index": index,
                    "lower": lower,
                    "upper": upper,
                    "count": 0,
                    "mean_confidence": None,
                    "accuracy": None,
                    "absolute_gap": None,
                }
            )
            continue
        mean_confidence = float(np.mean(values[mask]))
        accuracy = float(np.mean(labels[mask]))
        gap = abs(mean_confidence - accuracy)
        ece += (count / len(values)) * gap
        reliability_bins.append(
            {
                "index": index,
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_confidence": mean_confidence,
                "accuracy": accuracy,
                "absolute_gap": gap,
            }
        )

    has_both_classes = len(set(labels.tolist())) == 2
    return {
        "count": len(values),
        "brier": brier,
        "log_loss": log_loss,
        "ece": float(ece),
        "roc_auc": (float(roc_auc_score(labels, values)) if has_both_classes else None),
        "average_precision": (
            float(average_precision_score(labels, values)) if has_both_classes else None
        ),
        "reliability_bins": reliability_bins,
        "risk_coverage": risk_coverage_metrics(values, labels),
    }


def _normalize_dev_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen_record_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"dev rows[{index}] must be an object")
        fields = set(row)
        if fields != _DEV_ROW_ALLOWED_FIELDS:
            difference = sorted(fields ^ _DEV_ROW_ALLOWED_FIELDS)
            raise ValueError(
                f"dev rows[{index}] field allowlist mismatch: {difference}"
            )
        if row["split"] != "dev":
            raise ValueError(
                "method selection accepts dev rows only; calibration/test "
                "inputs are forbidden"
            )
        record_id = _nonempty_text(row["record_id"], f"dev rows[{index}].record_id")
        if record_id in seen_record_ids:
            raise ValueError(f"duplicate dev record_id: {record_id}")
        seen_record_ids.add(record_id)
        group_id = _nonempty_text(row["group_id"], f"dev rows[{index}].group_id")
        raw_score = row["raw_score"]
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, (int, float))
            or not math.isfinite(float(raw_score))
        ):
            raise ValueError(f"dev rows[{index}].raw_score must be finite")
        outcome = row["outcome"]
        if type(outcome) not in (bool, int) or int(outcome) not in (0, 1):
            raise ValueError(f"dev rows[{index}].outcome must be binary")
        normalized.append(
            {
                "record_id": record_id,
                "split": "dev",
                "group_id": group_id,
                "raw_score": float(raw_score),
                "outcome": int(outcome),
            }
        )
    if not normalized:
        raise ValueError("dev method selection requires at least one row")
    normalized.sort(key=lambda row: row["record_id"])
    return normalized


def _group_fold_assignments(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_splits: int,
    seed: int,
) -> tuple[dict[str, int], dict[str, int]]:
    groups = sorted({str(row["group_id"]) for row in rows})
    if (
        isinstance(n_splits, bool)
        or not isinstance(n_splits, int)
        or n_splits < 2
        or len(groups) < n_splits
    ):
        raise ValueError("n_splits requires at least two folds and as many groups")
    ordered_groups = sorted(
        groups,
        key=lambda group: (
            hashlib.sha256(f"{seed}\0{group}".encode()).hexdigest(),
            group,
        ),
    )
    group_folds = {
        group: index % n_splits for index, group in enumerate(ordered_groups)
    }
    record_folds = {
        str(row["record_id"]): group_folds[str(row["group_id"])] for row in rows
    }
    return group_folds, record_folds


def select_calibration_method_dev(
    rows: Sequence[Mapping[str, Any]],
    *,
    methods: Sequence[str] = SUPPORTED_METHODS,
    n_splits: int = 5,
    seed: int = 0,
    bins: int,
    primary_metric: str = "log_loss",
    regularization_c: float = 1.0,
    beta_epsilon: float = 1e-6,
) -> dict[str, Any]:
    """Select a calibration method using deterministic group CV on dev only."""

    dev_rows = _normalize_dev_rows(rows)
    selected_methods = tuple(methods)
    if (
        not selected_methods
        or len(set(selected_methods)) != len(selected_methods)
        or any(method not in SUPPORTED_METHODS for method in selected_methods)
    ):
        raise ValueError("methods must be unique supported calibration methods")
    if primary_metric not in {"log_loss", "brier", "ece"}:
        raise ValueError("primary_metric must be log_loss, brier, or ece")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise ValueError("bins must be an integer >= 2")
    c_value = _positive_finite(regularization_c, "regularization_c")
    epsilon = _positive_finite(beta_epsilon, "beta_epsilon")
    if epsilon >= 0.5:
        raise ValueError("beta_epsilon must be smaller than 0.5")

    group_folds, record_folds = _group_fold_assignments(
        dev_rows,
        n_splits=n_splits,
        seed=seed,
    )
    candidate_results: list[dict[str, Any]] = []
    successful: list[tuple[float, int, str]] = []
    for method_index, method in enumerate(selected_methods):
        out_of_fold: dict[str, float] = {}
        failure: str | None = None
        for fold in range(n_splits):
            train = [
                row for row in dev_rows if record_folds[str(row["record_id"])] != fold
            ]
            validation = [
                row for row in dev_rows if record_folds[str(row["record_id"])] == fold
            ]
            try:
                fitted = fit_calibrator(
                    [float(row["raw_score"]) for row in train],
                    [int(row["outcome"]) for row in train],
                    method=method,
                    regularization_c=c_value,
                    beta_epsilon=epsilon,
                )
                predictions = predict_calibrated(
                    fitted,
                    [float(row["raw_score"]) for row in validation],
                )
            except ValueError as exc:
                failure = f"{type(exc).__name__}: {exc}"
                break
            out_of_fold.update(
                {
                    str(row["record_id"]): prediction
                    for row, prediction in zip(
                        validation,
                        predictions,
                        strict=True,
                    )
                }
            )
        if failure is not None:
            candidate_results.append(
                {
                    "method": method,
                    "status": "invalid",
                    "failure": failure,
                    "metrics": None,
                }
            )
            continue
        ordered_probabilities = [out_of_fold[str(row["record_id"])] for row in dev_rows]
        metrics = calibration_metrics(
            ordered_probabilities,
            [int(row["outcome"]) for row in dev_rows],
            bins=bins,
        )
        candidate_results.append(
            {
                "method": method,
                "status": "eligible",
                "failure": None,
                "metrics": metrics,
            }
        )
        successful.append((float(metrics[primary_metric]), method_index, method))
    if not successful:
        raise ValueError("all calibration methods failed deterministic dev CV")
    winner = min(successful)[2]
    fold_rows = [
        {
            "record_id": str(row["record_id"]),
            "group_id": str(row["group_id"]),
            "fold": record_folds[str(row["record_id"])],
        }
        for row in dev_rows
    ]
    core = {
        "schema_version": METHOD_SELECTION_SCHEMA_VERSION,
        "protocol_version": METHOD_SELECTION_PROTOCOL_VERSION,
        "status": "selected_on_dev_only",
        "split_roles_consumed": ["dev"],
        "dev_rows_content_sha256": canonical_sha256(dev_rows),
        "group_folds_sha256": canonical_sha256(
            [
                {"group_id": group, "fold": fold}
                for group, fold in sorted(group_folds.items())
            ]
        ),
        "record_folds_sha256": canonical_sha256(fold_rows),
        "n_splits": n_splits,
        "seed": seed,
        "primary_metric": primary_metric,
        "calibration_bins": bins,
        "selection_direction": "minimize",
        "tie_break_order": list(selected_methods),
        "fit_config": {
            "regularization_c": c_value,
            "beta_epsilon": epsilon,
        },
        "candidates": candidate_results,
        "selected_method": winner,
        "calibration_or_test_rows_consumed": False,
    }
    return {**core, "artifact_sha256": canonical_sha256(core)}


__all__ = [
    "CALIBRATOR_SCHEMA_VERSION",
    "METHOD_SELECTION_SCHEMA_VERSION",
    "SUPPORTED_METHODS",
    "TEST_COHORT_BINDING_SCHEMA_VERSION",
    "TEST_CONSUMPTION_LEDGER_VERSION",
    "TEST_CONSUMPTION_SCHEMA_VERSION",
    "calibration_metrics",
    "canonical_json_dumps",
    "canonical_sha256",
    "fit_calibrator",
    "predict_calibrated",
    "reserve_frozen_test_consumption_v2",
    "reserve_test_cohort_consumption",
    "risk_coverage_metrics",
    "select_calibration_method_dev",
    "test_cohort_sha256",
]
