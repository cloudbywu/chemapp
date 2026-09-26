"""Preregistered, single-use joint v3/v4 NMR comparison.

The runner in this module deliberately keeps candidate scoring and outcome
evaluation in separate phases:

1. validate a pre-existing run specification and roleless candidate pools;
2. call the forward predictor exactly once per complete candidate pool;
3. reuse those predictions for both frozen v3 and v4 scoring;
4. open dev Gold for calibration-method selection and calibration Gold for fit;
5. bind and reserve the roleless test cohort in the canonical v2 ledger;
6. only then JSON-decode the captured test Gold bytes and evaluate once.

The three Gold splits are separate files.  In particular, the test file is only
streamed as opaque bytes for SHA-256 validation before ledger reservation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import tempfile
from typing import Any, Protocol

import numpy as np
from rdkit import rdBase
import scipy
import sklearn

from app.ml.nmr_calibration_v4 import (
    SUPPORTED_METHODS,
    TEST_CONSUMPTION_LEDGER_VERSION,
    TEST_CONSUMPTION_SCHEMA_VERSION,
    calibration_metrics,
    canonical_json_dumps,
    canonical_sha256,
    fit_calibrator,
    predict_calibrated,
    reserve_test_cohort_consumption,
    select_calibration_method_dev,
    test_cohort_sha256,
)
from app.ml.nmr_candidate_scorer_v4 import (
    PROTOCOL_VERSION as SCORER_PROTOCOL_VERSION,
    rank_candidate_evidence,
)
from app.ml.dp5q_runtime_pin import (
    DP5Q_NUMPY_VERSION,
    DP5Q_PANDAS_VERSION,
    DP5Q_PYTHON_VERSION,
    DP5Q_RDKIT_VERSION,
    DP5Q_SCIPY_VERSION,
    DP5Q_SCIKIT_LEARN_VERSION,
    DP5Q_TQDM_VERSION,
    DP5Q_UPSTREAM_SOURCE_BUNDLE_SHA256,
)
from app.ml.nmr_forward import (
    DP5Q_CONFORMER_PREFLIGHT_PROTOCOL_VERSION,
    DP5Q_MEAN_MODEL_SHA256,
    DP5Q_PREPROCESSOR_SHA256,
    DP5Q_REPOSITORY_COMMIT,
    PROTOCOL_VERSION as FORWARD_PROTOCOL_VERSION,
    NMRForwardAdapter,
    NMRForwardConfig,
    dp5q_conformer_policy,
)
from app.ml.nmr_v4_candidate_pools import (
    CANDIDATE_GENERATION_VERSION,
    DEFAULT_MAX_DECOYS,
    DEFAULT_SPLIT_SEED,
    ELIGIBILITY_SCHEMA_VERSION,
    SPLIT_MANIFEST_SCHEMA_VERSION,
    SPLIT_PROTOCOL_VERSION,
    SUMMARY_SCHEMA_VERSION as POOL_SUMMARY_SCHEMA_VERSION,
    _normalise_source_record,
    _validate_eligibility_row,
    implementation_binding,
    load_reviewed_release,
    recompute_frozen_candidate_connected_splits,
)


PIPELINE_VERSION = "chemapp.nmr.v4-joint-experiment.v1"
RUN_SPEC_SCHEMA_VERSION = "chemapp.nmr.v4-run-spec.v1"
RUN_SPEC_VERSION = 1
ROLELESS_POOL_SCHEMA_VERSION = "chemapp.nmr.roleless-candidate-pool.v4"
SEALED_GOLD_SCHEMA_VERSION = "chemapp.nmr.sealed-candidate-gold.v4"
ROLELESS_RANKING_SCHEMA_VERSION = "chemapp.nmr.roleless-ranking.v4"
OUTCOME_ROW_SCHEMA_VERSION = "chemapp.nmr.rank-outcome.v4"
REPORT_SCHEMA_VERSION = "chemapp.nmr.v4-joint-report.v1"
RUN_SPEC_AUDIT_SCHEMA_VERSION = "chemapp.nmr.v4-run-spec-audit.v1"
ARTIFACT_MANIFEST_SCHEMA_VERSION = "chemapp.nmr.v4-artifact-manifest.v1"

SPLITS = ("dev", "calibration", "test")
RANKERS = ("v3", "v4")
MAX_POOL_CANDIDATES = 8
CANONICAL_TEST_CONSUMPTION_LEDGER_RELATIVE_PATH = (
    "docs/nmr-v4-frozen-test-consumption-v2.jsonl"
)
CANONICAL_TEST_CONSUMPTION_LEDGER_PATH = (
    Path(__file__).resolve().parents[3]
    / CANONICAL_TEST_CONSUMPTION_LEDGER_RELATIVE_PATH
)
CONFIDENCE_SEMANTICS = {
    "v3": (
        "diagnostic_candidate_score_margin_not_calibrated_because_v3_uses_"
        "a_non_scalar_lexicographic_rank_key"
    ),
    "v4": "top1_candidate_local_set_similarity",
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

_POOL_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "record_id",
        "split",
        "split_group",
        "spectrum_fingerprint_sha256",
        "nucleus",
        "formula",
        "observed_13c",
        "candidates",
    }
)
_POOL_CANDIDATE_FIELDS = frozenset({"candidate_id", "smiles"})
_GOLD_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "record_id",
        "split",
        "truth_candidate_id",
    }
)
_SPLIT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "record_id",
        "split",
        "split_group",
        "spectrum_fingerprint_sha256",
        "dp5_upstream_structure_overlap",
        "dp5_upstream_scaffold_overlap",
        "base_index_structure_overlap",
        "nmrexp_overlap",
        "candidate_set_sha256",
        "roleless_pool_row_sha256",
    }
)
_ELIGIBILITY_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "record_id",
        "evaluated_before_model_scoring",
        "eligible",
        "reason_codes",
        "base_index_sha256",
        "candidate_generation_version",
        "max_decoys",
        "base_decoy_audit",
        "preflight_audit",
        "selected_hard_decoy_count",
        "base_index_structure_overlap",
    }
)
_POOL_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "outcome_free",
        "model_scores_created",
        "calibrator_fitted",
        "candidate_generation",
        "split_protocol",
        "source_binding",
        "base_index",
        "counts",
        "ineligibility_reason_counts",
        "artifacts",
        "implementation_binding",
        "formal_protocol",
    }
)
_POOL_ARTIFACT_NAMES = frozenset(
    {
        "roleless-pools.jsonl",
        "sealed-gold-dev.jsonl",
        "sealed-gold-calibration.jsonl",
        "sealed-gold-test.jsonl",
        "eligibility.jsonl",
        "split-manifest.jsonl",
    }
)
_RUN_SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "spec_version",
        "run_id",
        "registered_at",
        "pipeline_version",
        "purpose",
        "research_only",
        "claim_boundaries",
        "protocol",
        "parameters",
        "input_sha256",
        "code_sha256",
        "forward_model",
        "host_runtime",
        "bindings",
        "frozen_test",
    }
)


class NMRV4ExperimentError(ValueError):
    """Raised when an experiment input or frozen protocol fails closed."""


class ForwardPredictor(Protocol):
    """Small adapter protocol used by the experiment and its tests."""

    def predict_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        formula: str | None = None,
    ) -> Mapping[str, Any]: ...


Scorer = Callable[
    [Sequence[float], Sequence[Mapping[str, Any]]],
    Mapping[str, Any],
]


def _strict_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    context: str,
) -> None:
    if set(value) != expected:
        difference = sorted(set(value) ^ expected)
        raise NMRV4ExperimentError(
            f"{context} field allowlist mismatch: {difference}"
        )


def _nonempty_text(value: Any, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise NMRV4ExperimentError(f"{field_name} must be non-empty stripped text")
    return value


def _opaque_id(value: Any, field_name: str) -> str:
    text = _nonempty_text(value, field_name)
    if _ID_RE.fullmatch(text) is None:
        raise NMRV4ExperimentError(f"{field_name} violates the opaque-ID contract")
    return text


def _sha256_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise NMRV4ExperimentError(f"{field_name} must be lowercase SHA-256")
    return value


def _utc_timestamp(value: Any, field_name: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise NMRV4ExperimentError(f"{field_name} must be an explicit UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NMRV4ExperimentError(
            f"{field_name} must be an explicit UTC timestamp"
        ) from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
    ):
        raise NMRV4ExperimentError(f"{field_name} must be an explicit UTC timestamp")
    return parsed


def _clock_utc_iso(
    clock: Callable[[], datetime] | None,
    *,
    field_name: str,
) -> str:
    observed = (clock or (lambda: datetime.now(timezone.utc)))()
    if not isinstance(observed, datetime):
        raise NMRV4ExperimentError(f"{field_name} clock must return datetime")
    rendered = observed.astimezone(timezone.utc).isoformat()
    _utc_timestamp(rendered, field_name)
    return rendered


def _finite_shift(value: Any, field_name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not -20.0 <= float(value) <= 300.0
    ):
        raise NMRV4ExperimentError(
            f"{field_name} must be finite and within the supported 13C range"
        )
    return float(value)


def file_sha256(path: str | Path) -> str:
    """Hash a file as opaque bytes without interpreting its contents."""

    digest = hashlib.sha256()
    with Path(path).resolve().open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_opaque_bytes_and_sha256(path: str | Path) -> tuple[bytes, str]:
    """Capture immutable opaque bytes and their hash without JSON decoding."""

    payload = Path(path).resolve().read_bytes()
    return payload, hashlib.sha256(payload).hexdigest()


def _canonical_jsonl_rows(path: str | Path, *, context: str) -> list[dict[str, Any]]:
    resolved = Path(path).resolve()
    try:
        text = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise NMRV4ExperimentError(f"{context} must be UTF-8 JSONL") from exc
    if not text or not text.endswith("\n"):
        raise NMRV4ExperimentError(f"{context} must be non-empty newline-terminated JSONL")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise NMRV4ExperimentError(f"{context}:{line_number}: blank line")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NMRV4ExperimentError(
                f"{context}:{line_number}: invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise NMRV4ExperimentError(
                f"{context}:{line_number}: row must be an object"
            )
        if canonical_json_dumps(value) != line:
            raise NMRV4ExperimentError(
                f"{context}:{line_number}: row is not canonical JSON"
            )
        rows.append(value)
    return rows


def load_roleless_pools(path: str | Path) -> list[dict[str, Any]]:
    """Load and strictly validate frozen roleless candidate pools."""

    raw_rows = _canonical_jsonl_rows(path, context="roleless pools")
    normalized: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    seen_fingerprints: set[str] = set()
    group_splits: dict[str, str] = {}
    release_id: str | None = None

    for index, row in enumerate(raw_rows):
        context = f"roleless pools[{index}]"
        _strict_fields(row, _POOL_FIELDS, context=context)
        if row["schema_version"] != ROLELESS_POOL_SCHEMA_VERSION:
            raise NMRV4ExperimentError(f"{context}.schema_version changed")
        release = _nonempty_text(row["release_id"], f"{context}.release_id")
        if release_id is None:
            release_id = release
        elif release != release_id:
            raise NMRV4ExperimentError("roleless pools mix source releases")
        record_id = _opaque_id(row["record_id"], f"{context}.record_id")
        if record_id in seen_records:
            raise NMRV4ExperimentError(f"duplicate roleless record_id: {record_id}")
        seen_records.add(record_id)
        split = row["split"]
        if split not in SPLITS:
            raise NMRV4ExperimentError(f"{context}.split must be one of {SPLITS}")
        split_group = _opaque_id(row["split_group"], f"{context}.split_group")
        previous_split = group_splits.setdefault(split_group, str(split))
        if previous_split != split:
            raise NMRV4ExperimentError(
                f"split_group {split_group} crosses experiment splits"
            )
        fingerprint = _sha256_text(
            row["spectrum_fingerprint_sha256"],
            f"{context}.spectrum_fingerprint_sha256",
        )
        if fingerprint in seen_fingerprints:
            raise NMRV4ExperimentError(
                f"duplicate spectrum fingerprint: {fingerprint}"
            )
        seen_fingerprints.add(fingerprint)
        if row["nucleus"] != "13C":
            raise NMRV4ExperimentError(f"{context}.nucleus must be 13C")
        formula = row["formula"]
        if formula is not None:
            formula = _nonempty_text(formula, f"{context}.formula")
        observed = row["observed_13c"]
        if not isinstance(observed, list) or not 1 <= len(observed) <= 256:
            raise NMRV4ExperimentError(
                f"{context}.observed_13c must contain 1..256 signals"
            )
        clean_observed = sorted(
            _finite_shift(value, f"{context}.observed_13c[{position}]")
            for position, value in enumerate(observed)
        )
        candidates = row["candidates"]
        if (
            not isinstance(candidates, list)
            or not 2 <= len(candidates) <= MAX_POOL_CANDIDATES
        ):
            raise NMRV4ExperimentError(
                f"{context}.candidates must contain 2..{MAX_POOL_CANDIDATES} rows"
            )
        clean_candidates: list[dict[str, str]] = []
        seen_candidate_ids: set[str] = set()
        for position, candidate in enumerate(candidates):
            candidate_context = f"{context}.candidates[{position}]"
            if not isinstance(candidate, Mapping):
                raise NMRV4ExperimentError(
                    f"{candidate_context} must be an object"
                )
            _strict_fields(
                candidate,
                _POOL_CANDIDATE_FIELDS,
                context=candidate_context,
            )
            candidate_id = _opaque_id(
                candidate["candidate_id"],
                f"{candidate_context}.candidate_id",
            )
            if candidate_id in seen_candidate_ids:
                raise NMRV4ExperimentError(
                    f"{context} has duplicate candidate_id {candidate_id}"
                )
            seen_candidate_ids.add(candidate_id)
            smiles = _nonempty_text(
                candidate["smiles"],
                f"{candidate_context}.smiles",
            )
            if len(smiles) > 4096:
                raise NMRV4ExperimentError(
                    f"{candidate_context}.smiles exceeds the length limit"
                )
            clean_candidates.append(
                {"candidate_id": candidate_id, "smiles": smiles}
            )
        normalized.append(
            {
                "schema_version": ROLELESS_POOL_SCHEMA_VERSION,
                "release_id": release,
                "record_id": record_id,
                "split": split,
                "split_group": split_group,
                "spectrum_fingerprint_sha256": fingerprint,
                "nucleus": "13C",
                "formula": formula,
                "observed_13c": clean_observed,
                "candidates": clean_candidates,
            }
        )

    if not normalized:
        raise NMRV4ExperimentError("roleless pools are empty")
    split_counts = Counter(str(row["split"]) for row in normalized)
    missing = [split for split in SPLITS if split_counts[split] == 0]
    if missing:
        raise NMRV4ExperimentError(f"roleless pools are missing splits: {missing}")
    return sorted(normalized, key=lambda row: str(row["record_id"]))


def _artifact_file_binding(path: str | Path) -> dict[str, Any]:
    payload = Path(path).resolve().read_bytes()
    return {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "rows": payload.count(b"\n"),
    }


def load_pool_summary(
    path: str | Path,
    *,
    roleless_pools_path: str | Path,
    eligibility_path: str | Path,
    sealed_gold_dev_path: str | Path,
    sealed_gold_calibration_path: str | Path,
    sealed_gold_test_path: str | Path,
    split_manifest_path: str | Path,
) -> dict[str, Any]:
    """Validate the outcome-free builder summary and its published artifacts."""

    resolved = Path(path).resolve()
    try:
        text = resolved.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise NMRV4ExperimentError("pool summary must be UTF-8 JSON") from exc
    if not text.endswith("\n") or text.count("\n") != 1:
        raise NMRV4ExperimentError(
            "pool summary must be one canonical newline-terminated JSON object"
        )
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise NMRV4ExperimentError("pool summary contains invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise NMRV4ExperimentError("pool summary must be an object")
    if canonical_json_dumps(value) + "\n" != text:
        raise NMRV4ExperimentError("pool summary is not canonical JSON")
    _strict_fields(value, _POOL_SUMMARY_FIELDS, context="pool summary")
    if (
        value["schema_version"] != POOL_SUMMARY_SCHEMA_VERSION
        or value["outcome_free"] is not True
        or value["model_scores_created"] is not False
        or value["calibrator_fitted"] is not False
    ):
        raise NMRV4ExperimentError("pool summary outcome-free semantics changed")
    release_id = _sha256_text(value["release_id"], "pool summary.release_id")

    candidate_generation = value["candidate_generation"]
    expected_candidate_fields = {
        "version",
        "formula_relation",
        "connectivity_relation",
        "hardness_metric",
        "selection_order",
        "maximum_decoys",
        "preflight_operation",
        "preflight_batch_size",
        "preflight_runtime_sha256",
        "rdkit_version",
    }
    if (
        not isinstance(candidate_generation, Mapping)
        or set(candidate_generation) != expected_candidate_fields
        or candidate_generation["version"] != CANDIDATE_GENERATION_VERSION
        or candidate_generation["formula_relation"] != "exact_canonical_formula"
        or candidate_generation["connectivity_relation"]
        != "different_first_14_inchi_key"
        or candidate_generation["hardness_metric"]
        != "Morgan radius=2 fpSize=2048 Tanimoto to target"
        or candidate_generation["selection_order"]
        != "descending_tanimoto_then_ascending_opaque_candidate_id"
        or candidate_generation["preflight_operation"]
        != "conformer_preflight_only"
        or candidate_generation["preflight_batch_size"] != 4
        or candidate_generation["rdkit_version"]
        != _current_host_runtime()["rdkit"]
        or candidate_generation["maximum_decoys"] != DEFAULT_MAX_DECOYS
    ):
        raise NMRV4ExperimentError("pool summary candidate generation changed")
    _sha256_text(
        candidate_generation["preflight_runtime_sha256"],
        "pool summary.candidate_generation.preflight_runtime_sha256",
    )
    if value["formal_protocol"] != {
        "maximum_decoys": DEFAULT_MAX_DECOYS,
        "split_seed": DEFAULT_SPLIT_SEED,
        "parameter_shopping_prohibited": True,
        "preflighter_injection_prohibited": True,
    }:
        raise NMRV4ExperimentError("pool summary formal protocol changed")

    split_protocol = value["split_protocol"]
    expected_split_protocol = {
        "version": SPLIT_PROTOCOL_VERSION,
        "seed": DEFAULT_SPLIT_SEED,
        "connected_by": [
            "duplicate_molecule_group",
            "connectivity_key",
            "nonempty_murcko_scaffold_smiles",
            "spectrum_fingerprint_sha256",
            "all_selected_candidate_ids",
            "all_selected_candidate_connectivity_keys",
        ],
        "bootstrap_unit": "split_group_connected_component",
        "candidate_identity_isolation": {
            "dimensions": ["candidate_id", "connectivity_key"],
            "cross_split_intersections_must_be_empty": True,
            "shared_identity_requires_same_split_group": True,
        },
        "target_ratios": {"dev": 0.4, "calibration": 0.3, "test": 0.3},
        "overlap_policy": (
            "any NMRexp/DP5 structure/scaffold or base-index structure "
            "overlap forces the whole connected component to dev"
        ),
        "test_requires_all_upstream_and_base_overlap_flags_false": True,
    }
    if (
        not isinstance(split_protocol, Mapping)
        or set(split_protocol) != set(expected_split_protocol)
        or not isinstance(split_protocol.get("seed"), str)
        or not split_protocol["seed"]
        or dict(split_protocol) != expected_split_protocol
    ):
        raise NMRV4ExperimentError("pool summary split protocol changed")

    source_binding = value["source_binding"]
    source_fields = {
        "schema_version",
        "source_schema_version",
        "release_id",
        "current_sha256",
        "records_sha256",
        "records_count",
        "summary_sha256",
        "attribution_sha256",
        "review_kind",
    }
    if (
        not isinstance(source_binding, Mapping)
        or set(source_binding) != source_fields
        or source_binding["schema_version"]
        != "chemapp.nmrsolver-source-binding.v1"
        or source_binding["release_id"] != release_id
        or source_binding["review_kind"]
        != "upstream_manually_curated_benchmark"
        or not isinstance(source_binding["source_schema_version"], str)
        or not source_binding["source_schema_version"]
        or isinstance(source_binding["records_count"], bool)
        or not isinstance(source_binding["records_count"], int)
        or source_binding["records_count"] < 1
    ):
        raise NMRV4ExperimentError("pool summary source binding changed")
    for field_name in (
        "release_id",
        "current_sha256",
        "records_sha256",
        "summary_sha256",
        "attribution_sha256",
    ):
        _sha256_text(
            source_binding[field_name],
            f"pool summary.source_binding.{field_name}",
        )

    base_index = value["base_index"]
    if (
        not isinstance(base_index, Mapping)
        or set(base_index) != {"schema_version", "sha256"}
        or not isinstance(base_index["schema_version"], str)
        or not base_index["schema_version"]
    ):
        raise NMRV4ExperimentError("pool summary base-index binding changed")
    _sha256_text(base_index["sha256"], "pool summary.base_index.sha256")

    counts = value["counts"]
    count_fields = {
        "source_records",
        "eligible_records",
        "ineligible_records",
        "by_split",
        "split_groups",
        "source_dp5_structure_scaffold_independent_records",
        "source_dp5_and_base_structure_independent_records",
        "source_base_index_structure_overlap_records",
        "dp5_structure_overlap_forced_dev",
        "dp5_scaffold_overlap_forced_dev",
        "base_index_structure_overlap_forced_dev",
        "nmrexp_overlap_forced_dev",
    }
    if not isinstance(counts, Mapping) or set(counts) != count_fields:
        raise NMRV4ExperimentError("pool summary counts schema changed")
    scalar_count_fields = count_fields - {"by_split"}
    if any(
        isinstance(counts[field_name], bool)
        or not isinstance(counts[field_name], int)
        or counts[field_name] < 0
        for field_name in scalar_count_fields
    ):
        raise NMRV4ExperimentError("pool summary counts must be non-negative")
    by_split = counts["by_split"]
    if (
        not isinstance(by_split, Mapping)
        or set(by_split) != set(SPLITS)
        or any(
            isinstance(by_split[split], bool)
            or not isinstance(by_split[split], int)
            or by_split[split] < 1
            for split in SPLITS
        )
        or sum(by_split.values()) != counts["eligible_records"]
        or counts["source_records"]
        != counts["eligible_records"] + counts["ineligible_records"]
    ):
        raise NMRV4ExperimentError("pool summary split counts changed")
    reason_counts = value["ineligibility_reason_counts"]
    if not isinstance(reason_counts, Mapping) or any(
        not isinstance(reason, str)
        or not reason
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for reason, count in reason_counts.items()
    ):
        raise NMRV4ExperimentError("pool summary ineligibility counts changed")

    artifacts = value["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != _POOL_ARTIFACT_NAMES:
        raise NMRV4ExperimentError("pool summary artifact allowlist changed")
    for name, binding in artifacts.items():
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"sha256", "bytes", "rows"}
        ):
            raise NMRV4ExperimentError(
                f"pool summary artifact binding changed: {name}"
            )
        _sha256_text(binding["sha256"], f"pool summary.artifacts.{name}.sha256")
        for metric in ("bytes", "rows"):
            if (
                isinstance(binding[metric], bool)
                or not isinstance(binding[metric], int)
                or binding[metric] < 0
            ):
                raise NMRV4ExperimentError(
                    f"pool summary artifact {name}.{metric} is invalid"
                )
    actual_paths = {
        "roleless-pools.jsonl": roleless_pools_path,
        "eligibility.jsonl": eligibility_path,
        "sealed-gold-dev.jsonl": sealed_gold_dev_path,
        "sealed-gold-calibration.jsonl": sealed_gold_calibration_path,
        "sealed-gold-test.jsonl": sealed_gold_test_path,
        "split-manifest.jsonl": split_manifest_path,
    }
    for name, artifact_path in actual_paths.items():
        if artifacts[name] != _artifact_file_binding(artifact_path):
            raise NMRV4ExperimentError(
                f"pool summary artifact does not match frozen bytes: {name}"
            )
    try:
        live_implementation_binding = implementation_binding()
    except ValueError as exc:
        raise NMRV4ExperimentError(
            f"candidate-pool implementation binding failed: {exc}"
        ) from exc
    if value["implementation_binding"] != live_implementation_binding:
        raise NMRV4ExperimentError(
            "candidate-pool implementation differs from frozen summary"
        )
    return json.loads(canonical_json_dumps(value))


def load_split_manifest(
    path: str | Path,
    *,
    pools: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Validate every published split/independence binding before scoring."""

    raw_rows = _canonical_jsonl_rows(path, context="v4 split manifest")
    pools_by_id = {str(row["record_id"]): row for row in pools}
    rows_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(raw_rows):
        context = f"v4 split manifest[{index}]"
        _strict_fields(row, _SPLIT_MANIFEST_FIELDS, context=context)
        if row["schema_version"] != SPLIT_MANIFEST_SCHEMA_VERSION:
            raise NMRV4ExperimentError(f"{context}.schema_version changed")
        release_id = _sha256_text(row["release_id"], f"{context}.release_id")
        record_id = _opaque_id(row["record_id"], f"{context}.record_id")
        if record_id in rows_by_id:
            raise NMRV4ExperimentError(
                f"duplicate split-manifest record_id: {record_id}"
            )
        if record_id not in pools_by_id:
            raise NMRV4ExperimentError(
                f"split manifest has unknown record_id: {record_id}"
            )
        pool = pools_by_id[record_id]
        if (
            release_id != pool["release_id"]
            or row["split"] != pool["split"]
            or row["split_group"] != pool["split_group"]
            or row["spectrum_fingerprint_sha256"]
            != pool["spectrum_fingerprint_sha256"]
        ):
            raise NMRV4ExperimentError(
                f"{record_id}: split-manifest public binding differs from pool"
            )
        _opaque_id(row["split_group"], f"{context}.split_group")
        _sha256_text(
            row["spectrum_fingerprint_sha256"],
            f"{context}.spectrum_fingerprint_sha256",
        )
        for field_name in (
            "candidate_set_sha256",
            "roleless_pool_row_sha256",
        ):
            _sha256_text(row[field_name], f"{context}.{field_name}")
        overlap_fields = (
            "dp5_upstream_structure_overlap",
            "dp5_upstream_scaffold_overlap",
            "base_index_structure_overlap",
            "nmrexp_overlap",
        )
        if any(not isinstance(row[field_name], bool) for field_name in overlap_fields):
            raise NMRV4ExperimentError(
                f"{record_id}: split-manifest overlap flags must be boolean"
            )
        expected_candidate_set_sha = canonical_sha256(
            sorted(
                [
                    {
                        "candidate_id": str(candidate["candidate_id"]),
                        "smiles": str(candidate["smiles"]),
                    }
                    for candidate in pool["candidates"]
                ],
                key=lambda candidate: candidate["candidate_id"],
            )
        )
        if row["candidate_set_sha256"] != expected_candidate_set_sha:
            raise NMRV4ExperimentError(
                f"{record_id}: candidate-set publication binding changed"
            )
        if row["roleless_pool_row_sha256"] != canonical_sha256(pool):
            raise NMRV4ExperimentError(
                f"{record_id}: roleless-pool publication binding changed"
            )
        if row["split"] == "test" and any(
            row[field_name] for field_name in overlap_fields
        ):
            raise NMRV4ExperimentError(
                f"{record_id}: final test overlap flags must all be false"
            )
        rows_by_id[record_id] = json.loads(canonical_json_dumps(row))
    if set(rows_by_id) != set(pools_by_id):
        missing = sorted(set(pools_by_id) - set(rows_by_id))
        raise NMRV4ExperimentError(
            f"split manifest omits roleless records: {missing[:10]}"
        )
    return [rows_by_id[record_id] for record_id in sorted(rows_by_id)]


def load_eligibility(
    path: str | Path,
    *,
    release_id: str,
    base_index_sha256: str,
    maximum_decoys: int,
) -> list[dict[str, Any]]:
    """Strictly parse the builder's complete pre-score eligibility audit."""

    raw_rows = _canonical_jsonl_rows(path, context="v4 pool eligibility")
    rows_by_id: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(raw_rows):
        context = f"v4 pool eligibility[{index}]"
        _strict_fields(row, _ELIGIBILITY_FIELDS, context=context)
        try:
            _validate_eligibility_row(row)
        except ValueError as exc:
            raise NMRV4ExperimentError(
                f"{context}: builder eligibility validation failed: {exc}"
            ) from exc
        if (
            row["schema_version"] != ELIGIBILITY_SCHEMA_VERSION
            or row["release_id"] != release_id
            or row["base_index_sha256"] != base_index_sha256
            or row["candidate_generation_version"]
            != CANDIDATE_GENERATION_VERSION
            or row["max_decoys"] != maximum_decoys
            or not isinstance(row["base_index_structure_overlap"], bool)
        ):
            raise NMRV4ExperimentError(
                f"{context}: eligibility release/config binding changed"
            )
        record_id = _opaque_id(row["record_id"], f"{context}.record_id")
        if record_id in rows_by_id:
            raise NMRV4ExperimentError(
                f"duplicate eligibility record_id: {record_id}"
            )
        selected_count = row["selected_hard_decoy_count"]
        if (
            isinstance(selected_count, bool)
            or not isinstance(selected_count, int)
            or not 0 <= selected_count <= maximum_decoys
            or not isinstance(row["base_decoy_audit"], Mapping)
            or not isinstance(row["preflight_audit"], Mapping)
        ):
            raise NMRV4ExperimentError(
                f"{context}: eligibility audit payload is invalid"
            )
        rows_by_id[record_id] = json.loads(canonical_json_dumps(row))
    if not rows_by_id:
        raise NMRV4ExperimentError("v4 pool eligibility is empty")
    return [rows_by_id[record_id] for record_id in sorted(rows_by_id)]


def validate_candidate_publication(
    *,
    pools: Sequence[Mapping[str, Any]],
    split_manifest: Sequence[Mapping[str, Any]],
    eligibility: Sequence[Mapping[str, Any]],
    pool_summary: Mapping[str, Any],
    source_records: Sequence[Mapping[str, Any]],
    source_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Recompute publication counts from the independently verified source."""

    if dict(source_binding) != dict(pool_summary["source_binding"]):
        raise NMRV4ExperimentError(
            "verified source CURRENT binding differs from pool summary"
        )
    release_id = str(pool_summary["release_id"])
    if source_binding.get("release_id") != release_id:
        raise NMRV4ExperimentError("verified source release_id changed")
    normalized_source: dict[str, dict[str, Any]] = {}
    for index, raw_record in enumerate(source_records):
        if not isinstance(raw_record, Mapping):
            raise NMRV4ExperimentError(
                f"verified source record[{index}] must be an object"
            )
        try:
            normalized = _normalise_source_record(raw_record)
        except ValueError as exc:
            raise NMRV4ExperimentError(
                f"verified source record[{index}] failed normalization: {exc}"
            ) from exc
        record_id = str(normalized["record_id"])
        if record_id in normalized_source:
            raise NMRV4ExperimentError(
                f"verified source duplicates record_id: {record_id}"
            )
        normalized_source[record_id] = normalized

    pools_by_id = {str(row["record_id"]): row for row in pools}
    manifest_by_id = {str(row["record_id"]): row for row in split_manifest}
    eligibility_by_id = {str(row["record_id"]): row for row in eligibility}
    if set(eligibility_by_id) != set(normalized_source):
        missing = sorted(set(normalized_source) - set(eligibility_by_id))
        unexpected = sorted(set(eligibility_by_id) - set(normalized_source))
        raise NMRV4ExperimentError(
            "eligibility/source record sets differ; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    eligible_ids = {
        record_id
        for record_id, row in eligibility_by_id.items()
        if row["eligible"] is True
    }
    if eligible_ids != set(pools_by_id) or set(manifest_by_id) != set(pools_by_id):
        raise NMRV4ExperimentError(
            "eligible, roleless, and split-manifest record sets differ"
        )

    preflight_runtime = str(
        pool_summary["candidate_generation"]["preflight_runtime_sha256"]
    )
    split_recomputation_rows: list[dict[str, Any]] = []
    for record_id, eligibility_row in eligibility_by_id.items():
        source_row = normalized_source[record_id]
        if eligibility_row["eligible"]:
            pool = pools_by_id[record_id]
            manifest = manifest_by_id[record_id]
            if (
                pool["formula"] != source_row["formula"]
                or list(pool["observed_13c"])
                != list(source_row["observed_13c"])
                or pool["spectrum_fingerprint_sha256"]
                != source_row["spectrum_fingerprint_sha256"]
            ):
                raise NMRV4ExperimentError(
                    f"{record_id}: roleless spectrum/formula binding differs "
                    "from verified source"
                )
            source_candidate = source_row.get("candidate")
            if not isinstance(source_candidate, Mapping):
                raise NMRV4ExperimentError(
                    f"{record_id}: verified source candidate is missing"
                )
            expected_truth_candidate = {
                "candidate_id": str(source_candidate["candidate_id"]),
                "smiles": str(source_candidate["smiles"]),
            }
            if expected_truth_candidate not in pool["candidates"]:
                raise NMRV4ExperimentError(
                    f"{record_id}: source structure is absent from candidate pool"
                )
            if eligibility_row["selected_hard_decoy_count"] != (
                len(pool["candidates"]) - 1
            ):
                raise NMRV4ExperimentError(
                    f"{record_id}: selected hard-decoy count differs from pool"
                )
            row_runtime = eligibility_row["preflight_audit"].get(
                "preflight_runtime_sha256"
            )
            if row_runtime != preflight_runtime:
                raise NMRV4ExperimentError(
                    f"{record_id}: preflight runtime differs from pool summary"
                )
            expected_overlap = {
                "dp5_upstream_structure_overlap": bool(
                    source_row["dp5_upstream_structure_overlap"]
                ),
                "dp5_upstream_scaffold_overlap": bool(
                    source_row["dp5_upstream_scaffold_overlap"]
                ),
                "base_index_structure_overlap": bool(
                    eligibility_row["base_index_structure_overlap"]
                ),
                "nmrexp_overlap": bool(source_row["nmrexp_overlap"]),
            }
            if any(
                manifest[field_name] != expected_value
                for field_name, expected_value in expected_overlap.items()
            ):
                raise NMRV4ExperimentError(
                    f"{record_id}: split overlap flags differ from source/eligibility"
                )
            if any(expected_overlap.values()) and manifest["split"] != "dev":
                raise NMRV4ExperimentError(
                    f"{record_id}: overlap-bearing component was not forced to dev"
                )
            split_recomputation_rows.append(
                {
                    "record_id": record_id,
                    "duplicate_molecule_group": source_row[
                        "duplicate_molecule_group"
                    ],
                    "connectivity_key": source_row["connectivity_key"],
                    "scaffold_smiles": source_row["scaffold_smiles"],
                    "spectrum_fingerprint_sha256": source_row[
                        "spectrum_fingerprint_sha256"
                    ],
                    **expected_overlap,
                    "selected_candidates": list(pool["candidates"]),
                }
            )

    try:
        recomputed_assignments = recompute_frozen_candidate_connected_splits(
            split_recomputation_rows
        )
    except ValueError as exc:
        raise NMRV4ExperimentError(
            f"candidate-connected split recomputation failed: {exc}"
        ) from exc
    if set(recomputed_assignments) != set(pools_by_id):
        raise NMRV4ExperimentError(
            "candidate-connected split recomputation record set changed"
        )
    for record_id, assignment in recomputed_assignments.items():
        pool = pools_by_id[record_id]
        manifest = manifest_by_id[record_id]
        if (
            assignment.get("split") != pool["split"]
            or assignment.get("split_group") != pool["split_group"]
            or assignment.get("split") != manifest["split"]
            or assignment.get("split_group") != manifest["split_group"]
        ):
            raise NMRV4ExperimentError(
                f"{record_id}: frozen candidate-connected split/group differs "
                "from recomputation"
            )

    counts = pool_summary["counts"]
    by_split = Counter(str(row["split"]) for row in pools)
    source_dp5_independent = sum(
        not row["dp5_upstream_structure_overlap"]
        and not row["dp5_upstream_scaffold_overlap"]
        for row in normalized_source.values()
    )
    source_dp5_base_independent = sum(
        not normalized_source[record_id]["dp5_upstream_structure_overlap"]
        and not normalized_source[record_id]["dp5_upstream_scaffold_overlap"]
        and not eligibility_by_id[record_id]["base_index_structure_overlap"]
        for record_id in normalized_source
    )
    recomputed_counts = {
        "source_records": len(normalized_source),
        "eligible_records": len(pools),
        "ineligible_records": len(normalized_source) - len(pools),
        "by_split": {split: by_split[split] for split in SPLITS},
        "split_groups": len({str(row["split_group"]) for row in pools}),
        "source_dp5_structure_scaffold_independent_records": (
            source_dp5_independent
        ),
        "source_dp5_and_base_structure_independent_records": (
            source_dp5_base_independent
        ),
        "source_base_index_structure_overlap_records": sum(
            bool(row["base_index_structure_overlap"])
            for row in eligibility
        ),
        "dp5_structure_overlap_forced_dev": sum(
            bool(row["dp5_upstream_structure_overlap"])
            for row in split_manifest
        ),
        "dp5_scaffold_overlap_forced_dev": sum(
            bool(row["dp5_upstream_scaffold_overlap"])
            for row in split_manifest
        ),
        "base_index_structure_overlap_forced_dev": sum(
            bool(row["base_index_structure_overlap"])
            for row in split_manifest
        ),
        "nmrexp_overlap_forced_dev": sum(
            bool(row["nmrexp_overlap"]) for row in split_manifest
        ),
    }
    if dict(counts) != recomputed_counts:
        raise NMRV4ExperimentError(
            "pool summary counts differ from source/eligibility/splits"
        )
    reason_counts = dict(
        sorted(
            Counter(
                reason
                for row in eligibility
                for reason in row["reason_codes"]
            ).items()
        )
    )
    if dict(pool_summary["ineligibility_reason_counts"]) != reason_counts:
        raise NMRV4ExperimentError(
            "pool summary ineligibility reason counts changed"
        )
    test_pool_ids = {
        str(row["record_id"]) for row in pools if row["split"] == "test"
    }
    test_manifest_ids = {
        str(row["record_id"])
        for row in split_manifest
        if row["split"] == "test"
    }
    if test_pool_ids != test_manifest_ids:
        raise NMRV4ExperimentError(
            "final test pools are not the complete frozen manifest test set"
        )
    audit_core = {
        "source_release_id": release_id,
        "source_records_sha256": source_binding["records_sha256"],
        "source_record_count": len(normalized_source),
        "eligible_record_count": len(pools),
        "test_record_count": len(test_pool_ids),
        "counts_recomputed": True,
        "reason_counts_recomputed": True,
        "source_rows_rebound": True,
        "candidate_connected_splits_recomputed": True,
    }
    return {
        **audit_core,
        "publication_audit_sha256": canonical_sha256(audit_core),
    }


def _pool_bindings(
    pools: Sequence[Mapping[str, Any]],
    *,
    pool_summary: Mapping[str, Any],
    split_manifest: Sequence[Mapping[str, Any]],
    publication_audit: Mapping[str, Any],
) -> dict[str, Any]:
    release_ids = {str(row["release_id"]) for row in pools}
    if len(release_ids) != 1:
        raise NMRV4ExperimentError("roleless pools must bind exactly one release")
    release_id = next(iter(release_ids))
    identity_rows = sorted(
        (
            {
                "record_id": str(row["record_id"]),
                "split": str(row["split"]),
                "split_group": str(row["split_group"]),
                "spectrum_fingerprint_sha256": str(
                    row["spectrum_fingerprint_sha256"]
                ),
            }
            for row in pools
        ),
        key=lambda row: row["record_id"],
    )
    split_counts = {
        split: sum(row["split"] == split for row in identity_rows)
        for split in SPLITS
    }
    split_bindings = {
        split: canonical_sha256(
            [row for row in identity_rows if row["split"] == split]
        )
        for split in SPLITS
    }
    test_rows = [
        {
            "record_id": row["record_id"],
            "spectrum_fingerprint_sha256": row[
                "spectrum_fingerprint_sha256"
            ],
            "split_group": row["split_group"],
        }
        for row in identity_rows
        if row["split"] == "test"
    ]
    source_binding = pool_summary["source_binding"]
    candidate_generation = pool_summary["candidate_generation"]
    split_protocol = pool_summary["split_protocol"]
    base_index = pool_summary["base_index"]
    return {
        "release_id": release_id,
        "roleless_record_count": len(identity_rows),
        "roleless_record_binding_sha256": canonical_sha256(identity_rows),
        "split_counts": split_counts,
        "split_binding_sha256": split_bindings,
        "test_cohort_sha256": test_cohort_sha256(release_id, test_rows),
        "pool_summary_content_sha256": canonical_sha256(pool_summary),
        "split_manifest_content_sha256": canonical_sha256(split_manifest),
        "source_release_records_sha256": source_binding["records_sha256"],
        "source_release_records_count": source_binding["records_count"],
        "base_index_sha256": base_index["sha256"],
        "preflight_runtime_sha256": candidate_generation[
            "preflight_runtime_sha256"
        ],
        "candidate_generation_sha256": canonical_sha256(candidate_generation),
        "split_protocol_sha256": canonical_sha256(split_protocol),
        "test_independence_binding_sha256": canonical_sha256(
            [
                {
                    "record_id": row["record_id"],
                    "dp5_upstream_structure_overlap": row[
                        "dp5_upstream_structure_overlap"
                    ],
                    "dp5_upstream_scaffold_overlap": row[
                        "dp5_upstream_scaffold_overlap"
                    ],
                    "base_index_structure_overlap": row[
                        "base_index_structure_overlap"
                    ],
                    "nmrexp_overlap": row["nmrexp_overlap"],
                }
                for row in split_manifest
                if row["split"] == "test"
            ]
        ),
        "publication_audit_sha256": publication_audit[
            "publication_audit_sha256"
        ],
    }


def _implementation_paths() -> dict[str, Path]:
    module_path = Path(__file__).resolve()
    backend = module_path.parents[2]
    return {
        "experiment_module_file_sha256": module_path,
        "candidate_scorer_module_file_sha256": (
            module_path.with_name("nmr_candidate_scorer_v4.py")
        ),
        "calibration_module_file_sha256": (
            module_path.with_name("nmr_calibration_v4.py")
        ),
        "candidate_pool_builder_module_file_sha256": (
            module_path.with_name("nmr_v4_candidate_pools.py")
        ),
        "forward_adapter_module_file_sha256": (
            module_path.with_name("nmr_forward.py")
        ),
        "dp5q_runtime_pin_module_file_sha256": (
            module_path.with_name("dp5q_runtime_pin.py")
        ),
        "dp5q_sidecar_script_file_sha256": (
            backend / "scripts" / "dp5q_sidecar.py"
        ),
        "runner_script_file_sha256": (
            backend / "scripts" / "run_nmr_v4_experiment.py"
        ),
    }


def _code_hashes() -> dict[str, str]:
    paths = _implementation_paths()
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise NMRV4ExperimentError(
            f"v4 implementation files are missing: {missing}"
        )
    return {name: file_sha256(path) for name, path in paths.items()}


def _expected_forward_model() -> dict[str, Any]:
    return {
        "name": "DP5q-CASCADE-mean",
        "repository_commit": DP5Q_REPOSITORY_COMMIT,
        "mean_model_sha256": DP5Q_MEAN_MODEL_SHA256,
        "preprocessor_sha256": DP5Q_PREPROCESSOR_SHA256,
        "upstream_source_bundle_sha256": DP5Q_UPSTREAM_SOURCE_BUNDLE_SHA256,
        "runtime_pin": {
            "python": DP5Q_PYTHON_VERSION,
            "tensorflow": "2.14.0",
            "keras": "2.14.0",
            "numpy": DP5Q_NUMPY_VERSION,
            "scipy": DP5Q_SCIPY_VERSION,
            "pandas": DP5Q_PANDAS_VERSION,
            "rdkit": DP5Q_RDKIT_VERSION,
            "scikit_learn": DP5Q_SCIKIT_LEARN_VERSION,
            "tqdm": DP5Q_TQDM_VERSION,
        },
    }


def _current_host_runtime() -> dict[str, str]:
    """Attest the interpreter used for pool/scorer/calibration computation."""

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "rdkit": rdBase.rdkitVersion,
    }


def default_parameters() -> dict[str, Any]:
    """Return the fixed default parameters written into a preregistration."""

    return {
        "calibration_methods": list(SUPPORTED_METHODS),
        "calibration_cv_folds": 3,
        "calibration_cv_seed": 20260729,
        "calibration_primary_metric": "log_loss",
        "regularization_c": 1.0,
        "beta_epsilon": 1e-6,
        "calibration_bins": 10,
        "minimum_class_count": {
            "dev": 2,
            "calibration": 2,
            "test": 2,
        },
        "bootstrap_replicates": 2000,
        "bootstrap_seed": 20260729,
        "maximum_candidate_pool_size": MAX_POOL_CANDIDATES,
    }


def _validate_parameters(value: Any) -> dict[str, Any]:
    expected_fields = {
        "calibration_methods",
        "calibration_cv_folds",
        "calibration_cv_seed",
        "calibration_primary_metric",
        "regularization_c",
        "beta_epsilon",
        "calibration_bins",
        "minimum_class_count",
        "bootstrap_replicates",
        "bootstrap_seed",
        "maximum_candidate_pool_size",
    }
    if not isinstance(value, Mapping) or set(value) != expected_fields:
        raise NMRV4ExperimentError("run-spec parameters schema changed")
    methods = value["calibration_methods"]
    if (
        not isinstance(methods, list)
        or methods != list(SUPPORTED_METHODS)
    ):
        raise NMRV4ExperimentError(
            "calibration_methods must preserve the frozen three-method order"
        )
    integers = (
        "calibration_cv_folds",
        "calibration_cv_seed",
        "calibration_bins",
        "bootstrap_replicates",
        "bootstrap_seed",
        "maximum_candidate_pool_size",
    )
    if any(
        isinstance(value[name], bool) or not isinstance(value[name], int)
        for name in integers
    ):
        raise NMRV4ExperimentError("integer run-spec parameters are invalid")
    if not 2 <= value["calibration_cv_folds"] <= 20:
        raise NMRV4ExperimentError("calibration_cv_folds must be in 2..20")
    if value["calibration_primary_metric"] not in {"log_loss", "brier", "ece"}:
        raise NMRV4ExperimentError("unsupported calibration_primary_metric")
    if value["calibration_bins"] < 2:
        raise NMRV4ExperimentError("calibration_bins must be >= 2")
    if not 1_000 <= value["bootstrap_replicates"] <= 100_000:
        raise NMRV4ExperimentError(
            "bootstrap_replicates must be in 1000..100000"
        )
    if value["maximum_candidate_pool_size"] != MAX_POOL_CANDIDATES:
        raise NMRV4ExperimentError("maximum_candidate_pool_size changed")
    for name in ("regularization_c", "beta_epsilon"):
        number = value[name]
        if (
            isinstance(number, bool)
            or not isinstance(number, (int, float))
            or not math.isfinite(float(number))
            or float(number) <= 0.0
        ):
            raise NMRV4ExperimentError(f"{name} must be finite and positive")
    if float(value["beta_epsilon"]) >= 0.5:
        raise NMRV4ExperimentError("beta_epsilon must be smaller than 0.5")
    minimum = value["minimum_class_count"]
    if (
        not isinstance(minimum, Mapping)
        or set(minimum) != set(SPLITS)
        or any(
            isinstance(minimum[split], bool)
            or not isinstance(minimum[split], int)
            or minimum[split] < 1
            for split in SPLITS
        )
    ):
        raise NMRV4ExperimentError("minimum_class_count schema is invalid")
    return json.loads(canonical_json_dumps(value))


def _build_run_spec_template_core(
    *,
    roleless_pools_path: str | Path,
    pool_summary_path: str | Path,
    eligibility_path: str | Path,
    split_manifest_path: str | Path,
    source_current_path: str | Path,
    sealed_gold_dev_path: str | Path,
    sealed_gold_calibration_path: str | Path,
    sealed_gold_test_path: str | Path,
    run_id: str,
    parameters: Mapping[str, Any] | None = None,
    _clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Create a complete outcome-blind preregistration from frozen file hashes.

    Gold files are not JSON-decoded by this function.  They are opaque byte
    inputs whose hashes are bound into the returned specification.
    """

    clean_run_id = _opaque_id(run_id, "run_id")
    registered_at = _clock_utc_iso(_clock, field_name="registered_at")
    pools = load_roleless_pools(roleless_pools_path)
    pool_summary = load_pool_summary(
        pool_summary_path,
        roleless_pools_path=roleless_pools_path,
        eligibility_path=eligibility_path,
        sealed_gold_dev_path=sealed_gold_dev_path,
        sealed_gold_calibration_path=sealed_gold_calibration_path,
        sealed_gold_test_path=sealed_gold_test_path,
        split_manifest_path=split_manifest_path,
    )
    split_manifest = load_split_manifest(
        split_manifest_path,
        pools=pools,
    )
    try:
        source_records, source_binding = load_reviewed_release(
            source_current_path
        )
    except ValueError as exc:
        raise NMRV4ExperimentError(
            f"verified source release failed validation: {exc}"
        ) from exc
    if pool_summary["release_id"] != pools[0]["release_id"]:
        raise NMRV4ExperimentError("pool summary release differs from roleless pools")
    eligibility = load_eligibility(
        eligibility_path,
        release_id=str(pool_summary["release_id"]),
        base_index_sha256=str(pool_summary["base_index"]["sha256"]),
        maximum_decoys=int(
            pool_summary["candidate_generation"]["maximum_decoys"]
        ),
    )
    publication_audit = validate_candidate_publication(
        pools=pools,
        split_manifest=split_manifest,
        eligibility=eligibility,
        pool_summary=pool_summary,
        source_records=source_records,
        source_binding=source_binding,
    )
    clean_parameters = _validate_parameters(
        default_parameters() if parameters is None else parameters
    )
    bindings = _pool_bindings(
        pools,
        pool_summary=pool_summary,
        split_manifest=split_manifest,
        publication_audit=publication_audit,
    )
    return {
        "schema_version": RUN_SPEC_SCHEMA_VERSION,
        "spec_version": RUN_SPEC_VERSION,
        "run_id": clean_run_id,
        "registered_at": registered_at,
        "pipeline_version": PIPELINE_VERSION,
        "purpose": (
            "preregistered_single_use_retrospective_joint_v3_v4_comparison"
        ),
        "research_only": True,
        "claim_boundaries": {
            "external_equation_reproduction": True,
            "nmr_solver_method_independence": False,
            "source_doi_per_row_available": False,
            "cryptographically_blinded": False,
            "public_source_truth_derivable": True,
            "evaluation_design": (
                "preregistered_single_use_retrospective_evaluation"
            ),
            "externally_signed": False,
            "artifact_provenance": "self_attested_local_build",
        },
        "protocol": {
            "forward_prediction_calls": "exactly_once_per_roleless_pool",
            "same_predictions_reused_for_v3_and_v4": True,
            "scorer_truth_access": False,
            "method_selection_split": "dev_only",
            "final_calibrator_fit_split": "calibration_only",
            "raw_confidence_semantics": dict(CONFIDENCE_SEMANTICS),
            "beta_calibration_candidate_rankers": ["v4"],
            "test_predictions_and_calibrators_frozen_before_test_gold_decode": (
                True
            ),
            "primary_endpoint": (
                "test_top1_accuracy_difference_v4_minus_v3"
            ),
            "primary_test": "two_sided_cluster_sign_flip",
            "null_hypothesis": (
                "candidate_connected_group_v4_minus_v3_top1_deltas_are_"
                "sign_exchangeable_around_zero"
            ),
            "alpha": 0.05,
            "multiplicity": (
                "one_primary_endpoint_no_adjustment_all_other_metrics_"
                "exploratory"
            ),
        },
        "parameters": clean_parameters,
        "input_sha256": {
            "roleless_pools_file_sha256": file_sha256(roleless_pools_path),
            "pool_summary_file_sha256": file_sha256(pool_summary_path),
            "eligibility_file_sha256": file_sha256(eligibility_path),
            "split_manifest_file_sha256": file_sha256(split_manifest_path),
            "source_current_file_sha256": file_sha256(source_current_path),
            "sealed_gold_dev_file_sha256": file_sha256(sealed_gold_dev_path),
            "sealed_gold_calibration_file_sha256": file_sha256(
                sealed_gold_calibration_path
            ),
            "sealed_gold_test_file_sha256": file_sha256(sealed_gold_test_path),
        },
        "code_sha256": _code_hashes(),
        "forward_model": _expected_forward_model(),
        "host_runtime": _current_host_runtime(),
        "bindings": bindings,
        "frozen_test": {
            "consumption_policy": "single_use_canonical_v2_ledger",
            "canonical_ledger_relative_path": (
                CANONICAL_TEST_CONSUMPTION_LEDGER_RELATIVE_PATH
            ),
            "ledger_schema_version": TEST_CONSUMPTION_SCHEMA_VERSION,
            "ledger_version": TEST_CONSUMPTION_LEDGER_VERSION,
            "cohort_identity_is_model_independent": True,
            "test_gold_before_reservation": "opaque_bytes_sha256_only",
            "json_decode_test_gold_only_after_ledger_reservation": True,
            "reuse_for_model_calibrator_or_threshold_selection": False,
        },
    }


def build_run_spec_template(
    *,
    roleless_pools_path: str | Path,
    pool_summary_path: str | Path,
    eligibility_path: str | Path,
    split_manifest_path: str | Path,
    source_current_path: str | Path,
    sealed_gold_dev_path: str | Path,
    sealed_gold_calibration_path: str | Path,
    sealed_gold_test_path: str | Path,
    run_id: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a canonical preregistration with an internal UTC timestamp."""

    return _build_run_spec_template_core(
        roleless_pools_path=roleless_pools_path,
        pool_summary_path=pool_summary_path,
        eligibility_path=eligibility_path,
        split_manifest_path=split_manifest_path,
        source_current_path=source_current_path,
        sealed_gold_dev_path=sealed_gold_dev_path,
        sealed_gold_calibration_path=sealed_gold_calibration_path,
        sealed_gold_test_path=sealed_gold_test_path,
        run_id=run_id,
        parameters=parameters,
        _clock=None,
    )


def load_run_spec(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NMRV4ExperimentError("run-spec must be one UTF-8 JSON object") from exc
    if not isinstance(value, dict):
        raise NMRV4ExperimentError("run-spec must be one JSON object")
    return value


def validate_run_spec(
    spec: Mapping[str, Any],
    *,
    run_spec_path: str | Path,
    roleless_pools_path: str | Path,
    pool_summary_path: str | Path,
    eligibility_path: str | Path,
    split_manifest_path: str | Path,
    source_current_path: str | Path,
    sealed_gold_dev_path: str | Path,
    sealed_gold_calibration_path: str | Path,
    sealed_gold_test_path: str | Path,
    pools: Sequence[Mapping[str, Any]],
    pool_summary: Mapping[str, Any],
    split_manifest: Sequence[Mapping[str, Any]],
    publication_audit: Mapping[str, Any],
    run_at: str,
) -> dict[str, Any]:
    """Strictly validate a pre-existing preregistration against live hashes."""

    _strict_fields(spec, _RUN_SPEC_FIELDS, context="run-spec")
    failures: list[str] = []
    if spec["schema_version"] != RUN_SPEC_SCHEMA_VERSION:
        failures.append("schema_version")
    if spec["spec_version"] != RUN_SPEC_VERSION:
        failures.append("spec_version")
    if spec["pipeline_version"] != PIPELINE_VERSION:
        failures.append("pipeline_version")
    if spec["purpose"] != (
        "preregistered_single_use_retrospective_joint_v3_v4_comparison"
    ):
        failures.append("purpose")
    if spec["research_only"] is not True:
        failures.append("research_only")
    try:
        _opaque_id(spec["run_id"], "run-spec.run_id")
        registered = _utc_timestamp(spec["registered_at"], "registered_at")
        started = _utc_timestamp(run_at, "run_at")
        if registered >= started:
            failures.append("registration_must_precede_run")
    except NMRV4ExperimentError as exc:
        failures.append(str(exc))
    expected_claims = {
        "external_equation_reproduction": True,
        "nmr_solver_method_independence": False,
        "source_doi_per_row_available": False,
        "cryptographically_blinded": False,
        "public_source_truth_derivable": True,
        "evaluation_design": (
            "preregistered_single_use_retrospective_evaluation"
        ),
        "externally_signed": False,
        "artifact_provenance": "self_attested_local_build",
    }
    if spec["claim_boundaries"] != expected_claims:
        failures.append("claim_boundaries")
    expected_protocol = {
        "forward_prediction_calls": "exactly_once_per_roleless_pool",
        "same_predictions_reused_for_v3_and_v4": True,
        "scorer_truth_access": False,
        "method_selection_split": "dev_only",
        "final_calibrator_fit_split": "calibration_only",
        "raw_confidence_semantics": dict(CONFIDENCE_SEMANTICS),
        "beta_calibration_candidate_rankers": ["v4"],
        "test_predictions_and_calibrators_frozen_before_test_gold_decode": (
            True
        ),
        "primary_endpoint": "test_top1_accuracy_difference_v4_minus_v3",
        "primary_test": "two_sided_cluster_sign_flip",
        "null_hypothesis": (
            "candidate_connected_group_v4_minus_v3_top1_deltas_are_"
            "sign_exchangeable_around_zero"
        ),
        "alpha": 0.05,
        "multiplicity": (
            "one_primary_endpoint_no_adjustment_all_other_metrics_exploratory"
        ),
    }
    if spec["protocol"] != expected_protocol:
        failures.append("protocol")
    try:
        parameters = _validate_parameters(spec["parameters"])
    except NMRV4ExperimentError as exc:
        parameters = {}
        failures.append(str(exc))
    expected_inputs = {
        "roleless_pools_file_sha256": file_sha256(roleless_pools_path),
        "pool_summary_file_sha256": file_sha256(pool_summary_path),
        "eligibility_file_sha256": file_sha256(eligibility_path),
        "split_manifest_file_sha256": file_sha256(split_manifest_path),
        "source_current_file_sha256": file_sha256(source_current_path),
        "sealed_gold_dev_file_sha256": file_sha256(sealed_gold_dev_path),
        "sealed_gold_calibration_file_sha256": file_sha256(
            sealed_gold_calibration_path
        ),
        "sealed_gold_test_file_sha256": file_sha256(sealed_gold_test_path),
    }
    if spec["input_sha256"] != expected_inputs:
        failures.append("input_sha256")
    expected_code = _code_hashes()
    if spec["code_sha256"] != expected_code:
        failures.append("code_sha256")
    expected_forward_model = _expected_forward_model()
    if spec["forward_model"] != expected_forward_model:
        failures.append("forward_model")
    expected_host_runtime = _current_host_runtime()
    if spec["host_runtime"] != expected_host_runtime:
        failures.append("host_runtime")
    expected_bindings = _pool_bindings(
        pools,
        pool_summary=pool_summary,
        split_manifest=split_manifest,
        publication_audit=publication_audit,
    )
    if spec["bindings"] != expected_bindings:
        failures.append("bindings")
    expected_frozen_test = {
        "consumption_policy": "single_use_canonical_v2_ledger",
        "canonical_ledger_relative_path": (
            CANONICAL_TEST_CONSUMPTION_LEDGER_RELATIVE_PATH
        ),
        "ledger_schema_version": TEST_CONSUMPTION_SCHEMA_VERSION,
        "ledger_version": TEST_CONSUMPTION_LEDGER_VERSION,
        "cohort_identity_is_model_independent": True,
        "test_gold_before_reservation": "opaque_bytes_sha256_only",
        "json_decode_test_gold_only_after_ledger_reservation": True,
        "reuse_for_model_calibrator_or_threshold_selection": False,
    }
    if spec["frozen_test"] != expected_frozen_test:
        failures.append("frozen_test")
    if failures:
        raise NMRV4ExperimentError(
            "invalid v4 run-spec: " + canonical_json_dumps(sorted(set(failures)))
        )
    return {
        "schema_version": RUN_SPEC_AUDIT_SCHEMA_VERSION,
        "status": "passed_before_scoring",
        "run_id": str(spec["run_id"]),
        "registered_at": str(spec["registered_at"]),
        "run_at": run_at,
        "run_spec_file_sha256": file_sha256(run_spec_path),
        "run_spec_content_sha256": canonical_sha256(spec),
        "parameters": parameters,
        "input_sha256": expected_inputs,
        "code_sha256": expected_code,
        "forward_model": expected_forward_model,
        "host_runtime": expected_host_runtime,
        "bindings": expected_bindings,
        "publication_audit": json.loads(
            canonical_json_dumps(publication_audit)
        ),
        "test_gold_json_decoded": False,
    }


def _validate_prediction_response(
    response: Any,
    pool: Mapping[str, Any],
    *,
    expected_forward_model: Mapping[str, Any],
    expected_sidecar_code_sha256: str,
    expected_preflight_runtime_sha256: str,
) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping):
        raise NMRV4ExperimentError("forward predictor response must be an object")
    model = response.get("model")
    response_model_fields = (
        "name",
        "repository_commit",
        "mean_model_sha256",
        "preprocessor_sha256",
    )
    if not isinstance(model, Mapping) or any(
        model.get(field_name) != expected_value
        for field_name, expected_value in expected_forward_model.items()
        if field_name in response_model_fields
    ):
        raise NMRV4ExperimentError(
            f"record {pool['record_id']} forward model binding changed"
        )
    runtime = response.get("runtime")
    expected_versions = dict(expected_forward_model["runtime_pin"])
    expected_runtime_fields = set(expected_versions) | {
        "source_bundle_sha256",
        "runtime_pin_code_sha256",
        "rdkit_version",
        "sidecar_code_sha256",
        "legacy_preflight_runtime_sha256",
        "protocol_version",
        "conformer_generation",
        "conformer_preflight",
    }
    expected_preflight_policy = {
        "protocol_version": DP5Q_CONFORMER_PREFLIGHT_PROTOCOL_VERSION,
        "candidate_failures_are_results": True,
        "operational_failures_abort_request": True,
        "uses_same_prepare_candidate_path_as_prediction": True,
    }
    if not isinstance(runtime, Mapping) or set(runtime) != expected_runtime_fields:
        raise NMRV4ExperimentError(
            f"record {pool['record_id']} forward runtime schema changed"
        )
    legacy_runtime = {
        "sidecar_code_sha256": runtime["sidecar_code_sha256"],
        "rdkit_version": runtime["rdkit_version"],
    }
    preflight_runtime = {
        **legacy_runtime,
        "conformer_generation": runtime["conformer_generation"],
        "conformer_preflight": runtime["conformer_preflight"],
    }
    if (
        any(runtime.get(key) != value for key, value in expected_versions.items())
        or runtime.get("source_bundle_sha256")
        != expected_forward_model["upstream_source_bundle_sha256"]
        or runtime.get("runtime_pin_code_sha256")
        != _code_hashes()["dp5q_runtime_pin_module_file_sha256"]
        or runtime.get("sidecar_code_sha256") != expected_sidecar_code_sha256
        or runtime.get("rdkit_version") != expected_versions["rdkit"]
        or runtime.get("legacy_preflight_runtime_sha256")
        != canonical_sha256(legacy_runtime)
        or runtime.get("protocol_version") != FORWARD_PROTOCOL_VERSION
        or runtime.get("conformer_generation") != dp5q_conformer_policy()
        or runtime.get("conformer_preflight") != expected_preflight_policy
        or canonical_sha256(preflight_runtime)
        != expected_preflight_runtime_sha256
    ):
        raise NMRV4ExperimentError(
            f"record {pool['record_id']} forward runtime binding changed"
        )
    predictions = response.get("predictions")
    candidates = pool["candidates"]
    if not isinstance(predictions, list) or len(predictions) != len(candidates):
        raise NMRV4ExperimentError(
            f"record {pool['record_id']} predictor returned wrong row count"
        )
    expected = {str(row["candidate_id"]): row for row in candidates}
    by_id: dict[str, Mapping[str, Any]] = {}
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise NMRV4ExperimentError("forward prediction must be an object")
        candidate_id = prediction.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or candidate_id not in expected
            or candidate_id in by_id
        ):
            raise NMRV4ExperimentError(
                f"record {pool['record_id']} predictor returned unknown/duplicate ID"
            )
        if not isinstance(prediction.get("atom_predictions"), list):
            raise NMRV4ExperimentError(
                f"record {pool['record_id']} prediction lacks atom_predictions"
            )
        by_id[candidate_id] = prediction
    if set(by_id) != set(expected):
        raise NMRV4ExperimentError(
            f"record {pool['record_id']} predictor omitted candidates"
        )
    return [
        {
            "candidate_id": candidate_id,
            "smiles": str(expected[candidate_id]["smiles"]),
            "atom_predictions": by_id[candidate_id]["atom_predictions"],
        }
        for candidate_id in sorted(expected)
    ]


def _validate_ranking(
    result: Any,
    *,
    expected_candidate_ids: set[str],
    record_id: str,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise NMRV4ExperimentError(f"record {record_id} scorer output is not an object")
    if result.get("protocol_version") != SCORER_PROTOCOL_VERSION:
        raise NMRV4ExperimentError(f"record {record_id} scorer protocol changed")
    if result.get("calibrated_probability") is not False:
        raise NMRV4ExperimentError(
            f"record {record_id} scorer emitted a calibrated probability"
        )
    for ranker in RANKERS:
        order = result.get(f"{ranker}_order")
        if (
            not isinstance(order, list)
            or len(order) != len(expected_candidate_ids)
            or set(order) != expected_candidate_ids
        ):
            raise NMRV4ExperimentError(
                f"record {record_id} {ranker} order is not a candidate permutation"
            )
    candidate_rows = result.get("candidates")
    if not isinstance(candidate_rows, list) or len(candidate_rows) != len(
        expected_candidate_ids
    ):
        raise NMRV4ExperimentError(
            f"record {record_id} scorer candidate evidence is incomplete"
        )
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in candidate_rows:
        if not isinstance(row, Mapping):
            raise NMRV4ExperimentError("scorer candidate evidence must be objects")
        candidate_id = row.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or candidate_id not in expected_candidate_ids
            or candidate_id in by_id
        ):
            raise NMRV4ExperimentError("scorer candidate evidence ID mismatch")
        for ranker in RANKERS:
            evidence = row.get(ranker)
            score = evidence.get("candidate_score") if isinstance(
                evidence, Mapping
            ) else None
            if (
                isinstance(score, bool)
                or not isinstance(score, (int, float))
                or not math.isfinite(float(score))
            ):
                raise NMRV4ExperimentError(
                    f"record {record_id} {ranker} candidate score is invalid"
                )
        by_id[candidate_id] = row
    if set(by_id) != expected_candidate_ids:
        raise NMRV4ExperimentError("scorer candidate evidence IDs are incomplete")
    return json.loads(canonical_json_dumps(result))


def _score_all_roleless(
    pools: Sequence[Mapping[str, Any]],
    *,
    predictor: ForwardPredictor,
    scorer: Scorer,
    expected_forward_model: Mapping[str, Any],
    expected_sidecar_code_sha256: str,
    expected_preflight_runtime_sha256: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pool in pools:
        roleless_candidates = [
            {
                "candidate_id": str(candidate["candidate_id"]),
                "smiles": str(candidate["smiles"]),
            }
            for candidate in pool["candidates"]
        ]
        # The sole model call for this complete pool.  Its output is reused by
        # both ranking equations below.
        response = predictor.predict_candidates(
            roleless_candidates,
            formula=pool["formula"],
        )
        scorer_candidates = _validate_prediction_response(
            response,
            pool,
            expected_forward_model=expected_forward_model,
            expected_sidecar_code_sha256=expected_sidecar_code_sha256,
            expected_preflight_runtime_sha256=(
                expected_preflight_runtime_sha256
            ),
        )
        result = scorer(pool["observed_13c"], scorer_candidates)
        candidate_ids = {
            str(candidate["candidate_id"]) for candidate in pool["candidates"]
        }
        clean_result = _validate_ranking(
            result,
            expected_candidate_ids=candidate_ids,
            record_id=str(pool["record_id"]),
        )
        rows.append(
            {
                "schema_version": ROLELESS_RANKING_SCHEMA_VERSION,
                "release_id": pool["release_id"],
                "record_id": pool["record_id"],
                "split": pool["split"],
                "split_group": pool["split_group"],
                "spectrum_fingerprint_sha256": pool[
                    "spectrum_fingerprint_sha256"
                ],
                "candidate_count": len(roleless_candidates),
                "forward_response_sha256": canonical_sha256(response),
                "ranking": clean_result,
            }
        )
    return sorted(rows, key=lambda row: str(row["record_id"]))


def _parse_gold_bytes(
    payload: bytes,
    *,
    source: str,
    expected_split: str,
    release_id: str,
    expected_record_ids: set[str],
) -> dict[str, str]:
    """Decode one split-specific Gold file.

    The caller is responsible for invoking this function for the test payload
    only after a successful ledger reservation.
    """

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NMRV4ExperimentError(f"{source} must be UTF-8 JSONL") from exc
    if not text or not text.endswith("\n"):
        raise NMRV4ExperimentError(
            f"{source} must be non-empty newline-terminated JSONL"
        )
    result: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise NMRV4ExperimentError(f"{source}:{line_number}: blank line")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NMRV4ExperimentError(
                f"{source}:{line_number}: invalid JSON"
            ) from exc
        if not isinstance(row, Mapping):
            raise NMRV4ExperimentError(f"{source}:{line_number}: row must be object")
        if canonical_json_dumps(row) != line:
            raise NMRV4ExperimentError(
                f"{source}:{line_number}: row is not canonical JSON"
            )
        _strict_fields(row, _GOLD_FIELDS, context=f"{source}:{line_number}")
        if row["schema_version"] != SEALED_GOLD_SCHEMA_VERSION:
            raise NMRV4ExperimentError(f"{source}:{line_number}: schema changed")
        if row["release_id"] != release_id:
            raise NMRV4ExperimentError(
                f"{source}:{line_number}: release binding mismatch"
            )
        if row["split"] != expected_split:
            raise NMRV4ExperimentError(
                f"{source}:{line_number}: Gold split mismatch"
            )
        record_id = _opaque_id(
            row["record_id"],
            f"{source}:{line_number}.record_id",
        )
        truth_candidate_id = _opaque_id(
            row["truth_candidate_id"],
            f"{source}:{line_number}.truth_candidate_id",
        )
        if record_id in result:
            raise NMRV4ExperimentError(f"{source}: duplicate record_id {record_id}")
        result[record_id] = truth_candidate_id
    if set(result) != expected_record_ids:
        missing = sorted(expected_record_ids - set(result))
        unexpected = sorted(set(result) - expected_record_ids)
        raise NMRV4ExperimentError(
            f"{source}: Gold record set mismatch; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    return result


def _load_gold_file(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_split: str,
    release_id: str,
    expected_record_ids: set[str],
) -> dict[str, str]:
    payload, observed_sha256 = _read_opaque_bytes_and_sha256(path)
    if observed_sha256 != expected_file_sha256:
        raise NMRV4ExperimentError(
            f"sealed {expected_split} Gold changed after run-spec validation"
        )
    return _parse_gold_bytes(
        payload,
        source=str(Path(path).resolve()),
        expected_split=expected_split,
        release_id=release_id,
        expected_record_ids=expected_record_ids,
    )


def _raw_confidence(ranking: Mapping[str, Any], ranker: str) -> float:
    if ranker not in RANKERS:
        raise NMRV4ExperimentError(f"unknown ranker confidence: {ranker}")
    order = ranking[f"{ranker}_order"]
    evidence = {
        str(row["candidate_id"]): row
        for row in ranking["candidates"]
    }
    first = float(evidence[str(order[0])][ranker]["candidate_score"])
    if ranker == "v4":
        value = first
        if not 0.0 <= value <= 1.0:
            raise NMRV4ExperimentError(
                "v4 top1 set-similarity confidence escaped [0,1]"
            )
    else:
        second = float(evidence[str(order[1])][ranker]["candidate_score"])
        value = first - second
    if not math.isfinite(value):
        raise NMRV4ExperimentError("rank confidence is non-finite")
    return value


def _join_gold(
    ranking_rows: Sequence[Mapping[str, Any]],
    *,
    split: str,
    gold: Mapping[str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    selected = [row for row in ranking_rows if row["split"] == split]
    if {str(row["record_id"]) for row in selected} != set(gold):
        raise NMRV4ExperimentError(f"{split} ranking/Gold record sets differ")
    for row in selected:
        record_id = str(row["record_id"])
        truth = gold[record_id]
        ranking = row["ranking"]
        candidate_ids = {
            str(candidate["candidate_id"]) for candidate in ranking["candidates"]
        }
        if truth not in candidate_ids:
            raise NMRV4ExperimentError(
                f"{split} Gold truth is absent from pool {record_id}"
            )
        outcome: dict[str, Any] = {
            "schema_version": OUTCOME_ROW_SCHEMA_VERSION,
            "release_id": row["release_id"],
            "record_id": record_id,
            "split": split,
            "split_group": row["split_group"],
            "spectrum_fingerprint_sha256": row[
                "spectrum_fingerprint_sha256"
            ],
            "candidate_count": row["candidate_count"],
            "truth_candidate_id": truth,
        }
        for ranker in RANKERS:
            order = list(ranking[f"{ranker}_order"])
            truth_rank = order.index(truth) + 1
            outcome[ranker] = {
                "top_candidate_id": order[0],
                "truth_rank": truth_rank,
                "top1_exact_correct": truth_rank == 1,
                "top3_exact_correct": truth_rank <= 3,
                "reciprocal_rank": 1.0 / truth_rank,
                "raw_confidence": _raw_confidence(ranking, ranker),
                "calibrated_probability": None,
            }
        rows.append(outcome)
    return sorted(rows, key=lambda row: str(row["record_id"]))


def _class_counts(rows: Sequence[Mapping[str, Any]], ranker: str) -> dict[str, int]:
    outcomes = [bool(row[ranker]["top1_exact_correct"]) for row in rows]
    positives = sum(outcomes)
    return {
        "total": len(outcomes),
        "correct": positives,
        "incorrect": len(outcomes) - positives,
    }


def _classes_sufficient(
    rows: Sequence[Mapping[str, Any]],
    *,
    ranker: str,
    minimum: int,
) -> bool:
    counts = _class_counts(rows, ranker)
    return counts["correct"] >= minimum and counts["incorrect"] >= minimum


def _fit_ranker_calibration(
    *,
    ranker: str,
    dev_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    dev_counts = _class_counts(dev_rows, ranker)
    calibration_counts = _class_counts(calibration_rows, ranker)
    minimum = parameters["minimum_class_count"]
    configured_methods = tuple(parameters["calibration_methods"])
    candidate_methods = (
        configured_methods
        if ranker == "v4"
        else ()
    )
    base = {
        "ranker": ranker,
        "target": "top1_exact_structure_correct",
        "confidence_semantics": CONFIDENCE_SEMANTICS[ranker],
        "candidate_calibration_methods": list(candidate_methods),
        "research_only": True,
        "probability_claim_allowed": False,
        "dev_class_counts": dev_counts,
        "calibration_class_counts": calibration_counts,
    }
    if ranker == "v3":
        return {
            **base,
            "status": "blocked",
            "blocking_reason": (
                "v3_lexicographic_rank_key_has_no_preregistered_order_"
                "preserving_scalar_confidence"
            ),
            "method_selection": None,
            "calibrator": None,
        }
    if not _classes_sufficient(
        dev_rows,
        ranker=ranker,
        minimum=int(minimum["dev"]),
    ):
        return {
            **base,
            "status": "blocked",
            "blocking_reason": "insufficient_dev_outcome_classes",
            "method_selection": None,
            "calibrator": None,
        }
    dev_examples = [
        {
            "record_id": str(row["record_id"]),
            "split": "dev",
            "group_id": str(row["split_group"]),
            "raw_score": float(row[ranker]["raw_confidence"]),
            "outcome": int(bool(row[ranker]["top1_exact_correct"])),
        }
        for row in dev_rows
    ]
    try:
        selection = select_calibration_method_dev(
            dev_examples,
            methods=candidate_methods,
            n_splits=int(parameters["calibration_cv_folds"]),
            seed=int(parameters["calibration_cv_seed"]),
            primary_metric=str(parameters["calibration_primary_metric"]),
            bins=int(parameters["calibration_bins"]),
            regularization_c=float(parameters["regularization_c"]),
            beta_epsilon=float(parameters["beta_epsilon"]),
        )
    except ValueError as exc:
        return {
            **base,
            "status": "blocked",
            "blocking_reason": f"dev_method_selection_failed: {exc}",
            "method_selection": None,
            "calibrator": None,
        }
    if not _classes_sufficient(
        calibration_rows,
        ranker=ranker,
        minimum=int(minimum["calibration"]),
    ):
        return {
            **base,
            "status": "blocked",
            "blocking_reason": "insufficient_calibration_outcome_classes",
            "method_selection": selection,
            "calibrator": None,
        }
    try:
        calibrator = fit_calibrator(
            [
                float(row[ranker]["raw_confidence"])
                for row in calibration_rows
            ],
            [
                int(bool(row[ranker]["top1_exact_correct"]))
                for row in calibration_rows
            ],
            method=str(selection["selected_method"]),
            regularization_c=float(parameters["regularization_c"]),
            beta_epsilon=float(parameters["beta_epsilon"]),
        )
    except ValueError as exc:
        return {
            **base,
            "status": "blocked",
            "blocking_reason": f"final_calibrator_fit_failed: {exc}",
            "method_selection": selection,
            "calibrator": None,
        }
    return {
        **base,
        "status": "fitted_research_only",
        "blocking_reason": None,
        "method_selection": selection,
        "calibrator": calibrator,
    }


def _ranking_metrics(
    rows: Sequence[Mapping[str, Any]],
    ranker: str,
) -> dict[str, Any]:
    if not rows:
        raise NMRV4ExperimentError("ranking metrics require non-empty rows")
    top1 = sum(bool(row[ranker]["top1_exact_correct"]) for row in rows)
    top3 = sum(bool(row[ranker]["top3_exact_correct"]) for row in rows)
    mrr = sum(float(row[ranker]["reciprocal_rank"]) for row in rows) / len(rows)
    return {
        "count": len(rows),
        "top1_correct": top1,
        "top1_accuracy": top1 / len(rows),
        "top3_correct": top3,
        "top3_accuracy": top3 / len(rows),
        "mrr": mrr,
    }


def _random_baseline(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    top1_values: list[float] = []
    top3_values: list[float] = []
    reciprocal_values: list[float] = []
    for row in rows:
        count = int(row["candidate_count"])
        top1_values.append(1.0 / count)
        top3_values.append(min(3, count) / count)
        reciprocal_values.append(
            sum(1.0 / rank for rank in range(1, count + 1)) / count
        )
    return {
        "semantics": "exact_expectation_under_uniform_random_candidate_order",
        "top1_accuracy": float(np.mean(top1_values)),
        "top3_accuracy": float(np.mean(top3_values)),
        "mrr": float(np.mean(reciprocal_values)),
    }


def _mcnemar(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    both_correct = 0
    both_wrong = 0
    v3_only = 0
    v4_only = 0
    for row in rows:
        left = bool(row["v3"]["top1_exact_correct"])
        right = bool(row["v4"]["top1_exact_correct"])
        if left and right:
            both_correct += 1
        elif left:
            v3_only += 1
        elif right:
            v4_only += 1
        else:
            both_wrong += 1
    discordant = v3_only + v4_only
    if discordant:
        tail = sum(
            math.comb(discordant, index)
            for index in range(0, min(v3_only, v4_only) + 1)
        ) / (2**discordant)
        exact_p = min(1.0, 2.0 * tail)
        corrected_statistic = (
            max(abs(v3_only - v4_only) - 1, 0) ** 2 / discordant
        )
    else:
        exact_p = 1.0
        corrected_statistic = 0.0
    return {
        "inference_status": "row_level_exploratory_only",
        "independence_assumption_satisfied": False,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "v3_only_correct": v3_only,
        "v4_only_correct": v4_only,
        "discordant": discordant,
        "mcnemar_exact_two_sided_p": None,
        "mcnemar_continuity_corrected_chi_square": None,
        "descriptive_unclustered_exact_p": exact_p,
        "descriptive_unclustered_continuity_corrected_chi_square": (
            corrected_statistic
        ),
    }


def _cluster_sign_flip(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, float] = defaultdict(float)
    for row in rows:
        by_group[str(row["split_group"])] += (
            float(bool(row["v4"]["top1_exact_correct"]))
            - float(bool(row["v3"]["top1_exact_correct"]))
        )
    groups = sorted(by_group)
    if len(groups) < 2:
        return {
            "status": "blocked",
            "blocking_reason": "fewer_than_two_independent_test_groups",
            "group_count": len(groups),
            "two_sided_p": None,
        }
    deltas = np.asarray([by_group[group] for group in groups], dtype=float)
    observed = abs(float(np.sum(deltas)))
    if len(groups) <= 20:
        exceed = 0
        total = 1 << len(groups)
        for mask in range(total):
            signed = sum(
                delta if mask & (1 << index) else -delta
                for index, delta in enumerate(deltas)
            )
            exceed += abs(float(signed)) >= observed - 1e-15
        p_value = exceed / total
        method = "exact_cluster_sign_flip"
    else:
        rng = np.random.default_rng(20260729)
        signs = rng.choice(
            np.asarray([-1.0, 1.0]),
            size=(100_000, len(groups)),
        )
        statistics = np.abs(signs @ deltas)
        p_value = (1 + int(np.sum(statistics >= observed - 1e-15))) / 100_001
        method = "deterministic_monte_carlo_cluster_sign_flip"
    return {
        "status": "evaluated",
        "method": method,
        "group_count": len(groups),
        "observed_v4_minus_v3_correct_count": float(np.sum(deltas)),
        "two_sided_p": float(p_value),
    }


def _paired_truth_ranks(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    v3_better = 0
    v4_better = 0
    tied = 0
    for row in rows:
        v3_rank = int(row["v3"]["truth_rank"])
        v4_rank = int(row["v4"]["truth_rank"])
        if v3_rank < v4_rank:
            v3_better += 1
        elif v4_rank < v3_rank:
            v4_better += 1
        else:
            tied += 1
    return {
        "v3_better": v3_better,
        "v4_better": v4_better,
        "tied": tied,
    }


def _interval(values: Sequence[float], point: float) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "point_estimate": float(point),
        "lower_95": float(np.quantile(array, 0.025)),
        "median": float(np.quantile(array, 0.5)),
        "upper_95": float(np.quantile(array, 0.975)),
    }


def _group_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    by_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["split_group"])].append(row)
    groups = sorted(by_group)
    if len(groups) < 2:
        return {
            "status": "blocked",
            "blocking_reason": "fewer_than_two_independent_test_groups",
            "group_count": len(groups),
            "replicates": replicates,
            "seed": seed,
            "intervals": None,
        }
    rng = np.random.default_rng(seed)
    samples: dict[str, list[float]] = {
        "v3_top1": [],
        "v3_top3": [],
        "v3_mrr": [],
        "v4_top1": [],
        "v4_top3": [],
        "v4_mrr": [],
        "delta_top1_v4_minus_v3": [],
        "delta_top3_v4_minus_v3": [],
        "delta_mrr_v4_minus_v3": [],
    }
    for _ in range(replicates):
        indices = rng.integers(0, len(groups), size=len(groups))
        sampled: list[Mapping[str, Any]] = []
        for index in indices:
            sampled.extend(by_group[groups[int(index)]])
        v3 = _ranking_metrics(sampled, "v3")
        v4 = _ranking_metrics(sampled, "v4")
        samples["v3_top1"].append(float(v3["top1_accuracy"]))
        samples["v3_top3"].append(float(v3["top3_accuracy"]))
        samples["v3_mrr"].append(float(v3["mrr"]))
        samples["v4_top1"].append(float(v4["top1_accuracy"]))
        samples["v4_top3"].append(float(v4["top3_accuracy"]))
        samples["v4_mrr"].append(float(v4["mrr"]))
        samples["delta_top1_v4_minus_v3"].append(
            float(v4["top1_accuracy"]) - float(v3["top1_accuracy"])
        )
        samples["delta_top3_v4_minus_v3"].append(
            float(v4["top3_accuracy"]) - float(v3["top3_accuracy"])
        )
        samples["delta_mrr_v4_minus_v3"].append(
            float(v4["mrr"]) - float(v3["mrr"])
        )
    point_v3 = _ranking_metrics(rows, "v3")
    point_v4 = _ranking_metrics(rows, "v4")
    points = {
        "v3_top1": float(point_v3["top1_accuracy"]),
        "v3_top3": float(point_v3["top3_accuracy"]),
        "v3_mrr": float(point_v3["mrr"]),
        "v4_top1": float(point_v4["top1_accuracy"]),
        "v4_top3": float(point_v4["top3_accuracy"]),
        "v4_mrr": float(point_v4["mrr"]),
        "delta_top1_v4_minus_v3": (
            float(point_v4["top1_accuracy"]) - float(point_v3["top1_accuracy"])
        ),
        "delta_top3_v4_minus_v3": (
            float(point_v4["top3_accuracy"]) - float(point_v3["top3_accuracy"])
        ),
        "delta_mrr_v4_minus_v3": (
            float(point_v4["mrr"]) - float(point_v3["mrr"])
        ),
    }
    return {
        "status": "evaluated",
        "method": "nonparametric_split_group_cluster_bootstrap",
        "group_count": len(groups),
        "replicates": replicates,
        "seed": seed,
        "intervals": {
            name: _interval(values, points[name])
            for name, values in samples.items()
        },
    }


def _predict_test_probabilities_outcome_blind(
    *,
    ranking_rows: Sequence[Mapping[str, Any]],
    ranker: str,
    fit_result: Mapping[str, Any],
) -> dict[str, float] | None:
    """Apply a frozen calibrator before any test Gold JSON is decoded."""

    if fit_result["status"] != "fitted_research_only":
        return None
    selected = sorted(
        (row for row in ranking_rows if row["split"] == "test"),
        key=lambda row: str(row["record_id"]),
    )
    scores = [
        _raw_confidence(row["ranking"], ranker)
        for row in selected
    ]
    probabilities = predict_calibrated(fit_result["calibrator"], scores)
    return {
        str(row["record_id"]): float(probability)
        for row, probability in zip(selected, probabilities, strict=True)
    }


def _evaluate_test_calibration(
    *,
    test_rows: list[dict[str, Any]],
    ranker: str,
    fit_result: dict[str, Any],
    probabilities_by_record: Mapping[str, float] | None,
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    counts = _class_counts(test_rows, ranker)
    if (
        fit_result["status"] != "fitted_research_only"
        or probabilities_by_record is None
    ):
        return {
            "status": "blocked",
            "blocking_reason": "calibrator_not_fitted",
            "test_class_counts": counts,
            "probabilities_emitted": False,
            "metrics": None,
        }
    if set(probabilities_by_record) != {
        str(row["record_id"]) for row in test_rows
    }:
        raise NMRV4ExperimentError(
            "outcome-blind test probability record set changed"
        )
    probabilities = [
        float(probabilities_by_record[str(row["record_id"])])
        for row in test_rows
    ]
    outcomes = [
        int(bool(row[ranker]["top1_exact_correct"])) for row in test_rows
    ]
    for row, probability in zip(test_rows, probabilities, strict=True):
        row[ranker]["calibrated_probability"] = probability
    return {
        "status": "evaluated_research_only",
        "blocking_reason": None,
        "test_class_counts": counts,
        "probabilities_emitted": True,
        "probabilities_generated_before_test_gold_decode": True,
        "probability_claim_allowed": False,
        "metrics": calibration_metrics(
            probabilities,
            outcomes,
            bins=int(parameters["calibration_bins"]),
        ),
    }


def _render_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _render_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json_dumps(row) + "\n" for row in rows)


def _publish_immutable_directory(
    destination: str | Path,
    rendered_files: Mapping[str, str],
) -> dict[str, str]:
    target = Path(destination).resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{target.name}.",
            suffix=".part",
            dir=target.parent,
        )
    )
    try:
        for name, rendered in rendered_files.items():
            if Path(name).name != name:
                raise NMRV4ExperimentError(f"artifact name must be basename: {name}")
            path = staging / name
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(rendered)
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(staging, target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {
        name: hashlib.sha256(rendered.encode("utf-8")).hexdigest()
        for name, rendered in rendered_files.items()
    }


def _run_v4_experiment_core(
    *,
    roleless_pools_path: str | Path,
    pool_summary_path: str | Path,
    eligibility_path: str | Path,
    split_manifest_path: str | Path,
    source_current_path: str | Path,
    sealed_gold_dev_path: str | Path,
    sealed_gold_calibration_path: str | Path,
    sealed_gold_test_path: str | Path,
    run_spec_path: str | Path,
    output_dir: str | Path,
    predictor: ForwardPredictor,
    scorer: Scorer,
    _clock: Callable[[], datetime] | None,
    _test_components: bool,
) -> dict[str, Any]:
    """Private execution seam; injected components are always test-labelled."""

    canonical_ledger = Path(CANONICAL_TEST_CONSUMPTION_LEDGER_PATH).resolve()
    if not _test_components and (
        not isinstance(predictor, NMRForwardAdapter)
        or scorer is not rank_candidate_evidence
    ):
        raise NMRV4ExperimentError(
            "canonical v4 execution requires the real forward adapter and "
            "frozen candidate scorer"
        )
    run_at = _clock_utc_iso(_clock, field_name="run_at")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output {output}")
    pools = load_roleless_pools(roleless_pools_path)
    pool_summary = load_pool_summary(
        pool_summary_path,
        roleless_pools_path=roleless_pools_path,
        eligibility_path=eligibility_path,
        sealed_gold_dev_path=sealed_gold_dev_path,
        sealed_gold_calibration_path=sealed_gold_calibration_path,
        sealed_gold_test_path=sealed_gold_test_path,
        split_manifest_path=split_manifest_path,
    )
    split_manifest = load_split_manifest(
        split_manifest_path,
        pools=pools,
    )
    try:
        source_records, source_binding = load_reviewed_release(
            source_current_path
        )
    except ValueError as exc:
        raise NMRV4ExperimentError(
            f"verified source release failed validation: {exc}"
        ) from exc
    if pool_summary["release_id"] != pools[0]["release_id"]:
        raise NMRV4ExperimentError("pool summary release differs from roleless pools")
    eligibility = load_eligibility(
        eligibility_path,
        release_id=str(pool_summary["release_id"]),
        base_index_sha256=str(pool_summary["base_index"]["sha256"]),
        maximum_decoys=int(
            pool_summary["candidate_generation"]["maximum_decoys"]
        ),
    )
    publication_audit = validate_candidate_publication(
        pools=pools,
        split_manifest=split_manifest,
        eligibility=eligibility,
        pool_summary=pool_summary,
        source_records=source_records,
        source_binding=source_binding,
    )
    spec = load_run_spec(run_spec_path)
    audit = validate_run_spec(
        spec,
        run_spec_path=run_spec_path,
        roleless_pools_path=roleless_pools_path,
        pool_summary_path=pool_summary_path,
        eligibility_path=eligibility_path,
        split_manifest_path=split_manifest_path,
        source_current_path=source_current_path,
        sealed_gold_dev_path=sealed_gold_dev_path,
        sealed_gold_calibration_path=sealed_gold_calibration_path,
        sealed_gold_test_path=sealed_gold_test_path,
        pools=pools,
        pool_summary=pool_summary,
        split_manifest=split_manifest,
        publication_audit=publication_audit,
        run_at=run_at,
    )
    parameters = audit["parameters"]
    release_id = str(audit["bindings"]["release_id"])

    # All forward computation and both rankers remain outcome blind.
    rankings = _score_all_roleless(
        pools,
        predictor=predictor,
        scorer=scorer,
        expected_forward_model=audit["forward_model"],
        expected_sidecar_code_sha256=audit["code_sha256"][
            "dp5q_sidecar_script_file_sha256"
        ],
        expected_preflight_runtime_sha256=str(
            audit["bindings"]["preflight_runtime_sha256"]
        ),
    )

    expected_by_split = {
        split: {
            str(row["record_id"]) for row in pools if row["split"] == split
        }
        for split in SPLITS
    }
    dev_gold = _load_gold_file(
        sealed_gold_dev_path,
        expected_file_sha256=str(
            audit["input_sha256"]["sealed_gold_dev_file_sha256"]
        ),
        expected_split="dev",
        release_id=release_id,
        expected_record_ids=expected_by_split["dev"],
    )
    dev_rows = _join_gold(rankings, split="dev", gold=dev_gold)

    calibration_gold = _load_gold_file(
        sealed_gold_calibration_path,
        expected_file_sha256=str(
            audit["input_sha256"]["sealed_gold_calibration_file_sha256"]
        ),
        expected_split="calibration",
        release_id=release_id,
        expected_record_ids=expected_by_split["calibration"],
    )
    calibration_rows = _join_gold(
        rankings,
        split="calibration",
        gold=calibration_gold,
    )
    calibration_results = {
        ranker: _fit_ranker_calibration(
            ranker=ranker,
            dev_rows=dev_rows,
            calibration_rows=calibration_rows,
            parameters=parameters,
        )
        for ranker in RANKERS
    }
    test_probabilities = {
        ranker: _predict_test_probabilities_outcome_blind(
            ranking_rows=rankings,
            ranker=ranker,
            fit_result=calibration_results[ranker],
        )
        for ranker in RANKERS
    }

    # Capture and verify test Gold only as opaque bytes.  No JSON decoding has
    # happened before the model-independent cohort is durably reserved.
    test_gold_bytes, live_test_gold_sha = _read_opaque_bytes_and_sha256(
        sealed_gold_test_path
    )
    expected_test_gold_sha = str(
        audit["input_sha256"]["sealed_gold_test_file_sha256"]
    )
    if live_test_gold_sha != expected_test_gold_sha:
        raise NMRV4ExperimentError("sealed test Gold changed before reservation")
    ledger_entry = reserve_test_cohort_consumption(
        canonical_ledger,
        run_id=str(spec["run_id"]),
        source_release_id=release_id,
        run_spec_file_sha256=str(audit["run_spec_file_sha256"]),
        test_cohort_sha256=str(audit["bindings"]["test_cohort_sha256"]),
        _clock=_clock,
    )
    reservation_time = str(ledger_entry["reserved_at"])

    # This is the first JSON decode of the test Gold file in the run.
    test_gold = _parse_gold_bytes(
        test_gold_bytes,
        source=str(Path(sealed_gold_test_path).resolve()),
        expected_split="test",
        release_id=release_id,
        expected_record_ids=expected_by_split["test"],
    )
    test_rows = _join_gold(rankings, split="test", gold=test_gold)
    calibration_test_results = {
        ranker: _evaluate_test_calibration(
            test_rows=test_rows,
            ranker=ranker,
            fit_result=calibration_results[ranker],
            probabilities_by_record=test_probabilities[ranker],
            parameters=parameters,
        )
        for ranker in RANKERS
    }
    for ranker in RANKERS:
        calibration_results[ranker]["test_evaluation"] = (
            calibration_test_results[ranker]
        )

    v3_metrics = _ranking_metrics(test_rows, "v3")
    v4_metrics = _ranking_metrics(test_rows, "v4")
    comparison = {
        "common_candidate_pool_count": len(test_rows),
        "common_candidate_pool_binding_sha256": canonical_sha256(
            [
                {
                    "record_id": row["record_id"],
                    "candidate_count": row["candidate_count"],
                    "spectrum_fingerprint_sha256": row[
                        "spectrum_fingerprint_sha256"
                    ],
                }
                for row in test_rows
            ]
        ),
        "rankers": {"v3": v3_metrics, "v4": v4_metrics},
        "random_baseline": _random_baseline(test_rows),
        "paired_top1": _mcnemar(test_rows),
        "cluster_top1_sign_flip": _cluster_sign_flip(test_rows),
        "paired_truth_rank": _paired_truth_ranks(test_rows),
        "group_bootstrap": _group_bootstrap(
            test_rows,
            replicates=int(parameters["bootstrap_replicates"]),
            seed=int(parameters["bootstrap_seed"]),
        ),
    }
    report_core = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "run_id": str(spec["run_id"]),
        "registered_at": str(spec["registered_at"]),
        "run_at": run_at,
        "reserved_at": reservation_time,
        "status": (
            "completed_test_fixture_single_use_test"
            if _test_components
            else "completed_single_use_test"
        ),
        "research_only": True,
        "claim_boundaries": {
            "external_equation_reproduction": True,
            "nmr_solver_method_independence": False,
            "source_doi_per_row_available": False,
            "cryptographically_blinded": False,
            "public_source_truth_derivable": True,
            "evaluation_design": (
                "preregistered_single_use_retrospective_evaluation"
            ),
            "externally_signed": False,
            "artifact_provenance": "self_attested_local_build",
            "note": (
                "This is a preregistered single-use retrospective evaluation "
                "of public-source-derived labels and an external reproduction "
                "of score equations, not a cryptographically blinded or "
                "method-independent reproduction of NMR-Solver."
            ),
        },
        "test_consumption": {
            **ledger_entry,
            "test_gold_json_decoded_after_reservation": True,
        },
        "input_bindings": audit["bindings"],
        "prediction_call_contract": {
            "expected_calls": len(pools),
            "semantics": "one_complete_pool_per_call_reused_by_both_rankers",
            "truth_fields_passed_to_predictor_or_scorer": False,
        },
        "comparison": comparison,
        "calibration": calibration_results,
        "probability_claim_allowed": False,
        "execution_attestation": {
            "canonical_entrypoint": not _test_components,
            "forward_adapter": (
                "app.ml.nmr_forward.NMRForwardAdapter"
                if not _test_components
                else f"test_fixture:{type(predictor).__name__}"
            ),
            "candidate_scorer": (
                "app.ml.nmr_candidate_scorer_v4.rank_candidate_evidence"
                if not _test_components
                else f"test_fixture:{getattr(scorer, '__name__', type(scorer).__name__)}"
            ),
        },
    }
    report = {
        **report_core,
        "report_content_sha256": canonical_sha256(report_core),
    }
    audit = {
        **audit,
        "status": "passed_and_executed",
        "test_gold_json_decoded": True,
        "ledger_reservation_sha256": canonical_sha256(ledger_entry),
    }

    rendered = {
        "rankings-roleless.jsonl": _render_jsonl(rankings),
        "dev-outcomes.jsonl": _render_jsonl(dev_rows),
        "calibration-outcomes.jsonl": _render_jsonl(calibration_rows),
        "test-outcomes.jsonl": _render_jsonl(test_rows),
        "calibration.json": _render_json(calibration_results),
        "run-spec-audit.json": _render_json(audit),
        "report.json": _render_json(report),
    }
    artifact_hashes = {
        name: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for name, content in rendered.items()
    }
    manifest_core = {
        "schema_version": ARTIFACT_MANIFEST_SCHEMA_VERSION,
        "run_id": str(spec["run_id"]),
        "immutable_output": True,
        "artifacts": artifact_hashes,
    }
    rendered["artifact-manifest.json"] = _render_json(
        {
            **manifest_core,
            "manifest_content_sha256": canonical_sha256(manifest_core),
        }
    )
    _publish_immutable_directory(output, rendered)
    return report


def run_v4_experiment(
    *,
    roleless_pools_path: str | Path,
    pool_summary_path: str | Path,
    eligibility_path: str | Path,
    split_manifest_path: str | Path,
    source_current_path: str | Path,
    sealed_gold_dev_path: str | Path,
    sealed_gold_calibration_path: str | Path,
    sealed_gold_test_path: str | Path,
    run_spec_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Execute the canonical frozen comparison with non-injectable components."""

    config = NMRForwardConfig.from_environment()
    if config.verify_local_install is not True:
        raise NMRV4ExperimentError(
            "canonical v4 execution requires verified local DP5q assets"
        )
    with NMRForwardAdapter(config) as predictor:
        return _run_v4_experiment_core(
            roleless_pools_path=roleless_pools_path,
            pool_summary_path=pool_summary_path,
            eligibility_path=eligibility_path,
            split_manifest_path=split_manifest_path,
            source_current_path=source_current_path,
            sealed_gold_dev_path=sealed_gold_dev_path,
            sealed_gold_calibration_path=sealed_gold_calibration_path,
            sealed_gold_test_path=sealed_gold_test_path,
            run_spec_path=run_spec_path,
            output_dir=output_dir,
            predictor=predictor,
            scorer=rank_candidate_evidence,
            _clock=None,
            _test_components=False,
        )


__all__ = [
    "ARTIFACT_MANIFEST_SCHEMA_VERSION",
    "CANONICAL_TEST_CONSUMPTION_LEDGER_PATH",
    "CANONICAL_TEST_CONSUMPTION_LEDGER_RELATIVE_PATH",
    "CONFIDENCE_SEMANTICS",
    "MAX_POOL_CANDIDATES",
    "NMRV4ExperimentError",
    "OUTCOME_ROW_SCHEMA_VERSION",
    "PIPELINE_VERSION",
    "POOL_SUMMARY_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "ROLELESS_POOL_SCHEMA_VERSION",
    "ROLELESS_RANKING_SCHEMA_VERSION",
    "RUN_SPEC_SCHEMA_VERSION",
    "RUN_SPEC_VERSION",
    "SEALED_GOLD_SCHEMA_VERSION",
    "build_run_spec_template",
    "default_parameters",
    "file_sha256",
    "load_roleless_pools",
    "load_pool_summary",
    "load_run_spec",
    "load_split_manifest",
    "run_v4_experiment",
    "validate_run_spec",
]
