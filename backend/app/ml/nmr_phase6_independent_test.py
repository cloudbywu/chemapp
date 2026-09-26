"""One-shot, end-to-end protocol for an externally held phase-6 NMR test.

This module is deliberately separate from the frozen v4 experiment.  It binds
the complete roleless cohort, generator output (including empty candidate
lists), ranking output, release manifests, and a holder-signed Gold artifact.
Gold is opened only after a matching one-time reservation has been committed.

The local ledger is an integrity and replay control, not a WORM store or a
trusted timestamp.  Production use additionally requires the externally held
artifacts and trust anchors described in the phase-6 protocol document.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any


PROTOCOL_VERSION = 1
RUN_SPEC_SCHEMA = "chemapp.nmr.phase6-run-spec.v1"
ROLELESS_SCHEMA = "chemapp.nmr.phase6-roleless-queries.v1"
GENERATOR_SCHEMA = "chemapp.nmr.phase6-generator-output.v1"
RANKING_SCHEMA = "chemapp.nmr.phase6-ranking-output.v1"
GOLD_SCHEMA = "chemapp.nmr.phase6-holder-gold.v1"
HOLDER_AUDIT_SCHEMA = "chemapp.nmr.phase6-holder-audit.v1"
CONSUMED_SET_SCHEMA = "chemapp.nmr.phase6-consumed-set.v1"
RESERVATION_SCHEMA = "chemapp.nmr.phase6-reservation.v2"
EVALUATION_SCHEMA = "chemapp.nmr.phase6-evaluation.v1"
RESULT_SCHEMA = "chemapp.nmr.phase6-result.v1"

EXTERNAL_HOLDER_REQUIRED = "external_holder_required"
EXTERNAL_HOLDER_VERIFIED = "external_holder_verified"
RECALL_KS = (1, 5, 10, 25, 50, 100)
MINIMUM_CONNECTED_GROUPS = 200
PRIMARY_ENDPOINT = "end_to_end_group_macro_exact_and_connectivity_recall_at_k"
SECONDARY_ENDPOINT = "conditional_ranker_group_macro_recall_at_k"
GROUP_AGGREGATION = "macro_average_of_within_group_query_means"
GOLD_ABSENT_POLICY = "generator_miss_is_failure_at_every_k"
ONE_TIME_POLICY = "reserve_complete_outputs_before_first_gold_read"
HOLDER_ATTESTATION = (
    "external_holder_kept_gold_private_from_model_team_until_reservation"
)
CONSUMED_ATTESTATION = (
    "external_holder_verified_no_cohort_query_or_connected_group_reuse"
)

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._:-]{0,127}$")
_MAX_JSON_BYTES = 128 * 1024 * 1024
_ZERO_HASH = "0" * 64


class NMRPhase6Error(ValueError):
    """Raised when a phase-6 trust or schema invariant is violated."""


class NMRPhase6DependencyError(RuntimeError):
    """Raised when Ed25519 verification is unavailable."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON with exactly one trailing newline."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NMRPhase6Error("value is not finite canonical JSON") from exc
    return (rendered + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NMRPhase6Error(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes, *, location: str) -> Any:
    if not payload or len(payload) > _MAX_JSON_BYTES:
        raise NMRPhase6Error(f"{location}: JSON size is invalid")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NMRPhase6Error(f"{location}: JSON must be UTF-8") from exc
    if text.startswith("\ufeff"):
        raise NMRPhase6Error(f"{location}: UTF-8 BOM is forbidden")

    def reject_constant(value: str) -> None:
        raise NMRPhase6Error(f"{location}: non-finite number {value!r} is forbidden")

    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=reject_constant,
        )
    except NMRPhase6Error:
        raise
    except (ValueError, RecursionError) as exc:
        raise NMRPhase6Error(f"{location}: invalid JSON") from exc
    _reject_nonfinite(value, location=location)
    return value


def _reject_nonfinite(value: Any, *, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise NMRPhase6Error(f"{location}: non-finite number is forbidden")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, location=f"{location}[{index}]")


def _safe_file(path: str | Path, *, label: str) -> Path:
    selected = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    parts = selected.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current /= part
        if current.is_symlink():
            raise NMRPhase6Error(f"{label}: symlinked path is forbidden")
    try:
        resolved = selected.resolve(strict=True)
    except OSError as exc:
        raise NMRPhase6Error(f"{label}: file does not exist") from exc
    if not resolved.is_file() or resolved.stat().st_size > _MAX_JSON_BYTES:
        raise NMRPhase6Error(f"{label}: expected a bounded regular file")
    return resolved


def _read_canonical_json(
    path: str | Path, *, label: str
) -> tuple[dict[str, Any], bytes]:
    selected = _safe_file(path, label=label)
    payload = selected.read_bytes()
    value = strict_json_bytes(payload, location=str(selected))
    if canonical_json_bytes(value) != payload:
        raise NMRPhase6Error(f"{label}: artifact must be canonical JSON")
    if not isinstance(value, dict):
        raise NMRPhase6Error(f"{label}: expected an object")
    return value, payload


def _exact(value: Any, fields: set[str], *, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NMRPhase6Error(f"{location}: expected an object")
    if set(value) != fields:
        raise NMRPhase6Error(
            f"{location}: field allowlist mismatch: {sorted(set(value) ^ fields)}"
        )
    return value


def _text(value: Any, *, location: str, maximum: int = 4096) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise NMRPhase6Error(f"{location}: invalid non-empty text")
    return value


def _identifier(value: Any, *, location: str) -> str:
    text = _text(value, location=location, maximum=128)
    if _ID_RE.fullmatch(text) is None:
        raise NMRPhase6Error(f"{location}: invalid identifier")
    return text


def _sha(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise NMRPhase6Error(f"{location}: expected lowercase SHA-256")
    return value


def _utc(value: Any, *, location: str) -> str:
    text = _text(value, location=location, maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NMRPhase6Error(f"{location}: expected explicit UTC timestamp") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
    ):
        raise NMRPhase6Error(f"{location}: expected explicit UTC timestamp")
    return text


def _utc_datetime(value: Any, *, location: str) -> datetime:
    return datetime.fromisoformat(_utc(value, location=location).replace("Z", "+00:00"))


def _now(clock: Callable[[], datetime] | None) -> str:
    current = (clock or (lambda: datetime.now(timezone.utc)))()
    if not isinstance(current, datetime) or current.tzinfo is None:
        raise NMRPhase6Error("clock must return a timezone-aware datetime")
    return current.astimezone(timezone.utc).isoformat()


def _hash_release(value: Any, *, location: str) -> dict[str, Any]:
    release = _exact(value, {"release_id", "manifest_sha256"}, location=location)
    _identifier(release["release_id"], location=f"{location}.release_id")
    _sha(release["manifest_sha256"], location=f"{location}.manifest_sha256")
    return release


def validate_run_spec(value: Any) -> dict[str, Any]:
    spec = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "protocol_id",
            "run_id",
            "registered_at",
            "external_holder",
            "minimum_connected_groups",
            "recall_ks",
            "primary_endpoint",
            "secondary_endpoint",
            "connected_group_aggregation",
            "gold_absent_policy",
            "one_time_policy",
            "v5_model_release",
            "v5_calibrator_release",
            "holder_public_key_spki_sha256",
            "roleless_queries_sha256",
            "cohort_commitment_sha256",
            "consumed_set_exclusion",
        },
        location="run_spec",
    )
    if spec["schema_version"] != RUN_SPEC_SCHEMA or spec["protocol_version"] != 1:
        raise NMRPhase6Error("run_spec: unsupported schema or protocol")
    _identifier(spec["protocol_id"], location="run_spec.protocol_id")
    _identifier(spec["run_id"], location="run_spec.run_id")
    _utc(spec["registered_at"], location="run_spec.registered_at")
    holder = _exact(
        spec["external_holder"],
        {"status", "holder_id", "attestation", "attestation_sha256"},
        location="run_spec.external_holder",
    )
    if holder["status"] != EXTERNAL_HOLDER_VERIFIED:
        raise NMRPhase6Error(EXTERNAL_HOLDER_REQUIRED)
    _identifier(holder["holder_id"], location="run_spec.external_holder.holder_id")
    if holder["attestation"] != HOLDER_ATTESTATION:
        raise NMRPhase6Error("run_spec.external_holder: invalid attestation")
    _sha(
        holder["attestation_sha256"],
        location="run_spec.external_holder.attestation_sha256",
    )
    minimum = spec["minimum_connected_groups"]
    if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 200:
        raise NMRPhase6Error("run_spec.minimum_connected_groups must be at least 200")
    if spec["recall_ks"] != list(RECALL_KS):
        raise NMRPhase6Error("run_spec.recall_ks must be [1,5,10,25,50,100]")
    fixed = {
        "primary_endpoint": PRIMARY_ENDPOINT,
        "secondary_endpoint": SECONDARY_ENDPOINT,
        "connected_group_aggregation": GROUP_AGGREGATION,
        "gold_absent_policy": GOLD_ABSENT_POLICY,
        "one_time_policy": ONE_TIME_POLICY,
    }
    for field, expected in fixed.items():
        if spec[field] != expected:
            raise NMRPhase6Error(f"run_spec.{field}: preregistered value changed")
    _hash_release(spec["v5_model_release"], location="run_spec.v5_model_release")
    _hash_release(
        spec["v5_calibrator_release"],
        location="run_spec.v5_calibrator_release",
    )
    for field in (
        "holder_public_key_spki_sha256",
        "roleless_queries_sha256",
        "cohort_commitment_sha256",
    ):
        _sha(spec[field], location=f"run_spec.{field}")
    exclusion = _exact(
        spec["consumed_set_exclusion"],
        {"registry_sha256", "checked_at", "attestation", "current_cohort_absent"},
        location="run_spec.consumed_set_exclusion",
    )
    _sha(
        exclusion["registry_sha256"],
        location="run_spec.consumed_set_exclusion.registry_sha256",
    )
    _utc(exclusion["checked_at"], location="run_spec.consumed_set_exclusion.checked_at")
    if (
        exclusion["attestation"] != CONSUMED_ATTESTATION
        or exclusion["current_cohort_absent"] is not True
    ):
        raise NMRPhase6Error(
            "run_spec.consumed_set_exclusion: invalid exclusion declaration"
        )
    return spec


def _validate_artifact(value: Any, *, location: str) -> dict[str, Any]:
    artifact = _exact(value, {"kind", "sha256", "bytes"}, location=location)
    _identifier(artifact["kind"], location=f"{location}.kind")
    _sha(artifact["sha256"], location=f"{location}.sha256")
    if (
        not isinstance(artifact["bytes"], int)
        or isinstance(artifact["bytes"], bool)
        or artifact["bytes"] < 1
    ):
        raise NMRPhase6Error(f"{location}.bytes: expected a positive integer")
    return artifact


def _cohort_commitment(records: Sequence[Mapping[str, Any]]) -> str:
    rows = sorted(
        (
            {
                "query_commitment_sha256": row["query_commitment_sha256"],
                "connected_group_commitment_sha256": row[
                    "connected_group_commitment_sha256"
                ],
            }
            for row in records
        ),
        key=lambda row: (
            row["query_commitment_sha256"],
            row["connected_group_commitment_sha256"],
        ),
    )
    commitment_input = {
        "schema_version": "chemapp.nmr.phase6-cohort-commitment-input.v1",
        "records": rows,
    }
    return sha256_bytes(canonical_json_bytes(commitment_input))


def validate_roleless(value: Any) -> dict[str, Any]:
    roleless = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "protocol_id",
            "run_id",
            "cohort_commitment_sha256",
            "records",
        },
        location="roleless",
    )
    if (
        roleless["schema_version"] != ROLELESS_SCHEMA
        or roleless["protocol_version"] != 1
    ):
        raise NMRPhase6Error("roleless: unsupported schema or protocol")
    _identifier(roleless["protocol_id"], location="roleless.protocol_id")
    _identifier(roleless["run_id"], location="roleless.run_id")
    _sha(
        roleless["cohort_commitment_sha256"],
        location="roleless.cohort_commitment_sha256",
    )
    records = roleless["records"]
    if not isinstance(records, list) or not records:
        raise NMRPhase6Error("roleless.records: expected a non-empty list")
    query_ids: set[str] = set()
    query_commitments: set[str] = set()
    group_by_id: dict[str, str] = {}
    group_id_by_commitment: dict[str, str] = {}
    for index, raw in enumerate(records):
        location = f"roleless.records[{index}]"
        record = _exact(
            raw,
            {
                "query_id",
                "connected_group_id",
                "query_commitment_sha256",
                "connected_group_commitment_sha256",
                "formula",
                "artifacts",
            },
            location=location,
        )
        query_id = _identifier(record["query_id"], location=f"{location}.query_id")
        group_id = _identifier(
            record["connected_group_id"], location=f"{location}.connected_group_id"
        )
        query_commitment = _sha(
            record["query_commitment_sha256"],
            location=f"{location}.query_commitment_sha256",
        )
        group_commitment = _sha(
            record["connected_group_commitment_sha256"],
            location=f"{location}.connected_group_commitment_sha256",
        )
        if query_id in query_ids or query_commitment in query_commitments:
            raise NMRPhase6Error(f"{location}: duplicate query identity or commitment")
        query_ids.add(query_id)
        query_commitments.add(query_commitment)
        existing_group = group_by_id.setdefault(group_id, group_commitment)
        if existing_group != group_commitment:
            raise NMRPhase6Error(f"{location}: group id maps to multiple commitments")
        existing_group_id = group_id_by_commitment.setdefault(
            group_commitment, group_id
        )
        if existing_group_id != group_id:
            raise NMRPhase6Error(
                f"{location}: group commitment maps to multiple group ids"
            )
        if record["formula"] is not None:
            _text(record["formula"], location=f"{location}.formula", maximum=256)
        if not isinstance(record["artifacts"], list) or not record["artifacts"]:
            raise NMRPhase6Error(f"{location}.artifacts: expected a non-empty list")
        kinds: set[str] = set()
        for artifact_index, artifact in enumerate(record["artifacts"]):
            checked = _validate_artifact(
                artifact, location=f"{location}.artifacts[{artifact_index}]"
            )
            if checked["kind"] in kinds:
                raise NMRPhase6Error(f"{location}.artifacts: duplicate kind")
            kinds.add(checked["kind"])
    expected = _cohort_commitment(records)
    if roleless["cohort_commitment_sha256"] != expected:
        raise NMRPhase6Error(
            "roleless: cohort commitment does not match complete record list"
        )
    return roleless


def validate_consumed_set(value: Any) -> dict[str, Any]:
    registry = _exact(
        value,
        {
            "schema_version",
            "updated_at",
            "cohort_commitment_sha256s",
            "query_commitment_sha256s",
            "connected_group_commitment_sha256s",
        },
        location="consumed_set",
    )
    if registry["schema_version"] != CONSUMED_SET_SCHEMA:
        raise NMRPhase6Error("consumed_set: unsupported schema")
    _utc(registry["updated_at"], location="consumed_set.updated_at")
    for field in (
        "cohort_commitment_sha256s",
        "query_commitment_sha256s",
        "connected_group_commitment_sha256s",
    ):
        values = registry[field]
        if not isinstance(values, list):
            raise NMRPhase6Error(f"consumed_set.{field}: must be sorted and unique")
        for index, digest in enumerate(values):
            _sha(digest, location=f"consumed_set.{field}[{index}]")
        if values != sorted(set(values)):
            raise NMRPhase6Error(f"consumed_set.{field}: must be sorted and unique")
    return registry


def _validate_public_binding(
    spec: Mapping[str, Any],
    spec_payload: bytes,
    roleless: Mapping[str, Any],
    roleless_payload: bytes,
) -> None:
    if (
        spec["protocol_id"] != roleless["protocol_id"]
        or spec["run_id"] != roleless["run_id"]
    ):
        raise NMRPhase6Error("run spec and roleless protocol/run binding mismatch")
    if spec["roleless_queries_sha256"] != sha256_bytes(roleless_payload):
        raise NMRPhase6Error("run spec roleless hash mismatch")
    if spec["cohort_commitment_sha256"] != roleless["cohort_commitment_sha256"]:
        raise NMRPhase6Error("run spec cohort commitment mismatch")
    if not spec_payload:
        raise NMRPhase6Error("run spec payload is empty")


def validate_generator_output(
    value: Any,
    *,
    spec: Mapping[str, Any],
    spec_payload: bytes,
    roleless: Mapping[str, Any],
    roleless_payload: bytes,
) -> dict[str, Any]:
    output = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "protocol_id",
            "run_id",
            "created_at",
            "run_spec_sha256",
            "roleless_queries_sha256",
            "generator_release_manifest_sha256",
            "records",
        },
        location="generator_output",
    )
    if output["schema_version"] != GENERATOR_SCHEMA or output["protocol_version"] != 1:
        raise NMRPhase6Error("generator_output: unsupported schema or protocol")
    for field in ("protocol_id", "run_id"):
        if output[field] != spec[field]:
            raise NMRPhase6Error(f"generator_output.{field}: binding mismatch")
    _utc(output["created_at"], location="generator_output.created_at")
    if _utc_datetime(
        output["created_at"], location="generator_output.created_at"
    ) <= _utc_datetime(spec["registered_at"], location="run_spec.registered_at"):
        raise NMRPhase6Error("generator_output must be created after registration")
    if output["run_spec_sha256"] != sha256_bytes(spec_payload) or output[
        "roleless_queries_sha256"
    ] != sha256_bytes(roleless_payload):
        raise NMRPhase6Error("generator_output: public artifact hash binding mismatch")
    _sha(
        output["generator_release_manifest_sha256"],
        location="generator_output.generator_release_manifest_sha256",
    )
    expected_ids = [row["query_id"] for row in roleless["records"]]
    records = output["records"]
    if (
        not isinstance(records, list)
        or [row.get("query_id") if isinstance(row, dict) else None for row in records]
        != expected_ids
    ):
        raise NMRPhase6Error(
            "generator_output.records must exactly cover roleless order"
        )
    for index, raw in enumerate(records):
        location = f"generator_output.records[{index}]"
        record = _exact(raw, {"query_id", "candidates"}, location=location)
        _identifier(record["query_id"], location=f"{location}.query_id")
        candidates = record["candidates"]
        if not isinstance(candidates, list) or len(candidates) > 100_000:
            raise NMRPhase6Error(f"{location}.candidates: invalid list")
        candidate_ids: set[str] = set()
        exact_hashes: set[str] = set()
        for candidate_index, raw_candidate in enumerate(candidates):
            candidate_location = f"{location}.candidates[{candidate_index}]"
            candidate = _exact(
                raw_candidate,
                {"candidate_id", "exact_structure_sha256", "connectivity_sha256"},
                location=candidate_location,
            )
            candidate_id = _identifier(
                candidate["candidate_id"], location=f"{candidate_location}.candidate_id"
            )
            exact_hash = _sha(
                candidate["exact_structure_sha256"],
                location=f"{candidate_location}.exact_structure_sha256",
            )
            _sha(
                candidate["connectivity_sha256"],
                location=f"{candidate_location}.connectivity_sha256",
            )
            if candidate_id in candidate_ids or exact_hash in exact_hashes:
                raise NMRPhase6Error(
                    f"{candidate_location}: duplicate candidate id or exact structure"
                )
            candidate_ids.add(candidate_id)
            exact_hashes.add(exact_hash)
    return output


def validate_ranking_output(
    value: Any,
    *,
    spec: Mapping[str, Any],
    spec_payload: bytes,
    roleless_payload: bytes,
    generator: Mapping[str, Any],
    generator_payload: bytes,
) -> dict[str, Any]:
    output = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "protocol_id",
            "run_id",
            "created_at",
            "run_spec_sha256",
            "roleless_queries_sha256",
            "generator_output_sha256",
            "v5_model_release_manifest_sha256",
            "v5_calibrator_release_manifest_sha256",
            "records",
        },
        location="ranking_output",
    )
    if output["schema_version"] != RANKING_SCHEMA or output["protocol_version"] != 1:
        raise NMRPhase6Error("ranking_output: unsupported schema or protocol")
    for field in ("protocol_id", "run_id"):
        if output[field] != spec[field]:
            raise NMRPhase6Error(f"ranking_output.{field}: binding mismatch")
    _utc(output["created_at"], location="ranking_output.created_at")
    if _utc_datetime(
        output["created_at"], location="ranking_output.created_at"
    ) <= _utc_datetime(generator["created_at"], location="generator_output.created_at"):
        raise NMRPhase6Error("ranking_output must be created after generator output")
    bindings = {
        "run_spec_sha256": sha256_bytes(spec_payload),
        "roleless_queries_sha256": sha256_bytes(roleless_payload),
        "generator_output_sha256": sha256_bytes(generator_payload),
        "v5_model_release_manifest_sha256": spec["v5_model_release"]["manifest_sha256"],
        "v5_calibrator_release_manifest_sha256": spec["v5_calibrator_release"][
            "manifest_sha256"
        ],
    }
    for field, expected in bindings.items():
        if output[field] != expected:
            raise NMRPhase6Error(
                f"ranking_output.{field}: frozen hash binding mismatch"
            )
    generated_records = generator["records"]
    records = output["records"]
    if not isinstance(records, list) or len(records) != len(generated_records):
        raise NMRPhase6Error("ranking_output.records: incomplete query coverage")
    for index, (raw, generated) in enumerate(
        zip(records, generated_records, strict=True)
    ):
        location = f"ranking_output.records[{index}]"
        record = _exact(raw, {"query_id", "ranked_candidate_ids"}, location=location)
        if record["query_id"] != generated["query_id"]:
            raise NMRPhase6Error(f"{location}.query_id: generator order mismatch")
        ranked = record["ranked_candidate_ids"]
        expected = [candidate["candidate_id"] for candidate in generated["candidates"]]
        if not isinstance(ranked, list):
            raise NMRPhase6Error(
                f"{location}: ranking must be a complete candidate permutation"
            )
        for candidate_index, candidate_id in enumerate(ranked):
            _identifier(
                candidate_id,
                location=f"{location}.ranked_candidate_ids[{candidate_index}]",
            )
        if len(ranked) != len(set(ranked)) or set(ranked) != set(expected):
            raise NMRPhase6Error(
                f"{location}: ranking must be a complete candidate permutation"
            )
    return output


def _validate_gold(
    value: Any, *, spec: Mapping[str, Any], roleless: Mapping[str, Any]
) -> dict[str, Any]:
    gold = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "protocol_id",
            "run_id",
            "cohort_commitment_sha256",
            "released_at",
            "holder_audit_sha256",
            "records",
        },
        location="holder_gold",
    )
    if gold["schema_version"] != GOLD_SCHEMA or gold["protocol_version"] != 1:
        raise NMRPhase6Error("holder_gold: unsupported schema or protocol")
    for field in ("protocol_id", "run_id"):
        if gold[field] != spec[field]:
            raise NMRPhase6Error(f"holder_gold.{field}: binding mismatch")
    if gold["cohort_commitment_sha256"] != roleless["cohort_commitment_sha256"]:
        raise NMRPhase6Error("holder_gold: cohort commitment mismatch")
    _utc(gold["released_at"], location="holder_gold.released_at")
    _sha(gold["holder_audit_sha256"], location="holder_gold.holder_audit_sha256")
    records = gold["records"]
    roleless_records = roleless["records"]
    if not isinstance(records, list) or len(records) != len(roleless_records):
        raise NMRPhase6Error("holder_gold.records: incomplete query coverage")
    for index, (record, query) in enumerate(
        zip(records, roleless_records, strict=True)
    ):
        location = f"holder_gold.records[{index}]"
        checked = _exact(
            record,
            {
                "query_id",
                "connected_group_id",
                "query_commitment_sha256",
                "connected_group_commitment_sha256",
                "exact_structure_sha256",
                "connectivity_sha256",
            },
            location=location,
        )
        for field in (
            "query_id",
            "connected_group_id",
            "query_commitment_sha256",
            "connected_group_commitment_sha256",
        ):
            if checked[field] != query[field]:
                raise NMRPhase6Error(f"{location}.{field}: roleless binding mismatch")
        _sha(
            checked["exact_structure_sha256"],
            location=f"{location}.exact_structure_sha256",
        )
        _sha(checked["connectivity_sha256"], location=f"{location}.connectivity_sha256")
    return gold


def _validate_holder_audit(
    value: Any,
    *,
    spec: Mapping[str, Any],
    roleless: Mapping[str, Any],
) -> dict[str, Any]:
    audit = _exact(
        value,
        {
            "schema_version",
            "protocol_version",
            "protocol_id",
            "run_id",
            "cohort_commitment_sha256",
            "completed_at",
            "dual_review",
            "rights_review",
            "overlap_review",
        },
        location="holder_audit",
    )
    if audit["schema_version"] != HOLDER_AUDIT_SCHEMA or audit["protocol_version"] != 1:
        raise NMRPhase6Error("holder_audit: unsupported schema or protocol")
    for field in ("protocol_id", "run_id"):
        if audit[field] != spec[field]:
            raise NMRPhase6Error(f"holder_audit.{field}: binding mismatch")
    if audit["cohort_commitment_sha256"] != roleless["cohort_commitment_sha256"]:
        raise NMRPhase6Error("holder_audit: cohort commitment mismatch")
    _utc(audit["completed_at"], location="holder_audit.completed_at")
    required = {
        "dual_review": "two_independent_reviews_with_third_adjudicator_on_disagreement",
        "rights_review": "rights_cleared_for_test_and_release",
        "overlap_review": "no_consumed_cohort_query_or_connected_group_overlap",
    }
    for field, expected_status in required.items():
        declaration = _exact(
            audit[field],
            {"status", "evidence_sha256"},
            location=f"holder_audit.{field}",
        )
        if declaration["status"] != expected_status:
            raise NMRPhase6Error(f"holder_audit.{field}: required status is absent")
        _sha(
            declaration["evidence_sha256"],
            location=f"holder_audit.{field}.evidence_sha256",
        )
    return audit


def _load_public_artifacts(
    run_spec_path: str | Path,
    roleless_path: str | Path,
    generator_output_path: str | Path,
    ranking_output_path: str | Path,
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
]:
    spec, spec_payload = _read_canonical_json(run_spec_path, label="run spec")
    roleless, roleless_payload = _read_canonical_json(
        roleless_path, label="roleless queries"
    )
    validate_run_spec(spec)
    validate_roleless(roleless)
    _validate_public_binding(spec, spec_payload, roleless, roleless_payload)
    generator, generator_payload = _read_canonical_json(
        generator_output_path, label="generator output"
    )
    validate_generator_output(
        generator,
        spec=spec,
        spec_payload=spec_payload,
        roleless=roleless,
        roleless_payload=roleless_payload,
    )
    ranking, ranking_payload = _read_canonical_json(
        ranking_output_path, label="ranking output"
    )
    validate_ranking_output(
        ranking,
        spec=spec,
        spec_payload=spec_payload,
        roleless_payload=roleless_payload,
        generator=generator,
        generator_payload=generator_payload,
    )
    return (
        spec,
        spec_payload,
        roleless,
        roleless_payload,
        generator,
        generator_payload,
        ranking,
        ranking_payload,
    )


def _entry_hash(entry_without_hash: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(entry_without_hash)))


def _sorted_hashes(value: Any, *, location: str, non_empty: bool) -> list[str]:
    if not isinstance(value, list) or (non_empty and not value):
        raise NMRPhase6Error(f"{location}: expected a sorted unique digest list")
    for index, digest in enumerate(value):
        _sha(digest, location=f"{location}[{index}]")
    if value != sorted(set(value)):
        raise NMRPhase6Error(f"{location}: expected a sorted unique digest list")
    return value


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
        raise NMRPhase6Error("reservation ledger must be a bounded regular file")
    entries: list[dict[str, Any]] = []
    previous = _ZERO_HASH
    reservations_by_run: dict[str, dict[str, Any]] = {}
    reserved_cohorts: set[str] = set()
    reserved_roleless: set[str] = set()
    reserved_queries: set[str] = set()
    reserved_groups: set[str] = set()
    evaluated_runs: set[str] = set()
    for index, line in enumerate(path.read_bytes().splitlines(keepends=True)):
        if not line.endswith(b"\n"):
            raise NMRPhase6Error("reservation ledger has an unterminated line")
        value = strict_json_bytes(line, location=f"reservation ledger line {index + 1}")
        if canonical_json_bytes(value) != line:
            raise NMRPhase6Error("reservation ledger is not canonical JSONL")
        location = f"reservation ledger line {index + 1}"
        if not isinstance(value, dict):
            raise NMRPhase6Error(f"{location}: expected an object")
        event = value.get("event")
        if event == "one_time_reservation":
            entry = _exact(
                value,
                {
                    "schema_version",
                    "sequence",
                    "previous_entry_sha256",
                    "entry_sha256",
                    "event",
                    "protocol_id",
                    "run_id",
                    "reserved_at",
                    "cohort_commitment_sha256",
                    "run_spec_sha256",
                    "roleless_queries_sha256",
                    "generator_output_sha256",
                    "ranking_output_sha256",
                    "consumed_set_sha256",
                    "connected_group_count",
                    "query_commitment_sha256s",
                    "connected_group_commitment_sha256s",
                },
                location=location,
            )
            if entry["schema_version"] != RESERVATION_SCHEMA:
                raise NMRPhase6Error("reservation ledger schema mismatch")
            _utc(entry["reserved_at"], location=f"{location}.reserved_at")
            for field in (
                "cohort_commitment_sha256",
                "run_spec_sha256",
                "roleless_queries_sha256",
                "generator_output_sha256",
                "ranking_output_sha256",
                "consumed_set_sha256",
            ):
                _sha(entry[field], location=f"{location}.{field}")
            if (
                not isinstance(entry["connected_group_count"], int)
                or isinstance(entry["connected_group_count"], bool)
                or entry["connected_group_count"] < MINIMUM_CONNECTED_GROUPS
            ):
                raise NMRPhase6Error("reservation connected-group count is invalid")
            _sorted_hashes(
                entry["query_commitment_sha256s"],
                location=f"{location}.query_commitment_sha256s",
                non_empty=True,
            )
            group_commitments = _sorted_hashes(
                entry["connected_group_commitment_sha256s"],
                location=f"{location}.connected_group_commitment_sha256s",
                non_empty=True,
            )
            if len(group_commitments) != entry["connected_group_count"]:
                raise NMRPhase6Error("reservation connected-group list/count mismatch")
        elif event == "one_time_evaluation":
            entry = _exact(
                value,
                {
                    "schema_version",
                    "sequence",
                    "previous_entry_sha256",
                    "entry_sha256",
                    "event",
                    "protocol_id",
                    "run_id",
                    "evaluated_at",
                    "cohort_commitment_sha256",
                    "reservation_entry_sha256",
                    "holder_gold_sha256",
                    "holder_audit_sha256",
                    "holder_gold_signature_sha256",
                    "holder_public_key_spki_sha256",
                    "metric_result_sha256",
                },
                location=location,
            )
            if entry["schema_version"] != EVALUATION_SCHEMA:
                raise NMRPhase6Error("evaluation ledger schema mismatch")
            _utc(entry["evaluated_at"], location=f"{location}.evaluated_at")
            for field in (
                "cohort_commitment_sha256",
                "reservation_entry_sha256",
                "holder_gold_sha256",
                "holder_audit_sha256",
                "holder_gold_signature_sha256",
                "holder_public_key_spki_sha256",
                "metric_result_sha256",
            ):
                _sha(entry[field], location=f"{location}.{field}")
        else:
            raise NMRPhase6Error("reservation ledger schema or event mismatch")
        if (
            not isinstance(entry["sequence"], int)
            or isinstance(entry["sequence"], bool)
            or entry["sequence"] != index + 1
            or entry["previous_entry_sha256"] != previous
        ):
            raise NMRPhase6Error("reservation ledger chain mismatch")
        _identifier(entry["protocol_id"], location="reservation.protocol_id")
        _identifier(entry["run_id"], location="reservation.run_id")
        for field in (
            "previous_entry_sha256",
            "entry_sha256",
        ):
            _sha(entry[field], location=f"reservation.{field}")
        without_hash = dict(entry)
        claimed = without_hash.pop("entry_sha256")
        if claimed != _entry_hash(without_hash):
            raise NMRPhase6Error("reservation ledger entry hash mismatch")
        if entry["event"] == "one_time_reservation":
            run_id = entry["run_id"]
            query_commitments = set(entry["query_commitment_sha256s"])
            group_commitments = set(entry["connected_group_commitment_sha256s"])
            if run_id in reservations_by_run:
                raise NMRPhase6Error("reservation ledger repeats a run id")
            if entry["cohort_commitment_sha256"] in reserved_cohorts:
                raise NMRPhase6Error("reservation ledger repeats a cohort")
            if entry["roleless_queries_sha256"] in reserved_roleless:
                raise NMRPhase6Error("reservation ledger repeats roleless bytes")
            if query_commitments.intersection(reserved_queries):
                raise NMRPhase6Error("reservation ledger contains query reuse")
            if group_commitments.intersection(reserved_groups):
                raise NMRPhase6Error(
                    "reservation ledger contains connected-group reuse"
                )
            reservations_by_run[run_id] = entry
            reserved_cohorts.add(entry["cohort_commitment_sha256"])
            reserved_roleless.add(entry["roleless_queries_sha256"])
            reserved_queries.update(query_commitments)
            reserved_groups.update(group_commitments)
        else:
            run_id = entry["run_id"]
            reservation = reservations_by_run.get(run_id)
            if reservation is None:
                raise NMRPhase6Error(
                    "evaluation ledger entry has no earlier reservation"
                )
            if run_id in evaluated_runs:
                raise NMRPhase6Error("reservation ledger repeats an evaluation")
            if (
                entry["protocol_id"] != reservation["protocol_id"]
                or entry["cohort_commitment_sha256"]
                != reservation["cohort_commitment_sha256"]
                or entry["reservation_entry_sha256"] != reservation["entry_sha256"]
            ):
                raise NMRPhase6Error(
                    "evaluation ledger entry does not bind its reservation"
                )
            if _utc_datetime(
                entry["evaluated_at"], location="evaluation.evaluated_at"
            ) <= _utc_datetime(
                reservation["reserved_at"], location="reservation.reserved_at"
            ):
                raise NMRPhase6Error("evaluation must occur after reservation")
            evaluated_runs.add(run_id)
        previous = sha256_bytes(line)
        entries.append(entry)
    return entries


@contextmanager
def _ledger_lock(path: Path):
    lock = path.with_name(path.name + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise NMRPhase6Error("reservation ledger is locked by another process") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        yield
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def reserve_phase6_once(
    *,
    run_spec_path: str | Path,
    roleless_path: str | Path,
    generator_output_path: str | Path,
    ranking_output_path: str | Path,
    consumed_set_path: str | Path,
    reservation_ledger_path: str | Path,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Reserve complete public outputs exactly once, without accepting Gold."""

    (
        spec,
        spec_payload,
        roleless,
        roleless_payload,
        _generator,
        generator_payload,
        _ranking,
        ranking_payload,
    ) = _load_public_artifacts(
        run_spec_path,
        roleless_path,
        generator_output_path,
        ranking_output_path,
    )
    registry, registry_payload = _read_canonical_json(
        consumed_set_path, label="consumed set"
    )
    validate_consumed_set(registry)
    registry_sha = sha256_bytes(registry_payload)
    if spec["consumed_set_exclusion"]["registry_sha256"] != registry_sha:
        raise NMRPhase6Error("run spec consumed-set hash mismatch")

    cohort = roleless["cohort_commitment_sha256"]
    query_commitments = sorted(
        {row["query_commitment_sha256"] for row in roleless["records"]}
    )
    group_commitments = sorted(
        {row["connected_group_commitment_sha256"] for row in roleless["records"]}
    )
    if cohort in registry["cohort_commitment_sha256s"]:
        raise NMRPhase6Error("cohort has already been consumed")
    if set(query_commitments).intersection(registry["query_commitment_sha256s"]):
        raise NMRPhase6Error("one or more queries have already been consumed")
    if set(group_commitments).intersection(
        registry["connected_group_commitment_sha256s"]
    ):
        raise NMRPhase6Error("one or more connected groups have already been consumed")
    group_count = len(group_commitments)
    if (
        group_count < spec["minimum_connected_groups"]
        or group_count < MINIMUM_CONNECTED_GROUPS
    ):
        raise NMRPhase6Error(
            "external cohort has fewer than 200 unique connected groups"
        )

    ledger = Path(
        os.path.abspath(os.fspath(Path(reservation_ledger_path).expanduser()))
    )
    if ledger.is_symlink():
        raise NMRPhase6Error("reservation ledger symlink is forbidden")
    with _ledger_lock(ledger):
        entries = _read_ledger(ledger)
        for prior in entries:
            if prior["run_id"] == spec["run_id"]:
                raise NMRPhase6Error("run id is already reserved or evaluated")
            if prior["event"] != "one_time_reservation":
                continue
            if prior["cohort_commitment_sha256"] == cohort:
                raise NMRPhase6Error("cohort is already reserved and cannot be reused")
            if prior["roleless_queries_sha256"] == sha256_bytes(roleless_payload):
                raise NMRPhase6Error("roleless cohort is already reserved")
            if set(query_commitments).intersection(prior["query_commitment_sha256s"]):
                raise NMRPhase6Error(
                    "one or more queries overlap a previously reserved cohort"
                )
            if set(group_commitments).intersection(
                prior["connected_group_commitment_sha256s"]
            ):
                raise NMRPhase6Error(
                    "one or more connected groups overlap a previously reserved cohort"
                )
        entry_without_hash: dict[str, Any] = {
            "schema_version": RESERVATION_SCHEMA,
            "sequence": len(entries) + 1,
            "previous_entry_sha256": sha256_bytes(canonical_json_bytes(entries[-1]))
            if entries
            else _ZERO_HASH,
            "event": "one_time_reservation",
            "protocol_id": spec["protocol_id"],
            "run_id": spec["run_id"],
            "reserved_at": _now(clock),
            "cohort_commitment_sha256": cohort,
            "run_spec_sha256": sha256_bytes(spec_payload),
            "roleless_queries_sha256": sha256_bytes(roleless_payload),
            "generator_output_sha256": sha256_bytes(generator_payload),
            "ranking_output_sha256": sha256_bytes(ranking_payload),
            "consumed_set_sha256": registry_sha,
            "connected_group_count": group_count,
            "query_commitment_sha256s": query_commitments,
            "connected_group_commitment_sha256s": group_commitments,
        }
        if _utc_datetime(
            entry_without_hash["reserved_at"], location="reservation.reserved_at"
        ) <= _utc_datetime(
            _ranking["created_at"], location="ranking_output.created_at"
        ):
            raise NMRPhase6Error("reservation must occur after ranking output creation")
        entry = {**entry_without_hash, "entry_sha256": _entry_hash(entry_without_hash)}
        ledger.parent.mkdir(parents=True, exist_ok=True)
        with ledger.open("ab") as handle:
            handle.write(canonical_json_bytes(entry))
            handle.flush()
            os.fsync(handle.fileno())
        _read_ledger(ledger)
    return entry


def _public_key_spki_sha256(public_key_path: str | Path):
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    except ImportError as exc:
        raise NMRPhase6DependencyError(
            "Ed25519 verification requires cryptography>=44.0"
        ) from exc
    path = _safe_file(public_key_path, label="holder Ed25519 public key")
    try:
        key = serialization.load_pem_public_key(path.read_bytes())
    except (TypeError, ValueError) as exc:
        raise NMRPhase6DependencyError("holder public key is invalid PEM") from exc
    if not isinstance(key, Ed25519PublicKey):
        raise NMRPhase6DependencyError("holder public key must be Ed25519")
    spki = key.public_bytes(
        serialization.Encoding.DER,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return key, sha256_bytes(spki)


def _find_reservation(
    entries: Sequence[Mapping[str, Any]],
    *,
    spec: Mapping[str, Any],
    spec_payload: bytes,
    roleless_payload: bytes,
    generator_payload: bytes,
    ranking_payload: bytes,
) -> Mapping[str, Any]:
    expected = {
        "run_spec_sha256": sha256_bytes(spec_payload),
        "roleless_queries_sha256": sha256_bytes(roleless_payload),
        "generator_output_sha256": sha256_bytes(generator_payload),
        "ranking_output_sha256": sha256_bytes(ranking_payload),
    }
    matches = [
        entry
        for entry in entries
        if entry["event"] == "one_time_reservation"
        and entry["run_id"] == spec["run_id"]
    ]
    if len(matches) != 1:
        raise NMRPhase6Error("exactly one matching pre-Gold reservation is required")
    reservation = matches[0]
    for field, digest in expected.items():
        if reservation[field] != digest:
            raise NMRPhase6Error(f"reservation {field} mismatch")
    return reservation


def _reject_prior_evaluation(
    entries: Sequence[Mapping[str, Any]], *, run_id: str
) -> None:
    if any(
        entry["event"] == "one_time_evaluation" and entry["run_id"] == run_id
        for entry in entries
    ):
        raise NMRPhase6Error("run has already been evaluated once")


def _group_macro(
    per_query: Mapping[str, Mapping[int, float]],
    query_to_group: Mapping[str, str],
) -> dict[str, float]:
    groups: dict[str, list[str]] = {}
    for query_id in per_query:
        groups.setdefault(query_to_group[query_id], []).append(query_id)
    if not groups:
        return {str(k): 0.0 for k in RECALL_KS}
    return {
        str(k): sum(
            sum(per_query[query_id][k] for query_id in query_ids) / len(query_ids)
            for query_ids in groups.values()
        )
        / len(groups)
        for k in RECALL_KS
    }


def evaluate_phase6_once(
    *,
    run_spec_path: str | Path,
    roleless_path: str | Path,
    generator_output_path: str | Path,
    ranking_output_path: str | Path,
    reservation_ledger_path: str | Path,
    holder_gold_path: str | Path,
    holder_audit_path: str | Path,
    holder_gold_signature_path: str | Path,
    holder_public_key_path: str | Path,
    expected_reservation_entry_sha256: str,
    expected_holder_public_key_spki_sha256: str,
    clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Run the evaluation while exclusively holding the authoritative ledger."""

    ledger_path = Path(
        os.path.abspath(os.fspath(Path(reservation_ledger_path).expanduser()))
    )
    # Lock before any holder-controlled path is touched. A concurrent evaluation
    # therefore fails closed without opening either version of Gold.
    with _ledger_lock(ledger_path):
        return _evaluate_phase6_once_locked(
            run_spec_path=run_spec_path,
            roleless_path=roleless_path,
            generator_output_path=generator_output_path,
            ranking_output_path=ranking_output_path,
            reservation_ledger_path=reservation_ledger_path,
            holder_gold_path=holder_gold_path,
            holder_audit_path=holder_audit_path,
            holder_gold_signature_path=holder_gold_signature_path,
            holder_public_key_path=holder_public_key_path,
            expected_reservation_entry_sha256=(expected_reservation_entry_sha256),
            expected_holder_public_key_spki_sha256=(
                expected_holder_public_key_spki_sha256
            ),
            clock=clock,
            ledger_lock_held=True,
        )


def _evaluate_phase6_once_locked(
    *,
    run_spec_path: str | Path,
    roleless_path: str | Path,
    generator_output_path: str | Path,
    ranking_output_path: str | Path,
    reservation_ledger_path: str | Path,
    holder_gold_path: str | Path,
    holder_audit_path: str | Path,
    holder_gold_signature_path: str | Path,
    holder_public_key_path: str | Path,
    expected_reservation_entry_sha256: str,
    expected_holder_public_key_spki_sha256: str,
    clock: Callable[[], datetime] | None,
    ledger_lock_held: bool,
) -> dict[str, Any]:
    """Evaluate after proving reservation; Gold is not opened before that proof."""

    (
        spec,
        spec_payload,
        roleless,
        roleless_payload,
        generator,
        generator_payload,
        ranking,
        ranking_payload,
    ) = _load_public_artifacts(
        run_spec_path,
        roleless_path,
        generator_output_path,
        ranking_output_path,
    )
    ledger_path = Path(
        os.path.abspath(os.fspath(Path(reservation_ledger_path).expanduser()))
    )
    ledger_entries = _read_ledger(ledger_path)
    reservation = _find_reservation(
        ledger_entries,
        spec=spec,
        spec_payload=spec_payload,
        roleless_payload=roleless_payload,
        generator_payload=generator_payload,
        ranking_payload=ranking_payload,
    )
    expected_reservation_digest = _sha(
        expected_reservation_entry_sha256,
        location="independently anchored reservation-entry digest",
    )
    if reservation["entry_sha256"] != expected_reservation_digest:
        raise NMRPhase6Error(
            "reservation does not match the independently anchored digest"
        )
    _reject_prior_evaluation(ledger_entries, run_id=spec["run_id"])
    expected_key_digest = _sha(
        expected_holder_public_key_spki_sha256,
        location="independently supplied expected holder public-key digest",
    )

    # Trust boundary: no holder-only path is touched before the reservation proof.
    public_key, public_key_digest = _public_key_spki_sha256(holder_public_key_path)
    if (
        public_key_digest != spec["holder_public_key_spki_sha256"]
        or public_key_digest != expected_key_digest
    ):
        raise NMRPhase6Error(
            "holder public-key digest does not match run spec trust anchor"
        )
    gold, gold_payload = _read_canonical_json(holder_gold_path, label="holder Gold")
    signature_path = _safe_file(
        holder_gold_signature_path, label="holder Gold signature"
    )
    signature = signature_path.read_bytes()
    if len(signature) != 64:
        raise NMRPhase6Error("holder Gold signature must be 64 raw Ed25519 bytes")
    try:
        public_key.verify(signature, gold_payload)
    except Exception as exc:
        raise NMRPhase6Error("holder Gold Ed25519 signature is invalid") from exc
    _validate_gold(gold, spec=spec, roleless=roleless)
    holder_audit, holder_audit_payload = _read_canonical_json(
        holder_audit_path,
        label="holder audit",
    )
    _validate_holder_audit(holder_audit, spec=spec, roleless=roleless)
    if gold["holder_audit_sha256"] != sha256_bytes(holder_audit_payload):
        raise NMRPhase6Error("holder Gold does not bind the supplied holder audit")
    if _utc_datetime(
        gold["released_at"], location="holder_gold.released_at"
    ) <= _utc_datetime(reservation["reserved_at"], location="reservation.reserved_at"):
        raise NMRPhase6Error("holder Gold release must be strictly after reservation")

    query_to_group = {
        row["query_id"]: row["connected_group_id"] for row in roleless["records"]
    }
    generator_by_query = {
        row["query_id"]: row["candidates"] for row in generator["records"]
    }
    ranking_by_query = {
        row["query_id"]: row["ranked_candidate_ids"] for row in ranking["records"]
    }
    gold_by_query = {row["query_id"]: row for row in gold["records"]}
    exact_per_query: dict[str, dict[int, float]] = {}
    connectivity_per_query: dict[str, dict[int, float]] = {}
    conditional_exact: dict[str, dict[int, float]] = {}
    conditional_connectivity: dict[str, dict[int, float]] = {}
    exact_misses = 0
    connectivity_misses = 0
    for query_id, candidates in generator_by_query.items():
        candidate_by_id = {
            candidate["candidate_id"]: candidate for candidate in candidates
        }
        ranked_candidates = [
            candidate_by_id[candidate_id] for candidate_id in ranking_by_query[query_id]
        ]
        truth = gold_by_query[query_id]
        exact_ranks = [
            index
            for index, candidate in enumerate(ranked_candidates, start=1)
            if candidate["exact_structure_sha256"] == truth["exact_structure_sha256"]
        ]
        connectivity_ranks = [
            index
            for index, candidate in enumerate(ranked_candidates, start=1)
            if candidate["connectivity_sha256"] == truth["connectivity_sha256"]
        ]
        exact_rank = min(exact_ranks) if exact_ranks else None
        connectivity_rank = min(connectivity_ranks) if connectivity_ranks else None
        exact_per_query[query_id] = {
            k: float(exact_rank is not None and exact_rank <= k) for k in RECALL_KS
        }
        connectivity_per_query[query_id] = {
            k: float(connectivity_rank is not None and connectivity_rank <= k)
            for k in RECALL_KS
        }
        if exact_rank is None:
            exact_misses += 1
        else:
            conditional_exact[query_id] = exact_per_query[query_id]
        if connectivity_rank is None:
            connectivity_misses += 1
        else:
            conditional_connectivity[query_id] = connectivity_per_query[query_id]

    result = {
        "schema_version": RESULT_SCHEMA,
        "protocol_version": 1,
        "status": "evaluated_once",
        "protocol_id": spec["protocol_id"],
        "run_id": spec["run_id"],
        "cohort_commitment_sha256": roleless["cohort_commitment_sha256"],
        "reservation_entry_sha256": reservation["entry_sha256"],
        "holder_gold_sha256": sha256_bytes(gold_payload),
        "denominators": {
            "queries": len(roleless["records"]),
            "connected_groups": len(set(query_to_group.values())),
            "conditional_exact_queries": len(conditional_exact),
            "conditional_connectivity_queries": len(conditional_connectivity),
        },
        "generator_misses": {
            "exact_queries": exact_misses,
            "connectivity_queries": connectivity_misses,
            "policy": GOLD_ABSENT_POLICY,
        },
        "primary": {
            "label": PRIMARY_ENDPOINT,
            "aggregation": GROUP_AGGREGATION,
            "full_denominator": True,
            "exact_recall_at_k": _group_macro(exact_per_query, query_to_group),
            "connectivity_recall_at_k": _group_macro(
                connectivity_per_query, query_to_group
            ),
        },
        "secondary": {
            "label": SECONDARY_ENDPOINT,
            "conditional_on_generator_coverage": True,
            "not_a_primary_claim": True,
            "exact_recall_at_k": _group_macro(conditional_exact, query_to_group),
            "connectivity_recall_at_k": _group_macro(
                conditional_connectivity, query_to_group
            ),
        },
    }
    metric_result_sha = sha256_bytes(canonical_json_bytes(result))
    with nullcontext() if ledger_lock_held else _ledger_lock(ledger_path):
        current_entries = _read_ledger(ledger_path)
        current_reservation = _find_reservation(
            current_entries,
            spec=spec,
            spec_payload=spec_payload,
            roleless_payload=roleless_payload,
            generator_payload=generator_payload,
            ranking_payload=ranking_payload,
        )
        if current_reservation["entry_sha256"] != expected_reservation_digest:
            raise NMRPhase6Error(
                "reservation does not match the independently anchored digest"
            )
        _reject_prior_evaluation(current_entries, run_id=spec["run_id"])
        evaluated_at = _now(clock)
        if _utc_datetime(
            evaluated_at, location="evaluation.evaluated_at"
        ) <= _utc_datetime(gold["released_at"], location="holder_gold.released_at"):
            raise NMRPhase6Error("evaluation must occur after holder Gold release")
        entry_without_hash: dict[str, Any] = {
            "schema_version": EVALUATION_SCHEMA,
            "sequence": len(current_entries) + 1,
            "previous_entry_sha256": sha256_bytes(
                canonical_json_bytes(current_entries[-1])
            ),
            "event": "one_time_evaluation",
            "protocol_id": spec["protocol_id"],
            "run_id": spec["run_id"],
            "evaluated_at": evaluated_at,
            "cohort_commitment_sha256": roleless["cohort_commitment_sha256"],
            "reservation_entry_sha256": current_reservation["entry_sha256"],
            "holder_gold_sha256": sha256_bytes(gold_payload),
            "holder_audit_sha256": sha256_bytes(holder_audit_payload),
            "holder_gold_signature_sha256": sha256_bytes(signature),
            "holder_public_key_spki_sha256": public_key_digest,
            "metric_result_sha256": metric_result_sha,
        }
        evaluation_entry = {
            **entry_without_hash,
            "entry_sha256": _entry_hash(entry_without_hash),
        }
        with ledger_path.open("ab") as handle:
            handle.write(canonical_json_bytes(evaluation_entry))
            handle.flush()
            os.fsync(handle.fileno())
        _read_ledger(ledger_path)
    return result


def execution_status() -> dict[str, Any]:
    """Return the repository's honest phase-6 execution state."""

    return {
        "schema_version": "chemapp.nmr.phase6-execution-status.v1",
        "status": EXTERNAL_HOLDER_REQUIRED,
        "executed": False,
        "real_external_results": None,
        "minimum_connected_groups": MINIMUM_CONNECTED_GROUPS,
        "reason": "no independently held private Gold cohort is present in this repository",
    }


__all__ = [
    "CONSUMED_SET_SCHEMA",
    "EVALUATION_SCHEMA",
    "EXTERNAL_HOLDER_REQUIRED",
    "GENERATOR_SCHEMA",
    "GOLD_SCHEMA",
    "HOLDER_AUDIT_SCHEMA",
    "MINIMUM_CONNECTED_GROUPS",
    "NMRPhase6DependencyError",
    "NMRPhase6Error",
    "RANKING_SCHEMA",
    "RECALL_KS",
    "RESULT_SCHEMA",
    "ROLELESS_SCHEMA",
    "RUN_SPEC_SCHEMA",
    "canonical_json_bytes",
    "evaluate_phase6_once",
    "execution_status",
    "reserve_phase6_once",
    "sha256_bytes",
    "validate_consumed_set",
    "validate_generator_output",
    "validate_ranking_output",
    "validate_roleless",
    "validate_run_spec",
]
