"""Fail-closed protocol for independently held NMR blind challenges.

The protocol deliberately separates four moments:

1. a holder validates two independent reviews and builds an outcome-free pack;
2. the model team submits a complete prediction artifact and receives a
   hash-chained local receipt;
3. the holder releases Gold only after that receipt exists, signing the
   release manifest with Ed25519; and
4. any party verifies the detached signature and every byte binding.

The receipt ledger detects ordinary replay and accidental mutation.  It is a
local file, not WORM storage, a trusted timestamp, or a transparency log.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import shutil
import tempfile
from typing import Any


PROTOCOL_VERSION = 1
REVIEW_BUNDLE_SCHEMA = "chemapp.nmr.blind-review-bundle.v1"
RUN_SPEC_SCHEMA = "chemapp.nmr.blind-run-spec.v2"
LEGACY_RUN_SPEC_SCHEMA = "chemapp.nmr.blind-run-spec.v1"
ROLELESS_SCHEMA = "chemapp.nmr.blind-roleless.v1"
GOLD_SCHEMA = "chemapp.nmr.blind-holder-gold.v1"
REVIEW_AUDIT_SCHEMA = "chemapp.nmr.blind-review-audit.v1"
PUBLIC_MANIFEST_SCHEMA = "chemapp.nmr.blind-public-manifest.v1"
HOLDER_MANIFEST_SCHEMA = "chemapp.nmr.blind-holder-manifest.v1"
PREDICTIONS_SCHEMA = "chemapp.nmr.blind-predictions.v1"
RECEIPT_SCHEMA = "chemapp.nmr.blind-receipt.v1"
RELEASE_MANIFEST_SCHEMA = "chemapp.nmr.blind-release-manifest.v1"
SIGNATURE_ALGORITHM = "Ed25519"
CRYPTOGRAPHY_REQUIREMENT = "cryptography>=44.0"

ROLELESS_FILENAME = "roleless.json"
RUN_SPEC_FILENAME = "run-spec.json"
PUBLIC_MANIFEST_FILENAME = "public-manifest.json"
GOLD_FILENAME = "holder-gold.json"
REVIEW_AUDIT_FILENAME = "review-audit.json"
HOLDER_MANIFEST_FILENAME = "holder-manifest.json"
RELEASE_GOLD_FILENAME = "gold.json"
RELEASE_MANIFEST_FILENAME = "release-manifest.json"
RELEASE_SIGNATURE_FILENAME = "release-manifest.ed25519"

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_RE = re.compile(r"^(?:nmrbc|rec|cand|sub)_[0-9a-f]{32}$")
_ZERO_HASH = "0" * 64
_MAX_JSON_BYTES = 128 * 1024 * 1024
_INDEPENDENCE_ATTESTATION = "completed_without_access_to_the_other_review"


class NMRBlindChallengeError(ValueError):
    """Raised when a blind-challenge invariant is violated."""


class NMRBlindChallengeDependencyError(RuntimeError):
    """Raised when Ed25519 support is unavailable or unsuitable."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes with a trailing newline."""

    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NMRBlindChallengeError("value is not finite canonical JSON") from exc
    return (rendered + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _duplicate_rejecting_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NMRBlindChallengeError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes, *, location: str) -> Any:
    """Decode strict JSON, rejecting duplicate keys and non-finite numbers."""

    if not payload or len(payload) > _MAX_JSON_BYTES:
        raise NMRBlindChallengeError(f"{location}: JSON size is invalid")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NMRBlindChallengeError(f"{location}: JSON must be UTF-8") from exc
    if text.startswith("\ufeff"):
        raise NMRBlindChallengeError(f"{location}: UTF-8 BOM is not allowed")

    def reject_constant(value: str) -> None:
        raise NMRBlindChallengeError(
            f"{location}: non-finite JSON number {value!r} is forbidden"
        )

    try:
        value = json.loads(
            text,
            object_pairs_hook=_duplicate_rejecting_object,
            parse_constant=reject_constant,
        )
    except json.JSONDecodeError as exc:
        raise NMRBlindChallengeError(f"{location}: invalid JSON") from exc
    _reject_nonfinite(value, location=location)
    return value


def _reject_nonfinite(value: Any, *, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise NMRBlindChallengeError(f"{location}: non-finite number is forbidden")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, location=f"{location}[{index}]")


def _safe_existing_file(path: str | Path, *, label: str) -> Path:
    selected = _absolute_without_resolving(path)
    _reject_symlink_components(selected, label=label, include_leaf=True)
    try:
        resolved = selected.resolve(strict=True)
    except OSError as exc:
        raise NMRBlindChallengeError(f"{label}: file does not exist") from exc
    if not resolved.is_file():
        raise NMRBlindChallengeError(f"{label}: expected a regular file")
    if resolved.stat().st_size > _MAX_JSON_BYTES:
        raise NMRBlindChallengeError(f"{label}: file is too large")
    return resolved


def _absolute_without_resolving(path: str | Path) -> Path:
    """Make an absolute lexical path without following filesystem links."""

    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _reject_symlink_components(
    path: Path,
    *,
    label: str,
    include_leaf: bool,
) -> None:
    parts = path.parts
    if not parts:
        raise NMRBlindChallengeError(f"{label}: empty path")
    current = Path(parts[0])
    stop = len(parts) if include_leaf else len(parts) - 1
    for part in parts[1:stop]:
        current /= part
        if current.is_symlink():
            raise NMRBlindChallengeError(
                f"{label}: symlinked path component is forbidden: {current}"
            )


def _read_strict_json(path: str | Path, *, label: str) -> tuple[Any, bytes, Path]:
    resolved = _safe_existing_file(path, label=label)
    payload = resolved.read_bytes()
    return strict_json_bytes(payload, location=str(resolved)), payload, resolved


def _exact_fields(value: Any, expected: set[str], *, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NMRBlindChallengeError(f"{location}: expected an object")
    actual = set(value)
    if actual != expected:
        raise NMRBlindChallengeError(
            f"{location}: field allowlist mismatch: {sorted(actual ^ expected)}"
        )
    return value


def _nonempty_text(
    value: Any,
    *,
    location: str,
    maximum: int = 4096,
) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
        or any(ord(character) < 32 for character in value)
    ):
        raise NMRBlindChallengeError(f"{location}: invalid non-empty text")
    return value


def _sha256(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise NMRBlindChallengeError(f"{location}: expected lowercase SHA-256")
    return value


def _utc(value: Any, *, location: str) -> str:
    text = _nonempty_text(value, location=location, maximum=64)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise NMRBlindChallengeError(
            f"{location}: expected an explicit UTC timestamp"
        ) from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
    ):
        raise NMRBlindChallengeError(f"{location}: expected an explicit UTC timestamp")
    return text


def _utc_datetime(value: Any, *, location: str) -> datetime:
    text = _utc(value, location=location)
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _now(clock: Callable[[], datetime] | None) -> str:
    current = (clock or (lambda: datetime.now(timezone.utc)))()
    if not isinstance(current, datetime) or current.tzinfo is None:
        raise NMRBlindChallengeError("clock must return a timezone-aware datetime")
    return current.astimezone(timezone.utc).isoformat()


def _opaque(value: Any, *, location: str, prefix: str | None = None) -> str:
    if not isinstance(value, str) or _OPAQUE_RE.fullmatch(value) is None:
        raise NMRBlindChallengeError(f"{location}: invalid opaque identifier")
    if prefix is not None and not value.startswith(f"{prefix}_"):
        raise NMRBlindChallengeError(f"{location}: wrong identifier kind")
    return value


def _new_opaque(prefix: str, token_hex: Callable[[int], str]) -> str:
    token = token_hex(16)
    if not isinstance(token, str) or re.fullmatch(r"[0-9a-f]{32}", token) is None:
        raise NMRBlindChallengeError("secure token provider returned invalid entropy")
    return f"{prefix}_{token}"


def _artifact_binding(filename: str, payload: bytes) -> dict[str, Any]:
    return {
        "filename": filename,
        "sha256": sha256_bytes(payload),
        "size_bytes": len(payload),
    }


def _validate_binding(value: Any, *, filename: str, location: str) -> dict[str, Any]:
    binding = _exact_fields(
        value,
        {"filename", "sha256", "size_bytes"},
        location=location,
    )
    if binding["filename"] != filename:
        raise NMRBlindChallengeError(f"{location}: filename changed")
    _sha256(binding["sha256"], location=f"{location}.sha256")
    if type(binding["size_bytes"]) is not int or binding["size_bytes"] <= 0:
        raise NMRBlindChallengeError(f"{location}.size_bytes: invalid size")
    return binding


def _check_binding(
    binding: Mapping[str, Any], payload: bytes, *, location: str
) -> None:
    if binding["size_bytes"] != len(payload):
        raise NMRBlindChallengeError(f"{location}: byte size mismatch")
    if binding["sha256"] != sha256_bytes(payload):
        raise NMRBlindChallengeError(f"{location}: SHA-256 mismatch")


def _safe_new_directory(path: str | Path, *, label: str) -> Path:
    selected = _absolute_without_resolving(path)
    _reject_symlink_components(selected, label=label, include_leaf=True)
    if selected.exists() or selected.is_symlink():
        raise NMRBlindChallengeError(f"{label}: destination already exists")
    parent = selected.parent.resolve(strict=True)
    return parent / selected.name


def _write_staged_directory(destination: Path, files: Mapping[str, bytes]) -> None:
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for filename, payload in files.items():
            if Path(filename).name != filename or filename in {"", ".", ".."}:
                raise NMRBlindChallengeError("unsafe artifact filename")
            target = staging / filename
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if destination.exists():
            raise NMRBlindChallengeError("destination appeared during publication")
        os.rename(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _validate_run_spec(value: Any) -> dict[str, Any]:
    if isinstance(value, dict) and value.get("schema_version") == LEGACY_RUN_SPEC_SCHEMA:
        raise NMRBlindChallengeError(
            "legacy run-spec v1 lacks a pre-submission holder-key commitment; "
            "rebuild with chemapp.nmr.blind-run-spec.v2"
        )
    spec = _exact_fields(
        value,
        {
            "schema_version",
            "protocol_id",
            "registered_at",
            "primary_endpoint",
            "holder_public_key_spki_sha256",
            "model_release_ids",
            "model_release_manifests",
            "analysis_plan",
            "analysis_plan_sha256",
        },
        location="run_spec",
    )
    if spec["schema_version"] != RUN_SPEC_SCHEMA:
        raise NMRBlindChallengeError("run_spec.schema_version changed")
    _nonempty_text(spec["protocol_id"], location="run_spec.protocol_id", maximum=128)
    _utc(spec["registered_at"], location="run_spec.registered_at")
    _nonempty_text(
        spec["primary_endpoint"],
        location="run_spec.primary_endpoint",
        maximum=512,
    )
    _sha256(
        spec["holder_public_key_spki_sha256"],
        location="run_spec.holder_public_key_spki_sha256",
    )
    model_ids = spec["model_release_ids"]
    if not isinstance(model_ids, list) or not model_ids or len(model_ids) > 32:
        raise NMRBlindChallengeError(
            "run_spec.model_release_ids must be a non-empty list"
        )
    validated_model_ids: list[str] = []
    for index, model_id in enumerate(model_ids):
        validated_model_ids.append(
            _nonempty_text(
                model_id,
                location=f"run_spec.model_release_ids[{index}]",
                maximum=256,
            )
        )
    if len(validated_model_ids) != len(set(validated_model_ids)):
        raise NMRBlindChallengeError("run_spec.model_release_ids must be unique")
    manifests = spec["model_release_manifests"]
    if not isinstance(manifests, dict) or set(manifests) != set(validated_model_ids):
        raise NMRBlindChallengeError(
            "run_spec.model_release_manifests must exactly cover model_release_ids"
        )
    for model_id in validated_model_ids:
        manifest = manifests[model_id]
        if not isinstance(manifest, dict) or not manifest:
            raise NMRBlindChallengeError(
                f"run_spec.model_release_manifests[{model_id!r}] must be a non-empty object"
            )
        canonical_json_bytes(manifest)
    analysis_plan = spec["analysis_plan"]
    if not isinstance(analysis_plan, dict) or not analysis_plan:
        raise NMRBlindChallengeError("run_spec.analysis_plan must be a non-empty object")
    expected_plan_sha256 = sha256_bytes(canonical_json_bytes(analysis_plan))
    actual_plan_sha256 = _sha256(
        spec["analysis_plan_sha256"],
        location="run_spec.analysis_plan_sha256",
    )
    if actual_plan_sha256 != expected_plan_sha256:
        raise NMRBlindChallengeError(
            "run_spec.analysis_plan_sha256 does not bind the embedded analysis_plan"
        )
    return spec


def _validate_decision(
    value: Any,
    *,
    candidate_ids: set[str],
    location: str,
    adjudication: bool,
) -> dict[str, Any]:
    base = {"decision", "selected_source_candidate_id"}
    expected = (
        base | {"adjudicator_id", "adjudicated_at", "rationale_sha256"}
        if adjudication
        else base
        | {
            "reviewer_id",
            "reviewed_at",
            "evidence_sha256",
            "independence_attestation",
        }
    )
    decision = _exact_fields(value, expected, location=location)
    if decision["decision"] not in {"include", "exclude"}:
        raise NMRBlindChallengeError(f"{location}.decision: invalid decision")
    selected = decision["selected_source_candidate_id"]
    if decision["decision"] == "include":
        if selected not in candidate_ids:
            raise NMRBlindChallengeError(f"{location}: selected candidate is unknown")
    elif selected is not None:
        raise NMRBlindChallengeError(
            f"{location}: excluded record cannot select a candidate"
        )
    if adjudication:
        _nonempty_text(
            decision["adjudicator_id"],
            location=f"{location}.adjudicator_id",
            maximum=256,
        )
        _utc(decision["adjudicated_at"], location=f"{location}.adjudicated_at")
        _sha256(
            decision["rationale_sha256"],
            location=f"{location}.rationale_sha256",
        )
    else:
        _nonempty_text(
            decision["reviewer_id"],
            location=f"{location}.reviewer_id",
            maximum=256,
        )
        _utc(decision["reviewed_at"], location=f"{location}.reviewed_at")
        _sha256(
            decision["evidence_sha256"],
            location=f"{location}.evidence_sha256",
        )
        if decision["independence_attestation"] != _INDEPENDENCE_ATTESTATION:
            raise NMRBlindChallengeError(
                f"{location}: independent-review attestation is required"
            )
    return decision


def _validate_adjudication_order(
    reviews: Sequence[Mapping[str, Any]],
    adjudication: Mapping[str, Any],
    *,
    location: str,
) -> None:
    """Require a disagreement adjudication to follow both source reviews."""

    latest_review = max(
        _utc_datetime(
            review["reviewed_at"],
            location=f"{location}.reviews[{index}].reviewed_at",
        )
        for index, review in enumerate(reviews)
    )
    adjudicated_at = _utc_datetime(
        adjudication["adjudicated_at"],
        location=f"{location}.adjudication.adjudicated_at",
    )
    if adjudicated_at <= latest_review:
        raise NMRBlindChallengeError(
            f"{location}: adjudication must be strictly later than both reviews"
        )


def _validate_review_bundle(value: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundle = _exact_fields(
        value,
        {"schema_version", "dataset_id", "records"},
        location="review_bundle",
    )
    if bundle["schema_version"] != REVIEW_BUNDLE_SCHEMA:
        raise NMRBlindChallengeError("review_bundle.schema_version changed")
    _nonempty_text(
        bundle["dataset_id"], location="review_bundle.dataset_id", maximum=256
    )
    rows = bundle["records"]
    if not isinstance(rows, list) or not rows or len(rows) > 100_000:
        raise NMRBlindChallengeError("review_bundle.records must be non-empty")

    normalized: list[dict[str, Any]] = []
    seen_source_records: set[tuple[str, str]] = set()
    seen_fingerprints: set[str] = set()
    for index, raw in enumerate(rows):
        location = f"review_bundle.records[{index}]"
        row = _exact_fields(
            raw,
            {"source", "spectrum", "candidates", "reviews", "adjudication"},
            location=location,
        )
        source = _exact_fields(
            row["source"],
            {"collection_id", "record_id", "record_locator_sha256"},
            location=f"{location}.source",
        )
        collection_id = _nonempty_text(
            source["collection_id"],
            location=f"{location}.source.collection_id",
            maximum=256,
        )
        source_record_id = _nonempty_text(
            source["record_id"],
            location=f"{location}.source.record_id",
            maximum=512,
        )
        source_key = (collection_id, source_record_id)
        if source_key in seen_source_records:
            raise NMRBlindChallengeError(f"{location}: duplicate source record")
        seen_source_records.add(source_key)
        _sha256(
            source["record_locator_sha256"],
            location=f"{location}.source.record_locator_sha256",
        )

        spectrum = _exact_fields(
            row["spectrum"],
            {"nucleus", "shifts_ppm", "spectrum_fingerprint_sha256"},
            location=f"{location}.spectrum",
        )
        nucleus = _nonempty_text(
            spectrum["nucleus"],
            location=f"{location}.spectrum.nucleus",
            maximum=32,
        )
        if nucleus not in {"13C", "1H", "13C+1H"}:
            raise NMRBlindChallengeError(f"{location}.spectrum.nucleus is unsupported")
        shifts = spectrum["shifts_ppm"]
        if not isinstance(shifts, list) or not shifts or len(shifts) > 8192:
            raise NMRBlindChallengeError(f"{location}.spectrum.shifts_ppm is invalid")
        for shift_index, shift in enumerate(shifts):
            if (
                isinstance(shift, bool)
                or not isinstance(shift, (int, float))
                or not math.isfinite(float(shift))
                or abs(float(shift)) > 100_000
            ):
                raise NMRBlindChallengeError(
                    f"{location}.spectrum.shifts_ppm[{shift_index}] is invalid"
                )
        fingerprint = _sha256(
            spectrum["spectrum_fingerprint_sha256"],
            location=f"{location}.spectrum.spectrum_fingerprint_sha256",
        )
        if fingerprint in seen_fingerprints:
            raise NMRBlindChallengeError(f"{location}: duplicate spectrum fingerprint")
        seen_fingerprints.add(fingerprint)

        candidates = row["candidates"]
        if not isinstance(candidates, list) or not 2 <= len(candidates) <= 10_000:
            raise NMRBlindChallengeError(f"{location}.candidates: expected 2..10000")
        source_candidate_ids: set[str] = set()
        structures: set[str] = set()
        for candidate_index, raw_candidate in enumerate(candidates):
            candidate_location = f"{location}.candidates[{candidate_index}]"
            candidate = _exact_fields(
                raw_candidate,
                {"source_candidate_id", "smiles"},
                location=candidate_location,
            )
            source_candidate_id = _nonempty_text(
                candidate["source_candidate_id"],
                location=f"{candidate_location}.source_candidate_id",
                maximum=512,
            )
            smiles = _nonempty_text(
                candidate["smiles"],
                location=f"{candidate_location}.smiles",
                maximum=8192,
            )
            if source_candidate_id in source_candidate_ids:
                raise NMRBlindChallengeError(f"{candidate_location}: duplicate ID")
            if smiles in structures:
                raise NMRBlindChallengeError(
                    f"{candidate_location}: duplicate structure"
                )
            source_candidate_ids.add(source_candidate_id)
            structures.add(smiles)

        reviews = row["reviews"]
        if not isinstance(reviews, list) or len(reviews) != 2:
            raise NMRBlindChallengeError(f"{location}: exactly two reviews required")
        validated_reviews = [
            _validate_decision(
                review,
                candidate_ids=source_candidate_ids,
                location=f"{location}.reviews[{review_index}]",
                adjudication=False,
            )
            for review_index, review in enumerate(reviews)
        ]
        reviewer_ids = [review["reviewer_id"] for review in validated_reviews]
        if len(set(reviewer_ids)) != 2:
            raise NMRBlindChallengeError(f"{location}: reviewers are not independent")
        decisions = [
            (review["decision"], review["selected_source_candidate_id"])
            for review in validated_reviews
        ]
        if decisions[0] == decisions[1]:
            if row["adjudication"] is not None:
                raise NMRBlindChallengeError(
                    f"{location}: agreement must not be post-hoc adjudicated"
                )
            final = validated_reviews[0]
        else:
            if row["adjudication"] is None:
                raise NMRBlindChallengeError(
                    f"{location}: reviewer disagreement requires adjudication"
                )
            final = _validate_decision(
                row["adjudication"],
                candidate_ids=source_candidate_ids,
                location=f"{location}.adjudication",
                adjudication=True,
            )
            if final["adjudicator_id"] in reviewer_ids:
                raise NMRBlindChallengeError(
                    f"{location}: adjudicator must differ from both reviewers"
                )
            _validate_adjudication_order(
                validated_reviews,
                final,
                location=location,
            )
        normalized.append(
            {
                "row": row,
                "included": final["decision"] == "include",
                "truth_source_candidate_id": final["selected_source_candidate_id"],
            }
        )
    if sum(item["included"] for item in normalized) < 2:
        raise NMRBlindChallengeError("at least two reviewed records must be included")
    return bundle, normalized


def build_blind_challenge(
    reviewed_input_path: str | Path,
    run_spec_path: str | Path,
    public_directory: str | Path,
    holder_directory: str | Path,
    *,
    _clock: Callable[[], datetime] | None = None,
    _token_hex: Callable[[int], str] = secrets.token_hex,
    _shuffle: Callable[[list[Any]], None] | None = None,
) -> dict[str, Any]:
    """Build disjoint public and holder-only directories without overwriting."""

    review_value, review_bytes, _ = _read_strict_json(
        reviewed_input_path,
        label="reviewed input",
    )
    run_value, _, _ = _read_strict_json(run_spec_path, label="run spec")
    bundle, normalized = _validate_review_bundle(review_value)
    run_spec = _validate_run_spec(run_value)
    public_candidate = _absolute_without_resolving(public_directory)
    holder_candidate = _absolute_without_resolving(holder_directory)
    if (
        public_candidate == holder_candidate
        or public_candidate in holder_candidate.parents
        or holder_candidate in public_candidate.parents
    ):
        raise NMRBlindChallengeError(
            "public and holder directories must be distinct and non-nested"
        )
    public = _safe_new_directory(public_directory, label="public directory")
    holder = _safe_new_directory(holder_directory, label="holder directory")

    created_at = _now(_clock)
    challenge_id = _new_opaque("nmrbc", _token_hex)
    used_ids: set[str] = {challenge_id}

    def fresh(prefix: str) -> str:
        for _ in range(1000):
            candidate = _new_opaque(prefix, _token_hex)
            if candidate not in used_ids:
                used_ids.add(candidate)
                return candidate
        raise NMRBlindChallengeError("could not allocate a unique opaque identifier")

    roleless_records: list[dict[str, Any]] = []
    gold_records: list[dict[str, Any]] = []
    audit_records: list[dict[str, Any]] = []
    shuffle = _shuffle or secrets.SystemRandom().shuffle
    for item in normalized:
        row = item["row"]
        if not item["included"]:
            audit_records.append(
                {
                    "record_id": None,
                    "source": row["source"],
                    "spectrum_fingerprint_sha256": row["spectrum"][
                        "spectrum_fingerprint_sha256"
                    ],
                    "candidate_map": [
                        {
                            "candidate_id": None,
                            "source_candidate_id": candidate["source_candidate_id"],
                            "candidate_smiles_sha256": sha256_bytes(
                                candidate["smiles"].encode("utf-8")
                            ),
                        }
                        for candidate in row["candidates"]
                    ],
                    "reviews": row["reviews"],
                    "adjudication": row["adjudication"],
                    "included": False,
                    "truth_candidate_id": None,
                }
            )
            continue
        record_id = fresh("rec")
        candidate_map: list[dict[str, Any]] = []
        public_candidates: list[dict[str, str]] = []
        source_to_opaque: dict[str, str] = {}
        for source_candidate in row["candidates"]:
            candidate_id = fresh("cand")
            source_candidate_id = source_candidate["source_candidate_id"]
            source_to_opaque[source_candidate_id] = candidate_id
            candidate_map.append(
                {
                    "candidate_id": candidate_id,
                    "source_candidate_id": source_candidate_id,
                    "candidate_smiles_sha256": sha256_bytes(
                        source_candidate["smiles"].encode("utf-8")
                    ),
                }
            )
            public_candidates.append(
                {"candidate_id": candidate_id, "smiles": source_candidate["smiles"]}
            )
        shuffle(public_candidates)
        truth_candidate_id = source_to_opaque[item["truth_source_candidate_id"]]
        roleless_records.append(
            {
                "record_id": record_id,
                "spectrum": row["spectrum"],
                "candidates": public_candidates,
            }
        )
        gold_records.append(
            {"record_id": record_id, "truth_candidate_id": truth_candidate_id}
        )
        audit_records.append(
            {
                "record_id": record_id,
                "source": row["source"],
                "spectrum_fingerprint_sha256": row["spectrum"][
                    "spectrum_fingerprint_sha256"
                ],
                "candidate_map": candidate_map,
                "reviews": row["reviews"],
                "adjudication": row["adjudication"],
                "included": True,
                "truth_candidate_id": truth_candidate_id,
            }
        )
    shuffle(roleless_records)
    gold_records.sort(key=lambda row: row["record_id"])

    roleless = {
        "schema_version": ROLELESS_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "challenge_id": challenge_id,
        "records": roleless_records,
    }
    gold = {
        "schema_version": GOLD_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "challenge_id": challenge_id,
        "sealing_nonce": secrets.token_hex(32),
        "records": gold_records,
    }
    audit = {
        "schema_version": REVIEW_AUDIT_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "challenge_id": challenge_id,
        "source_dataset_id": bundle["dataset_id"],
        "source_review_bundle_sha256": sha256_bytes(review_bytes),
        "records": audit_records,
    }
    roleless_payload = canonical_json_bytes(roleless)
    run_spec_payload = canonical_json_bytes(run_spec)
    gold_payload = canonical_json_bytes(gold)
    audit_payload = canonical_json_bytes(audit)
    public_manifest = {
        "schema_version": PUBLIC_MANIFEST_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "challenge_id": challenge_id,
        "created_at": created_at,
        "counts": {
            "included_records": len(roleless_records),
            "excluded_records": len(normalized) - len(roleless_records),
            "candidates": sum(len(row["candidates"]) for row in roleless_records),
        },
        "artifacts": {
            "roleless": _artifact_binding(ROLELESS_FILENAME, roleless_payload),
            "run_spec": _artifact_binding(RUN_SPEC_FILENAME, run_spec_payload),
        },
        "gold_commitment": {
            "algorithm": "SHA-256",
            "sha256": sha256_bytes(gold_payload),
            "hidden_nonce_bits": 256,
        },
        "outcome_disclosure": (
            "gold bytes and hidden nonce withheld; salted Gold digest committed "
            "before predictions"
        ),
    }
    public_manifest_payload = canonical_json_bytes(public_manifest)
    holder_manifest = {
        "schema_version": HOLDER_MANIFEST_SCHEMA,
        "protocol_version": PROTOCOL_VERSION,
        "challenge_id": challenge_id,
        "created_at": created_at,
        "counts": public_manifest["counts"],
        "artifacts": {
            "public_manifest": _artifact_binding(
                PUBLIC_MANIFEST_FILENAME,
                public_manifest_payload,
            ),
            "roleless": _artifact_binding(ROLELESS_FILENAME, roleless_payload),
            "run_spec": _artifact_binding(RUN_SPEC_FILENAME, run_spec_payload),
            "gold": _artifact_binding(GOLD_FILENAME, gold_payload),
            "review_audit": _artifact_binding(REVIEW_AUDIT_FILENAME, audit_payload),
        },
        "release_policy": (
            "gold may be signed and released only after one complete prediction "
            "artifact has a valid hash-chain submission receipt"
        ),
    }
    holder_manifest_payload = canonical_json_bytes(holder_manifest)

    _write_staged_directory(
        public,
        {
            ROLELESS_FILENAME: roleless_payload,
            RUN_SPEC_FILENAME: run_spec_payload,
            PUBLIC_MANIFEST_FILENAME: public_manifest_payload,
        },
    )
    try:
        _write_staged_directory(
            holder,
            {
                GOLD_FILENAME: gold_payload,
                REVIEW_AUDIT_FILENAME: audit_payload,
                HOLDER_MANIFEST_FILENAME: holder_manifest_payload,
            },
        )
    except Exception:
        shutil.rmtree(public)
        raise
    return {
        "challenge_id": challenge_id,
        "public_directory": str(public),
        "holder_directory": str(holder),
        "included_records": len(roleless_records),
        "excluded_records": len(normalized) - len(roleless_records),
        "public_manifest_sha256": sha256_bytes(public_manifest_payload),
        "holder_manifest_sha256": sha256_bytes(holder_manifest_payload),
    }


def _validate_roleless(value: Any) -> dict[str, Any]:
    roleless = _exact_fields(
        value,
        {"schema_version", "protocol_version", "challenge_id", "records"},
        location="roleless",
    )
    if roleless["schema_version"] != ROLELESS_SCHEMA:
        raise NMRBlindChallengeError("roleless.schema_version changed")
    if roleless["protocol_version"] != PROTOCOL_VERSION:
        raise NMRBlindChallengeError("roleless.protocol_version changed")
    _opaque(roleless["challenge_id"], location="roleless.challenge_id", prefix="nmrbc")
    records = roleless["records"]
    if not isinstance(records, list) or len(records) < 2:
        raise NMRBlindChallengeError("roleless.records must contain at least two rows")
    record_ids: set[str] = set()
    candidate_ids_global: set[str] = set()
    fingerprints: set[str] = set()
    for index, raw in enumerate(records):
        location = f"roleless.records[{index}]"
        row = _exact_fields(
            raw,
            {"record_id", "spectrum", "candidates"},
            location=location,
        )
        record_id = _opaque(
            row["record_id"], location=f"{location}.record_id", prefix="rec"
        )
        if record_id in record_ids:
            raise NMRBlindChallengeError(f"{location}: duplicate record ID")
        record_ids.add(record_id)
        spectrum = _exact_fields(
            row["spectrum"],
            {"nucleus", "shifts_ppm", "spectrum_fingerprint_sha256"},
            location=f"{location}.spectrum",
        )
        if spectrum["nucleus"] not in {"13C", "1H", "13C+1H"}:
            raise NMRBlindChallengeError(f"{location}: invalid nucleus")
        shifts = spectrum["shifts_ppm"]
        if not isinstance(shifts, list) or not shifts:
            raise NMRBlindChallengeError(f"{location}: shifts are missing")
        for shift in shifts:
            if (
                isinstance(shift, bool)
                or not isinstance(shift, (int, float))
                or not math.isfinite(float(shift))
            ):
                raise NMRBlindChallengeError(f"{location}: invalid shift")
        fingerprint = _sha256(
            spectrum["spectrum_fingerprint_sha256"],
            location=f"{location}.spectrum.spectrum_fingerprint_sha256",
        )
        if fingerprint in fingerprints:
            raise NMRBlindChallengeError(f"{location}: duplicate spectrum fingerprint")
        fingerprints.add(fingerprint)
        candidates = row["candidates"]
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise NMRBlindChallengeError(f"{location}: too few candidates")
        local_ids: set[str] = set()
        local_smiles: set[str] = set()
        for candidate_index, raw_candidate in enumerate(candidates):
            candidate_location = f"{location}.candidates[{candidate_index}]"
            candidate = _exact_fields(
                raw_candidate,
                {"candidate_id", "smiles"},
                location=candidate_location,
            )
            candidate_id = _opaque(
                candidate["candidate_id"],
                location=f"{candidate_location}.candidate_id",
                prefix="cand",
            )
            smiles = _nonempty_text(
                candidate["smiles"],
                location=f"{candidate_location}.smiles",
                maximum=8192,
            )
            if candidate_id in local_ids or candidate_id in candidate_ids_global:
                raise NMRBlindChallengeError(f"{candidate_location}: duplicate ID")
            if smiles in local_smiles:
                raise NMRBlindChallengeError(
                    f"{candidate_location}: duplicate structure"
                )
            local_ids.add(candidate_id)
            candidate_ids_global.add(candidate_id)
            local_smiles.add(smiles)
    return roleless


def _validate_gold(value: Any, roleless: Mapping[str, Any]) -> dict[str, Any]:
    gold = _exact_fields(
        value,
        {
            "schema_version",
            "protocol_version",
            "challenge_id",
            "sealing_nonce",
            "records",
        },
        location="gold",
    )
    if gold["schema_version"] != GOLD_SCHEMA:
        raise NMRBlindChallengeError("gold.schema_version changed")
    if gold["protocol_version"] != PROTOCOL_VERSION:
        raise NMRBlindChallengeError("gold.protocol_version changed")
    if gold["challenge_id"] != roleless["challenge_id"]:
        raise NMRBlindChallengeError("gold challenge does not match roleless")
    if (
        not isinstance(gold["sealing_nonce"], str)
        or re.fullmatch(r"[0-9a-f]{64}", gold["sealing_nonce"]) is None
    ):
        raise NMRBlindChallengeError("gold.sealing_nonce is invalid")
    expected = {
        row["record_id"]: {candidate["candidate_id"] for candidate in row["candidates"]}
        for row in roleless["records"]
    }
    rows = gold["records"]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise NMRBlindChallengeError("gold record count changed")
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        location = f"gold.records[{index}]"
        row = _exact_fields(
            raw,
            {"record_id", "truth_candidate_id"},
            location=location,
        )
        record_id = row["record_id"]
        if record_id not in expected or record_id in seen:
            raise NMRBlindChallengeError(f"{location}: unknown or duplicate record")
        if row["truth_candidate_id"] not in expected[record_id]:
            raise NMRBlindChallengeError(f"{location}: truth candidate is not in pool")
        seen.add(record_id)
    if seen != set(expected):
        raise NMRBlindChallengeError("gold does not cover every roleless record")
    return gold


def _validate_review_audit(
    value: Any,
    roleless: Mapping[str, Any],
    gold: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate recorded review semantics and every source-to-opaque truth mapping.

    Each reviewer record carries the required independence declaration.  The
    reviewers do not sign it individually: the holder's final release signature
    covers the complete review audit.  Neither mechanism can technically prove
    that the humans did not communicate or view one another's work.
    """

    audit = _exact_fields(
        value,
        {
            "schema_version",
            "protocol_version",
            "challenge_id",
            "source_dataset_id",
            "source_review_bundle_sha256",
            "records",
        },
        location="review_audit",
    )
    if audit["schema_version"] != REVIEW_AUDIT_SCHEMA:
        raise NMRBlindChallengeError("review_audit.schema_version changed")
    if audit["protocol_version"] != PROTOCOL_VERSION:
        raise NMRBlindChallengeError("review_audit.protocol_version changed")
    if audit["challenge_id"] != roleless["challenge_id"]:
        raise NMRBlindChallengeError("review audit challenge changed")
    _nonempty_text(
        audit["source_dataset_id"],
        location="review_audit.source_dataset_id",
        maximum=256,
    )
    _sha256(
        audit["source_review_bundle_sha256"],
        location="review_audit.source_review_bundle_sha256",
    )
    rows = audit["records"]
    if not isinstance(rows, list) or len(rows) < len(roleless["records"]):
        raise NMRBlindChallengeError("review audit row count is invalid")
    public_by_record = {row["record_id"]: row for row in roleless["records"]}
    gold_by_record = {
        row["record_id"]: row["truth_candidate_id"] for row in gold["records"]
    }
    seen_source: set[tuple[str, str]] = set()
    seen_fingerprints: set[str] = set()
    seen_included: set[str] = set()
    seen_opaque_candidates: set[str] = set()
    for index, raw in enumerate(rows):
        location = f"review_audit.records[{index}]"
        row = _exact_fields(
            raw,
            {
                "record_id",
                "source",
                "spectrum_fingerprint_sha256",
                "candidate_map",
                "reviews",
                "adjudication",
                "included",
                "truth_candidate_id",
            },
            location=location,
        )
        source = _exact_fields(
            row["source"],
            {"collection_id", "record_id", "record_locator_sha256"},
            location=f"{location}.source",
        )
        source_key = (
            _nonempty_text(
                source["collection_id"],
                location=f"{location}.source.collection_id",
                maximum=256,
            ),
            _nonempty_text(
                source["record_id"],
                location=f"{location}.source.record_id",
                maximum=512,
            ),
        )
        if source_key in seen_source:
            raise NMRBlindChallengeError(f"{location}: duplicate source record")
        seen_source.add(source_key)
        _sha256(
            source["record_locator_sha256"],
            location=f"{location}.source.record_locator_sha256",
        )
        fingerprint = _sha256(
            row["spectrum_fingerprint_sha256"],
            location=f"{location}.spectrum_fingerprint_sha256",
        )
        if fingerprint in seen_fingerprints:
            raise NMRBlindChallengeError(f"{location}: duplicate spectrum fingerprint")
        seen_fingerprints.add(fingerprint)

        candidate_map = row["candidate_map"]
        if not isinstance(candidate_map, list) or len(candidate_map) < 2:
            raise NMRBlindChallengeError(f"{location}.candidate_map is invalid")
        source_candidate_ids: set[str] = set()
        opaque_to_source: dict[str, str] = {}
        source_to_opaque: dict[str, str | None] = {}
        for candidate_index, raw_mapping in enumerate(candidate_map):
            mapping_location = f"{location}.candidate_map[{candidate_index}]"
            mapping = _exact_fields(
                raw_mapping,
                {
                    "candidate_id",
                    "source_candidate_id",
                    "candidate_smiles_sha256",
                },
                location=mapping_location,
            )
            source_candidate_id = _nonempty_text(
                mapping["source_candidate_id"],
                location=f"{mapping_location}.source_candidate_id",
                maximum=512,
            )
            if source_candidate_id in source_candidate_ids:
                raise NMRBlindChallengeError(f"{mapping_location}: duplicate source ID")
            source_candidate_ids.add(source_candidate_id)
            smiles_sha = _sha256(
                mapping["candidate_smiles_sha256"],
                location=f"{mapping_location}.candidate_smiles_sha256",
            )
            candidate_id = mapping["candidate_id"]
            if candidate_id is not None:
                candidate_id = _opaque(
                    candidate_id,
                    location=f"{mapping_location}.candidate_id",
                    prefix="cand",
                )
                if (
                    candidate_id in opaque_to_source
                    or candidate_id in seen_opaque_candidates
                ):
                    raise NMRBlindChallengeError(
                        f"{mapping_location}: opaque mapping is not one-to-one"
                    )
                opaque_to_source[candidate_id] = source_candidate_id
                seen_opaque_candidates.add(candidate_id)
            source_to_opaque[source_candidate_id] = candidate_id
            mapping["candidate_smiles_sha256"] = smiles_sha

        reviews = row["reviews"]
        if not isinstance(reviews, list) or len(reviews) != 2:
            raise NMRBlindChallengeError(f"{location}: exactly two reviews required")
        validated_reviews = [
            _validate_decision(
                review,
                candidate_ids=source_candidate_ids,
                location=f"{location}.reviews[{review_index}]",
                adjudication=False,
            )
            for review_index, review in enumerate(reviews)
        ]
        reviewer_ids = [review["reviewer_id"] for review in validated_reviews]
        if len(set(reviewer_ids)) != 2:
            raise NMRBlindChallengeError(f"{location}: reviewer IDs are not distinct")
        decisions = [
            (review["decision"], review["selected_source_candidate_id"])
            for review in validated_reviews
        ]
        if decisions[0] == decisions[1]:
            if row["adjudication"] is not None:
                raise NMRBlindChallengeError(
                    f"{location}: agreed reviews cannot have adjudication"
                )
            final = validated_reviews[0]
        else:
            if row["adjudication"] is None:
                raise NMRBlindChallengeError(
                    f"{location}: disagreement lacks adjudication"
                )
            final = _validate_decision(
                row["adjudication"],
                candidate_ids=source_candidate_ids,
                location=f"{location}.adjudication",
                adjudication=True,
            )
            if final["adjudicator_id"] in reviewer_ids:
                raise NMRBlindChallengeError(
                    f"{location}: adjudicator is one of the reviewers"
                )
            _validate_adjudication_order(
                validated_reviews,
                final,
                location=location,
            )
        expected_included = final["decision"] == "include"
        if type(row["included"]) is not bool or row["included"] != expected_included:
            raise NMRBlindChallengeError(
                f"{location}: included flag contradicts reviews"
            )
        if expected_included:
            record_id = _opaque(
                row["record_id"],
                location=f"{location}.record_id",
                prefix="rec",
            )
            if record_id not in public_by_record or record_id in seen_included:
                raise NMRBlindChallengeError(f"{location}: included record is unknown")
            seen_included.add(record_id)
            public_row = public_by_record[record_id]
            public_candidate_ids = {
                candidate["candidate_id"] for candidate in public_row["candidates"]
            }
            if set(opaque_to_source) != public_candidate_ids:
                raise NMRBlindChallengeError(
                    f"{location}: candidate map is not a roleless-pool bijection"
                )
            public_smiles_hashes = {
                candidate["candidate_id"]: sha256_bytes(
                    candidate["smiles"].encode("utf-8")
                )
                for candidate in public_row["candidates"]
            }
            mapped_smiles_hashes = {
                mapping["candidate_id"]: mapping["candidate_smiles_sha256"]
                for mapping in candidate_map
            }
            if mapped_smiles_hashes != public_smiles_hashes:
                raise NMRBlindChallengeError(
                    f"{location}: source candidate-to-SMILES binding changed"
                )
            if public_row["spectrum"]["spectrum_fingerprint_sha256"] != fingerprint:
                raise NMRBlindChallengeError(
                    f"{location}: source spectrum maps to another roleless record"
                )
            truth_candidate_id = _opaque(
                row["truth_candidate_id"],
                location=f"{location}.truth_candidate_id",
                prefix="cand",
            )
            selected_source = final["selected_source_candidate_id"]
            if source_to_opaque[selected_source] != truth_candidate_id:
                raise NMRBlindChallengeError(
                    f"{location}: final review does not map to opaque truth"
                )
            if gold_by_record.get(record_id) != truth_candidate_id:
                raise NMRBlindChallengeError(
                    f"{location}: review truth does not match holder Gold"
                )
        else:
            if row["record_id"] is not None or row["truth_candidate_id"] is not None:
                raise NMRBlindChallengeError(
                    f"{location}: excluded row exposes opaque record or truth"
                )
            if opaque_to_source:
                raise NMRBlindChallengeError(
                    f"{location}: excluded row must not allocate opaque candidates"
                )
    if seen_included != set(public_by_record):
        raise NMRBlindChallengeError(
            "review audit does not cover every roleless record"
        )
    return audit


def _reviewed_counts(
    roleless: Mapping[str, Any],
    audit: Mapping[str, Any],
) -> dict[str, int]:
    return {
        "included_records": len(roleless["records"]),
        "excluded_records": sum(not row["included"] for row in audit["records"]),
        "candidates": sum(len(row["candidates"]) for row in roleless["records"]),
    }


def _validate_public_manifest(value: Any) -> dict[str, Any]:
    manifest = _exact_fields(
        value,
        {
            "schema_version",
            "protocol_version",
            "challenge_id",
            "created_at",
            "counts",
            "artifacts",
            "gold_commitment",
            "outcome_disclosure",
        },
        location="public_manifest",
    )
    if manifest["schema_version"] != PUBLIC_MANIFEST_SCHEMA:
        raise NMRBlindChallengeError("public_manifest.schema_version changed")
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise NMRBlindChallengeError("public_manifest.protocol_version changed")
    _opaque(
        manifest["challenge_id"],
        location="public_manifest.challenge_id",
        prefix="nmrbc",
    )
    _utc(manifest["created_at"], location="public_manifest.created_at")
    counts = _exact_fields(
        manifest["counts"],
        {"included_records", "excluded_records", "candidates"},
        location="public_manifest.counts",
    )
    for key, count in counts.items():
        if type(count) is not int or count < (2 if key == "included_records" else 0):
            raise NMRBlindChallengeError(f"public_manifest.counts.{key} is invalid")
    artifacts = _exact_fields(
        manifest["artifacts"],
        {"roleless", "run_spec"},
        location="public_manifest.artifacts",
    )
    _validate_binding(
        artifacts["roleless"],
        filename=ROLELESS_FILENAME,
        location="public_manifest.artifacts.roleless",
    )
    _validate_binding(
        artifacts["run_spec"],
        filename=RUN_SPEC_FILENAME,
        location="public_manifest.artifacts.run_spec",
    )
    commitment = _exact_fields(
        manifest["gold_commitment"],
        {"algorithm", "sha256", "hidden_nonce_bits"},
        location="public_manifest.gold_commitment",
    )
    if commitment["algorithm"] != "SHA-256" or commitment["hidden_nonce_bits"] != 256:
        raise NMRBlindChallengeError("public Gold commitment parameters changed")
    _sha256(
        commitment["sha256"],
        location="public_manifest.gold_commitment.sha256",
    )
    if (
        manifest["outcome_disclosure"]
        != "gold bytes and hidden nonce withheld; salted Gold digest committed before predictions"
    ):
        raise NMRBlindChallengeError("public manifest outcome policy changed")
    return manifest


def _validate_holder_manifest(value: Any) -> dict[str, Any]:
    manifest = _exact_fields(
        value,
        {
            "schema_version",
            "protocol_version",
            "challenge_id",
            "created_at",
            "counts",
            "artifacts",
            "release_policy",
        },
        location="holder_manifest",
    )
    if manifest["schema_version"] != HOLDER_MANIFEST_SCHEMA:
        raise NMRBlindChallengeError("holder_manifest.schema_version changed")
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise NMRBlindChallengeError("holder_manifest.protocol_version changed")
    _opaque(
        manifest["challenge_id"],
        location="holder_manifest.challenge_id",
        prefix="nmrbc",
    )
    _utc(manifest["created_at"], location="holder_manifest.created_at")
    _exact_fields(
        manifest["counts"],
        {"included_records", "excluded_records", "candidates"},
        location="holder_manifest.counts",
    )
    artifacts = _exact_fields(
        manifest["artifacts"],
        {"public_manifest", "roleless", "run_spec", "gold", "review_audit"},
        location="holder_manifest.artifacts",
    )
    filenames = {
        "public_manifest": PUBLIC_MANIFEST_FILENAME,
        "roleless": ROLELESS_FILENAME,
        "run_spec": RUN_SPEC_FILENAME,
        "gold": GOLD_FILENAME,
        "review_audit": REVIEW_AUDIT_FILENAME,
    }
    for key, filename in filenames.items():
        _validate_binding(
            artifacts[key],
            filename=filename,
            location=f"holder_manifest.artifacts.{key}",
        )
    _nonempty_text(
        manifest["release_policy"],
        location="holder_manifest.release_policy",
        maximum=512,
    )
    return manifest


def _load_and_verify_public(
    public_manifest_path: str | Path,
    roleless_path: str | Path,
    run_spec_path: str | Path,
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
]:
    manifest_value, manifest_payload, _ = _read_strict_json(
        public_manifest_path,
        label="public manifest",
    )
    roleless_value, roleless_payload, _ = _read_strict_json(
        roleless_path,
        label="roleless artifact",
    )
    run_value, run_payload, _ = _read_strict_json(run_spec_path, label="run spec")
    manifest = _validate_public_manifest(manifest_value)
    roleless = _validate_roleless(roleless_value)
    run_spec = _validate_run_spec(run_value)
    if manifest_payload != canonical_json_bytes(manifest):
        raise NMRBlindChallengeError("public manifest is not canonical JSON")
    if roleless_payload != canonical_json_bytes(roleless):
        raise NMRBlindChallengeError("roleless artifact is not canonical JSON")
    if run_payload != canonical_json_bytes(run_spec):
        raise NMRBlindChallengeError("run spec artifact is not canonical JSON")
    if roleless["challenge_id"] != manifest["challenge_id"]:
        raise NMRBlindChallengeError("roleless challenge does not match manifest")
    _check_binding(
        manifest["artifacts"]["roleless"],
        roleless_payload,
        location="roleless artifact",
    )
    _check_binding(
        manifest["artifacts"]["run_spec"],
        run_payload,
        location="run spec artifact",
    )
    expected_counts = {
        "included_records": len(roleless["records"]),
        "excluded_records": manifest["counts"]["excluded_records"],
        "candidates": sum(len(row["candidates"]) for row in roleless["records"]),
    }
    if manifest["counts"] != expected_counts:
        raise NMRBlindChallengeError("public manifest counts do not match roleless")
    return (
        manifest,
        manifest_payload,
        roleless,
        roleless_payload,
        run_spec,
        run_payload,
    )


def _validate_predictions(
    value: Any,
    roleless: Mapping[str, Any],
    run_spec: Mapping[str, Any],
) -> dict[str, Any]:
    predictions = _exact_fields(
        value,
        {
            "schema_version",
            "protocol_version",
            "challenge_id",
            "roleless_sha256",
            "run_spec_sha256",
            "model_release_id",
            "model_release_manifest_sha256",
            "created_at",
            "records",
        },
        location="predictions",
    )
    if predictions["schema_version"] != PREDICTIONS_SCHEMA:
        raise NMRBlindChallengeError("predictions.schema_version changed")
    if predictions["protocol_version"] != PROTOCOL_VERSION:
        raise NMRBlindChallengeError("predictions.protocol_version changed")
    if predictions["challenge_id"] != roleless["challenge_id"]:
        raise NMRBlindChallengeError("predictions challenge does not match roleless")
    _sha256(predictions["roleless_sha256"], location="predictions.roleless_sha256")
    _sha256(predictions["run_spec_sha256"], location="predictions.run_spec_sha256")
    _nonempty_text(
        predictions["model_release_id"],
        location="predictions.model_release_id",
        maximum=256,
    )
    if predictions["model_release_id"] not in run_spec["model_release_ids"]:
        raise NMRBlindChallengeError(
            "predictions.model_release_id was not frozen in the run spec"
        )
    manifest_sha256 = _sha256(
        predictions["model_release_manifest_sha256"],
        location="predictions.model_release_manifest_sha256",
    )
    expected_manifest_sha256 = sha256_bytes(
        canonical_json_bytes(
            run_spec["model_release_manifests"][predictions["model_release_id"]]
        )
    )
    if manifest_sha256 != expected_manifest_sha256:
        raise NMRBlindChallengeError(
            "predictions model-release manifest does not match the frozen run spec"
        )
    _utc(predictions["created_at"], location="predictions.created_at")
    expected = {
        row["record_id"]: [candidate["candidate_id"] for candidate in row["candidates"]]
        for row in roleless["records"]
    }
    rows = predictions["records"]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise NMRBlindChallengeError("predictions must cover every record exactly once")
    seen: set[str] = set()
    for index, raw in enumerate(rows):
        location = f"predictions.records[{index}]"
        row = _exact_fields(
            raw,
            {"record_id", "ranked_candidate_ids"},
            location=location,
        )
        record_id = row["record_id"]
        if record_id not in expected or record_id in seen:
            raise NMRBlindChallengeError(f"{location}: unknown or duplicate record")
        ranking = row["ranked_candidate_ids"]
        if not isinstance(ranking, list) or len(ranking) != len(expected[record_id]):
            raise NMRBlindChallengeError(
                f"{location}: ranking must be a complete candidate permutation"
            )
        validated_ranking = [
            _opaque(
                candidate_id,
                location=f"{location}.ranked_candidate_ids[{candidate_index}]",
                prefix="cand",
            )
            for candidate_index, candidate_id in enumerate(ranking)
        ]
        if set(validated_ranking) != set(expected[record_id]) or len(
            validated_ranking
        ) != len(set(validated_ranking)):
            raise NMRBlindChallengeError(
                f"{location}: ranking must be a complete candidate permutation"
            )
        seen.add(record_id)
    return predictions


def _receipt_hash(entry_without_hash: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(entry_without_hash))


def _validate_receipt_entry(
    value: Any,
    *,
    previous: str,
    location: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise NMRBlindChallengeError(f"{location}: receipt must be an object")
    event = value.get("event")
    common = {
        "schema_version",
        "protocol_version",
        "event",
        "challenge_id",
        "occurred_at",
        "previous_receipt_sha256",
        "receipt_sha256",
    }
    if event == "predictions_submitted_before_gold_release":
        expected = common | {
            "submission_id",
            "public_manifest_sha256",
            "roleless_sha256",
            "run_spec_sha256",
            "predictions_sha256",
            "predictions_size_bytes",
        }
    elif event == "gold_signed_and_released":
        expected = common | {
            "submission_receipt_sha256",
            "release_manifest_sha256",
            "signature_sha256",
        }
    else:
        raise NMRBlindChallengeError(f"{location}: unknown receipt event")
    entry = _exact_fields(value, expected, location=location)
    if entry["schema_version"] != RECEIPT_SCHEMA:
        raise NMRBlindChallengeError(f"{location}: receipt schema changed")
    if entry["protocol_version"] != PROTOCOL_VERSION:
        raise NMRBlindChallengeError(f"{location}: protocol changed")
    _opaque(entry["challenge_id"], location=f"{location}.challenge_id", prefix="nmrbc")
    _utc(entry["occurred_at"], location=f"{location}.occurred_at")
    if entry["previous_receipt_sha256"] != previous:
        raise NMRBlindChallengeError(f"{location}: receipt chain is broken")
    for key in expected:
        if key.endswith("_sha256"):
            _sha256(entry[key], location=f"{location}.{key}")
    if event == "predictions_submitted_before_gold_release":
        _opaque(
            entry["submission_id"], location=f"{location}.submission_id", prefix="sub"
        )
        if (
            type(entry["predictions_size_bytes"]) is not int
            or entry["predictions_size_bytes"] <= 0
        ):
            raise NMRBlindChallengeError(f"{location}: invalid prediction size")
    expected_hash = _receipt_hash(
        {key: item for key, item in entry.items() if key != "receipt_sha256"}
    )
    if entry["receipt_sha256"] != expected_hash:
        raise NMRBlindChallengeError(f"{location}: receipt hash mismatch")
    return entry


def _read_receipt_ledger_unlocked(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    if path.is_symlink() or not path.is_file():
        raise NMRBlindChallengeError(
            "receipt ledger must be a regular non-symlink file"
        )
    payload = path.read_bytes()
    if len(payload) > _MAX_JSON_BYTES:
        raise NMRBlindChallengeError("receipt ledger is too large")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NMRBlindChallengeError("receipt ledger must be UTF-8") from exc
    if not text or not text.endswith("\n"):
        raise NMRBlindChallengeError("receipt ledger is truncated or empty")
    entries: list[dict[str, Any]] = []
    previous = _ZERO_HASH
    challenge_events: dict[str, list[str]] = {}
    prediction_hashes: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise NMRBlindChallengeError(f"receipt ledger line {line_number} is blank")
        parsed = strict_json_bytes(
            line.encode("utf-8"), location=f"ledger:{line_number}"
        )
        entry = _validate_receipt_entry(
            parsed,
            previous=previous,
            location=f"ledger:{line_number}",
        )
        if canonical_json_bytes(entry).rstrip(b"\n") != line.encode("utf-8"):
            raise NMRBlindChallengeError(f"ledger:{line_number}: non-canonical receipt")
        challenge_id = entry["challenge_id"]
        history = challenge_events.setdefault(challenge_id, [])
        if entry["event"] == "predictions_submitted_before_gold_release":
            if history:
                raise NMRBlindChallengeError(
                    "challenge prediction receipt was replayed"
                )
            if entry["predictions_sha256"] in prediction_hashes:
                raise NMRBlindChallengeError("prediction artifact hash was replayed")
            prediction_hashes.add(entry["predictions_sha256"])
        else:
            if history != ["predictions_submitted_before_gold_release"]:
                raise NMRBlindChallengeError(
                    "gold release has no unique prior submission"
                )
        history.append(entry["event"])
        previous = entry["receipt_sha256"]
        entries.append(entry)
    return entries


@contextmanager
def _locked_ledger(path: str | Path):
    selected = _absolute_without_resolving(path)
    _reject_symlink_components(
        selected,
        label="receipt ledger",
        include_leaf=True,
    )
    parent = selected.parent.resolve(strict=True)
    ledger = parent / selected.name
    if ledger.is_symlink():
        raise NMRBlindChallengeError("symlinked receipt ledger is forbidden")
    lock = parent / f".{selected.name}.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise NMRBlindChallengeError(
            f"receipt ledger is locked; inspect stale lock {lock}"
        ) from exc
    try:
        yield ledger, _read_receipt_ledger_unlocked(ledger)
    finally:
        lock.rmdir()


def _write_receipt_ledger_unlocked(
    path: Path, entries: Sequence[Mapping[str, Any]]
) -> None:
    payload = b"".join(canonical_json_bytes(entry) for entry in entries)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".part",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def submit_blind_predictions(
    public_manifest_path: str | Path,
    roleless_path: str | Path,
    run_spec_path: str | Path,
    predictions_path: str | Path,
    receipt_ledger_path: str | Path,
    *,
    _clock: Callable[[], datetime] | None = None,
    _token_hex: Callable[[int], str] = secrets.token_hex,
) -> dict[str, Any]:
    """Commit one complete prediction artifact before any Gold is released."""

    (
        public_manifest,
        public_manifest_payload,
        roleless,
        roleless_payload,
        run_spec,
        run_payload,
    ) = _load_and_verify_public(public_manifest_path, roleless_path, run_spec_path)
    predictions_value, predictions_payload, _ = _read_strict_json(
        predictions_path,
        label="predictions",
    )
    predictions = _validate_predictions(predictions_value, roleless, run_spec)
    if predictions_payload != canonical_json_bytes(predictions):
        raise NMRBlindChallengeError("predictions must be canonical JSON")
    if predictions["roleless_sha256"] != sha256_bytes(roleless_payload):
        raise NMRBlindChallengeError("predictions roleless binding changed")
    if predictions["run_spec_sha256"] != sha256_bytes(run_payload):
        raise NMRBlindChallengeError("predictions run-spec binding changed")
    registered_at = _utc_datetime(
        run_spec["registered_at"],
        location="run_spec.registered_at",
    )
    predictions_at = _utc_datetime(
        predictions["created_at"],
        location="predictions.created_at",
    )
    if predictions_at < registered_at:
        raise NMRBlindChallengeError("predictions predate run-spec registration")

    with _locked_ledger(receipt_ledger_path) as (ledger, entries):
        if any(entry["challenge_id"] == roleless["challenge_id"] for entry in entries):
            raise NMRBlindChallengeError(
                "challenge already has a prediction receipt; replay is forbidden"
            )
        prediction_hash = sha256_bytes(predictions_payload)
        if any(entry.get("predictions_sha256") == prediction_hash for entry in entries):
            raise NMRBlindChallengeError("prediction artifact was already submitted")
        previous = entries[-1]["receipt_sha256"] if entries else _ZERO_HASH
        occurred_at = _now(_clock)
        if (
            _utc_datetime(occurred_at, location="submission occurred_at")
            < predictions_at
        ):
            raise NMRBlindChallengeError("prediction receipt predates predictions")
        body = {
            "schema_version": RECEIPT_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "event": "predictions_submitted_before_gold_release",
            "challenge_id": roleless["challenge_id"],
            "submission_id": _new_opaque("sub", _token_hex),
            "public_manifest_sha256": sha256_bytes(public_manifest_payload),
            "roleless_sha256": sha256_bytes(roleless_payload),
            "run_spec_sha256": sha256_bytes(run_payload),
            "predictions_sha256": prediction_hash,
            "predictions_size_bytes": len(predictions_payload),
            "occurred_at": occurred_at,
            "previous_receipt_sha256": previous,
        }
        entry = {**body, "receipt_sha256": _receipt_hash(body)}
        _validate_receipt_entry(entry, previous=previous, location="new receipt")
        _write_receipt_ledger_unlocked(ledger, [*entries, entry])
    return entry


def _require_ed25519() -> dict[str, Any]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except (ImportError, ModuleNotFoundError) as exc:
        raise NMRBlindChallengeDependencyError(
            "Ed25519 release is disabled: install " + CRYPTOGRAPHY_REQUIREMENT
        ) from exc
    return {
        "serialization": serialization,
        "private_type": Ed25519PrivateKey,
        "public_type": Ed25519PublicKey,
    }


def _load_private_key(path: str | Path) -> tuple[Any, Any, str]:
    crypto = _require_ed25519()
    key_path = _safe_existing_file(path, label="Ed25519 private key")
    if os.name != "nt" and key_path.stat().st_mode & 0o077:
        raise NMRBlindChallengeDependencyError(
            "Ed25519 private key must not be group- or world-readable"
        )
    try:
        key = crypto["serialization"].load_pem_private_key(
            key_path.read_bytes(),
            password=None,
        )
    except (TypeError, ValueError) as exc:
        raise NMRBlindChallengeDependencyError(
            "private key must be an unencrypted Ed25519 PEM key"
        ) from exc
    if not isinstance(key, crypto["private_type"]):
        raise NMRBlindChallengeDependencyError("private key is not Ed25519")
    public_key = key.public_key()
    public_der = public_key.public_bytes(
        encoding=crypto["serialization"].Encoding.DER,
        format=crypto["serialization"].PublicFormat.SubjectPublicKeyInfo,
    )
    return key, public_key, sha256_bytes(public_der)


def _load_public_key(path: str | Path) -> tuple[Any, str]:
    crypto = _require_ed25519()
    key_path = _safe_existing_file(path, label="Ed25519 public key")
    try:
        key = crypto["serialization"].load_pem_public_key(key_path.read_bytes())
    except (TypeError, ValueError) as exc:
        raise NMRBlindChallengeDependencyError(
            "public key must be an Ed25519 PEM key"
        ) from exc
    if not isinstance(key, crypto["public_type"]):
        raise NMRBlindChallengeDependencyError("public key is not Ed25519")
    public_der = key.public_bytes(
        encoding=crypto["serialization"].Encoding.DER,
        format=crypto["serialization"].PublicFormat.SubjectPublicKeyInfo,
    )
    return key, sha256_bytes(public_der)


def release_blind_gold(
    *,
    holder_manifest_path: str | Path,
    public_manifest_path: str | Path,
    roleless_path: str | Path,
    run_spec_path: str | Path,
    holder_gold_path: str | Path,
    review_audit_path: str | Path,
    predictions_path: str | Path,
    receipt_ledger_path: str | Path,
    private_key_path: str | Path,
    release_directory: str | Path,
    _clock: Callable[[], datetime] | None = None,
) -> dict[str, Any]:
    """Sign and release Gold after a unique prediction receipt is committed."""

    release = _safe_new_directory(release_directory, label="release directory")
    private_key, _, public_key_sha = _load_private_key(private_key_path)
    (
        public_manifest,
        public_manifest_payload,
        roleless,
        roleless_payload,
        run_spec,
        run_payload,
    ) = _load_and_verify_public(public_manifest_path, roleless_path, run_spec_path)
    if public_key_sha != run_spec["holder_public_key_spki_sha256"]:
        raise NMRBlindChallengeError(
            "signing key differs from the holder key committed before prediction submission"
        )
    holder_value, holder_payload, _ = _read_strict_json(
        holder_manifest_path,
        label="holder manifest",
    )
    gold_value, gold_payload, _ = _read_strict_json(
        holder_gold_path,
        label="holder Gold",
    )
    audit_value, audit_payload, _ = _read_strict_json(
        review_audit_path,
        label="review audit",
    )
    predictions_value, predictions_payload, _ = _read_strict_json(
        predictions_path,
        label="predictions",
    )
    holder_manifest = _validate_holder_manifest(holder_value)
    gold = _validate_gold(gold_value, roleless)
    audit = _validate_review_audit(audit_value, roleless, gold)
    predictions = _validate_predictions(predictions_value, roleless, run_spec)
    if holder_payload != canonical_json_bytes(holder_manifest):
        raise NMRBlindChallengeError("holder manifest is not canonical JSON")
    if gold_payload != canonical_json_bytes(gold):
        raise NMRBlindChallengeError("holder Gold is not canonical JSON")
    if predictions_payload != canonical_json_bytes(predictions):
        raise NMRBlindChallengeError("predictions are not canonical JSON")
    if (
        not isinstance(audit_value, dict)
        or audit_value.get("schema_version") != REVIEW_AUDIT_SCHEMA
    ):
        raise NMRBlindChallengeError("review audit schema changed")
    if audit_payload != canonical_json_bytes(audit_value):
        raise NMRBlindChallengeError("review audit is not canonical JSON")
    challenge_id = roleless["challenge_id"]
    if holder_manifest["challenge_id"] != challenge_id:
        raise NMRBlindChallengeError("holder manifest challenge changed")
    if audit_value.get("challenge_id") != challenge_id:
        raise NMRBlindChallengeError("review audit challenge changed")
    holder_bindings = holder_manifest["artifacts"]
    bound_payloads = {
        "public_manifest": public_manifest_payload,
        "roleless": roleless_payload,
        "run_spec": run_payload,
        "gold": gold_payload,
        "review_audit": audit_payload,
    }
    for key, payload in bound_payloads.items():
        _check_binding(holder_bindings[key], payload, location=f"holder {key}")
    if holder_manifest["counts"] != public_manifest["counts"]:
        raise NMRBlindChallengeError("holder and public counts differ")
    if public_manifest["counts"] != _reviewed_counts(roleless, audit):
        raise NMRBlindChallengeError("manifest counts do not match review audit")
    if public_manifest["gold_commitment"]["sha256"] != sha256_bytes(gold_payload):
        raise NMRBlindChallengeError(
            "holder Gold differs from pre-prediction commitment"
        )
    if predictions["roleless_sha256"] != sha256_bytes(roleless_payload):
        raise NMRBlindChallengeError("predictions roleless binding changed")
    if predictions["run_spec_sha256"] != sha256_bytes(run_payload):
        raise NMRBlindChallengeError("predictions run-spec binding changed")

    with _locked_ledger(receipt_ledger_path) as (ledger, entries):
        submissions = [
            entry
            for entry in entries
            if entry["challenge_id"] == challenge_id
            and entry["event"] == "predictions_submitted_before_gold_release"
        ]
        releases = [
            entry
            for entry in entries
            if entry["challenge_id"] == challenge_id
            and entry["event"] == "gold_signed_and_released"
        ]
        if len(submissions) != 1:
            raise NMRBlindChallengeError(
                "Gold remains sealed until one unique prediction receipt exists"
            )
        if releases:
            raise NMRBlindChallengeError(
                "Gold was already released; replay is forbidden"
            )
        submission = submissions[0]
        expected_submission = {
            "public_manifest_sha256": sha256_bytes(public_manifest_payload),
            "roleless_sha256": sha256_bytes(roleless_payload),
            "run_spec_sha256": sha256_bytes(run_payload),
            "predictions_sha256": sha256_bytes(predictions_payload),
            "predictions_size_bytes": len(predictions_payload),
        }
        for key, expected in expected_submission.items():
            if submission[key] != expected:
                raise NMRBlindChallengeError(
                    f"prediction receipt does not bind current {key}"
                )

        released_at = _now(_clock)
        if _utc_datetime(
            released_at,
            location="release released_at",
        ) < _utc_datetime(
            submission["occurred_at"],
            location="submission occurred_at",
        ):
            raise NMRBlindChallengeError("Gold release predates prediction submission")
        release_manifest = {
            "schema_version": RELEASE_MANIFEST_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "challenge_id": challenge_id,
            "released_at": released_at,
            "signature": {
                "algorithm": SIGNATURE_ALGORITHM,
                "public_key_spki_sha256": public_key_sha,
                "signed_bytes": "exact canonical UTF-8 release-manifest.json bytes",
            },
            "prediction_submission_receipt_sha256": submission["receipt_sha256"],
            "artifacts": {
                "public_manifest": _artifact_binding(
                    PUBLIC_MANIFEST_FILENAME,
                    public_manifest_payload,
                ),
                "holder_manifest": _artifact_binding(
                    HOLDER_MANIFEST_FILENAME,
                    holder_payload,
                ),
                "roleless": _artifact_binding(ROLELESS_FILENAME, roleless_payload),
                "run_spec": _artifact_binding(RUN_SPEC_FILENAME, run_payload),
                "gold": _artifact_binding(RELEASE_GOLD_FILENAME, gold_payload),
                "predictions": _artifact_binding(
                    "predictions.json", predictions_payload
                ),
                "review_audit": _artifact_binding(
                    REVIEW_AUDIT_FILENAME,
                    audit_payload,
                ),
            },
            "receipt_ledger_limit": (
                "local hash-chain detects mutation/replay but is not WORM, a trusted "
                "timestamp, or an external transparency log"
            ),
        }
        release_manifest_payload = canonical_json_bytes(release_manifest)
        signature = private_key.sign(release_manifest_payload)
        if not isinstance(signature, bytes) or len(signature) != 64:
            raise NMRBlindChallengeDependencyError(
                "Ed25519 signer returned invalid bytes"
            )
        _write_staged_directory(
            release,
            {
                RELEASE_GOLD_FILENAME: gold_payload,
                RELEASE_MANIFEST_FILENAME: release_manifest_payload,
                RELEASE_SIGNATURE_FILENAME: signature,
            },
        )
        previous = entries[-1]["receipt_sha256"] if entries else _ZERO_HASH
        release_body = {
            "schema_version": RECEIPT_SCHEMA,
            "protocol_version": PROTOCOL_VERSION,
            "event": "gold_signed_and_released",
            "challenge_id": challenge_id,
            "submission_receipt_sha256": submission["receipt_sha256"],
            "release_manifest_sha256": sha256_bytes(release_manifest_payload),
            "signature_sha256": sha256_bytes(signature),
            "occurred_at": released_at,
            "previous_receipt_sha256": previous,
        }
        release_entry = {
            **release_body,
            "receipt_sha256": _receipt_hash(release_body),
        }
        _validate_receipt_entry(
            release_entry,
            previous=previous,
            location="new release receipt",
        )
        try:
            _write_receipt_ledger_unlocked(ledger, [*entries, release_entry])
        except Exception as exc:
            raise NMRBlindChallengeError(
                "signed release was published but its local release receipt could not "
                f"be appended; preserve {release} and investigate"
            ) from exc
    return {
        "challenge_id": challenge_id,
        "release_directory": str(release),
        "release_manifest_sha256": sha256_bytes(release_manifest_payload),
        "signature_sha256": sha256_bytes(signature),
        "submission_receipt_sha256": submission["receipt_sha256"],
        "release_receipt_sha256": release_entry["receipt_sha256"],
    }


def _validate_release_manifest(value: Any) -> dict[str, Any]:
    manifest = _exact_fields(
        value,
        {
            "schema_version",
            "protocol_version",
            "challenge_id",
            "released_at",
            "signature",
            "prediction_submission_receipt_sha256",
            "artifacts",
            "receipt_ledger_limit",
        },
        location="release_manifest",
    )
    if manifest["schema_version"] != RELEASE_MANIFEST_SCHEMA:
        raise NMRBlindChallengeError("release manifest schema changed")
    if manifest["protocol_version"] != PROTOCOL_VERSION:
        raise NMRBlindChallengeError("release manifest protocol changed")
    _opaque(
        manifest["challenge_id"],
        location="release_manifest.challenge_id",
        prefix="nmrbc",
    )
    _utc(manifest["released_at"], location="release_manifest.released_at")
    signature = _exact_fields(
        manifest["signature"],
        {"algorithm", "public_key_spki_sha256", "signed_bytes"},
        location="release_manifest.signature",
    )
    if signature["algorithm"] != SIGNATURE_ALGORITHM:
        raise NMRBlindChallengeError("release signature algorithm changed")
    _sha256(
        signature["public_key_spki_sha256"],
        location="release_manifest.signature.public_key_spki_sha256",
    )
    if signature["signed_bytes"] != "exact canonical UTF-8 release-manifest.json bytes":
        raise NMRBlindChallengeError("signed-byte declaration changed")
    _sha256(
        manifest["prediction_submission_receipt_sha256"],
        location="release_manifest.prediction_submission_receipt_sha256",
    )
    artifacts = _exact_fields(
        manifest["artifacts"],
        {
            "public_manifest",
            "holder_manifest",
            "roleless",
            "run_spec",
            "gold",
            "predictions",
            "review_audit",
        },
        location="release_manifest.artifacts",
    )
    filenames = {
        "public_manifest": PUBLIC_MANIFEST_FILENAME,
        "holder_manifest": HOLDER_MANIFEST_FILENAME,
        "roleless": ROLELESS_FILENAME,
        "run_spec": RUN_SPEC_FILENAME,
        "gold": RELEASE_GOLD_FILENAME,
        "predictions": "predictions.json",
        "review_audit": REVIEW_AUDIT_FILENAME,
    }
    for key, filename in filenames.items():
        _validate_binding(
            artifacts[key],
            filename=filename,
            location=f"release_manifest.artifacts.{key}",
        )
    _nonempty_text(
        manifest["receipt_ledger_limit"],
        location="release_manifest.receipt_ledger_limit",
        maximum=512,
    )
    return manifest


def verify_blind_release(
    *,
    release_manifest_path: str | Path,
    signature_path: str | Path,
    public_key_path: str | Path,
    expected_public_key_spki_sha256: str,
    public_manifest_path: str | Path,
    holder_manifest_path: str | Path,
    roleless_path: str | Path,
    run_spec_path: str | Path,
    gold_path: str | Path,
    predictions_path: str | Path,
    review_audit_path: str | Path,
    receipt_ledger_path: str | Path,
) -> dict[str, Any]:
    """Verify the detached signature, artifacts, schemas, and receipt chain."""

    expected_key_sha = _sha256(
        expected_public_key_spki_sha256,
        location="expected_public_key_spki_sha256 trust anchor",
    )
    public_key, public_key_sha = _load_public_key(public_key_path)
    if public_key_sha != expected_key_sha:
        raise NMRBlindChallengeError(
            "public key does not match the independent expected-key trust anchor"
        )
    release_value, release_payload, _ = _read_strict_json(
        release_manifest_path,
        label="release manifest",
    )
    release_manifest = _validate_release_manifest(release_value)
    if release_payload != canonical_json_bytes(release_manifest):
        raise NMRBlindChallengeError("release manifest is not canonical JSON")
    signature_file = _safe_existing_file(signature_path, label="detached signature")
    signature = signature_file.read_bytes()
    if len(signature) != 64:
        raise NMRBlindChallengeError("detached Ed25519 signature must be 64 bytes")
    if release_manifest["signature"]["public_key_spki_sha256"] != public_key_sha:
        raise NMRBlindChallengeError("public key does not match signed manifest")
    try:
        public_key.verify(signature, release_payload)
    except Exception as exc:
        # Cryptography deliberately exposes InvalidSignature, but importing it
        # here would broaden the fail-closed dependency surface.
        raise NMRBlindChallengeError("detached Ed25519 signature is invalid") from exc

    artifact_paths = {
        "public_manifest": public_manifest_path,
        "holder_manifest": holder_manifest_path,
        "roleless": roleless_path,
        "run_spec": run_spec_path,
        "gold": gold_path,
        "predictions": predictions_path,
        "review_audit": review_audit_path,
    }
    payloads: dict[str, bytes] = {}
    values: dict[str, Any] = {}
    for key, path in artifact_paths.items():
        value, payload, _ = _read_strict_json(path, label=key)
        values[key] = value
        payloads[key] = payload
        if payload != canonical_json_bytes(value):
            raise NMRBlindChallengeError(f"{key}: artifact is not canonical JSON")
        _check_binding(
            release_manifest["artifacts"][key],
            payload,
            location=f"release {key}",
        )

    public_manifest = _validate_public_manifest(values["public_manifest"])
    holder_manifest = _validate_holder_manifest(values["holder_manifest"])
    roleless = _validate_roleless(values["roleless"])
    run_spec = _validate_run_spec(values["run_spec"])
    if run_spec["holder_public_key_spki_sha256"] != public_key_sha:
        raise NMRBlindChallengeError(
            "public key differs from the pre-submission run-spec commitment"
        )
    gold = _validate_gold(values["gold"], roleless)
    audit = _validate_review_audit(values["review_audit"], roleless, gold)
    predictions = _validate_predictions(values["predictions"], roleless, run_spec)
    challenge_id = release_manifest["challenge_id"]
    if any(
        item.get("challenge_id") != challenge_id
        for item in (
            public_manifest,
            holder_manifest,
            roleless,
            values["gold"],
            predictions,
        )
    ):
        raise NMRBlindChallengeError("release artifacts disagree on challenge ID")
    expected_counts = _reviewed_counts(roleless, audit)
    if (
        public_manifest["counts"] != expected_counts
        or holder_manifest["counts"] != expected_counts
    ):
        raise NMRBlindChallengeError("manifest counts do not match review audit")
    if public_manifest["gold_commitment"]["sha256"] != sha256_bytes(payloads["gold"]):
        raise NMRBlindChallengeError("released Gold differs from public commitment")
    _check_binding(
        public_manifest["artifacts"]["roleless"],
        payloads["roleless"],
        location="public roleless",
    )
    _check_binding(
        public_manifest["artifacts"]["run_spec"],
        payloads["run_spec"],
        location="public run spec",
    )
    for key in ("public_manifest", "roleless", "run_spec", "gold", "review_audit"):
        holder_key = key
        if key == "public_manifest":
            holder_key = "public_manifest"
        _check_binding(
            holder_manifest["artifacts"][holder_key],
            payloads[key],
            location=f"holder {key}",
        )
    if predictions["roleless_sha256"] != sha256_bytes(payloads["roleless"]):
        raise NMRBlindChallengeError("predictions roleless binding changed")
    if predictions["run_spec_sha256"] != sha256_bytes(payloads["run_spec"]):
        raise NMRBlindChallengeError("predictions run-spec binding changed")

    ledger = _absolute_without_resolving(receipt_ledger_path)
    _reject_symlink_components(
        ledger,
        label="receipt ledger",
        include_leaf=True,
    )
    ledger = ledger.resolve(strict=True)
    entries = _read_receipt_ledger_unlocked(ledger)
    submissions = [
        entry
        for entry in entries
        if entry["challenge_id"] == challenge_id
        and entry["event"] == "predictions_submitted_before_gold_release"
    ]
    releases = [
        entry
        for entry in entries
        if entry["challenge_id"] == challenge_id
        and entry["event"] == "gold_signed_and_released"
    ]
    if len(submissions) != 1 or len(releases) != 1:
        raise NMRBlindChallengeError(
            "receipt ledger lacks unique submit/release events"
        )
    submission = submissions[0]
    release = releases[0]
    if (
        submission["receipt_sha256"]
        != release_manifest["prediction_submission_receipt_sha256"]
    ):
        raise NMRBlindChallengeError("signed manifest binds another submission receipt")
    expected_submission = {
        "public_manifest_sha256": sha256_bytes(payloads["public_manifest"]),
        "roleless_sha256": sha256_bytes(payloads["roleless"]),
        "run_spec_sha256": sha256_bytes(payloads["run_spec"]),
        "predictions_sha256": sha256_bytes(payloads["predictions"]),
        "predictions_size_bytes": len(payloads["predictions"]),
    }
    for key, expected in expected_submission.items():
        if submission[key] != expected:
            raise NMRBlindChallengeError(f"submission receipt {key} changed")
    if release["submission_receipt_sha256"] != submission["receipt_sha256"]:
        raise NMRBlindChallengeError("release receipt binds another submission")
    if release["release_manifest_sha256"] != sha256_bytes(release_payload):
        raise NMRBlindChallengeError("release receipt manifest hash changed")
    if release["signature_sha256"] != sha256_bytes(signature):
        raise NMRBlindChallengeError("release receipt signature hash changed")
    registered_at = _utc_datetime(
        run_spec["registered_at"],
        location="run_spec.registered_at",
    )
    predictions_at = _utc_datetime(
        predictions["created_at"],
        location="predictions.created_at",
    )
    submitted_at = _utc_datetime(
        submission["occurred_at"],
        location="submission occurred_at",
    )
    released_at = _utc_datetime(
        release_manifest["released_at"],
        location="release_manifest.released_at",
    )
    release_receipt_at = _utc_datetime(
        release["occurred_at"],
        location="release receipt occurred_at",
    )
    if not (
        registered_at <= predictions_at <= submitted_at <= released_at
        and released_at == release_receipt_at
    ):
        raise NMRBlindChallengeError("challenge event timestamps are out of order")
    return {
        "status": "verified",
        "challenge_id": challenge_id,
        "record_count": len(roleless["records"]),
        "release_manifest_sha256": sha256_bytes(release_payload),
        "signature_sha256": sha256_bytes(signature),
        "holder_public_key_spki_sha256": public_key_sha,
        "receipt_chain_head_sha256": entries[-1]["receipt_sha256"],
        "receipt_ledger_is_worm": False,
    }
