"""Four-stage sealed-test protocol state machine.

Stages (each transition is enforced, never skipped, never replayed):

0. reservation_signed     -- holder signs and externally anchors a cohort
                              commitment (>=500 records / >=200 groups).
1. attempt_consumed       -- atomic exactly-once consumption before any query
                              bundle is loaded; holder nonce + append-only
                              ledger; terminal (crash/timeout/failure included).
2. predictions_committed  -- complete-denominator pre-Gold prediction manifest
                              (one success or explicit failure per query),
                              anchored and signed before Gold is released.
3. evaluation_complete    -- post-Gold result receipt with full-denominator
                              outcome accounting and dual signatures.

``simulation=True`` relaxes the record/group thresholds for dry runs and marks
every produced artifact with ``simulation: true`` so it can never be mistaken
for a real gate.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Mapping, Sequence

from .crypto import (
    object_sha256,
    payload_sha256,
    public_spki_sha256,
    sign_object,
    verify_object,
)
from .ledger import Denylist, Ledger

RESERVATION_SCHEMA = "chemapp.nmr.v8-external-holder-reservation.v2"
ATTEMPT_SCHEMA = "chemapp.nmr.v8-attempt-start.v1"
COMMITMENT_SCHEMA = "chemapp.nmr.v8-prediction-commitment.v1"
RECEIPT_SCHEMA = "chemapp.nmr.v8-external-holder-result-receipt.v2"
DENYLIST_SCHEMA = "chemapp.nmr.v8-denylist-entry.v1"

CANONICALIZATION = "chemapp-canonical-json.v1"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE_RE = re.compile(r"^[0-9a-f]{32,64}$")

REQUIRED_RESERVATION_HASHES = (
    "protocol_sha256",
    "task_manifest_sha256",
    "model_release_sha256",
    "generator_release_sha256",
    "cohort_commitment_sha256",
    "review_commitment_sha256",
    "license_review_commitment_sha256",
    "similarity_audit_commitment_sha256",
    "query_bundle_sha256",
    "prior_influence_index_sha256",
    "denylist_commitment_sha256",
)


class SealedTestError(ValueError):
    """Raised when a sealed-test protocol invariant is violated."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise SealedTestError(message)


def _hex(value: Any, location: str) -> str:
    _check(isinstance(value, str) and _HASH_RE.match(value), f"{location}: expected SHA-256 hex")
    return value


def _utc(value: Any, location: str) -> datetime:
    _check(isinstance(value, str), f"{location}: expected ISO-8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SealedTestError(f"{location}: invalid timestamp") from exc
    _check(parsed.tzinfo is not None, f"{location}: timestamp must carry timezone")
    return parsed


def _opaque(value: Any, location: str, prefix: str) -> str:
    _check(
        isinstance(value, str) and value.startswith(prefix) and len(value) > len(prefix),
        f"{location}: expected opaque {prefix} id",
    )
    return value


def _verify_signature_field(
    obj: Mapping[str, Any],
    field: str,
    public_key: Any,
    *,
    location: str,
    omit: Sequence[str] = (),
) -> None:
    signature = obj.get(field)
    _check(isinstance(signature, Mapping), f"{location}.{field}: missing signature")
    _check(signature.get("algorithm") == "Ed25519", f"{location}.{field}: algorithm must be Ed25519")
    signed_payload = {key: item for key, item in obj.items() if key not in set(omit) | {field}}
    signed_hash = object_sha256(signed_payload)
    _check(
        signature.get("signed_payload_sha256") == signed_hash,
        f"{location}.{field}: signed payload hash mismatch",
    )
    verify_object(signed_payload, signature["signature_base64"], public_key)


def _check_anchor(obj: Mapping[str, Any], anchor: Mapping[str, Any], *, location: str) -> None:
    _check(anchor.get("type") in ("rfc3161", "worm", "trusted_external_ledger"), f"{location}: bad anchor type")
    _hex(anchor.get("anchored_payload_sha256"), f"{location}.anchored_payload_sha256")
    _utc(anchor.get("timestamp_utc"), f"{location}.timestamp_utc")
    _hex(anchor.get("receipt_sha256"), f"{location}.receipt_sha256")
    _check(anchor.get("anchored_payload_rule"), f"{location}: anchored_payload_rule required")


class SealedTestStateMachine:
    """Enforces the four-stage sealed-test transitions."""

    def __init__(self, *, simulation: bool = False) -> None:
        self.simulation = bool(simulation)

    # ---- stage 0 ---------------------------------------------------------
    def verify_reservation(self, obj: Mapping[str, Any], holder_public_key: Any) -> dict[str, Any]:
        location = "reservation"
        _check(obj.get("schema_version") == RESERVATION_SCHEMA, f"{location}: schema mismatch")
        _check(obj.get("x-chemapp-contract-status") == "executable_awaiting_holder", f"{location}: not executable")
        if self.simulation:
            _check(obj.get("simulation") is True, f"{location}: simulation mode requires simulation=true")
        else:
            _check(obj.get("simulation") is not True, f"{location}: real reservations must not be marked simulation")

        _check(obj.get("canonicalization", {}).get("encoding") == CANONICALIZATION, f"{location}: canonicalization mismatch")
        holder = obj.get("holder")
        _check(isinstance(holder, Mapping), f"{location}.holder: missing")
        _check(holder.get("independent_of_model_team") is True, f"{location}.holder: must be independent")
        _check(public_spki_sha256(holder_public_key) == holder.get("ed25519_public_key_spki_sha256"), f"{location}: SPKI hash mismatch")

        minimum_records = 10 if self.simulation else 500
        minimum_groups = 5 if self.simulation else 200
        _check(isinstance(obj.get("record_count"), int) and obj["record_count"] >= minimum_records, f"{location}: record_count below threshold")
        _check(isinstance(obj.get("connected_group_count"), int) and obj["connected_group_count"] >= minimum_groups, f"{location}: connected_group_count below threshold")

        for key in REQUIRED_RESERVATION_HASHES:
            _hex(obj.get(key), f"{location}.{key}")
        _check(obj.get("attempt_nonce_policy", {}).get("required") is True, f"{location}: nonce must be required")
        _check(obj.get("attempt_nonce_policy", {}).get("min_entropy_bits", 0) >= 128, f"{location}: nonce entropy below 128 bits")
        _utc(obj.get("created_at_utc"), f"{location}.created_at_utc")

        without_signature = {key: item for key, item in obj.items() if key not in ("holder_signature", "external_anchor")}
        _check_anchor(obj, obj.get("external_anchor", {}), location=f"{location}.external_anchor")
        _check(
            obj["external_anchor"]["anchored_payload_sha256"] == object_sha256(without_signature),
            f"{location}.external_anchor: anchored payload hash mismatch",
        )
        _verify_signature_field(obj, "holder_signature", holder_public_key, location=location)
        return {"reservation_sha256": object_sha256(obj), "reservation": obj}

    # ---- stage 1 ---------------------------------------------------------
    def _verify_attempt_object(
        self,
        reservation: Mapping[str, Any],
        attempt: Mapping[str, Any],
        holder_public_key: Any,
    ) -> dict[str, Any]:
        location = "attempt"
        reservation_sha256 = object_sha256(reservation)
        _check(attempt.get("schema_version") == ATTEMPT_SCHEMA, f"{location}: schema mismatch")
        _check(attempt.get("reservation_sha256") == reservation_sha256, f"{location}: reservation binding mismatch")
        _opaque(attempt.get("attempt_id"), f"{location}.attempt_id", "att_")
        nonce = attempt.get("holder_nonce")
        _check(isinstance(nonce, str) and len(nonce) >= 64, f"{location}: holder_nonce must be >= 256 bits hex")
        _check(attempt.get("query_bundle_sha256") == reservation.get("query_bundle_sha256"), f"{location}: query bundle mismatch")
        _utc(attempt.get("consumed_at_utc"), f"{location}.consumed_at_utc")
        _check(attempt.get("state") == "consumed", f"{location}: state must be consumed")

        receipt = attempt.get("ledger_receipt", {})
        anchored = {
            "reservation_sha256": reservation_sha256,
            "attempt_id": attempt["attempt_id"],
            "state": "consumed",
            "consumed_at_utc": attempt["consumed_at_utc"],
        }
        _check(receipt.get("anchored_payload_sha256") == object_sha256(anchored), f"{location}: ledger anchor mismatch")
        _check(receipt.get("reservation_uniqueness_enforced") is True, f"{location}: ledger must enforce uniqueness")
        _verify_signature_field(attempt, "holder_signature", holder_public_key, location=location)
        return {
            "reservation_sha256": reservation_sha256,
            "attempt_sha256": object_sha256(attempt),
            "holder_nonce_sha256": object_sha256({"nonce": nonce}),
        }

    def consume(
        self,
        reservation: Mapping[str, Any],
        attempt: Mapping[str, Any],
        holder_public_key: Any,
        ledger: Ledger,
        *,
        denylist: Denylist | None = None,
        identities: Sequence[str] = (),
    ) -> dict[str, Any]:
        verified = self._verify_attempt_object(reservation, attempt, holder_public_key)
        if denylist is not None and identities:
            denylist.check(identities)

        entry = ledger.append(
            {
                "schema_version": ATTEMPT_SCHEMA,
                "event": "attempt_consumed",
                "reservation_sha256": verified["reservation_sha256"],
                "attempt_id": attempt["attempt_id"],
                "attempt_sha256": verified["attempt_sha256"],
                "holder_nonce_sha256": verified["holder_nonce_sha256"],
                "query_bundle_sha256": attempt["query_bundle_sha256"],
                "occurred_at_utc": attempt["consumed_at_utc"],
            }
        )
        if denylist is not None and identities:
            denylist.record(
                reservation_sha256=verified["reservation_sha256"],
                attempt_id=attempt["attempt_id"],
                identities=identities,
                occurred_at_utc=attempt["consumed_at_utc"],
                holder_nonce_sha256=entry["holder_nonce_sha256"],
            )
        return {"attempt": attempt, "ledger_entry": entry}

    # ---- stage 2 ---------------------------------------------------------
    def commit(
        self,
        reservation: Mapping[str, Any],
        attempt: Mapping[str, Any],
        ledger_entry: Mapping[str, Any],
        commitment: Mapping[str, Any],
        holder_public_key: Any,
        executor_public_key: Any,
        *,
        query_ids: Sequence[str] | None = None,
        prediction_manifest: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        location = "commitment"
        reservation_sha256 = object_sha256(reservation)
        _check(commitment.get("schema_version") == COMMITMENT_SCHEMA, f"{location}: schema mismatch")
        _check(commitment.get("reservation_sha256") == reservation_sha256, f"{location}: reservation binding mismatch")
        for key in ("model_release_sha256", "generator_release_sha256", "task_manifest_sha256"):
            _check(commitment.get(key) == reservation.get(key), f"{location}: {key} must match reservation")
        _check(commitment.get("attempt_id") == attempt.get("attempt_id"), f"{location}: attempt binding mismatch")
        _check(commitment.get("attempt_sha256") == ledger_entry.get("attempt_sha256"), f"{location}: ledger attempt binding mismatch")

        query_count = commitment.get("query_count")
        prediction_count = commitment.get("prediction_count")
        _check(isinstance(query_count, int) and query_count >= 1, f"{location}: query_count invalid")
        _check(isinstance(prediction_count, int) and prediction_count == query_count, f"{location}: complete denominator required (prediction_count == query_count)")
        succeeded = commitment.get("succeeded_count")
        failed = commitment.get("failed_count")
        _check(isinstance(succeeded, int) and isinstance(failed, int) and succeeded + failed == query_count, f"{location}: outcome accounting must be complete")
        _hex(commitment.get("query_manifest_sha256"), f"{location}.query_manifest_sha256")
        _hex(commitment.get("prediction_manifest_sha256"), f"{location}.prediction_manifest_sha256")

        _utc(attempt.get("consumed_at_utc"), f"{location}: attempt timestamp")
        committed_at = _utc(commitment.get("prediction_committed_at_utc"), f"{location}.prediction_committed_at_utc")
        _check(committed_at >= _utc(attempt["consumed_at_utc"], "attempt.consumed_at_utc"), f"{location}: commitment must follow consumption")
        anchor = commitment.get("prediction_anchor", {})
        _check_anchor(commitment, anchor, location=f"{location}.prediction_anchor")
        _check(_utc(anchor["timestamp_utc"], f"{location}.prediction_anchor.timestamp_utc") >= committed_at, f"{location}: anchor must follow commitment")
        _check(anchor.get("anchored_payload_sha256") == commitment.get("prediction_artifact_sha256"), f"{location}: prediction artifact anchor mismatch")

        if query_ids is not None:
            _check(len(query_ids) == query_count, f"{location}: query_ids length mismatch")
            _check(len(set(query_ids)) == len(query_ids), f"{location}: query_ids must be unique")
        if prediction_manifest is not None:
            outcomes = prediction_manifest.get("outcomes")
            _check(isinstance(outcomes, dict), f"{location}: prediction manifest outcomes missing")
            _check(
                set(outcomes) == set(query_ids or []),
                f"{location}: prediction manifest must cover every query exactly once",
            )
            for query_id, outcome in outcomes.items():
                _check(
                    outcome in ("success", "failure"),
                    f"{location}: outcome for {query_id} must be success or failure",
                )
            success_count = sum(1 for outcome in outcomes.values() if outcome == "success")
            failure_count = sum(1 for outcome in outcomes.values() if outcome == "failure")
            _check(success_count == commitment.get("succeeded_count"), f"{location}: succeeded_count mismatch")
            _check(failure_count == commitment.get("failed_count"), f"{location}: failed_count mismatch")
            from .crypto import canonical_object_bytes
            manifest_payload = canonical_object_bytes(prediction_manifest)
            _check(
                payload_sha256(manifest_payload) == commitment.get("prediction_manifest_sha256"),
                f"{location}: prediction manifest hash mismatch",
            )
        _verify_signature_field(commitment, "executor_signature", executor_public_key, location=location, omit=("holder_signature",))
        _verify_signature_field(commitment, "holder_signature", holder_public_key, location=location, omit=("executor_signature",))
        return {"commitment_sha256": object_sha256(commitment), "commitment": commitment}

    # ---- stage 3 ---------------------------------------------------------
    def evaluate(
        self,
        reservation: Mapping[str, Any],
        attempt: Mapping[str, Any],
        commitment: Mapping[str, Any],
        receipt: Mapping[str, Any],
        holder_public_key: Any,
        evaluator_public_key: Any,
        *,
        evaluation_artifact_bytes: bytes | None = None,
    ) -> dict[str, Any]:
        location = "receipt"
        reservation_sha256 = object_sha256(reservation)
        _check(receipt.get("schema_version") == RECEIPT_SCHEMA, f"{location}: schema mismatch")
        _check(receipt.get("reservation_sha256") == reservation_sha256, f"{location}: reservation binding mismatch")
        _check(receipt.get("execution_commitment_sha256") == object_sha256(commitment), f"{location}: commitment binding mismatch")
        _check(receipt.get("evaluation_mode") in ("holder_private", "public_reveal"), f"{location}: evaluation_mode invalid")

        total = receipt.get("total_queries")
        evaluated = receipt.get("evaluated_queries")
        succeeded = receipt.get("succeeded_queries")
        failed = receipt.get("failed_queries")
        _check(isinstance(total, int) and total == commitment.get("query_count"), f"{location}: total_queries must equal query_count")
        _check(isinstance(evaluated, int) and evaluated == total, f"{location}: full denominator must be evaluated")
        _check(isinstance(succeeded, int) and isinstance(failed, int) and succeeded + failed == total, f"{location}: outcome accounting must be complete")

        gold = receipt.get("gold_release_receipt", {})
        _check_anchor(receipt, gold, location=f"{location}.gold_release_receipt")
        gold_released = _utc(gold.get("gold_released_at_utc"), f"{location}.gold_release_receipt.gold_released_at_utc")
        _check(gold_released >= _utc(attempt["consumed_at_utc"], "attempt.consumed_at_utc"), f"{location}: Gold must follow consumption")
        _check(gold_released >= _utc(commitment["prediction_committed_at_utc"], "commitment.prediction_committed_at_utc"), f"{location}: Gold must follow pre-Gold commitment")
        _check(
            _utc(gold.get("timestamp_utc"), f"{location}.gold_release_receipt.timestamp_utc") >= gold_released,
            f"{location}: gold anchor must follow release",
        )
        completed = _utc(receipt.get("evaluation_completed_at_utc"), f"{location}.evaluation_completed_at_utc")
        _check(completed >= gold_released, f"{location}: evaluation must follow Gold release")
        _hex(receipt.get("evaluation_artifact_sha256"), f"{location}.evaluation_artifact_sha256")
        if evaluation_artifact_bytes is not None:
            _check(
                payload_sha256(evaluation_artifact_bytes) == receipt["evaluation_artifact_sha256"],
                f"{location}: evaluation artifact hash mismatch",
            )
        _verify_signature_field(receipt, "evaluator_signature", evaluator_public_key, location=location, omit=("holder_signature",))
        _verify_signature_field(receipt, "holder_signature", holder_public_key, location=location, omit=("evaluator_signature",))
        return {"receipt": receipt}

    # ---- chain verification ---------------------------------------------
    def verify_chain(
        self,
        reservation: Mapping[str, Any],
        attempt: Mapping[str, Any],
        commitment: Mapping[str, Any],
        receipt: Mapping[str, Any],
        holder_public_key: Any,
        executor_public_key: Any,
        evaluator_public_key: Any,
        *,
        ledger: Ledger | None = None,
        denylist: Denylist | None = None,
        identities: Sequence[str] = (),
    ) -> dict[str, Any]:
        problems: list[str] = []
        try:
            self.verify_reservation(reservation, holder_public_key)
        except SealedTestError as exc:
            problems.append(str(exc))
        try:
            verified_attempt = self._verify_attempt_object(reservation, attempt, holder_public_key)
            if ledger is None:
                raise SealedTestError("ledger is required for chain verification")
            entries = ledger.read()
            matches = [
                entry
                for entry in entries
                if entry.get("reservation_sha256") == verified_attempt["reservation_sha256"]
                and entry.get("attempt_id") == attempt["attempt_id"]
                and entry.get("event") == "attempt_consumed"
            ]
            if len(matches) != 1:
                raise SealedTestError(
                    f"ledger must contain exactly one consumed entry for this attempt, found {len(matches)}"
                )
            ledger_entry = matches[0]
            if ledger_entry["attempt_sha256"] != verified_attempt["attempt_sha256"]:
                raise SealedTestError("ledger attempt binding mismatch")
        except (SealedTestError, ValueError) as exc:
            ledger_entry = {}
            problems.append(str(exc))
        try:
            self.commit(
                reservation, attempt, ledger_entry, commitment,
                holder_public_key, executor_public_key,
            )
        except SealedTestError as exc:
            problems.append(str(exc))
        try:
            self.evaluate(
                reservation, attempt, commitment, receipt,
                holder_public_key, evaluator_public_key,
            )
        except SealedTestError as exc:
            problems.append(str(exc))
        if denylist is not None and identities:
            recorded: set[str] = set()
            for entry in denylist.read():
                recorded.update(entry.get("identities", []))
            missing = [identity for identity in identities if identity not in recorded]
            if missing:
                problems.append("denylist missing consumed identities: " + ", ".join(missing))
        return {
            "status": "verified" if not problems else "rejected",
            "problems": problems,
            "simulation": self.simulation,
        }


def _dummy_path() -> str:
    import tempfile
    from pathlib import Path
    return str(Path(tempfile.gettempdir()) / "sealed-test-unused-ledger.jsonl")


# ---- deterministic builders for holders / executors -----------------------
def build_reservation_draft(
    *,
    holder_id: str,
    holder_public_key: Any,
    record_count: int,
    connected_group_count: int,
    protocol_sha256: str,
    task_manifest_sha256: str,
    model_release_sha256: str,
    generator_release_sha256: str,
    cohort_commitment_sha256: str,
    review_commitment_sha256: str,
    license_review_commitment_sha256: str,
    similarity_audit_commitment_sha256: str,
    query_bundle_sha256: str,
    prior_influence_index_sha256: str,
    denylist_commitment_sha256: str,
    created_at_utc: str,
    simulation: bool = False,
) -> dict[str, Any]:
    draft = {
        "schema_version": RESERVATION_SCHEMA,
        "x-chemapp-contract-status": "executable_awaiting_holder",
        "canonicalization": {"encoding": CANONICALIZATION, "algorithm": "SHA-256"},
        "commitment_semantics": {
            "algorithm": "SHA-256",
            "canonicalization": CANONICALIZATION,
            "salt_policy": "minimum 128-bit random per-reservation salt held by external holder until release",
        },
        "holder": {
            "holder_id": holder_id,
            "independent_of_model_team": True,
            "ed25519_public_key_spki_base64": _spki_b64(holder_public_key),
            "ed25519_public_key_spki_sha256": public_spki_sha256(holder_public_key),
        },
        "created_at_utc": created_at_utc,
        "protocol_sha256": protocol_sha256,
        "task_manifest_sha256": task_manifest_sha256,
        "model_release_sha256": model_release_sha256,
        "generator_release_sha256": generator_release_sha256,
        "cohort_commitment_sha256": cohort_commitment_sha256,
        "review_commitment_sha256": review_commitment_sha256,
        "license_review_commitment_sha256": license_review_commitment_sha256,
        "similarity_audit_commitment_sha256": similarity_audit_commitment_sha256,
        "query_bundle_sha256": query_bundle_sha256,
        "prior_influence_index_sha256": prior_influence_index_sha256,
        "denylist_commitment_sha256": denylist_commitment_sha256,
        "attempt_nonce_policy": {"required": True, "min_entropy_bits": 128},
        "record_count": record_count,
        "connected_group_count": connected_group_count,
    }
    if simulation:
        draft["simulation"] = True
    return draft


def sign_reservation(draft: Mapping[str, Any], holder_private_key: Any, *, anchor_timestamp_utc: str) -> dict[str, Any]:
    without_anchor = {key: item for key, item in draft.items() if key != "holder_signature"}
    anchored_payload = object_sha256(without_anchor)
    anchor = {
        "anchored_payload_rule": "chemapp-canonical-json.v1 of the complete reservation with holder_signature and external_anchor omitted",
        "anchored_payload_sha256": anchored_payload,
        "canonicalization": CANONICALIZATION,
        "type": "worm",
        "provider": "simulation-local-ledger",
        "timestamp_utc": anchor_timestamp_utc,
        "receipt_sha256": object_sha256({"payload": anchored_payload, "at": anchor_timestamp_utc}),
        "receipt_uri": "https://example.invalid/simulation-anchor",
        "validator": "simulation",
    }
    signed = {**draft, "external_anchor": anchor}
    signature = {
        "algorithm": "Ed25519",
        "signed_payload_rule": "chemapp-canonical-json.v1 of the complete reservation with holder_signature omitted",
        "signed_payload_sha256": object_sha256({key: item for key, item in signed.items() if key != "holder_signature"}),
        "signature_base64": sign_object({key: item for key, item in signed.items() if key != "holder_signature"}, holder_private_key),
    }
    return {**signed, "holder_signature": signature}


def build_attempt(
    *,
    reservation: Mapping[str, Any],
    attempt_id: str,
    holder_nonce: str,
    consumed_at_utc: str,
    ledger_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    receipt = dict(ledger_receipt)
    anchored = {
        "reservation_sha256": object_sha256(reservation),
        "attempt_id": attempt_id,
        "state": "consumed",
        "consumed_at_utc": consumed_at_utc,
    }
    receipt.setdefault("anchored_payload_sha256", object_sha256(anchored))
    receipt.setdefault("reservation_uniqueness_enforced", True)
    return {
        "schema_version": ATTEMPT_SCHEMA,
        "reservation_sha256": object_sha256(reservation),
        "attempt_id": attempt_id,
        "holder_nonce": holder_nonce,
        "query_bundle_sha256": reservation["query_bundle_sha256"],
        "state": "consumed",
        "consumed_at_utc": consumed_at_utc,
        "ledger_receipt": receipt,
    }


def sign_attempt(attempt: Mapping[str, Any], holder_private_key: Any) -> dict[str, Any]:
    payload = {key: item for key, item in attempt.items() if key != "holder_signature"}
    return {
        **attempt,
        "holder_signature": {
            "algorithm": "Ed25519",
            "signed_payload_rule": "chemapp-canonical-json.v1 of the complete attempt with holder_signature omitted",
            "signed_payload_sha256": object_sha256(payload),
            "signature_base64": sign_object(payload, holder_private_key),
        },
    }


def build_commitment(
    *,
    reservation: Mapping[str, Any],
    attempt: Mapping[str, Any],
    ledger_entry: Mapping[str, Any],
    query_manifest_sha256: str,
    prediction_manifest_sha256: str,
    query_count: int,
    prediction_count: int,
    succeeded_count: int,
    failed_count: int,
    prediction_committed_at_utc: str,
    prediction_anchor: Mapping[str, Any],
    executor_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": COMMITMENT_SCHEMA,
        "reservation_sha256": object_sha256(reservation),
        "model_release_sha256": reservation["model_release_sha256"],
        "generator_release_sha256": reservation["generator_release_sha256"],
        "task_manifest_sha256": reservation["task_manifest_sha256"],
        "attempt_id": attempt["attempt_id"],
        "attempt_sha256": ledger_entry["attempt_sha256"],
        "query_manifest_sha256": query_manifest_sha256,
        "prediction_manifest_sha256": prediction_manifest_sha256,
        "query_count": query_count,
        "prediction_count": prediction_count,
        "succeeded_count": succeeded_count,
        "failed_count": failed_count,
        "prediction_artifact_sha256": prediction_manifest_sha256,
        "prediction_committed_at_utc": prediction_committed_at_utc,
        "prediction_anchor": prediction_anchor,
        "executor": {"executor_id": executor_id},
    }


def sign_commitment(commitment: Mapping[str, Any], executor_private_key: Any, holder_private_key: Any) -> dict[str, Any]:
    no_holder = {key: item for key, item in commitment.items() if key != "holder_signature"}
    no_signatures = {key: item for key, item in no_holder.items() if key != "executor_signature"}
    return {
        **no_holder,
        "executor_signature": {
            "algorithm": "Ed25519",
            "signed_payload_rule": "chemapp-canonical-json.v1 of the commitment with executor_signature omitted",
            "signed_payload_sha256": object_sha256(no_signatures),
            "signature_base64": sign_object(no_signatures, executor_private_key),
        },
        "holder_signature": {
            "algorithm": "Ed25519",
            "signed_payload_rule": "chemapp-canonical-json.v1 of the commitment with all signatures omitted",
            "signed_payload_sha256": object_sha256(no_signatures),
            "signature_base64": sign_object(no_signatures, holder_private_key),
        },
    }


def build_receipt(
    *,
    reservation: Mapping[str, Any],
    commitment: Mapping[str, Any],
    evaluation_mode: str,
    total_queries: int,
    evaluated_queries: int,
    succeeded_queries: int,
    failed_queries: int,
    gold_released_at_utc: str,
    gold_anchor: Mapping[str, Any],
    evaluation_completed_at_utc: str,
    evaluation_artifact_sha256: str,
    evaluator_id: str,
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA,
        "reservation_sha256": object_sha256(reservation),
        "execution_commitment_sha256": object_sha256(commitment),
        "evaluation_mode": evaluation_mode,
        "total_queries": total_queries,
        "evaluated_queries": evaluated_queries,
        "succeeded_queries": succeeded_queries,
        "failed_queries": failed_queries,
        "gold_release_receipt": {
            **gold_anchor,
            "anchored_payload_rule": "chemapp-canonical-json.v1 of {reservation_sha256,execution_commitment_sha256,gold_released_at_utc}",
            "anchored_payload_sha256": object_sha256(
                {
                    "reservation_sha256": object_sha256(reservation),
                    "execution_commitment_sha256": object_sha256(commitment),
                    "gold_released_at_utc": gold_released_at_utc,
                }
            ),
            "gold_released_at_utc": gold_released_at_utc,
        },
        "evaluation_artifact_sha256": evaluation_artifact_sha256,
        "evaluation_completed_at_utc": evaluation_completed_at_utc,
        "evaluator": {"evaluator_id": evaluator_id},
    }


def sign_receipt(receipt: Mapping[str, Any], evaluator_private_key: Any, holder_private_key: Any) -> dict[str, Any]:
    no_holder = {key: item for key, item in receipt.items() if key != "holder_signature"}
    no_signatures = {key: item for key, item in no_holder.items() if key != "evaluator_signature"}
    return {
        **no_holder,
        "evaluator_signature": {
            "algorithm": "Ed25519",
            "signed_payload_rule": "chemapp-canonical-json.v1 of the receipt with evaluator_signature omitted",
            "signed_payload_sha256": object_sha256(no_signatures),
            "signature_base64": sign_object(no_signatures, evaluator_private_key),
        },
        "holder_signature": {
            "algorithm": "Ed25519",
            "signed_payload_rule": "chemapp-canonical-json.v1 of the receipt with all signatures omitted",
            "signed_payload_sha256": object_sha256(no_signatures),
            "signature_base64": sign_object(no_signatures, holder_private_key),
        },
    }


def _spki_b64(public_key: Any) -> str:
    from .crypto import public_spki_base64
    return public_spki_base64(public_key)
