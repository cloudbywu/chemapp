"""Independent-reviewer commitment toolchain for the v8 sealed test.

The sealed-test promotion gate requires at least two independent reviewers
(dual review) covering structure/spectrum sanity, license/rights, and
prior-influence/prior-consumption exclusion before a reservation may be
signed.  This module lets each reviewer generate an Ed25519 keypair, complete
a checklist, and sign a machine-readable review commitment.  A separate
verifier enforces that the commitment set has:

- at least two reviewers,
- distinct identities and keypairs,
- independence attestations (not the same operator / machine / session),
- coverage of all three required review types with pass=true,
- valid Ed25519 signatures.

Independence beyond attestation cannot be proven cryptographically; it is a
holder-side commitment recorded for external audit.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from .crypto import (
    object_sha256,
    public_spki_base64,
    public_spki_sha256,
    sign_object,
    verify_object,
)

SCHEMA_VERSION = "chemapp.nmr.v8-reviewer-commitment.v1"
REVIEW_TYPES = (
    "structure_and_spectrum",
    "license_and_rights",
    "prior_influence_consumption",
)
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
MIN_REVIEWERS = 2


class ReviewerError(ValueError):
    """Raised when a reviewer commitment invariant is violated."""


def build_draft(
    *,
    reviewer_id: str,
    reviewer_public_key: Any,
    review_type: str,
    cohort_commitment_sha256: str,
    reviewed_at_utc: str,
    scope: Mapping[str, Any],
) -> dict[str, Any]:
    if review_type not in REVIEW_TYPES:
        raise ReviewerError(f"unknown review_type: {review_type!r}")
    if not _HASH_RE.match(cohort_commitment_sha256):
        raise ReviewerError("cohort_commitment_sha256 must be a SHA-256 hex digest")
    return {
        "schema_version": SCHEMA_VERSION,
        "reviewer": {
            "reviewer_id": reviewer_id,
            "independent_of_model_team": True,
            "ed25519_public_key_spki_base64": public_spki_base64(reviewer_public_key),
            "ed25519_public_key_spki_sha256": public_spki_sha256(reviewer_public_key),
        },
        "review_type": review_type,
        "cohort_commitment_sha256": cohort_commitment_sha256,
        "reviewed_at_utc": reviewed_at_utc,
        "scope": dict(scope),
        "findings": {
            "checked_items": [],
            "pass": False,
            "issues": [],
        },
        "independence_attestation": {
            "same_operator_as_model_team": False,
            "shared_machine_with_model_team": False,
            "shared_session_with_other_reviewer": False,
        },
    }


def sign_draft(
    draft: Mapping[str, Any],
    private_key: Any,
) -> dict[str, Any]:
    findings = draft.get("findings", {})
    checked = findings.get("checked_items", [])
    if not isinstance(checked, list) or not checked:
        raise ReviewerError("findings.checked_items must not be empty before signing")
    if any(item.get("status") == "pending" for item in checked if isinstance(item, Mapping)):
        raise ReviewerError("all checklist items must be completed (pass/fail) before signing")
    if not isinstance(findings.get("pass"), bool):
        raise ReviewerError("findings.pass must be a boolean")
    if not isinstance(findings.get("issues"), list):
        raise ReviewerError("findings.issues must be a list")
    payload = {key: item for key, item in draft.items() if key != "reviewer_signature"}
    return {
        **payload,
        "reviewer_signature": {
            "algorithm": "Ed25519",
            "signed_payload_rule": "chemapp-canonical-json.v1 of the commitment with reviewer_signature omitted",
            "signed_payload_sha256": object_sha256(payload),
            "signature_base64": sign_object(payload, private_key),
        },
    }


def verify_commitment(
    commitment: Mapping[str, Any],
    public_key: Any | None = None,
) -> list[str]:
    problems: list[str] = []
    if commitment.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version mismatch")
    reviewer = commitment.get("reviewer")
    if not isinstance(reviewer, Mapping):
        problems.append("reviewer identity missing")
        return problems
    if reviewer.get("independent_of_model_team") is not True:
        problems.append("reviewer must attest independence from the model team")
    review_type = commitment.get("review_type")
    if review_type not in REVIEW_TYPES:
        problems.append(f"unknown review_type: {review_type!r}")
    if not _HASH_RE.match(str(commitment.get("cohort_commitment_sha256") or "")):
        problems.append("cohort_commitment_sha256 invalid")
    if not str(commitment.get("reviewed_at_utc") or ""):
        problems.append("reviewed_at_utc missing")
    findings = commitment.get("findings")
    if not isinstance(findings, Mapping):
        problems.append("findings missing")
        return problems
    checked = findings.get("checked_items")
    if not isinstance(checked, list) or not checked:
        problems.append("checked_items missing")
    else:
        for index, item in enumerate(checked):
            if not isinstance(item, Mapping) or not item.get("item") or item.get("status") not in ("pass", "fail"):
                problems.append(f"checked_items[{index}] invalid")
    if findings.get("pass") is not True:
        problems.append("findings.pass must be true for the commitment to count")
    attestation = commitment.get("independence_attestation", {})
    for key in (
        "same_operator_as_model_team",
        "shared_machine_with_model_team",
        "shared_session_with_other_reviewer",
    ):
        if attestation.get(key) is not False:
            problems.append(f"independence_attestation.{key} must be false")

    signature = commitment.get("reviewer_signature")
    if not isinstance(signature, Mapping):
        problems.append("reviewer_signature missing")
        return problems
    payload = {key: item for key, item in commitment.items() if key != "reviewer_signature"}
    if signature.get("signed_payload_sha256") != object_sha256(payload):
        problems.append("signed payload hash mismatch")
    if public_key is not None:
        if public_spki_sha256(public_key) != reviewer.get("ed25519_public_key_spki_sha256"):
            problems.append("public key SPKI does not match commitment")
        try:
            verify_object(payload, signature.get("signature_base64", ""), public_key)
        except Exception as exc:
            problems.append(f"signature verification failed: {exc}")
    return problems


def verify_reviewer_set(
    commitments: Sequence[Mapping[str, Any]],
    public_keys: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    problems: list[str] = []
    by_spki: dict[str, dict[str, Any]] = {}
    for index, commitment in enumerate(commitments):
        reviewer = commitment.get("reviewer", {})
        spki = str(reviewer.get("ed25519_public_key_spki_sha256") or "")
        reviewer_id = str(reviewer.get("reviewer_id") or "")
        if not spki or not reviewer_id:
            problems.append(f"commitment[{index}] missing reviewer identity")
            continue
        existing = by_spki.get(spki)
        if existing is None:
            by_spki[spki] = {"reviewer_id": reviewer_id, "commitments": []}
        elif existing["reviewer_id"] != reviewer_id:
            problems.append(f"commitment[{index}] keypair reused with a different reviewer_id")
        existing = by_spki[spki]
        existing["commitments"].append(commitment)

    if len(by_spki) < MIN_REVIEWERS:
        problems.append(
            f"need at least {MIN_REVIEWERS} distinct reviewers, got {len(by_spki)}"
        )
    id_to_spki: dict[str, str] = {}
    for spki, group in by_spki.items():
        if group["reviewer_id"] in id_to_spki and id_to_spki[group["reviewer_id"]] != spki:
            problems.append(f"reviewer_id reused under different keypairs: {group['reviewer_id']}")
        id_to_spki[group["reviewer_id"]] = spki
    covered: set[str] = set()
    for spki, group in by_spki.items():
        key = public_keys.get(spki) if public_keys else None
        for commitment in group["commitments"]:
            problems.extend(verify_commitment(commitment, public_key=key))
            review_type = commitment.get("review_type")
            if review_type in REVIEW_TYPES and commitment.get("findings", {}).get("pass") is True:
                covered.add(review_type)
    missing = [review_type for review_type in REVIEW_TYPES if review_type not in covered]
    if missing:
        problems.append("review coverage missing: " + ", ".join(missing))
    bindings = [
        {
            "reviewer_id": group["reviewer_id"],
            "spki_sha256": spki,
            "review_type": commitment.get("review_type"),
            "pass": commitment.get("findings", {}).get("pass"),
            "issues": commitment.get("findings", {}).get("issues", []),
        }
        for spki, group in sorted(by_spki.items())
        for commitment in sorted(group["commitments"], key=lambda item: str(item.get("review_type", "")))
    ]
    commitment_set_sha256 = object_sha256(bindings)
    return {
        "schema_version": "chemapp.nmr.v8-reviewer-set.v1",
        "status": "verified" if not problems else "rejected",
        "problems": problems,
        "reviewer_count": len(by_spki),
        "commitment_count": len(commitments),
        "covered_review_types": sorted(covered),
        "commitment_set_sha256": commitment_set_sha256,
    }
