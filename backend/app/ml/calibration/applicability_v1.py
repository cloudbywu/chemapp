"""Versioned strict applicability signature for calibrated probability output.

Phase-4 policy: a calibrated probability may only be emitted when the frozen
signature matches the exact runtime context.  The committed default signature
is blocked because no production-distribution calibration pool with the
required error-event count has been frozen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "chemapp.nmr.calibration-applicability.v1"
REQUIRED_FIELDS = (
    "model_sha256",
    "calibrator_sha256",
    "feature_schema",
    "generator_version",
    "provider_ids",
    "nucleus",
    "solvent",
    "pool_size",
    "qc_passed",
    "protocol_version",
)

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_SIGNATURE_PATH = _PACKAGE_DIR / "applicability-v1.json"


def load_signature(
    path: Path | None = None,
) -> dict[str, Any]:
    selected = path or DEFAULT_SIGNATURE_PATH
    return json.loads(selected.read_text(encoding="utf-8"))


def signature_problems(value: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if value.get("schema_version") != SCHEMA_VERSION:
        problems.append("schema_version mismatch")
    if value.get("status") not in {"active", "blocked_no_eligible_calibration_data"}:
        problems.append("status must be active or blocked_no_eligible_calibration_data")
    for field in REQUIRED_FIELDS:
        if field not in value:
            problems.append(f"missing field: {field}")
    if value.get("status") == "active":
        for field in REQUIRED_FIELDS:
            if value.get(field) in (None, "", []):
                problems.append(f"active signature requires non-empty {field}")
    return problems


def check_applicability(
    signature: Mapping[str, Any],
    *,
    model_sha256: str | None,
    calibrator_sha256: str | None,
    feature_schema: str | None,
    generator_version: str | None,
    provider_ids: Sequence[str],
    nucleus: str | None,
    solvent: str | None,
    pool_size: int | None,
    qc_passed: bool,
    protocol_version: str | None,
) -> tuple[bool, str | None]:
    problems = signature_problems(signature)
    if problems:
        return False, "; ".join(problems)
    if signature.get("status") != "active":
        return (
            False,
            str(
                signature.get("reason")
                or "calibration applicability signature is not active"
            ),
        )
    expected = {
        "model_sha256": model_sha256,
        "calibrator_sha256": calibrator_sha256,
        "feature_schema": feature_schema,
        "generator_version": generator_version,
        "provider_ids": sorted(str(item) for item in provider_ids),
        "nucleus": nucleus,
        "solvent": solvent,
        "pool_size": pool_size,
        "qc_passed": qc_passed,
        "protocol_version": protocol_version,
    }
    for field, actual in expected.items():
        frozen = signature.get(field)
        if field == "provider_ids":
            if sorted(str(item) for item in (frozen or ())) != actual:
                return False, "provider_ids mismatch"
        elif frozen != actual:
            return False, f"{field} mismatch"
    return True, None


__all__ = [
    "DEFAULT_SIGNATURE_PATH",
    "REQUIRED_FIELDS",
    "SCHEMA_VERSION",
    "check_applicability",
    "load_signature",
    "signature_problems",
]
