"""Single source for the canonical JSON encodings used across NMR contracts.

Status: production (frozen bytes).  The encoding produced here
(``ensure_ascii=False``, sorted keys, minimal separators, no NaN/Infinity)
is a frozen contract: any change would invalidate committed artifact
hashes, ledger entries, and model artifacts across the versioned scorer
and calibration modules.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def canonical_json_text(value: Any) -> str:
    """Render deterministic JSON text and reject non-finite JSON numbers."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return the one canonical UTF-8 JSON byte string used by every hash."""

    return canonical_json_text(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hash a value's canonical JSON representation."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def artifact_with_hash(core: Mapping[str, Any]) -> dict[str, Any]:
    """Attach the canonical ``artifact_sha256`` digest to a fresh artifact."""

    materialized = dict(core)
    return {**materialized, "artifact_sha256": canonical_sha256(materialized)}
