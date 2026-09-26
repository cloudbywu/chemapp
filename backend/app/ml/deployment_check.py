"""Startup deployment verification for the CSP5 production forward scorer.

Policy (per owner decision, 2026-08-07):

* ``CHEMAPP_CSP5_MODE=on`` (Docker/compose default): CSP5 is the production
  forward scorer and its bundled weights MUST be present and byte-identical
  to the pinned manifest — otherwise the app fails to start (fail closed).
* ``CHEMAPP_CSP5_MODE=auto`` (local default): CSP5 is used when the weights
  verify; a mismatch only logs a warning so CPU/local GNN fallback remains
  available.
* ``CHEMAPP_CSP5_MODE=off``: verification is skipped.

GPU is preferred automatically by the scorer (CUDA when available, otherwise
CPU), so no separate switch is needed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger("chemapp.deployment")

_BACKEND_ROOT = Path(__file__).resolve().parents[2]
_MANIFEST_PATH = _BACKEND_ROOT / "vendor" / "csp5" / "weights-manifest.json"
_ON_VALUES = {"1", "on", "true", "yes"}


def _normalise_weight_rel(value: str) -> str:
    """Normalise manifest paths so Windows separators work on every OS."""
    return str(value or "").replace("\\", "/")


def csp5_mode() -> str:
    """Return the effective CSP5 mode (auto by default)."""

    return os.environ.get("CHEMAPP_CSP5_MODE", "auto").strip().casefold()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_csp5_weights(
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Verify all CSP5 weight files against the pinned manifest."""

    path = manifest_path or _MANIFEST_PATH
    if not path.is_file():
        raise RuntimeError(f"CSP5 weights manifest missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    vendor_root = path.parent
    checked: list[dict[str, Any]] = []
    for entry in manifest.get("files", []):
        rel = _normalise_weight_rel(str(entry.get("path") or ""))
        weight_path = vendor_root / rel
        if not weight_path.is_file():
            raise RuntimeError(f"CSP5 weight file missing: {rel}")
        actual = _sha256(weight_path)
        expected = str(entry.get("sha256") or "")
        if actual != expected:
            raise RuntimeError(
                f"CSP5 weight hash mismatch for {rel}: "
                f"expected {expected}, got {actual}"
            )
        checked.append(
            {"path": rel, "sha256": actual, "bytes": weight_path.stat().st_size}
        )
    return {
        "mode": csp5_mode(),
        "verified_files": checked,
        "status": "ok",
    }


def verify_deployment() -> dict[str, Any]:
    """Startup check: strict when on, warning when auto, skip when off."""

    mode = csp5_mode()
    if mode in _ON_VALUES:
        return verify_csp5_weights()
    if mode == "auto":
        try:
            return verify_csp5_weights()
        except Exception as exc:
            logger.warning(
                "CSP5 verification failed in auto mode; "
                "production forward scorer unavailable: %s",
                exc,
            )
            return {
                "mode": mode,
                "status": "fallback",
                "reason": str(exc),
            }
    return {"mode": mode, "status": "disabled"}
