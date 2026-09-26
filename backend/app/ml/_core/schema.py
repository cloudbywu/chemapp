"""Strict key-allowlist validation shared by the frozen scorer contracts.

Status: production.  The two message wordings below are frozen contract
text: ``nmr_candidate_scorer_v4``/``v5`` report an allowlist mismatch while
``nmr_candidate_scorer_v8``/``v9`` report a plain field mismatch.  The
comparison logic is identical; only the raised message differs.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def strict_allowlist_keys(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    context: str,
    error: type[ValueError],
) -> None:
    """Require exactly ``expected`` keys, with the frozen allowlist wording."""

    actual = set(value)
    if actual != expected:
        raise error(
            f"{context} fields do not match the strict allowlist; "
            f"missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def strict_fields_mismatch(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    context: str,
    error: type[ValueError],
) -> None:
    """Require exactly ``expected`` keys, with the frozen mismatch wording."""

    actual = set(value)
    if actual != expected:
        raise error(
            f"{context} fields mismatch; missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )
