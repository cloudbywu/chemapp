"""Finite-number and token guards shared by the versioned ML contracts.

Status: production.  Error classes and message wording are supplied by each
versioned module so its frozen exception contract is untouched; the guard
logic lives only here.
"""

from __future__ import annotations

import math
import re
from typing import Any

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def finite_float(
    value: Any,
    *,
    context: str,
    error: type[ValueError],
    minimum: float | None = None,
) -> float:
    """Return ``value`` as a finite float, rejecting bools and non-numbers."""

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise error(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise error(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise error(f"{context} must be at least {minimum}")
    return result


def nonnegative_int(value: Any, *, context: str, error: type[ValueError]) -> int:
    """Return ``value`` when it is a non-negative (non-bool) integer."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise error(f"{context} must be a non-negative integer")
    return value


def positive_rank(
    value: Any, *, context: str, maximum: int, error: type[ValueError]
) -> int:
    """Return ``value`` when it is a 1-based rank within ``1..maximum``."""

    rank = nonnegative_int(value, context=context, error=error)
    if not 1 <= rank <= maximum:
        raise error(f"{context} must be within 1..{maximum}")
    return rank


def sha256_text(
    value: Any, *, context: str, error: type[ValueError], description: str
) -> str:
    """Return ``value`` when it is a lowercase SHA-256 hex digest."""

    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise error(f"{context} must be {description}")
    return value


def finite_or_default(value: Any, default: float | None) -> float | None:
    """Leniently coerce ``value`` to a finite float, else return ``default``."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def clipped_logistic(linear: float) -> float:
    """Sigmoid with the frozen +/-30 logit clip used by the frozen calibrator."""

    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, linear))))
