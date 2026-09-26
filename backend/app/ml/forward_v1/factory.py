"""Forward scorer factory: prefer the CSP5 quantile model, fall back to GNN."""

from __future__ import annotations

import logging
import threading
from typing import Any

from .csp5_scorer import Csp5ForwardScorer
from .scorer import LocalForwardScorer

logger = logging.getLogger("chemapp.ml.forward_v1")

_cache_lock = threading.Lock()
_scorer_cache: dict[str, Any] = {}


def get_forward_scorer(prefer: str = "csp5") -> Any | None:
    """Return a process-wide cached forward scorer (created once)."""

    with _cache_lock:
        cached = _scorer_cache.get(prefer)
        if cached is not None:
            return cached
    scorer = create_forward_scorer(prefer=prefer)
    if scorer is not None:
        with _cache_lock:
            _scorer_cache[prefer] = scorer
    return scorer


def create_forward_scorer(prefer: str = "csp5") -> Any | None:
    """Create the best available 13C forward scorer.

    ``prefer="csp5"`` (default) tries the vendored CSP5q-13C quantile model
    first and falls back to the local ForwardGNN-13C checkpoint.  Use
    ``prefer="local"`` to force the local GNN.
    """
    order = ("csp5", "local") if prefer == "csp5" else ("local", "csp5")
    last_error: Exception | None = None
    for name in order:
        try:
            if name == "csp5":
                return Csp5ForwardScorer()
            return LocalForwardScorer()
        except Exception as exc:
            last_error = exc
            logger.warning("forward scorer %r unavailable: %s", name, exc)
            continue
    if last_error is not None:
        logger.error("no forward scorer available; last error: %s", last_error)
    return None


__all__ = ["create_forward_scorer"]
