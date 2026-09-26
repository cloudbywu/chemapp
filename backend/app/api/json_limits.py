"""Shared client-controlled JSON payload guardrails for API routes.

Historically :mod:`app.api.routes.analysis` and :mod:`app.api.routes.ai`
grew two separate validators with different limits:

* AI chat/action payloads (``ai.py``) are small operator inputs, so they use
  the stricter limits (2000 nodes, depth 8, 20k-char strings).
* Analysis manual-result payloads (``analysis.py``) can legitimately carry
  up to 10k peaks/integrals/multiplets, each with several fields, so they
  need the looser limits (20000 nodes, depth 12, 100k-char strings plus
  object/list cardinality caps).

Unifying on the stricter limits would reject currently legal analysis
requests, so both routes keep their own parameters and share this single
fail-closed implementation: oversized, overly deep, or non-finite payloads
are always rejected before any business logic runs.
"""

from __future__ import annotations

import math
from typing import Any


def validate_json_tree(
    value: Any,
    *,
    max_nodes: int,
    max_depth: int,
    max_string_length: int,
    max_object_fields: int | None = None,
    max_list_length: int | None = None,
) -> Any:
    """Reject payloads that are too large, too deep, or non-finite.

    ``max_object_fields``/``max_list_length`` of ``None`` disable the
    per-container cardinality cap (node and depth limits still apply).
    Returns ``value`` unchanged when valid.
    """

    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        if nodes > max_nodes or depth > max_depth:
            raise ValueError("Request payload is too large or deeply nested")
        if isinstance(item, float) and not math.isfinite(item):
            raise ValueError("NaN and Infinity are not allowed")
        if isinstance(item, str) and len(item) > max_string_length:
            raise ValueError("Request text value is too long")
        if isinstance(item, dict):
            if max_object_fields is not None and len(item) > max_object_fields:
                raise ValueError("Request object has too many fields")
            for key, child in item.items():
                if not isinstance(key, str) or len(key) > 256:
                    raise ValueError("Object keys must be short strings")
                visit(child, depth + 1)
        elif isinstance(item, list):
            if max_list_length is not None and len(item) > max_list_length:
                raise ValueError("Request list is too long")
            for child in item:
                visit(child, depth + 1)

    visit(value, 0)
    return value
