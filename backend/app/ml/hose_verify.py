"""Deprecated HOSE-verification experiment (removed from the ML path).

The previous implementation imported ``app.ml.graphdiff.models.nmr_encoder``
and ``app.ml.graphdiff.featurizer``, which do not exist in this repository,
and loaded pretrained files that are not published.  It also fabricated
retrieval identities from atom-type strings.  Per the 2026-08 ML review, this
code is not production-eligible and has been replaced by:

- ``app/ml/forward_v1/``: ForwardGNN-13C / CSP5q-13C forward evidence;
- ``app/ml/nmr_candidate_generation_v2.py``: hybrid candidate generation;
- ``app/ml/nmr_hybrid_predictor.py``: the auditable hybrid ranking pipeline.

Any HOSE-style per-atom environment verification must be rebuilt from RDKit
HOSE codes with a versioned dataset manifest before it can be considered.
"""

from __future__ import annotations


class HoseVerifyRemovedError(NotImplementedError):
    """Raised when legacy HOSE verification code is invoked."""


def __getattr__(name: str):
    raise HoseVerifyRemovedError(
        f"app.ml.hose_verify.{name} was removed in the 2026-08 ML review; "
        "use the forward_v1 / hybrid predictor modules instead."
    )


__all__: list[str] = []
