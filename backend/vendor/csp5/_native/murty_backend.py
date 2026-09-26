"""Native Murty backend wrapper."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from . import fastmurty_clink as _fm


def murty_k_best(cost_matrix: np.ndarray, *, nsolutions: int) -> List[Tuple[np.ndarray, float]]:
    """
    Run native Murty k-best ranking on a cost matrix.

    Returns a list of `(matches, total_cost)` tuples where `matches` is an int32
    array with shape `(n_matches, 2)` containing `(row_idx, col_idx)` pairs.
    """
    cost_matrix = np.asarray(cost_matrix, dtype=np.float64)
    if cost_matrix.ndim != 2:
        raise ValueError(f"cost_matrix must be 2D, got shape {cost_matrix.shape}")

    nrows, ncols = cost_matrix.shape
    if nrows == 0 or ncols == 0:
        return []

    nsolutions = int(nsolutions)
    if nsolutions < 1:
        raise ValueError(f"nsolutions must be >= 1, got {nsolutions}")

    # fastmurty treats misses as zero-cost. Shift all entries negative so real
    # assignments dominate over misses for standard positive-cost matrices.
    offset = float(np.max(cost_matrix)) + 1.0
    shifted_cost = cost_matrix - offset

    if _fm.sparse:
        cost_use = _fm.sparsifyByRow(shifted_cost, ncols)
    else:  # pragma: no cover
        cost_use = shifted_cost

    row_priors = np.ones((1, nrows), dtype=bool)
    col_priors = np.ones((1, ncols), dtype=bool)
    row_prior_weights = np.zeros(1, dtype=np.float64)
    col_prior_weights = np.zeros(1, dtype=np.float64)

    out_costs = np.zeros(nsolutions, dtype=np.float64)
    out_associations = np.zeros((nsolutions, nrows + ncols, 2), dtype=np.int32)

    workvars = _fm.allocateWorkvarsforDA(nrows, ncols, nsolutions)
    try:
        _fm.mhtda(
            cost_use,
            row_priors,
            row_prior_weights,
            col_priors,
            col_prior_weights,
            out_associations,
            out_costs,
            workvars,
        )
    finally:
        _fm.deallocateWorkvarsforDA(workvars)

    solutions: List[Tuple[np.ndarray, float]] = []
    for idx in range(nsolutions):
        assoc = out_associations[idx]
        mask = (assoc[:, 0] >= 0) & (assoc[:, 1] >= 0)
        matches = assoc[mask]
        corrected_total = float(out_costs[idx] + offset * matches.shape[0])
        solutions.append((matches.copy(), corrected_total))
    return solutions
