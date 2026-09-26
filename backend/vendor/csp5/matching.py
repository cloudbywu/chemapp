"""Assignment utilities for matching predicted and experimental shifts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ._native import match_indices_dp as _match_indices_dp_native
from ._native import murty_k_best as _murty_k_best_native


MatchingSolver = Literal["dp", "scipy", "murty"]
KBestPolicy = Literal["strict", "clip"]


@dataclass(frozen=True)
class RankedAssignment:
    """One ranked assignment candidate."""

    rank: int
    assignment: Dict[int, int]
    total_cost: float
    mean_abs_error: float


@dataclass(frozen=True)
class MatchingResult:
    """Matching output with optional k-best ambiguity metrics."""

    solver: MatchingSolver
    dummy_cost: float
    best_assignment: Dict[int, int]
    best_total_cost: float
    best_mean_abs_error: float
    ranked_assignments: List[RankedAssignment]
    assignment_entropy: float
    num_competing_assignments: int
    matching_count: int


def _as_vector(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _build_effective_cost(
    predicted: np.ndarray,
    experimental: np.ndarray,
    *,
    row_penalties: Sequence[float] | np.ndarray | None,
    pairwise_cost: Sequence[Sequence[float]] | np.ndarray | None,
) -> Tuple[np.ndarray, np.ndarray]:
    abs_cost = np.abs(predicted[:, None] - experimental[None, :])
    if pairwise_cost is not None:
        effective = np.asarray(pairwise_cost, dtype=np.float64)
        expected = (predicted.shape[0], experimental.shape[0])
        if effective.shape != expected:
            raise ValueError(
                f"pairwise_cost shape mismatch: got {effective.shape}, expected {expected}"
            )
        if not np.isfinite(effective).all():
            raise ValueError("pairwise_cost must be finite")
    else:
        effective = abs_cost.copy()

    if row_penalties is not None:
        penalties = np.asarray(row_penalties, dtype=np.float64).reshape(-1)
        if penalties.shape[0] != predicted.shape[0]:
            raise ValueError(
                f"row_penalties length mismatch: got {penalties.shape[0]}, expected {predicted.shape[0]}"
            )
        if not np.isfinite(penalties).all():
            raise ValueError("row_penalties must contain only finite values")
        effective = effective + penalties[:, None]

    return abs_cost, effective


def _resolve_dummy_cost(effective_cost: np.ndarray, dummy_cost: float | None) -> float:
    if dummy_cost is not None:
        value = float(dummy_cost)
        if not np.isfinite(value):
            raise ValueError(f"dummy_cost must be finite, got {dummy_cost}")
        return value
    finite = effective_cost[np.isfinite(effective_cost)]
    if finite.size == 0:
        raise ValueError("effective cost matrix has no finite values")
    return float(np.max(finite) + 1.0)


def _solve_hungarian(
    effective_cost: np.ndarray,
    *,
    dummy_cost: float,
) -> Tuple[List[int], List[int], float]:
    n_pred, n_obs = effective_cost.shape
    cost = effective_cost.copy()
    if n_pred > n_obs:
        pad = np.full((n_pred, n_pred - n_obs), float(dummy_cost), dtype=np.float64)
        cost = np.concatenate([cost, pad], axis=1)
    elif n_obs > n_pred:
        pad = np.full((n_obs - n_pred, n_obs), float(dummy_cost), dtype=np.float64)
        cost = np.concatenate([cost, pad], axis=0)

    row_ind, col_ind = linear_sum_assignment(cost)
    total = float(cost[row_ind, col_ind].sum())
    rows: List[int] = []
    cols: List[int] = []
    for row_idx, col_idx in zip(row_ind, col_ind):
        if int(row_idx) < n_pred and int(col_idx) < n_obs:
            rows.append(int(row_idx))
            cols.append(int(col_idx))
    return rows, cols, total


def _pad_square(effective_cost: np.ndarray, dummy_cost: float) -> np.ndarray:
    n_pred, n_obs = effective_cost.shape
    n_dim = max(n_pred, n_obs)
    square = np.full((n_dim, n_dim), float(dummy_cost), dtype=np.float64)
    square[:n_pred, :n_obs] = effective_cost
    return square


def _compute_entropy(
    maes: np.ndarray,
    *,
    temperature: float,
    mae_delta_threshold: float,
) -> Tuple[float, int]:
    if maes.size == 0:
        return 0.0, 0
    best = float(np.min(maes))
    close_mask = maes <= best + float(mae_delta_threshold)
    close_maes = maes[close_mask]
    if close_maes.size <= 1:
        return 0.0, int(close_maes.size)
    delta = close_maes - float(np.min(close_maes))
    weights = np.exp(-delta / float(temperature))
    probs = weights / float(np.sum(weights))
    entropy = float(-(probs * np.log(probs)).sum())
    return entropy, int(close_maes.size)


def _parse_murty_solutions(
    raw_solutions: List[Tuple[np.ndarray, float]],
    *,
    n_pred: int,
    n_obs: int,
    effective_cost: np.ndarray,
    abs_cost: np.ndarray,
    dummy_cost: float,
) -> List[Tuple[Dict[int, int], float, float]]:
    required_matches = min(int(n_pred), int(n_obs))
    by_assignment: Dict[Tuple[Tuple[int, int], ...], Tuple[float, float, Dict[int, int]]] = {}

    unmatched_penalty = float(abs(int(n_pred) - int(n_obs)) * float(dummy_cost))

    for matches, _total_cost in raw_solutions:
        assignment: Dict[int, int] = {}
        for pair in np.asarray(matches, dtype=np.int32):
            row_idx = int(pair[0])
            col_idx = int(pair[1])
            if row_idx < int(n_pred) and col_idx < int(n_obs):
                assignment[row_idx] = col_idx

        if len(assignment) != required_matches:
            continue

        assign_key = tuple(sorted(assignment.items()))
        mae = (
            float(np.mean([abs_cost[r, c] for r, c in assignment.items()])) if assignment else 0.0
        )
        total = (
            float(np.sum([effective_cost[r, c] for r, c in assignment.items()]))
            + unmatched_penalty
        )

        prev = by_assignment.get(assign_key)
        if prev is None or total < prev[0]:
            by_assignment[assign_key] = (total, mae, assignment)

    parsed = [(v[2], v[0], v[1]) for v in by_assignment.values()]
    parsed.sort(key=lambda item: (item[1], tuple(sorted(item[0].items()))))
    return parsed


def _request_native_murty(
    square_cost: np.ndarray,
    *,
    requested: int,
    policy: KBestPolicy,
    n_pred: int,
    n_obs: int,
    effective_cost: np.ndarray,
    abs_cost: np.ndarray,
    dummy_cost: float,
) -> List[Tuple[Dict[int, int], float, float]]:
    def _solve(nsolutions: int) -> List[Tuple[Dict[int, int], float, float]]:
        raw = _murty_k_best_native(square_cost, nsolutions=int(nsolutions))
        return _parse_murty_solutions(
            raw,
            n_pred=n_pred,
            n_obs=n_obs,
            effective_cost=effective_cost,
            abs_cost=abs_cost,
            dummy_cost=dummy_cost,
        )

    def _largest_feasible(max_n: int) -> List[Tuple[Dict[int, int], float, float]]:
        lo = 1
        hi = int(max_n)
        best: List[Tuple[Dict[int, int], float, float]] = []
        while lo <= hi:
            mid = (lo + hi) // 2
            try:
                cur = _solve(mid)
            except RuntimeError:
                hi = mid - 1
                continue
            best = cur
            lo = mid + 1
        if not best:
            raise RuntimeError("Murty produced zero unique feasible assignments")
        return best

    try:
        parsed = _solve(requested)
    except RuntimeError:
        if policy == "clip":
            parsed = _largest_feasible(requested)
        else:
            parsed = _largest_feasible(requested)
            raise RuntimeError(
                f"Murty produced only {len(parsed)} unique feasible assignments but k_best={requested}"
            )

    return parsed


def match_shifts(
    predicted_shifts: Sequence[float] | np.ndarray,
    experimental_shifts: Sequence[float] | np.ndarray,
    *,
    solver: MatchingSolver = "dp",
    k_best: int = 1,
    k_best_policy: KBestPolicy = "clip",
    temperature: float = 0.5,
    mae_delta_threshold: float = 0.2,
    dummy_cost: float | None = None,
    row_penalties: Sequence[float] | np.ndarray | None = None,
    pairwise_cost: Sequence[Sequence[float]] | np.ndarray | None = None,
) -> MatchingResult:
    """
    Match predicted vs experimental shifts.

    Solver modes:
      - `dp`: native C++ order-preserving dynamic programming (default).
      - `scipy`: Hungarian assignment via SciPy (`k_best` must be 1).
      - `murty`: native fastmurty k-best ranking.
    """
    if solver not in {"dp", "scipy", "murty"}:
        raise ValueError(f"Unsupported solver {solver!r}. Use one of: dp, scipy, murty")
    if int(k_best) < 1:
        raise ValueError(f"k_best must be >= 1, got {k_best}")
    if float(temperature) <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if float(mae_delta_threshold) < 0:
        raise ValueError(f"mae_delta_threshold must be >= 0, got {mae_delta_threshold}")
    if k_best_policy not in {"strict", "clip"}:
        raise ValueError(
            f"Unsupported k_best_policy {k_best_policy!r}. Use one of: strict, clip"
        )
    if solver in {"dp", "scipy"} and int(k_best) != 1:
        raise ValueError(
            f"solver={solver!r} only supports k_best=1; use solver='murty' for k-best"
        )

    predicted = _as_vector(predicted_shifts, name="predicted_shifts")
    experimental = _as_vector(experimental_shifts, name="experimental_shifts")
    penalties_arr = None
    if row_penalties is not None:
        penalties_arr = np.asarray(row_penalties, dtype=np.float64).reshape(-1)

    abs_cost, effective_cost = _build_effective_cost(
        predicted,
        experimental,
        row_penalties=penalties_arr,
        pairwise_cost=pairwise_cost,
    )
    resolved_dummy = _resolve_dummy_cost(effective_cost, dummy_cost)
    n_pred, n_obs = effective_cost.shape

    if solver == "dp":
        if pairwise_cost is not None:
            raise ValueError("solver='dp' does not support pairwise_cost; use solver='scipy' or 'murty'")

        rows, cols = _match_indices_dp_native(
            predicted,
            experimental,
            dummy_cost=resolved_dummy,
            row_penalties=penalties_arr,
        )
        assignment = {int(r): int(c) for r, c in zip(rows, cols)}
        matched_cost = float(sum(effective_cost[r, c] for r, c in assignment.items()))
        total_cost = float(matched_cost + abs(n_pred - n_obs) * resolved_dummy)
        mae = float(np.mean([abs_cost[r, c] for r, c in assignment.items()])) if assignment else 0.0

        ranked = [
            RankedAssignment(
                rank=1,
                assignment=assignment,
                total_cost=total_cost,
                mean_abs_error=mae,
            )
        ]
        return MatchingResult(
            solver="dp",
            dummy_cost=resolved_dummy,
            best_assignment=assignment,
            best_total_cost=total_cost,
            best_mean_abs_error=mae,
            ranked_assignments=ranked,
            assignment_entropy=0.0,
            num_competing_assignments=1,
            matching_count=len(assignment),
        )

    if solver == "scipy":
        rows, cols, total_cost = _solve_hungarian(effective_cost, dummy_cost=resolved_dummy)
        assignment = {int(r): int(c) for r, c in zip(rows, cols)}
        mae = float(np.mean([abs_cost[r, c] for r, c in assignment.items()])) if assignment else 0.0

        ranked = [
            RankedAssignment(
                rank=1,
                assignment=assignment,
                total_cost=float(total_cost),
                mean_abs_error=mae,
            )
        ]
        return MatchingResult(
            solver="scipy",
            dummy_cost=resolved_dummy,
            best_assignment=assignment,
            best_total_cost=float(total_cost),
            best_mean_abs_error=mae,
            ranked_assignments=ranked,
            assignment_entropy=0.0,
            num_competing_assignments=1,
            matching_count=len(assignment),
        )

    # solver == "murty"
    square_cost = _pad_square(effective_cost, resolved_dummy)
    parsed = _request_native_murty(
        square_cost,
        requested=int(k_best),
        policy=k_best_policy,
        n_pred=int(n_pred),
        n_obs=int(n_obs),
        effective_cost=effective_cost,
        abs_cost=abs_cost,
        dummy_cost=resolved_dummy,
    )

    if len(parsed) < int(k_best) and k_best_policy == "strict":
        raise RuntimeError(
            f"Murty produced only {len(parsed)} unique feasible assignments but k_best={k_best}"
        )
    if not parsed:
        raise RuntimeError("Murty produced zero unique feasible assignments")

    k_effective = min(int(k_best), len(parsed))
    ranked = [
        RankedAssignment(
            rank=idx + 1,
            assignment=assignment,
            total_cost=total_cost,
            mean_abs_error=mae,
        )
        for idx, (assignment, total_cost, mae) in enumerate(parsed[:k_effective])
    ]

    best = ranked[0]
    maes = np.asarray([item.mean_abs_error for item in ranked], dtype=np.float64)
    entropy, competing = _compute_entropy(
        maes,
        temperature=float(temperature),
        mae_delta_threshold=float(mae_delta_threshold),
    )

    return MatchingResult(
        solver="murty",
        dummy_cost=resolved_dummy,
        best_assignment=dict(best.assignment),
        best_total_cost=float(best.total_cost),
        best_mean_abs_error=float(best.mean_abs_error),
        ranked_assignments=ranked,
        assignment_entropy=float(entropy),
        num_competing_assignments=int(competing),
        matching_count=len(best.assignment),
    )
