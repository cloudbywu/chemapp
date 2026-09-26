"""Research-only group-weighted ridge calibration for hybrid Top-1 evidence.

The pre-registered v5 calibration protocol remains the product path.  This
module implements the same group-weighted ridge logistic method over
outcome-free features derived from the hybrid retrospective pipeline
(2D GNN pre-rank + CSP5q-13C refine).  It always emits
``probability_claim_allowed=false`` until an external holder test exists.

Status: research.  The Newton ridge-logistic kernel and the lenient numeric
guard live in :mod:`app.ml._core`; the hybrid feature row, group folds, and
gates remain frozen here.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np

from app.ml._core.numeric import finite_or_default
from app.ml._core.ridge import fit_weighted_ridge_logistic_newton


FEATURE_NAMES = (
    "pool_log_odds",
    "log_candidate_count",
    "normalized_pool_entropy",
    "log1p_top1_mae_ppm",
    "log1p_top1_max_abs_error_ppm",
    "top1_bidirectional_coverage",
    "model_agreement",
)
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
MIN_ERRORS = 20
ECE_GATE = 0.05


def _finite(value: Any, default: float | None = None) -> float | None:
    return finite_or_default(value, default)


def feature_row(case: Mapping[str, Any]) -> dict[str, Any] | None:
    """Build one outcome-free conditional Top-1 row from a retrospective case."""
    if case.get("status") != "ok":
        return None
    if not case.get("truth_in_pool"):
        return None
    features = case.get("calibration_features") or {}
    scores = case.get("candidate_scores") or []
    n = int(features.get("pool_size") or len(scores))
    if n < 2:
        return None
    top1_mae = _finite(features.get("top1_mae_ppm"))
    top1_max_abs = _finite(features.get("top1_max_abs_error_ppm"))
    top1_coverage = _finite(features.get("top1_bidirectional_coverage"))
    if top1_mae is None or top1_coverage is None:
        return None
    z = [
        -_finite(item.get("mae_ppm"))
        for item in scores
        if _finite(item.get("mae_ppm")) is not None
    ]
    if len(z) < 2:
        return None
    z_max = max(z)
    logsumexp = z_max + math.log(sum(math.exp(v - z_max) for v in z))
    pool_log_odds = z[0] - logsumexp
    probs = [
        math.exp(v - logsumexp) for v in z
    ]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    normalized_entropy = entropy / math.log(len(z)) if len(z) > 1 else 0.0
    top1_max_abs_value = (
        top1_max_abs
        if top1_max_abs is not None
        else top1_mae
    )
    gcn_top1 = case.get("gcn_top1_smiles")
    csp5_top1 = case.get("csp5_top1_smiles")
    agreement = (
        1.0
        if gcn_top1 is not None and csp5_top1 is not None and gcn_top1 == csp5_top1
        else 0.0
    )
    values = [
        pool_log_odds,
        math.log(max(n, 1)),
        normalized_entropy,
        math.log1p(max(top1_mae, 0.0)),
        math.log1p(max(top1_max_abs_value, 0.0)),
        top1_coverage,
        agreement,
    ]
    if not all(math.isfinite(v) for v in values):
        return None
    return {
        "record_id": str(case.get("record_id") or ""),
        "group_id": str(case.get("molecule_group_id") or case.get("record_id")),
        "label": 1 if case.get("top1_correct") else 0,
        "values": values,
    }


def conditional_rows(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = [row for case in cases if (row := feature_row(case)) is not None]
    rows.sort(key=lambda row: (row["group_id"], row["record_id"]))
    return rows


def group_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts = Counter(row["group_id"] for row in rows)
    weights = np.asarray(
        [1.0 / counts[row["group_id"]] for row in rows], dtype=float
    )
    return weights / weights.sum() * len(rows)


def _matrix(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([row["values"] for row in rows], dtype=float)
    y = np.asarray([row["label"] for row in rows], dtype=float)
    return x, y


def fit_ridge_logistic(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lam: float,
) -> tuple[float, np.ndarray, bool]:
    """Weighted ridge logistic via Newton-Raphson (intercept included)."""

    return fit_weighted_ridge_logistic_newton(x, y, weights, lam)


def _predict_probabilities(
    x: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
) -> np.ndarray:
    eta = intercept + x @ coefficients
    return 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))


def _group_kfold(rows: Sequence[Mapping[str, Any]], k: int = 5):
    groups = sorted({row["group_id"] for row in rows})
    groups.sort(key=lambda g: -sum(1 for r in rows if r["group_id"] == g))
    folds: list[list[str]] = [[] for _ in range(k)]
    for index, group in enumerate(groups):
        folds[index % k].append(group)
    return folds


def _group_weighted_logloss(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    intercept: float,
    coefficients: np.ndarray,
) -> float:
    p = np.clip(
        _predict_probabilities(x, intercept, coefficients), 1e-7, 1 - 1e-7
    )
    return float(
        np.sum(weights * (-(y * np.log(p) + (1 - y) * np.log(1 - p))))
        / weights.sum()
    )


def select_lambda(
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: int = 5,
) -> dict[str, Any]:
    k = min(folds, len({row["group_id"] for row in rows}))
    if k < 2:
        return {"selected_lambda": None, "blocked": ["too_few_groups_for_cv"]}
    fold_groups = _group_kfold(rows, k)
    results = []
    for lam in LAMBDA_GRID:
        losses = []
        for fold in range(k):
            test_groups = set(fold_groups[fold])
            train = [
                row for row in rows if row["group_id"] not in test_groups
            ]
            test = [row for row in rows if row["group_id"] in test_groups]
            if not train or not test:
                continue
            x_train, y_train = _matrix(train)
            weights_train = group_weights(train)
            intercept, coefficients, _ = fit_ridge_logistic(
                x_train, y_train, weights_train, lam
            )
            x_test, y_test = _matrix(test)
            weights_test = group_weights(test)
            losses.append(
                _group_weighted_logloss(
                    x_test, y_test, weights_test, intercept, coefficients
                )
            )
        if losses:
            results.append(
                {
                    "lambda": lam,
                    "cv_logloss": float(np.mean(losses)),
                    "folds": len(losses),
                }
            )
    if not results:
        return {"selected_lambda": None, "blocked": ["no_cv_results"]}
    results.sort(key=lambda item: item["cv_logloss"])
    return {"selected_lambda": results[0]["lambda"], "cv_results": results}


def evaluate_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    selected_lambda: float,
    folds: int = 5,
) -> dict[str, Any]:
    """Nested group-CV ECE/Brier with the selected lambda."""
    k = min(folds, len({row["group_id"] for row in rows}))
    fold_groups = _group_kfold(rows, k)
    probabilities = np.zeros(len(rows), dtype=float)
    for fold in range(k):
        test_groups = set(fold_groups[fold])
        train = [row for row in rows if row["group_id"] not in test_groups]
        test_indices = [
            index for index, row in enumerate(rows) if row["group_id"] in test_groups
        ]
        if not train or not test_indices:
            continue
        x_train, y_train = _matrix(train)
        weights_train = group_weights(train)
        intercept, coefficients, _ = fit_ridge_logistic(
            x_train, y_train, weights_train, selected_lambda
        )
        x_test, _ = _matrix([rows[i] for i in test_indices])
        probabilities[test_indices] = _predict_probabilities(
            x_test, intercept, coefficients
        )
    _, labels = _matrix(rows)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    brier = float(np.mean((probabilities - labels) ** 2))
    base_brier = float(np.mean((labels.mean() - labels) ** 2))
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (probabilities >= left) & (probabilities < right)
        if not mask.any():
            continue
        ece += (
            float(np.mean(np.abs(probabilities[mask] - labels[mask])))
            * mask.sum()
            / len(labels)
        )
    errors = int(np.sum(labels == 0))
    blocked = []
    if errors < MIN_ERRORS:
        blocked.append(f"insufficient_errors:{errors}_lt_{MIN_ERRORS}")
    if ece > ECE_GATE:
        blocked.append(f"ece_above_gate:{ece:.4f}")
    if brier >= base_brier:
        blocked.append("brier_not_better_than_baseline")
    return {
        "ece": ece,
        "brier": brier,
        "baseline_brier": base_brier,
        "errors": errors,
        "n_rows": len(rows),
        "blocked": blocked,
        "status": "blocked" if blocked else "passed",
    }


def run_hybrid_calibration(
    cases: Sequence[Mapping[str, Any]],
    *,
    folds: int = 5,
) -> dict[str, Any]:
    rows = conditional_rows(cases)
    if not rows:
        return {
            "schema_version": "chemapp.nmr.hybrid-calibration-research.v1",
            "status": "blocked",
            "blocking_reasons": ["no_conditional_rows"],
            "n_rows": 0,
            "errors": 0,
            "probability_claim_allowed": False,
        }
    selection = select_lambda(rows, folds=folds)
    if selection.get("blocked"):
        return {
            "schema_version": "chemapp.nmr.hybrid-calibration-research.v1",
            "status": "blocked",
            "blocking_reasons": selection["blocked"],
            "n_rows": len(rows),
            "errors": sum(1 for row in rows if row["label"] == 0),
            "probability_claim_allowed": False,
        }
    selected = float(selection["selected_lambda"])
    gate = evaluate_gate(rows, selected_lambda=selected, folds=folds)
    x, y = _matrix(rows)
    weights = group_weights(rows)
    intercept, coefficients, converged = fit_ridge_logistic(x, y, weights, selected)
    if not converged:
        return {
            "schema_version": "chemapp.nmr.hybrid-calibration-research.v1",
            "status": "blocked",
            "blocking_reasons": [
                *(gate.get("blocked") or []),
                "ridge_logistic_not_converged",
            ],
            "n_rows": len(rows),
            "errors": gate["errors"],
            "probability_claim_allowed": False,
        }
    return {
        "schema_version": "chemapp.nmr.hybrid-calibration-research.v1",
        "status": gate["status"],
        "blocking_reasons": gate["blocked"],
        "method": "group_weighted_ridge_top1_context_logistic",
        "feature_names": list(FEATURE_NAMES),
        "selected_lambda": selected,
        "lambda_cv": selection.get("cv_results"),
        "gate": gate,
        "parameters": {
            "intercept": intercept,
            "coefficients": coefficients.tolist(),
        },
        "n_rows": len(rows),
        "errors": gate["errors"],
        "probability_claim_allowed": False,
        "conditional_target_semantics": (
            "P(Top-1 is exact structure | truth retrieved, pool size >= 2, "
            "QC passed, fixed generator and ranker)"
        ),
        "not_retrieval_probability": True,
        "not_end_to_end_probability": True,
    }


__all__ = [
    "FEATURE_NAMES",
    "conditional_rows",
    "run_hybrid_calibration",
]
