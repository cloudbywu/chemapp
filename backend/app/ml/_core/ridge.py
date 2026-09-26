"""Weighted ridge-logistic fit kernels shared by the NMR calibration modules.

Status: production (frozen numerics).  The two kernels below fit the same
mathematical model -- a group-weighted L2-regularised logistic regression
with intercept -- but they are numerically distinct frozen implementations
and both are preserved exactly:

* :func:`fit_weighted_ridge_logistic_lbfgs` is the preregistered v5
  development kernel (scipy L-BFGS-B with an analytic gradient).
* :func:`fit_weighted_ridge_logistic_newton` is the hybrid research kernel
  (Newton-Raphson with a diagonal ridge load and a +/-30 logit clip).

Consumers map the shared failure modes onto their own frozen exception
types.  For identical inputs both kernels return bit-identical results to
the implementations they replace (locked by
``tests/test_ml_core_golden.py``).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


class RidgeLambdaError(ValueError):
    """The ridge strength was not a positive finite number."""


class RidgeOptimizerError(ValueError):
    """The weighted ridge logistic optimizer failed or went non-finite."""


def fit_weighted_ridge_logistic_lbfgs(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    regularization_lambda: float,
) -> tuple[float, np.ndarray]:
    """Fit intercept and coefficients by weighted ridge logistic (L-BFGS-B).

    The prevalence-clipped initial intercept, objective, gradient, solver
    options, and failure predicates are frozen from the v5 preregistration.
    """

    if regularization_lambda <= 0.0 or not math.isfinite(regularization_lambda):
        raise RidgeLambdaError("regularization lambda must be positive")
    prevalence = float(np.dot(weights, labels) / np.sum(weights))
    prevalence = min(max(prevalence, 1e-9), 1.0 - 1e-9)
    initial = np.zeros(matrix.shape[1] + 1, dtype=float)
    initial[0] = math.log(prevalence / (1.0 - prevalence))

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        linear = parameters[0] + matrix @ parameters[1:]
        loss = np.logaddexp(0.0, linear) - labels * linear
        value = float(np.dot(weights, loss))
        value += 0.5 * regularization_lambda * float(
            np.dot(parameters[1:], parameters[1:])
        )
        residual = weights * (expit(linear) - labels)
        gradient = np.empty_like(parameters)
        gradient[0] = float(np.sum(residual))
        gradient[1:] = matrix.T @ residual + regularization_lambda * parameters[1:]
        return value, gradient

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-9},
    )
    if not result.success or not np.isfinite(result.x).all():
        raise RidgeOptimizerError("weighted ridge logistic optimizer failed")
    return float(result.x[0]), np.asarray(result.x[1:], dtype=float)


def fit_weighted_ridge_logistic_newton(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    lam: float,
) -> tuple[float, np.ndarray, bool]:
    """Weighted ridge logistic via Newton-Raphson (intercept included)."""

    design = np.column_stack([np.ones(x.shape[0]), x])
    beta = np.zeros(design.shape[1], dtype=float)
    converged = False
    for _ in range(50):
        eta = design @ beta
        p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30.0, 30.0)))
        grad = design.T @ (weights * (p - y))
        w_diag = weights * p * (1.0 - p)
        hessian = design.T @ (w_diag[:, None] * design)
        hessian[np.diag_indices_from(hessian)] += lam
        try:
            step = np.linalg.solve(hessian, grad)
        except np.linalg.LinAlgError:
            break
        beta -= step
        if float(np.max(np.abs(step))) < 1e-8:
            converged = True
            break
    return float(beta[0]), beta[1:], converged
