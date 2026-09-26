"""Frozen group-weighted ridge logistic for conditional Top-1 probability.

The parameters were trained on retrospective v3+v4 error cases.  v5 and v6
were excluded from that fit, but their ECE/Brier results are retrospective
diagnostics and are not independent of CSP5 training.  Production probability
output is gated by the committed policy file plus
``CHEMAPP_CALIBRATION_MODE`` (computation defaults on); the mode cannot
override ``probability_claim_allowed=false`` in policy.

Semantics: P(Top-1 is the exact structure | truth retrieved, pool size >= 2,
QC passed, fixed generator and ranker).  It is *not* a retrieval or
end-to-end correctness probability, and automatic structure selection stays
disabled by policy.

Status: production (frozen parameters).  The lenient numeric guard and the
clipped sigmoid live in :mod:`app.ml._core`; the frozen parameters, policy
gates, and applicability envelope remain unchanged here.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.ml._core.numeric import clipped_logistic, finite_or_default

_PACKAGE_DIR = Path(__file__).resolve().parent
_DISABLED_VALUES = {"0", "off", "false", "no"}

CALIBRATION_POOL_SIZE = 12
CALIBRATION_MODALITY = "13c"
CALIBRATION_SOLVENTS = {"cdcl3"}
CALIBRATION_PROVIDER = "csp5_forward_13c_v1"
CALIBRATION_MODEL_NAME = "CSP5q-13C"


def _normalise_solvent(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().casefold().replace(" ", "")


def applicability_reason(
    pool: Sequence[Mapping[str, Any]],
    *,
    formula: str | None,
    modality: str | None,
    n_13c: int | None,
    solvent: str | None,
    supplied_candidate_count: int | None,
    generation_status: str | None,
    provider_count: int | None,
    provider_failures: Sequence[str] | None,
    forward_model: Mapping[str, Any] | None,
) -> str | None:
    """Return a machine-readable reason when calibration must not apply.

    The frozen calibrator was trained and validated only on the fixed
    12-candidate, same-formula, CDCl3-only 13C closed-pool blind protocol.
    Production probability must not be emitted outside that envelope until a
    broader validation exists.
    """

    if not formula:
        return "formula_required"
    if modality != CALIBRATION_MODALITY:
        return "modality_not_13c_only"
    if n_13c is None or n_13c < 1:
        return "no_13c_peaks"
    if _normalise_solvent(solvent) not in CALIBRATION_SOLVENTS:
        return "solvent_not_cdcl3"
    if len(pool) != CALIBRATION_POOL_SIZE:
        return f"pool_size_not_{CALIBRATION_POOL_SIZE}"
    if supplied_candidate_count:
        return "user_supplied_candidates_not_supported"
    if generation_status != "completed":
        return f"candidate_generation_{generation_status or 'unknown'}"
    if provider_count is None or provider_count <= 0:
        return "fixed_candidate_provider_required"
    if provider_failures:
        return "candidate_provider_failure"
    if not isinstance(forward_model, Mapping):
        return "forward_model_identity_missing"
    if forward_model.get("provider") != CALIBRATION_PROVIDER:
        return "forward_provider_mismatch"
    if forward_model.get("model_name") != CALIBRATION_MODEL_NAME:
        return "forward_model_mismatch"
    if not forward_model.get("sha256"):
        return "forward_weights_hash_missing"
    for candidate in pool:
        if not isinstance(candidate, Mapping):
            return "invalid_candidate"
        if candidate.get("molecular_formula") != formula:
            return "formula_mismatch_in_pool"
        evidence = candidate.get("_forward_evidence")
        if not isinstance(evidence, Mapping):
            return "forward_evidence_incomplete"
        if evidence.get("prediction_failed") is True:
            return "prediction_failed_in_pool"
    return None


def _load_json(name: str) -> dict[str, Any]:
    path = Path(
        os.environ.get("CHEMAPP_CALIBRATION_DIR", str(_PACKAGE_DIR))
    ) / name
    return json.loads(path.read_text(encoding="utf-8"))


def policy() -> dict[str, Any]:
    """Return the committed calibration policy (production gate)."""

    return _load_json("policy-v1.json")


def enabled() -> bool:
    """True when the external-holder policy allows probability output."""

    mode = os.environ.get("CHEMAPP_CALIBRATION_MODE", "on").strip().casefold()
    if mode in _DISABLED_VALUES:
        return False
    return bool(policy().get("probability_claim_allowed"))


def _finite(value: Any) -> float | None:
    return finite_or_default(value, None)


def build_features(
    candidates: Sequence[Mapping[str, Any]],
) -> list[float] | None:
    """Build the five calibration features from ranked forward evidence.

    ``candidates`` must be in final rank order and each contain
    ``forward_evidence`` (or ``_forward_evidence``) with the standard
    mae/max_abs/coverage fields.
    """

    z: list[float] = []
    for candidate in candidates:
        evidence = candidate.get("forward_evidence")
        if evidence is None:
            evidence = candidate.get("_forward_evidence")
        if not isinstance(evidence, Mapping):
            continue
        mae = _finite(evidence.get("mae_ppm"))
        if mae is not None:
            z.append(-mae)
    if len(z) < 2:
        return None
    z_max = max(z)
    logsumexp = z_max + math.log(sum(math.exp(v - z_max) for v in z))
    pool_log_odds = z[0] - logsumexp
    probs = [math.exp(v - logsumexp) for v in z]
    entropy = -sum(p * math.log(p) for p in probs if p > 0)
    normalized_entropy = entropy / math.log(len(z))

    top_evidence = candidates[0].get("forward_evidence")
    if top_evidence is None:
        top_evidence = candidates[0].get("_forward_evidence")
    if not isinstance(top_evidence, Mapping):
        return None
    top1_mae = _finite(top_evidence.get("mae_ppm"))
    top1_max_abs = _finite(top_evidence.get("max_abs_error_ppm"))
    top1_coverage = _finite(top_evidence.get("bidirectional_coverage"))
    if top1_mae is None or top1_coverage is None:
        return None
    top1_max_abs_value = top1_max_abs if top1_max_abs is not None else top1_mae
    values = [
        pool_log_odds,
        normalized_entropy,
        math.log1p(max(top1_mae, 0.0)),
        math.log1p(max(top1_max_abs_value, 0.0)),
        top1_coverage,
    ]
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def _probability(features: Sequence[float]) -> float | None:
    params = _load_json("calibrator-v1.json")
    coefficients = [float(value) for value in params["coefficients"]]
    if len(coefficients) != len(features):
        return None
    linear = float(params["intercept"]) + sum(
        coefficient * feature
        for coefficient, feature in zip(coefficients, features)
    )
    if not math.isfinite(linear):
        return None
    return clipped_logistic(linear)


def calibrate(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Return calibrated probability metadata, or None when unavailable."""

    if not enabled():
        return None
    features = build_features(candidates)
    if features is None:
        return None
    probability = _probability(features)
    if probability is None:
        return None
    params = _load_json("calibrator-v1.json")
    gate = policy()
    return {
        "probability": round(probability, 6),
        "method": "group_weighted_ridge_top1_context_logistic",
        "feature_names": list(params["feature_names"]),
        "feature_values": [round(value, 6) for value in features],
        "selected_lambda": float(params["selected_lambda"]),
        "validated_ece": {
            name: float(value["ece"])
            for name, value in gate.get("validations", {}).items()
        },
        "semantics": gate.get(
            "semantics",
            "P(Top-1 is exact structure | truth retrieved, pool size >= 2, "
            "QC passed, fixed generator and ranker)",
        ),
        "automatic_selection_allowed": bool(
            gate.get("automatic_selection_allowed", False)
        ),
        "external_holder_pending": bool(
            gate.get("external_holder_pending", True)
        ),
    }


def calibrate_pool(
    pool: Sequence[Mapping[str, Any]],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Calibrate from an internal hybrid pool (``_forward_evidence`` keys)."""

    if context is None:
        return None
    reason = applicability_reason(
        pool,
        formula=context.get("formula"),
        modality=context.get("modality"),
        n_13c=context.get("n_13c"),
        solvent=context.get("solvent"),
        supplied_candidate_count=context.get("supplied_candidate_count"),
        generation_status=context.get("generation_status"),
        provider_count=context.get("provider_count"),
        provider_failures=context.get("provider_failures"),
        forward_model=context.get("forward_model"),
    )
    if reason is not None:
        return None
    from app.ml.calibration.applicability_v1 import (
        check_applicability,
        load_signature,
    )

    signature_ok, signature_reason = check_applicability(
        load_signature(),
        model_sha256=(
            context.get("forward_model", {}).get("sha256")
            if isinstance(context.get("forward_model"), Mapping)
            else None
        ),
        calibrator_sha256=None,
        feature_schema="chemapp.nmr.calibration-features.v1",
        generator_version=context.get("generator_version"),
        provider_ids=context.get("provider_ids", ()),
        nucleus="13C" if context.get("modality") == "13c" else None,
        solvent=context.get("solvent"),
        pool_size=len(pool),
        qc_passed=bool(context.get("qc_passed", False)),
        protocol_version=context.get(
            "protocol_version",
            "chemapp.nmr.calibration-protocol.v2",
        ),
    )
    if not signature_ok:
        return None
    candidates = [
        {"forward_evidence": item.get("_forward_evidence")}
        for item in pool
        if item.get("_forward_evidence") is not None
    ]
    if not candidates:
        return None
    return calibrate(candidates)
