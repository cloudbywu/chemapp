"""Tests for the frozen conditional Top-1 calibration module."""

from __future__ import annotations

import math

import pytest

from app.ml.calibration import calibrate, calibrate_pool, enabled, policy
from app.ml.calibration.calibrator import applicability_reason, build_features


def _evidence(mae: float, max_abs: float, coverage: float = 1.0) -> dict:
    return {
        "mae_ppm": mae,
        "max_abs_error_ppm": max_abs,
        "bidirectional_coverage": coverage,
    }


def _candidates() -> list[dict]:
    return [
        {"forward_evidence": _evidence(0.2, 0.4, 1.0)},
        {"forward_evidence": _evidence(1.2, 1.8, 1.0)},
        {"forward_evidence": _evidence(2.0, 2.5, 1.0)},
    ]


def _pool12(*, formula: str = "C2H6O", failed: bool = False) -> list[dict]:
    return [
        {
            "molecular_formula": formula,
            "_forward_evidence": (
                {"prediction_failed": True}
                if failed and index == 0
                else _evidence(0.2 + index * 0.1, 0.4 + index * 0.1)
            ),
        }
        for index in range(12)
    ]


def _valid_context() -> dict:
    return {
        "formula": "C2H6O",
        "modality": "13c",
        "n_13c": 6,
        "solvent": "CDCl3",
        "supplied_candidate_count": 0,
        "generation_status": "completed",
        "provider_count": 1,
        "provider_failures": [],
        "forward_model": {
            "provider": "csp5_forward_13c_v1",
            "model_name": "CSP5q-13C",
            "sha256": "0123456789abcdef",
        },
    }


def test_policy_blocks_probability_claims_until_external_holder() -> None:
    gate = policy()
    assert gate["probability_claim_allowed"] is False
    assert gate["external_holder_pending"] is True
    assert gate["automatic_selection_allowed"] is False


def test_enabled_reflects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    assert enabled() is False
    monkeypatch.setenv("CHEMAPP_CALIBRATION_MODE", "off")
    assert enabled() is False
    monkeypatch.setenv("CHEMAPP_CALIBRATION_MODE", "on")
    assert enabled() is False


def test_build_features_returns_five_values() -> None:
    features = build_features(_candidates())
    assert features is not None
    assert len(features) == 5
    assert all(math.isfinite(value) for value in features)


def test_build_features_requires_two_finite_maes() -> None:
    candidates = [
        {"forward_evidence": _evidence(0.2, 0.4, 1.0)},
        {"forward_evidence": {"mae_ppm": None, "bidirectional_coverage": 1.0}},
    ]
    assert build_features(candidates) is None


def test_calibrate_returns_none_while_gate_closed() -> None:
    result = calibrate(_candidates())
    assert result is None


def test_calibrate_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHEMAPP_CALIBRATION_MODE", "off")
    assert calibrate(_candidates()) is None


def test_calibrate_pool_uses_internal_forward_key() -> None:
    assert calibrate_pool(_pool12()) is None
    assert calibrate_pool(_pool12(), context=_valid_context()) is None


def test_applicability_reason_accepts_validated_envelope() -> None:
    assert applicability_reason(_pool12(), **_valid_context()) is None


def test_applicability_reason_fails_closed_outside_envelope() -> None:
    context = _valid_context()
    assert (
        applicability_reason(_pool12(), **{**context, "formula": None})
        == "formula_required"
    )
    assert (
        applicability_reason(_pool12()[:3], **context)
        == "pool_size_not_12"
    )
    assert (
        applicability_reason(_pool12(), **{**context, "modality": "1h+13c"})
        == "modality_not_13c_only"
    )
    assert (
        applicability_reason(_pool12(), **{**context, "solvent": "DMSO"})
        == "solvent_not_cdcl3"
    )
    assert (
        applicability_reason(
            _pool12(),
            **{**context, "supplied_candidate_count": 1},
        )
        == "user_supplied_candidates_not_supported"
    )
    assert (
        applicability_reason(
            _pool12(),
            **{**context, "generation_status": "partial"},
        )
        == "candidate_generation_partial"
    )
    assert (
        applicability_reason(_pool12(), **{**context, "provider_count": 0})
        == "fixed_candidate_provider_required"
    )
    assert (
        applicability_reason(_pool12(failed=True), **context)
        == "prediction_failed_in_pool"
    )
