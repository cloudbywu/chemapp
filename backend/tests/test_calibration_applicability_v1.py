from __future__ import annotations

from app.ml.calibration.applicability_v1 import (
    check_applicability,
    load_signature,
    signature_problems,
)


def _active_signature() -> dict:
    return {
        "schema_version": "chemapp.nmr.calibration-applicability.v1",
        "status": "active",
        "model_sha256": "a" * 64,
        "calibrator_sha256": "b" * 64,
        "feature_schema": "chemapp.nmr.calibration-features.v1",
        "generator_version": "hybrid-index-formula-plus-pubchem-fastformula-v1",
        "provider_ids": ["local_index", "pubchem_fastformula"],
        "nucleus": "13C",
        "solvent": "CDCl3",
        "pool_size": 12,
        "qc_passed": True,
        "protocol_version": "chemapp.nmr.calibration-protocol.v2",
    }


def test_default_signature_is_blocked() -> None:
    signature = load_signature()
    assert signature["status"] == "blocked_no_eligible_calibration_data"
    ok, reason = check_applicability(
        signature,
        model_sha256="a" * 64,
        calibrator_sha256="b" * 64,
        feature_schema="chemapp.nmr.calibration-features.v1",
        generator_version="hybrid-index-formula-plus-pubchem-fastformula-v1",
        provider_ids=["local_index", "pubchem_fastformula"],
        nucleus="13C",
        solvent="CDCl3",
        pool_size=12,
        qc_passed=True,
        protocol_version="chemapp.nmr.calibration-protocol.v2",
    )
    assert ok is False
    assert reason is not None


def test_active_signature_matches_runtime_context() -> None:
    signature = _active_signature()
    assert signature_problems(signature) == []
    ok, reason = check_applicability(
        signature,
        model_sha256="a" * 64,
        calibrator_sha256="b" * 64,
        feature_schema="chemapp.nmr.calibration-features.v1",
        generator_version="hybrid-index-formula-plus-pubchem-fastformula-v1",
        provider_ids=["pubchem_fastformula", "local_index"],
        nucleus="13C",
        solvent="CDCl3",
        pool_size=12,
        qc_passed=True,
        protocol_version="chemapp.nmr.calibration-protocol.v2",
    )
    assert ok is True
    assert reason is None


def test_active_signature_rejects_mismatch() -> None:
    signature = _active_signature()
    ok, reason = check_applicability(
        signature,
        model_sha256="a" * 64,
        calibrator_sha256="b" * 64,
        feature_schema="chemapp.nmr.calibration-features.v1",
        generator_version="different-generator",
        provider_ids=["local_index", "pubchem_fastformula"],
        nucleus="13C",
        solvent="CDCl3",
        pool_size=12,
        qc_passed=True,
        protocol_version="chemapp.nmr.calibration-protocol.v2",
    )
    assert ok is False
    assert reason == "generator_version mismatch"


def test_invalid_active_signature_is_rejected() -> None:
    signature = _active_signature()
    signature.pop("pool_size")
    assert any("missing field: pool_size" in item for item in signature_problems(signature))
