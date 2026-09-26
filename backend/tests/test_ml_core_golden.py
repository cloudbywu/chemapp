"""Golden equivalence tests for the app.ml._core extraction.

Every constant below was generated from the pre-refactor implementations.
The refactor may not change canonical JSON bytes, content hashes, schema
acceptance sets, error wording, or ridge numerics; these tests fail on any
drift.
"""

from __future__ import annotations

import pytest

from app.ml import nmr_candidate_scorer_v4 as scorer_v4
from app.ml import nmr_candidate_scorer_v5 as scorer_v5
from app.ml import nmr_candidate_scorer_v8 as scorer_v8
from app.ml import nmr_candidate_scorer_v9 as scorer_v9
from app.ml import nmr_calibration_v4 as calibration_v4
from app.ml import nmr_calibration_v5 as calibration_v5
from app.ml import nmr_hybrid_calibration_v1 as hybrid_v1
from app.ml._core.canonical import (
    artifact_with_hash,
    canonical_json_bytes,
    canonical_sha256,
)
from app.ml._core.ridge import (
    RidgeLambdaError,
    fit_weighted_ridge_logistic_lbfgs,
    fit_weighted_ridge_logistic_newton,
)
from app.ml.calibration.calibrator import build_features

CANONICAL_FIXTURE = {
    "zeta": [3, 1.5, None, True, {"b": "umlaut-u00fc", "a": -0.25}],
    "alpha": "text",
}
CANONICAL_BYTES = (
    b'{"alpha":"text","zeta":[3,1.5,null,true,{"a":-0.25,"b":"umlaut-u00fc"}]}'
)
CANONICAL_SHA256 = (
    "db3a7bb0051b6714e4e0d26468a41813e10d79856b950f37b5baa61183fe3d0d"
)

V4_OBSERVED_SHA256 = (
    "5f6cab3a3d19f4e40b7f06b882d9e60c578e6276fe4ed379f1ccb450492a34cf"
)
V4_CLASSES_SHA256 = (
    "ab852a5723dec820037b3b97f5d07724d33031e29667f67aeab3586bb3f223a3"
)

V5_MODEL_ARTIFACT_SHA256 = (
    "732824c34e15f1b9a21e4b8e6782056bc23443f7624e04c7cd4c65f4d62ede1b"
)

CAL4_SIGMOID_ARTIFACT = {
    "artifact_sha256": (
        "4f71b828519f85ee1e5d8b751fb6a69638ae5339dd6b587f4e463f3936edae2b"
    ),
    "fit_config": {
        "max_iter": 2000,
        "random_state": 0,
        "regularization_c": 1.0,
        "solver": "lbfgs",
    },
    "input_domain": "finite_real",
    "method": "regularized_sigmoid",
    "parameters": {
        "coefficient": 1.353173487436869,
        "intercept": 0.00567851619467725,
        "score_center": 0.5,
        "score_scale": 0.2680951323690902,
    },
    "schema_version": "chemapp.nmr.correctness-calibrator.v2",
    "score_direction": "higher_is_more_confident",
    "target": "top1_exact_structure_correct",
}

_LBFGS_MATRIX = [
    [0.5, -1.2, 0.3],
    [1.5, 0.4, -0.7],
    [-0.8, 1.1, 0.9],
    [0.2, 0.3, -1.5],
    [1.1, -0.6, 0.8],
    [-1.3, 0.7, 0.2],
]
_LBFGS_LABELS = [0, 1, 0, 1, 1, 0]
_LBFGS_WEIGHTS = [1.0] * 6
LBFGS_INTERCEPT = float.fromhex("-0x1.12931f8624b47p-1")
LBFGS_COEFFICIENTS = [
    float.fromhex(value)
    for value in (
        "0x1.6b42bc17646e6p+1",
        "0x1.31ce41305dc91p+0",
        "-0x1.1fe5542877caep-0",
    )
]

_NEWTON_X = [
    [0.2, 0.9],
    [1.4, 0.1],
    [0.7, 1.3],
    [1.9, 0.4],
    [0.3, 0.8],
    [1.6, 1.1],
]
_NEWTON_Y = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
_NEWTON_WEIGHTS = [1.0] * 6
NEWTON_INTERCEPT = float.fromhex("-0x1.d6ccc5a07968fp+2")
NEWTON_COEFFICIENTS = [
    float.fromhex(value)
    for value in ("0x1.d8d7f59bbd5c8p+3", "-0x1.ff00d1620aea6p+2")
]
NEWTON_CONVERGED = False

CALIBRATOR_FEATURES = [
    float.fromhex(value)
    for value in (
        "-0x1.aa6c0b69df090p-2",
        "0x1.684d1ddaf7669p-1",
        "0x1.588c2d9133490p-2",
        "0x1.48a11293d785cp-1",
        "0x1.999999999999ap-1",
    )
]


def test_canonical_bytes_and_hash_match_golden():
    assert canonical_json_bytes(CANONICAL_FIXTURE) == CANONICAL_BYTES
    assert canonical_sha256(CANONICAL_FIXTURE) == CANONICAL_SHA256


def test_all_versioned_hashers_agree_with_golden():
    assert scorer_v5.canonical_json_bytes(CANONICAL_FIXTURE) == CANONICAL_BYTES
    assert scorer_v5.canonical_sha256(CANONICAL_FIXTURE) == CANONICAL_SHA256
    assert scorer_v8.canonical_json_bytes(CANONICAL_FIXTURE) == CANONICAL_BYTES
    assert scorer_v8.canonical_sha256(CANONICAL_FIXTURE) == CANONICAL_SHA256
    assert scorer_v9.canonical_sha256(CANONICAL_FIXTURE) == CANONICAL_SHA256
    assert calibration_v4.canonical_sha256(CANONICAL_FIXTURE) == CANONICAL_SHA256
    assert calibration_v4.canonical_json_dumps(CANONICAL_FIXTURE).encode(
        "utf-8"
    ) == CANONICAL_BYTES
    assert calibration_v5.canonical_sha256(CANONICAL_FIXTURE) == CANONICAL_SHA256
    assert calibration_v5.canonical_json_dumps(CANONICAL_FIXTURE).encode(
        "utf-8"
    ) == CANONICAL_BYTES


def test_v4_ranking_hashes_match_golden():
    ranking = scorer_v4.rank_candidate_evidence(
        [25.5, 10.0],
        [
            {
                "candidate_id": "c1",
                "smiles": "CC",
                "atom_predictions": [
                    {"atom_index": 0, "shift_ppm": 12.0},
                    {"atom_index": 1, "shift_ppm": 24.0},
                ],
            }
        ],
    )
    assert ranking["observed_13c_sha256"] == V4_OBSERVED_SHA256
    assert (
        ranking["candidates"][0]["v4"]["symmetry_classes_sha256"]
        == V4_CLASSES_SHA256
    )


def test_v5_model_artifact_hash_matches_golden():
    feature_count = len(scorer_v5.FEATURE_NAMES)
    model_core = {
        "schema_version": scorer_v5.MODEL_SCHEMA_VERSION,
        "protocol_version": scorer_v5.PROTOCOL_VERSION,
        "feature_schema_version": scorer_v5.FEATURE_SCHEMA_VERSION,
        "feature_names": list(scorer_v5.FEATURE_NAMES),
        "model_kind": "symmetric_pairwise_l2_logistic_v1",
        "fit_intercept": False,
        "intercept": 0.0,
        "normalization": {
            "means": [0.1] * feature_count,
            "scales": [1.0] * feature_count,
        },
        "coefficients": [0.01 * index for index in range(feature_count)],
        "calibrated_probability": False,
        "conditional_probability": False,
        "development_binding": {},
        "runtime": {},
    }
    model = scorer_v5.finalize_model_artifact(model_core)
    assert model["artifact_sha256"] == V5_MODEL_ARTIFACT_SHA256


def test_calibration_v4_sigmoid_artifact_matches_golden():
    artifact = calibration_v4.fit_calibrator(
        [0.1, 0.4, 0.35, 0.8, 0.9, 0.55, 0.2, 0.7],
        [0, 0, 0, 1, 1, 1, 0, 1],
        method="regularized_sigmoid",
        regularization_c=1.0,
    )
    assert artifact == CAL4_SIGMOID_ARTIFACT


def test_artifact_with_hash_is_single_source():
    core = {"b": 2, "a": 1}
    assert artifact_with_hash(core) == {
        "a": 1,
        "b": 2,
        "artifact_sha256": canonical_sha256(core),
    }
    assert "artifact_sha256" not in core


def test_ridge_lbfgs_golden():
    import numpy as np

    matrix = np.asarray(_LBFGS_MATRIX, dtype=float)
    labels = np.asarray(_LBFGS_LABELS, dtype=int)
    weights = np.asarray(_LBFGS_WEIGHTS, dtype=float)
    intercept, coefficients = fit_weighted_ridge_logistic_lbfgs(
        matrix, labels, weights, 0.1
    )
    # L-BFGS 结果在不同平台/BLAS 间可能有末位差异（1 ulp 量级），
    # 用极严容差而非逐位相等来锁定 golden 值。
    assert intercept == pytest.approx(LBFGS_INTERCEPT, rel=0, abs=1e-12)
    assert coefficients.tolist() == pytest.approx(LBFGS_COEFFICIENTS, rel=0, abs=1e-12)
    # The frozen v5 wrapper maps core failures onto its own error types.
    v5_intercept, v5_coefficients = calibration_v5._fit_ridge(
        matrix, labels, weights, 0.1
    )
    assert v5_intercept == pytest.approx(LBFGS_INTERCEPT, rel=0, abs=1e-12)
    assert v5_coefficients.tolist() == pytest.approx(LBFGS_COEFFICIENTS, rel=0, abs=1e-12)
    with pytest.raises(RidgeLambdaError):
        fit_weighted_ridge_logistic_lbfgs(matrix, labels, weights, -1.0)
    with pytest.raises(calibration_v5.NMRCalibrationV5Error) as excinfo:
        calibration_v5._fit_ridge(matrix, labels, weights, -1.0)
    assert str(excinfo.value) == "regularization lambda must be positive"


def test_ridge_newton_golden():
    import numpy as np

    x = np.asarray(_NEWTON_X, dtype=float)
    y = np.asarray(_NEWTON_Y, dtype=float)
    weights = np.asarray(_NEWTON_WEIGHTS, dtype=float)
    intercept, coefficients, converged = fit_weighted_ridge_logistic_newton(
        x, y, weights, 0.01
    )
    assert intercept == NEWTON_INTERCEPT
    assert coefficients.tolist() == NEWTON_COEFFICIENTS
    assert converged is NEWTON_CONVERGED
    # The hybrid module keeps its frozen public entry point.
    hybrid_intercept, hybrid_coefficients, hybrid_converged = (
        hybrid_v1.fit_ridge_logistic(x, y, weights, 0.01)
    )
    assert hybrid_intercept == NEWTON_INTERCEPT
    assert hybrid_coefficients.tolist() == NEWTON_COEFFICIENTS
    assert hybrid_converged is NEWTON_CONVERGED


def test_calibrator_build_features_golden():
    candidates = [
        {
            "forward_evidence": {
                "mae_ppm": 0.4,
                "max_abs_error_ppm": 0.9,
                "bidirectional_coverage": 0.8,
            }
        },
        {
            "forward_evidence": {
                "mae_ppm": 1.2,
                "max_abs_error_ppm": 2.5,
                "bidirectional_coverage": 0.7,
            }
        },
        {
            "forward_evidence": {
                "mae_ppm": 3.1,
                "max_abs_error_ppm": 4.0,
                "bidirectional_coverage": 0.5,
            }
        },
    ]
    features = build_features(candidates)
    assert features is not None
    assert features == CALIBRATOR_FEATURES


def test_strict_key_message_wording_is_frozen():
    with pytest.raises(scorer_v4.NMRCandidateScorerV4InputError) as v4_exc:
        scorer_v4._strict_keys({"a": 1}, frozenset({"b"}), context="ctx")
    assert str(v4_exc.value) == (
        "ctx fields do not match the strict allowlist; "
        "missing=['b'], unexpected=['a']"
    )
    with pytest.raises(scorer_v5.NMRCandidateScorerV5InputError) as v5_exc:
        scorer_v5._strict_keys({"a": 1}, frozenset({"b"}), context="ctx")
    assert str(v5_exc.value) == (
        "ctx fields do not match the strict allowlist; "
        "missing=['b'], unexpected=['a']"
    )
    for module, error_type in (
        (scorer_v8, scorer_v8.NMRCandidateScorerV8InputError),
        (scorer_v9, scorer_v9.NMRCandidateScorerV9InputError),
    ):
        with pytest.raises(error_type) as excinfo:
            module._strict_keys({"a": 1}, frozenset({"b"}), context="ctx")
        assert str(excinfo.value) == (
            "ctx fields mismatch; missing=['b'], unexpected=['a']"
        )


def test_numeric_guard_message_wording_is_frozen():
    with pytest.raises(scorer_v5.NMRCandidateScorerV5InputError) as excinfo:
        scorer_v5._finite_float("nan-value", context="ctx")
    assert str(excinfo.value) == "ctx must be numeric"
    with pytest.raises(scorer_v9.NMRCandidateScorerV9InputError) as excinfo:
        scorer_v9._finite(float("nan"), context="ctx")
    assert str(excinfo.value) == "ctx must be finite"
    with pytest.raises(scorer_v8.NMRCandidateScorerV8InputError) as excinfo:
        scorer_v8._finite(1.0, context="ctx", minimum=2.0)
    assert str(excinfo.value) == "ctx must be at least 2.0"
    with pytest.raises(scorer_v5.NMRCandidateScorerV5InputError) as excinfo:
        scorer_v5._sha256("zz", context="ctx")
    assert str(excinfo.value) == "ctx must be a lowercase SHA-256"
    with pytest.raises(scorer_v9.NMRCandidateScorerV9InputError) as excinfo:
        scorer_v9._sha256("zz", context="ctx")
    assert str(excinfo.value) == "ctx must be SHA-256"


def test_v4_shift_guard_keeps_its_own_frozen_wording():
    with pytest.raises(scorer_v4.NMRCandidateScorerV4InputError) as excinfo:
        scorer_v4._finite_shift("nan-value", context="ctx")
    assert str(excinfo.value) == "ctx must be a finite number"
    with pytest.raises(scorer_v4.NMRCandidateScorerV4InputError) as excinfo:
        scorer_v4._finite_shift(400.0, context="ctx")
    assert str(excinfo.value) == "ctx must be within the supported 13C range"


def test_canonical_error_wrapping_differs_frozenly_per_version():
    unserializable = {1, 2, 3}
    with pytest.raises(TypeError):
        scorer_v4._canonical_sha256(unserializable)
    with pytest.raises(scorer_v5.NMRCandidateScorerV5InputError) as excinfo:
        scorer_v5.canonical_json_bytes(unserializable)
    assert str(excinfo.value) == "value is not finite canonical JSON"
    with pytest.raises(scorer_v8.NMRCandidateScorerV8InputError) as excinfo:
        scorer_v8.canonical_json_bytes(unserializable)
    assert str(excinfo.value) == "value is not finite canonical JSON"
