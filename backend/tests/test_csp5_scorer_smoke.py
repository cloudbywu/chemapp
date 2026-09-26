"""Smoke tests for the vendored CSP5 forward scorer.

These tests exercise the small bundled quantile model on a few SMILES and are
skipped automatically when the vendored ``csp5`` package is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from app.ml.forward_v1.factory import create_forward_scorer


pytest.importorskip("csp5", reason="vendored csp5 package is unavailable")

from app.ml.forward_v1.csp5_scorer import (  # noqa: E402
    Csp5ForwardScorer,
    Csp5PredictionIntegrityError,
    PROVIDER_ID,
    _mae_sort_key,
)


_MODELS_DIR = (
    Path(__file__).resolve().parents[1]
    / "vendor"
    / "csp5"
    / "models"
    / "CSP5q-13C"
)


@pytest.fixture(scope="module", autouse=True)
def _require_bundled_weights() -> None:
    if not (_MODELS_DIR / "best_model.pt").exists():
        pytest.skip(
            "bundled CSP5 model weights are not present; restore them from "
            "the csp5 source distribution or Zenodo record 19486118"
        )


@pytest.fixture(scope="module")
def scorer() -> Csp5ForwardScorer:
    return Csp5ForwardScorer(device="cpu")


def test_ethanol_atom_predictions_plausible(scorer: Csp5ForwardScorer) -> None:
    predictions = scorer.predict_molecule("CCO")
    assert len(predictions) == 2
    by_index = {int(item["atom_index"]): item for item in predictions}
    assert set(by_index) == {1, 2}
    methyl = by_index[1]
    methylene = by_index[2]
    assert 10.0 <= methyl["shift_ppm"] <= 25.0
    assert 50.0 <= methylene["shift_ppm"] <= 65.0
    for item in predictions:
        assert item["q10_ppm"] <= item["shift_ppm"] <= item["q90_ppm"]
        assert item["shift_std_ppm"] > 0.0


@pytest.mark.parametrize(
    ("rows", "reason"),
    [
        ([], "missing_target_predictions"),
        (
            [{"molecule_id": 0, "atom_index": 0, "shift_ppm": 18.3}],
            "incomplete_target_predictions",
        ),
        (
            [
                {"molecule_id": 0, "atom_index": 0, "shift_ppm": 18.3},
                {"molecule_id": 0, "atom_index": 0, "shift_ppm": 58.1},
            ],
            "invalid_target_predictions",
        ),
        (
            [
                {"molecule_id": 0, "atom_index": 0, "shift_ppm": float("nan")},
                {"molecule_id": 0, "atom_index": 1, "shift_ppm": 58.1},
            ],
            "invalid_target_predictions",
        ),
    ],
)
def test_predict_molecule_rejects_invalid_carbon_prediction_sets(
    scorer: Csp5ForwardScorer,
    monkeypatch: pytest.MonkeyPatch,
    rows: list[dict[str, float | int]],
    reason: str,
) -> None:
    import app.ml.forward_v1.csp5_scorer as scorer_module

    class _FakeResult:
        predictions = pd.DataFrame(rows)
        failures: list[str] = []

    monkeypatch.setattr(
        scorer_module,
        "predict_mols",
        lambda *_args, **_kwargs: _FakeResult(),
    )

    with pytest.raises(Csp5PredictionIntegrityError, match=reason):
        scorer.predict_molecule("CCO")


def test_score_candidates_ranks_ethanol_first(scorer: Csp5ForwardScorer) -> None:
    observed = [{"shift": 18.3}, {"shift": 58.1}]
    candidates = [
        {"candidate_id": "wrong", "smiles": "c1ccccc1"},
        {"candidate_id": "acid", "smiles": "CC(=O)O"},
        {"candidate_id": "right", "smiles": "CCO"},
    ]
    result = scorer.score_candidates(observed, candidates)
    assert result["status"] == "ok"
    assert result["quantile_enabled"] is True
    assert result["calibrated_probability"] is False
    assert result["model"]["provider"] == PROVIDER_ID
    ranked = [item["candidate_id"] for item in result["candidates"]]
    assert ranked[0] == "right"
    top = result["candidates"][0]
    assert top["mae_ppm"] < 2.0
    assert top["bidirectional_coverage"] == 1.0
    assert len(top["prediction"]["atom_predictions"]) == 2


def test_score_candidates_requires_observed_13c(scorer: Csp5ForwardScorer) -> None:
    result = scorer.score_candidates([], [{"candidate_id": "x", "smiles": "CCO"}])
    assert result["status"] == "unsupported_modality"
    assert result["candidates"] == []


def test_score_candidates_keeps_failed_candidates_at_end(
    scorer: Csp5ForwardScorer,
) -> None:
    observed = [{"shift": 18.3}, {"shift": 58.1}]
    candidates = [
        {"candidate_id": "ok", "smiles": "CCO"},
        {"candidate_id": "bad", "smiles": "not-a-smiles"},
    ]
    result = scorer.score_candidates(observed, candidates)
    assert result["status"] == "partial_failure"
    assert result["failed_candidate_count"] == 1
    ranked = result["candidates"]
    assert len(ranked) == len(candidates)
    assert ranked[-1]["candidate_id"] == "bad"
    assert ranked[-1]["prediction_failed"] is True
    assert ranked[-1]["assignment_mode"] == "none_prediction_failed"
    assert ranked[-1]["mae_ppm"] is None
    assert ranked[-1]["prediction"]["status"] == "prediction_failed"
    assert ranked[0]["candidate_id"] == "ok"


def test_score_candidates_preserves_duplicate_observed_shifts(
    scorer: Csp5ForwardScorer,
) -> None:
    observed = [{"shift": 18.3}, {"shift": 18.3}, {"shift": 58.1}]
    result = scorer.score_candidates(
        observed,
        [{"candidate_id": "right", "smiles": "CCO"}],
    )
    assert result["status"] == "ok"
    assert result["observed_shifts_ppm"] == [18.3, 18.3, 58.1]
    top = result["candidates"][0]
    assert top["observed_count"] == 3
    assert top["predicted_atom_count"] == 2
    assert top["matched_count"] == 2
    assert top["unmatched_observed_count"] == 1


def test_score_candidates_batch_failure_marks_all_failed(
    scorer: Csp5ForwardScorer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.ml.forward_v1.csp5_scorer as scorer_module

    def boom(*_: object, **__: object) -> object:
        raise RuntimeError("simulated upstream batch failure")

    monkeypatch.setattr(scorer_module, "predict_mols", boom)
    result = scorer.score_candidates(
        [18.3, 58.1],
        [
            {"candidate_id": "a", "smiles": "CCO"},
            {"candidate_id": "b", "smiles": "CCN"},
        ],
    )
    assert result["status"] == "prediction_failed"
    assert result["failed_candidate_count"] == 2
    ranked = result["candidates"]
    assert len(ranked) == 2
    assert [item["candidate_id"] for item in ranked] == ["a", "b"]
    assert all(item["prediction_failed"] for item in ranked)
    assert all(item["mae_ppm"] is None for item in ranked)
    assert all(
        item["prediction"]["reason"] == "batch_prediction_failed"
        for item in ranked
    )


def test_score_candidates_empty_predictions_fail_closed(
    scorer: Csp5ForwardScorer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.ml.forward_v1.csp5_scorer as scorer_module

    class _FakeResult:
        predictions = pd.DataFrame(
            columns=["molecule_id", "atom_index", "shift_ppm"]
        )
        failures: list[str] = []

    monkeypatch.setattr(
        scorer_module,
        "predict_mols",
        lambda *_args, **_kwargs: _FakeResult(),
    )
    result = scorer.score_candidates(
        [18.3, 58.1],
        [{"candidate_id": "ethanol", "smiles": "CCO"}],
    )

    assert result["status"] == "prediction_failed"
    assert result["failed_candidate_count"] == 1
    failed = result["candidates"][0]
    assert failed["prediction_failed"] is True
    assert failed["mae_ppm"] is None
    assert failed["prediction"]["reason"] == "missing_target_predictions"


def test_score_candidates_missing_carbon_prediction_fails_closed(
    scorer: Csp5ForwardScorer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.ml.forward_v1.csp5_scorer as scorer_module

    class _FakeResult:
        predictions = pd.DataFrame(
            [{"molecule_id": 0, "atom_index": 0, "shift_ppm": 18.3}]
        )
        failures: list[str] = []

    monkeypatch.setattr(
        scorer_module,
        "predict_mols",
        lambda *_args, **_kwargs: _FakeResult(),
    )
    result = scorer.score_candidates(
        [18.3, 58.1],
        [{"candidate_id": "ethanol", "smiles": "CCO"}],
    )

    assert result["status"] == "prediction_failed"
    failed = result["candidates"][0]
    assert failed["prediction"]["reason"] == "incomplete_target_predictions"


def test_score_candidates_partial_failure_marks_mapped_candidate(
    scorer: Csp5ForwardScorer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.ml.forward_v1.csp5_scorer as scorer_module

    class _FakeResult:
        def __init__(self) -> None:
            self.predictions = pd.DataFrame(
                [
                    {
                        "molecule_id": 0,
                        "atom_index": 0,
                        "shift_ppm": 18.3,
                        "shift_q10_ppm": 12.0,
                        "shift_q90_ppm": 24.0,
                        "shift_std_ppm": 3.0,
                    },
                    {
                        "molecule_id": 0,
                        "atom_index": 1,
                        "shift_ppm": 58.1,
                        "shift_q10_ppm": 50.0,
                        "shift_q90_ppm": 65.0,
                        "shift_std_ppm": 4.0,
                    },
                ]
            )
            self.failures = ["no_target\tCCN"]

    def fake_predict_mols(*_: object, **__: object) -> _FakeResult:
        return _FakeResult()

    monkeypatch.setattr(scorer_module, "predict_mols", fake_predict_mols)
    result = scorer.score_candidates(
        [18.3, 58.1],
        [
            {"candidate_id": "a", "smiles": "CCO"},
            {"candidate_id": "b", "smiles": "CCN"},
        ],
    )
    assert result["status"] == "partial_failure"
    assert result["failed_candidate_count"] == 1
    ranked = result["candidates"]
    assert [item["candidate_id"] for item in ranked] == ["a", "b"]
    assert ranked[0].get("prediction_failed", False) is False
    assert ranked[1]["prediction_failed"] is True
    assert ranked[1]["prediction"]["reason"] == "batch_prediction_failed"
    assert ranked[1]["prediction"]["detail"] == "no_target\tCCN"


def test_score_candidates_unmappable_failure_marks_all_failed(
    scorer: Csp5ForwardScorer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.ml.forward_v1.csp5_scorer as scorer_module

    class _FakeResult:
        def __init__(self) -> None:
            self.predictions = pd.DataFrame(
                [
                    {
                        "molecule_id": 0,
                        "atom_index": 0,
                        "shift_ppm": 18.3,
                        "shift_q10_ppm": 12.0,
                        "shift_q90_ppm": 24.0,
                        "shift_std_ppm": 3.0,
                    }
                ]
            )
            self.failures = ["geometry failure without a SMILES suffix"]

    monkeypatch.setattr(
        scorer_module,
        "predict_mols",
        lambda *_: _FakeResult(),
    )
    result = scorer.score_candidates(
        [18.3, 58.1],
        [{"candidate_id": "a", "smiles": "CCO"}],
    )
    assert result["status"] == "prediction_failed"
    assert result["failed_candidate_count"] == 1
    ranked = result["candidates"]
    assert len(ranked) == 1
    assert ranked[0]["prediction_failed"] is True
    assert ranked[0]["prediction"]["reason"] == "batch_prediction_failed"


def test_mae_sort_key_keeps_zero_before_none() -> None:
    assert _mae_sort_key({"mae_ppm": 0.0}) == 0.0
    assert _mae_sort_key({"mae_ppm": 0.1}) == 0.1
    assert _mae_sort_key({"mae_ppm": None}) == float("inf")


def test_factory_prefers_csp5() -> None:
    scorer = create_forward_scorer(prefer="csp5")
    assert isinstance(scorer, Csp5ForwardScorer)


def test_hybrid_predictor_end_to_end_with_csp5(scorer: Csp5ForwardScorer) -> None:
    from typing import Any

    from app.ml.nmr_hybrid_predictor import HybridPredictorV1

    def no_references(**_: Any) -> dict[str, Any]:
        return {
            "candidates": [],
            "candidate_pool_status": "no_formula_match",
            "warnings": [],
        }

    predictor = HybridPredictorV1(
        reference_ranker=no_references,
        forward_scorer=scorer,
    )
    result = predictor.predict(
        peaks_13c=[{"shift": 18.3}, {"shift": 58.1}],
        formula="C2H6O",
        candidate_smiles=["CC(=O)O", "CCO", "c1ccccc1"],
    )

    assert result["status"] == "completed"
    assert result["candidates"][0]["smiles"] == "CCO"
    forward_stage = next(
        stage for stage in result["stages"] if stage["stage"] == "forward_scoring"
    )
    assert forward_stage["status"] == "completed"
    assert forward_stage["used_for_ranking"] is True
    assert forward_stage["model"]["provider"] == PROVIDER_ID
    assert result["uncertainty_kind"] == "predictive_interval_uncalibrated"
