from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.ml.nmr_calibration_v4 import (
    CALIBRATOR_SCHEMA_VERSION,
    TEST_CONSUMPTION_SCHEMA_VERSION,
    calibration_metrics,
    canonical_sha256,
    fit_calibrator,
    predict_calibrated,
    reserve_test_cohort_consumption,
    risk_coverage_metrics,
    select_calibration_method_dev,
    test_cohort_sha256,
)


def _digest(value: str) -> str:
    return canonical_sha256(value)


def _clock(hour: int):
    return lambda: datetime(2026, 7, 29, hour, 0, tzinfo=timezone.utc)


def _cohort_rows() -> list[dict]:
    return [
        {
            "record_id": "spectrum-b",
            "spectrum_fingerprint_sha256": _digest("fingerprint-b"),
            "split_group": "source-group-2",
            "model_name": "ignored-model-field",
        },
        {
            "record_id": "spectrum-a",
            "spectrum_fingerprint_sha256": _digest("fingerprint-a"),
            "split_group": "source-group-1",
            "prediction": "ignored-prediction-field",
        },
    ]


def _dev_rows(groups: int = 8) -> list[dict]:
    rows: list[dict] = []
    for index in range(groups):
        rows.extend(
            [
                {
                    "record_id": f"group-{index}-negative",
                    "split": "dev",
                    "group_id": f"group-{index}",
                    "raw_score": 0.08 + index * 0.002,
                    "outcome": 0,
                },
                {
                    "record_id": f"group-{index}-positive",
                    "split": "dev",
                    "group_id": f"group-{index}",
                    "raw_score": 0.92 - index * 0.002,
                    "outcome": 1,
                },
            ]
        )
    return rows


def test_test_cohort_hash_is_order_and_model_independent() -> None:
    rows = _cohort_rows()
    expected = test_cohort_sha256("release-v4", rows)
    assert test_cohort_sha256("release-v4", list(reversed(rows))) == expected

    changed_model_fields = [dict(row) for row in rows]
    changed_model_fields[0]["model_name"] = "another-ranker"
    changed_model_fields[1]["prediction"] = "another-prediction"
    assert test_cohort_sha256("release-v4", changed_model_fields) == expected

    changed_group = [dict(row) for row in rows]
    changed_group[0]["split_group"] = "new-group"
    changed_group[0]["record_id"] = "renamed-record"
    assert test_cohort_sha256("release-v4", changed_group) == expected
    assert test_cohort_sha256("release-v5", rows) == expected


def test_test_cohort_hash_rejects_duplicates_and_bad_fingerprints() -> None:
    rows = _cohort_rows()
    duplicate = [rows[0], dict(rows[0], record_id="renamed")]
    with pytest.raises(ValueError, match="duplicate"):
        test_cohort_sha256("release-v4", duplicate)

    invalid = [dict(rows[0], spectrum_fingerprint_sha256="not-a-hash")]
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        test_cohort_sha256("release-v4", invalid)


def test_v2_ledger_rejects_same_cohort_across_changed_runs(tmp_path: Path) -> None:
    ledger = tmp_path / "external-test-consumption.jsonl"
    cohort_hash = test_cohort_sha256("release-v4", _cohort_rows())
    first = reserve_test_cohort_consumption(
        ledger,
        run_id="run-one",
        source_release_id=_digest("release-v4"),
        run_spec_file_sha256=_digest("run-spec-one"),
        test_cohort_sha256=cohort_hash,
        _clock=_clock(1),
    )
    assert first["schema_version"] == TEST_CONSUMPTION_SCHEMA_VERSION

    with pytest.raises(ValueError, match="already reserved/consumed"):
        reserve_test_cohort_consumption(
            ledger,
            run_id="run-two-with-a-different-model",
            source_release_id=_digest("release-v5"),
            run_spec_file_sha256=_digest("run-spec-two"),
            test_cohort_sha256=cohort_hash,
            _clock=_clock(2),
        )
    assert len(ledger.read_text(encoding="utf-8").splitlines()) == 1


def test_v2_ledger_strictly_rejects_unknown_existing_fields(tmp_path: Path) -> None:
    ledger = tmp_path / "external-test-consumption.jsonl"
    cohort_hash = test_cohort_sha256("release-v4", _cohort_rows())
    reserve_test_cohort_consumption(
        ledger,
        run_id="run-one",
        source_release_id=_digest("release-v4"),
        run_spec_file_sha256=_digest("run-spec-one"),
        test_cohort_sha256=cohort_hash,
        _clock=_clock(1),
    )
    existing = json.loads(ledger.read_text(encoding="utf-8"))
    existing["ranker_name"] = "covert-ledger-field"
    ledger.write_text(json.dumps(existing) + "\n", encoding="utf-8")

    other_rows = [dict(row) for row in _cohort_rows()]
    other_rows[0]["record_id"] = "different-record"
    with pytest.raises(ValueError, match="field allowlist"):
        reserve_test_cohort_consumption(
            ledger,
            run_id="run-two",
            source_release_id=_digest("release-v5"),
            run_spec_file_sha256=_digest("run-spec-two"),
            test_cohort_sha256=test_cohort_sha256("release-v4", other_rows),
            _clock=_clock(2),
        )


def test_v2_ledger_rejects_same_release_with_changed_cohort(tmp_path: Path) -> None:
    ledger = tmp_path / "external-test-consumption.jsonl"
    first_rows = _cohort_rows()
    reserve_test_cohort_consumption(
        ledger,
        run_id="run-one",
        source_release_id=_digest("release-v4"),
        run_spec_file_sha256=_digest("run-spec-one"),
        test_cohort_sha256=test_cohort_sha256("release-v4", first_rows),
        _clock=_clock(1),
    )
    changed_rows = [
        {
            **row,
            "spectrum_fingerprint_sha256": _digest(
                f"changed-{index}"
            ),
        }
        for index, row in enumerate(first_rows)
    ]
    with pytest.raises(ValueError, match="source release was already"):
        reserve_test_cohort_consumption(
            ledger,
            run_id="run-two",
            source_release_id=_digest("release-v4"),
            run_spec_file_sha256=_digest("run-spec-two"),
            test_cohort_sha256=test_cohort_sha256(
                "renamed-release",
                changed_rows,
            ),
            _clock=_clock(2),
        )


def test_v2_ledger_allows_different_release_and_cohort(tmp_path: Path) -> None:
    ledger = tmp_path / "external-test-consumption.jsonl"
    reserve_test_cohort_consumption(
        ledger,
        run_id="run-one",
        source_release_id=_digest("release-v4"),
        run_spec_file_sha256=_digest("run-spec-one"),
        test_cohort_sha256=test_cohort_sha256("release-v4", _cohort_rows()),
        _clock=_clock(1),
    )
    changed_rows = [
        {
            **row,
            "spectrum_fingerprint_sha256": _digest(
                f"different-{index}"
            ),
        }
        for index, row in enumerate(_cohort_rows())
    ]
    second = reserve_test_cohort_consumption(
        ledger,
        run_id="run-two",
        source_release_id=_digest("release-v5"),
        run_spec_file_sha256=_digest("run-spec-two"),
        test_cohort_sha256=test_cohort_sha256("release-v5", changed_rows),
        _clock=_clock(2),
    )
    assert second["source_release_id"] == _digest("release-v5")


@pytest.mark.parametrize("method", ["regularized_sigmoid", "isotonic", "beta"])
def test_v4_calibrators_round_trip_monotonically(method: str) -> None:
    scores = [0.02, 0.08, 0.15, 0.25, 0.72, 0.82, 0.91, 0.98]
    outcomes = [0, 0, 0, 0, 1, 1, 1, 1]
    calibrator = fit_calibrator(scores, outcomes, method=method)
    assert calibrator["schema_version"] == CALIBRATOR_SCHEMA_VERSION
    assert calibrator["artifact_sha256"] == canonical_sha256(
        {key: value for key, value in calibrator.items() if key != "artifact_sha256"}
    )

    probabilities = predict_calibrated(calibrator, scores)
    assert probabilities == sorted(probabilities)
    assert all(0.0 <= value <= 1.0 for value in probabilities)


def test_beta_requires_unit_interval_and_fails_closed_if_nonmonotonic() -> None:
    with pytest.raises(ValueError, match="unit-interval"):
        fit_calibrator(
            [-0.1, 0.2, 0.8, 1.1],
            [0, 0, 1, 1],
            method="beta",
        )

    with pytest.raises(ValueError, match="not monotonic"):
        fit_calibrator(
            [0.05, 0.15, 0.85, 0.95],
            [1, 1, 0, 0],
            method="beta",
        )


def test_calibrator_artifacts_reject_unknown_fields_and_hash_changes() -> None:
    calibrator = fit_calibrator(
        [0.1, 0.2, 0.8, 0.9],
        [0, 0, 1, 1],
        method="regularized_sigmoid",
    )
    with_extra = {**calibrator, "covert": True}
    with pytest.raises(ValueError, match="field allowlist"):
        predict_calibrated(with_extra, [0.5])

    forged = {
        **calibrator,
        "parameters": {**calibrator["parameters"], "intercept": 100.0},
    }
    with pytest.raises(ValueError, match="hash mismatch"):
        predict_calibrated(forged, [0.5])

    changed_config_core = {
        **{key: value for key, value in calibrator.items() if key != "artifact_sha256"},
        "fit_config": {**calibrator["fit_config"], "solver": "changed"},
    }
    changed_config = {
        **changed_config_core,
        "artifact_sha256": canonical_sha256(changed_config_core),
    }
    with pytest.raises(ValueError, match="fit configuration changed"):
        predict_calibrated(changed_config, [0.5])


def test_dev_only_group_cv_selection_is_deterministic() -> None:
    rows = _dev_rows()
    first = select_calibration_method_dev(
        rows,
        methods=("regularized_sigmoid", "isotonic", "beta"),
        n_splits=4,
        seed=42,
        bins=7,
    )
    second = select_calibration_method_dev(
        list(reversed(rows)),
        methods=("regularized_sigmoid", "isotonic", "beta"),
        n_splits=4,
        seed=42,
        bins=7,
    )
    assert first == second
    assert first["calibration_bins"] == 7
    assert first["artifact_sha256"] == canonical_sha256(
        {key: value for key, value in first.items() if key != "artifact_sha256"}
    )
    assert first["split_roles_consumed"] == ["dev"]
    assert first["calibration_or_test_rows_consumed"] is False
    assert first["selected_method"] in {
        candidate["method"]
        for candidate in first["candidates"]
        if candidate["status"] == "eligible"
    }
    assert all(
        len(candidate["metrics"]["reliability_bins"]) == 7
        for candidate in first["candidates"]
        if candidate["status"] == "eligible"
    )


@pytest.mark.parametrize("forbidden_split", ["calibration", "test"])
def test_method_selection_rejects_non_dev_rows(forbidden_split: str) -> None:
    rows = _dev_rows()
    rows[0] = {**rows[0], "split": forbidden_split}
    with pytest.raises(ValueError, match="dev rows only"):
        select_calibration_method_dev(rows, n_splits=4, bins=10)


@pytest.mark.parametrize("bins", [True, 1, 2.5])
def test_method_selection_strictly_validates_calibration_bins(
    bins: object,
) -> None:
    with pytest.raises(ValueError, match="bins must be an integer >= 2"):
        select_calibration_method_dev(
            _dev_rows(),
            n_splits=4,
            bins=bins,  # type: ignore[arg-type]
        )


def test_metrics_cover_calibration_discrimination_and_risk() -> None:
    metrics = calibration_metrics(
        [0.1, 0.2, 0.8, 0.9],
        [0, 0, 1, 1],
        bins=5,
    )
    assert metrics["brier"] == pytest.approx(0.025)
    assert metrics["log_loss"] > 0.0
    assert metrics["ece"] == pytest.approx(0.15)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["average_precision"] == pytest.approx(1.0)
    assert metrics["risk_coverage"]["aurc"] >= 0.0
    assert metrics["risk_coverage"]["curve"][-1] == {
        "accepted": 4,
        "coverage": 1.0,
        "risk": 0.5,
        "minimum_confidence": 0.1,
    }


def test_risk_coverage_ties_are_order_independent() -> None:
    probabilities = [0.9, 0.5, 0.5, 0.1]
    first = risk_coverage_metrics(probabilities, [1, 1, 0, 0])
    swapped_tie = risk_coverage_metrics(probabilities, [1, 0, 1, 0])

    assert first == swapped_tie
    assert first["aurc"] == pytest.approx(13 / 48)
    assert first["aurc_semantics"] == (
        "expected_rankwise_mean_cumulative_risk_under_uniform_random_"
        "ordering_within_confidence_ties"
    )
    assert first["curve_semantics"] == (
        "unique_confidence_thresholds_accept_all_rows_in_each_tie"
    )
    assert first["curve"] == [
        {
            "accepted": 1,
            "coverage": 0.25,
            "risk": 0.0,
            "minimum_confidence": 0.9,
        },
        {
            "accepted": 3,
            "coverage": 0.75,
            "risk": 1 / 3,
            "minimum_confidence": 0.5,
        },
        {
            "accepted": 4,
            "coverage": 1.0,
            "risk": 0.5,
            "minimum_confidence": 0.1,
        },
    ]
