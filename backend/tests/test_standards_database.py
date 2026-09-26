"""Tests for the one-to-one consumed standards peak matching."""

from __future__ import annotations

from app.standards.database import StandardDatabase, _match_values


def test_match_values_consumes_each_observation_once() -> None:
    ref = [1.0, 1.05]
    obs = [1.0, 1.05, 2.0]
    errors = _match_values(ref, obs, tolerance=0.08)
    assert errors == [0.0, 0.0]


def test_match_values_does_not_reuse_a_single_observation() -> None:
    ref = [1.0, 1.1]
    obs = [1.05]
    errors = _match_values(ref, obs, tolerance=0.1)
    assert len(errors) == 1


def test_match_values_out_of_tolerance_is_unmatched() -> None:
    assert _match_values([1.0], [2.0], tolerance=0.08) == []


def test_match_peaks_uses_consumed_matching(tmp_path) -> None:
    db = StandardDatabase(str(tmp_path / "standards.db"))
    db.add_record(
        {
            "id": "nmr-test",
            "name": "Test",
            "technique": "NMR",
            "peaks": [{"shift": 1.0}, {"shift": 1.1}],
        }
    )
    matches = db.match_peaks(
        "NMR",
        [{"shift": 1.05}, {"shift": 3.0}],
        tolerance=0.1,
    )
    assert len(matches) == 1
    assert matches[0]["matched_peaks"] == 1
    assert matches[0]["reference_peaks"] == 2
