"""Tests for the one-to-one consumed HPLC peak matching."""

from __future__ import annotations

from app.api.routes.analysis import _match_sample_peaks


def _peak(position: float, name: str = "") -> dict:
    return {"position": position, "area": 1.0, "name": name}


def test_each_sample_peak_is_consumed_at_most_once() -> None:
    reference = [_peak(1.0), _peak(1.05)]
    sample = [_peak(1.0), _peak(1.05), _peak(2.0)]
    matched = _match_sample_peaks(reference, sample, rt_tolerance=0.08)
    assert [item["position"] if item else None for item in matched] == [1.0, 1.05]
    # The 1.05 sample peak is not reused for the first reference peak.
    assert matched[0]["position"] == 1.0


def test_out_of_tolerance_is_unmatched() -> None:
    reference = [_peak(1.0)]
    sample = [_peak(2.0)]
    matched = _match_sample_peaks(reference, sample, rt_tolerance=0.08)
    assert matched == [None]


def test_empty_sample_returns_no_matches() -> None:
    reference = [_peak(1.0), _peak(2.0)]
    assert _match_sample_peaks(reference, [], rt_tolerance=0.08) == [None, None]


def test_closest_within_window_wins() -> None:
    reference = [_peak(1.0)]
    sample = [_peak(0.97), _peak(1.02)]
    matched = _match_sample_peaks(reference, sample, rt_tolerance=0.08)
    assert matched[0]["position"] == 1.02
