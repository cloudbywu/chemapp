from __future__ import annotations

import json

import numpy as np
import pytest

from app.ml import nmr_structure_elucidation as elucidation


def _record(
    *,
    peaks_13c: list[float] | None = None,
    peaks_1h: list[float] | list[dict[str, float]] | None = None,
) -> dict:
    return {
        "peaks_13c": peaks_13c or [],
        "peaks_1h": peaks_1h or [],
        "formula": "C2H6",
        "mw": 30.07,
    }


def _insert_record(
    conn,
    *,
    peaks_13c: list[float],
    peaks_1h: list[float],
    source_id: str = "candidate",
) -> None:
    conn.execute(
        """
        INSERT INTO nmr_records
        (source, source_id, name, smiles, formula, mw, peaks_13c, peaks_1h, metadata)
        VALUES ('test', ?, '', 'CC', 'C2H6', 30.07, ?, ?, '{}')
        """,
        (source_id, json.dumps(peaks_13c), json.dumps(peaks_1h)),
    )


def _configure_index(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHEMAPP_NMR_INDEX", str(tmp_path / "index.sqlite"))
    monkeypatch.setenv("CHEMAPP_NMR_RANKER", str(tmp_path / "missing.joblib"))
    elucidation._load_records_cached.cache_clear()


def test_zero_matches_have_zero_score_even_when_peak_counts_balance():
    result = elucidation._match_score(
        [1.0, 2.0],
        [5.0, 6.0],
        tolerance=0.18,
        nucleus_weight=1.0,
    )

    assert result["matched"] == 0
    assert result["score"] == 0.0
    assert result["mae"] is None
    assert result["count_balance"] == 0.0
    assert result["query_recall"] == 0.0
    assert result["reference_coverage"] == 0.0


def test_count_balance_only_contributes_after_a_valid_assignment():
    result = elucidation._match_score(
        [1.0, 8.0],
        [1.01],
        tolerance=0.18,
        nucleus_weight=1.0,
    )

    assert result["matched"] == 1
    assert result["count_balance"] == pytest.approx(0.5)
    assert result["score"] > 0.0


def test_mae_missing_indicator_distinguishes_no_match_from_exact_match():
    exact_features, exact = elucidation._feature_vector(
        [100.0],
        [],
        _record(peaks_13c=[100.0]),
    )
    missing_features, missing = elucidation._feature_vector(
        [100.0],
        [],
        _record(peaks_13c=[150.0]),
    )

    mae_index = elucidation._FEATURE_NAMES.index("c13_mae")
    missing_index = elucidation._FEATURE_NAMES.index("c13_mae_missing")
    assert exact_features[mae_index] == 0.0
    assert missing_features[mae_index] == 0.0
    assert exact_features[missing_index] == 0.0
    assert missing_features[missing_index] == 1.0
    assert exact["c13"]["mae"] == 0.0
    assert missing["c13"]["mae"] is None


def test_resonance_normalisation_is_shared_and_keeps_close_13c_signals():
    proton_groups = elucidation._normalise_resonance_shifts(
        [
            {"shift": 1.0, "intensity": 1.0},
            {"shift": 1.02, "intensity": 3.0},
            {"shift": 7.0, "intensity": 1.0},
        ],
        "1H",
    )
    carbon_groups = elucidation._normalise_resonance_shifts(
        [100.0, 100.00006, 100.006],
        "13C",
    )

    assert proton_groups == pytest.approx([1.015, 7.0])
    assert carbon_groups == pytest.approx([100.00003, 100.006])

    _, breakdown = elucidation._feature_vector(
        [],
        proton_groups,
        _record(
            peaks_1h=[
                {"shift": 1.0, "intensity": 1.0},
                {"shift": 1.02, "intensity": 3.0},
                {"shift": 7.0, "intensity": 1.0},
            ]
        ),
    )
    assert breakdown["h1"]["query_count"] == 2
    assert breakdown["h1"]["reference_count"] == 2
    assert breakdown["h1"]["matched"] == 2


def test_evidence_is_weak_when_errors_are_near_tolerance_despite_full_recall(
    tmp_path,
    monkeypatch,
):
    _configure_index(tmp_path, monkeypatch)
    conn = elucidation._connect()
    _insert_record(
        conn,
        peaks_13c=[100.0, 120.0],
        peaks_1h=[1.0, 2.0],
    )
    conn.commit()
    conn.close()

    # 13C offsets are kept at ~95% of the 13C match tolerance (2.85/3.0),
    # mirroring the original 2.1/2.2 construction; the 1H offsets stay at
    # ~94% of the 1H tolerance (0.17/0.18).
    result = elucidation.rank_candidates(
        peaks_13c=[{"shift": 102.85}, {"shift": 122.85}],
        peaks_1h=[{"shift": 1.17}, {"shift": 2.17}],
        formula="C2H6",
    )
    candidate = result["candidates"][0]

    assert candidate["evidence"]["c13_recall"] == 1.0
    assert candidate["evidence"]["h1_recall"] == 1.0
    assert candidate["evidence"]["c13_reference_coverage"] == 1.0
    assert candidate["evidence"]["h1_reference_coverage"] == 1.0
    assert candidate["evidence"]["c13_normalized_mae"] > 0.9
    assert candidate["evidence"]["h1_normalized_mae"] > 0.9
    assert candidate["evidence_level"] == "weak"


def test_evidence_requires_reference_coverage_and_both_nuclei_for_strong(
    tmp_path,
    monkeypatch,
):
    _configure_index(tmp_path, monkeypatch)
    conn = elucidation._connect()
    _insert_record(
        conn,
        peaks_13c=[],
        peaks_1h=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    conn.commit()
    conn.close()

    result = elucidation.rank_candidates(
        peaks_1h=[{"shift": 1.0}, {"shift": 2.0}],
        formula="C2H6",
    )
    candidate = result["candidates"][0]

    assert candidate["evidence"]["h1_recall"] == 1.0
    assert candidate["evidence"]["h1_reference_coverage"] == 0.4
    assert candidate["evidence_level"] == "weak"


def test_exact_dual_nucleus_match_can_be_strong(tmp_path, monkeypatch):
    _configure_index(tmp_path, monkeypatch)
    conn = elucidation._connect()
    _insert_record(
        conn,
        peaks_13c=[20.0, 30.0],
        peaks_1h=[1.0, 2.0],
    )
    conn.commit()
    conn.close()

    result = elucidation.rank_candidates(
        peaks_13c=[{"shift": 20.0}, {"shift": 30.0}],
        peaks_1h=[{"shift": 1.0}, {"shift": 2.0}],
        formula="C2H6",
    )

    assert result["candidates"][0]["evidence_level"] == "strong"


def test_ranker_with_old_feature_schema_is_disabled(tmp_path, monkeypatch):
    _configure_index(tmp_path, monkeypatch)
    conn = elucidation._connect()
    _insert_record(conn, peaks_13c=[], peaks_1h=[1.0, 2.0])
    conn.commit()
    conn.close()
    old_features = [
        name
        for name in elucidation._FEATURE_NAMES
        if not name.endswith("_mae_missing")
    ]
    monkeypatch.setattr(
        elucidation,
        "_load_ranker",
        lambda: {
            "model": object(),
            "meta": {
                "evaluation_protocol": "independent_spectrum_scaffold_split_v2",
                "production_eligible": True,
                "supports_modalities": ["1h"],
            },
            "features": old_features,
        },
    )

    result = elucidation.rank_candidates(
        peaks_1h=[{"shift": 1.0}, {"shift": 2.0}],
        formula="C2H6",
    )

    assert result["ranker"]["loaded"] is False
    assert result["ranker"]["reason"] == "feature_schema_mismatch"


def test_pairwise_self_retrieval_ranker_is_not_production_eligible(
    tmp_path,
    monkeypatch,
):
    _configure_index(tmp_path, monkeypatch)
    conn = elucidation._connect()
    _insert_record(conn, peaks_13c=[], peaks_1h=[1.0, 2.0])
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        elucidation,
        "_load_ranker",
        lambda: {
            "model": object(),
            "meta": {
                "evaluation_protocol": "bemis_murcko_group_split_v1",
                "supports_modalities": ["1h"],
            },
            "features": elucidation._FEATURE_NAMES,
        },
    )

    result = elucidation.rank_candidates(
        peaks_1h=[{"shift": 1.0}, {"shift": 2.0}],
        formula="C2H6",
    )

    assert result["ranker"]["loaded"] is False
    assert result["ranker"]["reason"] == "unvalidated_training_protocol_disabled"


def test_ranker_cannot_assign_positive_score_without_any_peak_match(
    tmp_path,
    monkeypatch,
):
    class AlwaysHighRanker:
        def predict_proba(self, features):
            return np.tile([0.01, 0.99], (len(features), 1))

    _configure_index(tmp_path, monkeypatch)
    conn = elucidation._connect()
    _insert_record(conn, peaks_13c=[], peaks_1h=[8.0, 9.0])
    conn.commit()
    conn.close()
    monkeypatch.setattr(
        elucidation,
        "_load_ranker",
        lambda: {
            "model": AlwaysHighRanker(),
            "meta": {
                "evaluation_protocol": "independent_spectrum_scaffold_split_v2",
                "production_eligible": True,
                "supports_modalities": ["1h"],
            },
            "features": elucidation._FEATURE_NAMES,
        },
    )

    result = elucidation.rank_candidates(
        peaks_1h=[{"shift": 1.0}, {"shift": 2.0}],
        formula="C2H6",
    )

    assert result["ranker"]["loaded"] is True
    assert result["candidates"][0]["matched_1h"] == 0
    assert result["candidates"][0]["ranking_score"] == 0.0
    assert result["candidates"][0]["evidence_level"] == "no_match"


def test_missing_same_nucleus_reference_is_not_reported_as_weak_evidence(
    tmp_path,
    monkeypatch,
):
    _configure_index(tmp_path, monkeypatch)
    conn = elucidation._connect()
    _insert_record(conn, peaks_13c=[20.0, 30.0], peaks_1h=[])
    conn.commit()
    conn.close()

    result = elucidation.rank_candidates(
        peaks_1h=[{"shift": 1.0}, {"shift": 2.0}],
        formula="C2H6",
    )

    candidate = result["candidates"][0]
    assert candidate["score_breakdown"]["h1"]["reference_count"] == 0
    assert candidate["matched_1h"] == 0
    assert candidate["ranking_score"] == 0.0
    assert candidate["evidence_level"] == "no_reference"
