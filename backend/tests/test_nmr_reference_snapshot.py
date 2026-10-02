"""A reference import must never split rows from their normalized spectra."""
from dataclasses import FrozenInstanceError
import json

import pytest

from app.ml import nmr_structure_elucidation as engine


@pytest.fixture
def index(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR_INDEX", str(tmp_path / "index.sqlite"))
    monkeypatch.setenv("CHEMAPP_NMR_RANKER", str(tmp_path / "missing.joblib"))
    engine._load_records_cached.cache_clear()
    _insert("old", "CC", [20.0])


def _insert(source_id, smiles, peaks):
    conn = engine._connect()
    try:
        conn.execute(
            """INSERT INTO nmr_records
            (source, source_id, name, smiles, formula, peaks_13c, peaks_1h, metadata)
            VALUES ('fixture', ?, ?, ?, 'C2H6', ?, ?, '{}')""",
            (source_id, source_id, smiles, json.dumps(peaks),
             json.dumps([{"shift": 1.0, "intensity": 1.0}])),
        )
        conn.commit()
    finally:
        conn.close()


def test_import_after_snapshot_does_not_mix_rows_and_normalization(index, monkeypatch):
    load_snapshot = engine._load_reference_snapshot
    calls = []

    def import_after_read(limit=0):
        snapshot = load_snapshot(limit)
        calls.append(snapshot)
        # Deterministic concurrent-import interleaving: commit after the query
        # has obtained its rows, but before it starts matching the references.
        _insert("new", "CCC", [30.0])
        return snapshot

    monkeypatch.setattr(engine, "_load_reference_snapshot", import_after_read)
    result = engine.rank_candidates(peaks_13c=[{"shift": 20.0}])
    assert len(calls) == 1
    assert [row["source_id"] for row in result["candidates"]] == ["old"]
    assert result["candidates"][0]["matched_13c"] == 1

    monkeypatch.setattr(engine, "_load_reference_snapshot", load_snapshot)
    next_result = engine.rank_candidates(peaks_13c=[{"shift": 30.0}])
    assert {row["source_id"] for row in next_result["candidates"]} == {"old", "new"}
    assert next_result["candidates"][0]["source_id"] == "new"
    assert next_result["candidates"][0]["matched_13c"] == 1


def test_cached_snapshot_is_deeply_immutable_and_reuses_normalization(index, monkeypatch):
    snapshot = engine._load_reference_snapshot()
    with pytest.raises(FrozenInstanceError):
        snapshot.records = ()
    with pytest.raises(TypeError):
        snapshot.records[0]["source_id"] = "corrupted"
    with pytest.raises(TypeError):
        snapshot.records[0]["peaks_13c"][0] = 999.0
    with pytest.raises(TypeError):
        snapshot.records[0]["peaks_1h"][0]["shift"] = 999.0
    with pytest.raises(TypeError):
        snapshot.normalized[0][0][0] = 999.0

    def unexpected_normalization(*args, **kwargs):
        raise AssertionError("A cache hit must not normalize records again")

    monkeypatch.setattr(engine, "_normalise_resonance_shifts", unexpected_normalization)
    assert engine._load_reference_snapshot() is snapshot


def test_snapshot_normalization_matches_uncached_scoring(index):
    snapshot = engine._load_reference_snapshot()
    for record, normalized in zip(snapshot.records, snapshot.normalized, strict=True):
        direct = engine._feature_vector([20.0], [1.0], record)
        cached = engine._feature_vector([20.0], [1.0], record, _ref_norm=normalized)
        assert cached == direct


def test_snapshot_length_invariant_is_checked(index, monkeypatch):
    snapshot = engine._load_reference_snapshot()
    corrupt = engine._ReferenceSnapshot(snapshot.records, ())
    monkeypatch.setattr(engine, "_load_reference_snapshot", lambda limit=0: corrupt)
    with pytest.raises(ValueError, match="zip"):
        engine.rank_candidates(peaks_13c=[{"shift": 20.0}])
