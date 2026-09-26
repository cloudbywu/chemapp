"""Test PersistentStore — SQLite persistence."""
import tempfile

import pytest

from app.api.store import PersistentStore
from app.core.models import Spectrum, Technique, SampleInfo
import numpy as np


@pytest.fixture
def store():
    db_path = tempfile.mktemp(suffix=".db")
    s = PersistentStore(db_path)
    yield s
    # SQLite may hold file lock; skip cleanup on Windows


@pytest.fixture
def sample_spectrum():
    return Spectrum(
        technique=Technique.NMR,
        x_data=np.linspace(0, 10, 100),
        y_data=np.random.randn(100),
        x_label="ppm",
        y_label="intensity",
        x_unit="ppm",
        y_unit="arb",
        metadata=SampleInfo(name="test", solvent="CDCl3"),
        source_file="test.fid",
    )


def test_add_and_get(store, sample_spectrum):
    stored = store.add(sample_spectrum)
    assert stored.id is not None
    assert len(stored.id) == 12

    retrieved = store.get(stored.id)
    assert retrieved is not None
    assert retrieved.id == stored.id
    assert retrieved.spectrum.technique == Technique.NMR
    assert np.allclose(retrieved.spectrum.x_data, sample_spectrum.x_data)


def test_list_all(store, sample_spectrum):
    store.add(sample_spectrum)
    store.add(sample_spectrum)
    items = store.list_all()
    assert len(items) == 2


def test_remove(store, sample_spectrum):
    stored = store.add(sample_spectrum)
    assert store.remove(stored.id) is True
    assert store.get(stored.id) is None
    assert store.remove("nonexistent") is False


def test_set_result(store, sample_spectrum):
    from app.analysis.models import AnalysisResult
    stored = store.add(sample_spectrum)
    result = AnalysisResult(
        technique=Technique.NMR,
        peaks=[],
        metrics={"n_peaks": 10},
        summary="Test analysis",
    )
    updated = store.set_result(stored.id, result)
    assert updated is not None
    assert updated.result is not None
    assert updated.result.summary == "Test analysis"

    # Verify persistence (new store instance)
    store2 = PersistentStore(store._db_path)
    retrieved = store2.get(stored.id)
    assert retrieved is not None
    assert retrieved.result is not None
    assert retrieved.result.summary == "Test analysis"


def test_get_nonexistent(store):
    assert store.get("nonexistent") is None


def test_set_result_nonexistent(store):
    from app.analysis.models import AnalysisResult
    result = AnalysisResult(technique=Technique.NMR, peaks=[], metrics={}, summary="")
    assert store.set_result("nonexistent", result) is None
