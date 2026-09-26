"""Tests for shared runtime asset path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

import app.paths as paths_module
from app.api.store import PersistentStore
from app.paths import chemapp_db_path, nmr_index_v2_path


def test_default_index_path_is_backend_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHEMAPP_NMR_INDEX_V2", raising=False)
    path = nmr_index_v2_path()
    assert path.name == "nmr_spectral_index_v2.sqlite"
    assert path.parent.name == "data"
    assert path.parent.parent.name == "backend"


def test_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom.sqlite"
    monkeypatch.setenv("CHEMAPP_NMR_INDEX_V2", str(custom))
    assert nmr_index_v2_path() == custom


def test_default_db_path_is_backend_data(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CHEMAPP_DB_PATH", raising=False)
    path = chemapp_db_path()
    assert path.name == "chemapp.db"
    assert path.parent.name == "data"
    assert path.parent.parent.name == "backend"


def test_db_env_override_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    custom = tmp_path / "custom.db"
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(custom))
    assert chemapp_db_path() == custom


def test_store_default_db_is_anchored_not_cwd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CHEMAPP_DB_PATH", raising=False)
    monkeypatch.setattr(paths_module, "_BACKEND_ROOT", tmp_path / "backend")
    monkeypatch.chdir(tmp_path)
    store = PersistentStore()
    assert Path(store._db_path) == tmp_path / "backend" / "data" / "chemapp.db"
