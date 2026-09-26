"""Shared runtime asset path resolution for ChemApp backend.

Keeping these in one module prevents health endpoints and route modules
from resolving the same asset to different default locations.
"""

from __future__ import annotations

import os
from pathlib import Path

_BACKEND_ROOT = Path(__file__).resolve().parents[1]


def nmr_index_v2_path() -> Path:
    """Return the NMR v2 spectral index location.

    ``CHEMAPP_NMR_INDEX_V2`` wins when set; otherwise the default is the
    repository's ``backend/data`` directory, independent of the process
    working directory.
    """

    env = os.environ.get("CHEMAPP_NMR_INDEX_V2")
    if env:
        return Path(env)
    return _BACKEND_ROOT / "data" / "nmr_spectral_index_v2.sqlite"


def chemapp_db_path() -> Path:
    """Return the main spectrum store (SQLite) location.

    ``CHEMAPP_DB_PATH`` wins when set. The default used to be
    ``Path("data") / "chemapp.db"`` relative to the process working
    directory, so a server launched from any other directory silently
    created and used a second, empty database. Anchoring the default to the
    backend root keeps one canonical store per checkout regardless of CWD.

    Deployment note: existing deployments that relied on the CWD-relative
    default keep the exact same file as long as they launch from the
    backend directory (the anchored default resolves to the same
    ``backend/data/chemapp.db``); any other launch directory must set
    ``CHEMAPP_DB_PATH`` explicitly. Deployments already setting
    ``CHEMAPP_DB_PATH`` are unaffected.
    """

    env = os.environ.get("CHEMAPP_DB_PATH")
    if env:
        return Path(env)
    return _BACKEND_ROOT / "data" / "chemapp.db"


__all__ = ["chemapp_db_path", "nmr_index_v2_path"]
