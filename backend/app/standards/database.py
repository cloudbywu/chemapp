from __future__ import annotations

import json
import os
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class StandardRecord:
    id: str
    name: str
    technique: str
    formula: str = ""
    source: str = "ChemApp built-in"
    tags: list[str] = field(default_factory=list)
    peaks: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "technique": self.technique,
            "formula": self.formula,
            "source": self.source,
            "tags": self.tags,
            "peaks": self.peaks,
            "metadata": self.metadata,
        }


_BUILTIN: list[StandardRecord] = [
    StandardRecord(
        id="nmr-cdcl3",
        name="Chloroform-d residual peak",
        technique="NMR",
        formula="CHCl3",
        tags=["solvent", "1H", "13C"],
        peaks=[{"shift": 7.26, "nucleus": "1H", "multiplicity": "s"}, {"shift": 77.16, "nucleus": "13C"}],
    ),
    StandardRecord(
        id="nmr-dmso",
        name="DMSO-d6 residual peak",
        technique="NMR",
        formula="C2D6OS",
        tags=["solvent", "1H", "13C"],
        peaks=[{"shift": 2.50, "nucleus": "1H", "multiplicity": "quint"}, {"shift": 39.52, "nucleus": "13C"}],
    ),
    StandardRecord(
        id="hplc-caffeine",
        name="Caffeine",
        technique="HPLC",
        formula="C8H10N4O2",
        tags=["uv", "reference"],
        peaks=[{"retention_time": 2.35, "wavelength_nm": 254, "type": "BB"}],
        metadata={"lambda_max_nm": [205, 273]},
    ),
    StandardRecord(
        id="xrd-cds-41-1049",
        name="Cadmium Sulfide",
        technique="XRD",
        formula="CdS",
        source="ICDD-style built-in reference",
        tags=["hexagonal", "semiconductor"],
        peaks=[
            {"two_theta": 24.8, "hkl": "100", "rel_intensity": 65},
            {"two_theta": 26.5, "hkl": "002", "rel_intensity": 100},
            {"two_theta": 28.2, "hkl": "101", "rel_intensity": 78},
            {"two_theta": 43.7, "hkl": "110", "rel_intensity": 45},
            {"two_theta": 51.9, "hkl": "112", "rel_intensity": 28},
        ],
        metadata={"card_number": "41-1049", "crystal_system": "hexagonal", "space_group": "P63mc"},
    ),
    StandardRecord(
        id="xrd-si-27-1402",
        name="Silicon",
        technique="XRD",
        formula="Si",
        source="ICDD-style built-in reference",
        tags=["cubic", "calibrant"],
        peaks=[
            {"two_theta": 28.44, "hkl": "111", "rel_intensity": 100},
            {"two_theta": 47.30, "hkl": "220", "rel_intensity": 55},
            {"two_theta": 56.12, "hkl": "311", "rel_intensity": 32},
        ],
        metadata={"card_number": "27-1402", "crystal_system": "cubic", "space_group": "Fd-3m"},
    ),
]


class StandardDatabase:
    def __init__(self, db_path: str | None = None):
        self._records = {record.id: record for record in _BUILTIN}
        self._db_path = db_path or os.environ.get(
            "CHEMAPP_DB_PATH",
            str(Path("data") / "chemapp.db"),
        )
        self._lock = threading.RLock()
        self._init_persistence()
        self._load_user_records()

    def _connect(self) -> sqlite3.Connection:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_persistence(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS user_standards (
                    id TEXT PRIMARY KEY,
                    record_json TEXT NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now'))
                )
                """
            )
            conn.commit()

    def _load_user_records(self) -> None:
        with self._lock, self._connect() as conn:
            rows = conn.execute("SELECT record_json FROM user_standards").fetchall()
        for (record_json,) in rows:
            try:
                raw = json.loads(record_json)
                record = _record_from_raw(raw, default_source="User")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            self._records[record.id] = record

    def list_records(self, technique: str | None = None, query: str | None = None) -> list[dict[str, Any]]:
        technique_norm = technique.lower() if technique else ""
        query_norm = query.lower() if query else ""
        rows = []
        with self._lock:
            records = list(self._records.values())
        for record in records:
            if technique_norm and record.technique.lower() != technique_norm:
                continue
            haystack = " ".join([record.id, record.name, record.formula, record.source, *record.tags]).lower()
            if query_norm and query_norm not in haystack:
                continue
            rows.append(record.to_dict())
        return sorted(rows, key=lambda row: (row["technique"], row["name"]))

    def add_record(self, raw: dict[str, Any]) -> dict[str, Any]:
        rid = str(raw.get("id") or f"user-{uuid.uuid4().hex[:12]}")
        if rid in {record.id for record in _BUILTIN}:
            raise ValueError("Built-in standard IDs cannot be replaced")
        record = _record_from_raw({**raw, "id": rid}, default_source="User")
        record_json = json.dumps(
            record.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO user_standards (id, record_json, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(id) DO UPDATE SET
                    record_json=excluded.record_json,
                    updated_at=datetime('now')
                """,
                (record.id, record_json),
            )
            conn.commit()
            self._records[record.id] = record
        return record.to_dict()

    def match_peaks(self, technique: str, peaks: list[dict[str, Any]], tolerance: float = 0.05) -> list[dict[str, Any]]:
        matches = []
        with self._lock:
            records = list(self._records.values())
        for record in records:
            if record.technique.lower() != technique.lower():
                continue
            ref_values = [_peak_value(record.technique, p) for p in record.peaks]
            obs_values = [_peak_value(technique, p) for p in peaks]
            ref_values = [v for v in ref_values if v is not None]
            obs_values = [v for v in obs_values if v is not None]
            if not ref_values or not obs_values:
                continue
            errors = _match_values(ref_values, obs_values, tolerance)
            hit_count = len(errors)
            score = hit_count / len(ref_values)
            if hit_count:
                matches.append({
                    "record": record.to_dict(),
                    "matched_peaks": hit_count,
                    "reference_peaks": len(ref_values),
                    "score": round(score, 3),
                    "mean_error": round(sum(errors) / len(errors), 4) if errors else None,
                })
        return sorted(matches, key=lambda row: (row["score"], row["matched_peaks"]), reverse=True)


def _record_from_raw(raw: dict[str, Any], *, default_source: str) -> StandardRecord:
    rid = str(raw.get("id") or "").strip()
    if not rid:
        raise ValueError("Standard ID is required")
    return StandardRecord(
        id=rid,
        name=str(raw.get("name") or rid),
        technique=str(raw.get("technique") or "NMR"),
        formula=str(raw.get("formula") or ""),
        source=str(raw.get("source") or default_source),
        tags=[str(t) for t in raw.get("tags", [])],
        peaks=list(raw.get("peaks") or []),
        metadata=dict(raw.get("metadata") or {}),
    )


def _peak_value(technique: str, peak: dict[str, Any]) -> float | None:
    keys = {
        "NMR": ("shift", "position", "center_ppm"),
        "HPLC": ("retention_time", "position", "rt"),
        "XRD": ("two_theta", "position"),
    }.get(technique, ("position",))
    for key in keys:
        if key in peak:
            try:
                return float(peak[key])
            except (TypeError, ValueError):
                return None
    return None


def _match_values(
    ref_values: list[float],
    obs_values: list[float],
    tolerance: float,
) -> list[float]:
    """One-to-one consumed matching: each observed value is used at most once."""

    ref_sorted = sorted(ref_values)
    obs_sorted = sorted(obs_values)
    used: set[int] = set()
    pointer = 0
    matched: list[float] = []
    for ref in ref_sorted:
        while pointer < len(obs_sorted) and obs_sorted[pointer] < ref - tolerance:
            pointer += 1
        best_index: int | None = None
        best_error = tolerance + 1.0
        scan = pointer
        while scan < len(obs_sorted) and obs_sorted[scan] <= ref + tolerance:
            if scan not in used:
                error = abs(obs_sorted[scan] - ref)
                if error < best_error:
                    best_error = error
                    best_index = scan
            scan += 1
        if best_index is not None:
            used.add(best_index)
            matched.append(best_error)
    return matched


_db: StandardDatabase | None = None
_db_path: str | None = None
_db_lock = threading.Lock()


def get_standard_database() -> StandardDatabase:
    global _db, _db_path
    path = os.environ.get("CHEMAPP_DB_PATH", str(Path("data") / "chemapp.db"))
    with _db_lock:
        if _db is None or _db_path != path:
            _db = StandardDatabase(path)
            _db_path = path
        return _db
