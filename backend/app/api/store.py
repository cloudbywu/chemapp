from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.analysis.models import AnalysisResult
from app.core.models import Spectrum, Technique, Peak
from app.paths import chemapp_db_path


@dataclass
class StoredSpectrum:
    id: str
    spectrum: Spectrum
    result: AnalysisResult | None = None
    spectrum_revision: int = 1
    result_revision: int = 0


@dataclass
class ResultVersion:
    version: int
    result: AnalysisResult
    note: str = ""
    created_at: str = ""
    spectrum_revision: int | None = None


@dataclass(frozen=True)
class AIActionCommit:
    action_id: int
    action_name: str
    previous_version: int | None
    version: int
    result_revision: int


class RevisionConflict(ValueError):
    def __init__(self, current_revision: int):
        self.current_revision = int(current_revision)
        super().__init__(f"Revision conflict; current revision is {self.current_revision}")


class SpectrumResultRevisionConflict(RevisionConflict):
    """Processing would clear a result saved after its initial read."""

    def __init__(self, spectrum_revision: int, result_revision: int):
        self.current_result_revision = int(result_revision)
        super().__init__(spectrum_revision)


class ResultVersionSourceConflict(ValueError):
    """A historical result cannot be attached to the current spectrum."""

    def __init__(self, source_revision: int | None, current_revision: int):
        self.source_spectrum_revision = source_revision
        self.current_spectrum_revision = int(current_revision)
        message = (
            "This historical result has no verifiable source spectrum revision; "
            "it remains in history but cannot be restored."
            if source_revision is None
            else "This result belongs to a different spectrum revision; "
            "reanalyze the current spectrum instead of restoring it."
        )
        super().__init__(message)

    def to_detail(self) -> dict[str, Any]:
        return {
            "code": "result_source_mismatch",
            "message": str(self),
            "source_spectrum_revision": self.source_spectrum_revision,
            "current_spectrum_revision": self.current_spectrum_revision,
        }


class AIUndoConflict(ValueError):
    """Undo cannot prove it will preserve later non-AI result edits."""

    def __init__(self, reason: str, current_revision: int):
        self.reason = reason
        self.current_revision = int(current_revision)
        message = (
            "This AI action has no verifiable edit history. Undo is blocked to "
            "protect the current result; its saved versions remain available."
            if reason == "unverifiable_history"
            else "The result was manually saved, restored, or reanalyzed after this "
            "AI action. Undo is blocked to preserve those later edits."
        )
        super().__init__(message)

    def to_detail(self) -> dict[str, Any]:
        return {
            "code": "ai_undo_blocked",
            "reason": self.reason,
            "message": str(self),
            "current_revision": self.current_revision,
        }


class SpectrumDeleteConflict(ValueError):
    """Raised when a delete precondition no longer matches stored state."""

    def __init__(self, spectrum_revision: int, result_revision: int):
        self.current_spectrum_revision = int(spectrum_revision)
        self.current_result_revision = int(result_revision)
        super().__init__(
            "Delete revision conflict; current revisions are "
            f"spectrum={self.current_spectrum_revision}, "
            f"result={self.current_result_revision}"
        )


def _json_dumps(value: Any) -> str:
    """Serialize persisted data strictly so NaN/Infinity never reach SQLite."""
    return json.dumps(value, allow_nan=False, ensure_ascii=False, separators=(",", ":"))


class PersistentStore:
    def __init__(self, db_path: str | None = None):
        if db_path is None:
            # CHEMAPP_DB_PATH still wins; only the unset default moves from
            # a CWD-relative "data/chemapp.db" to the backend-root-anchored
            # canonical location (see app.paths.chemapp_db_path).
            db_path = str(chemapp_db_path())
        db_file = Path(db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(db_file)
        # This mutex serializes calls on one store instance. Transactions using
        # BEGIN IMMEDIATE protect read/check/write sequences across independent
        # store instances and server workers; the Python lock cannot do that.
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10.0)
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=10000")
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            # Serialize additive migrations across independently starting workers.
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS spectra (
                    id TEXT PRIMARY KEY,
                    technique TEXT NOT NULL,
                    name TEXT DEFAULT '',
                    points INTEGER DEFAULT 0,
                    spectrum_json TEXT NOT NULL,
                    result_json TEXT,
                    spectrum_revision INTEGER NOT NULL DEFAULT 1,
                    result_revision INTEGER NOT NULL DEFAULT 0,
                    non_ai_revision INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now'))
                )
            """)
            columns = {row[1] for row in conn.execute("PRAGMA table_info(spectra)").fetchall()}
            if "spectrum_revision" not in columns:
                conn.execute(
                    "ALTER TABLE spectra ADD COLUMN spectrum_revision INTEGER NOT NULL DEFAULT 1"
                )
            if "result_revision" not in columns:
                conn.execute(
                    "ALTER TABLE spectra ADD COLUMN result_revision INTEGER NOT NULL DEFAULT 0"
                )
            if "non_ai_revision" not in columns:
                # This is an opaque edit generation, not reconstructed history.
                # Old AI actions retain NULL below and cannot cross this boundary.
                conn.execute(
                    "ALTER TABLE spectra ADD COLUMN non_ai_revision INTEGER NOT NULL DEFAULT 0"
                )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_technique ON spectra(technique)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS result_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    spectrum_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    result_json TEXT NOT NULL,
                    spectrum_revision INTEGER,
                    note TEXT DEFAULT '',
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(spectrum_id, version),
                    FOREIGN KEY(spectrum_id) REFERENCES spectra(id) ON DELETE CASCADE
                )
            """)
            version_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(result_versions)").fetchall()
            }
            if "spectrum_revision" not in version_columns:
                # Historical rows cannot be reliably linked to their source:
                # retain them with NULL rather than inventing a provenance.
                conn.execute("ALTER TABLE result_versions ADD COLUMN spectrum_revision INTEGER")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_result_versions_sid ON result_versions(spectrum_id)")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ai_action_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    spectrum_id TEXT NOT NULL,
                    action_name TEXT NOT NULL,
                    args_json TEXT NOT NULL,
                    previous_version INTEGER,
                    new_version INTEGER,
                    non_ai_revision INTEGER,
                    undone INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    FOREIGN KEY(spectrum_id) REFERENCES spectra(id) ON DELETE CASCADE
                )
            """)
            action_columns = {
                row[1] for row in conn.execute("PRAGMA table_info(ai_action_history)").fetchall()
            }
            if "non_ai_revision" not in action_columns:
                # Never guess whether manual edits followed a legacy action.
                conn.execute("ALTER TABLE ai_action_history ADD COLUMN non_ai_revision INTEGER")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ai_action_history_sid ON ai_action_history(spectrum_id)")
            # Existing databases predate the foreign keys above. The trigger gives
            # those installations the same cascade behaviour without rebuilding
            # user tables in-place.
            conn.execute("""
                CREATE TRIGGER IF NOT EXISTS trg_spectra_delete_children
                AFTER DELETE ON spectra
                BEGIN
                    DELETE FROM result_versions WHERE spectrum_id=OLD.id;
                    DELETE FROM ai_action_history WHERE spectrum_id=OLD.id;
                END
            """)
            conn.commit()

    def add(self, spectrum: Spectrum) -> StoredSpectrum:
        return self.add_many([spectrum])[0]

    def add_many(self, spectra: list[Spectrum]) -> list[StoredSpectrum]:
        """Persist one logical upload atomically, including serialization failures."""

        stored_items: list[StoredSpectrum] = []
        rows = []
        for spectrum in spectra:
            sid = uuid.uuid4().hex[:12]
            payload = _json_dumps(spectrum.to_dict(include_internal=True))
            name = Path(spectrum.source_file).name if spectrum.source_file else ""
            rows.append((sid, spectrum.technique.value, name, spectrum.num_points, payload))
            stored_items.append(StoredSpectrum(id=sid, spectrum=spectrum))
        if not rows:
            return []
        with self._lock, self._connect() as conn:
            conn.executemany(
                "INSERT INTO spectra (id, technique, name, points, spectrum_json) VALUES (?,?,?,?,?)",
                rows,
            )
            conn.commit()
        return stored_items

    def get(self, sid: str) -> StoredSpectrum | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, spectrum_json, result_json, spectrum_revision, result_revision
                FROM spectra WHERE id=?
                """,
                (sid,),
            ).fetchone()
            if row is None:
                return None
            spec = Spectrum.from_dict(json.loads(row[1]))
            result = None
            if row[2]:
                raw = json.loads(row[2])
                result = _deserialize_result(raw)
            return StoredSpectrum(
                id=row[0],
                spectrum=spec,
                result=result,
                spectrum_revision=int(row[3] or 1),
                result_revision=int(row[4] or 0),
            )

    def list_all(
        self,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[StoredSpectrum]:
        offset = max(0, int(offset))
        sql = """
            SELECT id, spectrum_json, result_json, spectrum_revision, result_revision
            FROM spectra ORDER BY created_at DESC
        """
        params: list[Any] = []
        if limit is not None:
            limit = max(1, int(limit))
            sql += " LIMIT ? OFFSET ?"
            params = [limit, offset]
        results: list[StoredSpectrum] = []
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        for row in rows:
            spec = Spectrum.from_dict(json.loads(row[1]))
            result = None
            if row[2]:
                raw = json.loads(row[2])
                result = _deserialize_result(raw)
            results.append(
                StoredSpectrum(
                    id=row[0],
                    spectrum=spec,
                    result=result,
                    spectrum_revision=int(row[3] or 1),
                    result_revision=int(row[4] or 0),
                )
            )
        return results

    def list_metadata(self, limit: int = 1000, offset: int = 0) -> list[dict[str, Any]]:
        """List spectra without deserializing potentially large numeric arrays."""
        limit = max(1, min(int(limit), 5000))
        offset = max(0, int(offset))
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, technique, name, points, result_json IS NOT NULL,
                       created_at, spectrum_revision, result_revision
                FROM spectra
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [
            {
                "id": row[0],
                "technique": row[1],
                "name": row[2] or "",
                "points": int(row[3] or 0),
                "has_result": bool(row[4]),
                "created_at": row[5] or "",
                "spectrum_revision": int(row[6] or 1),
                "result_revision": int(row[7] or 0),
            }
            for row in rows
        ]

    def count(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM spectra").fetchone()
        return int(row[0] if row else 0)

    def remove(
        self,
        sid: str,
        *,
        expected_spectrum_revision: int | None = None,
        expected_result_revision: int | None = None,
    ) -> bool:
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT spectrum_revision, result_revision FROM spectra WHERE id=?",
                (sid,),
            ).fetchone()
            if row is None:
                conn.rollback()
                return False
            current_spectrum_revision = int(row[0] or 1)
            current_result_revision = int(row[1] or 0)
            if (
                expected_spectrum_revision is not None
                and int(expected_spectrum_revision) != current_spectrum_revision
            ) or (
                expected_result_revision is not None
                and int(expected_result_revision) != current_result_revision
            ):
                conn.rollback()
                raise SpectrumDeleteConflict(
                    current_spectrum_revision,
                    current_result_revision,
                )
            cursor = conn.execute("DELETE FROM spectra WHERE id=?", (sid,))
            conn.commit()
            return cursor.rowcount > 0

    def set_result(
        self,
        sid: str,
        result: AnalysisResult,
        *,
        expected_revision: int | None = None,
    ) -> StoredSpectrum | None:
        result_json = _json_dumps(result.to_dict())
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT spectrum_json, spectrum_revision, result_revision
                FROM spectra WHERE id=?
                """,
                (sid,),
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            current_revision = int(row[2] or 0)
            if expected_revision is not None and int(expected_revision) != current_revision:
                conn.rollback()
                raise RevisionConflict(current_revision)
            new_revision = current_revision + 1
            conn.execute(
                """UPDATE spectra SET result_json=?, result_revision=?,
                   non_ai_revision=non_ai_revision+1 WHERE id=?""",
                (result_json, new_revision, sid),
            )
            conn.commit()
        return StoredSpectrum(
            id=sid,
            spectrum=Spectrum.from_dict(json.loads(row[0])),
            result=result,
            spectrum_revision=int(row[1] or 1),
            result_revision=new_revision,
        )

    def save_result_version(
        self, sid: str, result: AnalysisResult, note: str = "", *, expected_revision: int
    ) -> int:
        result_json = _json_dumps(result.to_dict())
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            exists = conn.execute(
                "SELECT spectrum_revision, result_revision FROM spectra WHERE id=?", (sid,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"Spectrum {sid} not found")
            if int(expected_revision) != int(exists[1] or 0):
                raise RevisionConflict(int(exists[1] or 0))
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM result_versions WHERE spectrum_id=?",
                (sid,),
            ).fetchone()
            version = int(row[0] or 0) + 1
            conn.execute(
                """INSERT INTO result_versions
                   (spectrum_id, version, result_json, note, spectrum_revision)
                   VALUES (?,?,?,?,?)""",
                (sid, version, result_json, note, int(exists[0] or 1)),
            )
            conn.commit()
            return version

    def set_result_and_version(
        self,
        sid: str,
        result: AnalysisResult,
        note: str = "",
        *,
        version_metric: str | None = None,
        expected_revision: int | None = None,
        restore_from_version: int | None = None,
    ) -> tuple[StoredSpectrum, int] | None:
        """Atomically update the current result and append its immutable version."""
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """
                SELECT spectrum_json, spectrum_revision, result_revision
                FROM spectra WHERE id=?
                """,
                (sid,),
            ).fetchone()
            if existing is None:
                conn.rollback()
                return None
            current_revision = int(existing[2] or 0)
            if expected_revision is not None and int(expected_revision) != current_revision:
                conn.rollback()
                raise RevisionConflict(current_revision)
            if restore_from_version is not None:
                source = conn.execute(
                    "SELECT spectrum_revision FROM result_versions WHERE spectrum_id=? AND version=?",
                    (sid, restore_from_version),
                ).fetchone()
                if source is None:
                    raise KeyError(f"Result version {restore_from_version} not found")
                source_revision = int(source[0]) if source[0] is not None else None
                if source_revision != int(existing[1] or 1):
                    raise ResultVersionSourceConflict(source_revision, int(existing[1] or 1))
            new_result_revision = current_revision + 1
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM result_versions WHERE spectrum_id=?",
                (sid,),
            ).fetchone()
            version = int(row[0] or 0) + 1
            if version_metric:
                result.metrics[version_metric] = version
            result_json = _json_dumps(result.to_dict())
            conn.execute(
                """UPDATE spectra SET result_json=?, result_revision=?,
                   non_ai_revision=non_ai_revision+1 WHERE id=?""",
                (result_json, new_result_revision, sid),
            )
            conn.execute(
                """INSERT INTO result_versions
                   (spectrum_id, version, result_json, note, spectrum_revision)
                   VALUES (?,?,?,?,?)""",
                (sid, version, result_json, note, int(existing[1] or 1)),
            )
            conn.commit()
        spectrum = Spectrum.from_dict(json.loads(existing[0]))
        return (
            StoredSpectrum(
                id=sid,
                spectrum=spectrum,
                result=result,
                spectrum_revision=int(existing[1] or 1),
                result_revision=new_result_revision,
            ),
            version,
        )

    def list_result_versions(self, sid: str) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT v.version, v.result_json, v.note, v.created_at,
                       v.spectrum_revision, s.spectrum_revision
                FROM result_versions v JOIN spectra s ON s.id=v.spectrum_id
                WHERE v.spectrum_id=?
                ORDER BY version DESC
                """,
                (sid,),
            ).fetchall()
        versions = []
        for version, result_json, note, created_at, source_revision, current_revision in rows:
            raw = json.loads(result_json)
            versions.append({
                "version": int(version),
                "note": note or "",
                "created_at": created_at,
                "spectrum_revision": int(source_revision) if source_revision is not None else None,
                "restorable": source_revision is not None and source_revision == current_revision,
                "n_peaks": len(raw.get("peaks", [])),
                "manual_confirmed": bool(raw.get("metrics", {}).get("manual_confirmed")),
                "summary": raw.get("summary", ""),
            })
        return versions

    def get_result_version(self, sid: str, version: int) -> ResultVersion | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT version, result_json, note, created_at, spectrum_revision
                FROM result_versions
                WHERE spectrum_id=? AND version=?
                """,
                (sid, version),
            ).fetchone()
        if row is None:
            return None
        return ResultVersion(
            version=int(row[0]),
            result=_deserialize_result(json.loads(row[1])),
            note=row[2] or "",
            created_at=row[3] or "",
            spectrum_revision=int(row[4]) if row[4] is not None else None,
        )

    def commit_ai_action(
        self,
        sid: str,
        result: AnalysisResult,
        action_name: str,
        args: dict[str, Any],
        note: str = "",
        *,
        expected_revision: int,
        expected_spectrum_revision: int,
    ) -> AIActionCommit | None:
        """Commit a computed result, its preimage, and audit row together.

        Scientific computation and serialization happen before taking a write
        lock. The preimage is copied from the checked persisted state, never
        from a caller's mutable result or an earlier, potentially stale read.
        """
        result_json = _json_dumps(result.to_dict())
        args_json = _json_dumps(args)
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                """SELECT result_json, spectrum_revision, result_revision, non_ai_revision
                   FROM spectra WHERE id=?""",
                (sid,),
            ).fetchone()
            if existing is None:
                return None
            current_revision = int(existing[2] or 0)
            if int(expected_revision) != current_revision:
                raise RevisionConflict(current_revision)
            source_revision = int(existing[1] or 1)
            if int(expected_spectrum_revision) != source_revision:
                raise ResultVersionSourceConflict(expected_spectrum_revision, source_revision)
            latest = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM result_versions WHERE spectrum_id=?",
                (sid,),
            ).fetchone()[0]
            version = int(latest or 0) + 1
            previous_version = None
            if existing[0] is not None:
                previous_version = version
                conn.execute(
                    """INSERT INTO result_versions
                       (spectrum_id, version, result_json, note, spectrum_revision)
                       VALUES (?,?,?,?,?)""",
                    (sid, previous_version, existing[0], f"Before AI action {action_name}", source_revision),
                )
                version += 1
            new_revision = current_revision + 1
            conn.execute(
                "UPDATE spectra SET result_json=?, result_revision=? WHERE id=?",
                (result_json, new_revision, sid),
            )
            conn.execute(
                """INSERT INTO result_versions
                   (spectrum_id, version, result_json, note, spectrum_revision)
                   VALUES (?,?,?,?,?)""",
                (sid, version, result_json, note, source_revision),
            )
            cursor = conn.execute(
                """INSERT INTO ai_action_history
                   (spectrum_id, action_name, args_json, previous_version, new_version, non_ai_revision)
                   VALUES (?,?,?,?,?,?)""",
                (sid, action_name, args_json, previous_version, version, int(existing[3])),
            )
            committed = AIActionCommit(
                action_id=int(cursor.lastrowid),
                action_name=action_name,
                previous_version=previous_version,
                version=version,
                result_revision=new_revision,
            )
            conn.commit()
        return committed

    def undo_last_ai_action(
        self, sid: str, *, expected_revision: int
    ) -> AIActionCommit | None:
        """Restore and mark the latest eligible action undone in one transaction.

        Revision, source, and non-AI edit checks run under the write lock. Only
        AI commits and undo keep the edit generation unchanged, so sequential
        undo remains possible without crossing a manual save, restore, or
        reanalysis. Legacy actions stay visible but lack verified provenance.
        """
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute(
                "SELECT spectrum_revision, result_revision, non_ai_revision FROM spectra WHERE id=?",
                (sid,),
            ).fetchone()
            if existing is None:
                raise KeyError(f"Spectrum {sid} not found")
            current_revision = int(existing[1] or 0)
            if int(expected_revision) != current_revision:
                raise RevisionConflict(current_revision)
            last = conn.execute(
                """SELECT id, action_name, previous_version, new_version, non_ai_revision
                   FROM ai_action_history
                   WHERE spectrum_id=? AND undone=0 AND previous_version IS NOT NULL
                   ORDER BY id DESC LIMIT 1""",
                (sid,),
            ).fetchone()
            if last is None:
                return None
            action_id, action_name, previous_version, new_version, non_ai_revision = last
            saved = conn.execute(
                "SELECT result_json, spectrum_revision FROM result_versions WHERE spectrum_id=? AND version=?",
                (sid, previous_version),
            ).fetchone()
            if saved is None:
                raise KeyError(f"Previous version {previous_version} not found")
            source_revision = int(saved[1]) if saved[1] is not None else None
            if source_revision != int(existing[0] or 1):
                raise ResultVersionSourceConflict(source_revision, int(existing[0] or 1))
            if non_ai_revision is None:
                raise AIUndoConflict("unverifiable_history", current_revision)
            if int(non_ai_revision) != int(existing[2]):
                raise AIUndoConflict("later_result_edit", current_revision)
            result = _deserialize_result(json.loads(saved[0]))
            result.metrics["restored_from_version"] = int(previous_version)
            result.metrics["ai_undo"] = {
                "undone_action": action_name,
                "undone_action_id": int(action_id),
                "undone_new_version": int(new_version) if new_version is not None else None,
            }
            result_json = _json_dumps(result.to_dict())
            latest = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM result_versions WHERE spectrum_id=?",
                (sid,),
            ).fetchone()[0]
            version = int(latest or 0) + 1
            new_revision = current_revision + 1
            conn.execute(
                "UPDATE spectra SET result_json=?, result_revision=? WHERE id=?",
                (result_json, new_revision, sid),
            )
            conn.execute(
                """INSERT INTO result_versions
                   (spectrum_id, version, result_json, note, spectrum_revision)
                   VALUES (?,?,?,?,?)""",
                (sid, version, result_json, f"Undo AI action {action_name}", source_revision),
            )
            conn.execute("UPDATE ai_action_history SET undone=1 WHERE id=?", (action_id,))
            committed = AIActionCommit(
                action_id=int(action_id),
                action_name=action_name,
                previous_version=int(previous_version),
                version=version,
                result_revision=new_revision,
            )
            conn.commit()
        return committed

    def record_ai_action(
        self,
        sid: str,
        action_name: str,
        args: dict[str, Any],
        previous_version: int | None,
        new_version: int | None,
    ) -> int:
        args_json = _json_dumps(args)
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO ai_action_history
                (spectrum_id, action_name, args_json, previous_version, new_version)
                VALUES (?,?,?,?,?)
                """,
                (sid, action_name, args_json, previous_version, new_version),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def get_last_ai_action(self, sid: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                """
                SELECT id, spectrum_id, action_name, args_json, previous_version, new_version, created_at
                FROM ai_action_history
                WHERE spectrum_id=? AND undone=0 AND previous_version IS NOT NULL
                ORDER BY id DESC
                LIMIT 1
                """,
                (sid,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": int(row[0]),
            "spectrum_id": row[1],
            "action_name": row[2],
            "args": json.loads(row[3]),
            "previous_version": int(row[4]),
            "new_version": int(row[5]) if row[5] is not None else None,
            "created_at": row[6] or "",
        }

    def mark_ai_action_undone(self, action_id: int) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("UPDATE ai_action_history SET undone=1 WHERE id=?", (action_id,))
            conn.commit()

    def latest_result_version_number(self, sid: str) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(version), 0) FROM result_versions WHERE spectrum_id=?",
                (sid,),
            ).fetchone()
        return int(row[0] or 0)

    def set_spectrum(
        self,
        sid: str,
        spectrum: Spectrum,
        clear_result: bool = True,
        *,
        expected_revision: int | None = None,
        expected_result_revision: int | None = None,
    ) -> StoredSpectrum | None:
        name = Path(spectrum.source_file).name if spectrum.source_file else ""
        spectrum_json = _json_dumps(spectrum.to_dict(include_internal=True))
        with self._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """
                SELECT result_json, spectrum_revision, result_revision
                FROM spectra WHERE id=?
                """,
                (sid,),
            ).fetchone()
            if row is None:
                conn.rollback()
                return None
            current_revision = int(row[1] or 1)
            if expected_revision is not None and int(expected_revision) != current_revision:
                conn.rollback()
                raise RevisionConflict(current_revision)
            if (
                expected_result_revision is not None
                and int(expected_result_revision) != int(row[2] or 0)
            ):
                raise SpectrumResultRevisionConflict(current_revision, int(row[2] or 0))
            new_spectrum_revision = current_revision + 1
            new_result_revision = int(row[2] or 0) + (1 if clear_result else 0)
            conn.execute(
                """
                UPDATE spectra
                SET technique=?, name=?, points=?, spectrum_json=?,
                    spectrum_revision=?, result_revision=?,
                    non_ai_revision=non_ai_revision+1,
                    result_json=CASE WHEN ? THEN NULL ELSE result_json END
                WHERE id=?
                """,
                (
                    spectrum.technique.value,
                    name,
                    spectrum.num_points,
                    spectrum_json,
                    new_spectrum_revision,
                    new_result_revision,
                    1 if clear_result else 0,
                    sid,
                ),
            )
            conn.commit()
        previous_result = (
            _deserialize_result(json.loads(row[0]))
            if row[0] and not clear_result
            else None
        )
        return StoredSpectrum(
            id=sid,
            spectrum=spectrum,
            result=previous_result,
            spectrum_revision=new_spectrum_revision,
            result_revision=new_result_revision,
        )


def _deserialize_result(data: dict) -> AnalysisResult:
    from app.analysis.models import (
        FluorescenceAnalysisResult,
        NMRAnalysisResult,
        UVVisAnalysisResult,
    )
    from app.analysis.xrd_analysis import XRDAnalysisResult
    technique = data.get("technique", "NMR")
    # Map technique string to Technique enum
    tech_map = {"NMR": Technique.NMR, "UV-Vis": Technique.UVVIS, "Fluorescence": Technique.FLUORESCENCE,
                "XRD": Technique.XRD, "HPLC": Technique.HPLC, "ElectroChem": Technique.ELECTROCHEM}
    tech_obj = tech_map.get(technique, Technique.NMR)
    pk_data = data.get("peaks", [])
    peaks = [Peak(position=p["position"], intensity=p["intensity"],
                  area=p.get("area"), width=p.get("width"),
                  assignment=p.get("assignment", ""),
                  multiplicity=p.get("multiplicity", ""),
                  coupling_constant=p.get("coupling_constant")) for p in pk_data]
    if technique == "NMR":
        return NMRAnalysisResult(
            technique=tech_obj,
            integrals=data.get("integrals", []),
            multiplets=data.get("multiplets", []),
            noise_level=data.get("noise_level", 0),
            total_integral=data.get("total_integral", 0),
            solvent_shift=data.get("solvent_shift"),
            reference_corrected=data.get("reference_corrected", False),
            peaks=peaks, metrics=data.get("metrics", {}),
            summary=data.get("summary", ""),
        )
    elif technique == "UV-Vis":
        return UVVisAnalysisResult(
            technique=tech_obj,
            lambda_max=data.get("lambda_max", []),
            calibration=data.get("calibration"),
            sample_concentration=data.get("sample_concentration"),
            concentration_unit=data.get("concentration_unit", ""),
            baseline_corrected=data.get("baseline_corrected", False),
            peaks=peaks, metrics=data.get("metrics", {}),
            summary=data.get("summary", ""),
        )
    elif technique == "Fluorescence":
        return FluorescenceAnalysisResult(
            technique=tech_obj,
            ex_peak=data.get("ex_peak"),
            em_peak=data.get("em_peak"),
            stokes_shift_nm=data.get("stokes_shift_nm"),
            stokes_shift_cm1=data.get("stokes_shift_cm1"),
            quantum_yield_ref=data.get("quantum_yield_ref"),
            normalized=data.get("normalized", False),
            peaks=peaks, metrics=data.get("metrics", {}),
            summary=data.get("summary", ""),
        )
    elif technique == "XRD":
        return XRDAnalysisResult(
            technique=tech_obj,
            peaks=peaks,
            metrics=data.get("metrics", {}),
            summary=data.get("summary", ""),
            d_spacings=data.get("d_spacings", []),
            crystallite_sizes=data.get("crystallite_sizes", []),
            peak_assignments=data.get("peak_assignments", []),
            phase_matches=data.get("phase_matches", []),
            lattice_parameters=data.get("lattice_parameters", []),
            crystallinity=data.get("crystallinity", {}),
            williamson_hall=data.get("williamson_hall", {}),
            size_distribution=data.get("size_distribution", {}),
            quantitative_analysis=data.get("quantitative_analysis", []),
            crystal_structure=data.get("crystal_structure", []),
            rietveld_refinement=data.get("rietveld_refinement", {}),
            wavelength_used=data.get("wavelength_used", 1.54059),
        )
    return AnalysisResult(
        technique=tech_obj,
        peaks=peaks,
        metrics=data.get("metrics", {}),
        summary=data.get("summary", ""),
    )


# Module-level instance
_store: PersistentStore | None = None

# Serializes lazy initialization so concurrent first callers cannot each
# create their own PersistentStore (and SQLite schema) for the same database.
_store_init_lock = threading.Lock()


def get_store() -> PersistentStore:
    """Return the shared store, lazily creating it exactly once."""

    global _store
    if _store is None:
        with _store_init_lock:
            if _store is None:
                _store = PersistentStore()
    return _store
