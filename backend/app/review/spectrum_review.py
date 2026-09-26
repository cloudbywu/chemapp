from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlsplit

from rdkit import Chem

from app.api.store import PersistentStore


REVIEW_SCHEMA_VERSION = "chemapp.spectrum-review.v1"
GOLD_MANIFEST_SCHEMA_VERSION = "chemapp.nmr-gold-manifest.v1"
CHECK_NAMES = ("structure", "nucleus", "axis", "peaks", "solvent")
CHECK_VALUES = {"pass", "fail", "uncertain"}
QUEUE_STATES = {
    "pending",
    "awaiting_second",
    "conflict",
    "accepted",
    "rejected",
    "stale",
}
DEFAULT_REVIEW_LICENSES = {
    "CC0-1.0",
    "CC-BY-4.0",
    "CC-BY-SA-4.0",
}
INCHI_KEY_PATTERN = re.compile(r"[A-Z]{14}-[A-Z]{10}-[A-Z]")
SUBJECT_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{1,63}")


class ReviewStateError(ValueError):
    pass


class ReviewConflict(ValueError):
    def __init__(self, current_revision: int, message: str = "Review revision conflict"):
        self.current_revision = int(current_revision)
        super().__init__(message)


@dataclass(frozen=True)
class Snapshot:
    snapshot_sha256: str
    spectrum_sha256: str
    result_sha256: str | None
    spectrum_revision: int
    result_revision: int
    facts: dict[str, Any]
    spectrum: dict[str, Any]
    result: dict[str, Any] | None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def _normalize_structure(value: str) -> str:
    return "".join(value.split())


def _required_text(name: str, value: str) -> str:
    normalized = _normalize_text(value)
    if not normalized:
        raise ReviewStateError(f"{name} must not be blank")
    return normalized


def _allowed_review_licenses() -> set[str]:
    configured = os.environ.get("CHEMAPP_REVIEW_ALLOWED_LICENSES", "").strip()
    if not configured:
        return set(DEFAULT_REVIEW_LICENSES)
    return {
        value.strip()
        for value in configured.split(",")
        if value.strip()
    }


def _validated_license(value: str) -> str:
    normalized = _required_text("license_id", value)
    if normalized not in _allowed_review_licenses():
        raise ReviewStateError(
            f"License {normalized!r} is not explicitly allowed for review"
        )
    return normalized


def _validated_provenance_uri(value: str) -> str:
    normalized = _required_text("provenance_uri", value)
    parsed = urlsplit(normalized)
    if (
        parsed.scheme.lower() != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.fragment
    ):
        raise ReviewStateError(
            "provenance_uri must be an absolute HTTPS URL without "
            "credentials or a fragment"
        )
    return normalized


def _canonical_structure(
    structure_smiles: str,
    molecule_id: str,
) -> tuple[str, str]:
    raw = _normalize_structure(structure_smiles)
    if not raw:
        raise ReviewStateError("structure_smiles must not be blank")
    molecule = Chem.MolFromSmiles(raw)
    if molecule is None:
        raise ReviewStateError("structure_smiles is not parseable")
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise ReviewStateError(
            "structure_smiles must contain exactly one connected fragment"
        )
    canonical = str(
        Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
        or ""
    )
    inchi_key = str(Chem.MolToInchiKey(molecule) or "").upper()
    supplied_id = _required_text("molecule_id", molecule_id).upper()
    if not canonical or not INCHI_KEY_PATTERN.fullmatch(inchi_key):
        raise ReviewStateError(
            "Unable to derive a canonical structure identity"
        )
    if not INCHI_KEY_PATTERN.fullmatch(supplied_id):
        raise ReviewStateError("molecule_id must be a valid InChIKey")
    if supplied_id != inchi_key:
        raise ReviewStateError(
            "molecule_id does not match structure_smiles"
        )
    return canonical, inchi_key


def _axis_direction(values: list[Any]) -> str:
    if len(values) < 2:
        return "unknown"
    numeric = [float(value) for value in values]
    deltas = [
        numeric[index + 1] - numeric[index]
        for index in range(len(numeric) - 1)
    ]
    if all(delta > 0 for delta in deltas):
        return "ascending"
    if all(delta < 0 for delta in deltas):
        return "descending"
    return "non_monotonic"


def _snapshot_from_row(
    row: sqlite3.Row,
    *,
    structure_smiles: str,
    structure_source: str,
    molecule_id: str,
) -> Snapshot:
    spectrum = json.loads(row["spectrum_json"])
    result = json.loads(row["result_json"]) if row["result_json"] else None
    spectrum_revision = int(row["spectrum_revision"] or 1)
    result_revision = int(row["result_revision"] or 0)
    spectrum_sha256 = _sha256(spectrum)
    result_sha256 = _sha256(result) if result is not None else None

    parameters = spectrum.get("parameters") or {}
    metadata = spectrum.get("metadata") or {}
    metadata_extra = metadata.get("extra") or {}
    x_data = spectrum.get("x_data") or []
    result_peaks = (result or {}).get("peaks")
    spectrum_peaks = spectrum.get("peaks") or []
    peaks = result_peaks if isinstance(result_peaks, list) else spectrum_peaks
    nucleus = str(
        parameters.get("nucleus")
        or metadata_extra.get("nucleus")
        or ""
    ).strip()
    solvent = str(
        metadata.get("solvent")
        or parameters.get("solvent")
        or metadata_extra.get("solvent")
        or ""
    ).strip()
    processing_source = parameters.get("processing_source") or {}
    source_checksum = str(
        processing_source.get("checksum_sha256")
        or parameters.get("source_checksum_sha256")
        or ""
    ).strip()
    facts = {
        "technique": str(spectrum.get("technique") or ""),
        "source_file": str(spectrum.get("source_file") or row["name"] or ""),
        "source_checksum_sha256": source_checksum or None,
        "nucleus": nucleus,
        "solvent": solvent,
        "axis_unit": str(spectrum.get("x_unit") or "").strip(),
        "axis_direction": _axis_direction(x_data),
        "point_count": len(x_data),
        "peak_count": len(peaks),
        "axis_min": min((float(value) for value in x_data), default=None),
        "axis_max": max((float(value) for value in x_data), default=None),
    }
    label = {
        "structure_smiles": _normalize_structure(structure_smiles),
        "structure_source": _normalize_text(structure_source),
        "molecule_id": _normalize_text(molecule_id),
    }
    snapshot_sha256 = _sha256(
        {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "spectrum_id": row["id"],
            "spectrum_revision": spectrum_revision,
            "result_revision": result_revision,
            "spectrum_sha256": spectrum_sha256,
            "result_sha256": result_sha256,
            "label": label,
        }
    )
    return Snapshot(
        snapshot_sha256=snapshot_sha256,
        spectrum_sha256=spectrum_sha256,
        result_sha256=result_sha256,
        spectrum_revision=spectrum_revision,
        result_revision=result_revision,
        facts=facts,
        spectrum=spectrum,
        result=result,
    )


class ReviewRepository:
    """SQLite-backed, append-audited spectrum review repository."""

    def __init__(self, store: PersistentStore):
        self.store = store
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = self.store._connect()
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self.store._lock, self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS spectrum_review_queue (
                    id TEXT PRIMARY KEY,
                    spectrum_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    queue_revision INTEGER NOT NULL DEFAULT 1,
                    structure_smiles TEXT NOT NULL,
                    structure_source TEXT NOT NULL,
                    molecule_id TEXT NOT NULL,
                    source_collection TEXT NOT NULL,
                    source_record_id TEXT NOT NULL,
                    independence_group TEXT NOT NULL,
                    license_id TEXT NOT NULL,
                    provenance_uri TEXT DEFAULT '',
                    rights_confirmed INTEGER NOT NULL,
                    snapshot_sha256 TEXT NOT NULL,
                    spectrum_sha256 TEXT NOT NULL,
                    result_sha256 TEXT,
                    spectrum_revision INTEGER NOT NULL,
                    result_revision INTEGER NOT NULL,
                    facts_json TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    final_decision TEXT,
                    final_reason TEXT DEFAULT '',
                    adjudicator_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(spectrum_id, cycle)
                );
                CREATE INDEX IF NOT EXISTS idx_review_queue_spectrum
                    ON spectrum_review_queue(spectrum_id, cycle DESC);
                CREATE INDEX IF NOT EXISTS idx_review_queue_status
                    ON spectrum_review_queue(status, updated_at);

                CREATE TABLE IF NOT EXISTS spectrum_review_submissions (
                    id TEXT PRIMARY KEY,
                    queue_id TEXT NOT NULL,
                    reviewer_id TEXT NOT NULL,
                    reviewer_key TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    checks_json TEXT NOT NULL,
                    observations_json TEXT NOT NULL,
                    notes TEXT DEFAULT '',
                    snapshot_sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(queue_id, reviewer_key),
                    FOREIGN KEY(queue_id) REFERENCES spectrum_review_queue(id)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_review_submissions_queue
                    ON spectrum_review_submissions(queue_id, created_at);

                CREATE TABLE IF NOT EXISTS spectrum_review_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    queue_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    previous_revision INTEGER NOT NULL,
                    new_revision INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(queue_id) REFERENCES spectrum_review_queue(id)
                        ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_review_events_queue
                    ON spectrum_review_events(queue_id, id);

                CREATE TRIGGER IF NOT EXISTS trg_review_submission_no_update
                BEFORE UPDATE ON spectrum_review_submissions
                BEGIN
                    SELECT RAISE(ABORT, 'review submissions are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_review_submission_no_delete
                BEFORE DELETE ON spectrum_review_submissions
                BEGIN
                    SELECT RAISE(ABORT, 'review submissions are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_review_queue_snapshot_no_update
                BEFORE UPDATE OF
                    spectrum_id, cycle, structure_smiles, structure_source,
                    molecule_id, source_collection, source_record_id,
                    independence_group, license_id, provenance_uri,
                    rights_confirmed, snapshot_sha256, spectrum_sha256,
                    result_sha256, spectrum_revision, result_revision,
                    facts_json, created_by, created_at
                ON spectrum_review_queue
                BEGIN
                    SELECT RAISE(
                        ABORT,
                        'review queue snapshots are immutable'
                    );
                END;
                CREATE TRIGGER IF NOT EXISTS trg_review_queue_no_delete
                BEFORE DELETE ON spectrum_review_queue
                BEGIN
                    SELECT RAISE(ABORT, 'review queue cycles are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_review_event_no_update
                BEFORE UPDATE ON spectrum_review_events
                BEGIN
                    SELECT RAISE(ABORT, 'review events are immutable');
                END;
                CREATE TRIGGER IF NOT EXISTS trg_review_event_no_delete
                BEFORE DELETE ON spectrum_review_events
                BEGIN
                    SELECT RAISE(ABORT, 'review events are immutable');
                END;

                CREATE TRIGGER IF NOT EXISTS trg_spectrum_review_stale_update
                AFTER UPDATE OF spectrum_json, result_json,
                                spectrum_revision, result_revision
                ON spectra
                BEGIN
                    INSERT INTO spectrum_review_events
                    (queue_id, event_type, actor_id, details_json,
                     previous_revision, new_revision, created_at)
                    SELECT id, 'snapshot_stale', 'system',
                           '{"reason":"spectrum_or_result_revision_changed"}',
                           queue_revision, queue_revision + 1,
                           strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    FROM spectrum_review_queue
                    WHERE spectrum_id=NEW.id AND status!='stale';

                    UPDATE spectrum_review_queue
                    SET status='stale',
                        queue_revision=queue_revision + 1,
                        final_decision=NULL,
                        final_reason='',
                        updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE spectrum_id=NEW.id AND status!='stale';
                END;

                CREATE TRIGGER IF NOT EXISTS trg_spectrum_review_stale_delete
                BEFORE DELETE ON spectra
                BEGIN
                    INSERT INTO spectrum_review_events
                    (queue_id, event_type, actor_id, details_json,
                     previous_revision, new_revision, created_at)
                    SELECT id, 'snapshot_stale', 'system',
                           '{"reason":"spectrum_deleted"}',
                           queue_revision, queue_revision + 1,
                           strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    FROM spectrum_review_queue
                    WHERE spectrum_id=OLD.id AND status!='stale';

                    UPDATE spectrum_review_queue
                    SET status='stale',
                        queue_revision=queue_revision + 1,
                        final_decision=NULL,
                        final_reason='',
                        updated_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                    WHERE spectrum_id=OLD.id AND status!='stale';
                END;
                """
            )
            conn.commit()

    @staticmethod
    def _spectrum_row(conn: sqlite3.Connection, spectrum_id: str) -> sqlite3.Row:
        row = conn.execute(
            """
            SELECT id, technique, name, spectrum_json, result_json,
                   spectrum_revision, result_revision
            FROM spectra
            WHERE id=?
            """,
            (spectrum_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"Spectrum {spectrum_id} not found")
        return row

    @staticmethod
    def _event(
        conn: sqlite3.Connection,
        *,
        queue_id: str,
        event_type: str,
        actor_id: str,
        details: dict[str, Any],
        previous_revision: int,
        new_revision: int,
    ) -> None:
        conn.execute(
            """
            INSERT INTO spectrum_review_events
            (queue_id, event_type, actor_id, details_json,
             previous_revision, new_revision, created_at)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                queue_id,
                event_type,
                actor_id,
                _canonical_json(details),
                previous_revision,
                new_revision,
                _utc_now(),
            ),
        )

    @staticmethod
    def _latest_queue(
        conn: sqlite3.Connection,
        spectrum_id: str,
    ) -> sqlite3.Row | None:
        return conn.execute(
            """
            SELECT * FROM spectrum_review_queue
            WHERE spectrum_id=?
            ORDER BY cycle DESC
            LIMIT 1
            """,
            (spectrum_id,),
        ).fetchone()

    def _current_snapshot(
        self,
        conn: sqlite3.Connection,
        queue: sqlite3.Row,
    ) -> Snapshot:
        spectrum = self._spectrum_row(conn, queue["spectrum_id"])
        return _snapshot_from_row(
            spectrum,
            structure_smiles=queue["structure_smiles"],
            structure_source=queue["structure_source"],
            molecule_id=queue["molecule_id"],
        )

    def _mark_stale_if_needed(
        self,
        conn: sqlite3.Connection,
        queue: sqlite3.Row,
    ) -> sqlite3.Row:
        try:
            current = self._current_snapshot(conn, queue)
            stale_reason = (
                "spectrum_or_result_revision_changed"
                if current.snapshot_sha256 != queue["snapshot_sha256"]
                else ""
            )
        except KeyError:
            stale_reason = "spectrum_deleted"
        if not stale_reason or queue["status"] == "stale":
            return queue

        previous_revision = int(queue["queue_revision"])
        new_revision = previous_revision + 1
        now = _utc_now()
        conn.execute(
            """
            UPDATE spectrum_review_queue
            SET status='stale', queue_revision=?, updated_at=?,
                final_decision=NULL, final_reason=''
            WHERE id=?
            """,
            (new_revision, now, queue["id"]),
        )
        self._event(
            conn,
            queue_id=queue["id"],
            event_type="snapshot_stale",
            actor_id="system",
            details={"reason": stale_reason},
            previous_revision=previous_revision,
            new_revision=new_revision,
        )
        return conn.execute(
            "SELECT * FROM spectrum_review_queue WHERE id=?",
            (queue["id"],),
        ).fetchone()

    @staticmethod
    def _review_count(conn: sqlite3.Connection, queue_id: str) -> int:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM spectrum_review_submissions
            WHERE queue_id=?
            """,
            (queue_id,),
        ).fetchone()
        return int(row["count"])

    @classmethod
    def _public_queue(
        cls,
        conn: sqlite3.Connection,
        queue: sqlite3.Row,
        *,
        reviewer_id: str | None = None,
    ) -> dict[str, Any]:
        own_review = None
        if reviewer_id:
            own = conn.execute(
                """
                SELECT id, verdict, checks_json, observations_json, notes,
                       snapshot_sha256, created_at
                FROM spectrum_review_submissions
                WHERE queue_id=? AND reviewer_key=?
                """,
                (queue["id"], reviewer_id.casefold()),
            ).fetchone()
            if own:
                own_review = {
                    "id": own["id"],
                    "verdict": own["verdict"],
                    "checks": json.loads(own["checks_json"]),
                    "observations": json.loads(own["observations_json"]),
                    "notes": own["notes"],
                    "snapshot_sha256": own["snapshot_sha256"],
                    "created_at": own["created_at"],
                }
        return {
            "schema_version": REVIEW_SCHEMA_VERSION,
            "id": queue["id"],
            "spectrum_id": queue["spectrum_id"],
            "cycle": int(queue["cycle"]),
            "status": queue["status"],
            "queue_revision": int(queue["queue_revision"]),
            "structure_smiles": queue["structure_smiles"],
            "structure_source": queue["structure_source"],
            "molecule_id": queue["molecule_id"],
            "source_collection": queue["source_collection"],
            "source_record_id": queue["source_record_id"],
            "independence_group": queue["independence_group"],
            "license_id": queue["license_id"],
            "provenance_uri": queue["provenance_uri"],
            "rights_confirmed": bool(queue["rights_confirmed"]),
            "snapshot_sha256": queue["snapshot_sha256"],
            "spectrum_sha256": queue["spectrum_sha256"],
            "result_sha256": queue["result_sha256"],
            "spectrum_revision": int(queue["spectrum_revision"]),
            "result_revision": int(queue["result_revision"]),
            "facts": json.loads(queue["facts_json"]),
            "review_count": cls._review_count(conn, queue["id"]),
            "has_submitted": own_review is not None,
            "own_review": own_review,
            "final_decision": queue["final_decision"],
            "final_reason": queue["final_reason"],
            "created_at": queue["created_at"],
            "updated_at": queue["updated_at"],
        }

    def enqueue(
        self,
        *,
        spectrum_id: str,
        structure_smiles: str,
        structure_source: str,
        molecule_id: str,
        source_collection: str,
        source_record_id: str,
        independence_group: str,
        license_id: str,
        provenance_uri: str,
        rights_confirmed: bool,
        expected_spectrum_revision: int,
        expected_result_revision: int,
        actor_id: str,
    ) -> dict[str, Any]:
        if not rights_confirmed:
            raise ReviewStateError(
                "Data rights must be confirmed before entering the review queue"
            )
        if not SUBJECT_PATTERN.fullmatch(actor_id):
            raise ReviewStateError(
                "Review administrator must have a stable pseudonymous subject"
            )
        canonical_smiles, canonical_molecule_id = _canonical_structure(
            structure_smiles,
            molecule_id,
        )
        normalized_structure_source = _required_text(
            "structure_source",
            structure_source,
        )
        normalized_source_collection = _required_text(
            "source_collection",
            source_collection,
        )
        normalized_source_record_id = _required_text(
            "source_record_id",
            source_record_id,
        )
        normalized_independence_group = _required_text(
            "independence_group",
            independence_group,
        )
        normalized_license = _validated_license(license_id)
        normalized_provenance = _validated_provenance_uri(provenance_uri)
        with self.store._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            spectrum = self._spectrum_row(conn, spectrum_id)
            if spectrum["technique"] != "NMR":
                conn.rollback()
                raise ReviewStateError("Only NMR spectra can enter this review queue")
            current_spectrum_revision = int(spectrum["spectrum_revision"] or 1)
            current_result_revision = int(spectrum["result_revision"] or 0)
            if (
                current_spectrum_revision != expected_spectrum_revision
                or current_result_revision != expected_result_revision
            ):
                conn.rollback()
                raise ReviewConflict(
                    current_spectrum_revision,
                    "Spectrum or result revision changed before enqueue",
                )

            previous = self._latest_queue(conn, spectrum_id)
            cycle = 1
            if previous:
                previous = self._mark_stale_if_needed(conn, previous)
                if previous["status"] != "stale":
                    conn.rollback()
                    raise ReviewStateError(
                        "The current spectrum snapshot is already in the review queue"
                    )
                cycle = int(previous["cycle"]) + 1

            snapshot = _snapshot_from_row(
                spectrum,
                structure_smiles=canonical_smiles,
                structure_source=normalized_structure_source,
                molecule_id=canonical_molecule_id,
            )
            duplicate = conn.execute(
                """
                SELECT spectrum_id, source_collection, source_record_id
                FROM spectrum_review_queue
                WHERE spectrum_id!=?
                  AND (
                    spectrum_sha256=?
                    OR (
                        source_collection=?
                        AND source_record_id=?
                    )
                  )
                LIMIT 1
                """,
                (
                    spectrum_id,
                    snapshot.spectrum_sha256,
                    normalized_source_collection,
                    normalized_source_record_id,
                ),
            ).fetchone()
            if duplicate:
                conn.rollback()
                raise ReviewStateError(
                    "An identical spectrum or source record is already present "
                    f"under spectrum {duplicate['spectrum_id']}"
                )
            queue_id = uuid.uuid4().hex
            now = _utc_now()
            conn.execute(
                """
                INSERT INTO spectrum_review_queue
                (id, spectrum_id, cycle, status, queue_revision,
                 structure_smiles, structure_source, molecule_id,
                 source_collection, source_record_id, independence_group,
                 license_id, provenance_uri, rights_confirmed,
                 snapshot_sha256, spectrum_sha256, result_sha256,
                 spectrum_revision, result_revision, facts_json, created_by,
                 created_at, updated_at)
                VALUES (?,?,?,'pending',1,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    queue_id,
                    spectrum_id,
                    cycle,
                    canonical_smiles,
                    normalized_structure_source,
                    canonical_molecule_id,
                    normalized_source_collection,
                    normalized_source_record_id,
                    normalized_independence_group,
                    normalized_license,
                    normalized_provenance,
                    1,
                    snapshot.snapshot_sha256,
                    snapshot.spectrum_sha256,
                    snapshot.result_sha256,
                    snapshot.spectrum_revision,
                    snapshot.result_revision,
                    _canonical_json(snapshot.facts),
                    actor_id,
                    now,
                    now,
                ),
            )
            self._event(
                conn,
                queue_id=queue_id,
                event_type="enqueued",
                actor_id=actor_id,
                details={
                    "snapshot_sha256": snapshot.snapshot_sha256,
                    "spectrum_revision": snapshot.spectrum_revision,
                    "result_revision": snapshot.result_revision,
                    "rights_confirmed": True,
                },
                previous_revision=0,
                new_revision=1,
            )
            conn.commit()
            queue = conn.execute(
                "SELECT * FROM spectrum_review_queue WHERE id=?",
                (queue_id,),
            ).fetchone()
            return self._public_queue(conn, queue)

    def get_latest(
        self,
        spectrum_id: str,
        *,
        reviewer_id: str | None = None,
    ) -> dict[str, Any] | None:
        with self.store._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            queue = self._latest_queue(conn, spectrum_id)
            if queue is None:
                conn.rollback()
                return None
            queue = self._mark_stale_if_needed(conn, queue)
            conn.commit()
            return self._public_queue(
                conn,
                queue,
                reviewer_id=reviewer_id,
            )

    def list_queue(
        self,
        *,
        status: str | None = None,
        reviewer_id: str | None = None,
        include_history: bool = False,
    ) -> list[dict[str, Any]]:
        if status and status not in QUEUE_STATES:
            raise ReviewStateError(f"Unsupported review status: {status}")
        with self.store._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if include_history:
                rows = conn.execute(
                    """
                    SELECT * FROM spectrum_review_queue
                    ORDER BY updated_at ASC, spectrum_id ASC, cycle ASC
                    """
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT q.*
                    FROM spectrum_review_queue AS q
                    JOIN (
                        SELECT spectrum_id, MAX(cycle) AS cycle
                        FROM spectrum_review_queue
                        GROUP BY spectrum_id
                    ) AS latest
                    ON latest.spectrum_id=q.spectrum_id
                       AND latest.cycle=q.cycle
                    ORDER BY q.updated_at ASC, q.spectrum_id ASC
                    """
                ).fetchall()
            current_rows = [
                self._mark_stale_if_needed(conn, row)
                for row in rows
            ]
            conn.commit()
            return [
                self._public_queue(conn, row, reviewer_id=reviewer_id)
                for row in current_rows
                if status is None or row["status"] == status
            ]

    @staticmethod
    def _expected_observations(queue: sqlite3.Row) -> dict[str, Any]:
        facts = json.loads(queue["facts_json"])
        return {
            "structure_smiles": _normalize_structure(
                queue["structure_smiles"]
            ),
            "nucleus": _normalize_text(str(facts["nucleus"])).casefold(),
            "axis_unit": _normalize_text(
                str(facts["axis_unit"])
            ).casefold(),
            "axis_direction": str(facts["axis_direction"]),
            "solvent": _normalize_text(str(facts["solvent"])).casefold(),
            "peak_count": int(facts["peak_count"]),
        }

    @staticmethod
    def _normalized_observations(
        observations: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "structure_smiles": _normalize_structure(
                str(observations.get("structure_smiles") or "")
            ),
            "nucleus": _normalize_text(
                str(observations.get("nucleus") or "")
            ).casefold(),
            "axis_unit": _normalize_text(
                str(observations.get("axis_unit") or "")
            ).casefold(),
            "axis_direction": str(
                observations.get("axis_direction") or ""
            ),
            "solvent": _normalize_text(
                str(observations.get("solvent") or "")
            ).casefold(),
            "peak_count": int(observations.get("peak_count") or 0),
        }

    @classmethod
    def _validate_mechanical_facts(cls, queue: sqlite3.Row) -> None:
        canonical_smiles, inchi_key = _canonical_structure(
            queue["structure_smiles"],
            queue["molecule_id"],
        )
        if (
            canonical_smiles != queue["structure_smiles"]
            or inchi_key != queue["molecule_id"]
        ):
            raise ReviewStateError(
                "The frozen structure identity is not canonical"
            )
        for field in (
            "structure_source",
            "source_collection",
            "source_record_id",
            "independence_group",
        ):
            _required_text(field, str(queue[field]))
        _validated_license(str(queue["license_id"]))
        _validated_provenance_uri(str(queue["provenance_uri"]))
        if not bool(queue["rights_confirmed"]):
            raise ReviewStateError("The frozen data rights are not confirmed")

        facts = json.loads(queue["facts_json"])
        expected = cls._expected_observations(queue)
        if str(facts.get("technique") or "") != "NMR":
            raise ReviewStateError("Only NMR facts can be accepted")
        if not all(
            (
                expected["structure_smiles"],
                expected["nucleus"],
                expected["axis_unit"],
                expected["solvent"],
            )
        ):
            raise ReviewStateError(
                "Missing structure, nucleus, axis unit, or solvent metadata "
                "must be corrected before this spectrum can be accepted"
            )
        if (
            expected["axis_unit"] != "ppm"
            or expected["axis_direction"] not in {"ascending", "descending"}
            or expected["peak_count"] <= 0
            or int(facts.get("point_count") or 0) < 2
        ):
            raise ReviewStateError(
                "Accepted NMR spectra require a monotonic ppm axis, at least "
                "two points, and at least one reviewed peak"
            )

    @classmethod
    def _validate_review(
        cls,
        *,
        verdict: str,
        checks: dict[str, str],
        observations: dict[str, Any],
        queue: sqlite3.Row,
    ) -> None:
        if set(checks) != set(CHECK_NAMES):
            raise ReviewStateError(
                "Exactly structure, nucleus, axis, peaks, and solvent checks "
                "are required"
            )
        if any(value not in CHECK_VALUES for value in checks.values()):
            raise ReviewStateError("Review checks must be pass, fail, or uncertain")
        if verdict not in {"accept", "reject"}:
            raise ReviewStateError("Review verdict must be accept or reject")
        if verdict != "accept":
            return
        if any(value != "pass" for value in checks.values()):
            raise ReviewStateError("An accepted review requires all checks to pass")
        cls._validate_mechanical_facts(queue)
        expected = cls._expected_observations(queue)
        observed = cls._normalized_observations(observations)
        if observed != expected:
            raise ReviewStateError(
                "Accepted observations must match the immutable queue snapshot"
            )

    @staticmethod
    def _review_signature(row: sqlite3.Row) -> str:
        observations = json.loads(row["observations_json"])
        normalized = {
            "verdict": row["verdict"],
            "checks": json.loads(row["checks_json"]),
            "observations": ReviewRepository._normalized_observations(
                observations
            ),
        }
        return _sha256(normalized)

    def submit(
        self,
        *,
        spectrum_id: str,
        reviewer_id: str,
        verdict: Literal["accept", "reject"],
        checks: dict[str, str],
        observations: dict[str, Any],
        notes: str,
        expected_queue_revision: int,
        expected_snapshot_sha256: str,
    ) -> dict[str, Any]:
        if not SUBJECT_PATTERN.fullmatch(reviewer_id):
            raise ReviewStateError(
                "Reviewer identity must be a stable pseudonymous subject"
            )
        with self.store._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            queue = self._latest_queue(conn, spectrum_id)
            if queue is None:
                conn.rollback()
                raise KeyError(f"Spectrum {spectrum_id} is not queued")
            queue = self._mark_stale_if_needed(conn, queue)
            if queue["status"] == "stale":
                conn.commit()
                raise ReviewStateError(
                    "The queued snapshot is stale and must be re-enqueued"
                )
            current_revision = int(queue["queue_revision"])
            if current_revision != expected_queue_revision:
                conn.rollback()
                raise ReviewConflict(current_revision)
            if queue["snapshot_sha256"] != expected_snapshot_sha256:
                conn.rollback()
                raise ReviewConflict(
                    current_revision,
                    "Review snapshot hash changed",
                )
            if reviewer_id.casefold() == str(queue["created_by"]).casefold():
                conn.rollback()
                raise ReviewStateError(
                    "The review administrator cannot act as a reviewer"
                )
            if queue["status"] not in {"pending", "awaiting_second"}:
                conn.rollback()
                raise ReviewStateError(
                    f"Reviews cannot be submitted while status is {queue['status']}"
                )
            existing = conn.execute(
                """
                SELECT 1 FROM spectrum_review_submissions
                WHERE queue_id=? AND reviewer_key=?
                """,
                (queue["id"], reviewer_id.casefold()),
            ).fetchone()
            if existing:
                conn.rollback()
                raise ReviewStateError(
                    "A reviewer may submit only once for a queue cycle"
                )

            self._validate_review(
                verdict=verdict,
                checks=checks,
                observations=observations,
                queue=queue,
            )
            review_id = uuid.uuid4().hex
            now = _utc_now()
            conn.execute(
                """
                INSERT INTO spectrum_review_submissions
                (id, queue_id, reviewer_id, reviewer_key, verdict, checks_json,
                 observations_json, notes, snapshot_sha256, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    review_id,
                    queue["id"],
                    reviewer_id,
                    reviewer_id.casefold(),
                    verdict,
                    _canonical_json(checks),
                    _canonical_json(observations),
                    notes.strip(),
                    queue["snapshot_sha256"],
                    now,
                ),
            )
            reviews = conn.execute(
                """
                SELECT * FROM spectrum_review_submissions
                WHERE queue_id=?
                ORDER BY created_at ASC, id ASC
                """,
                (queue["id"],),
            ).fetchall()
            if len(reviews) == 1:
                next_status = "awaiting_second"
                final_decision = None
                final_reason = ""
            else:
                signatures = {self._review_signature(row) for row in reviews}
                if len(signatures) == 1:
                    final_decision = reviews[0]["verdict"]
                    next_status = (
                        "accepted"
                        if final_decision == "accept"
                        else "rejected"
                    )
                    final_reason = "independent_reviews_agree"
                else:
                    next_status = "conflict"
                    final_decision = None
                    final_reason = "independent_reviews_disagree"

            new_revision = current_revision + 1
            conn.execute(
                """
                UPDATE spectrum_review_queue
                SET status=?, queue_revision=?, final_decision=?,
                    final_reason=?, updated_at=?
                WHERE id=?
                """,
                (
                    next_status,
                    new_revision,
                    final_decision,
                    final_reason,
                    now,
                    queue["id"],
                ),
            )
            self._event(
                conn,
                queue_id=queue["id"],
                event_type="review_submitted",
                actor_id=reviewer_id,
                details={
                    "review_id": review_id,
                    "verdict": verdict,
                    "checks": checks,
                    "resulting_status": next_status,
                },
                previous_revision=current_revision,
                new_revision=new_revision,
            )
            conn.commit()
            updated = conn.execute(
                "SELECT * FROM spectrum_review_queue WHERE id=?",
                (queue["id"],),
            ).fetchone()
            return self._public_queue(
                conn,
                updated,
                reviewer_id=reviewer_id,
            )

    def adjudicate(
        self,
        *,
        spectrum_id: str,
        decision: Literal["accept", "reject"],
        checks: dict[str, str],
        reason: str,
        expected_queue_revision: int,
        expected_snapshot_sha256: str,
        actor_id: str,
    ) -> dict[str, Any]:
        if not SUBJECT_PATTERN.fullmatch(actor_id):
            raise ReviewStateError(
                "Adjudicator identity must be a stable pseudonymous subject"
            )
        normalized_reason = _required_text("adjudication reason", reason)
        if decision not in {"accept", "reject"}:
            raise ReviewStateError(
                "Adjudication decision must be accept or reject"
            )
        with self.store._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            queue = self._latest_queue(conn, spectrum_id)
            if queue is None:
                conn.rollback()
                raise KeyError(f"Spectrum {spectrum_id} is not queued")
            queue = self._mark_stale_if_needed(conn, queue)
            current_revision = int(queue["queue_revision"])
            if current_revision != expected_queue_revision:
                conn.rollback()
                raise ReviewConflict(current_revision)
            if queue["snapshot_sha256"] != expected_snapshot_sha256:
                conn.rollback()
                raise ReviewConflict(
                    current_revision,
                    "Review snapshot hash changed",
                )
            if queue["status"] != "conflict":
                conn.rollback()
                raise ReviewStateError(
                    "Only a conflicting pair of reviews can be adjudicated"
                )
            if set(checks) != set(CHECK_NAMES):
                conn.rollback()
                raise ReviewStateError("A complete adjudication checklist is required")
            if any(value not in CHECK_VALUES for value in checks.values()):
                conn.rollback()
                raise ReviewStateError(
                    "Adjudication checks must be pass, fail, or uncertain"
                )
            if decision == "accept" and any(
                value != "pass" for value in checks.values()
            ):
                conn.rollback()
                raise ReviewStateError(
                    "An accepted adjudication requires all checks to pass"
                )
            if decision == "accept":
                try:
                    self._validate_mechanical_facts(queue)
                except ReviewStateError:
                    conn.rollback()
                    raise

            reviewer_rows = conn.execute(
                    """
                    SELECT reviewer_key FROM spectrum_review_submissions
                    WHERE queue_id=?
                    """,
                    (queue["id"],),
                ).fetchall()
            reviewer_ids = {
                row["reviewer_key"]
                for row in reviewer_rows
            }
            if len(reviewer_rows) != 2 or len(reviewer_ids) != 2:
                conn.rollback()
                raise ReviewStateError(
                    "Adjudication requires exactly two independent reviews"
                )
            if actor_id.casefold() in reviewer_ids:
                conn.rollback()
                raise ReviewStateError(
                    "The adjudicator must be independent of both reviewers"
                )
            new_revision = current_revision + 1
            now = _utc_now()
            next_status = "accepted" if decision == "accept" else "rejected"
            conn.execute(
                """
                UPDATE spectrum_review_queue
                SET status=?, queue_revision=?, final_decision=?,
                    final_reason=?, adjudicator_id=?, updated_at=?
                WHERE id=?
                """,
                (
                    next_status,
                    new_revision,
                    decision,
                    normalized_reason,
                    actor_id,
                    now,
                    queue["id"],
                ),
            )
            self._event(
                conn,
                queue_id=queue["id"],
                event_type="conflict_adjudicated",
                actor_id=actor_id,
                details={
                    "decision": decision,
                    "checks": checks,
                    "reason": normalized_reason,
                    "resulting_status": next_status,
                },
                previous_revision=current_revision,
                new_revision=new_revision,
            )
            conn.commit()
            updated = conn.execute(
                "SELECT * FROM spectrum_review_queue WHERE id=?",
                (queue["id"],),
            ).fetchone()
            return self._public_queue(conn, updated)

    def audit(self, spectrum_id: str) -> dict[str, Any] | None:
        with self.store._lock, self._connect() as conn:
            queue = self._latest_queue(conn, spectrum_id)
            if queue is None:
                return None
            reviews = conn.execute(
                """
                SELECT id, reviewer_id, verdict, checks_json, observations_json,
                       notes, snapshot_sha256, created_at
                FROM spectrum_review_submissions
                WHERE queue_id=?
                ORDER BY created_at ASC, id ASC
                """,
                (queue["id"],),
            ).fetchall()
            events = conn.execute(
                """
                SELECT id, event_type, actor_id, details_json,
                       previous_revision, new_revision, created_at
                FROM spectrum_review_events
                WHERE queue_id=?
                ORDER BY id ASC
                """,
                (queue["id"],),
            ).fetchall()
            return {
                "queue": self._public_queue(conn, queue),
                "reviews": [
                    {
                        "id": row["id"],
                        "reviewer_id": row["reviewer_id"],
                        "verdict": row["verdict"],
                        "checks": json.loads(row["checks_json"]),
                        "observations": json.loads(row["observations_json"]),
                        "notes": row["notes"],
                        "snapshot_sha256": row["snapshot_sha256"],
                        "created_at": row["created_at"],
                    }
                    for row in reviews
                ],
                "events": [
                    {
                        "id": int(row["id"]),
                        "event_type": row["event_type"],
                        "actor_id": row["actor_id"],
                        "details": json.loads(row["details_json"]),
                        "previous_revision": int(row["previous_revision"]),
                        "new_revision": int(row["new_revision"]),
                        "created_at": row["created_at"],
                    }
                    for row in events
                ],
            }

    @classmethod
    def _gold_record_is_valid(
        cls,
        queue: sqlite3.Row,
        reviews: list[sqlite3.Row],
        events: list[sqlite3.Row],
        *,
        current_snapshot: Snapshot,
    ) -> bool:
        try:
            if (
                queue["status"] != "accepted"
                or queue["final_decision"] != "accept"
                or not SUBJECT_PATTERN.fullmatch(str(queue["created_by"]))
            ):
                return False
            if (
                current_snapshot.snapshot_sha256
                != queue["snapshot_sha256"]
                or current_snapshot.spectrum_sha256
                != queue["spectrum_sha256"]
                or current_snapshot.result_sha256
                != queue["result_sha256"]
                or current_snapshot.spectrum_revision
                != int(queue["spectrum_revision"])
                or current_snapshot.result_revision
                != int(queue["result_revision"])
                or current_snapshot.facts
                != json.loads(queue["facts_json"])
            ):
                return False
            cls._validate_mechanical_facts(queue)
            if len(reviews) != 2:
                return False
            reviewer_keys = {
                str(review["reviewer_key"]).casefold()
                for review in reviews
            }
            if (
                len(reviewer_keys) != 2
                or str(queue["created_by"]).casefold() in reviewer_keys
                or any(
                    not SUBJECT_PATTERN.fullmatch(
                        str(review["reviewer_id"])
                    )
                    for review in reviews
                )
                or any(
                    review["snapshot_sha256"] != queue["snapshot_sha256"]
                    for review in reviews
                )
            ):
                return False

            for review in reviews:
                cls._validate_review(
                    verdict=str(review["verdict"]),
                    checks=json.loads(review["checks_json"]),
                    observations=json.loads(review["observations_json"]),
                    queue=queue,
                )

            if not events or events[0]["event_type"] != "enqueued":
                return False
            expected_revision = 0
            for event in events:
                previous = int(event["previous_revision"])
                current = int(event["new_revision"])
                if previous != expected_revision or current != previous + 1:
                    return False
                expected_revision = current
            if expected_revision != int(queue["queue_revision"]):
                return False

            enqueued_details = json.loads(events[0]["details_json"])
            if (
                events[0]["actor_id"] != queue["created_by"]
                or enqueued_details.get("snapshot_sha256")
                != queue["snapshot_sha256"]
                or enqueued_details.get("spectrum_revision")
                != int(queue["spectrum_revision"])
                or enqueued_details.get("result_revision")
                != int(queue["result_revision"])
                or enqueued_details.get("rights_confirmed") is not True
            ):
                return False

            review_events = [
                event
                for event in events
                if event["event_type"] == "review_submitted"
            ]
            if len(review_events) != 2:
                return False
            reviews_by_id = {review["id"]: review for review in reviews}
            seen_review_ids: set[str] = set()
            for index, event in enumerate(review_events):
                details = json.loads(event["details_json"])
                review_id = str(details.get("review_id") or "")
                review = reviews_by_id.get(review_id)
                if (
                    review is None
                    or review_id in seen_review_ids
                    or event["actor_id"] != review["reviewer_id"]
                    or details.get("verdict") != review["verdict"]
                    or details.get("checks")
                    != json.loads(review["checks_json"])
                ):
                    return False
                resulting_status = details.get("resulting_status")
                if (
                    index == 0
                    and resulting_status != "awaiting_second"
                ) or (
                    index == 1
                    and resulting_status
                    not in {"accepted", "rejected", "conflict"}
                ):
                    return False
                seen_review_ids.add(review_id)

            signatures = {cls._review_signature(review) for review in reviews}
            terminal_event = events[-1]
            if len(signatures) == 1:
                if (
                    len(events) != 3
                    or any(review["verdict"] != "accept" for review in reviews)
                    or review_events[-1]["id"] != terminal_event["id"]
                    or json.loads(terminal_event["details_json"]).get(
                        "resulting_status"
                    )
                    != "accepted"
                    or queue["final_reason"] != "independent_reviews_agree"
                    or queue["adjudicator_id"] is not None
                ):
                    return False
                return True

            if (
                len(events) != 4
                or json.loads(review_events[-1]["details_json"]).get(
                    "resulting_status"
                )
                != "conflict"
                or terminal_event["event_type"] != "conflict_adjudicated"
            ):
                return False
            adjudicator = str(queue["adjudicator_id"] or "")
            adjudication = json.loads(terminal_event["details_json"])
            if (
                not SUBJECT_PATTERN.fullmatch(adjudicator)
                or adjudicator.casefold() in reviewer_keys
                or terminal_event["actor_id"] != adjudicator
                or adjudication.get("decision") != "accept"
                or adjudication.get("resulting_status") != "accepted"
                or adjudication.get("reason") != queue["final_reason"]
                or set(adjudication.get("checks") or {}) != set(CHECK_NAMES)
                or any(
                    value != "pass"
                    for value in (adjudication.get("checks") or {}).values()
                )
            ):
                return False
            return True
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
            ReviewStateError,
        ):
            return False

    def gold_manifest(self) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        with self.store._lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows = conn.execute(
                """
                SELECT q.*
                FROM spectrum_review_queue AS q
                JOIN (
                    SELECT spectrum_id, MAX(cycle) AS cycle
                    FROM spectrum_review_queue
                    GROUP BY spectrum_id
                ) AS latest
                ON latest.spectrum_id=q.spectrum_id
                   AND latest.cycle=q.cycle
                ORDER BY q.spectrum_id ASC
                """
            ).fetchall()
            for row in rows:
                current = self._mark_stale_if_needed(conn, row)
                if current["status"] != "accepted":
                    continue
                try:
                    current_snapshot = self._current_snapshot(conn, current)
                except (KeyError, TypeError, ValueError):
                    continue
                review_rows = conn.execute(
                    """
                    SELECT *
                    FROM spectrum_review_submissions
                    WHERE queue_id=?
                    ORDER BY created_at ASC, id ASC
                    """,
                    (current["id"],),
                ).fetchall()
                event_rows = conn.execute(
                    """
                    SELECT *
                    FROM spectrum_review_events
                    WHERE queue_id=?
                    ORDER BY id ASC
                    """,
                    (current["id"],),
                ).fetchall()
                if not self._gold_record_is_valid(
                    current,
                    list(review_rows),
                    list(event_rows),
                    current_snapshot=current_snapshot,
                ):
                    continue
                facts = json.loads(current["facts_json"])
                item = {
                    "spectrum_id": current["spectrum_id"],
                    "review_cycle": int(current["cycle"]),
                    "snapshot_sha256": current["snapshot_sha256"],
                    "spectrum_sha256": current["spectrum_sha256"],
                    "result_sha256": current["result_sha256"],
                    "spectrum_revision": int(current["spectrum_revision"]),
                    "result_revision": int(current["result_revision"]),
                    "structure_smiles": current["structure_smiles"],
                    "structure_source": current["structure_source"],
                    "molecule_id": current["molecule_id"],
                    "source_collection": current["source_collection"],
                    "source_record_id": current["source_record_id"],
                    "independence_group": current["independence_group"],
                    "license_id": current["license_id"],
                    "provenance_uri": current["provenance_uri"],
                    "rights_confirmed": bool(current["rights_confirmed"]),
                    "facts": facts,
                    "review": {
                        "decision": current["final_decision"],
                        "reason": current["final_reason"],
                        "review_count": len(review_rows),
                        "review_ids": [entry["id"] for entry in review_rows],
                        "reviewers": [
                            entry["reviewer_id"] for entry in review_rows
                        ],
                        "adjudicator_id": current["adjudicator_id"],
                    },
                    "data_locator": (
                        f"chemapp://spectra/{current['spectrum_id']}"
                        f"@{current['spectrum_revision']}"
                    ),
                }
                item["gold_item_sha256"] = _sha256(item)
                items.append(item)
            conn.commit()

        stable = {
            "schema_version": GOLD_MANIFEST_SCHEMA_VERSION,
            "intended_use": "calibration_candidate_pool",
            "benchmark_eligible": False,
            "requires_group_leakage_audit": True,
            "item_count": len(items),
            "items": items,
        }
        return {
            **stable,
            "generated_at": _utc_now(),
            "manifest_sha256": _sha256(stable),
        }
