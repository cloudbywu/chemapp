"""Append-only, hash-chained ledgers for sealed-test attempts and denylists.

The attempt ledger makes reservation consumption exactly-once: every entry is
terminal, entries are keyed by (reservation_sha256, attempt_id), and a
concurrent or repeated consume is rejected before any query bundle may be
loaded.  The denylist is a permanent, holder-private record of every consumed
source identity so no sealed cohort can be reused as calibration or training
data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .crypto import canonical_object_bytes, object_sha256

ZERO_HASH = "0" * 64
MAX_LEDGER_BYTES = 512 * 1024 * 1024


class LedgerError(ValueError):
    """Raised when a ledger invariant is violated."""


def _strict_loads(line: bytes, location: str) -> dict[str, Any]:
    try:
        value = json.loads(line.decode("utf-8"), object_pairs_hook=_reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LedgerError(f"{location}: invalid JSON line: {exc}") from exc
    if not isinstance(value, dict):
        raise LedgerError(f"{location}: ledger entry must be an object")
    return value


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise LedgerError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


class _FileLedger:
    """Hash-chained JSONL ledger with an atomic, lock-protected append."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def _ensure_parent(self) -> None:
        parent = self.path.parent
        if not parent.exists():
            parent.mkdir(parents=True, exist_ok=True)

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        payload = self.path.read_bytes()
        if len(payload) > MAX_LEDGER_BYTES:
            raise LedgerError("ledger is too large")
        text = payload.decode("utf-8")
        if not text or not text.endswith("\n"):
            raise LedgerError("ledger is truncated or missing trailing newline")
        entries: list[dict[str, Any]] = []
        previous = ZERO_HASH
        for number, line in enumerate(text.splitlines(), start=1):
            if not line:
                raise LedgerError(f"ledger:{number}: blank line is forbidden")
            entry = _strict_loads(line.encode("utf-8"), f"ledger:{number}")
            self._validate_entry(entry, previous=previous, location=f"ledger:{number}")
            if canonical_object_bytes(entry) != (line + "\n").encode("utf-8"):
                raise LedgerError(f"ledger:{number}: entry is not canonical JSON")
            previous = entry["entry_sha256"]
            entries.append(entry)
        return entries

    def append(self, entry: dict[str, Any]) -> dict[str, Any]:
        with self._lock():
            entries = self.read()
            previous = entries[-1]["entry_sha256"] if entries else ZERO_HASH
            entry = {**entry, "previous_sha256": previous}
            body = {key: item for key, item in entry.items() if key != "entry_sha256"}
            entry["entry_sha256"] = object_sha256(body)
            self._validate_entry(entry, previous=previous, location="new entry")
            for existing in entries:
                if (
                    existing.get("reservation_sha256") == entry.get("reservation_sha256")
                    and existing.get("attempt_id") == entry.get("attempt_id")
                    and entry.get("event") == "attempt_consumed"
                ):
                    raise LedgerError("reservation/attempt was already consumed")
            line = canonical_object_bytes(entry)
            self._ensure_parent()
            with self.path.open("ab") as handle:
                handle.write(line)
                handle.flush()
            return entry

    def _lock(self):
        import contextlib

        @contextlib.contextmanager
        def manager():
            parent = self.path.parent
            self._ensure_parent()
            lock = parent / f".{self.path.name}.lock"
            try:
                lock.mkdir()
            except FileExistsError as exc:
                raise LedgerError(
                    f"ledger is locked; inspect stale lock {lock}"
                ) from exc
            try:
                yield
            finally:
                lock.rmdir()

        return manager()

    def _validate_entry(
        self, entry: dict[str, Any], *, previous: str, location: str
    ) -> None:
        required = {
            "schema_version",
            "event",
            "reservation_sha256",
            "attempt_id",
            "occurred_at_utc",
            "previous_sha256",
            "entry_sha256",
        }
        missing = required - set(entry)
        if missing:
            raise LedgerError(f"{location}: missing fields {sorted(missing)}")
        if entry["previous_sha256"] != previous:
            raise LedgerError(f"{location}: ledger chain is broken")
        for key in ("reservation_sha256", "entry_sha256", "previous_sha256"):
            if not isinstance(entry[key], str) or len(entry[key]) != 64:
                raise LedgerError(f"{location}: {key} must be a SHA-256 hex digest")
        if not isinstance(entry["attempt_id"], str) or not entry["attempt_id"].startswith(
            "att_"
        ):
            raise LedgerError(f"{location}: attempt_id must be an opaque att_ id")


class Ledger(_FileLedger):
    """Exactly-once attempt ledger for sealed-test reservations."""


class Denylist(_FileLedger):
    """Permanent record of consumed source identities.

    Entries carry ``identities`` (a list of strings such as record IDs,
    canonical SMILES or InChI keys) and ``reservation_sha256``/``attempt_id``.
    Any identity present here is permanently excluded from future calibration
    and sealed cohorts.
    """

    def check(self, identities: Sequence[str]) -> None:
        present: set[str] = set()
        for entry in self.read():
            for identity in entry.get("identities", []):
                present.add(identity)
        overlap = sorted(set(identities) & present)
        if overlap:
            raise LedgerError(
                "denylist overlap: identities already consumed: "
                + ", ".join(overlap[:5])
                + (f" (+{len(overlap)-5} more)" if len(overlap) > 5 else "")
            )

    def record(
        self,
        *,
        reservation_sha256: str,
        attempt_id: str,
        identities: Sequence[str],
        occurred_at_utc: str,
        holder_nonce_sha256: str,
    ) -> dict[str, Any]:
        entry = {
            "schema_version": "chemapp.nmr.v8-denylist-entry.v1",
            "event": "identities_consumed",
            "reservation_sha256": reservation_sha256,
            "attempt_id": attempt_id,
            "holder_nonce_sha256": holder_nonce_sha256,
            "identities": sorted(set(identities)),
            "occurred_at_utc": occurred_at_utc,
        }
        return self.append(entry)
