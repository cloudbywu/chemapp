"""Outcome-blind NMR candidate generation and holder-only coverage audit.

The module exposes three deliberately separate surfaces:

* :func:`build_open_world_bundle` enumerates a read-only NMR index using only
  an opaque case id, a canonical molecular formula and an optional split.
* :func:`build_holder_challenge` may use a held structure to construct a
  closed challenge set, but keeps selection tiers in a holder-only audit.
* :func:`evaluate_open_world_coverage` joins a sealed mapping only after an
  open-world bundle exists and reports generator coverage, never ranking
  performance.

The separation is part of the protocol, not merely documentation.  The public
open-world query schema has no field through which a known structure or model
outcome can be supplied, and the coverage evaluator rejects challenge bundles
whose held structure was inserted by construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from app.ml.nmr_data_v2 import connect_readonly, schema_version
from app.ml.nmr_evidence import FormulaError, canonical_formula

ROLELESS_BUNDLE_SCHEMA_VERSION = "chemapp.nmr.roleless-candidate-bundle.v1"
HOLDER_GOLD_SCHEMA_VERSION = "chemapp.nmr.holder-candidate-gold.v1"
HOLDER_CHALLENGE_AUDIT_SCHEMA_VERSION = "chemapp.nmr.holder-challenge-audit.v1"
COVERAGE_REPORT_SCHEMA_VERSION = "chemapp.nmr.candidate-coverage-report.v1"
INDEX_BINDING_SCHEMA_VERSION = "chemapp.nmr-readonly-index-binding.v1"

OPEN_WORLD_MODE = "deployment_open_world_formula_enumeration"
HOLDER_CHALLENGE_MODE = "holder_only_closed_challenge"
GENERATION_PROTOCOL_VERSION = "nmr-index-exact-canonical-formula-v2"
CHALLENGE_PROTOCOL_VERSION = "holder-fixed-hard-negative-hierarchy-v2"
CASE_ID_PATTERN = re.compile(r"^case-[0-9a-f]{24}$")
CANDIDATE_ID_PATTERN = re.compile(r"^candidate-[0-9a-f]{24}$")
CONNECTIVITY_ID_PATTERN = re.compile(r"^connectivity-[0-9a-f]{24}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GENERATION_ID_PATTERN = re.compile(r"^generation-[0-9a-f]{24}$")

DEFAULT_RECALL_K = (1, 5, 10, 25, 50, 100)
CHALLENGE_TIER_ORDER = (
    "same_standard_inchi_connectivity_different_canonical_identity",
    "same_scaffold_and_topological_13c_count",
    "same_scaffold",
    "same_topological_13c_count",
    "morgan_similarity_fallback",
)

_QUERY_FIELDS = frozenset({"case_id", "formula", "split"})
_ROLELESS_ROW_FIELDS = frozenset(
    {"case_id", "formula", "split", "candidate_count", "candidates"}
)
_ROLELESS_CANDIDATE_FIELDS = frozenset({"candidate_id", "smiles"})
_ROLELESS_BUNDLE_FIELDS = frozenset(
    {
        "schema_version",
        "generation_mode",
        "generation_protocol",
        "label_free",
        "generation_id",
        "index_binding",
        "query_count",
        "rows",
        "enumeration_audit",
        "split_isolation",
    }
)
_INDEX_BINDING_FIELDS = frozenset(
    {
        "schema_version",
        "sqlite_schema_version",
        "sqlite_bytes",
        "sqlite_sha256",
        "source_snapshots_sha256",
        "rdkit_version",
        "binding_sha256",
    }
)
_GOLD_ROW_FIELDS = frozenset({"case_id", "candidate_id", "connectivity_id"})
_HOLDER_CASE_FIELDS = frozenset(
    {"case_id", "formula", "split", "held_structure_smiles"}
)
_FORBIDDEN_ROLELESS_KEYS = frozenset(
    {
        "truth",
        "truth_candidate_id",
        "gold",
        "label",
        "outcome",
        "correct",
        "score",
        "similarity",
        "tier",
        "rank",
        "inchi",
        "inchi_key",
        "connectivity_key",
    }
)


class NMRCandidateGenerationError(ValueError):
    """Raised when generation, isolation, or coverage invariants fail."""


def canonical_json_dumps(value: Any) -> str:
    """Return deterministic JSON and reject non-finite floats."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def opaque_case_id(source_identifier: str) -> str:
    """Derive a stable opaque case id without exposing the source identifier."""

    if not isinstance(source_identifier, str) or not source_identifier.strip():
        raise NMRCandidateGenerationError("source identifier must be non-empty text")
    digest = hashlib.sha256(
        b"chemapp-nmr-candidate-case-v1\0" + source_identifier.strip().encode("utf-8")
    ).hexdigest()
    return f"case-{digest[:24]}"


def _candidate_id(canonical_isomeric_smiles: str) -> str:
    digest = hashlib.sha256(
        b"chemapp-opaque-nmr-canonical-isomeric-smiles-v2\0"
        + canonical_isomeric_smiles.encode("utf-8")
    ).hexdigest()
    return f"candidate-{digest[:24]}"


def _connectivity_id(connectivity_key: str) -> str:
    digest = hashlib.sha256(
        b"chemapp-opaque-nmr-connectivity-v1\0" + connectivity_key.encode("ascii")
    ).hexdigest()
    return f"connectivity-{digest[:24]}"


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NMRCandidateGenerationError(f"{field} must be non-empty text")
    return value.strip()


def _normalise_formula(value: Any, field: str) -> str:
    raw = _require_text(value, field)
    try:
        normalised = canonical_formula(raw)
    except FormulaError as exc:
        raise NMRCandidateGenerationError(
            f"{field} is not a supported formula"
        ) from exc
    if not normalised:
        raise NMRCandidateGenerationError(f"{field} is not a supported formula")
    return normalised


def _require_case_id(value: Any, field: str = "case_id") -> str:
    case_id = _require_text(value, field)
    if CASE_ID_PATTERN.fullmatch(case_id) is None:
        raise NMRCandidateGenerationError(
            f"{field} must be an opaque case-<24 lowercase hex> identifier"
        )
    return case_id


def _require_candidate_id(value: Any, field: str = "candidate_id") -> str:
    candidate_id = _require_text(value, field)
    if CANDIDATE_ID_PATTERN.fullmatch(candidate_id) is None:
        raise NMRCandidateGenerationError(
            f"{field} must be an opaque candidate-<24 lowercase hex> identifier"
        )
    return candidate_id


def _require_connectivity_id(value: Any, field: str = "connectivity_id") -> str:
    connectivity_id = _require_text(value, field)
    if CONNECTIVITY_ID_PATTERN.fullmatch(connectivity_id) is None:
        raise NMRCandidateGenerationError(
            f"{field} must be an opaque connectivity-<24 lowercase hex> identifier"
        )
    return connectivity_id


def _normalise_split(value: Any, field: str = "split") -> str | None:
    if value is None:
        return None
    split = _require_text(value, field)
    if len(split) > 64 or re.fullmatch(r"[a-z][a-z0-9_-]*", split) is None:
        raise NMRCandidateGenerationError(f"{field} is invalid")
    return split


def _walk_mapping_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.append(str(key))
            keys.extend(_walk_mapping_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(_walk_mapping_keys(nested))
    return keys


def _reject_roleless_leakage(value: Mapping[str, Any]) -> None:
    leaked = sorted(
        {
            key
            for key in _walk_mapping_keys(value)
            if key.casefold() in _FORBIDDEN_ROLELESS_KEYS
        }
    )
    if leaked:
        raise NMRCandidateGenerationError(
            f"roleless candidate bundle contains prohibited fields: {leaked}"
        )


def _generation_id(value_without_generation_id: Mapping[str, Any]) -> str:
    return f"generation-{canonical_sha256(value_without_generation_id)[:24]}"


def _is_link_or_reparse(path: Path, metadata: os.stat_result) -> bool:
    """Treat POSIX symlinks and Windows reparse points as links."""

    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    is_junction = getattr(path, "is_junction", lambda: False)()
    return bool(
        stat.S_ISLNK(metadata.st_mode)
        or is_junction
        or (reparse_flag and file_attributes & reparse_flag)
    )


def _sqlite_sidecars(path: Path) -> tuple[Path, Path, Path]:
    return tuple(Path(f"{path}{suffix}") for suffix in ("-wal", "-shm", "-journal"))


def _reject_sqlite_sidecars(path: Path, *, phase: str) -> None:
    present = [sidecar for sidecar in _sqlite_sidecars(path) if os.path.lexists(sidecar)]
    if present:
        raise NMRCandidateGenerationError(
            f"NMR index has an unbound SQLite sidecar during {phase}: "
            + ", ".join(str(item) for item in present)
        )


def _validate_index_binding(value: Any) -> None:
    if not isinstance(value, Mapping) or set(value) != _INDEX_BINDING_FIELDS:
        raise NMRCandidateGenerationError("read-only index binding field mismatch")
    if (
        value.get("schema_version") != INDEX_BINDING_SCHEMA_VERSION
        or value.get("sqlite_schema_version") != "2"
    ):
        raise NMRCandidateGenerationError("read-only index binding schema changed")
    byte_size = value.get("sqlite_bytes")
    if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 1:
        raise NMRCandidateGenerationError("read-only index byte count is invalid")
    for name in ("sqlite_sha256", "source_snapshots_sha256", "binding_sha256"):
        if (
            not isinstance(value.get(name), str)
            or SHA256_PATTERN.fullmatch(str(value[name])) is None
        ):
            raise NMRCandidateGenerationError(f"read-only index {name} is invalid")
    _require_text(value.get("rdkit_version"), "read-only index rdkit_version")
    core = {key: nested for key, nested in value.items() if key != "binding_sha256"}
    if canonical_sha256(core) != value["binding_sha256"]:
        raise NMRCandidateGenerationError("read-only index binding hash changed")


def _canonical_structure(
    smiles: str,
    *,
    expected_formula: str,
) -> dict[str, Any] | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        return None
    try:
        formula = canonical_formula(rdMolDescriptors.CalcMolFormula(molecule))
    except FormulaError:
        return None
    if formula != expected_formula:
        return None
    canonical_smiles = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    inchi_key = Chem.MolToInchiKey(molecule)
    if not inchi_key:
        return None
    return {
        "candidate_id": _candidate_id(canonical_smiles),
        "smiles": canonical_smiles,
        "inchi_key": inchi_key,
        "connectivity_key": inchi_key[:14],
        "formula": formula,
        "molecule": molecule,
    }


class ReadonlyNMRIndex:
    """Hash-bound, query-only interface to the v2 NMR structure index."""

    def __init__(self, path: str | Path) -> None:
        supplied = Path(os.path.abspath(Path(path).expanduser()))
        current = Path(supplied.anchor) if supplied.anchor else Path()
        for part in supplied.parts[1:] if supplied.anchor else supplied.parts:
            current /= part
            try:
                component_metadata = current.lstat()
            except FileNotFoundError:
                continue
            if _is_link_or_reparse(current, component_metadata):
                raise NMRCandidateGenerationError(
                    f"NMR index path contains a symlink component: {current}"
                )
        try:
            supplied_metadata = supplied.lstat()
        except OSError as exc:
            raise NMRCandidateGenerationError(
                f"cannot inspect NMR index: {supplied}"
            ) from exc
        if _is_link_or_reparse(supplied, supplied_metadata) or not stat.S_ISREG(
            supplied_metadata.st_mode
        ):
            raise NMRCandidateGenerationError(
                "NMR index must be a regular, non-symlink SQLite file"
            )
        selected = supplied.resolve(strict=True)
        metadata = selected.stat()
        _reject_sqlite_sidecars(selected, phase="open preflight")
        self.path = selected
        self._initial_stat = (metadata.st_size, metadata.st_mtime_ns)
        self._initial_sha256 = file_sha256(selected)
        self.connection = connect_readonly(selected)
        self.connection.execute("PRAGMA query_only = ON")
        try:
            self.schema = schema_version(self.connection)
            self._validate_schema()
            self.binding = self._build_binding(metadata.st_size)
        except Exception:
            self.connection.close()
            _reject_sqlite_sidecars(selected, phase="failed-open close verification")
            raise
        self._closed = False

    def _validate_schema(self) -> None:
        molecule_columns = {
            str(row[1])
            for row in self.connection.execute("PRAGMA table_info(molecules)")
        }
        required = {"id", "smiles", "inchi_key", "formula"}
        if not required.issubset(molecule_columns):
            raise NMRCandidateGenerationError(
                "NMR index molecules table is missing required columns"
            )
        integrity = self.connection.execute("PRAGMA quick_check").fetchone()
        if integrity is None or str(integrity[0]).casefold() != "ok":
            raise NMRCandidateGenerationError("NMR index failed SQLite quick_check")

    def _build_binding(self, byte_size: int) -> dict[str, Any]:
        snapshots = [
            {
                "source_name": str(row[0]),
                "source_version": None if row[1] is None else str(row[1]),
                "sha256": str(row[2]),
            }
            for row in self.connection.execute(
                """
                SELECT source_name, source_version, sha256
                FROM source_snapshots
                ORDER BY id
                """
            )
        ]
        if not snapshots or any(
            SHA256_PATTERN.fullmatch(item["sha256"]) is None for item in snapshots
        ):
            raise NMRCandidateGenerationError(
                "NMR index source snapshot binding is missing or invalid"
            )
        core = {
            "schema_version": INDEX_BINDING_SCHEMA_VERSION,
            "sqlite_schema_version": self.schema,
            "sqlite_bytes": byte_size,
            "sqlite_sha256": self._initial_sha256,
            "source_snapshots_sha256": canonical_sha256(snapshots),
            "rdkit_version": rdBase.rdkitVersion,
        }
        return {**core, "binding_sha256": canonical_sha256(core)}

    def enumerate_formulas(
        self,
        formulas: Sequence[str],
    ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, int]]]:
        """Enumerate unique structures for formulas without any held labels."""

        unique_formulas = sorted(set(formulas))
        structures: dict[str, dict[str, dict[str, Any]]] = {
            formula: {} for formula in unique_formulas
        }
        audits: dict[str, Counter[str]] = {
            formula: Counter() for formula in unique_formulas
        }
        for offset in range(0, len(unique_formulas), 800):
            batch = unique_formulas[offset : offset + 800]
            if not batch:
                continue
            placeholders = ",".join("?" for _ in batch)
            rows = self.connection.execute(
                f"""
                SELECT id, formula, smiles, inchi_key
                FROM molecules
                WHERE formula IN ({placeholders}) AND smiles IS NOT NULL
                ORDER BY id
                """,
                batch,
            ).fetchall()
            for row in rows:
                formula = str(row["formula"])
                audits[formula]["index_rows"] += 1
                candidate = _canonical_structure(
                    str(row["smiles"]),
                    expected_formula=formula,
                )
                if candidate is None:
                    audits[formula]["invalid_structure_rows"] += 1
                    continue
                stored_inchi = row["inchi_key"]
                if stored_inchi and str(stored_inchi) != candidate["inchi_key"]:
                    audits[formula]["stored_identity_mismatch_rows"] += 1
                    continue
                candidate_id = str(candidate["candidate_id"])
                previous = structures[formula].get(candidate_id)
                if previous is None or candidate["smiles"] < previous["smiles"]:
                    structures[formula][candidate_id] = candidate
                else:
                    audits[formula]["duplicate_structure_rows"] += 1

        result: dict[str, list[dict[str, Any]]] = {}
        summary: dict[str, dict[str, int]] = {}
        for formula in unique_formulas:
            candidates = sorted(
                structures[formula].values(),
                key=lambda candidate: str(candidate["candidate_id"]),
            )
            result[formula] = candidates
            summary[formula] = {
                "index_rows": audits[formula]["index_rows"],
                "candidate_count": len(candidates),
                "invalid_structure_rows": audits[formula]["invalid_structure_rows"],
                "stored_identity_mismatch_rows": audits[formula][
                    "stored_identity_mismatch_rows"
                ],
                "duplicate_structure_rows": max(
                    audits[formula]["duplicate_structure_rows"],
                    audits[formula]["index_rows"]
                    - audits[formula]["invalid_structure_rows"]
                    - audits[formula]["stored_identity_mismatch_rows"]
                    - len(candidates),
                ),
            }
        return result, summary

    def verify_unchanged_and_close(self) -> None:
        if self._closed:
            return
        self.connection.close()
        self._closed = True
        _reject_sqlite_sidecars(self.path, phase="close verification")
        metadata = self.path.lstat()
        final_stat = (metadata.st_size, metadata.st_mtime_ns)
        if (
            final_stat != self._initial_stat
            or file_sha256(self.path) != self._initial_sha256
        ):
            raise NMRCandidateGenerationError(
                "NMR index changed while the formula enumeration was running"
            )

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.verify_unchanged_and_close()
        elif not self._closed:
            self.connection.close()
            self._closed = True
            _reject_sqlite_sidecars(self.path, phase="exception close verification")


def _validate_queries(queries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not queries:
        raise NMRCandidateGenerationError("at least one open-world query is required")
    normalised: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, raw in enumerate(queries):
        if not isinstance(raw, Mapping) or set(raw) != _QUERY_FIELDS:
            raise NMRCandidateGenerationError(
                f"queries[{position}] must contain exactly {sorted(_QUERY_FIELDS)}"
            )
        case_id = _require_case_id(raw.get("case_id"), f"queries[{position}].case_id")
        if case_id in seen:
            raise NMRCandidateGenerationError(f"duplicate case_id: {case_id}")
        seen.add(case_id)
        normalised.append(
            {
                "case_id": case_id,
                "formula": _normalise_formula(
                    raw.get("formula"), f"queries[{position}].formula"
                ),
                "split": _normalise_split(
                    raw.get("split"), f"queries[{position}].split"
                ),
            }
        )
    return sorted(normalised, key=lambda row: row["case_id"])


def audit_split_isolation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Check candidate ids and connectivity for every assigned split pair."""

    candidate_by_split: dict[str, set[str]] = defaultdict(set)
    connectivity_by_split: dict[str, set[str]] = defaultdict(set)
    assigned_rows = 0
    for position, row in enumerate(rows):
        split = _normalise_split(row.get("split"), f"rows[{position}].split")
        if split is None:
            continue
        assigned_rows += 1
        candidates = row.get("candidates")
        if not isinstance(candidates, list):
            raise NMRCandidateGenerationError(
                f"rows[{position}].candidates must be a list"
            )
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                raise NMRCandidateGenerationError("candidate must be an object")
            candidate_id = _require_candidate_id(candidate.get("candidate_id"))
            molecule = Chem.MolFromSmiles(
                _require_text(candidate.get("smiles"), "smiles")
            )
            if molecule is None:
                raise NMRCandidateGenerationError("candidate SMILES is invalid")
            inchi_key = Chem.MolToInchiKey(molecule)
            canonical_smiles = Chem.MolToSmiles(
                molecule, canonical=True, isomericSmiles=True
            )
            if not inchi_key or _candidate_id(canonical_smiles) != candidate_id:
                raise NMRCandidateGenerationError(
                    "candidate id does not match candidate structure"
                )
            candidate_by_split[split].add(candidate_id)
            connectivity_by_split[split].add(inchi_key[:14])

    splits = sorted(candidate_by_split)
    pairwise: list[dict[str, Any]] = []
    total_candidate_overlap = 0
    total_connectivity_overlap = 0
    for left_index, left in enumerate(splits):
        for right in splits[left_index + 1 :]:
            candidate_overlap = candidate_by_split[left] & candidate_by_split[right]
            connectivity_overlap = (
                connectivity_by_split[left] & connectivity_by_split[right]
            )
            total_candidate_overlap += len(candidate_overlap)
            total_connectivity_overlap += len(connectivity_overlap)
            pairwise.append(
                {
                    "left_split": left,
                    "right_split": right,
                    "candidate_id_overlap_count": len(candidate_overlap),
                    "connectivity_overlap_count": len(connectivity_overlap),
                }
            )
    checked = len(splits) >= 2
    return {
        "checked": checked,
        "assigned_row_count": assigned_rows,
        "unassigned_row_count": len(rows) - assigned_rows,
        "splits": splits,
        "pairwise": pairwise,
        "candidate_id_overlap_count": total_candidate_overlap,
        "connectivity_overlap_count": total_connectivity_overlap,
        "isolated": checked
        and total_candidate_overlap == 0
        and total_connectivity_overlap == 0,
    }


def validate_roleless_bundle(bundle: Mapping[str, Any]) -> None:
    """Validate schema, stable identities, formula consistency, and no leakage."""

    if set(bundle) != _ROLELESS_BUNDLE_FIELDS:
        raise NMRCandidateGenerationError("roleless bundle field allowlist mismatch")
    if bundle.get("schema_version") != ROLELESS_BUNDLE_SCHEMA_VERSION:
        raise NMRCandidateGenerationError("roleless bundle schema changed")
    mode = bundle.get("generation_mode")
    if mode not in {OPEN_WORLD_MODE, HOLDER_CHALLENGE_MODE}:
        raise NMRCandidateGenerationError("unsupported candidate generation mode")
    if bundle.get("label_free") is not True:
        raise NMRCandidateGenerationError("roleless bundle must be label-free")
    generation_id = bundle.get("generation_id")
    if (
        not isinstance(generation_id, str)
        or GENERATION_ID_PATTERN.fullmatch(generation_id) is None
    ):
        raise NMRCandidateGenerationError("roleless generation id is invalid")
    expected_protocol = (
        GENERATION_PROTOCOL_VERSION
        if mode == OPEN_WORLD_MODE
        else CHALLENGE_PROTOCOL_VERSION
    )
    if bundle.get("generation_protocol") != expected_protocol:
        raise NMRCandidateGenerationError("candidate generation protocol changed")
    _validate_index_binding(bundle.get("index_binding"))
    rows = bundle.get("rows")
    if not isinstance(rows, list) or bundle.get("query_count") != len(rows):
        raise NMRCandidateGenerationError("roleless row count binding changed")
    _reject_roleless_leakage(bundle)
    seen_cases: set[str] = set()
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _ROLELESS_ROW_FIELDS:
            raise NMRCandidateGenerationError(
                f"roleless rows[{position}] field allowlist mismatch"
            )
        case_id = _require_case_id(row.get("case_id"))
        if case_id in seen_cases:
            raise NMRCandidateGenerationError(f"duplicate roleless case: {case_id}")
        seen_cases.add(case_id)
        formula = _normalise_formula(row.get("formula"), f"rows[{position}].formula")
        if row.get("formula") != formula:
            raise NMRCandidateGenerationError("roleless formula is not canonical")
        _normalise_split(row.get("split"), f"rows[{position}].split")
        candidates = row.get("candidates")
        if not isinstance(candidates, list) or row.get("candidate_count") != len(
            candidates
        ):
            raise NMRCandidateGenerationError("candidate count binding changed")
        candidate_ids: list[str] = []
        for candidate_position, candidate in enumerate(candidates):
            if (
                not isinstance(candidate, Mapping)
                or set(candidate) != _ROLELESS_CANDIDATE_FIELDS
            ):
                raise NMRCandidateGenerationError(
                    f"rows[{position}].candidates[{candidate_position}] field mismatch"
                )
            candidate_id = _require_candidate_id(candidate.get("candidate_id"))
            supplied_smiles = _require_text(candidate.get("smiles"), "candidate.smiles")
            structure = _canonical_structure(
                supplied_smiles,
                expected_formula=formula,
            )
            if (
                structure is None
                or structure["candidate_id"] != candidate_id
                or structure["smiles"] != supplied_smiles
            ):
                raise NMRCandidateGenerationError(
                    "roleless candidate identity or formula does not validate"
                )
            candidate_ids.append(candidate_id)
        if candidate_ids != sorted(set(candidate_ids)):
            raise NMRCandidateGenerationError(
                "roleless candidates must be unique and ordered by opaque id"
            )
    isolation = audit_split_isolation(rows)
    if isolation != bundle.get("split_isolation"):
        raise NMRCandidateGenerationError("split-isolation audit binding changed")
    generation_core = {
        key: value for key, value in bundle.items() if key != "generation_id"
    }
    if generation_id != _generation_id(generation_core):
        raise NMRCandidateGenerationError("roleless generation id binding changed")


def build_open_world_bundle(
    index_path: str | Path,
    queries: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a formula-only deployment bundle from a read-only NMR index.

    The accepted query allowlist is exactly ``case_id``, ``formula`` and
    ``split``.  There is intentionally no override to insert a known structure.
    """

    normalised_queries = _validate_queries(queries)
    with ReadonlyNMRIndex(index_path) as index:
        by_formula, formula_audit = index.enumerate_formulas(
            [query["formula"] for query in normalised_queries]
        )
        rows = [
            {
                "case_id": query["case_id"],
                "formula": query["formula"],
                "split": query["split"],
                "candidate_count": len(by_formula[query["formula"]]),
                "candidates": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "smiles": candidate["smiles"],
                    }
                    for candidate in by_formula[query["formula"]]
                ],
            }
            for query in normalised_queries
        ]
        audit_totals = Counter()
        for audit in formula_audit.values():
            audit_totals.update(audit)
        bundle_core = {
            "schema_version": ROLELESS_BUNDLE_SCHEMA_VERSION,
            "generation_mode": OPEN_WORLD_MODE,
            "generation_protocol": GENERATION_PROTOCOL_VERSION,
            "label_free": True,
            "index_binding": dict(index.binding),
            "query_count": len(rows),
            "rows": rows,
            "enumeration_audit": {
                "unique_formula_count": len(formula_audit),
                "index_row_count": audit_totals["index_rows"],
                "unique_candidate_count_across_formulae": sum(
                    audit["candidate_count"] for audit in formula_audit.values()
                ),
                "invalid_structure_row_count": audit_totals["invalid_structure_rows"],
                "stored_identity_mismatch_row_count": audit_totals[
                    "stored_identity_mismatch_rows"
                ],
                "duplicate_structure_row_count": audit_totals[
                    "duplicate_structure_rows"
                ],
            },
            "split_isolation": audit_split_isolation(rows),
        }
        bundle = {**bundle_core, "generation_id": _generation_id(bundle_core)}
    validate_roleless_bundle(bundle)
    return bundle


def build_holder_gold(
    holder_cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create the minimal sealed mapping used only after bundle generation."""

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for position, raw in enumerate(holder_cases):
        if not isinstance(raw, Mapping) or set(raw) != _HOLDER_CASE_FIELDS:
            raise NMRCandidateGenerationError(
                f"holder_cases[{position}] field allowlist mismatch"
            )
        case_id = _require_case_id(raw.get("case_id"))
        if case_id in seen:
            raise NMRCandidateGenerationError(f"duplicate holder case: {case_id}")
        seen.add(case_id)
        formula = _normalise_formula(raw.get("formula"), "holder formula")
        structure = _canonical_structure(
            _require_text(raw.get("held_structure_smiles"), "held structure"),
            expected_formula=formula,
        )
        if structure is None:
            raise NMRCandidateGenerationError(
                f"{case_id}: held structure does not match formula"
            )
        rows.append(
            {
                "case_id": case_id,
                "candidate_id": structure["candidate_id"],
                "connectivity_id": _connectivity_id(structure["connectivity_key"]),
            }
        )
    rows.sort(key=lambda row: row["case_id"])
    return {
        "schema_version": HOLDER_GOLD_SCHEMA_VERSION,
        "rows": rows,
        "mapping_sha256": canonical_sha256(rows),
    }


def _murcko_scaffold(molecule: Chem.Mol) -> str | None:
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=False,
    )
    return scaffold or None


def _topological_13c_count(molecule: Chem.Mol) -> int:
    ranks = Chem.CanonicalRankAtoms(
        molecule,
        breakTies=False,
        includeChirality=False,
        includeIsotopes=False,
    )
    return len(
        {
            int(ranks[index])
            for index, atom in enumerate(molecule.GetAtoms())
            if atom.GetAtomicNum() == 6
        }
    )


def _challenge_decoy_metadata(
    held: Mapping[str, Any],
    candidate: Mapping[str, Any],
    fingerprint_generator: Any,
) -> dict[str, Any]:
    held_molecule = held["molecule"]
    candidate_molecule = candidate["molecule"]
    same_connectivity = held["connectivity_key"] == candidate["connectivity_key"]
    held_scaffold = _murcko_scaffold(held_molecule)
    candidate_scaffold = _murcko_scaffold(candidate_molecule)
    same_scaffold = (
        held_scaffold is not None and held_scaffold == candidate_scaffold
    )
    same_topological_count = _topological_13c_count(
        held_molecule
    ) == _topological_13c_count(candidate_molecule)
    if same_connectivity:
        tier = CHALLENGE_TIER_ORDER[0]
    elif same_scaffold and same_topological_count:
        tier = CHALLENGE_TIER_ORDER[1]
    elif same_scaffold:
        tier = CHALLENGE_TIER_ORDER[2]
    elif same_topological_count:
        tier = CHALLENGE_TIER_ORDER[3]
    else:
        tier = CHALLENGE_TIER_ORDER[4]
    similarity = float(
        DataStructs.TanimotoSimilarity(
            fingerprint_generator.GetFingerprint(held_molecule),
            fingerprint_generator.GetFingerprint(candidate_molecule),
        )
    )
    return {
        **candidate,
        "selection_tier": tier,
        "tier_ordinal": CHALLENGE_TIER_ORDER.index(tier),
        "morgan_tanimoto": similarity,
    }


def build_holder_challenge(
    open_world_bundle: Mapping[str, Any],
    holder_cases: Sequence[Mapping[str, Any]],
    *,
    max_decoys: int = 7,
    require_split_isolation: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a holder-only hard-negative challenge and a label-free export.

    Held structures are inserted only here.  Tier and similarity metadata are
    emitted exclusively in the returned holder audit; candidate order in the
    roleless bundle is by opaque id and therefore does not encode tier order.
    """

    if (
        isinstance(max_decoys, bool)
        or not isinstance(max_decoys, int)
        or max_decoys < 1
    ):
        raise NMRCandidateGenerationError("max_decoys must be a positive integer")
    validate_roleless_bundle(open_world_bundle)
    if open_world_bundle.get("generation_mode") != OPEN_WORLD_MODE:
        raise NMRCandidateGenerationError(
            "holder challenge must start from a deployment open-world bundle"
        )
    source_rows = {str(row["case_id"]): row for row in open_world_bundle["rows"]}
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
    )
    challenge_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, raw in enumerate(holder_cases):
        if not isinstance(raw, Mapping) or set(raw) != _HOLDER_CASE_FIELDS:
            raise NMRCandidateGenerationError(
                f"holder_cases[{position}] field allowlist mismatch"
            )
        case_id = _require_case_id(raw.get("case_id"))
        if case_id in seen:
            raise NMRCandidateGenerationError(f"duplicate holder case: {case_id}")
        seen.add(case_id)
        source_row = source_rows.get(case_id)
        if source_row is None:
            raise NMRCandidateGenerationError(
                f"holder case is absent from open-world bundle: {case_id}"
            )
        formula = _normalise_formula(raw.get("formula"), "holder formula")
        split = _normalise_split(raw.get("split"), "holder split")
        if source_row["formula"] != formula or source_row["split"] != split:
            raise NMRCandidateGenerationError(
                f"holder metadata does not match open-world query: {case_id}"
            )
        held = _canonical_structure(
            _require_text(raw.get("held_structure_smiles"), "held structure"),
            expected_formula=formula,
        )
        if held is None:
            raise NMRCandidateGenerationError(
                f"{case_id}: held structure does not match formula"
            )
        decoys: list[dict[str, Any]] = []
        for candidate_row in source_row["candidates"]:
            candidate = _canonical_structure(
                str(candidate_row["smiles"]),
                expected_formula=formula,
            )
            if candidate is None:
                raise NMRCandidateGenerationError(
                    "validated open-world candidate failed holder re-validation"
                )
            if candidate["candidate_id"] == held["candidate_id"]:
                continue
            decoys.append(
                _challenge_decoy_metadata(held, candidate, fingerprint_generator)
            )
        decoys.sort(
            key=lambda item: (
                int(item["tier_ordinal"]),
                -float(item["morgan_tanimoto"]),
                str(item["candidate_id"]),
            )
        )
        selected = decoys[:max_decoys]
        roleless_candidates = sorted(
            [held, *selected],
            key=lambda candidate: str(candidate["candidate_id"]),
        )
        challenge_rows.append(
            {
                "case_id": case_id,
                "formula": formula,
                "split": split,
                "candidate_count": len(roleless_candidates),
                "candidates": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "smiles": candidate["smiles"],
                    }
                    for candidate in roleless_candidates
                ],
            }
        )
        gold_rows.append(
            {
                "case_id": case_id,
                "candidate_id": held["candidate_id"],
                "connectivity_id": _connectivity_id(held["connectivity_key"]),
            }
        )
        audit_rows.append(
            {
                "case_id": case_id,
                "available_decoy_count": len(decoys),
                "selected_decoy_count": len(selected),
                "selected": [
                    {
                        "candidate_id": item["candidate_id"],
                        "selection_tier": item["selection_tier"],
                        "morgan_tanimoto": round(float(item["morgan_tanimoto"]), 12),
                    }
                    for item in selected
                ],
            }
        )

    challenge_rows.sort(key=lambda row: row["case_id"])
    gold_rows.sort(key=lambda row: row["case_id"])
    audit_rows.sort(key=lambda row: row["case_id"])
    isolation = audit_split_isolation(challenge_rows)
    if require_split_isolation and isolation["checked"] and not isolation["isolated"]:
        raise NMRCandidateGenerationError(
            "holder challenge has candidate or connectivity overlap across splits"
        )
    challenge_bundle_core = {
        "schema_version": ROLELESS_BUNDLE_SCHEMA_VERSION,
        "generation_mode": HOLDER_CHALLENGE_MODE,
        "generation_protocol": CHALLENGE_PROTOCOL_VERSION,
        "label_free": True,
        "index_binding": dict(open_world_bundle["index_binding"]),
        "query_count": len(challenge_rows),
        "rows": challenge_rows,
        "enumeration_audit": {
            "source_open_world_bundle_sha256": canonical_sha256(open_world_bundle),
            "maximum_decoys_per_case": max_decoys,
            "selected_decoy_count": sum(
                row["selected_decoy_count"] for row in audit_rows
            ),
        },
        "split_isolation": isolation,
    }
    challenge_bundle = {
        **challenge_bundle_core,
        "generation_id": _generation_id(challenge_bundle_core),
    }
    gold = {
        "schema_version": HOLDER_GOLD_SCHEMA_VERSION,
        "rows": gold_rows,
        "mapping_sha256": canonical_sha256(gold_rows),
    }
    holder_audit = {
        "schema_version": HOLDER_CHALLENGE_AUDIT_SCHEMA_VERSION,
        "selection_protocol": CHALLENGE_PROTOCOL_VERSION,
        "selection_hierarchy": list(CHALLENGE_TIER_ORDER),
        "morgan_fingerprint": {"radius": 2, "bits": 2048},
        "roleless_bundle_sha256": canonical_sha256(challenge_bundle),
        "rows": audit_rows,
    }
    validate_roleless_bundle(challenge_bundle)
    return challenge_bundle, gold, holder_audit


def _validate_holder_gold(gold: Mapping[str, Any]) -> dict[str, tuple[str, str]]:
    if set(gold) != {"schema_version", "rows", "mapping_sha256"}:
        raise NMRCandidateGenerationError("holder Gold field allowlist mismatch")
    if gold.get("schema_version") != HOLDER_GOLD_SCHEMA_VERSION:
        raise NMRCandidateGenerationError("holder Gold schema changed")
    rows = gold.get("rows")
    if not isinstance(rows, list) or gold.get("mapping_sha256") != canonical_sha256(
        rows
    ):
        raise NMRCandidateGenerationError("holder Gold mapping hash changed")
    mapping: dict[str, tuple[str, str]] = {}
    for position, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _GOLD_ROW_FIELDS:
            raise NMRCandidateGenerationError(
                f"holder Gold rows[{position}] field allowlist mismatch"
            )
        case_id = _require_case_id(row.get("case_id"))
        candidate_id = _require_candidate_id(row.get("candidate_id"))
        connectivity_id = _require_connectivity_id(row.get("connectivity_id"))
        if case_id in mapping:
            raise NMRCandidateGenerationError(f"duplicate Gold case: {case_id}")
        mapping[case_id] = (candidate_id, connectivity_id)
    return mapping


def _coverage_counts(
    rows: Sequence[Mapping[str, Any]],
    gold_by_case: Mapping[str, tuple[str, str]],
    recall_k: Sequence[int],
) -> dict[str, Any]:
    formula_available = 0
    exact_covered = 0
    connectivity_covered = 0
    no_candidate = 0
    exact_missing_with_candidates = 0
    connectivity_missing_with_candidates = 0
    exact_hits_at_k = {int(k): 0 for k in recall_k}
    connectivity_hits_at_k = {int(k): 0 for k in recall_k}
    candidate_counts: list[int] = []
    for row in rows:
        candidate_ids = [str(item["candidate_id"]) for item in row["candidates"]]
        connectivity_ids: list[str] = []
        for item in row["candidates"]:
            molecule = Chem.MolFromSmiles(str(item["smiles"]))
            inchi_key = Chem.MolToInchiKey(molecule) if molecule is not None else ""
            if not inchi_key:
                raise NMRCandidateGenerationError(
                    "validated roleless candidate lost its connectivity identity"
                )
            connectivity_ids.append(_connectivity_id(inchi_key[:14]))
        candidate_counts.append(len(candidate_ids))
        if not candidate_ids:
            no_candidate += 1
        else:
            formula_available += 1
        selected, selected_connectivity = gold_by_case[str(row["case_id"])]
        try:
            exact_position = candidate_ids.index(selected) + 1
        except ValueError:
            if candidate_ids:
                exact_missing_with_candidates += 1
        else:
            exact_covered += 1
            for k in exact_hits_at_k:
                if exact_position <= k:
                    exact_hits_at_k[k] += 1
        try:
            connectivity_position = connectivity_ids.index(selected_connectivity) + 1
        except ValueError:
            if candidate_ids:
                connectivity_missing_with_candidates += 1
        else:
            connectivity_covered += 1
            for k in connectivity_hits_at_k:
                if connectivity_position <= k:
                    connectivity_hits_at_k[k] += 1
    total = len(rows)
    sorted_counts = sorted(candidate_counts)
    median = (
        0.0
        if not sorted_counts
        else float(sorted_counts[(len(sorted_counts) - 1) // 2])
        if len(sorted_counts) % 2 == 1
        else (
            sorted_counts[len(sorted_counts) // 2 - 1]
            + sorted_counts[len(sorted_counts) // 2]
        )
        / 2.0
    )
    return {
        "case_count": total,
        "formula_candidate_availability": {
            "case_count": formula_available,
            "rate": formula_available / total if total else 0.0,
        },
        "exact_identity_coverage": {
            "case_count": exact_covered,
            "rate": exact_covered / total if total else 0.0,
        },
        "connectivity_coverage": {
            "case_count": connectivity_covered,
            "rate": connectivity_covered / total if total else 0.0,
        },
        "no_candidate_case_count": no_candidate,
        "no_candidate_rate": no_candidate / total if total else 0.0,
        "exact_identity_missing_with_candidates_count": exact_missing_with_candidates,
        "connectivity_missing_with_candidates_count": connectivity_missing_with_candidates,
        "recall_at_k": {
            str(k): {
                "exact_identity": {
                    "hit_count": exact_hits_at_k[k],
                    "recall": exact_hits_at_k[k] / total if total else 0.0,
                },
                "connectivity": {
                    "hit_count": connectivity_hits_at_k[k],
                    "recall": connectivity_hits_at_k[k] / total if total else 0.0,
                },
            }
            for k in sorted(exact_hits_at_k)
        },
        "candidate_count": {
            "minimum": min(candidate_counts, default=0),
            "median": median,
            "maximum": max(candidate_counts, default=0),
            "mean": sum(candidate_counts) / total if total else 0.0,
        },
    }


def evaluate_open_world_coverage(
    roleless_bundle: Mapping[str, Any],
    holder_gold: Mapping[str, Any],
    *,
    recall_k: Sequence[int] = DEFAULT_RECALL_K,
) -> dict[str, Any]:
    """Join held identities after generation and report coverage-only metrics."""

    validate_roleless_bundle(roleless_bundle)
    if roleless_bundle.get("generation_mode") != OPEN_WORLD_MODE:
        raise NMRCandidateGenerationError(
            "coverage evaluation rejects holder-constructed challenge bundles"
        )
    k_values: list[int] = []
    for value in recall_k:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise NMRCandidateGenerationError(
                "recall K values must be positive integers"
            )
        k_values.append(value)
    if not k_values:
        raise NMRCandidateGenerationError("at least one recall K is required")
    k_values = sorted(set(k_values))
    gold_by_case = _validate_holder_gold(holder_gold)
    rows = roleless_bundle["rows"]
    row_cases = {str(row["case_id"]) for row in rows}
    if set(gold_by_case) != row_cases:
        missing_gold = len(row_cases - set(gold_by_case))
        unexpected_gold = len(set(gold_by_case) - row_cases)
        raise NMRCandidateGenerationError(
            "coverage join must be one-to-one: "
            f"missing={missing_gold}, unexpected={unexpected_gold}"
        )
    overall = _coverage_counts(rows, gold_by_case, k_values)
    by_split: dict[str, Any] = {}
    split_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        split_rows[str(row["split"] or "unassigned")].append(row)
    for split, selected_rows in sorted(split_rows.items()):
        by_split[split] = _coverage_counts(selected_rows, gold_by_case, k_values)
    report = {
        "schema_version": COVERAGE_REPORT_SCHEMA_VERSION,
        "evaluation_scope": "candidate_generator_coverage_only",
        "generation_mode": OPEN_WORLD_MODE,
        "roleless_bundle_sha256": canonical_sha256(roleless_bundle),
        "holder_mapping_sha256": holder_gold["mapping_sha256"],
        "join_performed_after_generation": True,
        "model_scores_consumed": False,
        "ranking_metrics_included": False,
        "overall": overall,
        "by_split": by_split,
        "split_isolation": dict(roleless_bundle["split_isolation"]),
    }
    if not math.isfinite(float(overall["exact_identity_coverage"]["rate"])):
        raise NMRCandidateGenerationError("coverage result is not finite")
    return report
