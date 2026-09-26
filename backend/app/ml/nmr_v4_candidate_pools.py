"""Prepare outcome-free candidate pools for the fourth NMR experiment.

This module deliberately stops before forward prediction.  The only runtime
operation it permits is :meth:`NMRForwardAdapter.preflight_candidates`, whose
protocol explicitly returns no model output.  Candidate selection uses a
frozen, structure-only Morgan similarity rule.  Gold identities are emitted
to split-specific files so the external-test Gold file can remain unopened
until after the permanent consumption ledger has been reserved.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import sqlite3
import tempfile
from typing import Any, Callable, Protocol

from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from app.ml import dp5q_runtime_pin
from app.ml.nmr_forward import (
    DP5Q_CONFORMER_PREFLIGHT_PROTOCOL_VERSION,
    NMRForwardAdapter,
    NMRForwardConfig,
    dp5q_conformer_policy,
)
from app.ml.nmr_data_v2 import connect_readonly, schema_version
from app.ml.nmr_evidence import FormulaError, canonical_formula
from app.ml.nmrsolver_reviewed_data import (
    DERIVED_RELEASE_SCHEMA_VERSION,
    load_derived_release,
)


ROLELESS_POOL_SCHEMA_VERSION = "chemapp.nmr.roleless-candidate-pool.v4"
SEALED_GOLD_SCHEMA_VERSION = "chemapp.nmr.sealed-candidate-gold.v4"
ELIGIBILITY_SCHEMA_VERSION = "chemapp.nmr.v4-pool-eligibility.v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "chemapp.nmr.v4-split-manifest.v2"
SUMMARY_SCHEMA_VERSION = "chemapp.nmr.v4-candidate-pool-summary.v3"
SOURCE_BINDING_SCHEMA_VERSION = "chemapp.nmrsolver-source-binding.v1"
IMPLEMENTATION_BINDING_SCHEMA_VERSION = (
    "chemapp.nmr.v4-implementation-binding.v1"
)

CANDIDATE_GENERATION_VERSION = (
    "same-formula-different-connectivity-morgan-r2-2048-hard-decoys-v1"
)
SPLIT_PROTOCOL_VERSION = (
    "candidate-structure-connected-components-v2"
)
DEFAULT_SPLIT_SEED = "chemapp-nmr-v4-external-freeze-20260729"
DEFAULT_MAX_DECOYS = 7
# Keep one conformer-preflight request comfortably below the adapter timeout.
# This is a batching parameter only: candidates are traversed in the same
# frozen hardness order, so changing it cannot change which passing decoys
# enter the cap.
MAX_PREFLIGHT_CANDIDATES = 4
MAX_HEAVY_ATOMS = 80
MAX_TOTAL_ATOMS = 200
MAX_ROTATABLE_BONDS = 20
ALLOWED_ATOMIC_NUMBERS = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35})

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SPLITS = ("dev", "calibration", "test")
_ROLELESS_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "record_id",
        "split",
        "split_group",
        "spectrum_fingerprint_sha256",
        "nucleus",
        "formula",
        "observed_13c",
        "candidates",
    }
)
_ROLELESS_CANDIDATE_FIELDS = frozenset({"candidate_id", "smiles"})
_SEALED_GOLD_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "record_id",
        "split",
        "truth_candidate_id",
    }
)
_ELIGIBILITY_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "record_id",
        "evaluated_before_model_scoring",
        "eligible",
        "reason_codes",
        "base_index_sha256",
        "candidate_generation_version",
        "max_decoys",
        "base_decoy_audit",
        "preflight_audit",
        "selected_hard_decoy_count",
        "base_index_structure_overlap",
    }
)
_SPLIT_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "record_id",
        "split",
        "split_group",
        "spectrum_fingerprint_sha256",
        "dp5_upstream_structure_overlap",
        "dp5_upstream_scaffold_overlap",
        "base_index_structure_overlap",
        "nmrexp_overlap",
        "candidate_set_sha256",
        "roleless_pool_row_sha256",
    }
)
_SUMMARY_FIELDS = frozenset(
    {
        "schema_version",
        "release_id",
        "outcome_free",
        "model_scores_created",
        "calibrator_fitted",
        "candidate_generation",
        "split_protocol",
        "source_binding",
        "base_index",
        "counts",
        "ineligibility_reason_counts",
        "artifacts",
        "implementation_binding",
        "formal_protocol",
    }
)
_IMPLEMENTATION_FILES = (
    "backend/app/ml/nmr_v4_candidate_pools.py",
    "backend/scripts/prepare_nmr_v4_candidate_pools.py",
    "backend/app/ml/nmr_forward.py",
    "backend/scripts/dp5q_sidecar.py",
    "backend/app/ml/dp5q_runtime_pin.py",
    "backend/app/ml/nmr_data_v2.py",
    "backend/app/ml/nmr_evidence.py",
    "backend/app/ml/nmrsolver_reviewed_data.py",
)
_SOURCE_OUTCOME_FIELDS = frozenset(
    {
        "raw_score",
        "outcome",
        "prediction",
        "predictions",
        "predicted_top1",
        "top1_exact_correct",
        "calibrated_probability",
        "model_score",
    }
)
_PREFLIGHT_RESPONSE_FIELDS = frozenset(
    {
        "status",
        "model_outputs_created",
        "protocol_version",
        "runtime",
        "results",
    }
)
_PREFLIGHT_RESULT_FIELDS = frozenset(
    {
        "candidate_id",
        "status",
        "reason_code",
        "canonical_smiles",
        "conformer_count",
        "warnings",
    }
)


class NMRV4CandidatePoolError(ValueError):
    """Raised when a source, split, preflight, or artifact fails closed."""


class ConformerPreflighter(Protocol):
    """The sole adapter surface allowed during candidate-pool preparation."""

    def preflight_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        formula: str | None = None,
    ) -> dict[str, Any]: ...


def canonical_json_dumps(value: Any) -> str:
    """Render deterministic JSON while rejecting NaN and infinities."""

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


def implementation_binding() -> dict[str, Any]:
    """Bind every local implementation file executed by pool preparation."""

    project_root = Path(__file__).resolve().parents[3]
    files: list[dict[str, Any]] = []
    for relative_path in _IMPLEMENTATION_FILES:
        path = project_root.joinpath(*relative_path.split("/"))
        if path.is_symlink() or not path.is_file():
            raise NMRV4CandidatePoolError(
                f"implementation file is missing or unsafe: {relative_path}"
            )
        try:
            path.resolve(strict=True).relative_to(project_root)
        except ValueError as exc:
            raise NMRV4CandidatePoolError(
                f"implementation file escapes project root: {relative_path}"
            ) from exc
        files.append(
            {
                "path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    core = {
        "schema_version": IMPLEMENTATION_BINDING_SCHEMA_VERSION,
        "files": files,
        "host_runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "rdkit_version": rdBase.rdkitVersion,
        },
    }
    return {**core, "binding_sha256": canonical_sha256(core)}


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return "".join(canonical_json_dumps(dict(row)) + "\n" for row in rows).encode(
        "utf-8"
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (canonical_json_dumps(dict(value)) + "\n").encode("utf-8")


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise NMRV4CandidatePoolError(f"{field} must be a non-empty string")
    return value.strip()


def _require_hash(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HASH_RE.fullmatch(value) is None:
        raise NMRV4CandidatePoolError(f"{field} must be lowercase SHA-256")
    return value


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NMRV4CandidatePoolError(f"{field} must be an object")
    return value


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise NMRV4CandidatePoolError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _parse_json(text: str, *, location: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_object_pairs)
    except json.JSONDecodeError as exc:
        raise NMRV4CandidatePoolError(f"{location}: invalid JSON") from exc


def _walk_keys(value: Any) -> list[str]:
    keys: list[str] = []
    if isinstance(value, Mapping):
        for key, nested in value.items():
            keys.append(str(key))
            keys.extend(_walk_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.extend(_walk_keys(nested))
    return keys


def _reject_source_outcomes(record: Mapping[str, Any], *, record_id: str) -> None:
    found = sorted(
        {
            key
            for key in _walk_keys(record)
            if key.casefold() in _SOURCE_OUTCOME_FIELDS
        }
    )
    if found:
        raise NMRV4CandidatePoolError(
            f"{record_id}: source contains model outcome fields: {found}"
        )


def load_reviewed_release(
    current_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the immutable NMR-Solver release selected by ``CURRENT.json``.

    The pointer is treated only as a transport binding.  Every source row is
    validated again by :func:`build_v4_candidate_pool_artifacts`.
    """

    current = Path(current_path).resolve()
    if current.name != "CURRENT.json" or not current.is_file():
        raise NMRV4CandidatePoolError(
            "source must be an existing NMR-Solver CURRENT.json"
        )
    parsed = _parse_json(current.read_text(encoding="utf-8"), location=str(current))
    pointer = _require_mapping(parsed, "CURRENT.json")
    schema = _require_text(pointer.get("schema_version"), "CURRENT.schema_version")
    if schema != DERIVED_RELEASE_SCHEMA_VERSION:
        raise NMRV4CandidatePoolError("CURRENT.json is not an NMR-Solver v1 release")
    release_id = _require_hash(pointer.get("release_id"), "CURRENT.release_id")
    relative_release = pointer.get(
        "release_directory",
        f"releases/{release_id}",
    )
    relative_text = _require_text(
        relative_release,
        "CURRENT.release_directory",
    )
    relative = Path(relative_text)
    if relative.is_absolute() or ".." in relative.parts:
        raise NMRV4CandidatePoolError("CURRENT.release_directory is unsafe")
    root = current.parent
    release_dir = (root / relative).resolve()
    try:
        release_dir.relative_to(root)
    except ValueError as exc:
        raise NMRV4CandidatePoolError(
            "CURRENT.release_directory escapes the release root"
        ) from exc
    if release_dir.name != release_id:
        raise NMRV4CandidatePoolError(
            "CURRENT.release_directory is not named by release_id"
        )
    records_path = release_dir / "records.jsonl"
    expected_records_sha256 = _require_hash(
        pointer.get("records_sha256"),
        "CURRENT.records_sha256",
    )
    if not records_path.is_file():
        raise NMRV4CandidatePoolError("CURRENT-bound records.jsonl is missing")
    actual_records_sha256 = file_sha256(records_path)
    if actual_records_sha256 != expected_records_sha256:
        raise NMRV4CandidatePoolError("CURRENT records SHA-256 mismatch")
    # Reuse the source importer's stricter publication validator so summary and
    # attribution bytes are also bound by CURRENT, not merely records.jsonl.
    importer_records, source_summary = load_derived_release(current)

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with records_path.open("r", encoding="utf-8", newline="") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise NMRV4CandidatePoolError(
                    f"{records_path}:{line_number}: JSONL line lacks LF terminator"
                )
            raw = line[:-1]
            if not raw:
                raise NMRV4CandidatePoolError(
                    f"{records_path}:{line_number}: blank JSONL line"
                )
            parsed_row = _parse_json(
                raw,
                location=f"{records_path}:{line_number}",
            )
            row = dict(_require_mapping(parsed_row, "source record"))
            record_id = _require_text(
                row.get("record_id"),
                f"{records_path}:{line_number}.record_id",
            )
            if record_id in seen:
                raise NMRV4CandidatePoolError(
                    f"duplicate source record_id: {record_id}"
                )
            seen.add(record_id)
            records.append(row)
    if not records:
        raise NMRV4CandidatePoolError("source release contains no records")
    if importer_records != records:
        raise NMRV4CandidatePoolError(
            "source importer and strict JSONL loader disagree"
        )
    review_semantics = _require_mapping(
        source_summary.get("review_semantics"),
        "source summary.review_semantics",
    )
    if (
        review_semantics.get("review_kind")
        != "upstream_manually_curated_benchmark"
    ):
        raise NMRV4CandidatePoolError(
            "source summary is not the frozen upstream manually curated benchmark"
        )
    declared_count = pointer.get("records_count")
    if declared_count is not None and declared_count != len(records):
        raise NMRV4CandidatePoolError("CURRENT records_count mismatch")
    return records, {
        "schema_version": SOURCE_BINDING_SCHEMA_VERSION,
        "source_schema_version": schema,
        "release_id": release_id,
        "current_sha256": file_sha256(current),
        "records_sha256": actual_records_sha256,
        "records_count": len(records),
        "summary_sha256": _require_hash(
            pointer.get("summary_sha256"),
            "CURRENT.summary_sha256",
        ),
        "attribution_sha256": _require_hash(
            pointer.get("attribution_sha256"),
            "CURRENT.attribution_sha256",
        ),
        "review_kind": review_semantics["review_kind"],
    }


def _candidate_id(inchi_key: str) -> str:
    digest = hashlib.sha256(
        f"chemapp-opaque-nmr-candidate-v4\0{inchi_key}".encode("ascii")
    ).hexdigest()
    return f"candidate-{digest[:24]}"


def _murcko_smiles(molecule: Chem.Mol) -> str:
    # Match the reviewed-source importer and the attested DP5q overlap export:
    # MurckoScaffoldSmiles removes stereochemical decorations, and acyclic
    # structures share one explicit group instead of bypassing isolation.
    return (
        MurckoScaffold.MurckoScaffoldSmiles(
            mol=molecule,
            includeChirality=False,
        )
        or "acyclic"
    )


def _structure_candidate(
    smiles: str,
    *,
    expected_formula: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, ["structure_parse_failed"]
    if len(Chem.GetMolFrags(molecule)) != 1:
        reasons.append("multiple_fragments")
    if molecule.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        reasons.append("heavy_atom_limit_exceeded")
    if Chem.AddHs(molecule).GetNumAtoms() > MAX_TOTAL_ATOMS:
        reasons.append("total_atom_limit_exceeded")
    if rdMolDescriptors.CalcNumRotatableBonds(molecule) > MAX_ROTATABLE_BONDS:
        reasons.append("rotatable_bond_limit_exceeded")
    unsupported = sorted(
        {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
        - ALLOWED_ATOMIC_NUMBERS
    )
    reasons.extend(f"unsupported_atomic_number_{number}" for number in unsupported)
    carbon_count = sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
    if carbon_count == 0:
        reasons.append("no_carbon_atoms")
    try:
        formula = canonical_formula(rdMolDescriptors.CalcMolFormula(molecule))
    except FormulaError:
        formula = ""
        reasons.append("formula_is_not_supported")
    if formula != expected_formula:
        reasons.append("formula_mismatch")
    inchi_key = Chem.MolToInchiKey(molecule)
    if not inchi_key:
        reasons.append("inchi_key_generation_failed")
    if reasons:
        return None, sorted(set(reasons))
    canonical_smiles = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    return (
        {
            "candidate_id": _candidate_id(inchi_key),
            "smiles": canonical_smiles,
            "inchi_key": inchi_key,
            "connectivity_key": inchi_key[:14],
            "formula": formula,
            "carbon_count": carbon_count,
            "molecule": molecule,
        },
        [],
    )


def _record_overlap_flag(record: Mapping[str, Any], name: str) -> bool:
    locations: list[Mapping[str, Any]] = [record]
    for container_name in ("overlap", "independence", "calibration"):
        nested = record.get(container_name)
        if isinstance(nested, Mapping):
            locations.append(nested)
    found: list[bool] = []
    aliases = {name}
    if name.startswith("dp5_"):
        aliases.add(name.replace("dp5_", "dp5q_", 1))
    for location in locations:
        for alias in aliases:
            if alias in location:
                value = location[alias]
                if not isinstance(value, bool):
                    raise NMRV4CandidatePoolError(f"{alias} must be boolean")
                found.append(value)
    if not found:
        raise NMRV4CandidatePoolError(f"missing required overlap flag: {name}")
    if len(set(found)) != 1:
        raise NMRV4CandidatePoolError(f"conflicting overlap flag: {name}")
    return found[0]


def _nmrexp_overlap(record: Mapping[str, Any]) -> bool:
    values: list[bool] = []
    stack: list[Any] = [record]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key, value in current.items():
                lowered = str(key).casefold()
                if "nmrexp" in lowered and "overlap" in lowered:
                    if not isinstance(value, bool):
                        raise NMRV4CandidatePoolError(
                            f"{key} must be boolean when present"
                        )
                    values.append(value)
                elif isinstance(value, (Mapping, list)):
                    stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)
    return any(values)


def _normalise_source_record(record: Mapping[str, Any]) -> dict[str, Any]:
    record_id = _require_text(record.get("record_id"), "record_id")
    _reject_source_outcomes(record, record_id=record_id)
    structure = _require_mapping(record.get("structure"), f"{record_id}.structure")
    formula = canonical_formula(
        _require_text(structure.get("formula"), f"{record_id}.structure.formula")
    )
    raw_smiles = _require_text(
        structure.get("canonical_smiles"),
        f"{record_id}.structure.canonical_smiles",
    )
    identity_molecule = Chem.MolFromSmiles(raw_smiles)
    if identity_molecule is None or len(Chem.GetMolFrags(identity_molecule)) != 1:
        raise NMRV4CandidatePoolError(
            f"{record_id}: reviewed canonical_smiles is not one molecule"
        )
    computed_formula = canonical_formula(
        rdMolDescriptors.CalcMolFormula(identity_molecule)
    )
    computed_inchi = Chem.MolToInchiKey(identity_molecule)
    if not computed_inchi or computed_formula != formula:
        raise NMRV4CandidatePoolError(
            f"{record_id}: reviewed formula/identity does not match canonical_smiles"
        )
    computed_connectivity = computed_inchi[:14]
    computed_carbon_count = sum(
        atom.GetAtomicNum() == 6 for atom in identity_molecule.GetAtoms()
    )
    computed_scaffold = _murcko_smiles(identity_molecule)
    candidate, reasons = _structure_candidate(
        raw_smiles,
        expected_formula=formula,
    )
    expected_inchi = _require_text(
        structure.get("inchi_key"),
        f"{record_id}.structure.inchi_key",
    )
    expected_connectivity = _require_text(
        structure.get("connectivity_key"),
        f"{record_id}.structure.connectivity_key",
    )
    expected_carbon_count = structure.get("carbon_count")
    if (
        computed_inchi != expected_inchi
        or computed_connectivity != expected_connectivity
    ):
        raise NMRV4CandidatePoolError(
            f"{record_id}: structure identity does not match canonical_smiles"
        )
    if (
        isinstance(expected_carbon_count, bool)
        or not isinstance(expected_carbon_count, int)
        or expected_carbon_count != computed_carbon_count
    ):
        raise NMRV4CandidatePoolError(
            f"{record_id}: carbon_count does not match canonical_smiles"
        )
    stored_scaffold = structure.get(
        "scaffold_smiles",
        structure.get("murcko_scaffold_smiles"),
    )
    if not isinstance(stored_scaffold, str):
        raise NMRV4CandidatePoolError(
            f"{record_id}: structure.scaffold_smiles must be a string"
        )
    if stored_scaffold != computed_scaffold:
        raise NMRV4CandidatePoolError(
            f"{record_id}: Murcko scaffold does not match canonical_smiles"
        )
    raw_duplicate_group = record.get("duplicate_molecule_group")
    if isinstance(raw_duplicate_group, Mapping):
        duplicate_group = _require_text(
            raw_duplicate_group.get("molecule_group_id"),
            f"{record_id}.duplicate_molecule_group.molecule_group_id",
        )
        declared_group = record.get("molecule_group_id")
        if declared_group is not None and declared_group != duplicate_group:
            raise NMRV4CandidatePoolError(
                f"{record_id}: duplicate molecule group binding changed"
            )
    else:
        duplicate_group = _require_text(
            raw_duplicate_group,
            f"{record_id}.duplicate_molecule_group",
        )
    fingerprint = _require_hash(
        record.get("spectrum_fingerprint_sha256"),
        f"{record_id}.spectrum_fingerprint_sha256",
    )
    spectra = _require_mapping(record.get("spectra"), f"{record_id}.spectra")
    carbon_spectrum = _require_mapping(
        spectra.get("13C"),
        f"{record_id}.spectra.13C",
    )
    raw_shifts = carbon_spectrum.get("shifts_ppm")
    if not isinstance(raw_shifts, list) or not raw_shifts:
        raise NMRV4CandidatePoolError(
            f"{record_id}.spectra.13C.shifts_ppm must be a non-empty list"
        )
    shifts: list[float] = []
    for index, value in enumerate(raw_shifts):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not -20.0 <= float(value) <= 300.0
        ):
            raise NMRV4CandidatePoolError(
                f"{record_id}.spectra.13C.shifts_ppm[{index}] is invalid"
            )
        shifts.append(float(value))
    dp5_structure = _record_overlap_flag(
        record,
        "dp5_upstream_structure_overlap",
    )
    dp5_scaffold = _record_overlap_flag(
        record,
        "dp5_upstream_scaffold_overlap",
    )
    nmrexp_overlap = _nmrexp_overlap(record)
    return {
        "record_id": record_id,
        "normalization_reasons": [f"truth_{reason}" for reason in reasons],
        "candidate": candidate,
        "formula": formula,
        "connectivity_key": computed_connectivity,
        "observed_13c": sorted(shifts),
        "spectrum_fingerprint_sha256": fingerprint,
        "duplicate_molecule_group": duplicate_group,
        "scaffold_smiles": computed_scaffold,
        "dp5_upstream_structure_overlap": dp5_structure,
        "dp5_upstream_scaffold_overlap": dp5_scaffold,
        "nmrexp_overlap": nmrexp_overlap,
    }


def _base_decoys(
    connection: sqlite3.Connection,
    *,
    formula: str,
    truth_connectivity_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id, smiles, inchi_key
        FROM molecules
        WHERE formula = ? AND smiles IS NOT NULL
        ORDER BY id
        """,
        (formula,),
    ).fetchall()
    rejected: Counter[str] = Counter()
    same_connectivity = 0
    by_connectivity: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate, reasons = _structure_candidate(
            str(row["smiles"]),
            expected_formula=formula,
        )
        if candidate is None:
            rejected.update(reasons)
            continue
        stored_inchi = row["inchi_key"]
        if stored_inchi and str(stored_inchi) != candidate["inchi_key"]:
            rejected["stored_computed_inchi_key_mismatch"] += 1
            continue
        connectivity = str(candidate["connectivity_key"])
        if connectivity == truth_connectivity_key:
            same_connectivity += 1
            continue
        previous = by_connectivity.get(connectivity)
        if previous is None or int(row["id"]) < int(previous["base_row_id"]):
            candidate["base_row_id"] = int(row["id"])
            by_connectivity[connectivity] = candidate
    return list(by_connectivity.values()), {
        "base_formula_rows": len(rows),
        "same_truth_connectivity_rows_excluded": same_connectivity,
        "invalid_base_rows_by_reason": dict(sorted(rejected.items())),
        "different_connectivity_candidates": len(by_connectivity),
    }


def _base_formula_connectivity_keys(
    connection: sqlite3.Connection,
    *,
    formula: str,
    cache: dict[str, frozenset[str]],
) -> frozenset[str]:
    """Return computed base connectivities for one formula, cached per build."""

    cached = cache.get(formula)
    if cached is not None:
        return cached
    rows = connection.execute(
        """
        SELECT smiles
        FROM molecules
        WHERE formula = ? AND smiles IS NOT NULL
        ORDER BY id
        """,
        (formula,),
    ).fetchall()
    keys: set[str] = set()
    for row in rows:
        molecule = Chem.MolFromSmiles(str(row["smiles"]))
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            continue
        inchi_key = Chem.MolToInchiKey(molecule)
        if inchi_key:
            keys.add(inchi_key[:14])
    result = frozenset(keys)
    cache[formula] = result
    return result


def _hard_decoy_order(
    truth: Mapping[str, Any],
    decoys: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    truth_fp = generator.GetFingerprint(truth["molecule"])
    ranked: list[dict[str, Any]] = []
    for raw in decoys:
        candidate = dict(raw)
        similarity = float(
            DataStructs.TanimotoSimilarity(
                truth_fp,
                generator.GetFingerprint(candidate["molecule"]),
            )
        )
        candidate["truth_tanimoto"] = similarity
        ranked.append(candidate)
    ranked.sort(
        key=lambda candidate: (
            -float(candidate["truth_tanimoto"]),
            str(candidate["candidate_id"]),
        )
    )
    return ranked


def _validate_preflight_response(
    response: Any,
    *,
    candidates: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], str]:
    if not isinstance(response, Mapping) or set(response) != _PREFLIGHT_RESPONSE_FIELDS:
        raise NMRV4CandidatePoolError(
            "conformer preflight response field allowlist mismatch"
        )
    if response.get("status") != "ok" or response.get("model_outputs_created") is not False:
        raise NMRV4CandidatePoolError(
            "conformer preflight returned model-derived or failed semantics"
        )
    protocol = response.get("protocol_version")
    if isinstance(protocol, bool) or not isinstance(protocol, (int, str)):
        raise NMRV4CandidatePoolError("conformer preflight protocol is invalid")
    runtime = _require_mapping(response.get("runtime"), "preflight.runtime")
    expected_runtime_versions = {
        "python": dp5q_runtime_pin.DP5Q_PYTHON_VERSION,
        "tensorflow": dp5q_runtime_pin.DP5Q_TENSORFLOW_VERSION,
        "keras": dp5q_runtime_pin.DP5Q_KERAS_VERSION,
        "numpy": dp5q_runtime_pin.DP5Q_NUMPY_VERSION,
        "pandas": dp5q_runtime_pin.DP5Q_PANDAS_VERSION,
        "scipy": dp5q_runtime_pin.DP5Q_SCIPY_VERSION,
        "rdkit": dp5q_runtime_pin.DP5Q_RDKIT_VERSION,
        "scikit_learn": dp5q_runtime_pin.DP5Q_SCIKIT_LEARN_VERSION,
        "tqdm": dp5q_runtime_pin.DP5Q_TQDM_VERSION,
    }
    expected_runtime_fields = set(expected_runtime_versions) | {
        "source_bundle_sha256",
        "runtime_pin_code_sha256",
        "rdkit_version",
        "sidecar_code_sha256",
        "legacy_preflight_runtime_sha256",
        "protocol_version",
        "conformer_generation",
        "conformer_preflight",
    }
    for field in (
        "source_bundle_sha256",
        "runtime_pin_code_sha256",
        "sidecar_code_sha256",
        "legacy_preflight_runtime_sha256",
    ):
        _require_hash(runtime.get(field), f"preflight.runtime.{field}")
    if (
        set(runtime) != expected_runtime_fields
        or any(runtime.get(key) != value for key, value in expected_runtime_versions.items())
        or runtime.get("source_bundle_sha256")
        != dp5q_runtime_pin.DP5Q_UPSTREAM_SOURCE_BUNDLE_SHA256
        or runtime.get("runtime_pin_code_sha256")
        != file_sha256(Path(str(dp5q_runtime_pin.__file__)))
        or runtime.get("rdkit_version") != runtime.get("rdkit")
        or runtime.get("protocol_version") != 1
        or runtime.get("conformer_generation") != dp5q_conformer_policy()
        or runtime.get("conformer_preflight")
        != {
            "protocol_version": DP5Q_CONFORMER_PREFLIGHT_PROTOCOL_VERSION,
            "candidate_failures_are_results": True,
            "operational_failures_abort_request": True,
            "uses_same_prepare_candidate_path_as_prediction": True,
        }
    ):
        raise NMRV4CandidatePoolError(
            "conformer preflight runtime attestation changed"
        )
    legacy_two_field = {
        "sidecar_code_sha256": runtime["sidecar_code_sha256"],
        "rdkit_version": runtime["rdkit_version"],
    }
    if runtime["legacy_preflight_runtime_sha256"] != canonical_sha256(
        legacy_two_field
    ):
        raise NMRV4CandidatePoolError(
            "conformer preflight legacy runtime binding changed"
        )
    # Preserve the exact four-field identity used by the already audited pool.
    # The expanded attestation is validated above but deliberately excluded
    # from this legacy hash so a rerun can prove the same sidecar/RDKit and
    # conformer policy without pretending the first run recorded new fields.
    runtime_hash = canonical_sha256(
        {
            **legacy_two_field,
            "conformer_generation": runtime["conformer_generation"],
            "conformer_preflight": runtime["conformer_preflight"],
        }
    )
    results = response.get("results")
    if not isinstance(results, list) or len(results) != len(candidates):
        raise NMRV4CandidatePoolError(
            "conformer preflight changed candidate coverage"
        )
    expected_ids = [str(candidate["candidate_id"]) for candidate in candidates]
    clean: list[dict[str, Any]] = []
    for index, result in enumerate(results):
        if not isinstance(result, Mapping) or set(result) != _PREFLIGHT_RESULT_FIELDS:
            raise NMRV4CandidatePoolError(
                "conformer preflight result field allowlist mismatch"
            )
        candidate_id = result.get("candidate_id")
        if candidate_id != expected_ids[index]:
            raise NMRV4CandidatePoolError(
                "conformer preflight changed candidate identity or order"
            )
        status = result.get("status")
        if status not in {"passed", "rejected"}:
            raise NMRV4CandidatePoolError("conformer preflight status is invalid")
        conformer_count = result.get("conformer_count")
        warnings = result.get("warnings")
        if (
            isinstance(conformer_count, bool)
            or not isinstance(conformer_count, int)
            or conformer_count < 0
            or not isinstance(warnings, list)
            or any(not isinstance(item, str) for item in warnings)
        ):
            raise NMRV4CandidatePoolError(
                "conformer preflight result payload is invalid"
            )
        if status == "passed":
            if (
                result.get("reason_code") is not None
                or not isinstance(result.get("canonical_smiles"), str)
                or conformer_count < 1
            ):
                raise NMRV4CandidatePoolError(
                    "conformer preflight pass payload is invalid"
                )
        elif (
            not isinstance(result.get("reason_code"), str)
            or result.get("canonical_smiles") is not None
            or conformer_count != 0
        ):
            raise NMRV4CandidatePoolError(
                "conformer preflight rejection payload is invalid"
            )
        clean.append(dict(result))
    return clean, runtime_hash


def _preflight_hard_candidates(
    preflighter: ConformerPreflighter,
    *,
    truth: Mapping[str, Any],
    ranked_decoys: Sequence[Mapping[str, Any]],
    formula: str,
    max_decoys: int,
) -> tuple[bool, list[dict[str, Any]], dict[str, Any]]:
    """Preflight in hardness order and stop once later rows cannot enter the cap."""

    pending = [dict(truth), *[dict(candidate) for candidate in ranked_decoys]]
    passed_decoy_ids: set[str] = set()
    rejected_by_reason: Counter[str] = Counter()
    truth_passed = False
    runtime_hash: str | None = None
    checked = 0
    while pending and (not truth_passed or len(passed_decoy_ids) < max_decoys):
        # Keep every request role-opaque: no target/decoy marker or position rule
        # is sent to the adapter.
        chunk_raw = pending[:MAX_PREFLIGHT_CANDIDATES]
        del pending[:MAX_PREFLIGHT_CANDIDATES]
        chunk_raw.sort(key=lambda candidate: str(candidate["candidate_id"]))
        chunk = [
            {
                "candidate_id": str(candidate["candidate_id"]),
                "smiles": str(candidate["smiles"]),
            }
            for candidate in chunk_raw
        ]
        response = preflighter.preflight_candidates(chunk, formula=formula)
        results, current_runtime_hash = _validate_preflight_response(
            response,
            candidates=chunk,
        )
        if runtime_hash is None:
            runtime_hash = current_runtime_hash
        elif runtime_hash != current_runtime_hash:
            raise NMRV4CandidatePoolError(
                "conformer preflight runtime changed during one build"
            )
        for result in results:
            candidate_id = str(result["candidate_id"])
            checked += 1
            if result["status"] == "passed":
                if candidate_id == truth["candidate_id"]:
                    truth_passed = True
                else:
                    passed_decoy_ids.add(candidate_id)
            else:
                rejected_by_reason[str(result["reason_code"])] += 1

    selected = [
        dict(candidate)
        for candidate in ranked_decoys
        if candidate["candidate_id"] in passed_decoy_ids
    ][:max_decoys]
    return truth_passed, selected, {
        "candidate_count_preflighted": checked,
        "candidate_count_not_needed_after_cap": len(pending),
        "preflight_rejections_by_reason": dict(sorted(rejected_by_reason.items())),
        "preflight_runtime_sha256": runtime_hash,
        "model_outputs_created": False,
    }


class _UnionFind:
    def __init__(self, values: Sequence[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if left_root < right_root:
            self.parent[right_root] = left_root
        else:
            self.parent[left_root] = right_root


def _selected_candidate_identifiers(
    row: Mapping[str, Any],
) -> tuple[frozenset[str], frozenset[str]]:
    """Return every selected candidate identity used by one pool.

    Split grouping happens only after candidate selection and conformer
    preflight.  Consequently these identifiers are outcome-free, while still
    capturing dependence introduced when one structure is reused as a decoy
    in several pools.
    """

    raw_candidates = row.get("selected_candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise NMRV4CandidatePoolError(
            "split grouping requires a non-empty selected_candidates list"
        )
    candidate_ids: set[str] = set()
    connectivity_keys: set[str] = set()
    for index, raw_candidate in enumerate(raw_candidates):
        candidate = _require_mapping(
            raw_candidate,
            f"selected_candidates[{index}]",
        )
        candidate_id = _require_text(
            candidate.get("candidate_id"),
            f"selected_candidates[{index}].candidate_id",
        )
        raw_connectivity_key = candidate.get("connectivity_key")
        if raw_connectivity_key is None:
            smiles = _require_text(
                candidate.get("smiles"),
                f"selected_candidates[{index}].smiles",
            )
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
                raise NMRV4CandidatePoolError(
                    "selected candidate SMILES cannot provide connectivity"
                )
            inchi_key = Chem.MolToInchiKey(molecule)
            if not inchi_key:
                raise NMRV4CandidatePoolError(
                    "selected candidate InChIKey generation failed"
                )
            connectivity_key = inchi_key[:14]
        else:
            connectivity_key = _require_text(
                raw_connectivity_key,
                f"selected_candidates[{index}].connectivity_key",
            )
        if candidate_id in candidate_ids:
            raise NMRV4CandidatePoolError(
                "selected_candidates contains a duplicate candidate_id"
            )
        if connectivity_key in connectivity_keys:
            raise NMRV4CandidatePoolError(
                "selected_candidates contains duplicate connectivity"
            )
        candidate_ids.add(candidate_id)
        connectivity_keys.add(connectivity_key)
    return frozenset(candidate_ids), frozenset(connectivity_keys)


def _connected_split_components(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    ids = [str(row["record_id"]) for row in rows]
    union = _UnionFind(ids)
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        record_id = str(row["record_id"])
        candidate_ids, candidate_connectivity_keys = (
            _selected_candidate_identifiers(row)
        )
        identifiers = [
            ("duplicate", str(row["duplicate_molecule_group"])),
            ("molecule", str(row["connectivity_key"])),
            ("fingerprint", str(row["spectrum_fingerprint_sha256"])),
            *(
                ("selected_candidate_id", candidate_id)
                for candidate_id in sorted(candidate_ids)
            ),
            *(
                ("selected_candidate_connectivity", connectivity_key)
                for connectivity_key in sorted(candidate_connectivity_keys)
            ),
        ]
        scaffold = str(row["scaffold_smiles"])
        if scaffold:
            identifiers.append(("scaffold", scaffold))
        for identifier in identifiers:
            previous = seen.setdefault(identifier, record_id)
            union.union(record_id, previous)
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[union.find(str(row["record_id"]))].append(row)
    components: list[dict[str, Any]] = []
    for members in grouped.values():
        sorted_members = sorted(members, key=lambda row: str(row["record_id"]))
        component_candidate_ids: set[str] = set()
        component_candidate_connectivity_keys: set[str] = set()
        for row in sorted_members:
            candidate_ids, connectivity_keys = (
                _selected_candidate_identifiers(row)
            )
            component_candidate_ids.update(candidate_ids)
            component_candidate_connectivity_keys.update(connectivity_keys)
        binding = {
            "record_ids": [str(row["record_id"]) for row in sorted_members],
            "duplicate_molecule_groups": sorted(
                {str(row["duplicate_molecule_group"]) for row in sorted_members}
            ),
            "connectivity_keys": sorted(
                {str(row["connectivity_key"]) for row in sorted_members}
            ),
            "scaffold_smiles": sorted(
                {
                    str(row["scaffold_smiles"])
                    for row in sorted_members
                    if str(row["scaffold_smiles"])
                }
            ),
            "spectrum_fingerprints": sorted(
                {
                    str(row["spectrum_fingerprint_sha256"])
                    for row in sorted_members
                }
            ),
            "selected_candidate_ids": sorted(component_candidate_ids),
            "selected_candidate_connectivity_keys": sorted(
                component_candidate_connectivity_keys
            ),
        }
        components.append(
            {
                "split_group": f"group-{canonical_sha256(binding)[:24]}",
                "members": sorted_members,
                "restricted_to_dev": any(
                    bool(row["dp5_upstream_structure_overlap"])
                    or bool(row["dp5_upstream_scaffold_overlap"])
                    or bool(row["base_index_structure_overlap"])
                    or bool(row["nmrexp_overlap"])
                    for row in sorted_members
                ),
            }
        )
    return sorted(components, key=lambda item: str(item["split_group"]))


def _assign_frozen_splits(
    rows: Sequence[Mapping[str, Any]],
    *,
    split_seed: str,
) -> dict[str, dict[str, str]]:
    components = _connected_split_components(rows)
    restricted = [item for item in components if item["restricted_to_dev"]]
    clean = [item for item in components if not item["restricted_to_dev"]]
    assignments: dict[str, dict[str, str]] = {}

    def assign(component: Mapping[str, Any], split: str) -> None:
        for row in component["members"]:
            assignments[str(row["record_id"])] = {
                "split": split,
                "split_group": str(component["split_group"]),
            }

    for component in restricted:
        assign(component, "dev")

    ordered_clean = sorted(
        clean,
        key=lambda item: canonical_sha256(
            {"seed": split_seed, "split_group": item["split_group"]}
        ),
    )
    # Seed every claim-bearing split whenever the available independent groups
    # permit it.  A restricted group already supplies development data.
    required: list[str] = ["calibration", "test"]
    if not restricted:
        required.append("dev")
    counts = Counter(
        {
            "dev": sum(len(item["members"]) for item in restricted),
            "calibration": 0,
            "test": 0,
        }
    )
    while ordered_clean and required:
        split = required.pop(0)
        component = ordered_clean.pop(0)
        assign(component, split)
        counts[split] += len(component["members"])

    total = sum(len(item["members"]) for item in components)
    target = {
        "dev": 0.40 * total,
        "calibration": 0.30 * total,
        "test": 0.30 * total,
    }
    for component in ordered_clean:
        size = len(component["members"])
        split = max(
            _SPLITS,
            key=lambda name: (
                (target[name] - counts[name]) / max(target[name], 1.0),
                -_SPLITS.index(name),
            ),
        )
        assign(component, split)
        counts[split] += size
    return assignments


def recompute_frozen_candidate_connected_splits(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    """Purely recompute the canonical v4 group and split assignments.

    Each row supplies source-derived grouping/overlap fields plus its final
    selected candidates.  Candidate rows may provide ``connectivity_key``
    directly, as the builder does, or provide roleless ``smiles`` so a runner
    can independently derive connectivity.  The seed is deliberately not an
    argument: formal v4 publication always uses the preregistered value.
    """

    return _assign_frozen_splits(
        rows,
        split_seed=DEFAULT_SPLIT_SEED,
    )


def _validate_roleless_row(row: Mapping[str, Any]) -> None:
    if set(row) != _ROLELESS_FIELDS:
        raise NMRV4CandidatePoolError("roleless pool field allowlist mismatch")
    if row.get("schema_version") != ROLELESS_POOL_SCHEMA_VERSION:
        raise NMRV4CandidatePoolError("roleless pool schema changed")
    if row.get("split") not in _SPLITS:
        raise NMRV4CandidatePoolError("roleless pool split is invalid")
    _require_hash(row.get("release_id"), "roleless.release_id")
    _require_text(row.get("record_id"), "roleless.record_id")
    _require_text(row.get("split_group"), "roleless.split_group")
    _require_hash(
        row.get("spectrum_fingerprint_sha256"),
        "roleless.spectrum_fingerprint_sha256",
    )
    if row.get("nucleus") != "13C":
        raise NMRV4CandidatePoolError("roleless pool nucleus must be 13C")
    formula = _require_text(row.get("formula"), "roleless.formula")
    if canonical_formula(formula) != formula:
        raise NMRV4CandidatePoolError("roleless pool formula is not canonical")
    observed = row.get("observed_13c")
    if (
        not isinstance(observed, list)
        or not observed
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in observed
        )
        or [float(value) for value in observed]
        != sorted(float(value) for value in observed)
    ):
        raise NMRV4CandidatePoolError(
            "roleless pool observed_13c must be sorted finite numbers"
        )
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 8:
        raise NMRV4CandidatePoolError(
            "roleless pool must contain 2 through 8 candidates"
        )
    candidate_ids: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping) or set(candidate) != _ROLELESS_CANDIDATE_FIELDS:
            raise NMRV4CandidatePoolError(
                "roleless candidate field allowlist mismatch"
            )
        candidate_ids.append(
            _require_text(candidate.get("candidate_id"), "candidate_id")
        )
        _require_text(candidate.get("smiles"), "candidate.smiles")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise NMRV4CandidatePoolError("roleless pool candidate IDs are not unique")
    if candidate_ids != sorted(candidate_ids):
        raise NMRV4CandidatePoolError(
            "roleless pool candidates must be ordered by opaque ID"
        )
    forbidden_fragments = ("truth", "role", "inchi", "origin", "source")
    for key in _walk_keys(row):
        if any(fragment in key.casefold() for fragment in forbidden_fragments):
            raise NMRV4CandidatePoolError(
                f"roleless pool leaks forbidden field label: {key}"
            )


def _validate_gold_row(row: Mapping[str, Any], *, split: str) -> None:
    if set(row) != _SEALED_GOLD_FIELDS:
        raise NMRV4CandidatePoolError("sealed Gold field allowlist mismatch")
    if (
        row.get("schema_version") != SEALED_GOLD_SCHEMA_VERSION
        or row.get("split") != split
    ):
        raise NMRV4CandidatePoolError("sealed Gold schema/split changed")
    _require_hash(row.get("release_id"), "sealed Gold.release_id")
    _require_text(row.get("record_id"), "sealed Gold.record_id")
    _require_text(row.get("truth_candidate_id"), "sealed Gold.truth_candidate_id")


def _validate_eligibility_row(row: Mapping[str, Any]) -> None:
    if set(row) != _ELIGIBILITY_FIELDS:
        raise NMRV4CandidatePoolError("eligibility field allowlist mismatch")
    if (
        row.get("schema_version") != ELIGIBILITY_SCHEMA_VERSION
        or row.get("evaluated_before_model_scoring") is not True
        or not isinstance(row.get("eligible"), bool)
    ):
        raise NMRV4CandidatePoolError("eligibility semantics changed")
    _require_hash(row.get("release_id"), "eligibility.release_id")
    _require_hash(row.get("base_index_sha256"), "eligibility.base_index_sha256")
    _require_text(row.get("record_id"), "eligibility.record_id")
    reasons = row.get("reason_codes")
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in reasons)
        or reasons != sorted(set(reasons))
        or bool(reasons) is bool(row["eligible"])
    ):
        raise NMRV4CandidatePoolError("eligibility reason binding changed")
    preflight = _require_mapping(
        row.get("preflight_audit"),
        "eligibility.preflight_audit",
    )
    if preflight.get("model_outputs_created") is not False:
        raise NMRV4CandidatePoolError("eligibility contains model output semantics")
    forbidden = _SOURCE_OUTCOME_FIELDS.intersection(
        key.casefold() for key in _walk_keys(row)
    )
    if forbidden:
        raise NMRV4CandidatePoolError(
            f"eligibility contains forbidden outcome fields: {sorted(forbidden)}"
        )


def _validate_split_manifest_row(row: Mapping[str, Any]) -> None:
    if set(row) != _SPLIT_MANIFEST_FIELDS:
        raise NMRV4CandidatePoolError("split manifest field allowlist mismatch")
    if (
        row.get("schema_version") != SPLIT_MANIFEST_SCHEMA_VERSION
        or row.get("split") not in _SPLITS
    ):
        raise NMRV4CandidatePoolError("split manifest schema/split changed")
    for field in (
        "release_id",
        "spectrum_fingerprint_sha256",
        "candidate_set_sha256",
        "roleless_pool_row_sha256",
    ):
        _require_hash(row.get(field), f"split manifest.{field}")
    _require_text(row.get("record_id"), "split manifest.record_id")
    _require_text(row.get("split_group"), "split manifest.split_group")
    for field in (
        "dp5_upstream_structure_overlap",
        "dp5_upstream_scaffold_overlap",
        "base_index_structure_overlap",
        "nmrexp_overlap",
    ):
        if not isinstance(row.get(field), bool):
            raise NMRV4CandidatePoolError(
                f"split manifest.{field} must be boolean"
            )
    if row["split"] == "test" and (
        row["dp5_upstream_structure_overlap"]
        or row["dp5_upstream_scaffold_overlap"]
        or row["base_index_structure_overlap"]
        or row["nmrexp_overlap"]
    ):
        raise NMRV4CandidatePoolError(
            "split manifest places upstream/base structure overlap in final test"
        )


def _validate_implementation_binding(value: Any) -> dict[str, Any]:
    binding = _require_mapping(value, "implementation_binding")
    if set(binding) != {
        "schema_version",
        "files",
        "host_runtime",
        "binding_sha256",
    } or binding.get("schema_version") != IMPLEMENTATION_BINDING_SCHEMA_VERSION:
        raise NMRV4CandidatePoolError(
            "implementation binding field allowlist/schema mismatch"
        )
    files = binding.get("files")
    if (
        not isinstance(files, list)
        or [item.get("path") for item in files if isinstance(item, Mapping)]
        != list(_IMPLEMENTATION_FILES)
        or any(
            not isinstance(item, Mapping)
            or set(item) != {"path", "bytes", "sha256"}
            or isinstance(item.get("bytes"), bool)
            or not isinstance(item.get("bytes"), int)
            or item["bytes"] < 0
            or _HASH_RE.fullmatch(str(item.get("sha256"))) is None
            for item in files
        )
    ):
        raise NMRV4CandidatePoolError("implementation file binding changed")
    host = _require_mapping(
        binding.get("host_runtime"),
        "implementation_binding.host_runtime",
    )
    if set(host) != {
        "python_implementation",
        "python_version",
        "rdkit_version",
    } or any(not isinstance(value, str) or not value for value in host.values()):
        raise NMRV4CandidatePoolError("implementation host runtime changed")
    core = {
        "schema_version": binding["schema_version"],
        "files": list(files),
        "host_runtime": dict(host),
    }
    if (
        _require_hash(
            binding.get("binding_sha256"),
            "implementation_binding.binding_sha256",
        )
        != canonical_sha256(core)
    ):
        raise NMRV4CandidatePoolError("implementation binding SHA-256 mismatch")
    return dict(binding)


def _assert_candidate_partition(
    rows: Sequence[Mapping[str, Any]],
    *,
    identifiers: Callable[
        [Mapping[str, Any]],
        tuple[frozenset[str], frozenset[str]],
    ],
) -> None:
    """Require each selected structure to belong to one bootstrap group."""

    owners: dict[str, dict[str, tuple[str, str]]] = {
        "candidate_id": {},
        "candidate_connectivity_key": {},
    }
    for row in rows:
        split = str(row["split"])
        split_group = str(row["split_group"])
        candidate_ids, connectivity_keys = identifiers(row)
        values_by_dimension = {
            "candidate_id": candidate_ids,
            "candidate_connectivity_key": connectivity_keys,
        }
        for dimension, values in values_by_dimension.items():
            for value in values:
                previous = owners[dimension].setdefault(
                    value,
                    (split, split_group),
                )
                if previous[0] != split:
                    raise NMRV4CandidatePoolError(
                        f"{dimension} split leakage between "
                        f"{previous[0]} and {split}"
                    )
                if previous[1] != split_group:
                    raise NMRV4CandidatePoolError(
                        f"{dimension} appears in multiple split_group "
                        "bootstrap units"
                    )


def _roleless_candidate_identifiers(
    row: Mapping[str, Any],
) -> tuple[frozenset[str], frozenset[str]]:
    raw_candidates = row.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise NMRV4CandidatePoolError(
            "roleless candidate identity audit requires candidates"
        )
    candidate_ids: set[str] = set()
    connectivity_keys: set[str] = set()
    for index, raw_candidate in enumerate(raw_candidates):
        candidate = _require_mapping(
            raw_candidate,
            f"roleless.candidates[{index}]",
        )
        candidate_id = _require_text(
            candidate.get("candidate_id"),
            f"roleless.candidates[{index}].candidate_id",
        )
        smiles = _require_text(
            candidate.get("smiles"),
            f"roleless.candidates[{index}].smiles",
        )
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
            raise NMRV4CandidatePoolError(
                "roleless candidate identity audit could not parse SMILES"
            )
        inchi_key = Chem.MolToInchiKey(molecule)
        if not inchi_key:
            raise NMRV4CandidatePoolError(
                "roleless candidate identity audit could not derive InChIKey"
            )
        connectivity_key = inchi_key[:14]
        if candidate_id in candidate_ids:
            raise NMRV4CandidatePoolError(
                "roleless pool contains a duplicate candidate_id"
            )
        if connectivity_key in connectivity_keys:
            raise NMRV4CandidatePoolError(
                "roleless pool contains duplicate candidate connectivity"
            )
        candidate_ids.add(candidate_id)
        connectivity_keys.add(connectivity_key)
    return frozenset(candidate_ids), frozenset(connectivity_keys)


def _assert_split_isolation(rows: Sequence[Mapping[str, Any]]) -> None:
    dimensions = (
        "connectivity_key",
        "spectrum_fingerprint_sha256",
        "scaffold_smiles",
    )
    by_split = {
        split: [row for row in rows if row["split"] == split] for split in _SPLITS
    }
    for left_index, left in enumerate(_SPLITS):
        for right in _SPLITS[left_index + 1 :]:
            for dimension in dimensions:
                left_values = {
                    str(row[dimension])
                    for row in by_split[left]
                    if str(row[dimension])
                }
                right_values = {
                    str(row[dimension])
                    for row in by_split[right]
                    if str(row[dimension])
                }
                if left_values & right_values:
                    raise NMRV4CandidatePoolError(
                        f"{dimension} split leakage between {left} and {right}"
                    )
    _assert_candidate_partition(
        rows,
        identifiers=_selected_candidate_identifiers,
    )
    for row in by_split["test"]:
        if (
            row["dp5_upstream_structure_overlap"]
            or row["dp5_upstream_scaffold_overlap"]
            or row["base_index_structure_overlap"]
            or row["nmrexp_overlap"]
        ):
            raise NMRV4CandidatePoolError(
                "final test contains upstream/base structure/scaffold overlap"
            )


def _build_v4_candidate_pool_artifacts_core(
    source_records: Sequence[Mapping[str, Any]],
    *,
    release_id: str,
    base_index_path: str | Path,
    preflighter: ConformerPreflighter,
    source_binding: Mapping[str, Any] | None = None,
    max_decoys: int = DEFAULT_MAX_DECOYS,
    split_seed: str = DEFAULT_SPLIT_SEED,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Testable core; formal callers must use the non-injectable wrapper."""

    implementation_at_start = implementation_binding()
    release = _require_hash(release_id, "release_id")
    if (
        isinstance(max_decoys, bool)
        or not isinstance(max_decoys, int)
        or not 1 <= max_decoys <= DEFAULT_MAX_DECOYS
    ):
        raise NMRV4CandidatePoolError(
            f"max_decoys must be between 1 and {DEFAULT_MAX_DECOYS}"
        )
    seed = _require_text(split_seed, "split_seed")
    index_path = Path(base_index_path).resolve()
    if not index_path.is_file():
        raise NMRV4CandidatePoolError(f"base index does not exist: {index_path}")
    index_hash = file_sha256(index_path)
    connection = connect_readonly(index_path)
    try:
        index_schema = schema_version(connection)
        normalized: list[dict[str, Any]] = []
        eligibility: list[dict[str, Any]] = []
        seen_record_ids: set[str] = set()
        global_preflight_runtime: str | None = None
        base_connectivity_cache: dict[str, frozenset[str]] = {}
        source_dp5_independent = 0
        source_dp5_and_base_independent = 0
        for ordinal, raw in enumerate(source_records, start=1):
            if not isinstance(raw, Mapping):
                raise NMRV4CandidatePoolError(
                    f"source_records[{ordinal - 1}] must be an object"
                )
            record = _normalise_source_record(raw)
            record_id = str(record["record_id"])
            if record_id in seen_record_ids:
                raise NMRV4CandidatePoolError(
                    f"duplicate source record_id: {record_id}"
                )
            seen_record_ids.add(record_id)
            reasons = list(record["normalization_reasons"])
            base_index_structure_overlap = str(record["connectivity_key"]) in (
                _base_formula_connectivity_keys(
                    connection,
                    formula=str(record["formula"]),
                    cache=base_connectivity_cache,
                )
            )
            record["base_index_structure_overlap"] = (
                base_index_structure_overlap
            )
            dp5_independent = not (
                record["dp5_upstream_structure_overlap"]
                or record["dp5_upstream_scaffold_overlap"]
            )
            source_dp5_independent += int(dp5_independent)
            source_dp5_and_base_independent += int(
                dp5_independent and not base_index_structure_overlap
            )
            decoy_audit: dict[str, Any] = {}
            selected_decoys: list[dict[str, Any]] = []
            preflight_audit: dict[str, Any] = {
                "candidate_count_preflighted": 0,
                "candidate_count_not_needed_after_cap": 0,
                "preflight_rejections_by_reason": {},
                "preflight_runtime_sha256": None,
                "model_outputs_created": False,
            }
            if not reasons:
                truth = record["candidate"]
                decoys, decoy_audit = _base_decoys(
                    connection,
                    formula=str(record["formula"]),
                    truth_connectivity_key=str(truth["connectivity_key"]),
                )
                if not decoys:
                    reasons.append(
                        "no_same_formula_different_connectivity_decoy"
                    )
                else:
                    ranked = _hard_decoy_order(truth, decoys)
                    truth_passed, selected_decoys, preflight_audit = (
                        _preflight_hard_candidates(
                            preflighter,
                            truth=truth,
                            ranked_decoys=ranked,
                            formula=str(record["formula"]),
                            max_decoys=max_decoys,
                        )
                    )
                    runtime_hash = preflight_audit["preflight_runtime_sha256"]
                    if global_preflight_runtime is None:
                        global_preflight_runtime = runtime_hash
                    elif runtime_hash != global_preflight_runtime:
                        raise NMRV4CandidatePoolError(
                            "conformer preflight runtime changed during build"
                        )
                    if not truth_passed:
                        reasons.append("truth_failed_conformer_preflight")
                    if not selected_decoys:
                        reasons.append("no_decoy_passed_conformer_preflight")
            eligible = not reasons
            audit_row = {
                "schema_version": ELIGIBILITY_SCHEMA_VERSION,
                "release_id": release,
                "record_id": record_id,
                "evaluated_before_model_scoring": True,
                "eligible": eligible,
                "reason_codes": sorted(set(reasons)),
                "base_index_sha256": index_hash,
                "candidate_generation_version": CANDIDATE_GENERATION_VERSION,
                "max_decoys": max_decoys,
                "base_decoy_audit": decoy_audit,
                "preflight_audit": preflight_audit,
                "base_index_structure_overlap": base_index_structure_overlap,
                # Per-candidate similarities/identities are intentionally not
                # published here: subtracting that list from the roleless pool
                # would disclose the held-out target before Gold is unsealed.
                "selected_hard_decoy_count": len(selected_decoys),
            }
            eligibility.append(audit_row)
            if eligible:
                truth = record["candidate"]
                normalized.append(
                    {
                        **record,
                        "truth_candidate_id": str(truth["candidate_id"]),
                        "connectivity_key": str(truth["connectivity_key"]),
                        "selected_candidates": [truth, *selected_decoys],
                        "candidate_set_sha256": canonical_sha256(
                            sorted(
                                [
                                    {
                                        "candidate_id": str(candidate["candidate_id"]),
                                        "smiles": str(candidate["smiles"]),
                                    }
                                    for candidate in [truth, *selected_decoys]
                                ],
                                key=lambda candidate: candidate["candidate_id"],
                            )
                        ),
                    }
                )
            if progress is not None:
                progress(
                    {
                        "stage": "outcome_free_candidate_pool",
                        "record": ordinal,
                        "total": len(source_records),
                        "record_id": record_id,
                        "eligible": eligible,
                    }
                )
    finally:
        connection.close()
    if file_sha256(index_path) != index_hash:
        raise NMRV4CandidatePoolError(
            "base index changed while candidate pools were generated"
        )
    if not normalized:
        raise NMRV4CandidatePoolError("source produced no eligible candidate pools")

    if seed == DEFAULT_SPLIT_SEED:
        assignments = recompute_frozen_candidate_connected_splits(normalized)
    else:
        assignments = _assign_frozen_splits(normalized, split_seed=seed)
    for row in normalized:
        row.update(assignments[str(row["record_id"])])
    _assert_split_isolation(normalized)

    roleless_rows: list[dict[str, Any]] = []
    gold_by_split: dict[str, list[dict[str, Any]]] = {
        split: [] for split in _SPLITS
    }
    split_rows: list[dict[str, Any]] = []
    for row in sorted(
        normalized,
        key=lambda item: (_SPLITS.index(str(item["split"])), str(item["record_id"])),
    ):
        candidates = sorted(
            [
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "smiles": str(candidate["smiles"]),
                }
                for candidate in row["selected_candidates"]
            ],
            key=lambda candidate: candidate["candidate_id"],
        )
        roleless = {
            "schema_version": ROLELESS_POOL_SCHEMA_VERSION,
            "release_id": release,
            "record_id": str(row["record_id"]),
            "split": str(row["split"]),
            "split_group": str(row["split_group"]),
            "spectrum_fingerprint_sha256": str(
                row["spectrum_fingerprint_sha256"]
            ),
            "nucleus": "13C",
            "formula": str(row["formula"]),
            "observed_13c": list(row["observed_13c"]),
            "candidates": candidates,
        }
        _validate_roleless_row(roleless)
        gold = {
            "schema_version": SEALED_GOLD_SCHEMA_VERSION,
            "release_id": release,
            "record_id": str(row["record_id"]),
            "split": str(row["split"]),
            "truth_candidate_id": str(row["truth_candidate_id"]),
        }
        _validate_gold_row(gold, split=str(row["split"]))
        if gold["truth_candidate_id"] not in {
            candidate["candidate_id"] for candidate in candidates
        }:
            raise NMRV4CandidatePoolError("sealed Gold target is absent from pool")
        roleless_rows.append(roleless)
        gold_by_split[str(row["split"])].append(gold)
        split_rows.append(
            {
                "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
                "release_id": release,
                "record_id": str(row["record_id"]),
                "split": str(row["split"]),
                "split_group": str(row["split_group"]),
                "spectrum_fingerprint_sha256": str(
                    row["spectrum_fingerprint_sha256"]
                ),
                "dp5_upstream_structure_overlap": bool(
                    row["dp5_upstream_structure_overlap"]
                ),
                "dp5_upstream_scaffold_overlap": bool(
                    row["dp5_upstream_scaffold_overlap"]
                ),
                "base_index_structure_overlap": bool(
                    row["base_index_structure_overlap"]
                ),
                "nmrexp_overlap": bool(row["nmrexp_overlap"]),
                "candidate_set_sha256": str(row["candidate_set_sha256"]),
                "roleless_pool_row_sha256": canonical_sha256(roleless),
            }
        )

    eligibility.sort(key=lambda row: str(row["record_id"]))
    split_rows.sort(key=lambda row: str(row["record_id"]))
    for split in _SPLITS:
        gold_by_split[split].sort(key=lambda row: str(row["record_id"]))
    payloads = {
        "roleless-pools.jsonl": _jsonl_bytes(roleless_rows),
        "sealed-gold-dev.jsonl": _jsonl_bytes(gold_by_split["dev"]),
        "sealed-gold-calibration.jsonl": _jsonl_bytes(
            gold_by_split["calibration"]
        ),
        "sealed-gold-test.jsonl": _jsonl_bytes(gold_by_split["test"]),
        "eligibility.jsonl": _jsonl_bytes(eligibility),
        "split-manifest.jsonl": _jsonl_bytes(split_rows),
    }
    artifact_bindings = {
        name: {
            "sha256": _bytes_sha256(payload),
            "bytes": len(payload),
            "rows": payload.count(b"\n"),
        }
        for name, payload in sorted(payloads.items())
    }
    counts = {
        "source_records": len(source_records),
        "eligible_records": len(normalized),
        "ineligible_records": len(source_records) - len(normalized),
        "by_split": dict(
            sorted(Counter(str(row["split"]) for row in normalized).items())
        ),
        "split_groups": len({str(row["split_group"]) for row in normalized}),
        "source_dp5_structure_scaffold_independent_records": (
            source_dp5_independent
        ),
        "source_dp5_and_base_structure_independent_records": (
            source_dp5_and_base_independent
        ),
        "source_base_index_structure_overlap_records": sum(
            bool(audit["base_index_structure_overlap"])
            for audit in eligibility
        ),
        "dp5_structure_overlap_forced_dev": sum(
            bool(row["dp5_upstream_structure_overlap"]) for row in normalized
        ),
        "dp5_scaffold_overlap_forced_dev": sum(
            bool(row["dp5_upstream_scaffold_overlap"]) for row in normalized
        ),
        "base_index_structure_overlap_forced_dev": sum(
            bool(row["base_index_structure_overlap"]) for row in normalized
        ),
        "nmrexp_overlap_forced_dev": sum(
            bool(row["nmrexp_overlap"]) for row in normalized
        ),
    }
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "release_id": release,
        "outcome_free": True,
        "model_scores_created": False,
        "calibrator_fitted": False,
        "candidate_generation": {
            "version": CANDIDATE_GENERATION_VERSION,
            "formula_relation": "exact_canonical_formula",
            "connectivity_relation": "different_first_14_inchi_key",
            "hardness_metric": "Morgan radius=2 fpSize=2048 Tanimoto to target",
            "selection_order": (
                "descending_tanimoto_then_ascending_opaque_candidate_id"
            ),
            "maximum_decoys": max_decoys,
            "preflight_operation": "conformer_preflight_only",
            "preflight_batch_size": MAX_PREFLIGHT_CANDIDATES,
            "preflight_runtime_sha256": global_preflight_runtime,
            "rdkit_version": rdBase.rdkitVersion,
        },
        "split_protocol": {
            "version": SPLIT_PROTOCOL_VERSION,
            "seed": seed,
            "connected_by": [
                "duplicate_molecule_group",
                "connectivity_key",
                "nonempty_murcko_scaffold_smiles",
                "spectrum_fingerprint_sha256",
                "all_selected_candidate_ids",
                "all_selected_candidate_connectivity_keys",
            ],
            "bootstrap_unit": "split_group_connected_component",
            "candidate_identity_isolation": {
                "dimensions": ["candidate_id", "connectivity_key"],
                "cross_split_intersections_must_be_empty": True,
                "shared_identity_requires_same_split_group": True,
            },
            "target_ratios": {"dev": 0.4, "calibration": 0.3, "test": 0.3},
            "overlap_policy": (
                "any NMRexp/DP5 structure/scaffold or base-index structure "
                "overlap forces the whole connected component to dev"
            ),
            "test_requires_all_upstream_and_base_overlap_flags_false": True,
        },
        "source_binding": dict(source_binding or {}),
        "base_index": {
            "schema_version": index_schema,
            "sha256": index_hash,
        },
        "counts": counts,
        "ineligibility_reason_counts": dict(
            sorted(
                Counter(
                    reason
                    for row in eligibility
                    for reason in row["reason_codes"]
                ).items()
            )
        ),
        "artifacts": artifact_bindings,
        "implementation_binding": implementation_at_start,
        "formal_protocol": {
            "maximum_decoys": DEFAULT_MAX_DECOYS,
            "split_seed": DEFAULT_SPLIT_SEED,
            "parameter_shopping_prohibited": True,
            "preflighter_injection_prohibited": True,
        },
    }
    implementation_at_end = implementation_binding()
    if implementation_at_end != implementation_at_start:
        raise NMRV4CandidatePoolError(
            "candidate-pool implementation changed during build"
        )
    return {
        "roleless_pools": roleless_rows,
        "sealed_gold": gold_by_split,
        "eligibility": eligibility,
        "split_manifest": split_rows,
        "summary": summary,
    }


def build_v4_candidate_pool_artifacts(
    *,
    source_current_path: str | Path,
    base_index_path: str | Path,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build the formal frozen v4 pool with no tunable protocol choices.

    The public entry point owns source loading and the real DP5q adapter.  This
    prevents callers from shopping the split seed or decoy cap, and prevents a
    caller-supplied fake preflighter from masquerading as the formal build.
    Tests use the private core above.
    """

    current_path = Path(source_current_path).resolve()
    source_records, source_binding = load_reviewed_release(current_path)
    source_records_sha256 = canonical_sha256(source_records)
    config = NMRForwardConfig.from_environment()
    with NMRForwardAdapter(config) as preflighter:
        artifacts = _build_v4_candidate_pool_artifacts_core(
            source_records,
            release_id=str(source_binding["release_id"]),
            base_index_path=base_index_path,
            preflighter=preflighter,
            source_binding=source_binding,
            max_decoys=DEFAULT_MAX_DECOYS,
            split_seed=DEFAULT_SPLIT_SEED,
            progress=progress,
        )
    final_records, final_binding = load_reviewed_release(current_path)
    if (
        final_binding != source_binding
        or canonical_sha256(final_records) != source_records_sha256
    ):
        raise NMRV4CandidatePoolError(
            "reviewed source changed while candidate pools were generated"
        )
    return artifacts


def _artifact_payloads(artifacts: Mapping[str, Any]) -> dict[str, bytes]:
    gold = _require_mapping(artifacts.get("sealed_gold"), "sealed_gold")
    if set(gold) != set(_SPLITS):
        raise NMRV4CandidatePoolError("sealed Gold split allowlist mismatch")
    roleless_rows = artifacts.get("roleless_pools")
    eligibility_rows = artifacts.get("eligibility")
    split_rows = artifacts.get("split_manifest")
    if (
        not isinstance(roleless_rows, list)
        or not isinstance(eligibility_rows, list)
        or not isinstance(split_rows, list)
        or any(not isinstance(gold.get(split), list) for split in _SPLITS)
    ):
        raise NMRV4CandidatePoolError("artifact row collections are invalid")
    for row in roleless_rows:
        _validate_roleless_row(_require_mapping(row, "roleless pool row"))
    for row in eligibility_rows:
        _validate_eligibility_row(_require_mapping(row, "eligibility row"))
    for row in split_rows:
        _validate_split_manifest_row(_require_mapping(row, "split manifest row"))
    for split in _SPLITS:
        for row in gold[split]:
            _validate_gold_row(
                _require_mapping(row, f"sealed Gold {split} row"),
                split=split,
            )
    _assert_candidate_partition(
        roleless_rows,
        identifiers=_roleless_candidate_identifiers,
    )

    roleless_by_id = {
        str(row["record_id"]): row for row in roleless_rows
    }
    split_by_id = {str(row["record_id"]): row for row in split_rows}
    gold_by_id = {
        str(row["record_id"]): row
        for split in _SPLITS
        for row in gold[split]
    }
    expected_count = len(roleless_rows)
    if (
        len(roleless_by_id) != expected_count
        or len(split_by_id) != expected_count
        or len(gold_by_id) != expected_count
        or not (
            set(roleless_by_id) == set(split_by_id) == set(gold_by_id)
        )
    ):
        raise NMRV4CandidatePoolError(
            "roleless, split-manifest, and sealed-Gold record sets differ"
        )
    for record_id, roleless in roleless_by_id.items():
        split_row = split_by_id[record_id]
        gold_row = gold_by_id[record_id]
        if (
            split_row["split"] != roleless["split"]
            or gold_row["split"] != roleless["split"]
            or split_row["split_group"] != roleless["split_group"]
            or split_row["roleless_pool_row_sha256"]
            != canonical_sha256(roleless)
            or gold_row["truth_candidate_id"]
            not in {
                candidate["candidate_id"]
                for candidate in roleless["candidates"]
            }
        ):
            raise NMRV4CandidatePoolError(
                f"{record_id}: artifact row bindings changed"
            )

    summary = _require_mapping(artifacts.get("summary"), "summary")
    if (
        set(summary) != _SUMMARY_FIELDS
        or summary.get("schema_version") != SUMMARY_SCHEMA_VERSION
        or summary.get("outcome_free") is not True
        or summary.get("model_scores_created") is not False
        or summary.get("calibrator_fitted") is not False
    ):
        raise NMRV4CandidatePoolError("summary schema/semantics changed")
    frozen_implementation = _validate_implementation_binding(
        summary.get("implementation_binding")
    )
    if frozen_implementation != implementation_binding():
        raise NMRV4CandidatePoolError(
            "current implementation differs from the frozen build binding"
        )
    release_id = _require_hash(summary.get("release_id"), "summary.release_id")
    if any(row["release_id"] != release_id for row in roleless_rows):
        raise NMRV4CandidatePoolError("summary release binding changed")
    payloads = {
        "roleless-pools.jsonl": _jsonl_bytes(roleless_rows),
        "sealed-gold-dev.jsonl": _jsonl_bytes(gold["dev"]),
        "sealed-gold-calibration.jsonl": _jsonl_bytes(gold["calibration"]),
        "sealed-gold-test.jsonl": _jsonl_bytes(gold["test"]),
        "eligibility.jsonl": _jsonl_bytes(eligibility_rows),
        "split-manifest.jsonl": _jsonl_bytes(split_rows),
    }
    bindings = _require_mapping(summary.get("artifacts"), "summary.artifacts")
    if set(bindings) != set(payloads):
        raise NMRV4CandidatePoolError("summary artifact allowlist mismatch")
    for name, payload in payloads.items():
        binding = _require_mapping(bindings.get(name), f"summary.artifacts.{name}")
        if (
            binding.get("sha256") != _bytes_sha256(payload)
            or binding.get("bytes") != len(payload)
            or binding.get("rows") != payload.count(b"\n")
        ):
            raise NMRV4CandidatePoolError(f"{name} summary binding mismatch")
    payloads["summary.json"] = _json_bytes(summary)
    return payloads


def write_v4_candidate_pool_artifacts(
    destination: str | Path,
    artifacts: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Atomically publish a complete artifact directory, refusing overwrite."""

    root = Path(destination).resolve()
    payloads = _artifact_payloads(artifacts)
    frozen_implementation = dict(
        _require_mapping(
            _require_mapping(artifacts.get("summary"), "summary").get(
                "implementation_binding"
            ),
            "summary.implementation_binding",
        )
    )
    targets = {name: root / name for name in payloads}
    if root.exists():
        if not root.is_dir():
            raise NMRV4CandidatePoolError(f"destination is not a directory: {root}")
        existing_names = {
            path.name for path in root.iterdir() if path.is_file()
        }
        expected_names = set(payloads)
        identical = existing_names == expected_names and all(
            targets[name].read_bytes() == payload for name, payload in payloads.items()
        )
        if identical:
            return targets
        if not overwrite:
            raise NMRV4CandidatePoolError(
                f"existing artifact set differs: {root}; use --overwrite only "
                "after review"
            )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent)
    )
    backup: Path | None = None
    try:
        for name, payload in payloads.items():
            target = staging / name
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if implementation_binding() != frozen_implementation:
            raise NMRV4CandidatePoolError(
                "implementation changed while artifacts were staged"
            )
        if root.exists():
            backup = root.with_name(
                f".{root.name}.backup-{hashlib.sha256(os.urandom(32)).hexdigest()[:12]}"
            )
            os.replace(root, backup)
        try:
            os.replace(staging, root)
        except Exception:
            if backup is not None and backup.exists() and not root.exists():
                os.replace(backup, root)
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if backup is not None and backup.exists() and not root.exists():
            os.replace(backup, root)
    return targets


__all__ = [
    "CANDIDATE_GENERATION_VERSION",
    "DEFAULT_MAX_DECOYS",
    "DEFAULT_SPLIT_SEED",
    "ELIGIBILITY_SCHEMA_VERSION",
    "IMPLEMENTATION_BINDING_SCHEMA_VERSION",
    "NMRV4CandidatePoolError",
    "ROLELESS_POOL_SCHEMA_VERSION",
    "SEALED_GOLD_SCHEMA_VERSION",
    "SPLIT_MANIFEST_SCHEMA_VERSION",
    "SUMMARY_SCHEMA_VERSION",
    "build_v4_candidate_pool_artifacts",
    "canonical_json_dumps",
    "canonical_sha256",
    "file_sha256",
    "implementation_binding",
    "load_reviewed_release",
    "recompute_frozen_candidate_connected_splits",
    "write_v4_candidate_pool_artifacts",
]
