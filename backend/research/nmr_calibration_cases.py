"""Build frozen, auditable 13C forward-score cases without fitting a calibrator.

The input is the independently derived NMRexp strict pool.  Eligibility is
decided before any DP5q call.  Every scoreable candidate pool contains the
human-corrected structure plus same-formula, different-connectivity decoys from
the read-only v2 base index.

The scorer receives opaque candidate IDs and structures only.  It never
receives the truth role, extraction-review labels, or a model outcome.  The
truth binding is applied after ``NMRForwardAdapter.score_candidates`` returns.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Callable, Protocol

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors

from app.ml.independent_nmr_data import (
    DERIVED_RELEASE_SCHEMA_VERSION,
    DERIVED_SCHEMA_VERSION,
    PARENT_STANDARDIZATION_VERSION,
)
from app.ml.nmr_data_v2 import connect_readonly, schema_version
from app.ml.nmr_evidence import FormulaError, canonical_formula
from app.ml.nmr_forward import (
    DP5Q_CONFORMER_PREFLIGHT_PROTOCOL_VERSION,
    DP5Q_CONFORMER_PROTOCOL_VERSION,
    DP5Q_MEAN_MODEL_SHA256,
    DP5Q_PREPROCESSOR_SHA256,
    DP5Q_REPOSITORY_COMMIT,
    PROTOCOL_VERSION as DP5Q_SIDECAR_PROTOCOL_VERSION,
    dp5q_conformer_policy,
)


PRE_SCORE_SCHEMA_VERSION = "chemapp.nmr.pre-score-eligibility.v3"
CANDIDATE_POOL_SCHEMA_VERSION = "chemapp.nmr.candidate-pool.v3"
SCOREABLE_MANIFEST_SCHEMA_VERSION = "chemapp.nmr.scoreable-source-manifest.v3"
FROZEN_SCORE_CASE_SCHEMA_VERSION = "chemapp.nmr.frozen-score-case.v3"
SUMMARY_SCHEMA_VERSION = "chemapp.nmr.calibration-case-build-summary.v3"
CRITERION_VERSION = (
    "strict-independent-parent-dp5q-13c-formula-decoys-conformer-preflight-v3"
)
CANDIDATE_GENERATION_VERSION = (
    "base-index-same-formula-parent-connectivity-decoys-preflight-before-cap-v3"
)
SCORE_SEMANTICS = "dp5q_frozen_top1_runnerup_negative_hungarian_mae_margin_v1"

MAX_FORWARD_CANDIDATES = 20
DEFAULT_MAX_DECOYS = MAX_FORWARD_CANDIDATES - 1
MAX_HEAVY_ATOMS = 80
MAX_TOTAL_ATOMS = 200
MAX_ROTATABLE_BONDS = 20
ALLOWED_ATOMIC_NUMBERS = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35})

_FORBIDDEN_PRE_SCORE_FIELDS = frozenset(
    {
        "raw_score",
        "top1_exact_correct",
        "predicted_top1",
        "predicted_top1_inchi_key",
        "outcome",
    }
)


class CalibrationCaseError(ValueError):
    """Raised when source, pool, score, or artifact semantics fail closed."""


class ForwardScorer(Protocol):
    """The public adapter surface used by the case builder."""

    def score_candidates(
        self,
        observed_13c: Sequence[float],
        candidates: Sequence[Mapping[str, Any]],
        *,
        formula: str | None = None,
    ) -> dict[str, Any]: ...


class ConformerPreflighter(Protocol):
    """Model-output-free applicability surface used before candidate capping."""

    def preflight_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        formula: str | None = None,
    ) -> dict[str, Any]: ...


def canonical_json_dumps(value: Any) -> str:
    """Render deterministic JSON suitable for artifact hashing."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def artifact_json_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the exact deterministic bytes used for JSON artifact binding."""

    return (canonical_json_dumps(dict(value)) + "\n").encode("utf-8")


def artifact_json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(artifact_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_dumps(dict(row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationCaseError(f"{field} must be a non-empty string")
    return value.strip()


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CalibrationCaseError(f"{field} must be an object")
    return value


def load_strict_source_records(
    path: str | Path,
    *,
    source_summary: Mapping[str, Any],
    base_index_path: str | Path,
    source_current_sha256: str,
) -> tuple[list[dict[str, Any]], int, dict[str, Any]]:
    """Load strict records only after verifying the v2 release and base index.

    Non-strict rows remain outside this builder.  Both 1H and 13C strict rows
    are retained so the pre-score eligibility artifact accounts for the full
    independent pool.
    """

    source_path = Path(path).resolve()
    index_path = Path(base_index_path).resolve()
    publication = source_summary.get("publication")
    overlap_summary = source_summary.get("overlap")
    if (
        source_summary.get("schema_version") != DERIVED_SCHEMA_VERSION
        or not isinstance(publication, Mapping)
        or publication.get("schema_version") != DERIVED_RELEASE_SCHEMA_VERSION
        or not isinstance(overlap_summary, Mapping)
    ):
        raise CalibrationCaseError("source summary is not a verified v2 release")
    records_binding = publication.get("records")
    release_id = publication.get("release_id")
    if (
        not isinstance(records_binding, Mapping)
        or records_binding.get("file") != "records.jsonl"
        or source_path.name != "records.jsonl"
        or source_path.parent.name != release_id
        or records_binding.get("sha256") != file_sha256(source_path)
        or not isinstance(source_current_sha256, str)
        or len(source_current_sha256) != 64
    ):
        raise CalibrationCaseError("source records are not bound to the v2 release")
    actual_base_index_sha256 = file_sha256(index_path)
    if (
        overlap_summary.get("status") != "checked"
        or overlap_summary.get("base_index_sha256") != actual_base_index_sha256
        or overlap_summary.get("parent_standardization_version")
        != PARENT_STANDARDIZATION_VERSION
    ):
        raise CalibrationCaseError(
            "source overlap proof does not bind the selected base index"
        )
    rows: list[dict[str, Any]] = []
    total_rows = 0
    seen: set[str] = set()
    try:
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                total_rows += 1
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise CalibrationCaseError(
                        f"{source_path}:{line_number}: invalid JSON"
                    ) from exc
                if not isinstance(value, dict):
                    raise CalibrationCaseError(
                        f"{source_path}:{line_number}: row must be an object"
                    )
                if value.get("schema_version") != DERIVED_SCHEMA_VERSION:
                    raise CalibrationCaseError(
                        f"{source_path}:{line_number}: source schema changed"
                    )
                calibration = _mapping(
                    value.get("calibration"),
                    f"{source_path}:{line_number}.calibration",
                )
                if calibration.get("strict_independent_eligible") is not True:
                    continue
                if (
                    calibration.get("base_index_connectivity_overlap") is not False
                    or calibration.get("base_index_exact_inchi_key_overlap")
                    is not False
                    or calibration.get("base_index_parent_connectivity_overlap")
                    is not False
                    or calibration.get("base_index_exact_parent_inchi_key_overlap")
                    is not False
                    or calibration.get("base_index_overlap_status") != "checked"
                    or calibration.get("parent_eligible") is not True
                    or calibration.get("parent_standardization_version")
                    != PARENT_STANDARDIZATION_VERSION
                ):
                    raise CalibrationCaseError(
                        f"{source_path}:{line_number}: strict row lacks a checked "
                        "non-overlap proof"
                    )
                structure = _mapping(
                    value.get("structure"),
                    f"{source_path}:{line_number}.structure",
                )
                if (
                    structure.get("parent_eligible") is not True
                    or structure.get("parent_status")
                    != "eligible_single_organic_parent"
                    or structure.get("parent_standardization_version")
                    != PARENT_STANDARDIZATION_VERSION
                ):
                    raise CalibrationCaseError(
                        f"{source_path}:{line_number}: strict row is not a "
                        "standardized eligible single-organic parent"
                    )
                record_id = _text(
                    value.get("record_id"),
                    f"{source_path}:{line_number}.record_id",
                )
                if record_id in seen:
                    raise CalibrationCaseError(
                        f"{source_path}:{line_number}: duplicate record_id {record_id}"
                    )
                seen.add(record_id)
                rows.append(value)
    except CalibrationCaseError:
        raise
    except (OSError, UnicodeError) as exc:
        raise CalibrationCaseError(f"cannot load source records: {exc}") from exc
    rows.sort(key=lambda row: str(row["record_id"]))
    if not rows:
        raise CalibrationCaseError("strict-independent source pool is empty")
    if records_binding.get("count") != total_rows:
        raise CalibrationCaseError("source publication record count changed")
    return (
        rows,
        total_rows,
        {
            "schema_version": "chemapp.nmr.calibration-source-release-binding.v2",
            "release_id": release_id,
            "records_sha256": records_binding["sha256"],
            "records_count": records_binding["count"],
            "source_current_sha256": source_current_sha256,
            "base_index_sha256": actual_base_index_sha256,
            "parent_standardization_version": PARENT_STANDARDIZATION_VERSION,
        },
    )


def _domain_config(*, max_decoys: int) -> dict[str, Any]:
    return {
        "criterion_version": CRITERION_VERSION,
        "required_nucleus": "13C",
        "forward_model": "DP5q-CASCADE-mean",
        "allowed_atomic_numbers": sorted(ALLOWED_ATOMIC_NUMBERS),
        "max_heavy_atoms": MAX_HEAVY_ATOMS,
        "max_total_atoms": MAX_TOTAL_ATOMS,
        "max_rotatable_bonds": MAX_ROTATABLE_BONDS,
        "maximum_candidate_pool_size": MAX_FORWARD_CANDIDATES,
        "maximum_decoys": max_decoys,
        "truth_structure_field": "smiles_actual via training_smiles",
        "decoy_formula_relation": "exact_canonical_formula_match",
        "decoy_structure_relation": "different_first_14_inchi_key_connectivity",
        "conformer_preflight_scope": (
            "all_candidates_of_statically_viable_pools_before_cap"
        ),
    }


def _normalise_max_decoys(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CalibrationCaseError("max_decoys must be an integer")
    if not 1 <= value <= DEFAULT_MAX_DECOYS:
        raise CalibrationCaseError(
            f"max_decoys must be between 1 and {DEFAULT_MAX_DECOYS}"
        )
    return value


def _preflight_runtime_binding(result: Mapping[str, Any]) -> dict[str, Any]:
    if (
        result.get("status") != "ok"
        or result.get("model_outputs_created") is not False
        or result.get("protocol_version") != DP5Q_SIDECAR_PROTOCOL_VERSION
    ):
        raise CalibrationCaseError(
            "conformer preflight returned unsupported or model-derived semantics"
        )
    runtime = _mapping(result.get("runtime"), "conformer preflight runtime")
    sidecar_sha256 = _text(
        runtime.get("sidecar_code_sha256"),
        "conformer preflight runtime.sidecar_code_sha256",
    )
    sidecar_path = Path(__file__).resolve().parents[1] / "scripts" / "dp5q_sidecar.py"
    if sidecar_sha256 != file_sha256(sidecar_path):
        raise CalibrationCaseError(
            "conformer preflight did not execute the shipped DP5q sidecar"
        )
    rdkit_version = _text(
        runtime.get("rdkit_version"),
        "conformer preflight runtime.rdkit_version",
    )
    version_parts = rdkit_version.split(".")
    if len(version_parts) != 3 or any(not part.isdigit() for part in version_parts):
        raise CalibrationCaseError("conformer preflight RDKit version is invalid")
    conformer_generation = _mapping(
        runtime.get("conformer_generation"),
        "conformer preflight runtime.conformer_generation",
    )
    if conformer_generation != dp5q_conformer_policy():
        raise CalibrationCaseError(
            "conformer preflight generation policy is not frozen"
        )
    preflight_policy = _mapping(
        runtime.get("conformer_preflight"),
        "conformer preflight runtime.conformer_preflight",
    )
    expected_preflight_policy = {
        "protocol_version": DP5Q_CONFORMER_PREFLIGHT_PROTOCOL_VERSION,
        "candidate_failures_are_results": True,
        "operational_failures_abort_request": True,
        "uses_same_prepare_candidate_path_as_prediction": True,
    }
    if preflight_policy != expected_preflight_policy:
        raise CalibrationCaseError("conformer preflight policy is not frozen")
    binding = {
        "sidecar_protocol_version": DP5Q_SIDECAR_PROTOCOL_VERSION,
        "sidecar_code_sha256": sidecar_sha256,
        "rdkit_version": rdkit_version,
        "conformer_protocol_version": DP5Q_CONFORMER_PROTOCOL_VERSION,
        "conformer_generation_config_sha256": canonical_sha256(conformer_generation),
        "conformer_preflight_protocol_version": (
            DP5Q_CONFORMER_PREFLIGHT_PROTOCOL_VERSION
        ),
        "conformer_preflight_policy_sha256": canonical_sha256(preflight_policy),
        "model_outputs_created": False,
    }
    return {**binding, "runtime_sha256": canonical_sha256(binding)}


def _run_conformer_preflight(
    preflighter: ConformerPreflighter,
    candidates: Sequence[Mapping[str, Any]],
    *,
    formula: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preflight every candidate in fixed-size protocol chunks, fail closed."""

    roleless = [
        {
            "candidate_id": _text(
                candidate.get("candidate_id"),
                "preflight candidate.candidate_id",
            ),
            "smiles": _text(candidate.get("smiles"), "preflight candidate.smiles"),
        }
        for candidate in candidates
    ]
    results: list[dict[str, Any]] = []
    runtime_binding: dict[str, Any] | None = None
    for offset in range(0, len(roleless), MAX_FORWARD_CANDIDATES):
        chunk = roleless[offset : offset + MAX_FORWARD_CANDIDATES]
        response = preflighter.preflight_candidates(chunk, formula=formula)
        if not isinstance(response, Mapping):
            raise CalibrationCaseError("conformer preflight response is not an object")
        current_binding = _preflight_runtime_binding(response)
        if runtime_binding is None:
            runtime_binding = current_binding
        elif runtime_binding != current_binding:
            raise CalibrationCaseError(
                "conformer preflight runtime changed within one candidate pool"
            )
        chunk_results = response.get("results")
        if not isinstance(chunk_results, list) or len(chunk_results) != len(chunk):
            raise CalibrationCaseError("conformer preflight changed candidate coverage")
        expected_ids = [candidate["candidate_id"] for candidate in chunk]
        actual_ids = [
            str(item.get("candidate_id"))
            for item in chunk_results
            if isinstance(item, Mapping)
        ]
        if actual_ids != expected_ids:
            raise CalibrationCaseError(
                "conformer preflight changed candidate identity or order"
            )
        results.extend(dict(item) for item in chunk_results)
    if roleless and runtime_binding is None:
        raise AssertionError("non-empty conformer preflight lacks runtime binding")
    return results, dict(runtime_binding or {})


def _candidate_id(inchi_key: str) -> str:
    # The ID deliberately contains no truth/decoy label and is the only ID sent
    # to DP5q.  This prevents role leakage through score tie-breaking.
    digest = hashlib.sha256(
        f"chemapp-opaque-nmr-candidate-v1\0{inchi_key}".encode("ascii")
    ).hexdigest()
    return f"candidate-{digest[:24]}"


def _candidate_from_smiles(
    smiles: str,
    *,
    expected_formula: str,
    origin: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    reasons: list[str] = []
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None, ["structure_parse_failed"]
    if len(Chem.GetMolFrags(molecule)) != 1:
        reasons.append("multiple_fragments")
    heavy_atoms = molecule.GetNumHeavyAtoms()
    if heavy_atoms > MAX_HEAVY_ATOMS:
        reasons.append("heavy_atom_limit_exceeded")
    total_atoms = Chem.AddHs(molecule).GetNumAtoms()
    if total_atoms > MAX_TOTAL_ATOMS:
        reasons.append("total_atom_limit_exceeded")
    rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(molecule)
    if rotatable_bonds > MAX_ROTATABLE_BONDS:
        reasons.append("rotatable_bond_limit_exceeded")
    atomic_numbers = sorted({atom.GetAtomicNum() for atom in molecule.GetAtoms()})
    unsupported = sorted(set(atomic_numbers) - ALLOWED_ATOMIC_NUMBERS)
    reasons.extend(f"unsupported_atomic_number_{number}" for number in unsupported)
    carbon_count = sum(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms())
    if carbon_count == 0:
        reasons.append("no_carbon_atoms")
    try:
        actual_formula = canonical_formula(rdMolDescriptors.CalcMolFormula(molecule))
    except FormulaError:
        reasons.append("formula_is_not_supported")
        actual_formula = ""
    if actual_formula != expected_formula:
        reasons.append("formula_mismatch")
    canonical_smiles = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    inchi_key = Chem.MolToInchiKey(molecule)
    if not inchi_key:
        reasons.append("inchi_key_generation_failed")
    if reasons:
        return None, sorted(set(reasons))
    return (
        {
            "candidate_id": _candidate_id(inchi_key),
            "smiles": canonical_smiles,
            "inchi_key": inchi_key,
            "connectivity_key": inchi_key[:14],
            "formula": actual_formula,
            "carbon_count": carbon_count,
            "origin": dict(origin),
        },
        [],
    )


def _record_bindings(record: Mapping[str, Any]) -> dict[str, Any]:
    source = _mapping(record.get("source"), "source")
    structure = _mapping(record.get("structure"), "structure")
    dataset_document_key = _text(
        source.get("document_key"),
        "source.document_key",
    )
    document_doi = _text(
        source.get("document_doi"),
        "source.document_doi",
    ).casefold()
    source_document_key = f"doi:{document_doi}"
    molecule_key = _text(structure.get("molecule_key"), "structure.molecule_key")
    scaffold_key = _text(structure.get("scaffold_key"), "structure.scaffold_key")
    content_hash = _text(
        record.get("source_content_sha256"),
        "source_content_sha256",
    )
    if len(content_hash) != 64 or any(
        character not in "0123456789abcdef" for character in content_hash
    ):
        raise CalibrationCaseError("source_content_sha256 is not lowercase SHA-256")
    return {
        "source_document_key": source_document_key,
        "source_document_dataset_key": dataset_document_key,
        "molecule_key": molecule_key,
        "scaffold_key": scaffold_key,
        "source_content_sha256": content_hash,
        "source_document_sha256": hashlib.sha256(
            source_document_key.encode("utf-8")
        ).hexdigest(),
        "molecule_sha256": hashlib.sha256(molecule_key.encode("utf-8")).hexdigest(),
        "scaffold_sha256": hashlib.sha256(scaffold_key.encode("utf-8")).hexdigest(),
    }


def _truth_candidate(
    record: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, list[str], str, str]:
    structure = _mapping(record.get("structure"), "structure")
    if structure.get("ground_truth_field") != "smiles_actual":
        raise CalibrationCaseError("ground truth must be bound to smiles_actual")
    formula = canonical_formula(_text(structure.get("formula"), "structure.formula"))
    truth_inchi_key = _text(
        structure.get("inchi_key"),
        "structure.inchi_key",
    )
    truth_smiles = _text(
        structure.get("training_smiles"),
        "structure.training_smiles",
    )
    candidate, reasons = _candidate_from_smiles(
        truth_smiles,
        expected_formula=formula,
        origin={"dataset": "NMRexp", "record_id": record["record_id"]},
    )
    if candidate is not None and candidate["inchi_key"] != truth_inchi_key:
        raise CalibrationCaseError(
            f"{record['record_id']}: truth InChIKey does not match training_smiles"
        )
    return candidate, reasons, formula, truth_inchi_key


def _find_decoys(
    connection: sqlite3.Connection,
    *,
    formula: str,
    truth_connectivity_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = connection.execute(
        """
        SELECT id, source_molecule_id, record_sha256, smiles, inchi_key
        FROM molecules
        WHERE formula = ? AND smiles IS NOT NULL
        ORDER BY id
        """,
        (formula,),
    ).fetchall()
    rejection_counts: Counter[str] = Counter()
    by_connectivity: dict[str, dict[str, Any]] = {}
    same_connectivity_rows = 0
    for row in rows:
        candidate, reasons = _candidate_from_smiles(
            str(row["smiles"]),
            expected_formula=formula,
            origin={
                "dataset": "nmrshiftdb2-v2-base-index",
                "base_molecule_id": int(row["id"]),
                "base_source_molecule_id": str(row["source_molecule_id"]),
                "base_record_sha256": str(row["record_sha256"]),
                "base_stored_inchi_key": (
                    str(row["inchi_key"]) if row["inchi_key"] else None
                ),
            },
        )
        if candidate is None:
            rejection_counts.update(reasons)
            continue
        stored_inchi_key = candidate["origin"].get("base_stored_inchi_key")
        computed_inchi_key = str(candidate["inchi_key"])
        if isinstance(stored_inchi_key, str) and stored_inchi_key:
            if stored_inchi_key != computed_inchi_key:
                mismatch_reason = (
                    "base_stored_computed_inchi_key_stereo_mismatch"
                    if stored_inchi_key[:14] == computed_inchi_key[:14]
                    else "base_stored_computed_inchi_key_connectivity_mismatch"
                )
                rejection_counts[mismatch_reason] += 1
                continue
        connectivity = str(candidate["connectivity_key"])
        if connectivity == truth_connectivity_key:
            same_connectivity_rows += 1
            continue
        previous = by_connectivity.get(connectivity)
        if previous is None or (
            int(candidate["origin"]["base_molecule_id"])
            < int(previous["origin"]["base_molecule_id"])
        ):
            by_connectivity[connectivity] = candidate
    decoys = sorted(
        by_connectivity.values(),
        key=lambda candidate: str(candidate["candidate_id"]),
    )
    return decoys, {
        "base_formula_rows": len(rows),
        "same_truth_connectivity_rows_excluded": same_connectivity_rows,
        "invalid_base_rows_by_reason": dict(sorted(rejection_counts.items())),
        "usable_different_connectivity_decoys": len(decoys),
    }


def _observed_shifts(record: Mapping[str, Any]) -> list[float]:
    spectrum = _mapping(record.get("spectrum"), "spectrum")
    peaks = spectrum.get("processed_peaks")
    if not isinstance(peaks, list) or not peaks:
        raise CalibrationCaseError(
            f"{record['record_id']}: processed 13C peaks are empty"
        )
    shifts: list[float] = []
    for index, peak in enumerate(peaks):
        if not isinstance(peak, Mapping):
            raise CalibrationCaseError(
                f"{record['record_id']}: peak {index} is not an object"
            )
        value = peak.get("shift_ppm")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not -20.0 <= float(value) <= 300.0
        ):
            raise CalibrationCaseError(
                f"{record['record_id']}: peak {index} shift is invalid"
            )
        shifts.append(float(value))
    return sorted(shifts)


def build_pre_score_artifacts(
    source_records: Sequence[Mapping[str, Any]],
    *,
    base_index_path: str | Path,
    source_release_binding: Mapping[str, Any],
    preflighter: ConformerPreflighter,
    max_decoys: int = DEFAULT_MAX_DECOYS,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Build eligibility, candidate pools and the scoreable source subset.

    No forward-model output is created here.  Every candidate in a statically
    viable truth-plus-decoy pool is conformer-preflighted before decoy capping.
    """

    selected_max_decoys = _normalise_max_decoys(max_decoys)
    index_path = Path(base_index_path).resolve()
    if not index_path.is_file():
        raise CalibrationCaseError(f"base index does not exist: {index_path}")
    index_hash = file_sha256(index_path)
    if (
        source_release_binding.get("schema_version")
        != "chemapp.nmr.calibration-source-release-binding.v2"
        or source_release_binding.get("base_index_sha256") != index_hash
        or source_release_binding.get("parent_standardization_version")
        != PARENT_STANDARDIZATION_VERSION
        or not all(
            isinstance(source_release_binding.get(field_name), str)
            and len(str(source_release_binding[field_name])) == 64
            for field_name in (
                "release_id",
                "records_sha256",
                "source_current_sha256",
            )
        )
    ):
        raise CalibrationCaseError(
            "source release binding is missing or does not match the base index"
        )
    domain_config = _domain_config(max_decoys=selected_max_decoys)
    domain_hash = canonical_sha256(domain_config)
    generation_config = {
        "version": CANDIDATE_GENERATION_VERSION,
        "base_index_schema_version": "2",
        "base_index_sha256": index_hash,
        "selection": {
            "formula": "exact canonical formula",
            "exclude_connectivity": "truth first 14 InChIKey characters",
            "deduplicate_decoys_by": "computed connectivity key",
            "representative": "lowest base molecule id",
            "preflight": (
                "all candidates of statically viable pools before cap; no model output"
            ),
            "failed_decoy": "exclude transparently before cap",
            "failed_truth": "record is outside the scoreability domain",
            "cap": selected_max_decoys,
            "cap_order": "opaque candidate_id ascending",
        },
        "candidate_id_semantics": (
            "sha256-derived opaque ID; role is never sent to scorer"
        ),
        "truth_always_included_when_eligible": True,
        "domain_config_sha256": domain_hash,
    }
    generation_hash = canonical_sha256(generation_config)

    eligibility_rows: list[dict[str, Any]] = []
    candidate_pools: list[dict[str, Any]] = []
    scoreable_manifest: list[dict[str, Any]] = []
    reason_counts: Counter[str] = Counter()
    candidate_generation_rejection_counts: Counter[str] = Counter()
    by_nucleus: Counter[str] = Counter()
    seen_ids: set[str] = set()
    preflight_manifest_rows: list[dict[str, Any]] = []
    global_preflight_runtime: dict[str, Any] | None = None

    connection = connect_readonly(index_path)
    try:
        schema_version(connection)
        ordered_records = sorted(
            source_records,
            key=lambda row: str(row.get("record_id") or ""),
        )
        for record_index, raw_record in enumerate(ordered_records, start=1):
            record = dict(raw_record)
            if record.get("schema_version") != DERIVED_SCHEMA_VERSION:
                raise CalibrationCaseError("source record schema changed")
            calibration = _mapping(record.get("calibration"), "calibration")
            if calibration.get("strict_independent_eligible") is not True:
                raise CalibrationCaseError(
                    "build_pre_score_artifacts accepts strict records only"
                )
            record_id = _text(record.get("record_id"), "record_id")
            if record_id in seen_ids:
                raise CalibrationCaseError(f"duplicate record_id: {record_id}")
            seen_ids.add(record_id)
            spectrum = _mapping(record.get("spectrum"), "spectrum")
            nucleus = _text(spectrum.get("nucleus"), "spectrum.nucleus")
            by_nucleus[nucleus] += 1
            bindings = _record_bindings(record)
            reasons: list[str] = []
            truth_candidate: dict[str, Any] | None = None
            decoys: list[dict[str, Any]] = []
            decoy_audit: dict[str, Any] = {
                "base_formula_rows": 0,
                "same_truth_connectivity_rows_excluded": 0,
                "invalid_base_rows_by_reason": {},
                "usable_different_connectivity_decoys": 0,
                "conformer_preflight_rejections_by_reason": {},
                "usable_different_connectivity_decoys_after_preflight": 0,
            }
            formula: str | None = None
            truth_inchi_key: str | None = None
            preflight_input: list[dict[str, str]] = []
            preflight_results: list[dict[str, Any]] = []
            passed_candidates: list[dict[str, Any]] = []
            passed_decoys: list[dict[str, Any]] = []
            if nucleus != "13C":
                reasons.append("nucleus_not_13c_dp5q_mean")
            else:
                (
                    truth_candidate,
                    truth_reasons,
                    formula,
                    truth_inchi_key,
                ) = _truth_candidate(record)
                reasons.extend(f"truth_{reason}" for reason in truth_reasons)
                if truth_candidate is not None:
                    decoys, decoy_audit = _find_decoys(
                        connection,
                        formula=formula,
                        truth_connectivity_key=str(truth_candidate["connectivity_key"]),
                    )
                    candidate_generation_rejection_counts.update(
                        decoy_audit["invalid_base_rows_by_reason"]
                    )
                    if not decoys:
                        reasons.append("no_same_formula_different_connectivity_decoy")
                    else:
                        preflight_candidates = sorted(
                            [truth_candidate, *decoys],
                            key=lambda candidate: str(candidate["candidate_id"]),
                        )
                        preflight_input = [
                            {
                                "candidate_id": str(candidate["candidate_id"]),
                                "smiles": str(candidate["smiles"]),
                            }
                            for candidate in preflight_candidates
                        ]
                        if progress is not None:
                            progress(
                                {
                                    "stage": "conformer_preflight",
                                    "pool_index": record_index,
                                    "pool_total": len(ordered_records),
                                    "record_id": record_id,
                                    "candidate_count": len(preflight_candidates),
                                    "model_outputs_created": False,
                                }
                            )
                        preflight_results, preflight_runtime = _run_conformer_preflight(
                            preflighter,
                            preflight_candidates,
                            formula=formula,
                        )
                        if global_preflight_runtime is None:
                            global_preflight_runtime = preflight_runtime
                        elif global_preflight_runtime != preflight_runtime:
                            raise CalibrationCaseError(
                                "conformer preflight runtime changed within one build"
                            )
                        results_by_id = {
                            str(result.get("candidate_id")): result
                            for result in preflight_results
                        }
                        if len(results_by_id) != len(preflight_candidates):
                            raise CalibrationCaseError(
                                f"{record_id}: conformer preflight duplicated a "
                                "candidate"
                            )
                        rejection_counts: Counter[str] = Counter()
                        for candidate in preflight_candidates:
                            candidate_id = str(candidate["candidate_id"])
                            result = _mapping(
                                results_by_id.get(candidate_id),
                                f"{record_id}.conformer_preflight[{candidate_id}]",
                            )
                            status = result.get("status")
                            if status == "passed":
                                passed_candidates.append(candidate)
                            elif status == "rejected":
                                reason_code = _text(
                                    result.get("reason_code"),
                                    "conformer preflight reason_code",
                                )
                                rejection_counts[reason_code] += 1
                            else:
                                raise CalibrationCaseError(
                                    f"{record_id}: conformer preflight status is "
                                    "invalid"
                                )
                        truth_result = results_by_id[
                            str(truth_candidate["candidate_id"])
                        ]
                        if truth_result.get("status") != "passed":
                            reasons.append(
                                "truth_conformer_preflight_"
                                + _text(
                                    truth_result.get("reason_code"),
                                    "truth conformer preflight reason",
                                )
                            )
                        passed_decoys = [
                            candidate
                            for candidate in passed_candidates
                            if candidate["candidate_id"]
                            != truth_candidate["candidate_id"]
                        ]
                        decoy_audit["conformer_preflight_rejections_by_reason"] = dict(
                            sorted(rejection_counts.items())
                        )
                        decoy_audit[
                            "usable_different_connectivity_decoys_after_preflight"
                        ] = len(passed_decoys)
                        if not passed_decoys:
                            reasons.append(
                                "no_conformer_preflight_eligible_same_formula_decoy"
                            )

            eligible = not reasons
            selected_decoys = passed_decoys[:selected_max_decoys]
            candidate_pool_size = (
                1 + len(selected_decoys)
                if eligible and truth_candidate is not None
                else 0
            )
            passed_candidate_set = [
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "smiles": str(candidate["smiles"]),
                }
                for candidate in passed_candidates
            ]
            preflight_hashes = {
                "conformer_preflight_input_candidate_set_sha256": canonical_sha256(
                    preflight_input
                ),
                "conformer_preflight_passed_candidate_set_sha256": canonical_sha256(
                    passed_candidate_set
                ),
                "conformer_preflight_decisions_sha256": canonical_sha256(
                    preflight_results
                ),
            }
            preflight_manifest_rows.append(
                {
                    "record_id": record_id,
                    **preflight_hashes,
                }
            )
            eligibility = {
                "schema_version": PRE_SCORE_SCHEMA_VERSION,
                "record_id": record_id,
                "eligible": eligible,
                "evaluated_before_scoring": True,
                "criterion_version": CRITERION_VERSION,
                "reason_codes": sorted(set(reasons)),
                "domain_config_sha256": domain_hash,
                "base_index_sha256": index_hash,
                "nucleus": nucleus,
                "candidate_pool_size_pre_score": candidate_pool_size,
                "candidate_count_preflight_input": len(preflight_input),
                "candidate_count_preflight_passed": len(passed_candidate_set),
                "usable_decoy_count_pre_cap": len(passed_decoys),
                **preflight_hashes,
                "source_document_sha256": bindings["source_document_sha256"],
                "molecule_sha256": bindings["molecule_sha256"],
                "scaffold_sha256": bindings["scaffold_sha256"],
                "source_content_sha256": bindings["source_content_sha256"],
                "source_release_id": source_release_binding["release_id"],
                "source_records_sha256": source_release_binding["records_sha256"],
                "source_current_sha256": source_release_binding[
                    "source_current_sha256"
                ],
            }
            if _FORBIDDEN_PRE_SCORE_FIELDS.intersection(eligibility):
                raise AssertionError("pre-score eligibility contains an outcome")
            eligibility_rows.append(eligibility)
            reason_counts.update(eligibility["reason_codes"])
            if not eligible:
                continue
            assert truth_candidate is not None
            assert truth_inchi_key is not None
            assert formula is not None
            selected_candidates = sorted(
                [truth_candidate, *selected_decoys],
                key=lambda candidate: str(candidate["candidate_id"]),
            )
            if len(selected_candidates) < 2:
                raise AssertionError("eligible candidate pool has no decoy")
            if len(selected_candidates) > MAX_FORWARD_CANDIDATES:
                raise AssertionError("candidate pool exceeds adapter limit")
            if (
                sum(
                    candidate["inchi_key"] == truth_inchi_key
                    for candidate in selected_candidates
                )
                != 1
            ):
                raise CalibrationCaseError(
                    f"{record_id}: candidate pool does not contain exactly one truth"
                )
            scorer_candidates = [
                {
                    "candidate_id": candidate["candidate_id"],
                    "smiles": candidate["smiles"],
                }
                for candidate in selected_candidates
            ]
            candidate_set_hash = canonical_sha256(scorer_candidates)
            candidate_pools.append(
                {
                    "schema_version": CANDIDATE_POOL_SCHEMA_VERSION,
                    "record_id": record_id,
                    "nucleus": "13C",
                    "formula": formula,
                    "observed_shifts_ppm": _observed_shifts(record),
                    "truth_candidate_id": truth_candidate["candidate_id"],
                    "target_inchi_key": truth_inchi_key,
                    "candidate_pool_size": len(selected_candidates),
                    "domain_config_sha256": domain_hash,
                    "candidate_generation_config_sha256": generation_hash,
                    "base_index_sha256": index_hash,
                    "candidate_set_sha256": candidate_set_hash,
                    "candidate_count_preflight_input": len(preflight_input),
                    "candidate_count_preflight_passed": len(passed_candidate_set),
                    **preflight_hashes,
                    "conformer_preflight_input": preflight_input,
                    "conformer_preflight_results": preflight_results,
                    "bindings": bindings,
                    "source_release_binding": dict(source_release_binding),
                    "decoy_audit": decoy_audit,
                    "candidates": [
                        {
                            **candidate,
                            "role": (
                                "truth"
                                if candidate["candidate_id"]
                                == truth_candidate["candidate_id"]
                                else "decoy"
                            ),
                        }
                        for candidate in selected_candidates
                    ],
                }
            )
            scoreable_manifest.append(
                {
                    "schema_version": SCOREABLE_MANIFEST_SCHEMA_VERSION,
                    "record_id": record_id,
                    "nucleus": "13C",
                    "target_inchi_key": truth_inchi_key,
                    **bindings,
                    "candidate_pool_size": len(selected_candidates),
                    "candidate_generation_config_sha256": generation_hash,
                    "base_index_sha256": index_hash,
                    "candidate_set_sha256": candidate_set_hash,
                    "candidate_count_preflight_input": len(preflight_input),
                    "candidate_count_preflight_passed": len(passed_candidate_set),
                    **preflight_hashes,
                    "source_release_id": source_release_binding["release_id"],
                    "source_records_sha256": source_release_binding["records_sha256"],
                    "source_current_sha256": source_release_binding[
                        "source_current_sha256"
                    ],
                }
            )
    finally:
        connection.close()
    if file_sha256(index_path) != index_hash:
        raise CalibrationCaseError(
            "base index changed while candidate pools were being generated"
        )
    if global_preflight_runtime is None:
        raise CalibrationCaseError(
            "strict source pool produced no conformer-preflight runtime binding"
        )
    preflight_manifest_hash = canonical_sha256(preflight_manifest_rows)
    eligibility_runtime_binding = {
        "conformer_preflight_sidecar_sha256": global_preflight_runtime[
            "sidecar_code_sha256"
        ],
        "conformer_preflight_rdkit_version": global_preflight_runtime["rdkit_version"],
        "conformer_protocol_version": global_preflight_runtime[
            "conformer_protocol_version"
        ],
        "conformer_generation_config_sha256": global_preflight_runtime[
            "conformer_generation_config_sha256"
        ],
        "conformer_preflight_protocol_version": global_preflight_runtime[
            "conformer_preflight_protocol_version"
        ],
        "conformer_preflight_policy_sha256": global_preflight_runtime[
            "conformer_preflight_policy_sha256"
        ],
        "conformer_preflight_runtime_sha256": global_preflight_runtime[
            "runtime_sha256"
        ],
        "conformer_preflight_manifest_sha256": preflight_manifest_hash,
    }
    for row in eligibility_rows:
        row.update(eligibility_runtime_binding)
    for row in candidate_pools:
        row.update(eligibility_runtime_binding)
        row["conformer_preflight"] = dict(global_preflight_runtime)
    for row in scoreable_manifest:
        row.update(eligibility_runtime_binding)

    strict_count = len(eligibility_rows)
    strict_13c_count = int(by_nucleus["13C"])
    scoreable_count = len(candidate_pools)
    scoreable_ids = [str(row["record_id"]) for row in scoreable_manifest]
    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "mode": "pre_score_audit",
        "criterion_version": CRITERION_VERSION,
        "change_notice": (
            "v3 additionally binds a model-output-free conformer preflight of "
            "truth and every same-formula decoy in statically viable pools "
            "before candidate capping. Candidate-local decoy failures are "
            "excluded transparently; truth or operational failures remain "
            "fail-closed. Statically ineligible rows never call the sidecar."
        ),
        "outcomes_created": False,
        "extraction_review_labels_used_as_outcomes": False,
        "counts": {
            "strict_pool_records": strict_count,
            "strict_by_nucleus": dict(sorted(by_nucleus.items())),
            "strict_13c_records": strict_13c_count,
            "pre_score_eligible_records": scoreable_count,
            "pre_score_ineligible_records": strict_count - scoreable_count,
            "downstream_score_coverage_denominator": scoreable_count,
        },
        "coverage": {
            "eligible_over_all_strict": (
                scoreable_count / strict_count if strict_count else 0.0
            ),
            "eligible_over_strict_13c": (
                scoreable_count / strict_13c_count if strict_13c_count else 0.0
            ),
            "interpretation": (
                "The immutable v2 strict pool is the eligibility-audit "
                "denominator; "
                "only eligible rows belong in the downstream score-case "
                "coverage denominator."
            ),
        },
        "ineligibility_reason_counts": dict(sorted(reason_counts.items())),
        "candidate_generation_rejection_counts": dict(
            sorted(candidate_generation_rejection_counts.items())
        ),
        "domain_config": domain_config,
        "domain_config_sha256": domain_hash,
        "candidate_generation_config": generation_config,
        "candidate_generation_config_sha256": generation_hash,
        "conformer_preflight": {
            **global_preflight_runtime,
            "manifest_sha256": preflight_manifest_hash,
            "manifest_rows": len(preflight_manifest_rows),
        },
        "base_index": {
            "path": str(index_path),
            "schema_version": "2",
            "sha256": index_hash,
        },
        "source_release_binding": dict(source_release_binding),
        "scoreable_record_ids": scoreable_ids,
        "scoreable_record_ids_sha256": canonical_sha256(scoreable_ids),
        "artifacts": {
            "pre_score_eligibility_jsonl_sha256": jsonl_sha256(eligibility_rows),
            "candidate_pools_jsonl_sha256": jsonl_sha256(candidate_pools),
            "scoreable_source_manifest_jsonl_sha256": jsonl_sha256(scoreable_manifest),
        },
        "runtime": {"rdkit_version": rdBase.rdkitVersion},
    }
    return {
        "eligibility": eligibility_rows,
        "candidate_pools": candidate_pools,
        "scoreable_manifest": scoreable_manifest,
        "summary": summary,
    }


def _finite_number(value: Any, field: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise CalibrationCaseError(f"{field} must be a finite number")
    return float(value)


def _score_artifact(
    model: Mapping[str, Any],
    runtime: Mapping[str, Any],
    rank_basis: str,
    *,
    base_index_sha256: str,
    domain_config_sha256: str,
    candidate_generation_config_sha256: str,
    conformer_preflight: Mapping[str, Any],
    conformer_preflight_manifest_sha256: str,
    source_release_binding: Mapping[str, Any],
) -> dict[str, Any]:
    adapter_path = (
        Path(__file__).resolve().parents[1] / "app" / "ml" / "nmr_forward.py"
    )
    sidecar_path = Path(__file__).resolve().parents[1] / "scripts" / "dp5q_sidecar.py"
    sidecar_code_sha256 = _text(
        runtime.get("sidecar_code_sha256"),
        "forward runtime.sidecar_code_sha256",
    )
    expected_sidecar_sha256 = file_sha256(sidecar_path)
    if sidecar_code_sha256 != expected_sidecar_sha256:
        raise CalibrationCaseError(
            "forward runtime did not execute the shipped DP5q sidecar"
        )
    conformer_generation = _mapping(
        runtime.get("conformer_generation"),
        "forward runtime.conformer_generation",
    )
    expected_conformer_generation = dp5q_conformer_policy()
    if conformer_generation != expected_conformer_generation:
        raise CalibrationCaseError(
            "forward runtime conformer-generation policy is not frozen"
        )
    scoring_preflight_policy = _mapping(
        runtime.get("conformer_preflight"),
        "forward runtime.conformer_preflight",
    )
    scoring_binding = {
        "sidecar_protocol_version": runtime.get("protocol_version"),
        "sidecar_code_sha256": sidecar_code_sha256,
        "rdkit_version": runtime.get("rdkit_version"),
        "conformer_protocol_version": DP5Q_CONFORMER_PROTOCOL_VERSION,
        "conformer_generation_config_sha256": canonical_sha256(conformer_generation),
        "conformer_preflight_protocol_version": scoring_preflight_policy.get(
            "protocol_version"
        ),
        "conformer_preflight_policy_sha256": canonical_sha256(scoring_preflight_policy),
        "model_outputs_created": False,
    }
    scoring_binding["runtime_sha256"] = canonical_sha256(scoring_binding)
    if scoring_binding != dict(conformer_preflight):
        raise CalibrationCaseError(
            "scoring runtime does not equal the frozen conformer-preflight runtime"
        )
    if (
        not isinstance(conformer_preflight_manifest_sha256, str)
        or len(conformer_preflight_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in conformer_preflight_manifest_sha256
        )
    ):
        raise CalibrationCaseError("conformer preflight manifest hash is invalid")
    artifact = {
        "schema_version": "chemapp.nmr.frozen-ranker-artifact.v3",
        "adapter": "NMRForwardAdapter.score_candidates",
        "adapter_code_sha256": file_sha256(adapter_path),
        "case_scoring_code_sha256": file_sha256(Path(__file__)),
        "sidecar_code_sha256": sidecar_code_sha256,
        "conformer_generation": dict(conformer_generation),
        "conformer_protocol_version": DP5Q_CONFORMER_PROTOCOL_VERSION,
        "conformer_generation_config_sha256": canonical_sha256(conformer_generation),
        "conformer_preflight": {
            **dict(conformer_preflight),
            "manifest_sha256": conformer_preflight_manifest_sha256,
        },
        "model": dict(model),
        "expected_pins": {
            "repository_commit": DP5Q_REPOSITORY_COMMIT,
            "mean_model_sha256": DP5Q_MEAN_MODEL_SHA256,
            "preprocessor_sha256": DP5Q_PREPROCESSOR_SHA256,
        },
        "rank_basis": rank_basis,
        "score_semantics": SCORE_SEMANTICS,
        "base_index_sha256": base_index_sha256,
        "gold_reference": {
            "scope": "chemapp_frozen_nmrshiftdb2_base_index",
            "sha256": base_index_sha256,
            "interpretation": (
                "Gold structures were checked against this ChemApp base index "
                "only; this is not a DP5q training-corpus audit."
            ),
        },
        "upstream_training_overlap": {
            "model": "DP5q-CASCADE-mean",
            "status": "unknown",
            "reason": (
                "DP5q upstream training structures are not exposed by the "
                "frozen runtime in an auditable structure-level manifest."
            ),
        },
        "domain_config_sha256": domain_config_sha256,
        "candidate_generation_config_sha256": (candidate_generation_config_sha256),
        "source_release_binding": dict(source_release_binding),
        "candidate_scalar_score": "negative_mae_ppm",
        "margin": "top1_score_minus_runnerup_score",
        "same_formula_invariant": (
            "coverage and unmatched counts must be identical within each pool"
        ),
        "calibrated_probability": False,
    }
    if artifact["model"] != {
        "name": "DP5q-CASCADE-mean",
        "repository_commit": DP5Q_REPOSITORY_COMMIT,
        "mean_model_sha256": DP5Q_MEAN_MODEL_SHA256,
        "preprocessor_sha256": DP5Q_PREPROCESSOR_SHA256,
    }:
        raise CalibrationCaseError("forward model artifact does not match DP5q pins")
    return artifact


def _normalise_evidence(
    raw: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    required_integer_fields = (
        "relative_rank",
        "matched_count",
        "observed_count",
        "predicted_atom_count",
        "unmatched_observed_count",
        "unmatched_predicted_atom_count",
    )
    integers: dict[str, int] = {}
    for field in required_integer_fields:
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise CalibrationCaseError(f"forward evidence {field} is invalid")
        integers[field] = value
    numeric_fields = (
        "observed_coverage",
        "predicted_atom_coverage",
        "bidirectional_coverage",
        "mae_ppm",
        "rmse_ppm",
        "max_abs_error_ppm",
    )
    numeric = {
        field: _finite_number(raw.get(field), f"forward evidence {field}")
        for field in numeric_fields
    }
    prediction = _mapping(raw.get("prediction"), "forward prediction")
    atom_predictions = prediction.get("atom_predictions")
    if not isinstance(atom_predictions, list):
        raise CalibrationCaseError("forward atom_predictions must be a list")
    predicted_shifts: list[float] = []
    for atom in atom_predictions:
        if not isinstance(atom, Mapping):
            raise CalibrationCaseError("forward atom prediction is invalid")
        predicted_shifts.append(
            _finite_number(atom.get("shift_ppm"), "predicted shift")
        )
    prediction_audit = {
        "canonical_smiles": _text(
            prediction.get("canonical_smiles"),
            "prediction.canonical_smiles",
        ),
        "conformer_count": prediction.get("conformer_count"),
        "warnings": prediction.get("warnings"),
        "predicted_shifts_ppm": predicted_shifts,
    }
    return {
        "candidate_id": candidate["candidate_id"],
        "inchi_key": candidate["inchi_key"],
        "connectivity_key": candidate["connectivity_key"],
        **integers,
        **numeric,
        "assignment_complete": raw.get("assignment_complete") is True,
        "prediction": prediction_audit,
        "prediction_sha256": canonical_sha256(prediction_audit),
    }


def score_candidate_pools(
    candidate_pools: Sequence[Mapping[str, Any]],
    *,
    scorer: ForwardScorer,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Score every eligible pool and create exact Top-1 outcomes afterward."""

    cases: list[dict[str, Any]] = []
    ranker_artifact: dict[str, Any] | None = None
    ranker_hash: str | None = None
    for pool_index, pool in enumerate(candidate_pools, start=1):
        record_id = _text(pool.get("record_id"), "pool.record_id")
        if pool.get("schema_version") != CANDIDATE_POOL_SCHEMA_VERSION:
            raise CalibrationCaseError(f"{record_id}: candidate-pool schema changed")
        candidates = pool.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise CalibrationCaseError(f"{record_id}: candidate pool is invalid")
        scorer_candidates = [
            {
                "candidate_id": _text(
                    candidate.get("candidate_id"),
                    "candidate.candidate_id",
                ),
                "smiles": _text(candidate.get("smiles"), "candidate.smiles"),
            }
            for candidate in candidates
            if isinstance(candidate, Mapping)
        ]
        if len(scorer_candidates) != len(candidates):
            raise CalibrationCaseError(f"{record_id}: candidate is not an object")
        if any(
            set(candidate) != {"candidate_id", "smiles"}
            for candidate in scorer_candidates
        ):
            raise AssertionError("truth role leaked into scorer input")
        if canonical_sha256(scorer_candidates) != pool.get("candidate_set_sha256"):
            raise CalibrationCaseError(f"{record_id}: candidate set hash changed")
        if progress is not None:
            progress(
                {
                    "stage": "forward_scoring",
                    "pool_index": pool_index,
                    "pool_total": len(candidate_pools),
                    "record_id": record_id,
                    "candidate_count": len(scorer_candidates),
                }
            )
        preflight_input = pool.get("conformer_preflight_input")
        preflight_results = pool.get("conformer_preflight_results")
        if (
            not isinstance(preflight_input, list)
            or not isinstance(preflight_results, list)
            or len(preflight_input) != len(preflight_results)
            or canonical_sha256(preflight_input)
            != pool.get("conformer_preflight_input_candidate_set_sha256")
            or canonical_sha256(preflight_results)
            != pool.get("conformer_preflight_decisions_sha256")
        ):
            raise CalibrationCaseError(
                f"{record_id}: conformer-preflight input/decision binding changed"
            )
        result_by_id = {
            str(result.get("candidate_id")): result
            for result in preflight_results
            if isinstance(result, Mapping)
        }
        if len(result_by_id) != len(preflight_results):
            raise CalibrationCaseError(
                f"{record_id}: conformer-preflight decisions are not unique"
            )
        passed_preflight_candidates = [
            dict(candidate)
            for candidate in preflight_input
            if isinstance(candidate, Mapping)
            and isinstance(
                result_by_id.get(str(candidate.get("candidate_id"))), Mapping
            )
            and result_by_id[str(candidate.get("candidate_id"))].get("status")
            == "passed"
        ]
        if (
            len(passed_preflight_candidates)
            != pool.get("candidate_count_preflight_passed")
            or canonical_sha256(passed_preflight_candidates)
            != pool.get("conformer_preflight_passed_candidate_set_sha256")
            or not {
                candidate["candidate_id"] for candidate in scorer_candidates
            }.issubset(
                {
                    str(candidate.get("candidate_id"))
                    for candidate in passed_preflight_candidates
                }
            )
        ):
            raise CalibrationCaseError(
                f"{record_id}: scored candidates are not bound preflight passes"
            )
        result = scorer.score_candidates(
            list(pool["observed_shifts_ppm"]),
            scorer_candidates,
            formula=str(pool["formula"]),
        )
        if (
            not isinstance(result, Mapping)
            or result.get("status") != "ok"
            or result.get("nucleus") != "13C"
            or result.get("evidence_kind") != "relative_13c_forward_evidence"
            or result.get("calibrated_probability") is not False
        ):
            raise CalibrationCaseError(
                f"{record_id}: forward scorer returned unsupported semantics"
            )
        raw_evidence = result.get("candidates")
        if not isinstance(raw_evidence, list) or len(raw_evidence) != len(candidates):
            raise CalibrationCaseError(
                f"{record_id}: forward scorer changed candidate coverage"
            )
        candidate_by_id = {
            str(candidate["candidate_id"]): candidate for candidate in candidates
        }
        if set(candidate_by_id) != {
            str(item.get("candidate_id"))
            for item in raw_evidence
            if isinstance(item, Mapping)
        }:
            raise CalibrationCaseError(
                f"{record_id}: forward scorer changed candidate identities"
            )
        evidence = [
            _normalise_evidence(
                _mapping(item, "forward candidate evidence"),
                candidate_by_id[str(item["candidate_id"])],
            )
            for item in raw_evidence
        ]
        evidence.sort(key=lambda item: int(item["relative_rank"]))
        if [item["relative_rank"] for item in evidence] != list(
            range(1, len(evidence) + 1)
        ):
            raise CalibrationCaseError(
                f"{record_id}: forward relative ranks are invalid"
            )

        # Exact-formula pools have the same carbon count.  Therefore coverage
        # and unmatched terms in the adapter's lexicographic rank are invariant,
        # and ranking reduces to MAE then RMSE.  Enforcing this invariant makes
        # the scalar margin transparent rather than an arbitrary weighted sum.
        invariant_fields = (
            "matched_count",
            "observed_count",
            "predicted_atom_count",
            "unmatched_observed_count",
            "unmatched_predicted_atom_count",
            "observed_coverage",
            "predicted_atom_coverage",
            "bidirectional_coverage",
        )
        for field in invariant_fields:
            if len({item[field] for item in evidence}) != 1:
                raise CalibrationCaseError(
                    f"{record_id}: same-formula {field} invariant failed"
                )
        top, runner_up = evidence[0], evidence[1]
        raw_score = float(runner_up["mae_ppm"]) - float(top["mae_ppm"])
        if raw_score < -1e-10:
            raise CalibrationCaseError(
                f"{record_id}: forward Top-1 has worse MAE than runner-up"
            )
        raw_score = max(raw_score, 0.0)

        current_artifact = _score_artifact(
            _mapping(result.get("model"), "forward model"),
            _mapping(result.get("runtime"), "forward runtime"),
            _text(result.get("rank_basis"), "rank_basis"),
            base_index_sha256=_text(
                pool.get("base_index_sha256"),
                "pool.base_index_sha256",
            ),
            domain_config_sha256=_text(
                pool.get("domain_config_sha256"),
                "pool.domain_config_sha256",
            ),
            candidate_generation_config_sha256=_text(
                pool.get("candidate_generation_config_sha256"),
                "pool.candidate_generation_config_sha256",
            ),
            conformer_preflight=_mapping(
                pool.get("conformer_preflight"),
                "pool.conformer_preflight",
            ),
            conformer_preflight_manifest_sha256=_text(
                pool.get("conformer_preflight_manifest_sha256"),
                "pool.conformer_preflight_manifest_sha256",
            ),
            source_release_binding=_mapping(
                pool.get("source_release_binding"),
                "pool.source_release_binding",
            ),
        )
        current_hash = artifact_json_sha256(current_artifact)
        if ranker_hash is None:
            ranker_hash = current_hash
            ranker_artifact = current_artifact
        elif ranker_hash != current_hash:
            raise CalibrationCaseError(
                "forward ranker artifact changed within one frozen run"
            )

        predicted_top1_inchi_key = str(top["inchi_key"])
        target_inchi_key = _text(
            pool.get("target_inchi_key"),
            "pool.target_inchi_key",
        )
        exact_correct = predicted_top1_inchi_key == target_inchi_key
        bindings = _mapping(pool.get("bindings"), "pool.bindings")
        cases.append(
            {
                "schema_version": FROZEN_SCORE_CASE_SCHEMA_VERSION,
                "case_id": f"nmrexp-13c-{record_id.split(':', 1)[-1]}",
                "record_ids": [record_id],
                "nucleus": "13C",
                "target_inchi_key": target_inchi_key,
                "predicted_top1_inchi_key": predicted_top1_inchi_key,
                "top1_exact_correct": exact_correct,
                "top_candidate_is_truth": exact_correct,
                "candidate_pool_size": len(candidates),
                "raw_score": raw_score,
                "score_semantics": SCORE_SEMANTICS,
                "ranker_artifact_sha256": current_hash,
                "candidate_generation_config_sha256": pool[
                    "candidate_generation_config_sha256"
                ],
                "domain_config_sha256": pool["domain_config_sha256"],
                "candidate_set_sha256": pool["candidate_set_sha256"],
                "candidate_count_preflight_input": pool[
                    "candidate_count_preflight_input"
                ],
                "candidate_count_preflight_passed": pool[
                    "candidate_count_preflight_passed"
                ],
                "conformer_preflight_sidecar_sha256": pool[
                    "conformer_preflight_sidecar_sha256"
                ],
                "conformer_preflight_rdkit_version": pool[
                    "conformer_preflight_rdkit_version"
                ],
                "conformer_protocol_version": pool["conformer_protocol_version"],
                "conformer_generation_config_sha256": pool[
                    "conformer_generation_config_sha256"
                ],
                "conformer_preflight_protocol_version": pool[
                    "conformer_preflight_protocol_version"
                ],
                "conformer_preflight_policy_sha256": pool[
                    "conformer_preflight_policy_sha256"
                ],
                "conformer_preflight_runtime_sha256": pool[
                    "conformer_preflight_runtime_sha256"
                ],
                "conformer_preflight_manifest_sha256": pool[
                    "conformer_preflight_manifest_sha256"
                ],
                "conformer_preflight_input_candidate_set_sha256": pool[
                    "conformer_preflight_input_candidate_set_sha256"
                ],
                "conformer_preflight_passed_candidate_set_sha256": pool[
                    "conformer_preflight_passed_candidate_set_sha256"
                ],
                "conformer_preflight_decisions_sha256": pool[
                    "conformer_preflight_decisions_sha256"
                ],
                "candidate_set": scorer_candidates,
                "candidate_identity_sha256": canonical_sha256(
                    [
                        {
                            "candidate_id": item["candidate_id"],
                            "inchi_key": item["inchi_key"],
                            "connectivity_key": item["connectivity_key"],
                        }
                        for item in evidence
                    ]
                ),
                "candidate_evidence_sha256": canonical_sha256(evidence),
                "base_index_sha256": pool["base_index_sha256"],
                "source_release_id": pool["source_release_binding"]["release_id"],
                "source_records_sha256": pool["source_release_binding"][
                    "records_sha256"
                ],
                "source_current_sha256": pool["source_release_binding"][
                    "source_current_sha256"
                ],
                "source_document_key": bindings["source_document_key"],
                "molecule_key": bindings["molecule_key"],
                "scaffold_key": bindings["scaffold_key"],
                "source_document_sha256": bindings["source_document_sha256"],
                "molecule_sha256": bindings["molecule_sha256"],
                "scaffold_sha256": bindings["scaffold_sha256"],
                "source_content_sha256": bindings["source_content_sha256"],
                "outcome_semantics": (
                    "DP5q Top-1 exact InChIKey equals smiles_actual gold exact InChIKey"
                ),
                "extraction_review_labels_used_as_outcome": False,
                "raw_features": {
                    "top1_mae_ppm": top["mae_ppm"],
                    "top1_rmse_ppm": top["rmse_ppm"],
                    "top1_bidirectional_coverage": top["bidirectional_coverage"],
                    "top1_observed_coverage": top["observed_coverage"],
                    "top1_predicted_atom_coverage": top["predicted_atom_coverage"],
                    "top1_unmatched_observed_count": top["unmatched_observed_count"],
                    "top1_unmatched_predicted_atom_count": top[
                        "unmatched_predicted_atom_count"
                    ],
                    "runner_up_mae_ppm": runner_up["mae_ppm"],
                    "runner_up_rmse_ppm": runner_up["rmse_ppm"],
                    "top_margin_mae_ppm": raw_score,
                    "pool_size": len(candidates),
                },
                "candidate_evidence": evidence,
            }
        )
    if candidate_pools and (ranker_hash is None or ranker_artifact is None):
        raise AssertionError("scored pools lack a ranker artifact")
    cases.sort(key=lambda row: str(row["case_id"]))
    return {
        "cases": cases,
        "ranker_artifact": ranker_artifact,
        "ranker_artifact_sha256": ranker_hash,
    }


def build_scored_manifest(
    scoreable_manifest: Sequence[Mapping[str, Any]],
    cases: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Bind only successfully scored cases to their source records."""

    manifest_by_record = {
        str(row["record_id"]): dict(row) for row in scoreable_manifest
    }
    output: list[dict[str, Any]] = []
    for case in cases:
        record_ids = case.get("record_ids")
        if not isinstance(record_ids, list) or len(record_ids) != 1:
            raise CalibrationCaseError("each frozen case must bind one record")
        record_id = str(record_ids[0])
        source = manifest_by_record.get(record_id)
        if source is None:
            raise CalibrationCaseError(
                f"scored case is absent from scoreable manifest: {record_id}"
            )
        output.append(
            {
                **source,
                "case_id": case["case_id"],
                "ranker_artifact_sha256": case["ranker_artifact_sha256"],
            }
        )
    output.sort(key=lambda row: str(row["record_id"]))
    return output


def _render_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json_dumps(dict(row)) + "\n" for row in rows)


def _render_json(value: Mapping[str, Any]) -> str:
    return (
        json.dumps(
            dict(value),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _write_artifact_set(
    root: Path,
    rendered: Mapping[str, str],
    *,
    overwrite: bool,
) -> dict[str, Path]:
    root.mkdir(parents=True, exist_ok=True)
    targets = {name: root / name for name in rendered}
    # Preflight every target so a refusal cannot publish a half-updated set.
    for name, target in targets.items():
        if not target.exists():
            continue
        try:
            existing = target.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise CalibrationCaseError(
                f"cannot inspect existing artifact {target}: {exc}"
            ) from exc
        if existing != rendered[name] and not overwrite:
            raise CalibrationCaseError(
                f"existing artifact differs: {target}; use --overwrite only "
                "after review"
            )
    temporary_paths: list[Path] = []
    try:
        for name, target in targets.items():
            if target.exists() and target.read_text(encoding="utf-8") == rendered[name]:
                continue
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                prefix=f".{name}.",
                suffix=".part",
                dir=root,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                temporary_paths.append(temporary)
                handle.write(rendered[name])
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, target)
            temporary_paths.remove(temporary)
    finally:
        for temporary in temporary_paths:
            temporary.unlink(missing_ok=True)
    return targets


def write_pre_score_artifacts(
    destination: str | Path,
    artifacts: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write audit-only artifacts under a mode-specific directory."""

    root = Path(destination).resolve() / "audit-only"
    return _write_artifact_set(
        root,
        {
            "pre-score-eligibility.jsonl": _render_jsonl(artifacts["eligibility"]),
            "candidate-pools.jsonl": _render_jsonl(artifacts["candidate_pools"]),
            "scoreable-source-manifest.jsonl": _render_jsonl(
                artifacts["scoreable_manifest"]
            ),
            "summary.json": _render_json(artifacts["summary"]),
        },
        overwrite=overwrite,
    )


def write_scored_artifacts(
    destination: str | Path,
    pre_score: Mapping[str, Any],
    scored: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write a complete scored set only after every eligible pool succeeded."""

    cases = list(scored["cases"])
    scored_manifest = build_scored_manifest(
        pre_score["scoreable_manifest"],
        cases,
    )
    if len(cases) != len(pre_score["candidate_pools"]):
        raise CalibrationCaseError(
            "partial scoring cannot be published as a frozen case set"
        )
    summary = json.loads(canonical_json_dumps(pre_score["summary"]))
    summary["mode"] = "scored"
    summary["outcomes_created"] = True
    summary["scoring"] = {
        "attempted_records": len(pre_score["candidate_pools"]),
        "scored_records": len(cases),
        "failed_records": 0,
        "score_case_coverage_denominator": len(pre_score["scoreable_manifest"]),
        "ranker_artifact": scored["ranker_artifact"],
        "ranker_artifact_sha256": scored["ranker_artifact_sha256"],
        "score_semantics": SCORE_SEMANTICS,
        "calibrator_fitted": False,
        "cases_jsonl_sha256": jsonl_sha256(cases),
        "scored_source_manifest_jsonl_sha256": jsonl_sha256(scored_manifest),
    }
    ranker_artifact = _mapping(
        scored.get("ranker_artifact"),
        "ranker_artifact",
    )
    if artifact_json_sha256(ranker_artifact) != scored.get("ranker_artifact_sha256"):
        raise CalibrationCaseError("ranker artifact byte hash changed")
    root = Path(destination).resolve() / "scored"
    return _write_artifact_set(
        root,
        {
            "pre-score-eligibility.jsonl": _render_jsonl(pre_score["eligibility"]),
            "candidate-pools.jsonl": _render_jsonl(pre_score["candidate_pools"]),
            "scoreable-source-manifest.jsonl": _render_jsonl(
                pre_score["scoreable_manifest"]
            ),
            "cases.jsonl": _render_jsonl(cases),
            "scored-source-manifest.jsonl": _render_jsonl(scored_manifest),
            "ranker-artifact.json": artifact_json_bytes(ranker_artifact).decode(
                "utf-8"
            ),
            "summary.json": _render_json(summary),
        },
        overwrite=overwrite,
    )


__all__ = [
    "CANDIDATE_GENERATION_VERSION",
    "CRITERION_VERSION",
    "CalibrationCaseError",
    "DEFAULT_MAX_DECOYS",
    "FROZEN_SCORE_CASE_SCHEMA_VERSION",
    "PRE_SCORE_SCHEMA_VERSION",
    "SCORE_SEMANTICS",
    "build_pre_score_artifacts",
    "build_scored_manifest",
    "load_strict_source_records",
    "score_candidate_pools",
    "write_pre_score_artifacts",
    "write_scored_artifacts",
]
