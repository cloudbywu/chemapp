"""Leak-resistant evaluation primitives for NMR structure elucidation.

The benchmark is intentionally strict.  A result case is only valid when it
is bound to a pre-registered run and to records in the frozen test split.
Closed-library and open-world tasks are never combined.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from app.ml.nmr_data_v2 import connect_readonly, schema_version

try:
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    from rdkit.Chem.MolStandardize import rdMolStandardize
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:  # pragma: no cover - validated by the caller
    Chem = None
    rdMolDescriptors = None
    rdMolStandardize = None
    MurckoScaffold = None


BENCHMARK_PROTOCOL_VERSION = "nmr-elucidation-benchmark-v2"
REGISTRATION_SCHEMA_VERSION = "nmr-benchmark-registration-v1"
BenchmarkTask = Literal[
    "closed_library",
    "open_exact_removed",
    "open_scaffold_removed",
]
MeasurementPolicy = Literal["measured_only", "include_inferred_measured"]
ReviewPolicy = Literal["reviewed_only", "exclude_rejected", "all"]

SPLIT_NAMES = ("train", "validation", "calibration", "test")
DEFAULT_SPLIT_RATIOS = {
    "train": 0.70,
    "validation": 0.10,
    "calibration": 0.10,
    "test": 0.10,
}
ALLOWED_MODALITIES = {"1h", "13c", "1h+13c"}
ALLOWED_NUCLEI = {"1H", "13C"}
MEASUREMENT_POLICIES = {"measured_only", "include_inferred_measured"}
REVIEW_POLICIES = {"reviewed_only", "exclude_rejected", "all"}

_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s\"<>]+", re.IGNORECASE)
_PMID_RE = re.compile(r"\bPMID\s*:?\s*(\d+)\b", re.IGNORECASE)


class StructureIdentityMismatch(ValueError):
    """Raised when source identifiers conflict with the authoritative MolBlock."""


def canonical_json_dumps(value: Any) -> str:
    """Return the canonical JSON representation used by benchmark hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash JSONL content independently from incidental whitespace."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_dumps(dict(row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def benchmark_registration_id(registration: Mapping[str, Any]) -> str:
    """Compute the immutable identifier of a registration payload."""

    payload = {
        key: value for key, value in registration.items() if key != "registration_id"
    }
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()


def _require_hash(value: Any, label: str) -> str:
    result = str(value or "").lower()
    if not _HASH_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _require_bool(mapping: Mapping[str, Any], key: str) -> bool:
    if key not in mapping or type(mapping[key]) is not bool:
        raise ValueError(f"{key} must be a JSON boolean")
    return mapping[key]


def _require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _connectivity_key(inchi_key: str) -> str:
    return inchi_key.split("-", 1)[0]


def _mol_inchi_key(mol: Any) -> str:
    key = str(Chem.MolToInchiKey(mol) or "").strip().upper()
    if not key:
        raise StructureIdentityMismatch("RDKit could not compute an InChIKey")
    return key


def _representation_connectivity(
    value: str | None,
    *,
    representation: str,
) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if representation == "smiles":
        mol = Chem.MolFromSmiles(text)
    else:
        mol = Chem.MolFromInchi(text)
    if mol is None:
        raise StructureIdentityMismatch(
            f"source {representation} is present but cannot be parsed"
        )
    return _connectivity_key(_mol_inchi_key(mol))


def _structure_identifiers(
    *,
    molblock: str,
    inchi: str | None,
    inchi_key: str | None,
    smiles: str | None,
    source_formula: str | None = None,
) -> dict[str, str] | None:
    """Derive identity from MolBlock and reject conflicting source metadata."""

    if (
        Chem is None
        or rdMolDescriptors is None
        or rdMolStandardize is None
        or MurckoScaffold is None
    ):
        raise RuntimeError("RDKit is required to build NMR benchmark manifests")

    mol = Chem.MolFromMolBlock(
        molblock,
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    computed_inchi_key = _mol_inchi_key(mol)
    computed_connectivity = _connectivity_key(computed_inchi_key)
    computed_formula = rdMolDescriptors.CalcMolFormula(mol)

    declared_key = str(inchi_key or "").strip().upper()
    if declared_key and declared_key != computed_inchi_key:
        raise StructureIdentityMismatch(
            "source InChIKey does not match the MolBlock-derived InChIKey"
        )
    for representation, value in (("inchi", inchi), ("smiles", smiles)):
        source_connectivity = _representation_connectivity(
            value,
            representation=representation,
        )
        if source_connectivity and source_connectivity != computed_connectivity:
            raise StructureIdentityMismatch(
                f"source {representation} connectivity does not match the MolBlock"
            )
    declared_formula = re.sub(r"\s+", "", str(source_formula or ""))
    if declared_formula and declared_formula != computed_formula:
        raise StructureIdentityMismatch(
            "source formula does not match the MolBlock-derived formula"
        )

    parent = rdMolStandardize.FragmentParent(rdMolStandardize.Cleanup(mol))
    parent = Chem.RemoveHs(parent)
    parent_inchi_key = _mol_inchi_key(parent)
    molecule_key = _connectivity_key(parent_inchi_key)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=parent,
        includeChirality=False,
    )
    generic_scaffold = ""
    if scaffold:
        scaffold_mol = Chem.MolFromSmiles(scaffold)
        if scaffold_mol is not None:
            generic = MurckoScaffold.MakeScaffoldGeneric(scaffold_mol)
            generic_scaffold = Chem.MolToSmiles(
                generic,
                canonical=True,
                isomericSmiles=False,
            )

    return {
        "canonical_smiles": Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        ),
        "parent_canonical_smiles": Chem.MolToSmiles(
            parent,
            canonical=True,
            isomericSmiles=True,
        ),
        "inchi_key": computed_inchi_key,
        "stereo_identity_key": computed_inchi_key,
        "molecule_key": molecule_key,
        "scaffold_key": scaffold or "acyclic",
        "generic_scaffold_key": generic_scaffold or "acyclic",
        "formula": computed_formula,
    }


def _normalize_source_document_key(literature: str, snapshot_sha: str) -> str:
    text = unicodedata.normalize("NFKC", literature or "").strip()
    doi_match = _DOI_RE.search(text)
    if doi_match:
        doi = doi_match.group(0).rstrip(".,;:)]}").casefold()
        return f"doi:{doi}"
    pmid_match = _PMID_RE.search(text)
    if pmid_match:
        return f"pmid:{pmid_match.group(1)}"
    if text:
        normalized = re.sub(r"\s+", " ", text.casefold())
        normalized = re.sub(r"\s*([,;:.])\s*", r"\1", normalized)
        digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        return f"literature-sha256:{digest}"
    return f"snapshot:{snapshot_sha}:unknown-document"


def _measurement_is_eligible(kind: str, policy: MeasurementPolicy) -> bool:
    if kind == "measured":
        return True
    return policy == "include_inferred_measured" and kind == "inferred_measured"


def _review_is_eligible(status: str, policy: ReviewPolicy) -> bool:
    if policy == "reviewed_only":
        return status == "reviewed"
    if policy == "exclude_rejected":
        return status != "rejected"
    return True


def load_v2_manifest_records(
    index_path: str,
    *,
    measurement_policy: MeasurementPolicy = "measured_only",
    review_policy: ReviewPolicy = "exclude_rejected",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read auditable, policy-filtered split records from a v2 spectrum index."""

    if (
        Chem is None
        or rdMolDescriptors is None
        or rdMolStandardize is None
        or MurckoScaffold is None
    ):
        raise RuntimeError("RDKit is required to build NMR benchmark manifests")
    if measurement_policy not in MEASUREMENT_POLICIES:
        raise ValueError(f"unsupported measurement policy: {measurement_policy}")
    if review_policy not in REVIEW_POLICIES:
        raise ValueError(f"unsupported review policy: {review_policy}")

    conn = connect_readonly(index_path)
    try:
        version = schema_version(conn)
        metadata_row = conn.execute(
            "SELECT value FROM schema_metadata WHERE key = 'build_options'"
        ).fetchone()
        build_options = json.loads(metadata_row[0]) if metadata_row else {}
        snapshot_rows = conn.execute(
            """
            SELECT
                id, source_name, source_version, source_uri, sha256, byte_size,
                license_uri, imported_at, validation_json
            FROM source_snapshots
            ORDER BY id
            """
        ).fetchall()
        snapshots: dict[int, dict[str, Any]] = {}
        for row in snapshot_rows:
            value = dict(row)
            value["validation"] = json.loads(value.pop("validation_json"))
            snapshots[int(row["id"])] = value
        rows = conn.execute(
            """
            SELECT
                s.id AS spectrum_id,
                s.source_spectrum_id,
                s.spectrum_tag,
                s.nucleus,
                s.measurement_kind,
                s.review_status,
                s.solvent,
                s.field_mhz,
                s.temperature_k,
                s.reference,
                s.literature,
                s.rawdata_uri,
                s.metadata_json AS spectrum_metadata_json,
                m.snapshot_id,
                m.source_record_ordinal,
                m.source_molecule_id,
                m.record_sha256,
                m.molblock,
                m.inchi,
                m.inchi_key,
                m.smiles,
                m.formula AS source_formula
            FROM spectra AS s
            JOIN molecules AS m ON m.id = s.molecule_id
            ORDER BY m.snapshot_id, m.source_record_ordinal, s.id
            """
        ).fetchall()
    finally:
        conn.close()

    records: list[dict[str, Any]] = []
    invalid_structures = 0
    identity_mismatches = 0
    filtered_measurement = 0
    filtered_review = 0
    measurement_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    mismatch_examples: list[dict[str, Any]] = []

    for row in rows:
        kind = str(row["measurement_kind"])
        review = str(row["review_status"])
        measurement_counts[kind] += 1
        review_counts[review] += 1
        if not _measurement_is_eligible(kind, measurement_policy):
            filtered_measurement += 1
            continue
        if not _review_is_eligible(review, review_policy):
            filtered_review += 1
            continue
        spectrum_metadata = json.loads(str(row["spectrum_metadata_json"]))
        source_mismatch_fields = [
            field
            for field in (
                "source_inchi_key_mismatch",
                "source_smiles_mismatch",
                "source_formula_mismatch",
            )
            if spectrum_metadata.get(field) is True
        ]
        if source_mismatch_fields:
            identity_mismatches += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(
                    {
                        "source_record_ordinal": int(row["source_record_ordinal"]),
                        "spectrum_tag": str(row["spectrum_tag"]),
                        "reason": (
                            "source metadata conflicts with MolBlock: "
                            + ", ".join(source_mismatch_fields)
                        ),
                    }
                )
            continue
        try:
            identity = _structure_identifiers(
                molblock=str(row["molblock"]),
                inchi=row["inchi"],
                inchi_key=row["inchi_key"],
                smiles=row["smiles"],
                source_formula=row["source_formula"],
            )
        except StructureIdentityMismatch as exc:
            identity_mismatches += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(
                    {
                        "source_record_ordinal": int(row["source_record_ordinal"]),
                        "spectrum_tag": str(row["spectrum_tag"]),
                        "reason": str(exc),
                    }
                )
            continue
        except (RuntimeError, ValueError):
            # Real database snapshots contain a small number of malformed
            # stereochemical graphs that can pass SD parsing but fail during
            # canonical scaffold generation. Quarantine the complete spectrum
            # rather than aborting or weakening the split policy.
            invalid_structures += 1
            continue
        if identity is None:
            invalid_structures += 1
            continue

        snapshot = snapshots[int(row["snapshot_id"])]
        snapshot_sha = str(snapshot["sha256"])
        stable_key = (
            f"{snapshot_sha}:{row['source_record_ordinal']}:{row['spectrum_tag']}"
        )
        records.append(
            {
                "record_id": hashlib.sha256(stable_key.encode()).hexdigest(),
                "source_snapshot_sha256": snapshot_sha,
                "source_name": str(snapshot["source_name"]),
                "source_version": snapshot["source_version"],
                "source_imported_at": str(snapshot["imported_at"]),
                "source_record_ordinal": int(row["source_record_ordinal"]),
                "source_molecule_id": str(row["source_molecule_id"]),
                "source_spectrum_id": str(row["source_spectrum_id"]),
                "spectrum_tag": str(row["spectrum_tag"]),
                "nucleus": str(row["nucleus"]),
                "measurement_kind": kind,
                "review_status": review,
                "eligibility_policy": {
                    "measurement": measurement_policy,
                    "review": review_policy,
                    "inferred_measured_included": (
                        measurement_policy == "include_inferred_measured"
                    ),
                },
                "solvent": row["solvent"],
                "field_mhz": row["field_mhz"],
                "temperature_k": row["temperature_k"],
                "reference": row["reference"],
                "record_sha256": str(row["record_sha256"]),
                "source_document_key": _normalize_source_document_key(
                    str(row["literature"] or ""),
                    snapshot_sha,
                ),
                **identity,
            }
        )

    return records, {
        "schema_version": version,
        "eligibility_policy": {
            "measurement": measurement_policy,
            "review": review_policy,
            "inferred_measured_included": (
                measurement_policy == "include_inferred_measured"
            ),
        },
        "index_build_options": build_options,
        "source_snapshots": list(snapshots.values()),
        "spectra_seen": len(rows),
        "records_eligible": len(records),
        "filtered_measurement": filtered_measurement,
        "filtered_review": filtered_review,
        "invalid_structures": invalid_structures,
        "identity_mismatches": identity_mismatches,
        "identity_mismatch_examples": mismatch_examples,
        "measurement_counts": dict(sorted(measurement_counts.items())),
        "review_counts": dict(sorted(review_counts.items())),
    }


def _stable_digest(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def _validate_ratios(ratios: Mapping[str, float]) -> None:
    required = set(SPLIT_NAMES)
    if set(ratios) != required:
        raise ValueError(f"split ratios must contain exactly {sorted(required)}")
    if any(not math.isfinite(value) or value <= 0 for value in ratios.values()):
        raise ValueError("all split ratios must be finite and positive")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("split ratios must sum to 1")


def _connected_components(
    records: Sequence[dict[str, Any]],
    *,
    group_fields: Sequence[str],
) -> dict[str, list[dict[str, Any]]]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    owners: dict[tuple[str, str], int] = {}
    record_ids: set[str] = set()
    for index, record in enumerate(records):
        record_id = str(record.get("record_id") or "")
        if not record_id:
            raise ValueError("every record requires a non-empty record_id")
        if record_id in record_ids:
            raise ValueError(f"duplicate record_id: {record_id}")
        record_ids.add(record_id)
        for field in group_fields:
            value = str(record.get(field) or "")
            if not value:
                raise ValueError(f"record {record_id} has no {field}")
            key = (field, value)
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index

    components: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        components[find(index)].append(record)

    result: dict[str, list[dict[str, Any]]] = {}
    for values in components.values():
        component_records = sorted(str(row["record_id"]) for row in values)
        component_key = hashlib.sha256(
            "\0".join(component_records).encode()
        ).hexdigest()
        result[component_key] = values
    return result


def assign_grouped_splits(
    records: Sequence[dict[str, Any]],
    *,
    group_field: str,
    seed: int,
    ratios: dict[str, float] | None = None,
    additional_group_fields: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Assign connected leakage groups to deterministic, count-balanced splits."""

    selected_ratios = dict(DEFAULT_SPLIT_RATIOS if ratios is None else ratios)
    _validate_ratios(selected_ratios)
    if not records:
        return []

    group_fields = tuple(dict.fromkeys((group_field, *additional_group_fields)))
    grouped = _connected_components(records, group_fields=group_fields)
    if len(grouped) < len(SPLIT_NAMES):
        raise ValueError("at least four independent leakage groups are required")

    total = len(records)
    targets = {name: selected_ratios[name] * total for name in SPLIT_NAMES}
    counts = {name: 0 for name in SPLIT_NAMES}
    assignments: dict[str, str] = {}
    ordered_groups = sorted(
        grouped,
        key=lambda key: (-len(grouped[key]), _stable_digest(key, seed)),
    )

    for position, group in enumerate(ordered_groups):
        group_size = len(grouped[group])
        empty = [name for name in SPLIT_NAMES if counts[name] == 0]
        remaining_after = len(ordered_groups) - position - 1
        candidates = (
            empty if empty and remaining_after < len(empty) else list(SPLIT_NAMES)
        )

        def objective(name: str) -> tuple[float, float, str]:
            trial = dict(counts)
            trial[name] += group_size
            normalized_error = sum(
                ((trial[item] - targets[item]) / targets[item]) ** 2
                for item in SPLIT_NAMES
            )
            overflow = sum(
                max(0.0, trial[item] - targets[item]) / targets[item]
                for item in SPLIT_NAMES
            )
            return (
                normalized_error,
                overflow,
                _stable_digest(f"{group}:{name}", seed),
            )

        split = min(candidates, key=objective)
        assignments[group] = split
        counts[split] += group_size

    component_by_record = {
        str(record["record_id"]): component
        for component, values in grouped.items()
        for record in values
    }
    output = []
    for record in sorted(records, key=lambda item: str(item["record_id"])):
        record_id = str(record["record_id"])
        component = component_by_record[record_id]
        output.append(
            {
                **record,
                "split": assignments[component],
                "split_group_field": group_field,
                "split_group_fields": list(group_fields),
                "split_group": component,
                "split_seed": seed,
                "split_ratios_requested": selected_ratios,
                "protocol_version": BENCHMARK_PROTOCOL_VERSION,
            }
        )
    for field in group_fields:
        audit_group_leakage(output, group_field=field)
    return output


def _parse_timestamp(value: Any, label: str) -> datetime:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is empty")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def assign_time_splits(
    records: Sequence[dict[str, Any]],
    *,
    group_field: str,
    time_field: str,
    ratios: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    """Create a chronological split while keeping every molecule group intact."""

    selected_ratios = dict(DEFAULT_SPLIT_RATIOS if ratios is None else ratios)
    _validate_ratios(selected_ratios)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for record in records:
        record_id = str(record.get("record_id") or "")
        group = str(record.get(group_field) or "")
        if not record_id or not group:
            raise ValueError(
                f"time split requires record_id, {group_field}, and {time_field}"
            )
        if record_id in seen_ids:
            raise ValueError(f"duplicate record_id: {record_id}")
        seen_ids.add(record_id)
        _parse_timestamp(record.get(time_field), f"{record_id}.{time_field}")
        grouped[group].append(record)
    if len(grouped) < 4:
        raise ValueError("at least four distinct chronological groups are required")

    group_times = []
    for group, values in grouped.items():
        latest = max(
            _parse_timestamp(item[time_field], f"{item['record_id']}.{time_field}")
            for item in values
        )
        group_times.append((latest, group))
    group_times.sort()

    total = len(records)
    cumulative = 0
    boundaries = [
        selected_ratios["train"] * total,
        (selected_ratios["train"] + selected_ratios["validation"]) * total,
        (
            selected_ratios["train"]
            + selected_ratios["validation"]
            + selected_ratios["calibration"]
        )
        * total,
    ]
    assignments: dict[str, str] = {}
    for _, group in group_times:
        split_index = sum(cumulative >= boundary for boundary in boundaries)
        assignments[group] = SPLIT_NAMES[min(split_index, 3)]
        cumulative += len(grouped[group])

    output = [
        {
            **record,
            "split": assignments[str(record[group_field])],
            "split_group_field": group_field,
            "split_group_fields": [group_field],
            "split_group": str(record[group_field]),
            "split_time_field": time_field,
            "split_ratios_requested": selected_ratios,
            "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        }
        for record in sorted(records, key=lambda item: str(item["record_id"]))
    ]
    audit_group_leakage(output, group_field=group_field)
    missing = set(SPLIT_NAMES) - {str(row["split"]) for row in output}
    if missing:
        raise ValueError(
            "chronological grouping produced empty split(s): "
            + ", ".join(sorted(missing))
        )
    return output


def audit_group_leakage(
    manifest: Sequence[dict[str, Any]],
    *,
    group_field: str,
) -> dict[str, Any]:
    """Raise on cross-split group leakage and return an audit summary."""

    locations: dict[str, set[str]] = defaultdict(set)
    record_ids: set[str] = set()
    for row in manifest:
        record_id = str(row.get("record_id") or "")
        group = str(row.get(group_field) or "")
        split = str(row.get("split") or "")
        if not record_id or not group or not split:
            raise ValueError("manifest row is missing record_id, group, or split")
        if split not in SPLIT_NAMES:
            raise ValueError(f"unsupported split name: {split}")
        if record_id in record_ids:
            raise ValueError(f"duplicate manifest record_id: {record_id}")
        record_ids.add(record_id)
        locations[group].add(split)
    leaked = {
        group: sorted(splits) for group, splits in locations.items() if len(splits) > 1
    }
    if leaked:
        example = next(iter(leaked.items()))
        raise ValueError(f"group leakage detected for {example[0]}: {example[1]}")
    counts: dict[str, int] = defaultdict(int)
    for row in manifest:
        counts[str(row["split"])] += 1
    return {
        "records": len(manifest),
        "groups": len(locations),
        "counts": dict(sorted(counts.items())),
        "leaked_groups": 0,
    }


def validate_split_manifest(
    manifest: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate a frozen split manifest and return its record lookup."""

    if not manifest:
        raise ValueError("split manifest is empty")
    required = {
        "record_id",
        "protocol_version",
        "split",
        "split_group_fields",
        "molecule_key",
        "scaffold_key",
        "inchi_key",
        "formula",
        "nucleus",
        "measurement_kind",
        "review_status",
        "eligibility_policy",
    }
    lookup: dict[str, dict[str, Any]] = {}
    group_fields: tuple[str, ...] | None = None
    eligibility_policy: dict[str, Any] | None = None
    for row_value in manifest:
        row = dict(row_value)
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"manifest row is missing fields: {', '.join(sorted(missing))}"
            )
        if row["protocol_version"] != BENCHMARK_PROTOCOL_VERSION:
            raise ValueError("manifest protocol_version is incompatible")
        record_id = str(row["record_id"] or "")
        if not record_id or record_id in lookup:
            raise ValueError(f"duplicate or empty manifest record_id: {record_id}")
        if row["split"] not in SPLIT_NAMES:
            raise ValueError(f"unsupported split name: {row['split']}")
        fields_value = row["split_group_fields"]
        if (
            not isinstance(fields_value, list)
            or not fields_value
            or not all(isinstance(field, str) and field for field in fields_value)
        ):
            raise ValueError("split_group_fields must be a non-empty string list")
        fields = tuple(fields_value)
        if group_fields is None:
            group_fields = fields
        elif fields != group_fields:
            raise ValueError("manifest rows disagree on split_group_fields")
        if row["nucleus"] not in ALLOWED_NUCLEI:
            raise ValueError(f"unsupported manifest nucleus: {row['nucleus']}")
        policy = row["eligibility_policy"]
        if (
            not isinstance(policy, Mapping)
            or set(policy) != {"measurement", "review", "inferred_measured_included"}
            or policy["measurement"] not in MEASUREMENT_POLICIES
            or policy["review"] not in REVIEW_POLICIES
            or type(policy["inferred_measured_included"]) is not bool
        ):
            raise ValueError("manifest eligibility_policy is invalid")
        expected_inferred = policy["measurement"] == "include_inferred_measured"
        if policy["inferred_measured_included"] != expected_inferred:
            raise ValueError("manifest inferred-measured policy is contradictory")
        if not _measurement_is_eligible(
            str(row["measurement_kind"]),
            policy["measurement"],
        ):
            raise ValueError("manifest contains an ineligible measurement kind")
        if not _review_is_eligible(
            str(row["review_status"]),
            policy["review"],
        ):
            raise ValueError("manifest contains an ineligible review status")
        normalized_policy = dict(policy)
        if eligibility_policy is None:
            eligibility_policy = normalized_policy
        elif normalized_policy != eligibility_policy:
            raise ValueError("manifest rows disagree on eligibility_policy")
        lookup[record_id] = row

    assert group_fields is not None
    materialized = list(lookup.values())
    for field in group_fields:
        audit_group_leakage(materialized, group_field=field)
    # Exact molecules may never cross splits, even in source-based protocols.
    audit_group_leakage(materialized, group_field="molecule_key")
    missing_splits = set(SPLIT_NAMES) - {str(row["split"]) for row in materialized}
    if missing_splits:
        raise ValueError(
            "manifest has empty split(s): " + ", ".join(sorted(missing_splits))
        )
    return lookup


def _validate_artifact(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise ValueError(f"{label} must contain exactly path and sha256")
    path = str(value["path"] or "")
    if not path:
        raise ValueError(f"{label}.path must be non-empty")
    return {
        "path": path,
        "sha256": _require_hash(value["sha256"], f"{label}.sha256"),
    }


def validate_benchmark_registration(
    registration_value: Mapping[str, Any],
    *,
    manifest: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate registration schema, identity, artifact hashes, and test set."""

    registration = dict(registration_value)
    required = {
        "registration_schema_version",
        "protocol_version",
        "registration_id",
        "task",
        "created_at",
        "split_roles",
        "split_manifest",
        "artifacts",
    }
    optional = {"notes"}
    missing = required - set(registration)
    extras = set(registration) - required - optional
    if missing or extras:
        raise ValueError(
            "registration schema mismatch; "
            f"missing={sorted(missing)}, unexpected={sorted(extras)}"
        )
    if registration["registration_schema_version"] != REGISTRATION_SCHEMA_VERSION:
        raise ValueError("registration schema version is incompatible")
    if registration["protocol_version"] != BENCHMARK_PROTOCOL_VERSION:
        raise ValueError("registration protocol version is incompatible")
    if registration["task"] not in {
        "closed_library",
        "open_exact_removed",
        "open_scaffold_removed",
    }:
        raise ValueError("registration task is unsupported")
    _parse_timestamp(registration["created_at"], "registration.created_at")
    expected_roles = {
        "model_fit": ["train"],
        "model_selection": ["validation"],
        "calibrator_fit": ["calibration"],
        "evaluation": ["test"],
    }
    if registration["split_roles"] != expected_roles:
        raise ValueError("registration split_roles are not strictly separated")

    split_manifest = registration["split_manifest"]
    if not isinstance(split_manifest, Mapping) or set(split_manifest) != {
        "path",
        "sha256",
        "content_sha256",
        "test_record_count",
    }:
        raise ValueError("split_manifest registration block has an invalid schema")
    _require_hash(split_manifest["sha256"], "split_manifest.sha256")
    content_sha = _require_hash(
        split_manifest["content_sha256"],
        "split_manifest.content_sha256",
    )
    if isinstance(split_manifest["test_record_count"], bool) or not isinstance(
        split_manifest["test_record_count"],
        int,
    ):
        raise ValueError("split_manifest.test_record_count must be an integer")

    manifest_lookup = validate_split_manifest(manifest)
    if canonical_jsonl_sha256(manifest) != content_sha:
        raise ValueError("split manifest content hash does not match registration")
    test_count = sum(row["split"] == "test" for row in manifest_lookup.values())
    if split_manifest["test_record_count"] != test_count:
        raise ValueError("registered test record count does not match manifest")

    artifacts = registration["artifacts"]
    required_artifacts = {
        "base_index",
        "evaluation_index",
        "model",
        "calibrator",
        "run_config",
        "exclusion_proof",
    }
    if not isinstance(artifacts, Mapping) or set(artifacts) != required_artifacts:
        raise ValueError("registration artifacts have an invalid schema")
    validated_artifacts = {}
    for name in required_artifacts - {"exclusion_proof"}:
        validated_artifacts[name] = _validate_artifact(
            artifacts[name],
            f"artifacts.{name}",
        )
    base_hash = validated_artifacts["base_index"]["sha256"]
    evaluation_hash = validated_artifacts["evaluation_index"]["sha256"]
    if registration["task"] == "closed_library" and base_hash != evaluation_hash:
        raise ValueError(
            "closed-library registration requires identical base/evaluation indexes"
        )
    if registration["task"] != "closed_library" and base_hash == evaluation_hash:
        raise ValueError("open-world registration requires a distinct evaluation index")
    exclusion = artifacts["exclusion_proof"]
    if registration["task"] == "closed_library":
        if exclusion is not None:
            _validate_artifact(exclusion, "artifacts.exclusion_proof")
    elif exclusion is None:
        raise ValueError("open-world registration requires an exclusion proof")
    else:
        _validate_artifact(exclusion, "artifacts.exclusion_proof")

    expected_id = benchmark_registration_id(registration)
    if registration["registration_id"] != expected_id:
        raise ValueError("registration_id does not match the registration payload")
    return registration


def _candidate_identity(candidate: Mapping[str, Any]) -> dict[str, str]:
    required = {"structure_id", "canonical_smiles", "ranking_score"}
    optional = {"formula", "metadata"}
    missing = required - set(candidate)
    extras = set(candidate) - required - optional
    if missing or extras:
        raise ValueError(
            "candidate schema mismatch; "
            f"missing={sorted(missing)}, unexpected={sorted(extras)}"
        )
    if Chem is None or rdMolDescriptors is None:
        raise RuntimeError("RDKit is required to validate benchmark candidates")
    smiles = str(candidate["canonical_smiles"] or "")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("candidate canonical_smiles cannot be parsed")
    computed_id = _mol_inchi_key(mol)
    declared_id = str(candidate["structure_id"] or "").upper()
    if computed_id != declared_id:
        raise ValueError("candidate structure_id does not match canonical_smiles")
    formula = rdMolDescriptors.CalcMolFormula(mol)
    if "formula" in candidate and str(candidate["formula"]) != formula:
        raise ValueError("candidate formula does not match canonical_smiles")
    score = _require_number(candidate["ranking_score"], "candidate.ranking_score")
    return {
        "structure_id": computed_id,
        "formula": formula,
        "ranking_score": score,
    }


def _modality_for_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    nuclei = {str(row["nucleus"]) for row in rows}
    if nuclei == {"1H"}:
        return "1h"
    if nuclei == {"13C"}:
        return "13c"
    if nuclei == {"1H", "13C"}:
        return "1h+13c"
    raise ValueError(f"unsupported case nuclei: {sorted(nuclei)}")


def _canonical_case(
    case_value: Mapping[str, Any],
    *,
    task: BenchmarkTask,
    registration: Mapping[str, Any],
    manifest_lookup: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    required = {
        "protocol_version",
        "registration_id",
        "task",
        "case_id",
        "record_ids",
        "target_id",
        "target_formula",
        "target_removed_from_index",
        "target_scaffold_removed_from_index",
        "candidate_pool_structure_ids",
        "candidate_pool_size",
        "candidate_pool_contains_target",
        "modality",
        "abstained",
        "selected_structure_id",
        "calibrated_probability",
        "candidates",
    }
    optional = {"metadata"}
    case = dict(case_value)
    missing = required - set(case)
    extras = set(case) - required - optional
    if missing or extras:
        raise ValueError(
            "case schema mismatch; "
            f"missing={sorted(missing)}, unexpected={sorted(extras)}"
        )
    if case["protocol_version"] != BENCHMARK_PROTOCOL_VERSION:
        raise ValueError("case has an incompatible protocol_version")
    if case["registration_id"] != registration["registration_id"]:
        raise ValueError("case registration_id does not match registration")
    if case["task"] != task or task != registration["task"]:
        raise ValueError("case task does not match the registered benchmark")
    case_id = str(case["case_id"] or "")
    if not case_id:
        raise ValueError("case_id must be non-empty")

    record_ids_value = case["record_ids"]
    if (
        not isinstance(record_ids_value, list)
        or not record_ids_value
        or not all(isinstance(item, str) and item for item in record_ids_value)
        or len(set(record_ids_value)) != len(record_ids_value)
    ):
        raise ValueError("record_ids must be a non-empty unique string list")
    try:
        records = [manifest_lookup[item] for item in record_ids_value]
    except KeyError as exc:
        raise ValueError(
            f"case references unknown manifest record: {exc.args[0]}"
        ) from exc
    if any(record["split"] != "test" for record in records):
        raise ValueError("benchmark cases may reference only the frozen test split")
    molecule_keys = {str(record["molecule_key"]) for record in records}
    target_ids = {str(record["inchi_key"]).upper() for record in records}
    formulas = {str(record["formula"]) for record in records}
    if len(molecule_keys) != 1 or len(target_ids) != 1 or len(formulas) != 1:
        raise ValueError("all records in a case must describe one target molecule")
    target_id = str(case["target_id"] or "").upper()
    target_formula = str(case["target_formula"] or "")
    if target_ids != {target_id} or formulas != {target_formula}:
        raise ValueError("case target identity/formula does not match split manifest")
    modality = str(case["modality"] or "")
    if modality not in ALLOWED_MODALITIES or modality != _modality_for_rows(records):
        raise ValueError("case modality does not match its manifest records")

    removed = _require_bool(case, "target_removed_from_index")
    scaffold_removed = _require_bool(
        case,
        "target_scaffold_removed_from_index",
    )
    if task == "closed_library" and (removed or scaffold_removed):
        raise ValueError("closed-library case cannot remove target or scaffold")
    if task == "open_exact_removed" and (not removed or scaffold_removed):
        raise ValueError("exact-removed case must remove only the exact target")
    if task == "open_scaffold_removed" and not (removed and scaffold_removed):
        raise ValueError("scaffold-removed case must remove target and scaffold")

    pool = case["candidate_pool_structure_ids"]
    if (
        not isinstance(pool, list)
        or not all(isinstance(item, str) and item for item in pool)
        or len(set(pool)) != len(pool)
    ):
        raise ValueError("candidate_pool_structure_ids must be a unique string list")
    normalized_pool = [item.upper() for item in pool]
    if len(set(normalized_pool)) != len(normalized_pool):
        raise ValueError(
            "candidate pool contains case-insensitive duplicate identities"
        )
    pool_size = case["candidate_pool_size"]
    if isinstance(pool_size, bool) or not isinstance(pool_size, int) or pool_size < 0:
        raise ValueError("candidate_pool_size must be a non-negative integer")
    if pool_size != len(normalized_pool):
        raise ValueError("candidate_pool_size does not match candidate pool")
    declared_coverage = _require_bool(case, "candidate_pool_contains_target")
    derived_coverage = target_id in normalized_pool
    if declared_coverage != derived_coverage:
        raise ValueError("candidate_pool_contains_target contradicts candidate pool")

    candidates_value = case["candidates"]
    if not isinstance(candidates_value, list):
        raise ValueError("candidates must be a list")
    candidates = [_candidate_identity(candidate) for candidate in candidates_value]
    candidate_ids = [candidate["structure_id"] for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("ranked candidates contain duplicate structure identities")
    if any(candidate_id not in normalized_pool for candidate_id in candidate_ids):
        raise ValueError(
            "ranked candidate is absent from the registered candidate pool"
        )
    scores = [float(candidate["ranking_score"]) for candidate in candidates]
    if any(left < right for left, right in zip(scores, scores[1:])):
        raise ValueError("candidates must be sorted by non-increasing ranking_score")

    abstained = _require_bool(case, "abstained")
    selected = case["selected_structure_id"]
    if selected is not None and not isinstance(selected, str):
        raise ValueError("selected_structure_id must be a string or null")
    normalized_selected = str(selected).upper() if selected else None
    if abstained and normalized_selected is not None:
        raise ValueError("an abstained case cannot select a structure")
    if not abstained:
        if not candidates:
            raise ValueError("a non-abstained case requires ranked candidates")
        if normalized_selected != candidate_ids[0]:
            raise ValueError(
                "selected_structure_id must equal the top-ranked candidate"
            )
    if normalized_selected and normalized_selected not in normalized_pool:
        raise ValueError("selected_structure_id is absent from candidate pool")

    probability = _require_number(
        case["calibrated_probability"],
        "calibrated_probability",
    )
    if not 0 <= probability <= 1:
        raise ValueError("calibrated_probability must be in [0, 1]")
    if not candidates and probability != 0:
        raise ValueError("a case without candidates must have zero probability")

    return {
        **case,
        "case_id": case_id,
        "record_ids": list(record_ids_value),
        "target_id": target_id,
        "target_formula": target_formula,
        "candidate_pool_structure_ids": normalized_pool,
        "candidate_pool_contains_target": derived_coverage,
        "candidates": candidates,
        "abstained": abstained,
        "selected_structure_id": normalized_selected,
        "calibrated_probability": probability,
        "_molecule_key": next(iter(molecule_keys)),
    }


def _cluster_bootstrap_interval(
    values: Sequence[float],
    clusters: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> list[float] | None:
    if not values:
        return None
    if len(values) != len(clusters):
        raise ValueError("bootstrap values and clusters have different lengths")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(value)
    cluster_names = sorted(grouped)
    rng = random.Random(seed)
    means = []
    for _ in range(replicates):
        sample: list[float] = []
        for _ in cluster_names:
            selected = cluster_names[rng.randrange(len(cluster_names))]
            sample.extend(grouped[selected])
        means.append(sum(sample) / len(sample))
    means.sort()
    low = means[int(0.025 * (replicates - 1))]
    high = means[int(0.975 * (replicates - 1))]
    return [round(low, 6), round(high, 6)]


def _calibration_metrics(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    bins: int = 10,
) -> dict[str, Any]:
    if not probabilities or len(probabilities) != len(outcomes):
        raise ValueError("calibration requires one probability per benchmark case")
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in probabilities):
        raise ValueError("calibrated probabilities must be finite values in [0, 1]")
    brier = sum(
        (probability - outcome) ** 2
        for probability, outcome in zip(probabilities, outcomes)
    ) / len(probabilities)
    rows = []
    ece = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            item
            for item, probability in enumerate(probabilities)
            if low <= probability < high or (index == bins - 1 and probability == 1)
        ]
        if not members:
            continue
        confidence = sum(probabilities[item] for item in members) / len(members)
        accuracy = sum(outcomes[item] for item in members) / len(members)
        weight = len(members) / len(probabilities)
        ece += weight * abs(confidence - accuracy)
        rows.append(
            {
                "lower": low,
                "upper": high,
                "count": len(members),
                "mean_probability": round(confidence, 6),
                "accuracy": round(accuracy, 6),
            }
        )
    return {
        "cases": len(probabilities),
        "missing_probabilities": 0,
        "brier": round(brier, 6),
        "ece": round(ece, 6),
        "reliability_bins": rows,
    }


def _accuracy_coverage(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
) -> list[dict[str, float]]:
    rows = []
    for threshold in sorted({0.0, *probabilities}, reverse=True):
        accepted = [
            outcome
            for probability, outcome in zip(probabilities, outcomes)
            if probability >= threshold
        ]
        if not accepted:
            continue
        accuracy = sum(accepted) / len(accepted)
        rows.append(
            {
                "threshold": round(threshold, 6),
                "coverage": round(len(accepted) / len(outcomes), 6),
                "accuracy": round(accuracy, 6),
                "risk": round(1.0 - accuracy, 6),
            }
        )
    return rows


def _area_under_risk_coverage(
    curve: Sequence[Mapping[str, float]],
) -> float:
    """Integrate the right-continuous selective-risk curve."""

    area = 0.0
    previous_coverage = 0.0
    for row in sorted(curve, key=lambda item: item["coverage"]):
        coverage = float(row["coverage"])
        area += (coverage - previous_coverage) * float(row["risk"])
        previous_coverage = coverage
    return area


def _top_k(rank_values: Iterable[int | None], k: int) -> float:
    values = list(rank_values)
    return sum(rank is not None and rank <= k for rank in values) / len(values)


def _pool_bucket(size: int) -> str:
    if size <= 1:
        return "1"
    if size <= 5:
        return "2-5"
    if size <= 20:
        return "6-20"
    return ">20"


def evaluate_rankings(
    cases: Sequence[Mapping[str, Any]],
    *,
    task: BenchmarkTask,
    manifest: Sequence[Mapping[str, Any]],
    registration: Mapping[str, Any],
    bootstrap_seed: int = 20260711,
    bootstrap_replicates: int = 1000,
) -> dict[str, Any]:
    """Evaluate a complete, pre-registered frozen-test ranking run."""

    if bootstrap_replicates < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    validated_registration = validate_benchmark_registration(
        registration,
        manifest=manifest,
    )
    manifest_lookup = validate_split_manifest(manifest)
    if task != validated_registration["task"]:
        raise ValueError("requested task does not match registration")
    if not cases:
        raise ValueError("benchmark requires at least one case")

    normalized: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    used_record_ids: set[str] = set()
    for case_value in cases:
        if not isinstance(case_value, Mapping):
            raise ValueError("every benchmark case must be a JSON object")
        case = _canonical_case(
            case_value,
            task=task,
            registration=validated_registration,
            manifest_lookup=manifest_lookup,
        )
        if case["case_id"] in case_ids:
            raise ValueError(f"duplicate case_id: {case['case_id']}")
        case_ids.add(case["case_id"])
        overlap = used_record_ids.intersection(case["record_ids"])
        if overlap:
            raise ValueError(
                f"test record appears in multiple cases: {sorted(overlap)[0]}"
            )
        used_record_ids.update(case["record_ids"])
        normalized.append(case)

    expected_test_records = {
        record_id
        for record_id, row in manifest_lookup.items()
        if row["split"] == "test"
    }
    if used_record_ids != expected_test_records:
        omitted = sorted(expected_test_records - used_record_ids)
        unexpected = sorted(used_record_ids - expected_test_records)
        raise ValueError(
            "cases must cover every frozen test record exactly once; "
            f"omitted={omitted[:3]}, unexpected={unexpected[:3]}"
        )

    ranks: list[int | None] = []
    pool_coverage: list[float] = []
    formula_violation_cases: list[float] = []
    formula_violations = 0
    returned = 0
    accepted: list[float] = []
    selected_correct: list[float] = []
    probabilities: list[float] = []
    outcomes: list[float] = []
    molecule_clusters: list[str] = []
    modalities: dict[str, list[int | None]] = defaultdict(list)
    pool_buckets: dict[str, list[int | None]] = defaultdict(list)

    for case in normalized:
        target = str(case["target_id"])
        candidates = list(case["candidates"])
        candidate_ids = [candidate["structure_id"] for candidate in candidates]
        found_rank = (
            candidate_ids.index(target) + 1 if target in candidate_ids else None
        )
        ranks.append(found_rank)
        covered = float(case["candidate_pool_contains_target"])
        pool_coverage.append(covered)
        violations = sum(
            candidate["formula"] != case["target_formula"] for candidate in candidates
        )
        formula_violations += violations
        returned += len(candidates)
        formula_violation_cases.append(float(violations > 0))
        is_accepted = float(not case["abstained"])
        accepted.append(is_accepted)
        selected_correct.append(
            float(not case["abstained"] and case["selected_structure_id"] == target)
        )
        probabilities.append(float(case["calibrated_probability"]))
        outcomes.append(float(found_rank == 1))
        molecule_clusters.append(str(case["_molecule_key"]))
        modalities[str(case["modality"])].append(found_rank)
        pool_buckets[_pool_bucket(int(case["candidate_pool_size"]))].append(found_rank)

    reciprocal_ranks = [0.0 if rank is None else 1.0 / rank for rank in ranks]
    top1_values = [float(rank == 1) for rank in ranks]
    accepted_count = int(sum(accepted))
    accuracy_coverage = _accuracy_coverage(probabilities, outcomes)
    metrics = {
        "exact_top1": round(_top_k(ranks, 1), 6),
        "exact_top5": round(_top_k(ranks, 5), 6),
        "exact_top10": round(_top_k(ranks, 10), 6),
        "mrr": round(sum(reciprocal_ranks) / len(ranks), 6),
        "decision_exact_top1": round(sum(selected_correct) / len(normalized), 6),
        "candidate_coverage": round(sum(pool_coverage) / len(pool_coverage), 6),
        "conditional_top1": (
            round(
                sum(rank == 1 for rank, covered in zip(ranks, pool_coverage) if covered)
                / sum(pool_coverage),
                6,
            )
            if sum(pool_coverage)
            else None
        ),
        "formula_violation_rate": (
            round(formula_violations / returned, 6) if returned else 0.0
        ),
        "formula_violation_case_rate": round(
            sum(formula_violation_cases) / len(normalized),
            6,
        ),
        "acceptance_coverage": round(accepted_count / len(normalized), 6),
        "refusal_rate": round(1.0 - accepted_count / len(normalized), 6),
        "selective_accuracy": (
            round(sum(selected_correct) / accepted_count, 6) if accepted_count else None
        ),
        "aurc": round(_area_under_risk_coverage(accuracy_coverage), 6),
        "molecule_clusters": len(set(molecule_clusters)),
        "top1_95ci": _cluster_bootstrap_interval(
            top1_values,
            molecule_clusters,
            seed=bootstrap_seed,
            replicates=bootstrap_replicates,
        ),
        "mrr_95ci": _cluster_bootstrap_interval(
            reciprocal_ranks,
            molecule_clusters,
            seed=bootstrap_seed + 1,
            replicates=bootstrap_replicates,
        ),
    }
    by_modality = {
        modality: {
            "cases": len(values),
            "exact_top1": round(_top_k(values, 1), 6),
            "exact_top5": round(_top_k(values, 5), 6),
        }
        for modality, values in sorted(modalities.items())
    }
    by_pool_size = {
        bucket: {
            "cases": len(values),
            "exact_top1": round(_top_k(values, 1), 6),
            "exact_top5": round(_top_k(values, 5), 6),
        }
        for bucket, values in sorted(pool_buckets.items())
    }
    return {
        "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "registration_id": validated_registration["registration_id"],
        "task": task,
        "cases": len(normalized),
        "test_records": len(used_record_ids),
        "metrics": metrics,
        "by_modality": by_modality,
        "by_candidate_pool_size": by_pool_size,
        "calibration": _calibration_metrics(probabilities, outcomes),
        "accuracy_coverage": accuracy_coverage,
        "interpretation": (
            "Top-K measures the complete ranked list. decision_exact_top1 and "
            "selective_accuracy apply the explicit abstention decision. "
            "Confidence calibration includes every frozen test case."
        ),
    }


def _forward_expected_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    required = {
        "case_id",
        "molecule_key",
        "nucleus",
        "atom_id",
        "observed_shift",
    }
    optional = {"equivalence_class"}
    missing = required - set(row)
    extras = set(row) - required - optional
    if missing or extras:
        raise ValueError(
            "expected atom schema mismatch; "
            f"missing={sorted(missing)}, unexpected={sorted(extras)}"
        )
    case_id = str(row["case_id"] or "")
    nucleus = str(row["nucleus"] or "")
    atom_id = str(row["atom_id"] or "")
    if not case_id or not atom_id or nucleus not in ALLOWED_NUCLEI:
        raise ValueError("expected atom requires case_id, atom_id, and 1H/13C nucleus")
    _require_number(row["observed_shift"], "observed_shift")
    if not str(row["molecule_key"] or ""):
        raise ValueError("expected atom requires molecule_key")
    return case_id, nucleus, atom_id


def _forward_prediction_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    required = {
        "case_id",
        "nucleus",
        "atom_id",
        "predicted_shift",
        "uncertainty",
        "uncertainty_kind",
    }
    if set(row) != required:
        raise ValueError(
            "forward prediction schema mismatch; "
            f"missing={sorted(required - set(row))}, "
            f"unexpected={sorted(set(row) - required)}"
        )
    case_id = str(row["case_id"] or "")
    nucleus = str(row["nucleus"] or "")
    atom_id = str(row["atom_id"] or "")
    if not case_id or not atom_id or nucleus not in ALLOWED_NUCLEI:
        raise ValueError("prediction requires case_id, atom_id, and 1H/13C nucleus")
    _require_number(row["predicted_shift"], "predicted_shift")
    uncertainty = row["uncertainty"]
    uncertainty_kind = row["uncertainty_kind"]
    if uncertainty is None:
        if uncertainty_kind is not None:
            raise ValueError("uncertainty_kind must be null when uncertainty is null")
    else:
        value = _require_number(uncertainty, "uncertainty")
        if value <= 0:
            raise ValueError("uncertainty must be positive")
        if uncertainty_kind not in {"gaussian_sigma", "interval_95_half_width"}:
            raise ValueError("uncertainty_kind is unsupported")
    return case_id, nucleus, atom_id


def evaluate_forward_predictions(
    predictions: Sequence[Mapping[str, Any]],
    *,
    expected_atoms: Sequence[Mapping[str, Any]],
    require_uncertainty: bool = True,
) -> dict[str, Any]:
    """Evaluate complete atom predictions, separately for each nucleus."""

    if type(require_uncertainty) is not bool:
        raise ValueError("require_uncertainty must be a boolean")
    if not expected_atoms:
        raise ValueError("forward evaluation requires expected atoms")
    expected_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in expected_atoms:
        key = _forward_expected_key(row)
        if key in expected_lookup:
            raise ValueError(f"duplicate expected atom: {key}")
        expected_lookup[key] = dict(row)

    prediction_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in predictions:
        key = _forward_prediction_key(row)
        if key in prediction_lookup:
            raise ValueError(f"duplicate forward prediction: {key}")
        prediction_lookup[key] = dict(row)
    missing = sorted(set(expected_lookup) - set(prediction_lookup))
    unexpected = sorted(set(prediction_lookup) - set(expected_lookup))
    if missing or unexpected:
        raise ValueError(
            "forward predictions must cover expected atoms exactly; "
            f"missing={missing[:3]}, unexpected={unexpected[:3]}"
        )
    if require_uncertainty and any(
        row["uncertainty"] is None for row in prediction_lookup.values()
    ):
        raise ValueError("complete uncertainty is required for this benchmark")

    by_nucleus_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    molecule_keys: set[str] = set()
    for key, expected in expected_lookup.items():
        prediction = prediction_lookup[key]
        observed = _require_number(expected["observed_shift"], "observed_shift")
        predicted = _require_number(prediction["predicted_shift"], "predicted_shift")
        signed_error = predicted - observed
        molecule_key = str(expected["molecule_key"])
        molecule_keys.add(molecule_key)
        uncertainty = prediction["uncertainty"]
        covered: float | None = None
        standardized: float | None = None
        if uncertainty is not None:
            width = float(uncertainty)
            if prediction["uncertainty_kind"] == "gaussian_sigma":
                covered = float(abs(signed_error) <= 1.959963984540054 * width)
                standardized = abs(signed_error) / width
            else:
                covered = float(abs(signed_error) <= width)
        by_nucleus_rows[key[1]].append(
            {
                "signed_error": signed_error,
                "absolute_error": abs(signed_error),
                "squared_error": signed_error**2,
                "molecule_key": molecule_key,
                "covered": covered,
                "standardized": standardized,
            }
        )

    by_nucleus: dict[str, dict[str, Any]] = {}
    for nucleus, rows in sorted(by_nucleus_rows.items()):
        absolute = [float(row["absolute_error"]) for row in rows]
        squared = [float(row["squared_error"]) for row in rows]
        signed = [float(row["signed_error"]) for row in rows]
        covered = [float(row["covered"]) for row in rows if row["covered"] is not None]
        standardized = [
            float(row["standardized"])
            for row in rows
            if row["standardized"] is not None
        ]
        per_molecule: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            per_molecule[str(row["molecule_key"])].append(float(row["absolute_error"]))
        molecule_mae = [sum(values) / len(values) for values in per_molecule.values()]
        by_nucleus[nucleus] = {
            "atoms_expected": len(rows),
            "atoms_predicted": len(rows),
            "prediction_completeness": 1.0,
            "molecules": len(per_molecule),
            "atom_micro_mae": round(float(np.mean(absolute)), 6),
            "atom_micro_rmse": round(float(np.sqrt(np.mean(squared))), 6),
            "signed_bias": round(float(np.mean(signed)), 6),
            "molecule_macro_mae": round(float(np.mean(molecule_mae)), 6),
            "uncertainty_atoms": len(covered),
            "uncertainty_completeness": round(len(covered) / len(rows), 6),
            "interval_95_coverage": (
                round(float(np.mean(covered)), 6) if covered else None
            ),
            "mean_absolute_standardized_error": (
                round(float(np.mean(standardized)), 6) if standardized else None
            ),
        }

    return {
        "atoms_expected": len(expected_lookup),
        "atoms_predicted": len(prediction_lookup),
        "prediction_completeness": 1.0,
        "molecules": len(molecule_keys),
        "pooled_raw_ppm_metric": None,
        "by_nucleus": by_nucleus,
        "interpretation": (
            "Raw ppm errors are not pooled across nuclei. Atom-micro and "
            "molecule-macro errors are both reported; uncertainty coverage "
            "uses the explicitly declared interval semantics."
        ),
    }
