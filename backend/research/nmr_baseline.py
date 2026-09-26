"""Deterministic, no-training baselines for the strict NMR benchmark.

The helpers in this module turn a frozen spectrum-level v2 index and split
manifest into formula-constrained benchmark cases.  They deliberately keep
candidate generation, spectral retrieval and confidence separate:

* the molecular formula is an exact candidate-pool constraint;
* the spectral score is the production heuristic, without a learned ranker;
* every result abstains because neither baseline has a calibrated confidence.

Closed-library self-reference and leave-source-record-out runs are distinct
protocols.  The former is only an optimistic library-lookup upper bound.  The
latter also removes sibling spectrum tags and exact peak-table duplicates; the
grouped split means that a target normally has no remaining exact-structure
reference spectrum.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any, Literal

from rdkit import Chem

from app.ml.nmr_benchmark import (
    BENCHMARK_PROTOCOL_VERSION,
    BenchmarkTask,
    canonical_json_dumps,
    validate_split_manifest,
)
from app.ml.nmr_data_v2 import connect_readonly, schema_version
from app.ml.nmr_structure_elucidation import _feature_vector


BASELINE_PROTOCOL_VERSION = "nmr-no-training-baseline-v2"
BaselineKind = Literal["formula_only", "spectral_library"]
ReferencePolicy = Literal[
    "not_applicable",
    "reference_in_library",
    "leave_source_record_out",
]


def file_sha256(path: str | Path) -> str:
    """Hash one immutable benchmark artifact."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    """Hash canonical JSONL rows without depending on incidental whitespace."""

    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_dumps(dict(row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a strict UTF-8 JSONL artifact."""

    result: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON on line {line_number} of {path}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"line {line_number} of {path} is not an object")
            result.append(value)
    return result


def load_bound_spectra(
    index_path: str | Path,
    manifest: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load peaks while proving every row belongs to the frozen manifest.

    Binding uses the source snapshot SHA, source-record ordinal and exact
    spectrum tag.  Nucleus and molecule-record hashes are checked again so a
    different SQLite file cannot be substituted behind an unchanged manifest.
    """

    manifest_lookup = validate_split_manifest(manifest)
    exact_identity_splits: dict[str, set[str]] = defaultdict(set)
    for row in manifest_lookup.values():
        exact_identity_splits[str(row["inchi_key"]).upper()].add(str(row["split"]))
    leaked_exact_identities = {
        identity: splits
        for identity, splits in exact_identity_splits.items()
        if len(splits) > 1
    }
    if leaked_exact_identities:
        example = next(iter(sorted(leaked_exact_identities.items())))
        raise ValueError(
            f"exact InChIKey leakage detected for {example[0]}: "
            f"{sorted(example[1])}"
        )
    index = Path(index_path).expanduser().resolve()
    before = index.stat()
    index_hash = file_sha256(index)
    conn = connect_readonly(index)
    try:
        version = schema_version(conn)
        rows = conn.execute(
            """
            SELECT
                ss.sha256 AS snapshot_sha256,
                m.source_record_ordinal,
                m.record_sha256,
                s.id AS spectrum_db_id,
                s.spectrum_tag,
                s.nucleus,
                p.ordinal,
                p.shift,
                p.intensity,
                p.multiplicity,
                p.atom_ref
            FROM spectra AS s
            JOIN molecules AS m ON m.id = s.molecule_id
            JOIN source_snapshots AS ss ON ss.id = m.snapshot_id
            LEFT JOIN peaks AS p ON p.spectrum_id = s.id
            ORDER BY ss.sha256, m.source_record_ordinal, s.spectrum_tag, p.ordinal
            """
        ).fetchall()
    finally:
        conn.close()
    after = index.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("v2 index changed while baseline spectra were being loaded")

    database_rows: dict[tuple[str, int, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            str(row["snapshot_sha256"]),
            int(row["source_record_ordinal"]),
            str(row["spectrum_tag"]),
        )
        current = database_rows.get(key)
        if current is None:
            current = {
                "record_sha256": str(row["record_sha256"]),
                "spectrum_db_id": int(row["spectrum_db_id"]),
                "nucleus": str(row["nucleus"]),
                "peaks": [],
            }
            database_rows[key] = current
        elif (
            current["record_sha256"] != str(row["record_sha256"])
            or current["spectrum_db_id"] != int(row["spectrum_db_id"])
            or current["nucleus"] != str(row["nucleus"])
        ):
            raise ValueError("v2 index contains an inconsistent spectrum identity")
        if row["ordinal"] is not None:
            current["peaks"].append(
                {
                    "shift": float(row["shift"]),
                    "intensity": float(row["intensity"]),
                    "multiplicity": str(row["multiplicity"] or ""),
                    "atom_ref": int(row["atom_ref"]),
                }
            )

    bound: list[dict[str, Any]] = []
    seen_database_keys: set[tuple[str, int, str]] = set()
    peak_counts: Counter[str] = Counter()
    for record_id, manifest_row in sorted(manifest_lookup.items()):
        key = (
            str(manifest_row.get("source_snapshot_sha256") or ""),
            int(manifest_row.get("source_record_ordinal", -1)),
            str(manifest_row.get("spectrum_tag") or ""),
        )
        if not all((key[0], key[2])) or key[1] < 0:
            raise ValueError(
                f"manifest record {record_id} lacks its source-spectrum binding"
            )
        database_row = database_rows.get(key)
        if database_row is None:
            raise ValueError(f"manifest spectrum is absent from v2 index: {record_id}")
        if key in seen_database_keys:
            raise ValueError("multiple manifest records bind to one database spectrum")
        seen_database_keys.add(key)
        if database_row["record_sha256"] != manifest_row.get("record_sha256"):
            raise ValueError(f"record hash mismatch for manifest spectrum: {record_id}")
        if database_row["nucleus"] != manifest_row["nucleus"]:
            raise ValueError(f"nucleus mismatch for manifest spectrum: {record_id}")
        peaks = list(database_row["peaks"])
        if not peaks:
            raise ValueError(f"manifest spectrum contains no indexed peaks: {record_id}")
        peak_counts[str(manifest_row["nucleus"])] += len(peaks)
        bound.append(
            {
                **dict(manifest_row),
                "spectrum_db_id": database_row["spectrum_db_id"],
                "peaks": peaks,
            }
        )

    return bound, {
        "baseline_protocol_version": BASELINE_PROTOCOL_VERSION,
        "schema_version": version,
        "index_path": str(index),
        "index_sha256": index_hash,
        "manifest_records": len(bound),
        "bound_spectra": len(seen_database_keys),
        "peaks_by_nucleus": dict(sorted(peak_counts.items())),
    }


def candidate_representation_audit(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Audit whether candidate SMILES round-trip to the exact target InChIKey."""

    grouped: dict[str, dict[str, Any]] = {}
    for record in records:
        structure_id = str(record["inchi_key"]).upper()
        value = grouped.setdefault(
            structure_id,
            {
                "smiles": set(),
                "formulas": set(),
                "splits": set(),
                "records": 0,
            },
        )
        value["smiles"].add(str(record["canonical_smiles"]))
        value["formulas"].add(str(record["formula"]))
        value["splits"].add(str(record["split"]))
        value["records"] += 1
    unrepresentable = []
    formula_conflicts = []
    for structure_id, value in sorted(grouped.items()):
        if len(value["formulas"]) != 1:
            formula_conflicts.append(structure_id)
        valid = []
        for smiles in sorted(value["smiles"]):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and Chem.MolToInchiKey(mol) == structure_id:
                valid.append(smiles)
        value["valid_smiles"] = valid
        if not valid:
            unrepresentable.append(
                {
                    "structure_id": structure_id,
                    "records": int(value["records"]),
                    "splits": sorted(value["splits"]),
                    "smiles_examples": sorted(value["smiles"])[:3],
                }
            )
    return {
        "structures": len(grouped),
        "records": len(records),
        "exact_roundtrip_structures": len(grouped) - len(unrepresentable),
        "unrepresentable_structures": len(unrepresentable),
        "unrepresentable_records": sum(row["records"] for row in unrepresentable),
        "unrepresentable_test_structures": sum(
            "test" in row["splits"] for row in unrepresentable
        ),
        "formula_conflicts": len(formula_conflicts),
        "formula_conflict_examples": formula_conflicts[:10],
        "unrepresentable_examples": unrepresentable[:10],
        "strict_candidate_identity_eligible": (
            not unrepresentable and not formula_conflicts
        ),
    }


def _structure_catalog(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["inchi_key"]).upper()].append(record)
    catalog: dict[str, dict[str, Any]] = {}
    for structure_id, values in sorted(grouped.items()):
        formulas = {str(record["formula"]) for record in values}
        if len(formulas) != 1:
            raise ValueError(f"formula conflict for structure {structure_id}")
        valid_smiles = []
        for smiles in sorted({str(record["canonical_smiles"]) for record in values}):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and Chem.MolToInchiKey(mol) == structure_id:
                valid_smiles.append(smiles)
        if not valid_smiles:
            raise ValueError(
                "candidate canonical_smiles cannot round-trip its exact "
                f"structure_id: {structure_id}"
            )
        catalog[structure_id] = {
            "structure_id": structure_id,
            "canonical_smiles": valid_smiles[0],
            "formula": next(iter(formulas)),
            # Tautomer-sensitive source depictions can disagree on scaffold and
            # standardized parent. These fields are metadata only here; global
            # scaffold exclusion is applied to every raw identity beforehand.
            "scaffold_key": min(str(record["scaffold_key"]) for record in values),
            "molecule_key": min(str(record["molecule_key"]) for record in values),
        }
    return catalog


def materialize_evaluation_records(
    records: Sequence[Mapping[str, Any]],
    *,
    task: BenchmarkTask,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply one global, pre-inference open-world exclusion rule."""

    test_records = [record for record in records if record["split"] == "test"]
    target_ids = {str(record["inchi_key"]).upper() for record in test_records}
    target_scaffolds = {str(record["scaffold_key"]) for record in test_records}
    scaffold_blocked_ids = {
        str(record["inchi_key"]).upper()
        for record in records
        if str(record["scaffold_key"]) in target_scaffolds
    }

    if task == "closed_library":

        def keep(_: Mapping[str, Any]) -> bool:
            return True

    elif task == "open_exact_removed":

        def keep(row: Mapping[str, Any]) -> bool:
            return str(row["inchi_key"]).upper() not in target_ids

    elif task == "open_scaffold_removed":

        def keep(row: Mapping[str, Any]) -> bool:
            # Remove the complete exact identity if any source depiction maps
            # it to a held-out scaffold. This prevents a tautomeric alternate
            # depiction from re-introducing the same structure.
            return str(row["inchi_key"]).upper() not in scaffold_blocked_ids

    else:  # pragma: no cover - Literal plus CLI choices
        raise ValueError(f"unsupported benchmark task: {task}")

    retained = [dict(row) for row in records if keep(row)]
    removed = [dict(row) for row in records if not keep(row)]
    retained_ids = sorted(str(row["record_id"]) for row in retained)
    removed_ids = sorted(str(row["record_id"]) for row in removed)
    retained_structures = _structure_catalog(retained)

    absent_target_ids = sorted(target_ids.intersection(retained_structures))
    retained_scaffolds = {str(row["scaffold_key"]) for row in retained}
    present_target_scaffolds = sorted(target_scaffolds.intersection(retained_scaffolds))
    if task != "closed_library" and absent_target_ids:
        raise ValueError("open-world evaluation library still contains a test target")
    if task == "open_scaffold_removed" and present_target_scaffolds:
        raise ValueError("scaffold-removed library still contains a test scaffold")

    proof = {
        "baseline_protocol_version": BASELINE_PROTOCOL_VERSION,
        "task": task,
        "exclusion_scope": (
            "none"
            if task == "closed_library"
            else "all_frozen_test_exact_structures"
            if task == "open_exact_removed"
            else "all_frozen_test_bemis_murcko_scaffolds"
        ),
        "records_before": len(records),
        "records_after": len(retained),
        "records_removed": len(removed),
        "structures_before": len(_structure_catalog(records)),
        "structures_after": len(retained_structures),
        "test_target_structures": len(target_ids),
        "test_target_scaffolds": len(target_scaffolds),
        "scaffold_blocked_structures": len(scaffold_blocked_ids),
        "retained_record_ids_sha256": hashlib.sha256(
            "\n".join(retained_ids).encode()
        ).hexdigest(),
        "removed_record_ids_sha256": hashlib.sha256(
            "\n".join(removed_ids).encode()
        ).hexdigest(),
        "target_structure_ids": sorted(target_ids),
        "target_scaffold_keys": sorted(target_scaffolds),
        "target_structures_remaining": absent_target_ids,
        "target_scaffolds_remaining": (
            present_target_scaffolds
            if task == "open_scaffold_removed"
            else "not_required"
        ),
        "verified_before_inference": True,
    }
    return retained, proof


def _test_cases(
    records: Sequence[Mapping[str, Any]],
) -> list[list[dict[str, Any]]]:
    """Return one case per spectrum; cross-condition spectra are never merged.

    A future dual-nucleus case requires a frozen acquisition/pairing identifier.
    Merely sharing a molecule identity is insufficient evidence that two spectra
    belong to one experiment.
    """

    result = [
        [dict(record)]
        for record in sorted(records, key=lambda row: str(row["record_id"]))
        if record["split"] == "test"
    ]
    if not result:
        raise ValueError("manifest contains no frozen test cases")
    return result


def _modality(case_records: Sequence[Mapping[str, Any]]) -> str:
    nuclei = {str(record["nucleus"]) for record in case_records}
    if nuclei == {"1H"}:
        return "1h"
    if nuclei == {"13C"}:
        return "13c"
    if nuclei == {"1H", "13C"}:
        return "1h+13c"
    raise ValueError(f"unsupported test-case nuclei: {sorted(nuclei)}")


def _peaks_by_nucleus(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {"1H": [], "13C": []}
    for record in records:
        nucleus = str(record["nucleus"])
        result[nucleus].extend(dict(peak) for peak in record["peaks"])
    return result


def _spectrum_fingerprint(record: Mapping[str, Any]) -> str:
    """Fingerprint a same-nucleus peak table for conservative deduplication."""

    payload = {
        "nucleus": str(record["nucleus"]),
        "shifts_ppm": sorted(
            round(float(peak["shift"]), 6)
            for peak in record["peaks"]
        ),
    }
    return hashlib.sha256(canonical_json_dumps(payload).encode("utf-8")).hexdigest()


def _heuristic_score(
    query: Mapping[str, list[dict[str, Any]]],
    references: Sequence[Mapping[str, Any]],
    *,
    formula: str,
) -> tuple[float, dict[str, Any]]:
    scored = []
    for reference in references:
        reference_peaks = _peaks_by_nucleus([reference])
        _, breakdown = _feature_vector(
            query["13C"],
            query["1H"],
            {
                "peaks_13c": reference_peaks["13C"],
                "peaks_1h": reference_peaks["1H"],
                "formula": formula,
                "mw": 0.0,
            },
            formula_constraint=formula,
        )
        c13 = breakdown["c13"]
        h1 = breakdown["h1"]
        if query["13C"] and query["1H"]:
            score = 0.5 * float(c13["score"]) + 0.5 * float(h1["score"])
        elif query["13C"]:
            score = float(c13["score"])
        else:
            score = float(h1["score"])
        if int(c13["matched"]) + int(h1["matched"]) == 0:
            score = 0.0
        scored.append(
            (
                score,
                str(reference["record_id"]),
                {
                    "c13": c13,
                    "h1": h1,
                    "selected_reference_record_id": str(reference["record_id"]),
                },
            )
        )
    if not scored:
        _, empty = _feature_vector(
            query["13C"],
            query["1H"],
            {
                "peaks_13c": [],
                "peaks_1h": [],
                "formula": formula,
                "mw": 0.0,
            },
            formula_constraint=formula,
        )
        return 0.0, {
            "c13": empty["c13"],
            "h1": empty["h1"],
            "reference_spectra_considered": 0,
            "selected_reference_record_id": None,
            "aggregation": "best_single_spectrum_no_cross_condition_merge",
        }
    scored.sort(key=lambda item: (-item[0], item[1]))
    score, _, detail = scored[0]
    return score, {
        **detail,
        "reference_spectra_considered": len(references),
        "aggregation": "best_single_spectrum_no_cross_condition_merge",
    }


def build_baseline_cases(
    records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    task: BenchmarkTask,
    baseline_kind: BaselineKind,
    reference_policy: ReferencePolicy,
    registration_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build complete, deterministic cases for one pre-registered run."""

    if baseline_kind not in {"formula_only", "spectral_library"}:
        raise ValueError(f"unsupported baseline kind: {baseline_kind}")
    if reference_policy not in {
        "not_applicable",
        "reference_in_library",
        "leave_source_record_out",
    }:
        raise ValueError(f"unsupported reference policy: {reference_policy}")
    if baseline_kind == "formula_only" and reference_policy != "not_applicable":
        raise ValueError("formula-only baseline cannot use a reference policy")
    if baseline_kind == "spectral_library" and reference_policy == "not_applicable":
        raise ValueError("spectral-library baseline requires a reference policy")
    if task != "closed_library" and reference_policy == "reference_in_library":
        raise ValueError("open-world baselines cannot use test references")

    catalog = _structure_catalog(evaluation_records)
    structures_by_formula: dict[str, list[dict[str, Any]]] = defaultdict(list)
    references_by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for structure in catalog.values():
        structures_by_formula[structure["formula"]].append(structure)
    for record in evaluation_records:
        references_by_structure[str(record["inchi_key"]).upper()].append(dict(record))
    for structures in structures_by_formula.values():
        structures.sort(key=lambda row: row["structure_id"])

    cases: list[dict[str, Any]] = []
    independent_target_references = 0
    self_reference_counts: list[int] = []
    formula_pool_sizes: list[int] = []
    for ordinal, case_records in enumerate(_test_cases(records), start=1):
        target = str(case_records[0]["inchi_key"]).upper()
        formula = str(case_records[0]["formula"])
        scaffold = str(case_records[0]["scaffold_key"])
        record_ids = {str(record["record_id"]) for record in case_records}
        source_record_keys = {
            (
                str(record["source_snapshot_sha256"]),
                int(record["source_record_ordinal"]),
            )
            for record in case_records
        }
        source_record_hashes = {
            str(record["record_sha256"])
            for record in case_records
        }
        query_spectrum_fingerprints = {
            _spectrum_fingerprint(record)
            for record in case_records
        }
        query = _peaks_by_nucleus(case_records)
        query_nuclei = {
            nucleus for nucleus, peaks in query.items() if peaks
        }
        pool = list(structures_by_formula.get(formula, ()))
        formula_pool_sizes.append(len(pool))
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        target_independent = False
        target_self_count = 0

        for structure in pool:
            structure_id = structure["structure_id"]
            all_references = references_by_structure.get(structure_id, [])
            self_references = [
                row
                for row in all_references
                if str(row["record_id"]) in record_ids
                and str(row["nucleus"]) in query_nuclei
            ]
            independent_references = [
                row
                for row in all_references
                if str(row["record_id"]) not in record_ids
                and str(row["nucleus"]) in query_nuclei
                and (
                    str(row["source_snapshot_sha256"]),
                    int(row["source_record_ordinal"]),
                )
                not in source_record_keys
                and str(row["record_sha256"]) not in source_record_hashes
                and _spectrum_fingerprint(row)
                not in query_spectrum_fingerprints
            ]
            if structure_id == target:
                target_self_count = len(self_references)
                target_independent = bool(independent_references)

            if baseline_kind == "formula_only":
                score = 0.0
                detail = {
                    "tie_policy": "all_formula_matches_tied_then_structure_id",
                    "reference_spectra": 0,
                }
            else:
                references = (
                    all_references
                    if reference_policy == "reference_in_library"
                    else independent_references
                )
                score, detail = _heuristic_score(
                    query,
                    references,
                    formula=formula,
                )
            ranked.append(
                (
                    score,
                    structure_id,
                    {
                        "structure_id": structure_id,
                        "canonical_smiles": structure["canonical_smiles"],
                        "formula": structure["formula"],
                        "ranking_score": round(float(score), 12),
                            "metadata": {
                                **detail,
                                "self_reference_spectra": len(self_references),
                                "source_record_independent_same_nucleus_"
                                "reference_spectra": len(independent_references),
                            },
                    },
                )
            )

        ranked.sort(key=lambda item: (-item[0], item[1]))
        candidates = [item[2] for item in ranked]
        independent_target_references += int(target_independent)
        self_reference_counts.append(target_self_count)
        pool_ids = [structure["structure_id"] for structure in pool]
        removed = task != "closed_library"
        scaffold_removed = task == "open_scaffold_removed"
        cases.append(
            {
                "protocol_version": BENCHMARK_PROTOCOL_VERSION,
                "registration_id": registration_id,
                "task": task,
                "case_id": f"{task}:{baseline_kind}:{ordinal:06d}:{target}",
                "record_ids": sorted(record_ids),
                "target_id": target,
                "target_formula": formula,
                "target_removed_from_index": removed,
                "target_scaffold_removed_from_index": scaffold_removed,
                "candidate_pool_structure_ids": pool_ids,
                "candidate_pool_size": len(pool_ids),
                "candidate_pool_contains_target": target in pool_ids,
                "modality": _modality(case_records),
                # A deterministic score is not a calibrated decision rule.
                "abstained": True,
                "selected_structure_id": None,
                "calibrated_probability": 0.0,
                "candidates": candidates,
                "metadata": {
                    "baseline_protocol_version": BASELINE_PROTOCOL_VERSION,
                    "baseline_kind": baseline_kind,
                    "reference_policy": reference_policy,
                    "confidence_policy": "fixed_zero_always_abstain",
                    "target_scaffold_key": scaffold,
                    "target_self_reference_spectra": target_self_count,
                    "source_record_independent_target_reference_available": (
                        target_independent
                    ),
                    "reference_exclusion": (
                        "query_record_id_plus_source_record_identity_plus_"
                        "same_nucleus_peak_fingerprint"
                        if reference_policy == "leave_source_record_out"
                        else "none"
                    ),
                },
            }
        )

    return cases, {
        "baseline_protocol_version": BASELINE_PROTOCOL_VERSION,
        "task": task,
        "baseline_kind": baseline_kind,
        "reference_policy": reference_policy,
        "cases": len(cases),
        "evaluation_records": len(evaluation_records),
        "evaluation_structures": len(catalog),
        "source_record_independent_target_reference_cases": (
            independent_target_references
        ),
        "source_record_independent_target_reference_rate": (
            independent_target_references / len(cases)
        ),
        "target_self_reference_spectra": sum(self_reference_counts),
        "formula_pool_size": {
            "minimum": min(formula_pool_sizes),
            "median": median(formula_pool_sizes),
            "maximum": max(formula_pool_sizes),
            "mean": sum(formula_pool_sizes) / len(formula_pool_sizes),
        },
        "confidence_policy": "fixed_zero_always_abstain",
    }


def formula_constraint_metrics(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Report tie-aware formula-only expectations without inventing a ranker."""

    if not cases:
        raise ValueError("formula constraint metrics require cases")
    expected_top = {1: [], 5: [], 10: []}
    expected_mrr: list[float] = []
    covered = 0
    pool_sizes: list[int] = []
    empty = 0
    for case in cases:
        pool = [str(value).upper() for value in case["candidate_pool_structure_ids"]]
        target = str(case["target_id"]).upper()
        size = len(pool)
        pool_sizes.append(size)
        empty += int(size == 0)
        if target not in pool:
            expected_mrr.append(0.0)
            for values in expected_top.values():
                values.append(0.0)
            continue
        covered += 1
        expected_mrr.append(sum(1.0 / rank for rank in range(1, size + 1)) / size)
        for k, values in expected_top.items():
            values.append(min(k, size) / size)

    return {
        "cases": len(cases),
        "candidate_coverage": round(covered / len(cases), 6),
        "empty_pool_refusal_rate": round(empty / len(cases), 6),
        "formula_violation_rate": 0.0,
        "pool_size": {
            "minimum": min(pool_sizes),
            "median": median(pool_sizes),
            "maximum": max(pool_sizes),
            "mean": round(sum(pool_sizes) / len(pool_sizes), 6),
        },
        "tie_aware_random_expectation": {
            "top1": round(sum(expected_top[1]) / len(cases), 6),
            "top5": round(sum(expected_top[5]) / len(cases), 6),
            "top10": round(sum(expected_top[10]) / len(cases), 6),
            "mrr": round(sum(expected_mrr) / len(cases), 6),
        },
        "interpretation": (
            "Every exact-formula candidate is tied. Expected metrics average "
            "uniform random ordering within each formula pool; deterministic "
            "structure-ID order is reported separately only as a reproducibility "
            "sanity check."
        ),
    }


def spectral_evidence_metrics(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Measure ranks only when the target has a non-zero heuristic score.

    The generic benchmark evaluator correctly reports candidate-list ranks, but
    an exact-formula singleton can be ranked first with a zero spectral score.
    This companion metric prevents that deterministic fallback from being
    mistaken for a successful spectrum match.  Non-zero is not an acceptance
    threshold and must not be described as qualified or calibrated evidence.
    """

    if not cases:
        raise ValueError("spectral evidence metrics require cases")
    ranks: list[int | None] = []
    any_positive: list[bool] = []
    target_positive: list[bool] = []
    by_modality: dict[str, list[int | None]] = defaultdict(list)
    for case in cases:
        target = str(case["target_id"]).upper()
        candidates = list(case["candidates"])
        positive = [
            candidate
            for candidate in candidates
            if float(candidate["ranking_score"]) > 0.0
        ]
        positive_ids = [
            str(candidate["structure_id"]).upper() for candidate in positive
        ]
        rank = positive_ids.index(target) + 1 if target in positive_ids else None
        ranks.append(rank)
        any_positive.append(bool(positive))
        target_positive.append(rank is not None)
        by_modality[str(case["modality"])].append(rank)

    reciprocal = [0.0 if rank is None else 1.0 / rank for rank in ranks]

    def top_k(values: Sequence[int | None], k: int) -> float:
        return sum(rank is not None and rank <= k for rank in values) / len(values)

    return {
        "cases": len(cases),
        "any_candidate_nonzero_score_case_rate": round(
            sum(any_positive) / len(cases),
            6,
        ),
        "target_nonzero_score_rate": round(
            sum(target_positive) / len(cases),
            6,
        ),
        "nonzero_score_top1": round(top_k(ranks, 1), 6),
        "nonzero_score_top5": round(top_k(ranks, 5), 6),
        "nonzero_score_top10": round(top_k(ranks, 10), 6),
        "nonzero_score_mrr": round(sum(reciprocal) / len(cases), 6),
        "by_modality": {
            modality: {
                "cases": len(values),
                "nonzero_score_top1": round(top_k(values, 1), 6),
                "nonzero_score_top5": round(top_k(values, 5), 6),
            }
            for modality, values in sorted(by_modality.items())
        },
        "zero_score_policy": (
            "A target with ranking_score <= 0 is counted as not retrieved, even "
            "when the exact-formula pool makes it first by deterministic tie-break."
        ),
        "interpretation": (
            "A positive heuristic score only proves that at least one peak was "
            "matched. It is not a pre-registered acceptance threshold, evidence "
            "grade or calibrated probability."
        ),
    }


def summarize_dp5q_shadow(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate target-only DP5q evidence without treating it as a probability."""

    completed = [row for row in rows if row.get("status") == "ok"]
    maes = [float(row["mae_ppm"]) for row in completed]
    rmses = [float(row["rmse_ppm"]) for row in completed]
    coverages = [float(row["bidirectional_coverage"]) for row in completed]
    return {
        "baseline_protocol_version": BASELINE_PROTOCOL_VERSION,
        "diagnostic_only": True,
        "calibrated_probability": False,
        "quantile_enabled": False,
        "assignment_mode": "unassigned_hungarian_atom_level",
        "cases_attempted": len(rows),
        "cases_completed": len(completed),
        "cases_failed_or_unsupported": len(rows) - len(completed),
        "mae_ppm_macro_mean": (
            round(sum(maes) / len(maes), 6) if maes else None
        ),
        "mae_ppm_median": round(median(maes), 6) if maes else None,
        "rmse_ppm_macro_mean": (
            round(sum(rmses) / len(rmses), 6) if rmses else None
        ),
        "bidirectional_coverage_macro_mean": (
            round(sum(coverages) / len(coverages), 6) if coverages else None
        ),
        "assignment_complete_rate": (
            round(
                sum(bool(row["assignment_complete"]) for row in completed)
                / len(completed),
                6,
            )
            if completed
            else None
        ),
        "limitation": (
            "This target-only shadow diagnostic uses unassigned Hungarian "
            "matching and does not validate atom correspondence, symmetry "
            "collapse, candidate ranking or calibrated structural probability."
        ),
    }
