"""Fail-closed admission and calibration for NMR ranking confidence.

This module deliberately does not train the structure ranker.  It calibrates a
single, frozen ranker score to the probability that the top-ranked exact
structure is correct.  Admission is evaluated before ``fit_calibrator`` is
called, and deployment eligibility is evaluated on a source-, molecule- and
scaffold-isolated frozen test split.

Artifacts are JSON rather than pickle/joblib so that they can be inspected
without executing code.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from app.ml.independent_nmr_data import (
    DERIVED_RELEASE_SCHEMA_VERSION,
    DERIVED_SCHEMA_VERSION,
    PARENT_STANDARDIZATION_VERSION,
    RECORD_ID as NMREXP_RECORD_ID,
    SOURCE_CREATORS,
    SOURCE_DOI,
    SOURCE_LICENSE,
)

try:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception:  # pragma: no cover - dependency failure is reported by the gate
    Chem = None
    MurckoScaffold = None


GOLD_MANIFEST_SCHEMA_VERSION = "chemapp.nmr.gold-spectrum-manifest.v1"
CALIBRATION_EXAMPLES_SCHEMA_VERSION = "chemapp.nmr.calibration-examples.v2"
FROZEN_SCORE_CASE_SCHEMA_VERSION = "chemapp.nmr.frozen-score-case.v3"
CALIBRATOR_SCHEMA_VERSION = "chemapp.nmr.correctness-calibrator.v1"
MODEL_CARD_SCHEMA_VERSION = "chemapp.nmr.calibration-model-card.v1"
TRAINING_PIPELINE_VERSION = "nmr-calibration-training-v3"
REVIEW_PROTOCOL_VERSION = "nmr-gold-spectrum-review-v1"
NMREXP_DERIVED_SCHEMA_VERSION = DERIVED_SCHEMA_VERSION

SPLITS = ("train", "validation", "calibration", "test")
CALIBRATOR_SPLITS = ("calibration", "test")
SUPPORTED_NUCLEI = ("1H", "13C")
PRE_SCORE_ELIGIBILITY_SCHEMA_VERSION = "chemapp.nmr.pre-score-eligibility.v3"
PRE_SCORE_CRITERION_VERSION = (
    "strict-independent-parent-dp5q-13c-formula-decoys-conformer-preflight-v3"
)
FROZEN_RANKER_SCHEMA_VERSION = "chemapp.nmr.frozen-ranker-artifact.v3"
RUN_SPEC_SCHEMA_VERSION = "chemapp.nmr.calibration-run-spec.v1"
TEST_CONSUMPTION_SCHEMA_VERSION = "chemapp.nmr.frozen-test-consumption.v1"
RUN_SPEC_VERSION = 1
TEST_CONSUMPTION_LEDGER_VERSION = 1
CONFORMER_PROTOCOL_VERSION = "chemapp.dp5q-conformer.v2"
CONFORMER_PREFLIGHT_PROTOCOL_VERSION = "chemapp.dp5q-conformer-preflight.v1"
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_RDKIT_VERSION_RE = re.compile(r"^\d{4}\.\d{2}\.\d+$")
_PRE_SCORE_REASON_RE = re.compile(
    r"^(?:nucleus_not_13c_dp5q_mean|"
    r"no_same_formula_different_connectivity_decoy|"
    r"no_conformer_preflight_eligible_same_formula_decoy|"
    r"truth_(?:structure_parse_failed|multiple_fragments|"
    r"heavy_atom_limit_exceeded|total_atom_limit_exceeded|"
    r"rotatable_bond_limit_exceeded|no_carbon_atoms|"
    r"formula_is_not_supported|formula_mismatch|"
    r"inchi_key_generation_failed|unsupported_atomic_number_\d+|"
    r"conformer_preflight_(?:invalid_smiles|multiple_fragments|"
    r"molecule_too_large|molecule_too_flexible|unsupported_element|"
    r"no_carbon|mmff_parameters_unavailable|"
    r"conformer_generation_failed|mmff_optimisation_failed|"
    r"conformer_filter_failed)))$"
)
_SOURCE_DOCUMENT_RE = re.compile(
    r"^(?:doi:10\.\d{4,9}/\S+|pmid:\d+|accession:\S+|"
    r"document-sha256:[0-9a-f]{64})$"
)
_PRE_SCORE_ALLOWED_FIELDS = frozenset(
    {
        "schema_version",
        "record_id",
        "eligible",
        "evaluated_before_scoring",
        "criterion_version",
        "reason_codes",
        "domain_config_sha256",
        "base_index_sha256",
        "nucleus",
        "candidate_pool_size_pre_score",
        "usable_decoy_count_pre_cap",
        "candidate_count_preflight_input",
        "candidate_count_preflight_passed",
        "conformer_preflight_sidecar_sha256",
        "conformer_preflight_rdkit_version",
        "conformer_protocol_version",
        "conformer_generation_config_sha256",
        "conformer_preflight_protocol_version",
        "conformer_preflight_policy_sha256",
        "conformer_preflight_input_candidate_set_sha256",
        "conformer_preflight_passed_candidate_set_sha256",
        "conformer_preflight_decisions_sha256",
        "conformer_preflight_runtime_sha256",
        "conformer_preflight_manifest_sha256",
        "source_document_sha256",
        "molecule_sha256",
        "scaffold_sha256",
        "source_content_sha256",
        "source_release_id",
        "source_records_sha256",
        "source_current_sha256",
    }
)
_CONFORMER_PREFLIGHT_HASH_FIELDS = (
    "conformer_preflight_sidecar_sha256",
    "conformer_generation_config_sha256",
    "conformer_preflight_policy_sha256",
    "conformer_preflight_input_candidate_set_sha256",
    "conformer_preflight_passed_candidate_set_sha256",
    "conformer_preflight_decisions_sha256",
    "conformer_preflight_runtime_sha256",
    "conformer_preflight_manifest_sha256",
)
_CONFORMER_PREFLIGHT_GLOBAL_FIELDS = (
    "conformer_preflight_sidecar_sha256",
    "conformer_preflight_rdkit_version",
    "conformer_protocol_version",
    "conformer_generation_config_sha256",
    "conformer_preflight_protocol_version",
    "conformer_preflight_policy_sha256",
    "conformer_preflight_runtime_sha256",
    "conformer_preflight_manifest_sha256",
)


class CalibrationAdmissionError(ValueError):
    """Raised before fitting when the gold data or score cases fail admission."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        reasons = self.report.get("blocking_reasons") or []
        codes = ", ".join(str(item.get("code")) for item in reasons[:5])
        super().__init__(f"NMR calibration admission blocked: {codes or 'unknown'}")


def validate_calibration_run_spec(
    spec: Mapping[str, Any],
    *,
    expected_input_sha256: Mapping[str, str | None],
    expected_parameters: Mapping[str, Any],
    expected_bindings: Mapping[str, str],
    run_created_at: str,
) -> dict[str, Any]:
    """Validate a pre-existing v1 preregistration; never create one here."""

    allowed_top_level = {
        "schema_version",
        "spec_version",
        "run_id",
        "registered_at",
        "pipeline_version",
        "purpose",
        "parameters",
        "input_sha256",
        "expected_bindings",
        "frozen_test",
    }
    failures: list[str] = []
    if set(spec) != allowed_top_level:
        failures.append("top_level_field_allowlist_mismatch")
    if spec.get("schema_version") != RUN_SPEC_SCHEMA_VERSION:
        failures.append("schema_version")
    if spec.get("spec_version") != RUN_SPEC_VERSION:
        failures.append("spec_version")
    if not _nonempty(spec.get("run_id")):
        failures.append("run_id")
    if spec.get("pipeline_version") != TRAINING_PIPELINE_VERSION:
        failures.append("pipeline_version")
    if spec.get("purpose") != "preregistered_frozen_test_calibration":
        failures.append("purpose")
    registered_at = _parse_utc(spec.get("registered_at"))
    created_at = _parse_utc(run_created_at)
    if registered_at is None or created_at is None or registered_at >= created_at:
        failures.append("registration_must_precede_run")
    parameters = spec.get("parameters")
    if not isinstance(parameters, Mapping) or dict(parameters) != dict(
        expected_parameters
    ):
        failures.append("parameters_mismatch")
    input_sha256 = spec.get("input_sha256")
    if not isinstance(input_sha256, Mapping) or dict(input_sha256) != dict(
        expected_input_sha256
    ):
        failures.append("input_sha256_mismatch")
    elif any(
        value is not None and not _is_sha256(value) for value in input_sha256.values()
    ):
        failures.append("input_sha256_invalid")
    bindings = spec.get("expected_bindings")
    if not isinstance(bindings, Mapping) or dict(bindings) != dict(expected_bindings):
        failures.append("expected_bindings_mismatch")
    elif any(not _is_sha256(value) for value in bindings.values()):
        failures.append("expected_bindings_invalid")
    frozen_test = spec.get("frozen_test")
    if not isinstance(frozen_test, Mapping) or dict(frozen_test) != {
        "consumption_policy": "single_use",
        "read_outcomes_only_after_ledger_reservation": True,
        "reuse_for_model_or_threshold_selection": False,
    }:
        failures.append("frozen_test_contract")
    if failures:
        raise ValueError(
            "invalid calibration run-spec: "
            + canonical_json_dumps(sorted(set(failures)))
        )
    return {
        "schema_version": "chemapp.nmr.calibration-run-spec-audit.v1",
        "status": "passed",
        "run_id": str(spec["run_id"]),
        "run_spec_content_sha256": canonical_sha256(spec),
        "registered_at": str(spec["registered_at"]),
        "outcomes_read": False,
    }


def reserve_frozen_test_consumption(
    ledger_path: str | Path,
    *,
    run_id: str,
    run_spec_file_sha256: str,
    frozen_score_cases_file_sha256: str,
    frozen_test_binding_sha256: str,
    reserved_at: str,
) -> dict[str, Any]:
    """Atomically reserve one immutable frozen test before outcomes are parsed."""

    for value, field_name in (
        (run_spec_file_sha256, "run_spec_file_sha256"),
        (frozen_score_cases_file_sha256, "frozen_score_cases_file_sha256"),
        (frozen_test_binding_sha256, "frozen_test_binding_sha256"),
    ):
        if not _is_sha256(value):
            raise ValueError(f"{field_name} must be lowercase SHA-256")
    if not _nonempty(run_id) or _parse_utc(reserved_at) is None:
        raise ValueError("ledger reservation requires run_id and UTC reserved_at")
    path = Path(ledger_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(f"{path.name}.lock")
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ValueError(
            f"frozen-test ledger is locked; inspect stale lock {lock}"
        ) from exc
    temporary: Path | None = None
    try:
        entries: list[dict[str, Any]] = []
        if path.exists():
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{line_number}: corrupt ledger JSON"
                    ) from exc
                if (
                    not isinstance(entry, dict)
                    or entry.get("schema_version") != TEST_CONSUMPTION_SCHEMA_VERSION
                    or entry.get("ledger_version") != TEST_CONSUMPTION_LEDGER_VERSION
                ):
                    raise ValueError(f"{path}:{line_number}: ledger schema changed")
                entries.append(entry)
        if any(
            entry.get("frozen_test_binding_sha256") == frozen_test_binding_sha256
            or entry.get("frozen_score_cases_file_sha256")
            == frozen_score_cases_file_sha256
            for entry in entries
        ):
            raise ValueError(
                "frozen test was already reserved/consumed and cannot be reused"
            )
        entry = {
            "schema_version": TEST_CONSUMPTION_SCHEMA_VERSION,
            "ledger_version": TEST_CONSUMPTION_LEDGER_VERSION,
            "event": "reserved_before_test_outcome_read",
            "run_id": run_id,
            "run_spec_file_sha256": run_spec_file_sha256,
            "frozen_score_cases_file_sha256": (frozen_score_cases_file_sha256),
            "frozen_test_binding_sha256": frozen_test_binding_sha256,
            "reserved_at": reserved_at,
        }
        entries.append(entry)
        rendered = "".join(canonical_json_dumps(value) + "\n" for value in entries)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".part",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        return entry
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        lock.rmdir()


@dataclass(frozen=True)
class CalibrationGatePolicy:
    """Production-oriented defaults.

    Tests and explicitly labelled research runs may provide a stricter or
    smaller policy, but a model card records the complete effective policy.
    Changing a policy therefore changes the run hash.
    """

    schema_version: str = "chemapp.nmr.calibration-gate-policy.v1"
    minimum_records: dict[str, int] = field(
        default_factory=lambda: {
            "train": 0,
            "validation": 0,
            "calibration": 100,
            "test": 100,
        }
    )
    minimum_molecules: dict[str, int] = field(
        default_factory=lambda: {
            "train": 0,
            "validation": 0,
            "calibration": 100,
            "test": 100,
        }
    )
    minimum_scaffolds: dict[str, int] = field(
        default_factory=lambda: {
            "train": 0,
            "validation": 0,
            "calibration": 60,
            "test": 60,
        }
    )
    minimum_source_documents: dict[str, int] = field(
        default_factory=lambda: {
            "train": 0,
            "validation": 0,
            "calibration": 50,
            "test": 50,
        }
    )
    minimum_records_per_nucleus: dict[str, int] = field(
        default_factory=lambda: {
            "train": 0,
            "validation": 0,
            "calibration": 40,
            "test": 40,
        }
    )
    minimum_paired_molecules: dict[str, int] = field(
        default_factory=lambda: {
            "train": 0,
            "validation": 0,
            "calibration": 0,
            "test": 0,
        }
    )
    required_human_reviews: dict[str, int] = field(
        default_factory=lambda: {
            "train": 1,
            "validation": 1,
            "calibration": 1,
            "test": 1,
        }
    )
    minimum_calibration_cases: int = 100
    minimum_test_cases: int = 100
    minimum_outcomes_per_class: int = 25
    minimum_unique_scores: int = 20
    reliability_bins: int = 10
    maximum_test_ece: float = 0.05
    maximum_test_brier: float = 0.20
    minimum_test_brier_skill: float = 0.0
    minimum_test_roc_auc: float = 0.70
    minimum_populated_reliability_bins: int = 5
    maximum_absolute_calibration_intercept: float = 0.25
    minimum_calibration_slope: float = 0.75
    maximum_calibration_slope: float = 1.25


def canonical_json_dumps(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def canonical_jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(canonical_json_dumps(dict(row)).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: row must be a JSON object")
            rows.append(value)
    return rows


def _nmrexp_review_is_eligible(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if row.get("schema_version") != NMREXP_DERIVED_SCHEMA_VERSION:
        failures.append("schema_version")
    calibration = row.get("calibration")
    review = row.get("human_review")
    spectrum = row.get("spectrum")
    structure = row.get("structure")
    source = row.get("source")
    if not isinstance(calibration, Mapping):
        failures.append("calibration_missing")
        calibration = {}
    if calibration.get("strict_independent_eligible") is not True:
        failures.append("not_strict_independent")
    if calibration.get("base_index_connectivity_overlap") is not False:
        failures.append("base_connectivity_overlap_or_unchecked")
    if calibration.get("base_index_exact_inchi_key_overlap") is not False:
        failures.append("base_exact_overlap_or_unchecked")
    if calibration.get("base_index_parent_connectivity_overlap") is not False:
        failures.append("base_parent_connectivity_overlap_or_unchecked")
    if calibration.get("base_index_exact_parent_inchi_key_overlap") is not False:
        failures.append("base_exact_parent_overlap_or_unchecked")
    if calibration.get("base_index_overlap_status") != "checked":
        failures.append("base_overlap_not_checked")
    if calibration.get("parent_eligible") is not True:
        failures.append("calibration_parent_not_eligible")
    if (
        calibration.get("parent_standardization_version")
        != PARENT_STANDARDIZATION_VERSION
    ):
        failures.append("calibration_parent_standardization_changed")
    if calibration.get("model_nucleus_supported") is not True:
        failures.append("unsupported_nucleus")
    if not isinstance(review, Mapping):
        failures.append("human_review_missing")
        review = {}
    expected_review = {
        "critical_spectrum_fields_all_right": True,
        "frequency_extraction": "right",
        "processed_peak_extraction": "right",
        "solvent_extraction": "right",
    }
    for field_name, expected in expected_review.items():
        if review.get(field_name) != expected:
            failures.append(f"human_review_{field_name}")
    semantics = str(review.get("semantics") or "")
    if "not chemical class labels or model probabilities" not in semantics:
        failures.append("review_semantics_missing")
    if not isinstance(spectrum, Mapping):
        failures.append("spectrum_missing")
        spectrum = {}
    if spectrum.get("nucleus") not in SUPPORTED_NUCLEI:
        failures.append("unsupported_nucleus")
    peaks = spectrum.get("processed_peaks")
    if (
        not isinstance(peaks, list)
        or not peaks
        or any(
            not isinstance(peak, Mapping)
            or isinstance(peak.get("shift_ppm"), bool)
            or not isinstance(peak.get("shift_ppm"), (int, float))
            or not math.isfinite(float(peak["shift_ppm"]))
            for peak in peaks
        )
    ):
        failures.append("processed_peaks_invalid")
    if not isinstance(structure, Mapping):
        failures.append("structure_missing")
        structure = {}
    if not _nonempty(structure.get("training_smiles")):
        failures.append("training_smiles_missing")
    if structure.get("ground_truth_field") != "smiles_actual":
        failures.append("ground_truth_not_smiles_actual")
    if structure.get("status") != "human_corrected_standardized_parent":
        failures.append("structure_not_human_corrected")
    if structure.get("parent_eligible") is not True:
        failures.append("structure_parent_not_eligible")
    if structure.get("parent_status") != "eligible_single_organic_parent":
        failures.append("structure_parent_status_invalid")
    if (
        structure.get("parent_standardization_version")
        != PARENT_STANDARDIZATION_VERSION
    ):
        failures.append("structure_parent_standardization_changed")
    if not isinstance(source, Mapping):
        failures.append("source_missing")
        source = {}
    if not _nonempty(source.get("document_doi")):
        failures.append("source_document_doi_missing")
    if source.get("zenodo_record_id") != NMREXP_RECORD_ID:
        failures.append("source_record_changed")
    if source.get("zenodo_doi") != SOURCE_DOI:
        failures.append("source_release_changed")
    if source.get("license_spdx") != SOURCE_LICENSE:
        failures.append("license_not_cc_by_4_0")
    if source.get("attribution_manifest_required") is not True:
        failures.append("attribution_manifest_not_required")
    if not _is_sha256(row.get("source_content_sha256")):
        failures.append("source_content_hash_invalid")
    return not failures, sorted(set(failures))


def _nmrexp_spectrum_fingerprint(spectrum: Mapping[str, Any]) -> str:
    """Hash normalized numeric spectrum content, independent of CSV aliases."""

    peaks = []
    for peak in spectrum.get("processed_peaks") or []:
        peaks.append(
            {
                "shift_ppm": float(peak["shift_ppm"]),
                "range_ppm": [float(value) for value in peak.get("range_ppm") or []],
                "multiplicity": peak.get("multiplicity"),
                "couplings": peak.get("couplings"),
                "reported_integral": peak.get("reported_integral"),
            }
        )
    return canonical_sha256(
        {
            "nucleus": spectrum.get("nucleus"),
            "frequency_mhz": spectrum.get("frequency_mhz"),
            "solvent": spectrum.get("solvent"),
            "representation": spectrum.get("representation"),
            "peaks": peaks,
        }
    )


def _validate_nmrexp_source_summary(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str]:
    """Validate the immutable v2 release binding before adapting any rows."""

    if not isinstance(summary, Mapping):
        raise ValueError("NMRexp v2 adaptation requires its immutable release summary")
    value = dict(summary)
    source = value.get("source")
    publication = value.get("publication")
    if value.get("schema_version") != DERIVED_SCHEMA_VERSION:
        raise ValueError("NMRexp source summary schema changed")
    if not isinstance(source, Mapping) or (
        source.get("record_id") != NMREXP_RECORD_ID
        or source.get("doi") != SOURCE_DOI
        or source.get("license_spdx") != SOURCE_LICENSE
        or source.get("creators") != list(SOURCE_CREATORS)
        or source.get("version_resolution") != "specific_record_only"
        or source.get("attribution_manifest") != "ATTRIBUTION.json"
    ):
        raise ValueError("NMRexp source attribution or frozen release changed")
    if (
        not isinstance(publication, Mapping)
        or publication.get("schema_version") != DERIVED_RELEASE_SCHEMA_VERSION
    ):
        raise ValueError("NMRexp publication binding is missing or changed")
    records_binding = publication.get("records")
    attribution_binding = publication.get("attribution")
    overlap = value.get("overlap")
    expected_records_sha256 = canonical_jsonl_sha256(rows)
    if (
        not isinstance(records_binding, Mapping)
        or records_binding.get("file") != "records.jsonl"
        or records_binding.get("sha256") != expected_records_sha256
        or records_binding.get("count") != len(rows)
    ):
        raise ValueError(
            "NMRexp publication records hash/count does not bind the input rows"
        )
    if (
        not isinstance(attribution_binding, Mapping)
        or attribution_binding.get("file") != "ATTRIBUTION.json"
        or not _is_sha256(attribution_binding.get("sha256"))
        or attribution_binding.get("required_for_redistribution") is not True
    ):
        raise ValueError("NMRexp attribution artifact binding is invalid")
    if not _is_sha256(publication.get("release_id")) or not _is_sha256(
        publication.get("summary_core_sha256")
    ):
        raise ValueError("NMRexp immutable release identity is invalid")
    if (
        not isinstance(overlap, Mapping)
        or overlap.get("status") != "checked"
        or not _is_sha256(overlap.get("base_index_sha256"))
        or overlap.get("parent_standardization_version")
        != PARENT_STANDARDIZATION_VERSION
    ):
        raise ValueError("NMRexp parent-level base-index overlap proof changed")
    return value, expected_records_sha256


def _connected_gold_components(
    records: Sequence[Mapping[str, Any]],
) -> list[list[int]]:
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
    fields = (
        "molecule_key",
        "inchi_key",
        "scaffold_key",
        "source_document_key",
        "record_sha256",
        "spectrum_fingerprint_sha256",
    )
    for index, row in enumerate(records):
        for field_name in fields:
            value = str(row.get(field_name) or "")
            if not value:
                continue
            key = (field_name, value)
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index
    components: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        components[find(index)].append(index)
    return list(components.values())


def _assign_calibration_test_splits(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    calibration_fraction: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be strictly between 0 and 1")
    components = _connected_gold_components(records)
    total = len(records)
    totals_by_nucleus = Counter(str(row["nucleus"]) for row in records)
    targets = {
        "calibration": {
            "total": total * calibration_fraction,
            **{
                nucleus: totals_by_nucleus[nucleus] * calibration_fraction
                for nucleus in SUPPORTED_NUCLEI
            },
        },
        "test": {
            "total": total * (1.0 - calibration_fraction),
            **{
                nucleus: totals_by_nucleus[nucleus] * (1.0 - calibration_fraction)
                for nucleus in SUPPORTED_NUCLEI
            },
        },
    }
    assigned_counts = {
        split: {"total": 0, **{nucleus: 0 for nucleus in SUPPORTED_NUCLEI}}
        for split in CALIBRATOR_SPLITS
    }

    def component_digest(component: Sequence[int]) -> str:
        ids = sorted(str(records[index]["record_id"]) for index in component)
        return hashlib.sha256(f"{seed}\0{chr(0).join(ids)}".encode()).hexdigest()

    ordered = sorted(
        components,
        key=lambda component: (-len(component), component_digest(component)),
    )
    assignments: dict[int, tuple[str, str]] = {}
    for component in ordered:
        component_counts = Counter(
            str(records[index]["nucleus"]) for index in component
        )
        component_id = component_digest(component)
        scored_splits = []
        for selected_candidate in CALIBRATOR_SPLITS:
            cost = 0.0
            for split in CALIBRATOR_SPLITS:
                selected = split == selected_candidate
                projected_total = assigned_counts[split]["total"] + (
                    len(component) if selected else 0
                )
                total_target = max(targets[split]["total"], 1.0)
                cost += ((projected_total - total_target) / total_target) ** 2
                for nucleus in SUPPORTED_NUCLEI:
                    projected = assigned_counts[split][nucleus] + (
                        component_counts.get(nucleus, 0) if selected else 0
                    )
                    target = max(targets[split][nucleus], 1.0)
                    cost += ((projected - target) / target) ** 2
            tie_break = hashlib.sha256(
                f"{seed}\0{component_id}\0{selected_candidate}".encode()
            ).hexdigest()
            scored_splits.append((cost, tie_break, selected_candidate))
        selected_split = min(scored_splits)[2]
        assigned_counts[selected_split]["total"] += len(component)
        for nucleus in SUPPORTED_NUCLEI:
            assigned_counts[selected_split][nucleus] += component_counts.get(
                nucleus,
                0,
            )
        for index in component:
            assignments[index] = (selected_split, component_id)

    assigned = [
        {
            **dict(row),
            "split": assignments[index][0],
            "split_group": assignments[index][1],
            "split_group_fields": [
                "source_document_key",
                "molecule_key",
                "inchi_key",
                "scaffold_key",
                "spectrum_fingerprint_sha256",
            ],
            "split_seed": seed,
        }
        for index, row in enumerate(records)
    ]
    return assigned, {
        "strategy": "connected_source_document_structure_scaffold_v1",
        "seed": seed,
        "calibration_fraction": calibration_fraction,
        "components": len(components),
        "largest_component": max((len(value) for value in components), default=0),
        "counts": assigned_counts,
    }


def adapt_nmrexp_reviewed_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_summary: Mapping[str, Any] | None = None,
    seed: int = 20260726,
    calibration_fraction: float = 0.5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert the pinned NMRexp checked release to the gold manifest contract.

    Only ``structure.training_smiles`` (derived from ``smiles_actual``) is used
    as structure truth.  Extraction right/wrong labels are used solely as
    review evidence and are never emitted as model correctness outcomes.
    """

    summary, source_records_sha256 = _validate_nmrexp_source_summary(
        rows,
        source_summary,
    )
    overlap_summary = summary.get("overlap")
    reference_index_sha256 = (
        overlap_summary.get("base_index_sha256")
        if isinstance(overlap_summary, Mapping)
        else None
    )
    summary_sha256 = canonical_sha256(summary) if summary else None
    normalized: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    rejected_examples: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        eligible, failures = _nmrexp_review_is_eligible(row)
        if not eligible:
            for failure in failures:
                rejection_counts[failure] += 1
            if len(rejected_examples) < 10:
                rejected_examples.append(
                    {
                        "row": row_index + 1,
                        "record_id": row.get("record_id"),
                        "reasons": failures,
                    }
                )
            continue
        source = row["source"]
        spectrum = row["spectrum"]
        structure = row["structure"]
        upstream_evidence_sha256 = canonical_sha256(
            {
                "schema_version": row["schema_version"],
                "record_id": row["record_id"],
                "source_content_sha256": row["source_content_sha256"],
                "source": source,
                "human_review": row["human_review"],
                "calibration": row["calibration"],
                "structure_truth": {
                    "ground_truth_field": structure["ground_truth_field"],
                    "training_smiles": structure["training_smiles"],
                    "inchi_key": structure["inchi_key"],
                    "molecule_key": structure["molecule_key"],
                    "scaffold_key": structure["scaffold_key"],
                },
                "spectrum": spectrum,
            }
        )
        record: dict[str, Any] = {
            "schema_version": GOLD_MANIFEST_SCHEMA_VERSION,
            "record_id": str(row["record_id"]),
            "record_sha256": str(row["source_content_sha256"]),
            "spectrum_fingerprint_sha256": _nmrexp_spectrum_fingerprint(spectrum),
            "molecule_key": str(structure["molecule_key"]),
            "inchi_key": str(structure["inchi_key"]),
            "canonical_smiles": str(structure["training_smiles"]),
            "scaffold_key": str(structure["scaffold_key"] or "acyclic"),
            "source_document_key": (
                f"doi:{str(source['document_doi']).strip().casefold()}"
            ),
            "dataset_release_id": (
                f"doi:{str(source['zenodo_doi']).strip().casefold()}"
            ),
            "source_name": "nmrexp",
            "nucleus": str(spectrum["nucleus"]),
            "measurement_kind": "measured",
            "base_index_overlap": {
                "status": row["calibration"]["base_index_overlap_status"],
                "connectivity": row["calibration"]["base_index_connectivity_overlap"],
                "exact_inchi_key": row["calibration"][
                    "base_index_exact_inchi_key_overlap"
                ],
                "parent_connectivity": row["calibration"][
                    "base_index_parent_connectivity_overlap"
                ],
                "exact_parent_inchi_key": row["calibration"][
                    "base_index_exact_parent_inchi_key_overlap"
                ],
                "parent_standardization_version": (PARENT_STANDARDIZATION_VERSION),
            },
            "ranker_training_overlap": {
                "status": row["calibration"]["base_index_overlap_status"],
                "reference_index_sha256": reference_index_sha256,
                "reference_scope": "chemapp_frozen_nmrshiftdb2_base_index",
                "connectivity": row["calibration"]["base_index_connectivity_overlap"],
                "exact_inchi_key": row["calibration"][
                    "base_index_exact_inchi_key_overlap"
                ],
                "parent_connectivity": row["calibration"][
                    "base_index_parent_connectivity_overlap"
                ],
                "exact_parent_inchi_key": row["calibration"][
                    "base_index_exact_parent_inchi_key_overlap"
                ],
                "upstream_dp5q_training_overlap": "unknown",
            },
            "source_aliases": row.get("source_aliases") or [],
            "spectrum": spectrum,
            "upstream_review_evidence_sha256": upstream_evidence_sha256,
            "source_summary_sha256": summary_sha256,
        }
        review = {
            "reviewer_id": "upstream-curation:nmrexp-checked-release",
            "reviewed_at": None,
            "protocol_version": REVIEW_PROTOCOL_VERSION,
            "decision": "accepted",
            "review_kind": "upstream_human_checked_release",
            "structure_verified": True,
            "spectrum_verified": True,
            "nucleus_verified": True,
            "source_document_verified": True,
            "duplicate_check_completed": True,
            "upstream_review_evidence_sha256": upstream_evidence_sha256,
        }
        review["review_record_sha256"] = manual_review_binding_sha256(
            record,
            review,
        )
        record["manual_reviews"] = [review]
        normalized.append(record)

    assigned, split_audit = _assign_calibration_test_splits(
        normalized,
        seed=seed,
        calibration_fraction=calibration_fraction,
    )
    return assigned, {
        "schema_version": "chemapp.nmr.nmrexp-adaptation-audit.v1",
        "input_rows": len(rows),
        "strict_gold_rows": len(assigned),
        "rejected_rows": len(rows) - len(assigned),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "rejected_examples": rejected_examples,
        "structure_truth_field": "structure.training_smiles_from_smiles_actual",
        "extraction_review_labels_used_as_model_outcomes": False,
        "source_summary_sha256": summary_sha256,
        "source_records_sha256": source_records_sha256,
        "source_release_id": summary["publication"]["release_id"],
        "source_attribution_sha256": summary["publication"]["attribution"]["sha256"],
        "reference_index_sha256": reference_index_sha256,
        "split": split_audit,
        "output_content_sha256": canonical_jsonl_sha256(assigned),
    }


def select_scoreable_gold_subset(
    manifest: Sequence[Mapping[str, Any]],
    eligibility_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a pre-score applicability subset without reading model outcomes.

    The eligibility audit must cover the complete input pool and conform to a
    fixed, versioned field allowlist.  This prevents both obvious outcome fields
    and covert post-outcome selectors from entering the applicability contract.
    """

    manifest_lookup = _manifest_lookup(manifest)
    expected_ids = set(manifest_lookup)
    seen: set[str] = set()
    eligible_ids: set[str] = set()
    reason_counts: Counter[str] = Counter()
    criteria: Counter[str] = Counter()
    domain_hashes: Counter[str] = Counter()
    base_index_hashes: Counter[str] = Counter()
    source_release_ids: Counter[str] = Counter()
    source_records_hashes: Counter[str] = Counter()
    source_current_hashes: Counter[str] = Counter()
    preflight_global_values: dict[str, Counter[str]] = {
        field_name: Counter() for field_name in _CONFORMER_PREFLIGHT_GLOBAL_FIELDS
    }
    invalid: list[dict[str, Any]] = []
    for index, row in enumerate(eligibility_rows):
        failures: list[str] = []
        record_id = str(row.get("record_id") or "")
        row_fields = {str(key) for key in row}
        if row_fields != _PRE_SCORE_ALLOWED_FIELDS:
            failures.append(
                "field_allowlist_mismatch:"
                + ",".join(sorted(row_fields ^ _PRE_SCORE_ALLOWED_FIELDS))
            )
        if row.get("schema_version") != PRE_SCORE_ELIGIBILITY_SCHEMA_VERSION:
            failures.append("schema_version")
        if not record_id or record_id not in manifest_lookup:
            failures.append("record_id_not_in_pool")
        elif record_id in seen:
            failures.append("duplicate_record_id")
        else:
            seen.add(record_id)
        if type(row.get("eligible")) is not bool:
            failures.append("eligible_not_boolean")
        if row.get("evaluated_before_scoring") is not True:
            failures.append("not_attested_pre_score")
        criterion = str(row.get("criterion_version") or "")
        if criterion != PRE_SCORE_CRITERION_VERSION:
            failures.append("criterion_version_changed")
        else:
            criteria[criterion] += 1
        domain_hash = row.get("domain_config_sha256")
        if not _is_sha256(domain_hash):
            failures.append("domain_config_hash_invalid")
        else:
            domain_hashes[str(domain_hash)] += 1
        base_index_hash = row.get("base_index_sha256")
        if not _is_sha256(base_index_hash):
            failures.append("base_index_hash_invalid")
        else:
            base_index_hashes[str(base_index_hash)] += 1
        nucleus = row.get("nucleus")
        if nucleus not in SUPPORTED_NUCLEI:
            failures.append("nucleus_invalid")
        reasons = row.get("reason_codes")
        if (
            not isinstance(reasons, list)
            or any(
                not isinstance(reason, str)
                or _PRE_SCORE_REASON_RE.fullmatch(reason) is None
                for reason in reasons
            )
            or reasons != sorted(set(reasons))
        ):
            failures.append("reason_codes_invalid")
            reasons = []
        if row.get("eligible") is False and not reasons:
            failures.append("ineligible_reason_missing")
        if row.get("eligible") is True and reasons:
            failures.append("eligible_row_has_reasons")
        pool_size = row.get("candidate_pool_size_pre_score")
        decoy_count = row.get("usable_decoy_count_pre_cap")
        if (
            isinstance(pool_size, bool)
            or not isinstance(pool_size, int)
            or pool_size < 0
            or isinstance(decoy_count, bool)
            or not isinstance(decoy_count, int)
            or decoy_count < 0
        ):
            failures.append("pre_score_counts_invalid")
        elif row.get("eligible") is True:
            if nucleus != "13C" or not 2 <= pool_size <= 20 or decoy_count < 1:
                failures.append("eligible_domain_counts_invalid")
        elif pool_size != 0:
            failures.append("ineligible_pool_must_be_empty")
        preflight_input_count = row.get("candidate_count_preflight_input")
        preflight_passed_count = row.get("candidate_count_preflight_passed")
        if (
            isinstance(preflight_input_count, bool)
            or not isinstance(preflight_input_count, int)
            or preflight_input_count < 0
            or isinstance(preflight_passed_count, bool)
            or not isinstance(preflight_passed_count, int)
            or preflight_passed_count < 0
            or preflight_passed_count > preflight_input_count
        ):
            failures.append("conformer_preflight_counts_invalid")
        elif row.get("eligible") is True and (
            preflight_passed_count < pool_size
            or preflight_passed_count < 2
            or preflight_input_count < 2
        ):
            failures.append("eligible_conformer_preflight_counts_invalid")
        for field_name in _CONFORMER_PREFLIGHT_HASH_FIELDS:
            if not _is_sha256(row.get(field_name)):
                failures.append(f"{field_name}_invalid")
        if row.get("conformer_protocol_version") != CONFORMER_PROTOCOL_VERSION:
            failures.append("conformer_protocol_version_changed")
        if (
            row.get("conformer_preflight_protocol_version")
            != CONFORMER_PREFLIGHT_PROTOCOL_VERSION
        ):
            failures.append("conformer_preflight_protocol_version_changed")
        rdkit_version = row.get("conformer_preflight_rdkit_version")
        if (
            not isinstance(rdkit_version, str)
            or _RDKIT_VERSION_RE.fullmatch(rdkit_version) is None
        ):
            failures.append("conformer_preflight_rdkit_version_invalid")
        for field_name, counter in preflight_global_values.items():
            value = row.get(field_name)
            valid = (
                _is_sha256(value)
                if field_name.endswith("_sha256")
                else field_name == "conformer_protocol_version"
                and value == CONFORMER_PROTOCOL_VERSION
                or field_name == "conformer_preflight_protocol_version"
                and value == CONFORMER_PREFLIGHT_PROTOCOL_VERSION
                or field_name == "conformer_preflight_rdkit_version"
                and isinstance(value, str)
                and _RDKIT_VERSION_RE.fullmatch(value) is not None
            )
            if valid:
                counter[str(value)] += 1
        for field_name in (
            "source_document_sha256",
            "molecule_sha256",
            "scaffold_sha256",
            "source_content_sha256",
            "source_release_id",
            "source_records_sha256",
            "source_current_sha256",
        ):
            if not _is_sha256(row.get(field_name)):
                failures.append(f"{field_name}_invalid")
        for field_name, counter in (
            ("source_release_id", source_release_ids),
            ("source_records_sha256", source_records_hashes),
            ("source_current_sha256", source_current_hashes),
        ):
            if _is_sha256(row.get(field_name)):
                counter[str(row[field_name])] += 1
        gold = manifest_lookup.get(record_id)
        if gold is not None:
            expected_bindings = {
                "source_document_sha256": hashlib.sha256(
                    str(gold.get("source_document_key") or "").encode("utf-8")
                ).hexdigest(),
                "molecule_sha256": hashlib.sha256(
                    str(gold.get("molecule_key") or "").encode("utf-8")
                ).hexdigest(),
                "scaffold_sha256": hashlib.sha256(
                    str(gold.get("scaffold_key") or "").encode("utf-8")
                ).hexdigest(),
                "source_content_sha256": gold.get("record_sha256"),
            }
            for field_name, expected_value in expected_bindings.items():
                if row.get(field_name) != expected_value:
                    failures.append(f"{field_name}_gold_mismatch")
            overlap = gold.get("ranker_training_overlap")
            if not isinstance(overlap, Mapping) or row.get(
                "base_index_sha256"
            ) != overlap.get("reference_index_sha256"):
                failures.append("base_index_gold_reference_mismatch")
        if failures:
            invalid.append(
                {
                    "row": index + 1,
                    "record_id": record_id,
                    "failures": failures,
                }
            )
            continue
        for reason in reasons:
            reason_counts[str(reason)] += 1
        if row["eligible"] is True:
            eligible_ids.add(record_id)
    omitted = sorted(expected_ids - seen)
    unexpected = sorted(seen - expected_ids)
    if omitted:
        invalid.append(
            {
                "failure": "pool_records_omitted",
                "count": len(omitted),
                "examples": omitted[:10],
            }
        )
    if unexpected:
        invalid.append(
            {
                "failure": "unexpected_records",
                "count": len(unexpected),
                "examples": unexpected[:10],
            }
        )
    if len(criteria) != 1:
        invalid.append(
            {
                "failure": "criterion_not_global",
                "values": dict(sorted(criteria.items())),
            }
        )
    if len(domain_hashes) != 1:
        invalid.append(
            {
                "failure": "domain_config_not_global",
                "values": dict(sorted(domain_hashes.items())),
            }
        )
    if len(base_index_hashes) != 1:
        invalid.append(
            {
                "failure": "base_index_not_global",
                "values": dict(sorted(base_index_hashes.items())),
            }
        )
    for field_name, counter in (
        ("source_release_id", source_release_ids),
        ("source_records_sha256", source_records_hashes),
        ("source_current_sha256", source_current_hashes),
    ):
        if len(counter) != 1:
            invalid.append(
                {
                    "failure": f"{field_name}_not_global",
                    "values": dict(sorted(counter.items())),
                }
            )
    for field_name, counter in preflight_global_values.items():
        if len(counter) != 1:
            invalid.append(
                {
                    "failure": f"{field_name}_not_global",
                    "values": dict(sorted(counter.items())),
                }
            )
    if invalid:
        raise ValueError(
            "invalid pre-score eligibility audit: " + canonical_json_dumps(invalid[:10])
        )
    subset = [
        dict(row) for row in manifest if str(row.get("record_id")) in eligible_ids
    ]
    by_nucleus = Counter(str(row.get("nucleus")) for row in subset)
    by_split = Counter(str(row.get("split")) for row in subset)
    return subset, {
        "schema_version": "chemapp.nmr.scoreable-subset-audit.v1",
        "selection_stage": "pre_score_applicability_only",
        "selection_used_model_outputs_or_outcomes": False,
        "pool_records": len(manifest),
        "eligibility_rows": len(eligibility_rows),
        "scoreable_records": len(subset),
        "scoreable_fraction": (len(subset) / len(manifest) if manifest else 0.0),
        "scoreable_by_nucleus": dict(sorted(by_nucleus.items())),
        "scoreable_by_split": dict(sorted(by_split.items())),
        "ineligibility_reason_counts": dict(sorted(reason_counts.items())),
        "criterion_versions": dict(sorted(criteria.items())),
        "domain_config_sha256_counts": dict(sorted(domain_hashes.items())),
        "base_index_sha256": (
            next(iter(base_index_hashes)) if len(base_index_hashes) == 1 else None
        ),
        "source_release_id": (
            next(iter(source_release_ids)) if len(source_release_ids) == 1 else None
        ),
        "source_records_sha256": (
            next(iter(source_records_hashes))
            if len(source_records_hashes) == 1
            else None
        ),
        "source_current_sha256": (
            next(iter(source_current_hashes))
            if len(source_current_hashes) == 1
            else None
        ),
        "conformer_preflight_bindings": {
            field_name: next(iter(counter)) if len(counter) == 1 else None
            for field_name, counter in sorted(preflight_global_values.items())
        },
        "pool_manifest_content_sha256": canonical_jsonl_sha256(manifest),
        "eligibility_content_sha256": canonical_jsonl_sha256(eligibility_rows),
        "subset_manifest_content_sha256": canonical_jsonl_sha256(subset),
    }


def adapt_frozen_score_cases(
    cases: Sequence[Mapping[str, Any]],
    manifest: Sequence[Mapping[str, Any]],
    *,
    ranker_artifact: Mapping[str, Any] | None = None,
    ranker_artifact_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Re-derive targets, outcomes and margins under frozen Gold authority.

    Scorer-authored split, target, correctness and margin claims are never
    trusted.  They are checked against the frozen Gold row and candidate
    evidence, then the compact calibration example is emitted from recomputed
    values only.
    """

    manifest_lookup: dict[str, Mapping[str, Any]] = {}
    duplicate_manifest_ids: set[str] = set()
    for row in manifest:
        record_id = str(row.get("record_id") or "")
        if not record_id:
            continue
        if record_id in manifest_lookup:
            duplicate_manifest_ids.add(record_id)
        else:
            manifest_lookup[record_id] = row
    if duplicate_manifest_ids:
        raise ValueError(
            "cannot bind score cases to duplicate Gold record IDs: "
            + canonical_json_dumps(sorted(duplicate_manifest_ids)[:10])
        )

    gold_manifest_sha256 = canonical_jsonl_sha256(manifest)
    reference_hashes = {
        str(overlap.get("reference_index_sha256"))
        for row in manifest
        if isinstance(
            (overlap := row.get("ranker_training_overlap")),
            Mapping,
        )
        and _is_sha256(overlap.get("reference_index_sha256"))
    }
    if len(reference_hashes) != 1:
        raise ValueError(
            "frozen Gold manifest must use one global base-index reference hash"
        )
    gold_reference_hash = next(iter(reference_hashes))
    artifact_base_index_sha256: str | None = None
    artifact_preflight_bindings: dict[str, str] = {}
    upstream_overlap_status = "unknown"
    if ranker_artifact is not None:
        if ranker_artifact.get("schema_version") != FROZEN_RANKER_SCHEMA_VERSION:
            raise ValueError("unsupported frozen ranker artifact schema")
        if not _is_sha256(ranker_artifact_sha256):
            raise ValueError("ranker artifact requires its exact file SHA-256")
        artifact_base_index_sha256 = str(ranker_artifact.get("base_index_sha256") or "")
        gold_reference = ranker_artifact.get("gold_reference")
        upstream_overlap = ranker_artifact.get("upstream_training_overlap")
        artifact_source_release = ranker_artifact.get("source_release_binding")
        if (
            not _is_sha256(artifact_base_index_sha256)
            or artifact_base_index_sha256 != gold_reference_hash
            or not isinstance(gold_reference, Mapping)
            or gold_reference.get("scope") != "chemapp_frozen_nmrshiftdb2_base_index"
            or gold_reference.get("sha256") != artifact_base_index_sha256
        ):
            raise ValueError(
                "ranker artifact base index does not equal the global Gold reference"
            )
        if not isinstance(upstream_overlap, Mapping):
            raise ValueError("ranker artifact lacks upstream training-overlap scope")
        if not isinstance(artifact_source_release, Mapping) or not all(
            _is_sha256(artifact_source_release.get(field_name))
            for field_name in (
                "release_id",
                "records_sha256",
                "source_current_sha256",
            )
        ):
            raise ValueError("ranker artifact lacks its immutable source release")
        upstream_overlap_status = str(upstream_overlap.get("status") or "")
        if upstream_overlap_status not in {"unknown", "checked_no_overlap"}:
            raise ValueError("ranker upstream training-overlap status is invalid")
        artifact_preflight_bindings, preflight_failures = (
            _ranker_conformer_preflight_bindings(ranker_artifact)
        )
        if preflight_failures:
            raise ValueError(
                "ranker artifact conformer-preflight binding is invalid: "
                + canonical_json_dumps(preflight_failures)
            )

    adapted: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    case_ranker_hashes: set[str] = set()
    case_base_hashes: set[str] = set()
    domain_hashes: set[str] = set()
    generation_hashes: set[str] = set()
    semantics_values: set[str] = set()
    source_release_ids: set[str] = set()
    source_records_hashes: set[str] = set()
    source_current_hashes: set[str] = set()
    preflight_global_values: dict[str, set[str]] = {
        field_name: set() for field_name in _CONFORMER_PREFLIGHT_GLOBAL_FIELDS
    }
    for index, case in enumerate(cases):
        failures: list[str] = []
        case_id = str(case.get("case_id") or "")
        record_ids = case.get("record_ids")
        if case.get("schema_version") != FROZEN_SCORE_CASE_SCHEMA_VERSION:
            failures.append("schema_version")
        if "split" in case:
            failures.append("scorer_must_not_assign_split")
        if not case_id:
            failures.append("case_id_missing")
        elif case_id in seen_case_ids:
            failures.append("duplicate_case_id")
        else:
            seen_case_ids.add(case_id)
        if (
            not isinstance(record_ids, list)
            or len(record_ids) != 1
            or any(not _nonempty(value) for value in record_ids)
            or len(set(str(value) for value in record_ids)) != len(record_ids)
        ):
            failures.append("record_ids_invalid")
            bound: list[Mapping[str, Any]] = []
        else:
            bound = [
                manifest_lookup[str(record_id)]
                for record_id in record_ids
                if str(record_id) in manifest_lookup
            ]
            if len(bound) != len(record_ids):
                failures.append("record_not_in_frozen_gold_manifest")
        splits = {
            str(row.get("split"))
            for row in bound
            if row.get("split") in CALIBRATOR_SPLITS
        }
        if bound and len(splits) != 1:
            failures.append("records_do_not_share_one_calibrator_split")
        gold = bound[0] if len(bound) == 1 else None
        expected_target = str(gold.get("inchi_key") or "") if gold else ""
        if case.get("nucleus") != "13C" or (
            gold is not None and gold.get("nucleus") != "13C"
        ):
            failures.append("nucleus_binding_mismatch")
        if case.get("target_inchi_key") != expected_target:
            failures.append("target_not_derived_from_gold")

        candidate_set = case.get("candidate_set")
        if (
            not isinstance(candidate_set, list)
            or len(candidate_set) < 2
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"candidate_id", "smiles"}
                or not _nonempty(item.get("candidate_id"))
                or not _nonempty(item.get("smiles"))
                for item in candidate_set
            )
        ):
            failures.append("candidate_set_invalid")
            candidate_set = []
        elif canonical_sha256(candidate_set) != case.get("candidate_set_sha256"):
            failures.append("candidate_set_hash_mismatch")
        preflight_input_count = case.get("candidate_count_preflight_input")
        preflight_passed_count = case.get("candidate_count_preflight_passed")
        if (
            isinstance(preflight_input_count, bool)
            or not isinstance(preflight_input_count, int)
            or preflight_input_count < 0
            or isinstance(preflight_passed_count, bool)
            or not isinstance(preflight_passed_count, int)
            or preflight_passed_count < 0
            or preflight_passed_count > preflight_input_count
            or (
                isinstance(candidate_set, list)
                and preflight_passed_count < len(candidate_set)
            )
        ):
            failures.append("conformer_preflight_counts_invalid")
        for field_name in _CONFORMER_PREFLIGHT_HASH_FIELDS:
            if not _is_sha256(case.get(field_name)):
                failures.append(f"{field_name}_invalid")
        if case.get("conformer_protocol_version") != CONFORMER_PROTOCOL_VERSION:
            failures.append("conformer_protocol_version_changed")
        if (
            case.get("conformer_preflight_protocol_version")
            != CONFORMER_PREFLIGHT_PROTOCOL_VERSION
        ):
            failures.append("conformer_preflight_protocol_version_changed")
        rdkit_version = case.get("conformer_preflight_rdkit_version")
        if (
            not isinstance(rdkit_version, str)
            or _RDKIT_VERSION_RE.fullmatch(rdkit_version) is None
        ):
            failures.append("conformer_preflight_rdkit_version_invalid")
        for field_name, values in preflight_global_values.items():
            value = case.get(field_name)
            if isinstance(value, str):
                values.add(value)
            expected = artifact_preflight_bindings.get(field_name)
            if expected is not None and value != expected:
                failures.append(f"{field_name}_ranker_artifact_mismatch")

        evidence = case.get("candidate_evidence")
        normalized_evidence: list[Mapping[str, Any]] = []
        if (
            not isinstance(evidence, list)
            or len(evidence) < 2
            or any(not isinstance(item, Mapping) for item in evidence)
        ):
            failures.append("candidate_evidence_invalid")
        else:
            normalized_evidence = sorted(
                evidence,
                key=lambda item: (
                    item.get("relative_rank")
                    if isinstance(item.get("relative_rank"), int)
                    else 10**9
                ),
            )
            ranks = [item.get("relative_rank") for item in normalized_evidence]
            candidate_ids = [
                str(item.get("candidate_id") or "") for item in normalized_evidence
            ]
            evidence_inchi_keys = [
                str(item.get("inchi_key") or "") for item in normalized_evidence
            ]
            maes = [item.get("mae_ppm") for item in normalized_evidence]
            if (
                ranks != list(range(1, len(normalized_evidence) + 1))
                or len(set(candidate_ids)) != len(candidate_ids)
                or any(not value for value in candidate_ids)
                or any(not value for value in evidence_inchi_keys)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    for value in maes
                )
            ):
                failures.append("candidate_evidence_binding_invalid")
            if candidate_set and {
                str(item["candidate_id"]) for item in candidate_set
            } != set(candidate_ids):
                failures.append("candidate_set_evidence_ids_mismatch")
            if canonical_sha256(evidence) != case.get("candidate_evidence_sha256"):
                failures.append("candidate_evidence_hash_mismatch")
            identity_binding = [
                {
                    "candidate_id": item.get("candidate_id"),
                    "inchi_key": item.get("inchi_key"),
                    "connectivity_key": item.get("connectivity_key"),
                }
                for item in normalized_evidence
            ]
            if canonical_sha256(identity_binding) != case.get(
                "candidate_identity_sha256"
            ):
                failures.append("candidate_identity_hash_mismatch")

        recomputed_top1 = ""
        recomputed_raw_score: float | None = None
        recomputed_outcome: bool | None = None
        if len(normalized_evidence) >= 2:
            recomputed_top1 = str(normalized_evidence[0].get("inchi_key") or "")
            top_mae = normalized_evidence[0].get("mae_ppm")
            runner_mae = normalized_evidence[1].get("mae_ppm")
            if all(
                not isinstance(value, bool)
                and isinstance(value, (int, float))
                and math.isfinite(float(value))
                for value in (top_mae, runner_mae)
            ):
                recomputed_raw_score = max(
                    float(runner_mae) - float(top_mae),
                    0.0,
                )
                if float(runner_mae) + 1e-10 < float(top_mae):
                    failures.append("evidence_rank_margin_inconsistent")
            recomputed_outcome = recomputed_top1 == expected_target
        if case.get("predicted_top1_inchi_key") != recomputed_top1:
            failures.append("predicted_top1_not_derived_from_evidence")
        if case.get("top1_exact_correct") is not recomputed_outcome:
            failures.append("outcome_not_derived_from_gold_and_evidence")
        if case.get("top_candidate_is_truth") is not recomputed_outcome:
            failures.append("truth_role_not_derived_from_gold_and_evidence")
        claimed_raw_score = case.get("raw_score")
        if (
            recomputed_raw_score is None
            or isinstance(claimed_raw_score, bool)
            or not isinstance(claimed_raw_score, (int, float))
            or not math.isfinite(float(claimed_raw_score))
            or not math.isclose(
                float(claimed_raw_score),
                recomputed_raw_score,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            failures.append("raw_score_not_derived_from_evidence")
        if case.get("candidate_pool_size") != len(normalized_evidence):
            failures.append("candidate_pool_size_mismatch")

        raw_features = case.get("raw_features")
        if not isinstance(raw_features, Mapping):
            failures.append("raw_features_missing")
        elif len(normalized_evidence) >= 2 and recomputed_raw_score is not None:
            expected_raw_features = {
                "top1_mae_ppm": float(normalized_evidence[0]["mae_ppm"]),
                "runner_up_mae_ppm": float(normalized_evidence[1]["mae_ppm"]),
                "top_margin_mae_ppm": recomputed_raw_score,
                "pool_size": len(normalized_evidence),
            }
            for field_name, expected_value in expected_raw_features.items():
                actual = raw_features.get(field_name)
                if isinstance(expected_value, float):
                    matches = (
                        not isinstance(actual, bool)
                        and isinstance(actual, (int, float))
                        and math.isclose(
                            float(actual),
                            expected_value,
                            rel_tol=1e-12,
                            abs_tol=1e-12,
                        )
                    )
                else:
                    matches = actual == expected_value
                if not matches:
                    failures.append(f"raw_features_{field_name}_mismatch")

        score_semantics = str(case.get("score_semantics") or "")
        if not score_semantics:
            failures.append("score_semantics_missing")
        else:
            semantics_values.add(score_semantics)
        case_ranker_hash = case.get("ranker_artifact_sha256")
        if not _is_sha256(case_ranker_hash):
            failures.append("ranker_artifact_sha256_invalid")
        else:
            case_ranker_hashes.add(str(case_ranker_hash))
            if (
                ranker_artifact_sha256 is not None
                and case_ranker_hash != ranker_artifact_sha256
            ):
                failures.append("ranker_artifact_sha256_mismatch")
        base_index_hash = case.get("base_index_sha256")
        if not _is_sha256(base_index_hash):
            failures.append("base_index_sha256_invalid")
        else:
            case_base_hashes.add(str(base_index_hash))
            if base_index_hash != gold_reference_hash:
                failures.append("base_index_gold_reference_mismatch")
            if (
                artifact_base_index_sha256 is not None
                and base_index_hash != artifact_base_index_sha256
            ):
                failures.append("base_index_ranker_artifact_mismatch")
        for field_name, values in (
            ("domain_config_sha256", domain_hashes),
            (
                "candidate_generation_config_sha256",
                generation_hashes,
            ),
        ):
            value = case.get(field_name)
            if not _is_sha256(value):
                failures.append(f"{field_name}_invalid")
            else:
                values.add(str(value))
                if ranker_artifact is not None and value != ranker_artifact.get(
                    field_name
                ):
                    failures.append(f"{field_name}_ranker_artifact_mismatch")
        artifact_source_release = (
            ranker_artifact.get("source_release_binding")
            if isinstance(ranker_artifact, Mapping)
            else None
        )
        for field_name, values in (
            ("source_release_id", source_release_ids),
            ("source_records_sha256", source_records_hashes),
            ("source_current_sha256", source_current_hashes),
        ):
            value = case.get(field_name)
            if not _is_sha256(value):
                failures.append(f"{field_name}_invalid")
            else:
                values.add(str(value))
                if isinstance(
                    artifact_source_release, Mapping
                ) and value != artifact_source_release.get(
                    {
                        "source_release_id": "release_id",
                        "source_records_sha256": "records_sha256",
                        "source_current_sha256": "source_current_sha256",
                    }[field_name]
                ):
                    failures.append(f"{field_name}_ranker_artifact_mismatch")
        if ranker_artifact is not None and score_semantics != ranker_artifact.get(
            "score_semantics"
        ):
            failures.append("score_semantics_ranker_artifact_mismatch")

        if gold is not None:
            expected_case_bindings = {
                "source_document_key": gold.get("source_document_key"),
                "molecule_key": gold.get("molecule_key"),
                "scaffold_key": gold.get("scaffold_key"),
                "source_content_sha256": gold.get("record_sha256"),
                "source_document_sha256": hashlib.sha256(
                    str(gold.get("source_document_key") or "").encode("utf-8")
                ).hexdigest(),
                "molecule_sha256": hashlib.sha256(
                    str(gold.get("molecule_key") or "").encode("utf-8")
                ).hexdigest(),
                "scaffold_sha256": hashlib.sha256(
                    str(gold.get("scaffold_key") or "").encode("utf-8")
                ).hexdigest(),
            }
            for field_name, expected_value in expected_case_bindings.items():
                if case.get(field_name) != expected_value:
                    failures.append(f"{field_name}_gold_mismatch")
        if failures:
            invalid.append(
                {
                    "row": index + 1,
                    "case_id": case_id,
                    "failures": sorted(set(failures)),
                }
            )
            continue
        split = next(iter(splits))
        assert recomputed_raw_score is not None
        assert recomputed_outcome is not None
        adapted.append(
            {
                "schema_version": CALIBRATION_EXAMPLES_SCHEMA_VERSION,
                "case_id": case_id,
                "split": split,
                "record_ids": [str(value) for value in record_ids],
                "raw_score": recomputed_raw_score,
                "top1_exact_correct": recomputed_outcome,
                "score_semantics": score_semantics,
                "ranker_artifact_sha256": str(case_ranker_hash),
                "base_index_sha256": str(base_index_hash),
                "gold_manifest_content_sha256": gold_manifest_sha256,
                "source_release_id": str(case["source_release_id"]),
                "source_records_sha256": str(case["source_records_sha256"]),
                "source_current_sha256": str(case["source_current_sha256"]),
                "gold_record_binding_sha256": canonical_sha256(
                    {
                        "record_id": gold["record_id"],
                        "record_sha256": gold["record_sha256"],
                        "spectrum_fingerprint_sha256": gold[
                            "spectrum_fingerprint_sha256"
                        ],
                        "inchi_key": gold["inchi_key"],
                        "molecule_key": gold["molecule_key"],
                        "scaffold_key": gold["scaffold_key"],
                        "source_document_key": gold["source_document_key"],
                        "split": split,
                    }
                ),
                "frozen_case_content_sha256": canonical_sha256(case),
            }
        )
    if invalid:
        raise ValueError(
            "invalid frozen score cases: " + canonical_json_dumps(invalid[:10])
        )

    global_failures: list[str] = []
    if len(case_ranker_hashes) != 1:
        global_failures.append("ranker_artifact_hash_not_global")
    if len(case_base_hashes) != 1:
        global_failures.append("base_index_hash_not_global")
    if len(domain_hashes) != 1:
        global_failures.append("domain_config_hash_not_global")
    if len(generation_hashes) != 1:
        global_failures.append("candidate_generation_hash_not_global")
    if len(semantics_values) != 1:
        global_failures.append("score_semantics_not_global")
    if len(source_release_ids) != 1:
        global_failures.append("source_release_id_not_global")
    if len(source_records_hashes) != 1:
        global_failures.append("source_records_hash_not_global")
    if len(source_current_hashes) != 1:
        global_failures.append("source_current_hash_not_global")
    for field_name, values in preflight_global_values.items():
        if len(values) != 1:
            global_failures.append(f"{field_name}_not_global")
    if global_failures:
        raise ValueError(
            "invalid frozen score cases: "
            + canonical_json_dumps({"global_failures": global_failures})
        )

    record_split_mapping = sorted(
        (
            str(row.get("record_id")),
            str(row.get("split")),
        )
        for row in manifest
        if _nonempty(row.get("record_id")) and row.get("split") in CALIBRATOR_SPLITS
    )
    by_split = Counter(str(row["split"]) for row in adapted)
    return adapted, {
        "schema_version": "chemapp.nmr.score-case-split-binding.v1",
        "binding_authority": "frozen_gold_manifest_only",
        "scorer_split_fields_accepted": False,
        "input_cases": len(cases),
        "output_examples": len(adapted),
        "examples_by_split": dict(sorted(by_split.items())),
        "input_cases_content_sha256": canonical_jsonl_sha256(cases),
        "gold_manifest_content_sha256": canonical_jsonl_sha256(manifest),
        "gold_reference_base_index_sha256": gold_reference_hash,
        "ranker_artifact_sha256": next(iter(case_ranker_hashes)),
        "domain_config_sha256": next(iter(domain_hashes)),
        "candidate_generation_config_sha256": next(iter(generation_hashes)),
        "upstream_dp5q_training_overlap": upstream_overlap_status,
        "source_release_id": next(iter(source_release_ids)),
        "source_records_sha256": next(iter(source_records_hashes)),
        "source_current_sha256": next(iter(source_current_hashes)),
        "conformer_preflight_bindings": {
            field_name: next(iter(values))
            for field_name, values in sorted(preflight_global_values.items())
        },
        "record_split_mapping_sha256": canonical_sha256(record_split_mapping),
        "output_examples_content_sha256": canonical_jsonl_sha256(adapted),
    }


def policy_from_mapping(value: Mapping[str, Any]) -> CalibrationGatePolicy:
    allowed = set(CalibrationGatePolicy.__dataclass_fields__)
    unexpected = sorted(set(value) - allowed)
    if unexpected:
        raise ValueError(f"unknown calibration policy keys: {unexpected}")
    policy = CalibrationGatePolicy(**dict(value))
    reasons = _policy_validation_reasons(policy)
    if reasons:
        examples = reasons[0].get("examples") or []
        raise ValueError(f"invalid calibration policy: {examples}")
    return policy


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _HASH_RE.fullmatch(value) is not None


def _ranker_conformer_preflight_bindings(
    artifact: Mapping[str, Any],
) -> tuple[dict[str, str], list[str]]:
    """Validate and flatten the v3 ranker's nested preflight runtime binding."""

    failures: list[str] = []
    value = artifact.get("conformer_preflight")
    required_fields = {
        "sidecar_protocol_version",
        "sidecar_code_sha256",
        "rdkit_version",
        "conformer_protocol_version",
        "conformer_generation_config_sha256",
        "conformer_preflight_protocol_version",
        "conformer_preflight_policy_sha256",
        "model_outputs_created",
        "runtime_sha256",
        "manifest_sha256",
    }
    if not isinstance(value, Mapping):
        return {}, ["conformer_preflight_missing"]
    if set(value) != required_fields:
        failures.append("conformer_preflight_field_allowlist_mismatch")
    bindings = {
        "conformer_preflight_sidecar_sha256": value.get("sidecar_code_sha256"),
        "conformer_preflight_rdkit_version": value.get("rdkit_version"),
        "conformer_protocol_version": value.get("conformer_protocol_version"),
        "conformer_generation_config_sha256": value.get(
            "conformer_generation_config_sha256"
        ),
        "conformer_preflight_protocol_version": value.get(
            "conformer_preflight_protocol_version"
        ),
        "conformer_preflight_policy_sha256": value.get(
            "conformer_preflight_policy_sha256"
        ),
        "conformer_preflight_runtime_sha256": value.get("runtime_sha256"),
        "conformer_preflight_manifest_sha256": value.get("manifest_sha256"),
    }
    for field_name in _CONFORMER_PREFLIGHT_GLOBAL_FIELDS:
        candidate = bindings.get(field_name)
        if field_name.endswith("_sha256"):
            valid = _is_sha256(candidate)
        elif field_name == "conformer_protocol_version":
            valid = candidate == CONFORMER_PROTOCOL_VERSION
        elif field_name == "conformer_preflight_protocol_version":
            valid = candidate == CONFORMER_PREFLIGHT_PROTOCOL_VERSION
        else:
            valid = (
                isinstance(candidate, str)
                and _RDKIT_VERSION_RE.fullmatch(candidate) is not None
            )
        if not valid:
            failures.append(f"{field_name}_invalid")
    runtime_core = {
        key: value.get(key)
        for key in sorted(required_fields - {"runtime_sha256", "manifest_sha256"})
    }
    if (
        value.get("sidecar_protocol_version") != 1
        or value.get("model_outputs_created") is not False
        or canonical_sha256(runtime_core) != value.get("runtime_sha256")
    ):
        failures.append("conformer_preflight_runtime_binding_mismatch")
    conformer_generation = artifact.get("conformer_generation")
    if (
        not isinstance(conformer_generation, Mapping)
        or canonical_sha256(conformer_generation)
        != bindings.get("conformer_generation_config_sha256")
        or artifact.get("conformer_generation_config_sha256")
        != bindings.get("conformer_generation_config_sha256")
        or artifact.get("conformer_protocol_version")
        != bindings.get("conformer_protocol_version")
        or artifact.get("sidecar_code_sha256")
        != bindings.get("conformer_preflight_sidecar_sha256")
    ):
        failures.append("conformer_preflight_ranker_binding_mismatch")
    return (
        {field_name: str(value) for field_name, value in bindings.items()},
        sorted(set(failures)),
    )


def _parse_utc(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    if parsed.utcoffset().total_seconds() != 0:
        return None
    return parsed


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _reason(
    code: str,
    message: str,
    *,
    count: int | None = None,
    examples: Iterable[Any] = (),
) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "message": message}
    if count is not None:
        result["count"] = int(count)
    examples_list = list(examples)[:10]
    if examples_list:
        result["examples"] = examples_list
    return result


def manual_review_binding_sha256(
    record: Mapping[str, Any],
    review: Mapping[str, Any],
) -> str:
    """Bind a human attestation to the exact spectrum and gold identity."""

    payload = {
        "record_id": record.get("record_id"),
        "record_sha256": record.get("record_sha256"),
        "spectrum_fingerprint_sha256": record.get("spectrum_fingerprint_sha256"),
        "molecule_key": record.get("molecule_key"),
        "inchi_key": record.get("inchi_key"),
        "canonical_smiles": record.get("canonical_smiles"),
        "scaffold_key": record.get("scaffold_key"),
        "source_document_key": record.get("source_document_key"),
        "dataset_release_id": record.get("dataset_release_id"),
        "nucleus": record.get("nucleus"),
        "reviewer_id": review.get("reviewer_id"),
        "reviewed_at": review.get("reviewed_at"),
        "protocol_version": review.get("protocol_version"),
        "decision": review.get("decision"),
        "review_kind": review.get("review_kind"),
        "structure_verified": review.get("structure_verified"),
        "spectrum_verified": review.get("spectrum_verified"),
        "nucleus_verified": review.get("nucleus_verified"),
        "source_document_verified": review.get("source_document_verified"),
        "duplicate_check_completed": review.get("duplicate_check_completed"),
        "upstream_review_evidence_sha256": review.get(
            "upstream_review_evidence_sha256"
        ),
    }
    return canonical_sha256(payload)


def _review_failures(
    record: Mapping[str, Any],
    *,
    required_reviews: int,
) -> list[str]:
    reviews = record.get("manual_reviews")
    if not isinstance(reviews, list):
        return ["manual_reviews_missing"]
    accepted_reviewers: set[str] = set()
    failures: list[str] = []
    for review in reviews:
        if not isinstance(review, Mapping):
            failures.append("manual_review_not_object")
            continue
        reviewer = str(review.get("reviewer_id") or "").strip()
        review_kind = review.get("review_kind")
        human_identity_valid = (
            review_kind == "human" and _parse_utc(review.get("reviewed_at")) is not None
        )
        upstream_release_valid = (
            review_kind == "upstream_human_checked_release"
            and review.get("reviewed_at") is None
            and _is_sha256(review.get("upstream_review_evidence_sha256"))
        )
        valid = (
            reviewer
            and (human_identity_valid or upstream_release_valid)
            and review.get("decision") == "accepted"
            and review.get("protocol_version") == REVIEW_PROTOCOL_VERSION
            and review.get("structure_verified") is True
            and review.get("spectrum_verified") is True
            and review.get("nucleus_verified") is True
            and review.get("source_document_verified") is True
            and review.get("duplicate_check_completed") is True
            and review.get("review_record_sha256")
            == manual_review_binding_sha256(record, review)
        )
        if valid:
            accepted_reviewers.add(reviewer)
        else:
            failures.append("manual_review_invalid_or_unbound")
    if len(accepted_reviewers) < required_reviews:
        failures.append("manual_review_quorum_not_met")
    return sorted(set(failures))


def _computed_structure_identity(smiles: Any) -> dict[str, str] | None:
    if Chem is None or MurckoScaffold is None:
        return None
    if not _nonempty(smiles):
        return None
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None:
        return None
    canonical_smiles = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    inchi_key = Chem.MolToInchiKey(molecule)
    if not inchi_key:
        return None
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=molecule,
        includeChirality=False,
    )
    return {
        "canonical_smiles": canonical_smiles,
        "inchi_key": inchi_key,
        "molecule_key": inchi_key.split("-", 1)[0],
        "scaffold_key": scaffold or "acyclic",
    }


def _value_counts_by_split(
    records: Sequence[Mapping[str, Any]],
    field_name: str,
) -> dict[str, int]:
    return {
        split: len(
            {
                str(row.get(field_name))
                for row in records
                if row.get("split") == split and _nonempty(row.get(field_name))
            }
        )
        for split in SPLITS
    }


def _group_leakage(
    records: Sequence[Mapping[str, Any]],
    field_name: str,
) -> dict[str, Any]:
    owners: dict[str, set[str]] = defaultdict(set)
    for row in records:
        value = row.get(field_name)
        split = row.get("split")
        if _nonempty(value) and split in SPLITS:
            owners[str(value)].add(str(split))
    leaked = {
        value: sorted(splits) for value, splits in owners.items() if len(splits) > 1
    }
    calibration_test = sorted(
        value
        for value, splits in owners.items()
        if {"calibration", "test"}.issubset(splits)
    )
    return {
        "groups": len(owners),
        "leaked_groups": len(leaked),
        "leak_examples": [
            {"value": value, "splits": splits}
            for value, splits in sorted(leaked.items())[:10]
        ],
        "calibration_test_overlap_count": len(calibration_test),
        "calibration_test_overlap_examples": calibration_test[:10],
    }


def _threshold_reasons(
    actual: Mapping[str, int],
    expected: Mapping[str, int],
    code_prefix: str,
    label: str,
) -> list[dict[str, Any]]:
    reasons = []
    for split in SPLITS:
        minimum = int(expected.get(split, 0))
        observed = int(actual.get(split, 0))
        if observed < minimum:
            reasons.append(
                _reason(
                    f"{code_prefix}_{split}",
                    f"{split} requires at least {minimum} {label}; found {observed}",
                    count=observed,
                )
            )
    return reasons


def production_policy_weaknesses(
    policy: CalibrationGatePolicy,
) -> list[dict[str, Any]]:
    """Return ways a policy is weaker than the built-in production floor."""

    baseline = CalibrationGatePolicy()
    weaknesses: list[dict[str, Any]] = []
    minimum_maps = (
        "minimum_records",
        "minimum_molecules",
        "minimum_scaffolds",
        "minimum_source_documents",
        "minimum_records_per_nucleus",
        "minimum_paired_molecules",
        "required_human_reviews",
    )
    for field_name in minimum_maps:
        actual = getattr(policy, field_name)
        required = getattr(baseline, field_name)
        for split in SPLITS:
            if int(actual.get(split, 0)) < int(required[split]):
                weaknesses.append(
                    {
                        "field": f"{field_name}.{split}",
                        "configured": int(actual.get(split, 0)),
                        "production_floor": int(required[split]),
                    }
                )
    minimum_scalars = (
        "minimum_calibration_cases",
        "minimum_test_cases",
        "minimum_outcomes_per_class",
        "minimum_unique_scores",
        "minimum_test_brier_skill",
        "minimum_test_roc_auc",
        "minimum_populated_reliability_bins",
        "minimum_calibration_slope",
    )
    for field_name in minimum_scalars:
        actual = float(getattr(policy, field_name))
        required = float(getattr(baseline, field_name))
        if actual < required:
            weaknesses.append(
                {
                    "field": field_name,
                    "configured": actual,
                    "production_floor": required,
                }
            )
    maximum_scalars = (
        "maximum_test_ece",
        "maximum_test_brier",
        "maximum_absolute_calibration_intercept",
        "maximum_calibration_slope",
    )
    for field_name in maximum_scalars:
        actual = float(getattr(policy, field_name))
        required = float(getattr(baseline, field_name))
        if actual > required:
            weaknesses.append(
                {
                    "field": field_name,
                    "configured": actual,
                    "production_floor": required,
                }
            )
    if policy.reliability_bins != baseline.reliability_bins:
        weaknesses.append(
            {
                "field": "reliability_bins",
                "configured": policy.reliability_bins,
                "production_floor": baseline.reliability_bins,
            }
        )
    return weaknesses


def _policy_validation_reasons(
    policy: CalibrationGatePolicy,
) -> list[dict[str, Any]]:
    errors: list[str] = []
    if policy.schema_version != "chemapp.nmr.calibration-gate-policy.v1":
        errors.append("unsupported schema_version")
    map_fields = (
        "minimum_records",
        "minimum_molecules",
        "minimum_scaffolds",
        "minimum_source_documents",
        "minimum_records_per_nucleus",
        "minimum_paired_molecules",
        "required_human_reviews",
    )
    for field_name in map_fields:
        value = getattr(policy, field_name)
        if not isinstance(value, Mapping) or set(value) != set(SPLITS):
            errors.append(f"{field_name} must contain exactly {list(SPLITS)}")
            continue
        if any(type(item) is not int or item < 0 for item in value.values()):
            errors.append(f"{field_name} values must be non-negative integers")
    integer_fields = (
        "minimum_calibration_cases",
        "minimum_test_cases",
        "minimum_outcomes_per_class",
        "minimum_unique_scores",
        "reliability_bins",
        "minimum_populated_reliability_bins",
    )
    for field_name in integer_fields:
        value = getattr(policy, field_name)
        if type(value) is not int or value < 0:
            errors.append(f"{field_name} must be a non-negative integer")
    if type(policy.reliability_bins) is int and policy.reliability_bins < 2:
        errors.append("reliability_bins must be at least 2")
    number_fields = (
        "maximum_test_ece",
        "maximum_test_brier",
        "minimum_test_brier_skill",
        "minimum_test_roc_auc",
        "maximum_absolute_calibration_intercept",
        "minimum_calibration_slope",
        "maximum_calibration_slope",
    )
    numbers_valid = True
    for field_name in number_fields:
        value = getattr(policy, field_name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            errors.append(f"{field_name} must be finite")
            numbers_valid = False
    if numbers_valid:
        if not 0 <= policy.maximum_test_ece <= 1:
            errors.append("maximum_test_ece must be in [0, 1]")
        if not 0 <= policy.maximum_test_brier <= 1:
            errors.append("maximum_test_brier must be in [0, 1]")
        if not 0 <= policy.minimum_test_roc_auc <= 1:
            errors.append("minimum_test_roc_auc must be in [0, 1]")
        if policy.maximum_absolute_calibration_intercept < 0:
            errors.append("maximum_absolute_calibration_intercept cannot be negative")
        if (
            policy.minimum_calibration_slope < 0
            or policy.maximum_calibration_slope < policy.minimum_calibration_slope
        ):
            errors.append("calibration slope interval is invalid")
    if not errors:
        return []
    return [
        _reason(
            "invalid_gate_policy",
            "calibration gate policy is invalid",
            count=len(errors),
            examples=errors,
        )
    ]


def audit_gold_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    policy: CalibrationGatePolicy | None = None,
) -> dict[str, Any]:
    """Audit a gold spectrum manifest without training or emitting probability."""

    selected_policy = policy or CalibrationGatePolicy()
    policy_reasons = _policy_validation_reasons(selected_policy)
    effective_policy = (
        selected_policy if not policy_reasons else CalibrationGatePolicy()
    )
    reasons: list[dict[str, Any]] = list(policy_reasons)
    required_fields = (
        "schema_version",
        "record_id",
        "record_sha256",
        "spectrum_fingerprint_sha256",
        "molecule_key",
        "inchi_key",
        "canonical_smiles",
        "scaffold_key",
        "source_document_key",
        "dataset_release_id",
        "source_name",
        "nucleus",
        "split",
        "measurement_kind",
        "manual_reviews",
        "ranker_training_overlap",
    )
    missing: Counter[str] = Counter()
    invalid_hashes: Counter[str] = Counter()
    invalid_enums: Counter[str] = Counter()
    provenance_failures: Counter[str] = Counter()
    identity_failures: Counter[str] = Counter()
    identity_failure_examples: list[dict[str, Any]] = []
    duplicate_record_ids: list[str] = []
    record_ids: set[str] = set()
    review_failure_counts: Counter[str] = Counter()
    review_failure_examples: list[dict[str, Any]] = []
    reference_index_hashes: Counter[str] = Counter()

    for index, row in enumerate(records):
        for field_name in required_fields:
            if field_name not in row or row.get(field_name) in (None, ""):
                missing[field_name] += 1
        record_id = str(row.get("record_id") or "")
        if record_id in record_ids:
            duplicate_record_ids.append(record_id)
        if record_id:
            record_ids.add(record_id)
        for field_name in (
            "record_sha256",
            "spectrum_fingerprint_sha256",
        ):
            if not _is_sha256(row.get(field_name)):
                invalid_hashes[field_name] += 1
        if row.get("split") not in SPLITS:
            invalid_enums["split"] += 1
        if row.get("schema_version") != GOLD_MANIFEST_SCHEMA_VERSION:
            invalid_enums["schema_version"] += 1
        if row.get("nucleus") not in SUPPORTED_NUCLEI:
            invalid_enums["nucleus"] += 1
        if row.get("measurement_kind") != "measured":
            invalid_enums["measurement_kind"] += 1
        source_document = str(row.get("source_document_key") or "")
        if (
            not _SOURCE_DOCUMENT_RE.fullmatch(source_document)
            or source_document != source_document.casefold()
        ):
            provenance_failures["source_document_key_not_normalized"] += 1
        if str(
            row.get("source_name") or ""
        ).casefold() == "nmrexp" and source_document == row.get("dataset_release_id"):
            provenance_failures["nmrexp_release_id_used_as_source_document"] += 1
        overlap = row.get("ranker_training_overlap")
        if not isinstance(overlap, Mapping):
            provenance_failures["ranker_training_overlap_missing"] += 1
        else:
            if overlap.get("status") != "checked":
                provenance_failures["ranker_training_overlap_unchecked"] += 1
            if not _is_sha256(overlap.get("reference_index_sha256")):
                provenance_failures["ranker_training_reference_hash_invalid"] += 1
            else:
                reference_index_hashes[str(overlap["reference_index_sha256"])] += 1
            if overlap.get("reference_scope") not in {
                None,
                "chemapp_frozen_nmrshiftdb2_base_index",
            }:
                provenance_failures["overlap_reference_scope_invalid"] += 1
            if overlap.get("connectivity") is not False:
                provenance_failures["ranker_training_connectivity_overlap"] += 1
            if overlap.get("exact_inchi_key") is not False:
                provenance_failures["ranker_training_exact_overlap"] += 1
            if overlap.get("parent_connectivity") not in {None, False}:
                provenance_failures["ranker_training_parent_overlap"] += 1
            if overlap.get("exact_parent_inchi_key") not in {None, False}:
                provenance_failures["ranker_training_exact_parent_overlap"] += 1
            if overlap.get("upstream_dp5q_training_overlap") not in {
                None,
                "unknown",
            }:
                provenance_failures["upstream_dp5q_overlap_claim_not_supported"] += 1
        identity = _computed_structure_identity(row.get("canonical_smiles"))
        identity_row_failures = []
        if identity is None:
            identity_row_failures.append("canonical_smiles_unparseable")
        else:
            for field_name, computed_value in identity.items():
                if row.get(field_name) != computed_value:
                    identity_row_failures.append(f"{field_name}_mismatch")
        for failure in identity_row_failures:
            identity_failures[failure] += 1
        if identity_row_failures and len(identity_failure_examples) < 10:
            identity_failure_examples.append(
                {
                    "row": index + 1,
                    "record_id": record_id,
                    "failures": identity_row_failures,
                }
            )
        split = str(row.get("split") or "")
        required_reviews = int(effective_policy.required_human_reviews.get(split, 1))
        failures = _review_failures(row, required_reviews=required_reviews)
        for failure in failures:
            review_failure_counts[failure] += 1
        if failures and len(review_failure_examples) < 10:
            review_failure_examples.append(
                {
                    "row": index + 1,
                    "record_id": record_id,
                    "failures": failures,
                }
            )

    if len(reference_index_hashes) > 1:
        provenance_failures["reference_index_hash_not_global"] += sum(
            reference_index_hashes.values()
        )
    if not records:
        reasons.append(_reason("manifest_empty", "gold spectrum manifest is empty"))
    if missing:
        reasons.append(
            _reason(
                "required_fields_missing",
                "gold records are missing required provenance or identity fields",
                count=sum(missing.values()),
                examples=[
                    {"field": key, "count": value} for key, value in missing.items()
                ],
            )
        )
    if invalid_hashes:
        reasons.append(
            _reason(
                "invalid_content_hashes",
                "gold records require lowercase SHA-256 content bindings",
                count=sum(invalid_hashes.values()),
                examples=[
                    {"field": key, "count": value}
                    for key, value in invalid_hashes.items()
                ],
            )
        )
    if invalid_enums:
        reasons.append(
            _reason(
                "invalid_gold_record_values",
                "gold records must be measured 1H/13C spectra in a defined split",
                count=sum(invalid_enums.values()),
                examples=[
                    {"field": key, "count": value}
                    for key, value in invalid_enums.items()
                ],
            )
        )
    if provenance_failures:
        reasons.append(
            _reason(
                "source_provenance_gate_failed",
                "source_document_key must identify the normalized original "
                "document, separately from dataset_release_id",
                count=sum(provenance_failures.values()),
                examples=[
                    {"failure": key, "count": value}
                    for key, value in sorted(provenance_failures.items())
                ],
            )
        )
    if Chem is None or MurckoScaffold is None:
        reasons.append(
            _reason(
                "rdkit_unavailable",
                "RDKit is required to recompute gold structure identities",
            )
        )
    elif identity_failures:
        reasons.append(
            _reason(
                "structure_identity_gate_failed",
                "canonical structure, InChIKey, molecule and scaffold bindings "
                "must agree with an independent RDKit recomputation",
                count=sum(identity_failures.values()),
                examples=identity_failure_examples,
            )
        )
    if duplicate_record_ids:
        reasons.append(
            _reason(
                "duplicate_record_ids",
                "record_id must be unique",
                count=len(duplicate_record_ids),
                examples=duplicate_record_ids,
            )
        )
    if review_failure_counts:
        reasons.append(
            _reason(
                "manual_review_gate_failed",
                "every spectrum requires bound, accepted human-review quorum",
                count=sum(review_failure_counts.values()),
                examples=review_failure_examples,
            )
        )

    counts = Counter(
        str(row.get("split")) for row in records if row.get("split") in SPLITS
    )
    record_counts = {split: counts.get(split, 0) for split in SPLITS}
    molecule_counts = _value_counts_by_split(records, "molecule_key")
    scaffold_counts = _value_counts_by_split(records, "scaffold_key")
    source_document_counts = _value_counts_by_split(records, "source_document_key")
    reasons.extend(
        _threshold_reasons(
            record_counts,
            effective_policy.minimum_records,
            "insufficient_records",
            "spectrum records",
        )
    )
    reasons.extend(
        _threshold_reasons(
            molecule_counts,
            effective_policy.minimum_molecules,
            "insufficient_molecules",
            "molecules",
        )
    )
    reasons.extend(
        _threshold_reasons(
            scaffold_counts,
            effective_policy.minimum_scaffolds,
            "insufficient_scaffolds",
            "scaffolds",
        )
    )
    reasons.extend(
        _threshold_reasons(
            source_document_counts,
            effective_policy.minimum_source_documents,
            "insufficient_source_documents",
            "source documents",
        )
    )

    nucleus_counts: dict[str, dict[str, int]] = {}
    paired_molecules: dict[str, int] = {}
    for split in SPLITS:
        nucleus_counts[split] = {
            nucleus: sum(
                row.get("split") == split and row.get("nucleus") == nucleus
                for row in records
            )
            for nucleus in SUPPORTED_NUCLEI
        }
        molecule_nuclei: dict[str, set[str]] = defaultdict(set)
        for row in records:
            if row.get("split") == split and row.get("nucleus") in SUPPORTED_NUCLEI:
                molecule_nuclei[str(row.get("molecule_key"))].add(str(row["nucleus"]))
        paired_molecules[split] = sum(
            set(SUPPORTED_NUCLEI).issubset(nuclei)
            for nuclei in molecule_nuclei.values()
        )
        minimum_nucleus = int(
            effective_policy.minimum_records_per_nucleus.get(split, 0)
        )
        for nucleus in SUPPORTED_NUCLEI:
            observed = nucleus_counts[split][nucleus]
            if observed < minimum_nucleus:
                reasons.append(
                    _reason(
                        f"insufficient_{nucleus.lower()}_{split}",
                        f"{split} requires at least {minimum_nucleus} "
                        f"{nucleus} spectra; found {observed}",
                        count=observed,
                    )
                )
        minimum_paired = int(effective_policy.minimum_paired_molecules.get(split, 0))
        if paired_molecules[split] < minimum_paired:
            reasons.append(
                _reason(
                    f"insufficient_paired_molecules_{split}",
                    f"{split} requires at least {minimum_paired} molecules "
                    "with both 1H and 13C spectra",
                    count=paired_molecules[split],
                )
            )

    leakage_fields = (
        "molecule_key",
        "inchi_key",
        "scaffold_key",
        "source_document_key",
        "record_sha256",
        "spectrum_fingerprint_sha256",
    )
    leakage = {
        field_name: _group_leakage(records, field_name) for field_name in leakage_fields
    }
    for field_name, audit in leakage.items():
        if audit["leaked_groups"]:
            reasons.append(
                _reason(
                    f"{field_name}_split_leakage",
                    f"{field_name} must not occur in more than one split",
                    count=audit["leaked_groups"],
                    examples=audit["leak_examples"],
                )
            )

    releases = Counter(
        str(row.get("dataset_release_id"))
        for row in records
        if _nonempty(row.get("dataset_release_id"))
    )
    sources = Counter(
        str(row.get("source_name"))
        for row in records
        if _nonempty(row.get("source_name"))
    )
    report = {
        "schema_version": "chemapp.nmr.calibration-admission-report.v1",
        "pipeline_version": TRAINING_PIPELINE_VERSION,
        "stage": "gold_manifest",
        "status": "passed" if not reasons else "blocked",
        "training_allowed": not reasons,
        "probability_claim_allowed": False,
        "records": len(records),
        "manifest_content_sha256": canonical_jsonl_sha256(records),
        "counts": {
            "records": record_counts,
            "molecules": molecule_counts,
            "scaffolds": scaffold_counts,
            "source_documents": source_document_counts,
            "nuclei": nucleus_counts,
            "paired_molecules": paired_molecules,
            "dataset_releases": dict(sorted(releases.items())),
            "source_names": dict(sorted(sources.items())),
        },
        "manual_review": {
            "protocol_version": REVIEW_PROTOCOL_VERSION,
            "required_reviews_by_split": dict(effective_policy.required_human_reviews),
            "failure_counts": dict(sorted(review_failure_counts.items())),
            "failed_record_examples": review_failure_examples,
        },
        "structure_identity": {
            "recomputed_with_rdkit": Chem is not None,
            "failure_counts": dict(sorted(identity_failures.items())),
            "failed_record_examples": identity_failure_examples,
        },
        "source_provenance": {
            "failure_counts": dict(sorted(provenance_failures.items())),
            "dataset_release_is_not_a_split_group": True,
        },
        "overlap_scope": {
            "chemapp_base_index": {
                "status": (
                    "checked_no_overlap"
                    if len(reference_index_hashes) == 1 and not provenance_failures
                    else "invalid_or_incomplete"
                ),
                "reference_index_sha256": (
                    next(iter(reference_index_hashes))
                    if len(reference_index_hashes) == 1
                    else None
                ),
            },
            "dp5q_upstream_training_corpus": {
                "status": "unknown",
                "interpretation": (
                    "The Gold non-overlap proof covers the frozen ChemApp "
                    "nmrshiftdb2 base index, not DP5q-CASCADE's upstream "
                    "structure-level training corpus."
                ),
            },
        },
        "leakage": leakage,
        "blocking_reasons": reasons,
        "policy": asdict(selected_policy),
        "interpretation": (
            "Passing this report permits calibrator fitting only. It does not "
            "permit a probability claim until the frozen-test calibration gate "
            "also passes."
        ),
    }
    return report


def _manifest_lookup(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    lookup: dict[str, Mapping[str, Any]] = {}
    for row in records:
        record_id = str(row.get("record_id") or "")
        if record_id and record_id not in lookup:
            lookup[record_id] = row
    return lookup


def audit_calibration_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    manifest: Sequence[Mapping[str, Any]],
    ranker_artifact_sha256: str,
    ranker_artifact: Mapping[str, Any] | None = None,
    policy: CalibrationGatePolicy | None = None,
) -> dict[str, Any]:
    """Bind score examples to every calibration/test spectrum exactly once."""

    selected_policy = policy or CalibrationGatePolicy()
    reasons: list[dict[str, Any]] = []
    lookup = _manifest_lookup(manifest)
    expected = {
        split: {
            str(row["record_id"])
            for row in manifest
            if row.get("split") == split and _nonempty(row.get("record_id"))
        }
        for split in CALIBRATOR_SPLITS
    }
    seen_case_ids: set[str] = set()
    seen_records: dict[str, set[str]] = {split: set() for split in CALIBRATOR_SPLITS}
    invalid: Counter[str] = Counter()
    invalid_examples: list[dict[str, Any]] = []
    semantics: set[str] = set()
    score_rows: dict[str, list[tuple[float, int]]] = {
        split: [] for split in CALIBRATOR_SPLITS
    }
    manifest_sha256 = canonical_jsonl_sha256(manifest)
    gold_reference_hashes = {
        str(overlap.get("reference_index_sha256"))
        for row in manifest
        if isinstance(
            (overlap := row.get("ranker_training_overlap")),
            Mapping,
        )
        and _is_sha256(overlap.get("reference_index_sha256"))
    }

    if not _is_sha256(ranker_artifact_sha256):
        reasons.append(
            _reason(
                "invalid_ranker_artifact_hash",
                "ranker artifact must be bound by a lowercase SHA-256 digest",
            )
        )
    artifact_base_index_sha256 = None
    if ranker_artifact is not None:
        artifact_base_index_sha256 = ranker_artifact.get("base_index_sha256")
        gold_reference = ranker_artifact.get("gold_reference")
        if (
            ranker_artifact.get("schema_version") != FROZEN_RANKER_SCHEMA_VERSION
            or not _is_sha256(artifact_base_index_sha256)
            or len(gold_reference_hashes) != 1
            or artifact_base_index_sha256 != next(iter(gold_reference_hashes))
            or not isinstance(gold_reference, Mapping)
            or gold_reference.get("sha256") != artifact_base_index_sha256
            or gold_reference.get("scope") != "chemapp_frozen_nmrshiftdb2_base_index"
        ):
            reasons.append(
                _reason(
                    "ranker_gold_reference_mismatch",
                    "ranker base index must equal the one global Gold overlap reference",
                )
            )
        _, invalid_preflight_bindings = _ranker_conformer_preflight_bindings(
            ranker_artifact
        )
        if invalid_preflight_bindings:
            reasons.append(
                _reason(
                    "ranker_conformer_preflight_binding_invalid",
                    "ranker artifact must freeze the conformer-preflight "
                    "implementation, runtime and policy",
                    examples=invalid_preflight_bindings,
                )
            )

    for index, example in enumerate(examples):
        row_failures: list[str] = []
        case_id = str(example.get("case_id") or "")
        split = example.get("split")
        record_ids = example.get("record_ids")
        raw_score = example.get("raw_score")
        outcome = example.get("top1_exact_correct")
        score_semantics = str(example.get("score_semantics") or "")
        if not case_id:
            row_failures.append("case_id_missing")
        elif case_id in seen_case_ids:
            row_failures.append("duplicate_case_id")
        else:
            seen_case_ids.add(case_id)
        if example.get("schema_version") != CALIBRATION_EXAMPLES_SCHEMA_VERSION:
            row_failures.append("schema_version_mismatch")
        if split not in CALIBRATOR_SPLITS:
            row_failures.append("invalid_split")
        if (
            isinstance(raw_score, bool)
            or not isinstance(raw_score, (int, float))
            or not math.isfinite(float(raw_score))
        ):
            row_failures.append("invalid_raw_score")
        if type(outcome) is not bool:
            row_failures.append("invalid_top1_outcome")
        if not score_semantics:
            row_failures.append("score_semantics_missing")
        else:
            semantics.add(score_semantics)
        if example.get("ranker_artifact_sha256") != ranker_artifact_sha256:
            row_failures.append("ranker_hash_mismatch")
        if example.get("gold_manifest_content_sha256") != manifest_sha256:
            row_failures.append("gold_manifest_hash_mismatch")
        if len(gold_reference_hashes) != 1 or example.get("base_index_sha256") != next(
            iter(gold_reference_hashes), None
        ):
            row_failures.append("base_index_gold_reference_mismatch")
        if (
            artifact_base_index_sha256 is not None
            and example.get("base_index_sha256") != artifact_base_index_sha256
        ):
            row_failures.append("base_index_ranker_artifact_mismatch")
        if not _is_sha256(example.get("frozen_case_content_sha256")):
            row_failures.append("frozen_case_hash_invalid")
        artifact_source_release = (
            ranker_artifact.get("source_release_binding")
            if isinstance(ranker_artifact, Mapping)
            else None
        )
        for field_name, artifact_field in (
            ("source_release_id", "release_id"),
            ("source_records_sha256", "records_sha256"),
            ("source_current_sha256", "source_current_sha256"),
        ):
            if not _is_sha256(example.get(field_name)):
                row_failures.append(f"{field_name}_invalid")
            elif isinstance(artifact_source_release, Mapping) and example.get(
                field_name
            ) != artifact_source_release.get(artifact_field):
                row_failures.append(f"{field_name}_ranker_artifact_mismatch")
        if (
            not isinstance(record_ids, list)
            or not record_ids
            or any(not _nonempty(value) for value in record_ids)
            or len(set(record_ids)) != len(record_ids)
        ):
            row_failures.append("invalid_record_ids")
            bound_records: list[Mapping[str, Any]] = []
        else:
            bound_records = [
                lookup[str(record_id)]
                for record_id in record_ids
                if str(record_id) in lookup
            ]
            if len(bound_records) != len(record_ids):
                row_failures.append("record_not_in_manifest")
            elif any(row.get("split") != split for row in bound_records):
                row_failures.append("record_split_mismatch")
            else:
                for field_name in (
                    "molecule_key",
                    "inchi_key",
                    "scaffold_key",
                    "source_document_key",
                ):
                    if len({row.get(field_name) for row in bound_records}) != 1:
                        row_failures.append(f"case_mixes_{field_name}")
                if split in CALIBRATOR_SPLITS:
                    overlap = seen_records[str(split)].intersection(
                        str(value) for value in record_ids
                    )
                    if overlap:
                        row_failures.append("record_used_by_multiple_cases")
                    seen_records[str(split)].update(str(value) for value in record_ids)
                if len(bound_records) != 1:
                    row_failures.append("case_must_bind_one_spectrum")
                else:
                    gold = bound_records[0]
                    expected_gold_binding = canonical_sha256(
                        {
                            "record_id": gold["record_id"],
                            "record_sha256": gold["record_sha256"],
                            "spectrum_fingerprint_sha256": gold[
                                "spectrum_fingerprint_sha256"
                            ],
                            "inchi_key": gold["inchi_key"],
                            "molecule_key": gold["molecule_key"],
                            "scaffold_key": gold["scaffold_key"],
                            "source_document_key": gold["source_document_key"],
                            "split": split,
                        }
                    )
                    if (
                        example.get("gold_record_binding_sha256")
                        != expected_gold_binding
                    ):
                        row_failures.append("gold_record_binding_mismatch")
        if row_failures:
            for failure in set(row_failures):
                invalid[failure] += 1
            if len(invalid_examples) < 10:
                invalid_examples.append(
                    {
                        "row": index + 1,
                        "case_id": case_id,
                        "failures": sorted(set(row_failures)),
                    }
                )
        elif split in CALIBRATOR_SPLITS:
            score_rows[str(split)].append((float(raw_score), int(outcome)))

    if not examples:
        reasons.append(_reason("calibration_examples_empty", "score cases are empty"))
    if invalid:
        reasons.append(
            _reason(
                "invalid_calibration_examples",
                "score cases are not completely bound to the frozen ranker/manifest",
                count=sum(invalid.values()),
                examples=invalid_examples,
            )
        )
    if len(semantics) != 1:
        reasons.append(
            _reason(
                "mixed_score_semantics",
                "all score cases must use one frozen score definition",
                count=len(semantics),
                examples=sorted(semantics),
            )
        )

    coverage: dict[str, Any] = {}
    for split in CALIBRATOR_SPLITS:
        omitted = sorted(expected[split] - seen_records[split])
        unexpected = sorted(seen_records[split] - expected[split])
        coverage[split] = {
            "expected_records": len(expected[split]),
            "seen_records": len(seen_records[split]),
            "omitted_count": len(omitted),
            "unexpected_count": len(unexpected),
            "omitted_examples": omitted[:10],
            "unexpected_examples": unexpected[:10],
        }
        if omitted or unexpected:
            reasons.append(
                _reason(
                    f"incomplete_{split}_record_coverage",
                    f"{split} score cases must cover every gold spectrum exactly once",
                    count=len(omitted) + len(unexpected),
                    examples=[*omitted[:5], *unexpected[:5]],
                )
            )

    cases_by_split = {split: len(score_rows[split]) for split in CALIBRATOR_SPLITS}
    required_cases = {
        "calibration": selected_policy.minimum_calibration_cases,
        "test": selected_policy.minimum_test_cases,
    }
    class_counts: dict[str, dict[str, int]] = {}
    unique_scores: dict[str, int] = {}
    for split in CALIBRATOR_SPLITS:
        if cases_by_split[split] < required_cases[split]:
            reasons.append(
                _reason(
                    f"insufficient_{split}_cases",
                    f"{split} requires at least {required_cases[split]} cases; "
                    f"found {cases_by_split[split]}",
                    count=cases_by_split[split],
                )
            )
        outcomes = [outcome for _, outcome in score_rows[split]]
        counts = Counter(outcomes)
        class_counts[split] = {
            "incorrect": counts.get(0, 0),
            "correct": counts.get(1, 0),
        }
        for class_name, count in class_counts[split].items():
            if count < selected_policy.minimum_outcomes_per_class:
                reasons.append(
                    _reason(
                        f"insufficient_{split}_{class_name}_outcomes",
                        f"{split} requires at least "
                        f"{selected_policy.minimum_outcomes_per_class} "
                        f"{class_name} cases; found {count}",
                        count=count,
                    )
                )
        unique_scores[split] = len({score for score, _ in score_rows[split]})
        if unique_scores[split] < selected_policy.minimum_unique_scores:
            reasons.append(
                _reason(
                    f"insufficient_unique_scores_{split}",
                    f"{split} requires at least "
                    f"{selected_policy.minimum_unique_scores} distinct raw scores; "
                    f"found {unique_scores[split]}",
                    count=unique_scores[split],
                )
            )

    return {
        "schema_version": "chemapp.nmr.calibration-score-audit.v1",
        "pipeline_version": TRAINING_PIPELINE_VERSION,
        "stage": "score_examples",
        "status": "passed" if not reasons else "blocked",
        "training_allowed": not reasons,
        "probability_claim_allowed": False,
        "examples": len(examples),
        "examples_content_sha256": canonical_jsonl_sha256(examples),
        "ranker_artifact_sha256": ranker_artifact_sha256,
        "base_index_sha256": (
            next(iter(gold_reference_hashes))
            if len(gold_reference_hashes) == 1
            else None
        ),
        "gold_manifest_content_sha256": manifest_sha256,
        "score_semantics": next(iter(semantics)) if len(semantics) == 1 else None,
        "conformer_preflight_bindings": (
            _ranker_conformer_preflight_bindings(ranker_artifact)[0]
            if isinstance(ranker_artifact, Mapping)
            else None
        ),
        "cases_by_split": cases_by_split,
        "class_counts": class_counts,
        "unique_scores": unique_scores,
        "record_coverage": coverage,
        "blocking_reasons": reasons,
        "interpretation": (
            "Only calibration rows may be passed to fit. Test outcomes are "
            "reserved for one frozen post-fit evaluation."
        ),
    }


def admission_report(
    manifest: Sequence[Mapping[str, Any]],
    *,
    examples: Sequence[Mapping[str, Any]] | None = None,
    ranker_artifact_sha256: str | None = None,
    ranker_artifact: Mapping[str, Any] | None = None,
    policy: CalibrationGatePolicy | None = None,
) -> dict[str, Any]:
    selected_policy = policy or CalibrationGatePolicy()
    manifest_report = audit_gold_manifest(manifest, policy=selected_policy)
    examples_report = None
    reasons = list(manifest_report["blocking_reasons"])
    if examples is not None:
        examples_report = audit_calibration_examples(
            examples,
            manifest=manifest,
            ranker_artifact_sha256=str(ranker_artifact_sha256 or ""),
            ranker_artifact=ranker_artifact,
            policy=selected_policy,
        )
        reasons.extend(examples_report["blocking_reasons"])
    status = (
        "passed"
        if not reasons and examples is not None
        else "passed_manifest_only"
        if not reasons
        else "blocked"
    )
    pending_checks: list[dict[str, Any]] = []
    if examples is None:
        pending_checks.append(
            _reason(
                "score_examples_not_audited",
                "manifest-only audit cannot authorize calibrator fitting",
            )
        )
    policy_invalid = _policy_validation_reasons(selected_policy)
    policy_weaknesses = (
        [] if policy_invalid else production_policy_weaknesses(selected_policy)
    )
    upstream_overlap = (
        ranker_artifact.get("upstream_training_overlap")
        if isinstance(ranker_artifact, Mapping)
        else None
    )
    upstream_overlap_status = (
        str(upstream_overlap.get("status") or "unknown")
        if isinstance(upstream_overlap, Mapping)
        else "unknown"
    )
    return {
        "schema_version": "chemapp.nmr.calibration-training-admission.v1",
        "pipeline_version": TRAINING_PIPELINE_VERSION,
        "status": status,
        "training_started": False,
        "training_allowed": status == "passed",
        "probability_claim_allowed": False,
        "gold_manifest": manifest_report,
        "score_examples": examples_report,
        "blocking_reasons": reasons,
        "pending_checks": pending_checks,
        "policy": asdict(selected_policy),
        "ranker_overlap_scope": {
            "chemapp_base_index": manifest_report["overlap_scope"][
                "chemapp_base_index"
            ],
            "dp5q_upstream_training_corpus": {
                "status": upstream_overlap_status,
                "production_independence_claim_allowed": False,
            },
        },
        "production_policy": {
            "compatible": not policy_invalid and not policy_weaknesses,
            "weaknesses": policy_weaknesses,
            "invalid": policy_invalid,
        },
    }


def _sigmoid(value: np.ndarray) -> np.ndarray:
    positive = value >= 0
    result = np.empty_like(value, dtype=float)
    result[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exp_value = np.exp(value[~positive])
    result[~positive] = exp_value / (1.0 + exp_value)
    return result


def fit_calibrator(
    raw_scores: Sequence[float],
    outcomes: Sequence[int],
    *,
    method: str = "sigmoid",
) -> dict[str, Any]:
    """Fit a JSON-serializable calibrator on the calibration split only."""

    x = np.asarray(raw_scores, dtype=float)
    y = np.asarray(outcomes, dtype=int)
    if x.ndim != 1 or y.ndim != 1 or len(x) != len(y) or not len(x):
        raise ValueError("calibrator requires aligned, non-empty score/outcome vectors")
    if not np.isfinite(x).all() or not set(y.tolist()).issubset({0, 1}):
        raise ValueError("calibrator inputs must be finite scores and binary outcomes")
    if len(set(y.tolist())) != 2:
        raise ValueError("calibrator requires both correctness outcome classes")
    if method == "sigmoid":
        center = float(np.mean(x))
        scale = float(np.std(x))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("sigmoid calibration requires non-constant scores")
        normalized = ((x - center) / scale).reshape(-1, 1)
        estimator = LogisticRegression(
            C=np.inf,
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
        )
        estimator.fit(normalized, y)
        coefficient = float(estimator.coef_[0, 0])
        intercept = float(estimator.intercept_[0])
        if (
            not math.isfinite(coefficient)
            or not math.isfinite(intercept)
            or coefficient <= 0.0
        ):
            raise ValueError(
                "sigmoid calibrator violates higher-is-more-confident direction"
            )
        artifact = {
            "schema_version": CALIBRATOR_SCHEMA_VERSION,
            "method": "sigmoid",
            "target": "top1_exact_structure_correct",
            "score_direction": "higher_is_more_confident",
            "deployment_status": "unassessed_diagnostic",
            "probability_claim_allowed": False,
            "parameters": {
                "score_center": center,
                "score_scale": scale,
                "coefficient": coefficient,
                "intercept": intercept,
            },
        }
    elif method == "isotonic":
        estimator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            increasing=True,
            out_of_bounds="clip",
        )
        estimator.fit(x, y)
        x_thresholds = [float(value) for value in estimator.X_thresholds_]
        y_thresholds = [float(value) for value in estimator.y_thresholds_]
        if (
            len(x_thresholds) < 2
            or any(not math.isfinite(value) for value in [*x_thresholds, *y_thresholds])
            or any(right <= left for left, right in zip(x_thresholds, x_thresholds[1:]))
            or any(not 0.0 <= value <= 1.0 for value in y_thresholds)
            or any(right < left for left, right in zip(y_thresholds, y_thresholds[1:]))
        ):
            raise ValueError("fitted isotonic calibrator is not valid monotonic [0,1]")
        artifact = {
            "schema_version": CALIBRATOR_SCHEMA_VERSION,
            "method": "isotonic",
            "target": "top1_exact_structure_correct",
            "score_direction": "higher_is_more_confident",
            "deployment_status": "unassessed_diagnostic",
            "probability_claim_allowed": False,
            "parameters": {
                "x_thresholds": x_thresholds,
                "y_thresholds": y_thresholds,
            },
        }
    else:
        raise ValueError("calibration method must be sigmoid or isotonic")
    return {
        **artifact,
        "artifact_sha256": canonical_sha256(artifact),
    }


def predict_calibrated(
    calibrator: Mapping[str, Any],
    raw_scores: Sequence[float],
    *,
    allow_diagnostic: bool = False,
) -> list[float]:
    if calibrator.get("schema_version") != CALIBRATOR_SCHEMA_VERSION:
        raise ValueError("unsupported calibrator schema")
    claimed_hash = calibrator.get("artifact_sha256")
    artifact_core = {
        key: value for key, value in calibrator.items() if key != "artifact_sha256"
    }
    if not _is_sha256(claimed_hash) or claimed_hash != canonical_sha256(artifact_core):
        raise ValueError("calibrator artifact hash mismatch")
    deployment_status = calibrator.get("deployment_status")
    probability_claim_allowed = calibrator.get("probability_claim_allowed")
    if (
        not (
            deployment_status == "deployment_eligible"
            and probability_claim_allowed is True
        )
        and not allow_diagnostic
    ):
        raise ValueError(
            "diagnostic calibrator prediction requires allow_diagnostic=True"
        )
    x = np.asarray(raw_scores, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("raw scores must be a finite one-dimensional vector")
    parameters = calibrator.get("parameters")
    if not isinstance(parameters, Mapping):
        raise ValueError("calibrator parameters are missing")
    method = calibrator.get("method")
    if method == "sigmoid":
        center = float(parameters["score_center"])
        scale = float(parameters["score_scale"])
        coefficient = float(parameters["coefficient"])
        intercept = float(parameters["intercept"])
        if (
            not all(
                math.isfinite(value)
                for value in (center, scale, coefficient, intercept)
            )
            or scale <= 0
            or coefficient <= 0
        ):
            raise ValueError(
                "sigmoid parameters must be finite with positive scale/direction"
            )
        probabilities = _sigmoid(coefficient * ((x - center) / scale) + intercept)
    elif method == "isotonic":
        thresholds = np.asarray(parameters["x_thresholds"], dtype=float)
        values = np.asarray(parameters["y_thresholds"], dtype=float)
        if (
            thresholds.ndim != 1
            or values.ndim != 1
            or len(thresholds) != len(values)
            or not len(thresholds)
            or not np.isfinite(thresholds).all()
            or not np.isfinite(values).all()
            or len(thresholds) < 2
            or np.any(np.diff(thresholds) <= 0)
            or np.any(np.diff(values) < 0)
            or np.any(values < 0.0)
            or np.any(values > 1.0)
        ):
            raise ValueError("invalid isotonic calibrator thresholds")
        probabilities = np.interp(x, thresholds, values)
    else:
        raise ValueError("unsupported calibrator method")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError("calibrator produced values outside finite [0,1]")
    return [float(value) for value in probabilities]


def reliability_metrics(
    probabilities: Sequence[float],
    outcomes: Sequence[int],
    *,
    bins: int = 10,
) -> dict[str, Any]:
    """Return fixed-width reliability-diagram data and calibration metrics."""

    probability_array = np.asarray(probabilities, dtype=float)
    outcome_array = np.asarray(outcomes, dtype=int)
    if (
        probability_array.ndim != 1
        or outcome_array.ndim != 1
        or len(probability_array) != len(outcome_array)
        or not len(probability_array)
        or bins < 2
    ):
        raise ValueError("reliability metrics require aligned non-empty vectors")
    if (
        not np.isfinite(probability_array).all()
        or np.any(probability_array < 0)
        or np.any(probability_array > 1)
        or not set(outcome_array.tolist()).issubset({0, 1})
    ):
        raise ValueError("probabilities/outcomes are outside their valid domains")

    rows: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        if index == bins - 1:
            mask = (probability_array >= lower) & (probability_array <= upper)
        else:
            mask = (probability_array >= lower) & (probability_array < upper)
        count = int(np.sum(mask))
        if count:
            mean_probability = float(np.mean(probability_array[mask]))
            empirical_accuracy = float(np.mean(outcome_array[mask]))
            gap = abs(mean_probability - empirical_accuracy)
            ece += (count / len(probability_array)) * gap
            mce = max(mce, gap)
        else:
            mean_probability = None
            empirical_accuracy = None
            gap = None
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": count,
                "mean_probability": mean_probability,
                "empirical_accuracy": empirical_accuracy,
                "absolute_gap": gap,
            }
        )

    epsilon = 1e-12
    clipped = np.clip(probability_array, epsilon, 1 - epsilon)
    brier = float(np.mean((probability_array - outcome_array) ** 2))
    log_loss = float(
        -np.mean(
            outcome_array * np.log(clipped) + (1 - outcome_array) * np.log(1 - clipped)
        )
    )
    prevalence = float(np.mean(outcome_array))
    null_brier = prevalence * (1.0 - prevalence)
    brier_skill = float(1.0 - brier / null_brier) if null_brier > 0 else None
    roc_auc = (
        float(roc_auc_score(outcome_array, probability_array))
        if len(set(outcome_array.tolist())) == 2
        else None
    )
    average_precision = (
        float(average_precision_score(outcome_array, probability_array))
        if len(set(outcome_array.tolist())) == 2
        else None
    )

    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    calibration_intercept = None
    calibration_slope = None
    if len(set(outcome_array.tolist())) == 2 and float(np.std(logits)) > 0:
        calibration_model = LogisticRegression(
            C=np.inf,
            solver="lbfgs",
            max_iter=2000,
            random_state=0,
        )
        calibration_model.fit(logits, outcome_array)
        calibration_intercept = float(calibration_model.intercept_[0])
        calibration_slope = float(calibration_model.coef_[0, 0])

    return {
        "cases": len(probability_array),
        "event_rate": prevalence,
        "mean_probability": float(np.mean(probability_array)),
        "calibration_in_the_large": float(np.mean(probability_array) - prevalence),
        "brier": brier,
        "null_brier": null_brier,
        "brier_skill_score": brier_skill,
        "log_loss": log_loss,
        "ece": float(ece),
        "mce": float(mce),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "calibration_intercept": calibration_intercept,
        "calibration_slope": calibration_slope,
        "populated_bins": sum(row["count"] > 0 for row in rows),
        "reliability_bins": rows,
    }


def reliability_svg(metrics: Mapping[str, Any], *, title: str) -> str:
    """Render a dependency-free, deterministic SVG reliability diagram."""

    width = 640
    height = 520
    left = 70
    top = 55
    plot = 400
    elements = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            f'<text x="{left}" y="28" font-family="sans-serif" '
            f'font-size="18">{_xml_escape(title)}</text>'
        ),
        (
            f'<line x1="{left}" y1="{top + plot}" x2="{left + plot}" '
            f'y2="{top}" stroke="#999" stroke-dasharray="5 5"/>'
        ),
        (
            f'<rect x="{left}" y="{top}" width="{plot}" height="{plot}" '
            'fill="none" stroke="#222"/>'
        ),
    ]
    reliability_bins = metrics.get("reliability_bins")
    if isinstance(reliability_bins, Sequence):
        for row in reliability_bins:
            if not isinstance(row, Mapping) or not row.get("count"):
                continue
            probability = float(row["mean_probability"])
            accuracy = float(row["empirical_accuracy"])
            x = left + probability * plot
            y = top + (1.0 - accuracy) * plot
            radius = min(11.0, 3.0 + math.sqrt(float(row["count"])))
            elements.append(
                f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{radius:.3f}" '
                'fill="#2563eb" fill-opacity="0.75" stroke="#1e3a8a"/>'
            )
    for index in range(6):
        fraction = index / 5
        x = left + fraction * plot
        y = top + (1.0 - fraction) * plot
        elements.extend(
            [
                (
                    f'<text x="{x:.2f}" y="{top + plot + 24}" '
                    'font-family="sans-serif" font-size="12" '
                    f'text-anchor="middle">{fraction:.1f}</text>'
                ),
                (
                    f'<text x="{left - 12}" y="{y + 4:.2f}" '
                    'font-family="sans-serif" font-size="12" '
                    f'text-anchor="end">{fraction:.1f}</text>'
                ),
            ]
        )
    elements.extend(
        [
            (
                f'<text x="{left + plot / 2}" y="{height - 22}" '
                'font-family="sans-serif" font-size="14" '
                'text-anchor="middle">Predicted probability</text>'
            ),
            (
                f'<text x="20" y="{top + plot / 2}" '
                'font-family="sans-serif" font-size="14" '
                'text-anchor="middle" '
                f'transform="rotate(-90 20 {top + plot / 2})">'
                "Observed exact-structure accuracy</text>"
            ),
            "</svg>",
        ]
    )
    return "\n".join(elements)


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _examples_for_split(
    examples: Sequence[Mapping[str, Any]],
    split: str,
) -> tuple[list[float], list[int]]:
    rows = [row for row in examples if row.get("split") == split]
    return (
        [float(row["raw_score"]) for row in rows],
        [int(row["top1_exact_correct"]) for row in rows],
    )


def _performance_gate(
    test_metrics: Mapping[str, Any],
    *,
    policy: CalibrationGatePolicy,
) -> list[dict[str, Any]]:
    checks = (
        (
            "test_ece_too_high",
            float(test_metrics["ece"]) <= policy.maximum_test_ece,
            f"test ECE must be <= {policy.maximum_test_ece}",
        ),
        (
            "test_brier_too_high",
            float(test_metrics["brier"]) <= policy.maximum_test_brier,
            f"test Brier score must be <= {policy.maximum_test_brier}",
        ),
        (
            "test_brier_skill_too_low",
            test_metrics["brier_skill_score"] is not None
            and float(test_metrics["brier_skill_score"])
            >= policy.minimum_test_brier_skill,
            "test Brier skill must beat the frozen prevalence baseline",
        ),
        (
            "test_roc_auc_too_low",
            test_metrics["roc_auc"] is not None
            and float(test_metrics["roc_auc"]) >= policy.minimum_test_roc_auc,
            f"test ROC AUC must be >= {policy.minimum_test_roc_auc}",
        ),
        (
            "too_few_populated_reliability_bins",
            int(test_metrics["populated_bins"])
            >= policy.minimum_populated_reliability_bins,
            "test reliability diagram has insufficient populated bins",
        ),
        (
            "calibration_intercept_out_of_range",
            test_metrics["calibration_intercept"] is not None
            and abs(float(test_metrics["calibration_intercept"]))
            <= policy.maximum_absolute_calibration_intercept,
            "absolute test calibration intercept is too large",
        ),
        (
            "calibration_slope_out_of_range",
            test_metrics["calibration_slope"] is not None
            and policy.minimum_calibration_slope
            <= float(test_metrics["calibration_slope"])
            <= policy.maximum_calibration_slope,
            "test calibration slope is outside the pre-registered range",
        ),
    )
    return [_reason(code, message) for code, passed, message in checks if not passed]


def _environment_inventory() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def run_calibration_training(
    manifest: Sequence[Mapping[str, Any]],
    examples: Sequence[Mapping[str, Any]],
    *,
    ranker_artifact_sha256: str,
    ranker_artifact: Mapping[str, Any] | None = None,
    method: str = "sigmoid",
    policy: CalibrationGatePolicy | None = None,
    created_at: str,
    implementation_sha256: str,
    fit_function: Callable[..., dict[str, Any]] = fit_calibrator,
    research_only: bool = False,
    data_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run admission, calibration-only fitting and frozen-test claim gating."""

    selected_policy = policy or CalibrationGatePolicy()
    if _parse_utc(created_at) is None:
        raise ValueError("created_at must be an explicit UTC ISO-8601 timestamp")
    if not _is_sha256(implementation_sha256):
        raise ValueError("implementation_sha256 must be a lowercase SHA-256 digest")
    admission = admission_report(
        manifest,
        examples=examples,
        ranker_artifact_sha256=ranker_artifact_sha256,
        ranker_artifact=ranker_artifact,
        policy=selected_policy,
    )
    if not admission["training_allowed"]:
        raise CalibrationAdmissionError(admission)
    policy_weaknesses = production_policy_weaknesses(selected_policy)
    if policy_weaknesses and not research_only:
        admission["status"] = "blocked"
        admission["training_allowed"] = False
        admission["blocking_reasons"].append(
            _reason(
                "policy_below_production_floor",
                "a weaker policy requires an explicit research-only run, "
                "which can never authorize probability claims",
                count=len(policy_weaknesses),
                examples=policy_weaknesses,
            )
        )
    if not admission["training_allowed"]:
        raise CalibrationAdmissionError(admission)
    upstream_overlap = (
        ranker_artifact.get("upstream_training_overlap")
        if isinstance(ranker_artifact, Mapping)
        else None
    )
    upstream_overlap_status = (
        str(upstream_overlap.get("status") or "unknown")
        if isinstance(upstream_overlap, Mapping)
        else "unknown"
    )
    if upstream_overlap_status != "checked_no_overlap" and not research_only:
        admission["status"] = "blocked"
        admission["training_allowed"] = False
        admission["blocking_reasons"].append(
            _reason(
                "dp5q_upstream_training_overlap_unknown",
                "Gold non-overlap was proved only against the frozen ChemApp "
                "nmrshiftdb2 base index; DP5q-CASCADE upstream structure-level "
                "training overlap is not auditable, so only research-only "
                "diagnostic fitting is permitted",
            )
        )
        raise CalibrationAdmissionError(admission)

    calibration_scores, calibration_outcomes = _examples_for_split(
        examples,
        "calibration",
    )
    test_scores, test_outcomes = _examples_for_split(examples, "test")
    fitted_calibrator = fit_function(
        calibration_scores,
        calibration_outcomes,
        method=method,
    )
    calibration_probabilities = predict_calibrated(
        fitted_calibrator,
        calibration_scores,
        allow_diagnostic=True,
    )
    test_probabilities = predict_calibrated(
        fitted_calibrator,
        test_scores,
        allow_diagnostic=True,
    )
    calibration_metrics = reliability_metrics(
        calibration_probabilities,
        calibration_outcomes,
        bins=selected_policy.reliability_bins,
    )
    test_metrics = reliability_metrics(
        test_probabilities,
        test_outcomes,
        bins=selected_policy.reliability_bins,
    )
    performance_reasons = _performance_gate(test_metrics, policy=selected_policy)
    probability_allowed = not performance_reasons and not research_only
    if research_only:
        performance_reasons.append(
            _reason(
                "research_only_run",
                "research-only runs are permanently ineligible for probability claims",
            )
        )
    calibrator_core = {
        key: value
        for key, value in fitted_calibrator.items()
        if key != "artifact_sha256"
    }
    calibrator_core["deployment_status"] = (
        "deployment_eligible" if probability_allowed else "diagnostic_rejected"
    )
    calibrator_core["probability_claim_allowed"] = probability_allowed
    calibrator = {
        **calibrator_core,
        "artifact_sha256": canonical_sha256(calibrator_core),
    }
    model_card_core = {
        "schema_version": MODEL_CARD_SCHEMA_VERSION,
        "pipeline_version": TRAINING_PIPELINE_VERSION,
        "created_at": created_at,
        "status": (
            "deployment_eligible" if probability_allowed else "diagnostic_rejected"
        ),
        "probability_claim_allowed": probability_allowed,
        "target_definition": (
            "Probability that the frozen ranker's top-ranked candidate is the "
            "exact gold structure for one independently reviewed spectrum case."
        ),
        "scope": {
            "supported_nuclei": list(SUPPORTED_NUCLEI),
            "score_semantics": admission["score_examples"]["score_semantics"],
            "calibration_method": method,
            "ranker_is_frozen": True,
            "test_used_for_fitting_or_selection": False,
            "research_only": research_only,
            "overlap_scope": {
                "chemapp_base_index": {
                    "status": "checked_no_overlap",
                    "sha256": admission["score_examples"]["base_index_sha256"],
                },
                "dp5q_upstream_training_corpus": {
                    "status": upstream_overlap_status,
                    "independence_claim_allowed": False,
                },
            },
        },
        "data_bindings": {
            "gold_manifest_content_sha256": admission["gold_manifest"][
                "manifest_content_sha256"
            ],
            "score_examples_content_sha256": admission["score_examples"][
                "examples_content_sha256"
            ],
            "ranker_artifact_sha256": ranker_artifact_sha256,
            "implementation_sha256": implementation_sha256,
            "policy_sha256": canonical_sha256(asdict(selected_policy)),
            "data_lineage_sha256": (
                canonical_sha256(dict(data_lineage))
                if data_lineage is not None
                else None
            ),
        },
        "split_roles": {
            "train": "upstream_ranker_fit_only",
            "validation": "upstream_ranker_selection_only",
            "calibration": "calibrator_fit_only",
            "test": "single_frozen_probability_claim_evaluation",
        },
        "admission_summary": {
            "status": admission["status"],
            "record_counts": admission["gold_manifest"]["counts"]["records"],
            "molecule_counts": admission["gold_manifest"]["counts"]["molecules"],
            "scaffold_counts": admission["gold_manifest"]["counts"]["scaffolds"],
            "source_document_counts": admission["gold_manifest"]["counts"][
                "source_documents"
            ],
            "case_counts": admission["score_examples"]["cases_by_split"],
            "class_counts": admission["score_examples"]["class_counts"],
            "leakage": admission["gold_manifest"]["leakage"],
        },
        "calibrator": {
            "schema_version": calibrator["schema_version"],
            "method": calibrator["method"],
            "artifact_sha256": calibrator["artifact_sha256"],
        },
        "metrics": {
            "calibration": calibration_metrics,
            "frozen_test": test_metrics,
        },
        "claim_gate": {
            "status": "passed" if probability_allowed else "blocked",
            "blocking_reasons": performance_reasons,
            "policy": asdict(selected_policy),
        },
        "limitations": [
            "This calibrates only one frozen scalar score and one exact-Top-1 target.",
            "It does not establish candidate-generation coverage or open-world recall.",
            "A new ranker, score definition, data release, split, or review change invalidates this artifact.",
            "Performance outside the reviewed nuclei, chemistry and acquisition domains is unknown.",
            (
                "Gold non-overlap is established only for the frozen ChemApp "
                "nmrshiftdb2 base index. DP5q-CASCADE upstream training-set "
                "structure overlap remains unknown and has not been audited."
            ),
        ],
        "environment": _environment_inventory(),
        "data_lineage": dict(data_lineage) if data_lineage is not None else None,
    }
    model_card = {
        **model_card_core,
        "model_card_content_sha256": canonical_sha256(model_card_core),
    }
    return {
        "schema_version": "chemapp.nmr.calibration-training-result.v1",
        "pipeline_version": TRAINING_PIPELINE_VERSION,
        "status": model_card["status"],
        "training_started": True,
        "training_completed": True,
        "probability_claim_allowed": probability_allowed,
        "calibrator": calibrator,
        "model_card": model_card,
        "admission": admission,
        "reliability": {
            "calibration": calibration_metrics,
            "frozen_test": test_metrics,
        },
        "blocking_reasons": performance_reasons,
    }


def implementation_bundle_sha256(paths: Sequence[str | Path]) -> str:
    rows = []
    for value in sorted(Path(path).resolve() for path in paths):
        rows.append({"name": value.name, "sha256": file_sha256(value)})
    return canonical_sha256(rows)


def hardware_inventory() -> dict[str, Any]:
    """Return a non-authoritative local inventory for the model card/run log."""

    inventory: dict[str, Any] = {
        "logical_cpu_count": None,
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": None,
    }
    try:
        import os

        inventory["logical_cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        import torch

        inventory["torch"] = {
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_version": torch.version.cuda,
            "device_count": int(torch.cuda.device_count()),
            "devices": [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:  # pragma: no cover - environment-specific
        inventory["torch"] = {"inventory_error": type(exc).__name__}
    return inventory


def exit_code_for_result(result: Mapping[str, Any]) -> int:
    if result.get("probability_claim_allowed") is True:
        return 0
    if (
        result.get("status") in {"passed", "passed_manifest_only"}
        and result.get("training_started") is not True
    ):
        return 0
    if result.get("training_started") is True:
        return 3
    return 2


__all__ = [
    "CALIBRATION_EXAMPLES_SCHEMA_VERSION",
    "CALIBRATOR_SCHEMA_VERSION",
    "FROZEN_SCORE_CASE_SCHEMA_VERSION",
    "GOLD_MANIFEST_SCHEMA_VERSION",
    "MODEL_CARD_SCHEMA_VERSION",
    "NMREXP_DERIVED_SCHEMA_VERSION",
    "REVIEW_PROTOCOL_VERSION",
    "RUN_SPEC_SCHEMA_VERSION",
    "TEST_CONSUMPTION_SCHEMA_VERSION",
    "TRAINING_PIPELINE_VERSION",
    "CalibrationAdmissionError",
    "CalibrationGatePolicy",
    "adapt_frozen_score_cases",
    "adapt_nmrexp_reviewed_records",
    "admission_report",
    "audit_calibration_examples",
    "audit_gold_manifest",
    "canonical_json_dumps",
    "canonical_jsonl_sha256",
    "canonical_sha256",
    "exit_code_for_result",
    "file_sha256",
    "fit_calibrator",
    "hardware_inventory",
    "implementation_bundle_sha256",
    "load_jsonl",
    "manual_review_binding_sha256",
    "policy_from_mapping",
    "production_policy_weaknesses",
    "predict_calibrated",
    "reliability_metrics",
    "reliability_svg",
    "reserve_frozen_test_consumption",
    "run_calibration_training",
    "select_scoreable_gold_subset",
    "validate_calibration_run_spec",
]
