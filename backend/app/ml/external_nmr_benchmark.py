"""Deterministic, non-extracting conversion of the fixed external NMR smoke set.

The six Zenodo samples are useful for parser and provenance smoke testing only.
This module deliberately refuses to promote them to an accuracy benchmark:
they are too small, come from one deposited collection, and may overlap the
local reference index by exact molecule.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
import zipfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from app.ml.external_smoke_data import (
    ExternalDatasetError,
    load_fixed_manifest,
    verify_dataset,
)
from app.ml.nmr_data_v2 import connect_readonly, schema_version
from app.parsers import NMRJCAMPParser, inspect_jcamp_numeric


DERIVED_SCHEMA_VERSION = "external-nmr-smoke-derived-v1"
SOURCE_RECORD_ID = 16881130
SOURCE_RECORD_REVISION = 4
SOURCE_DOI = "10.5281/zenodo.16881130"
SOURCE_LICENSE = "CC0-1.0"
EXPECTED_SAMPLE_COUNT = 6
MIN_ACCURACY_BENCHMARK_MOLECULES = 200
DERIVED_FILENAMES = (
    "structures.jsonl",
    "spectra.jsonl",
    "groups.jsonl",
)
SUMMARY_FILENAME = "summary.json"
MAX_SELECTED_MEMBER_BYTES = 512 * 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_TOC_BYTES = 8 * 1024 * 1024
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_JCAMP_LINE_RE = re.compile(r"^##([^=]+)=\s*(.*)$")
_DATA_MARKERS = (
    "XYDATA",
    "XYPOINTS",
    "PEAK TABLE",
    "PEAKTABLE",
    "DATATABLE",
    "PAGE",
)


class ExternalNMRBenchmarkError(ValueError):
    """Raised when source lineage or a derived smoke bundle is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_path(path: Path) -> str:
    with path.open("rb") as handle:
        return _sha256_stream(handle)


def _sha256_stream(handle: BinaryIO) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_relative_member(value: object) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ExternalNMRBenchmarkError("archive member path is empty or invalid")
    normalized = value.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(value)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in posix.parts
        or ".." in windows.parts
        or any(part in {"", "."} for part in posix.parts)
    ):
        raise ExternalNMRBenchmarkError(f"unsafe archive member path: {value!r}")
    return posix.as_posix()


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)


def _read_member(
    archive: zipfile.ZipFile,
    name: str,
    *,
    max_bytes: int = MAX_SELECTED_MEMBER_BYTES,
) -> bytes:
    safe_name = _safe_relative_member(name)
    infos = [item for item in archive.infolist() if item.filename == safe_name]
    if len(infos) != 1:
        raise ExternalNMRBenchmarkError(
            f"expected exactly one archive member {safe_name!r}, found {len(infos)}"
        )
    info = infos[0]
    if info.is_dir() or _is_symlink(info) or info.flag_bits & 0x1:
        raise ExternalNMRBenchmarkError(
            f"selected archive member is not a regular unencrypted file: {safe_name!r}"
        )
    if info.file_size < 1 or info.file_size > max_bytes:
        raise ExternalNMRBenchmarkError(
            f"selected archive member size is outside the safety limit: {safe_name!r}"
        )
    payload = bytearray()
    try:
        with archive.open(info, "r") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                payload.extend(chunk)
                if len(payload) > info.file_size or len(payload) > max_bytes:
                    raise ExternalNMRBenchmarkError(
                        f"archive member exceeded its declared safe size: {safe_name!r}"
                    )
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise ExternalNMRBenchmarkError(
            f"cannot read validated archive member {safe_name!r}: {exc}"
        ) from exc
    if len(payload) != info.file_size:
        raise ExternalNMRBenchmarkError(
            f"archive member byte count changed while reading: {safe_name!r}"
        )
    return bytes(payload)


def _load_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    if len(payload) > MAX_TOC_BYTES:
        raise ExternalNMRBenchmarkError(f"{label} exceeds the metadata size limit")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalNMRBenchmarkError(
            f"{label} is not valid UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise ExternalNMRBenchmarkError(f"{label} root must be an object")
    return value


def _sample_content(sample: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    content = sample.get("$content")
    if not isinstance(content, dict):
        raise ExternalNMRBenchmarkError(
            f"toc sample {ordinal} has no object-valued $content"
        )
    general = content.get("general")
    spectra = content.get("spectra")
    if not isinstance(general, dict) or not isinstance(spectra, dict):
        raise ExternalNMRBenchmarkError(
            f"toc sample {ordinal} is missing general or spectra metadata"
        )
    return content


def _mol_from_block(value: str, label: str) -> Any:
    if not value.strip():
        raise ExternalNMRBenchmarkError(f"{label} structure is empty")
    mol = Chem.MolFromMolBlock(
        value,
        sanitize=True,
        removeHs=False,
        strictParsing=True,
    )
    if mol is None:
        raise ExternalNMRBenchmarkError(f"{label} structure cannot be parsed by RDKit")
    return Chem.RemoveHs(mol)


def _structure_identity(mol: Any) -> dict[str, str]:
    cleaned = rdMolStandardize.Cleanup(mol)
    parent = Chem.RemoveHs(rdMolStandardize.FragmentParent(cleaned))
    inchi_key = str(Chem.MolToInchiKey(mol) or "").upper()
    parent_key = str(Chem.MolToInchiKey(parent) or "").upper()
    if not inchi_key or not parent_key:
        raise ExternalNMRBenchmarkError("RDKit could not derive a stable InChIKey")
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        mol=parent,
        includeChirality=False,
    )
    generic_scaffold = ""
    if scaffold:
        scaffold_mol = Chem.MolFromSmiles(scaffold)
        if scaffold_mol is not None:
            generic_scaffold = Chem.MolToSmiles(
                MurckoScaffold.MakeScaffoldGeneric(scaffold_mol),
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
        "inchi_key": inchi_key,
        "molecule_key": parent_key.split("-", 1)[0],
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "scaffold_key": scaffold or "acyclic",
        "generic_scaffold_key": generic_scaffold or "acyclic",
    }


def _first_name(general: Mapping[str, Any]) -> str | None:
    names = general.get("name")
    if not isinstance(names, list):
        return None
    for item in names:
        if isinstance(item, Mapping):
            value = str(item.get("value") or "").strip()
            if value:
                return value
    return None


def _sample_source_id(sample: Mapping[str, Any], ordinal: int) -> str:
    identifiers = sample.get("$id")
    if isinstance(identifiers, list):
        for value in reversed(identifiers):
            text = str(value or "").strip()
            if text:
                return text
    opaque_id = str(sample.get("_id") or "").strip()
    return opaque_id or f"sample-{ordinal}"


def _normalise_label(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().upper())


def _jcamp_headers(payload: bytes, member: str) -> dict[str, str]:
    prefix = payload[:MAX_METADATA_BYTES]
    text = prefix.decode("utf-8", errors="replace")
    headers: dict[str, str] = {}
    for line in text.splitlines():
        match = _JCAMP_LINE_RE.match(line.strip())
        if not match:
            continue
        key = _normalise_label(match.group(1))
        if key in _DATA_MARKERS:
            break
        headers.setdefault(key, match.group(2).strip().strip("<>"))
    if "TITLE" not in headers or not any(
        key in headers for key in ("JCAMPDX", "JCAMP-DX")
    ):
        raise ExternalNMRBenchmarkError(
            f"selected spectrum is not recognisable JCAMP-DX: {member!r}"
        )
    if b"##END=" not in payload[-65536:].upper():
        raise ExternalNMRBenchmarkError(
            f"selected JCAMP-DX member has no terminal ##END=: {member!r}"
        )
    return headers


def _header_value(headers: Mapping[str, str], *keys: str) -> str | None:
    for key in keys:
        value = str(headers.get(_normalise_label(key)) or "").strip()
        if value:
            return value
    return None


def _header_float(headers: Mapping[str, str], *keys: str) -> float | None:
    value = _header_value(headers, *keys)
    if value is None:
        return None
    match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?", value)
    return _finite_number(match.group(0)) if match else None


def _canonical_nucleus(value: object) -> str:
    text = re.sub(r"[\s^{}<>]", "", str(value or "")).upper()
    aliases = {
        "H1": "1H",
        "1H": "1H",
        "C13": "13C",
        "13C": "13C",
        "F19": "19F",
        "19F": "19F",
        "P31": "31P",
        "31P": "31P",
        "N15": "15N",
        "15N": "15N",
    }
    return aliases.get(text, text)


def _normalise_nuclei(value: object) -> list[str]:
    values = value if isinstance(value, list) else [value]
    nuclei = [_canonical_nucleus(item) for item in values]
    if not nuclei or any(not item for item in nuclei):
        raise ExternalNMRBenchmarkError("spectrum has no valid nucleus metadata")
    return nuclei


def _condition(
    toc_value: Any,
    header_value: Any,
    *,
    numeric: bool = False,
    comparable: bool = True,
) -> dict[str, Any]:
    toc_parsed = _finite_number(toc_value) if numeric else str(toc_value or "").strip()
    header_parsed = (
        _finite_number(header_value) if numeric else str(header_value or "").strip()
    )
    value = toc_parsed if toc_parsed not in (None, "") else header_parsed
    source = "toc.json" if toc_parsed not in (None, "") else "jcamp_header"
    if value in (None, ""):
        source = "unavailable"
    conflict: bool | None = None
    if comparable and toc_parsed not in (None, "") and header_parsed not in (None, ""):
        if numeric:
            conflict = not math.isclose(
                float(toc_parsed),
                float(header_parsed),
                rel_tol=1e-5,
                abs_tol=1e-3,
            )
        else:
            conflict = str(toc_parsed).casefold() != str(header_parsed).casefold()
    return {
        "value": value,
        "source": source,
        "toc_value": toc_parsed,
        "jcamp_header_value": header_parsed,
        "conflict": conflict,
    }


def _normalise_ranges(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    if len(value) > 10000:
        raise ExternalNMRBenchmarkError("spectrum annotation range count is unsafe")
    output: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        signals: list[dict[str, Any]] = []
        raw_signals = item.get("signal")
        if isinstance(raw_signals, list):
            for signal in raw_signals[:1000]:
                if not isinstance(signal, Mapping):
                    continue
                couplings: list[float] = []
                raw_couplings = signal.get("j")
                if isinstance(raw_couplings, list):
                    for coupling in raw_couplings[:100]:
                        if isinstance(coupling, Mapping):
                            number = _finite_number(coupling.get("coupling"))
                            if number is not None:
                                couplings.append(number)
                signals.append(
                    {
                        "delta_ppm": _finite_number(signal.get("delta")),
                        "multiplicity": (
                            str(signal.get("multiplicity") or "").strip() or None
                        ),
                        "couplings_hz": couplings,
                        "assignment_ids": (
                            [
                                str(entry)
                                for entry in signal.get("diaID", [])
                                if str(entry).strip()
                            ]
                            if isinstance(signal.get("diaID"), list)
                            else []
                        ),
                    }
                )
        output.append(
            {
                "from_ppm": _finite_number(item.get("from")),
                "to_ppm": _finite_number(item.get("to")),
                "integral": _finite_number(item.get("integral")),
                "signals": signals,
            }
        )
    return output


def _spectrum_member(
    spectrum: Mapping[str, Any],
    *,
    key: str,
) -> str | None:
    value = spectrum.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ExternalNMRBenchmarkError(f"spectrum {key} metadata must be an object")
    filename = value.get("filename")
    if filename is None:
        return None
    return _safe_relative_member(filename)


def _member_provenance(
    archive: zipfile.ZipFile,
    member: str,
    *,
    max_bytes: int = MAX_SELECTED_MEMBER_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    payload = _read_member(archive, member, max_bytes=max_bytes)
    return payload, {
        "member": member,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
    }


def _inspect_transient_jcamp(
    payload: bytes,
    *,
    source_sha256: str,
) -> dict[str, Any]:
    """Use the common decoder through a deleted-on-exit transient spool."""

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix="chemapp-external-nmr-",
            suffix=".jdx",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        inspection = inspect_jcamp_numeric(temporary_path)
        if not isinstance(inspection, dict):
            raise ExternalNMRBenchmarkError(
                "common JCAMP decoder returned a non-object inspection"
            )
        if inspection.get("numeric_payload_decoded") is not True:
            reason = str(inspection.get("unsupported_reason") or "unknown reason")
            raise ExternalNMRBenchmarkError(
                f"processed JCAMP numeric payload could not be decoded: {reason}"
            )
        if inspection.get("source_sha256") != source_sha256:
            raise ExternalNMRBenchmarkError(
                "transient JCAMP decoder source hash differs from the ZIP member"
            )
        if int(inspection.get("dimension", 0)) == 1:
            spectrum = NMRJCAMPParser().parse(temporary_path)
            if len(spectrum.x_data) != int(inspection["point_count"]) or len(
                spectrum.y_data
            ) != int(inspection["point_count"]):
                raise ExternalNMRBenchmarkError(
                    "common JCAMP Spectrum length differs from numeric inspection"
                )
            inspection["representation"] = {
                "spectrum_model_supported": True,
                "matrix_only": False,
                "verified_by": "NMRJCAMPParser.parse",
                "x_points": len(spectrum.x_data),
                "y_points": len(spectrum.y_data),
                "x_unit": spectrum.x_unit,
                "x_axis_direction": (
                    "descending"
                    if float(spectrum.x_data[0]) > float(spectrum.x_data[-1])
                    else "ascending"
                ),
                "nucleus": spectrum.parameters.get("nucleus"),
                "processing_qc_available": isinstance(
                    spectrum.parameters.get("processing_quality"),
                    dict,
                ),
            }
        else:
            inspection["representation"] = {
                "spectrum_model_supported": False,
                "matrix_only": True,
                "verified_by": "inspect_jcamp_numeric",
                "reason": (
                    "multi-dimensional numeric matrix is preserved and must not "
                    "be flattened into the one-dimensional Spectrum model"
                ),
            }
        return inspection
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _group_audit(
    rows: Sequence[Mapping[str, Any]],
    *,
    partition_field: str = "partition",
) -> dict[str, Any]:
    fields = ("source_record_key", "molecule_key", "scaffold_key")
    result: dict[str, Any] = {}
    ids: set[str] = set()
    for row in rows:
        record_id = str(row.get("record_id") or "")
        if not record_id or record_id in ids:
            raise ExternalNMRBenchmarkError(
                f"duplicate or empty derived spectrum record_id: {record_id!r}"
            )
        ids.add(record_id)
    for field in fields:
        locations: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            group = str(row.get(field) or "")
            partition = str(row.get(partition_field) or "")
            if not group or not partition:
                raise ExternalNMRBenchmarkError(
                    f"derived spectrum is missing {field} or {partition_field}"
                )
            locations[group].add(partition)
        leaked = {
            group: sorted(partitions)
            for group, partitions in locations.items()
            if len(partitions) > 1
        }
        if leaked:
            example = next(iter(leaked.items()))
            raise ExternalNMRBenchmarkError(
                f"{field} leakage detected for {example[0]}: {example[1]}"
            )
        result[field] = {
            "groups": len(locations),
            "leaked_groups": 0,
            "partitions": sorted(
                {
                    partition
                    for locations_ in locations.values()
                    for partition in locations_
                }
            ),
        }
    return result


def _base_index_overlap(
    structures: Sequence[Mapping[str, Any]],
    index_path: str | Path | None,
) -> dict[str, Any]:
    if index_path is None:
        return {
            "status": "not_checked",
            "base_index": None,
            "base_index_sha256": None,
            "base_index_schema_version": None,
            "base_source_snapshots": [],
            "external_molecules": len(structures),
            "exact_molecule_overlaps": None,
            "exact_overlap_rate": None,
            "overlapping_molecule_keys": [],
        }
    path = Path(index_path).expanduser().resolve()
    if not path.is_file():
        raise ExternalNMRBenchmarkError(f"base index does not exist: {path}")
    before_sha256 = _sha256_path(path)
    conn = connect_readonly(path)
    try:
        version = schema_version(conn)
        snapshots = [
            {
                "source_name": str(row["source_name"]),
                "source_version": row["source_version"],
                "sha256": str(row["sha256"]),
            }
            for row in conn.execute(
                """
                SELECT source_name, source_version, sha256
                FROM source_snapshots
                ORDER BY id
                """
            ).fetchall()
        ]
        overlaps: list[str] = []
        counts: dict[str, int] = {}
        for structure in structures:
            molecule_key = str(structure["molecule_key"])
            row = conn.execute(
                """
                SELECT COUNT(*)
                FROM molecules
                WHERE UPPER(inchi_key) LIKE ?
                """,
                (f"{molecule_key}%",),
            ).fetchone()
            count = int(row[0]) if row is not None else 0
            counts[molecule_key] = count
            if count:
                overlaps.append(molecule_key)
    finally:
        conn.close()
    after_sha256 = _sha256_path(path)
    if before_sha256 != after_sha256:
        raise ExternalNMRBenchmarkError(
            "base index changed while exact-molecule overlap was audited"
        )
    return {
        "status": "checked",
        "base_index": "base-index.sqlite",
        "base_index_sha256": after_sha256,
        "base_index_schema_version": version,
        "base_source_snapshots": snapshots,
        "external_molecules": len(structures),
        "exact_molecule_overlaps": len(overlaps),
        "exact_overlap_rate": len(overlaps) / max(len(structures), 1),
        "overlapping_molecule_keys": sorted(overlaps),
        "base_source_record_counts_by_molecule": dict(sorted(counts.items())),
    }


def _accuracy_eligibility(
    structures: Sequence[Mapping[str, Any]],
    overlap: Mapping[str, Any],
) -> dict[str, Any]:
    reasons = [
        (
            f"only {len(structures)} molecules are present; at least "
            f"{MIN_ACCURACY_BENCHMARK_MOLECULES} are required by this local policy"
        ),
        "all samples come from one deposited Zenodo collection",
        "peak assignments and numeric JCAMP axes have not been independently reviewed",
    ]
    exact_overlaps = overlap.get("exact_molecule_overlaps")
    if exact_overlaps is None:
        reasons.append(
            "exact molecule overlap with the model/reference index was not checked"
        )
    elif int(exact_overlaps) > 0:
        reasons.append(
            f"{exact_overlaps}/{len(structures)} molecules overlap the base index exactly"
        )
    return {
        "headline_accuracy_allowed": False,
        "open_world_claim_allowed": False,
        "independent_accuracy_test": False,
        "role": "external_source_structure_overlapping_parser_smoke",
        "reasons": reasons,
    }


def derive_external_nmr_records(
    manifest: Mapping[str, Any],
    source_directory: str | Path,
    *,
    base_index_path: str | Path | None = None,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    """Read verified archives without extraction and create unified records."""

    if (
        int(manifest.get("record_id", -1)) != SOURCE_RECORD_ID
        or int(manifest.get("record_revision", -1)) != SOURCE_RECORD_REVISION
        or str(manifest.get("doi")) != SOURCE_DOI
        or str(manifest.get("license", {}).get("spdx_id")) != SOURCE_LICENSE
    ):
        raise ExternalNMRBenchmarkError(
            "unexpected external dataset identity or license"
        )
    scope = manifest.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("sample_count") != EXPECTED_SAMPLE_COUNT
        or scope.get("role") != "proof_of_concept_external_smoke_only"
        or scope.get("headline_benchmark_allowed") is not False
        or not _HASH_RE.fullmatch(str(manifest.get("inventory_sha256") or ""))
    ):
        raise ExternalNMRBenchmarkError(
            "external dataset scope or frozen inventory identity is unsafe"
        )
    source_root = Path(source_directory).expanduser().resolve()
    frozen_files = {
        str(item["name"]): item
        for item in manifest.get("files", [])
        if isinstance(item, Mapping)
    }
    expected_files = {
        "1.zip",
        "2.zip",
        "3.zip",
        "4.zip",
        "5.zip",
        "6.zip",
        "README.md",
        "toc.json",
    }
    if set(frozen_files) != expected_files:
        raise ExternalNMRBenchmarkError(
            "external dataset must contain the exact eight-file frozen inventory"
        )
    toc_path = source_root / "toc.json"
    if not toc_path.is_file() or toc_path.stat().st_size > MAX_TOC_BYTES:
        raise ExternalNMRBenchmarkError(
            "toc.json is missing or exceeds the safety limit"
        )
    toc_payload = toc_path.read_bytes()
    toc_record = frozen_files.get("toc.json")
    if toc_record is None or _sha256_bytes(toc_payload) != str(
        toc_record.get("sha256") or ""
    ):
        raise ExternalNMRBenchmarkError("toc.json differs from its frozen SHA-256")
    toc = _load_json_bytes(toc_payload, "toc.json")
    samples = toc.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_SAMPLE_COUNT:
        raise ExternalNMRBenchmarkError(
            f"toc.json must contain exactly {EXPECTED_SAMPLE_COUNT} samples"
        )
    structures: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    groups: list[dict[str, Any]] = []
    for ordinal, raw_sample in enumerate(samples, start=1):
        if not isinstance(raw_sample, Mapping):
            raise ExternalNMRBenchmarkError(f"toc sample {ordinal} is not an object")
        content = _sample_content(raw_sample, ordinal)
        general = content["general"]
        nmr = content["spectra"].get("nmr")
        if not isinstance(nmr, list) or not nmr:
            raise ExternalNMRBenchmarkError(f"toc sample {ordinal} has no NMR spectra")
        archive_name = f"{ordinal}.zip"
        archive_record = frozen_files.get(archive_name)
        if archive_record is None:
            raise ExternalNMRBenchmarkError(f"fixed manifest has no {archive_name}")
        archive_path = source_root / archive_name
        archive_sha256 = str(archive_record.get("sha256") or "")
        if not _HASH_RE.fullmatch(archive_sha256):
            raise ExternalNMRBenchmarkError(f"{archive_name} has no frozen SHA-256")
        sample_prefix = f"{ordinal}"
        archive_stream: BinaryIO | None = None
        try:
            archive_stream = archive_path.open("rb")
            before_stat = os.fstat(archive_stream.fileno())
            if _sha256_stream(archive_stream) != archive_sha256:
                raise ExternalNMRBenchmarkError(
                    f"{archive_name} differs from its frozen SHA-256 before conversion"
                )
            archive_stream.seek(0)
            archive_handle = zipfile.ZipFile(archive_stream)
        except ExternalNMRBenchmarkError:
            if archive_stream is not None:
                archive_stream.close()
            raise
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
            if archive_stream is not None:
                archive_stream.close()
            raise ExternalNMRBenchmarkError(
                f"cannot open verified archive {archive_name}: {exc}"
            ) from exc
        try:
            with archive_handle as archive:
                available_processed = {
                    info.filename
                    for info in archive.infolist()
                    if "/spectra/nmr/" in info.filename
                    and info.filename.casefold().endswith(".jdx")
                    and not info.filename.casefold().endswith(".fid.jdx")
                }
                available_fid = {
                    info.filename
                    for info in archive.infolist()
                    if "/spectra/nmr/" in info.filename
                    and info.filename.casefold().endswith(".fid.jdx")
                }
                selected_processed: set[str] = set()
                selected_fid: set[str] = set()
                structure_member = f"{sample_prefix}/structure.mol"
                index_member = f"{sample_prefix}/index.json"
                structure_bytes, structure_member_info = _member_provenance(
                    archive,
                    structure_member,
                    max_bytes=MAX_METADATA_BYTES,
                )
                index_bytes, index_member_info = _member_provenance(
                    archive,
                    index_member,
                    max_bytes=MAX_METADATA_BYTES,
                )
                index = _load_json_bytes(index_bytes, index_member)
                index_general = index.get("general")
                if not isinstance(index_general, Mapping):
                    raise ExternalNMRBenchmarkError(
                        f"{index_member} has no general structure metadata"
                    )
                try:
                    archive_molblock = structure_bytes.decode("utf-8")
                except UnicodeError as exc:
                    raise ExternalNMRBenchmarkError(
                        f"{structure_member} is not UTF-8 text"
                    ) from exc
                toc_molblock = str(general.get("molfile") or "")
                index_molblock = str(index_general.get("molfile") or "")
                identities = [
                    _structure_identity(_mol_from_block(value, label))
                    for value, label in (
                        (archive_molblock, structure_member),
                        (toc_molblock, f"toc sample {ordinal}"),
                        (index_molblock, index_member),
                    )
                ]
                identity = identities[0]
                if any(item != identity for item in identities[1:]):
                    raise ExternalNMRBenchmarkError(
                        f"structure identity differs across toc/index/archive for sample {ordinal}"
                    )
                declared_formula = re.sub(r"\s+", "", str(general.get("mf") or ""))
                if declared_formula and declared_formula != identity["formula"]:
                    raise ExternalNMRBenchmarkError(
                        f"declared formula differs from structure for sample {ordinal}"
                    )
                sample_source_id = _sample_source_id(raw_sample, ordinal)
                source_record_key = (
                    f"doi:{SOURCE_DOI}:revision:{SOURCE_RECORD_REVISION}:"
                    f"archive:{archive_name}:sample:{sample_source_id}"
                )
                structure_record_id = hashlib.sha256(
                    f"{source_record_key}:structure".encode()
                ).hexdigest()
                structure_record = {
                    "schema_version": DERIVED_SCHEMA_VERSION,
                    "record_id": structure_record_id,
                    "partition": "external_smoke",
                    "source_collection_key": f"doi:{SOURCE_DOI}",
                    "source_record_key": source_record_key,
                    "source_record_ordinal": ordinal,
                    "source_sample_id": sample_source_id,
                    "source_sample_uuid": str(raw_sample.get("_id") or "") or None,
                    "name": _first_name(general),
                    "declared_formula": declared_formula or None,
                    **identity,
                    "source": {
                        "record_id": SOURCE_RECORD_ID,
                        "record_revision": SOURCE_RECORD_REVISION,
                        "doi": SOURCE_DOI,
                        "license_spdx": SOURCE_LICENSE,
                        "archive_name": archive_name,
                        "archive_sha256": archive_sha256,
                        "structure_member": structure_member_info,
                        "index_member": index_member_info,
                        "toc_sha256": str(frozen_files["toc.json"]["sha256"]),
                    },
                }
                structures.append(structure_record)

                spectrum_ids: list[str] = []
                for spectrum_ordinal, spectrum_value in enumerate(nmr, start=1):
                    if not isinstance(spectrum_value, Mapping):
                        raise ExternalNMRBenchmarkError(
                            f"sample {ordinal} spectrum {spectrum_ordinal} is not an object"
                        )
                    jcamp_relative = _spectrum_member(spectrum_value, key="jcamp")
                    if jcamp_relative is None:
                        raise ExternalNMRBenchmarkError(
                            f"sample {ordinal} spectrum {spectrum_ordinal} has no JCAMP member"
                        )
                    jcamp_member = f"{sample_prefix}/{jcamp_relative}"
                    if jcamp_member in selected_processed:
                        raise ExternalNMRBenchmarkError(
                            f"duplicate processed spectrum member in toc.json: {jcamp_member}"
                        )
                    selected_processed.add(jcamp_member)
                    payload, processed_info = _member_provenance(archive, jcamp_member)
                    headers = _jcamp_headers(payload, jcamp_member)
                    numeric_inspection = _inspect_transient_jcamp(
                        payload,
                        source_sha256=str(processed_info["sha256"]),
                    )
                    fid_relative = _spectrum_member(spectrum_value, key="jcampFID")
                    fid_info = None
                    if fid_relative is not None:
                        fid_member = f"{sample_prefix}/{fid_relative}"
                        if fid_member in selected_fid:
                            raise ExternalNMRBenchmarkError(
                                f"duplicate FID spectrum member in toc.json: {fid_member}"
                            )
                        selected_fid.add(fid_member)
                        _, fid_info = _member_provenance(
                            archive,
                            fid_member,
                        )
                    nuclei = _normalise_nuclei(spectrum_value.get("nucleus"))
                    dimension_value = spectrum_value.get("dimension")
                    dimension = (
                        int(dimension_value)
                        if dimension_value is not None
                        else len(nuclei)
                    )
                    if dimension not in {1, 2} or len(nuclei) != dimension:
                        raise ExternalNMRBenchmarkError(
                            f"sample {ordinal} spectrum {spectrum_ordinal} has inconsistent dimension/nuclei"
                        )
                    if int(numeric_inspection["dimension"]) != dimension:
                        raise ExternalNMRBenchmarkError(
                            f"toc/common-decoder dimension conflict for {jcamp_member}: "
                            f"toc={dimension}, decoded={numeric_inspection['dimension']}"
                        )
                    header_nucleus = _canonical_nucleus(
                        _header_value(headers, ".OBSERVE NUCLEUS") or ""
                    )
                    if header_nucleus and header_nucleus not in nuclei:
                        raise ExternalNMRBenchmarkError(
                            f"toc/JCAMP nucleus conflict for {jcamp_member}"
                        )
                    field_header = _header_float(
                        headers,
                        ".OBSERVE FREQUENCY",
                        "$SFO1",
                        "$BF1",
                    )
                    solvent_header = _header_value(
                        headers,
                        ".SOLVENT NAME",
                        "$SOLVENT",
                    )
                    annotations = _normalise_ranges(spectrum_value.get("range"))
                    spectrum_key = (
                        f"{source_record_key}:nmr:{spectrum_ordinal}:"
                        f"{processed_info['sha256']}"
                    )
                    spectrum_record_id = hashlib.sha256(
                        spectrum_key.encode()
                    ).hexdigest()
                    spectrum_ids.append(spectrum_record_id)
                    spectra.append(
                        {
                            "schema_version": DERIVED_SCHEMA_VERSION,
                            "record_id": spectrum_record_id,
                            "structure_record_id": structure_record_id,
                            "partition": "external_smoke",
                            "source_collection_key": f"doi:{SOURCE_DOI}",
                            "source_record_key": source_record_key,
                            "molecule_key": identity["molecule_key"],
                            "scaffold_key": identity["scaffold_key"],
                            "generic_scaffold_key": identity["generic_scaffold_key"],
                            "inchi_key": identity["inchi_key"],
                            "formula": identity["formula"],
                            "spectrum_ordinal": spectrum_ordinal,
                            "dimension": dimension,
                            "nuclei": nuclei,
                            "modality": "+".join(
                                nucleus.casefold() for nucleus in nuclei
                            ),
                            "title": str(spectrum_value.get("title") or "").strip()
                            or None,
                            "experiment": (
                                str(spectrum_value.get("experiment") or "").strip()
                                or None
                            ),
                            "pulse_sequence": (
                                str(spectrum_value.get("pulse") or "").strip() or None
                            ),
                            "processed_frequency_domain": (
                                spectrum_value.get("isFt")
                                if type(spectrum_value.get("isFt")) is bool
                                else None
                            ),
                            "complex_data": (
                                spectrum_value.get("isComplex")
                                if type(spectrum_value.get("isComplex")) is bool
                                else None
                            ),
                            "conditions": {
                                "solvent": _condition(
                                    spectrum_value.get("solvent"),
                                    solvent_header,
                                ),
                                "field_mhz": _condition(
                                    spectrum_value.get("frequency"),
                                    field_header,
                                    numeric=True,
                                ),
                                "temperature_k": _condition(
                                    spectrum_value.get("temperature"),
                                    None,
                                    numeric=True,
                                ),
                                "acquisition_date": _condition(
                                    spectrum_value.get("date"),
                                    _header_value(headers, "LONG DATE", "DATE"),
                                    comparable=False,
                                ),
                                "sample_reference": _condition(
                                    spectrum_value.get("reference"),
                                    None,
                                    comparable=False,
                                ),
                                "shift_reference": _condition(
                                    None,
                                    _header_value(headers, ".SHIFT REFERENCE"),
                                    comparable=False,
                                ),
                            },
                            "annotations": {
                                "ranges": annotations,
                                "range_count": len(annotations),
                                "signal_count": sum(
                                    len(item["signals"]) for item in annotations
                                ),
                                "source": "toc.json",
                                "independently_reviewed": False,
                            },
                            "jcamp": {
                                "format_version": _header_value(
                                    headers,
                                    "JCAMPDX",
                                    "JCAMP-DX",
                                ),
                                "data_type": _header_value(headers, "DATA TYPE"),
                                "data_class": _header_value(headers, "DATA CLASS"),
                                "x_units": _header_value(headers, "XUNITS"),
                                "y_units": _header_value(headers, "YUNITS"),
                                "first_x": _header_float(headers, "FIRSTX"),
                                "last_x": _header_float(headers, "LASTX"),
                                "n_points": _header_float(headers, "NPOINTS"),
                                "numeric_payload_decoded": True,
                                "numeric_inspection": numeric_inspection,
                                "axis_validation": (
                                    "numeric_payload_shape_and_finiteness_verified_"
                                    "axis_not_independently_reviewed"
                                ),
                            },
                            "source": {
                                "record_id": SOURCE_RECORD_ID,
                                "record_revision": SOURCE_RECORD_REVISION,
                                "doi": SOURCE_DOI,
                                "license_spdx": SOURCE_LICENSE,
                                "archive_name": archive_name,
                                "archive_sha256": archive_sha256,
                                "processed_member": processed_info,
                                "fid_member": (
                                    {
                                        **fid_info,
                                        "numeric_payload_decoded": False,
                                        "status": "metadata_only_not_requested",
                                    }
                                    if fid_info is not None
                                    else None
                                ),
                            },
                        }
                    )
            if selected_processed != available_processed:
                raise ExternalNMRBenchmarkError(
                    f"toc/archive processed NMR member inventory differs for {archive_name}"
                )
            if selected_fid != available_fid:
                raise ExternalNMRBenchmarkError(
                    f"toc/archive FID NMR member inventory differs for {archive_name}"
                )
            groups.append(
                {
                    "schema_version": DERIVED_SCHEMA_VERSION,
                    "partition": "external_smoke",
                    "source_record_key": source_record_key,
                    "molecule_key": identity["molecule_key"],
                    "scaffold_key": identity["scaffold_key"],
                    "generic_scaffold_key": identity["generic_scaffold_key"],
                    "structure_record_id": structure_record_id,
                    "spectrum_record_ids": sorted(spectrum_ids),
                }
            )
            archive_stream.seek(0)
            after_sha256 = _sha256_stream(archive_stream)
            after_stat = os.fstat(archive_stream.fileno())
            if (
                after_sha256 != archive_sha256
                or after_stat.st_size != before_stat.st_size
                or after_stat.st_mtime_ns != before_stat.st_mtime_ns
                or (
                    before_stat.st_ino
                    and after_stat.st_ino
                    and after_stat.st_ino != before_stat.st_ino
                )
            ):
                raise ExternalNMRBenchmarkError(
                    f"{archive_name} changed while derived records were being built"
                )
        finally:
            archive_stream.close()

    structures.sort(key=lambda item: str(item["record_id"]))
    spectra.sort(key=lambda item: str(item["record_id"]))
    groups.sort(key=lambda item: str(item["source_record_key"]))
    group_audit = _group_audit(spectra)
    overlap = _base_index_overlap(structures, base_index_path)
    metadata_conflicts = Counter()
    for spectrum in spectra:
        for condition, value in spectrum["conditions"].items():
            if value["conflict"] is True:
                metadata_conflicts[condition] += 1
    status = {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "status": "verified_smoke_only",
        "source": {
            "record_id": SOURCE_RECORD_ID,
            "record_revision": SOURCE_RECORD_REVISION,
            "doi": SOURCE_DOI,
            "license_spdx": SOURCE_LICENSE,
            "source_inventory_sha256": manifest["inventory_sha256"],
        },
        "scope": {
            "samples": len(structures),
            "spectra": len(spectra),
            "one_dimensional_spectra": sum(item["dimension"] == 1 for item in spectra),
            "two_dimensional_spectra": sum(item["dimension"] == 2 for item in spectra),
            "nuclei": dict(
                sorted(
                    Counter(
                        nucleus
                        for spectrum in spectra
                        for nucleus in spectrum["nuclei"]
                    ).items()
                )
            ),
            "numeric_payloads_decoded": sum(
                item["jcamp"]["numeric_payload_decoded"] is True for item in spectra
            ),
            "numeric_payloads_unsupported": sum(
                item["jcamp"]["numeric_payload_decoded"] is not True for item in spectra
            ),
            "numeric_value_count": sum(
                int(item["jcamp"]["numeric_inspection"]["numeric_value_count"])
                for item in spectra
            ),
            "one_dimensional_spectrum_model_supported": sum(
                item["dimension"] == 1
                and item["jcamp"]["numeric_inspection"]["representation"][
                    "spectrum_model_supported"
                ]
                is True
                for item in spectra
            ),
            "two_dimensional_matrix_only": sum(
                item["dimension"] == 2
                and item["jcamp"]["numeric_inspection"]["representation"]["matrix_only"]
                is True
                for item in spectra
            ),
            "independently_reviewed_peak_assignments": 0,
        },
        "conversion": {
            "archives_extracted": False,
            "processed_members_transiently_spooled_for_common_decoder": True,
            "transient_spools_removed": True,
            "original_assets_modified": False,
            "derived_format": "canonical UTF-8 JSON Lines",
            "deterministic": True,
            "structure_identity_cross_checks": [
                "toc.json molfile",
                "archive index.json molfile",
                "archive structure.mol",
            ],
            "metadata_conflicts": dict(sorted(metadata_conflicts.items())),
        },
        "partitioning": {
            "partition": "external_smoke",
            "policy": "single_smoke_partition_no_train_test_split",
            "reason": "six samples are too small for a meaningful four-way benchmark split",
            "group_leakage_audit": group_audit,
        },
        "base_index_overlap": overlap,
        "accuracy_eligibility": _accuracy_eligibility(structures, overlap),
    }
    return structures, spectra, groups, status


def _render_jsonl(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join((_canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows)


def _atomic_publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{path.name}.",
            suffix=".part",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _artifact_record(name: str, payload: bytes, row_count: int) -> dict[str, Any]:
    return {
        "name": name,
        "bytes": len(payload),
        "sha256": _sha256_bytes(payload),
        "rows": row_count,
    }


def _ensure_separate_output(source: Path, output: Path) -> None:
    if output == source or source in output.parents:
        raise ExternalNMRBenchmarkError(
            "derived output must not be the source directory or one of its descendants"
        )
    source_files = {path.resolve() for path in source.iterdir() if path.is_file()}
    if any(
        (output / name).resolve() in source_files
        for name in (*DERIVED_FILENAMES, SUMMARY_FILENAME)
    ):
        raise ExternalNMRBenchmarkError("derived output would overwrite a source asset")


def build_external_nmr_bundle(
    *,
    manifest_path: str | Path,
    source_directory: str | Path,
    output_directory: str | Path,
    base_index_path: str | Path | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Verify all source bytes and atomically publish a deterministic bundle."""

    manifest = load_fixed_manifest(manifest_path)
    source = Path(source_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    _ensure_separate_output(source, output)
    try:
        verify_dataset(manifest, source)
    except ExternalDatasetError as exc:
        raise ExternalNMRBenchmarkError(str(exc)) from exc
    structures, spectra, groups, status = derive_external_nmr_records(
        manifest,
        source,
        base_index_path=base_index_path,
    )
    payloads = {
        "structures.jsonl": _render_jsonl(structures),
        "spectra.jsonl": _render_jsonl(spectra),
        "groups.jsonl": _render_jsonl(groups),
    }
    summary = {
        **status,
        "artifacts": {
            name: _artifact_record(
                name,
                payload,
                {
                    "structures.jsonl": len(structures),
                    "spectra.jsonl": len(spectra),
                    "groups.jsonl": len(groups),
                }[name],
            )
            for name, payload in sorted(payloads.items())
        },
    }
    summary_payload = (
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    targets = [output / name for name in (*DERIVED_FILENAMES, SUMMARY_FILENAME)]
    existing = [path for path in targets if path.exists()]
    if existing and not overwrite:
        raise ExternalNMRBenchmarkError(
            "refusing to replace derived artifacts without --overwrite: "
            + ", ".join(path.name for path in existing)
        )
    output.mkdir(parents=True, exist_ok=True)
    for name, payload in payloads.items():
        _atomic_publish(output / name, payload)
    _atomic_publish(output / SUMMARY_FILENAME, summary_payload)
    return {
        **summary,
        "summary_artifact": {
            "name": SUMMARY_FILENAME,
            "bytes": len(summary_payload),
            "sha256": _sha256_bytes(summary_payload),
        },
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise ExternalNMRBenchmarkError(
                        f"{path.name}:{line_number} is an empty JSONL line"
                    )
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ExternalNMRBenchmarkError(
                        f"{path.name}:{line_number} is not a JSON object"
                    )
                rows.append(value)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        if isinstance(exc, ExternalNMRBenchmarkError):
            raise
        raise ExternalNMRBenchmarkError(f"cannot read {path.name}: {exc}") from exc
    return rows


def verify_external_nmr_bundle(
    *,
    manifest_path: str | Path,
    source_directory: str | Path,
    output_directory: str | Path,
    base_index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Recompute the expected bundle and verify without writing any file."""

    manifest = load_fixed_manifest(manifest_path)
    source = Path(source_directory).expanduser().resolve()
    output = Path(output_directory).expanduser().resolve()
    _ensure_separate_output(source, output)
    try:
        verify_dataset(manifest, source)
    except ExternalDatasetError as exc:
        raise ExternalNMRBenchmarkError(str(exc)) from exc
    structures, spectra, groups, status = derive_external_nmr_records(
        manifest,
        source,
        base_index_path=base_index_path,
    )
    expected_rows = {
        "structures.jsonl": structures,
        "spectra.jsonl": spectra,
        "groups.jsonl": groups,
    }
    try:
        summary = json.loads((output / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalNMRBenchmarkError(f"cannot read summary.json: {exc}") from exc
    if not isinstance(summary, dict):
        raise ExternalNMRBenchmarkError("summary.json root must be an object")
    expected_summary = {
        **status,
        "artifacts": {
            name: _artifact_record(name, _render_jsonl(rows), len(rows))
            for name, rows in sorted(expected_rows.items())
        },
    }
    if _canonical_json(summary) != _canonical_json(expected_summary):
        raise ExternalNMRBenchmarkError(
            "summary.json differs from the recomputed deterministic summary"
        )
    for name, rows in expected_rows.items():
        path = output / name
        actual_rows = _load_jsonl(path)
        if _canonical_json(actual_rows) != _canonical_json(rows):
            raise ExternalNMRBenchmarkError(
                f"{name} differs from the recomputed source-derived records"
            )
        artifact = summary["artifacts"][name]
        if (
            path.stat().st_size != artifact["bytes"]
            or _sha256_path(path) != artifact["sha256"]
            or len(actual_rows) != artifact["rows"]
        ):
            raise ExternalNMRBenchmarkError(
                f"{name} artifact digest or row count changed"
            )
    _group_audit(spectra)
    summary_payload = (output / SUMMARY_FILENAME).read_bytes()
    return {
        **summary,
        "verification": {
            "status": "verified",
            "verify_only": True,
            "files_written": 0,
            "source_reverified": True,
            "derived_recomputed": True,
        },
        "summary_artifact": {
            "name": SUMMARY_FILENAME,
            "bytes": len(summary_payload),
            "sha256": _sha256_bytes(summary_payload),
        },
    }


def load_external_nmr_status(path: str | Path) -> dict[str, Any]:
    """Load the small machine-readable status payload for API/status surfaces."""

    summary_path = Path(path)
    try:
        value = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalNMRBenchmarkError(
            f"cannot load external NMR status: {exc}"
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != DERIVED_SCHEMA_VERSION
        or value.get("status") != "verified_smoke_only"
        or not isinstance(value.get("accuracy_eligibility"), dict)
        or value["accuracy_eligibility"].get("headline_accuracy_allowed") is not False
        or value["accuracy_eligibility"].get("open_world_claim_allowed") is not False
        or value["accuracy_eligibility"].get("independent_accuracy_test") is not False
    ):
        raise ExternalNMRBenchmarkError("external NMR status is incompatible or unsafe")
    return value
