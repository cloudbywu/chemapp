"""Safely import and freeze the manually curated NMR-Solver test set.

The only dataset payload this module will read from the pinned Zenodo archive
is ``data/experiment/test.txt``.  In particular, it never extracts the archive
and never opens or deserializes the bundled LMDB files.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile, ZipInfo

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold


SOURCE_SCHEMA_VERSION = "chemapp.nmrsolver-reviewed-source.v1"
DERIVED_SCHEMA_VERSION = "chemapp.nmrsolver-human-reviewed.v1"
SCHEMA_VERSION = DERIVED_SCHEMA_VERSION
DERIVED_RELEASE_SCHEMA_VERSION = "chemapp.nmrsolver-human-reviewed-release.v1"
ATTRIBUTION_SCHEMA_VERSION = "chemapp.third-party-attribution.v1"
DP5Q_EXPORT_SCHEMA_VERSION = "chemapp.dp5q-upstream-structure-export.v1"
DP5Q_EXPORT_ROW_SCHEMA_VERSION = "chemapp.dp5q-upstream-structure.v1"
DP5Q_ATTESTATION_SCHEMA_VERSION = (
    "chemapp.dp5q-upstream-export-attestation.v1"
)
PARENT_STANDARDIZATION_VERSION = (
    "rdkit-cleanup-fragment-parent-uncharger-single-organic-v1"
)

RECORD_ID = 16952024
CONCEPT_RECORD_ID = 16952023
SOURCE_DOI = "10.5281/zenodo.16952024"
SOURCE_TITLE = (
    "NMR-Solver: Automated Molecular Structure Elucidation via Large-Scale "
    "Spectra Matching and Physics-Guided Fragment Optimization"
)
PAPER_DOI = "10.1038/s41467-026-71315-0"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_LICENSE_URI = "https://creativecommons.org/licenses/by/4.0/legalcode"
SOURCE_CREATORS = (
    "Yongqi, Jin",
    "Junjie, Wang",
    "Fanjie, Xu",
    "Xiaohong, Ji",
    "Zhifeng, Gao",
    "Linfeng, Zhang",
    "Guolin, Ke",
    "Rong, Zhu",
    "Weinan, E",
)

ARCHIVE_NAME = "data.zip"
ARCHIVE_BYTES = 665347
ARCHIVE_MD5 = "1b22c12c977ef4e6e8c1f689c3d8a04b"
ARCHIVE_SHA256 = (
    "f250509ba4a7c56aa0aaf8d4135ca09cffdc4e6f55254e324cbab2f86f72b3ba"
)
ALLOWED_MEMBER = "data/experiment/test.txt"
ALLOWED_MEMBER_BYTES = 215710
ALLOWED_MEMBER_CRC32 = "b10d24e9"
ALLOWED_MEMBER_SHA256 = (
    "c0cf29264c07392dd9859686c03f2117a315bacfc95499110cf57a526b9762e9"
)
ALLOWED_RESPONSE_HOSTS = {"zenodo.org", "files.zenodo.org"}
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_DP5Q_EXPORT_BYTES = 16 * 1024 * 1024
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
INCHI_KEY_RE = re.compile(r"^[A-Z]{14}-[A-Z]{10}-[A-Z]$")

EXPECTED_ARCHIVE_MEMBERS = {
    "data/": (0, "directory"),
    "data/experiment/": (0, "directory"),
    "data/experiment/test.lmdb": (1241088, "forbidden_lmdb"),
    ALLOWED_MEMBER: (ALLOWED_MEMBER_BYTES, "allowed_text"),
    "data/demo/": (0, "directory"),
    "data/demo/test.lmdb": (12288, "forbidden_lmdb"),
    "data/demo/test.txt": (713, "out_of_scope_text"),
    "data/simulation/": (0, "directory"),
    "data/simulation/test.lmdb": (1933312, "forbidden_lmdb"),
    "data/README.md": (25, "out_of_scope_text"),
}

DP5Q_COMMIT = "b79968cf63cb282e8871d5595ea6cef5b4dc0d49"
DP5Q_MANIFEST_SHA256 = (
    "6be4fa6f95680d99e7b6391adfdf537fc2fc570ae4931c86896b422b884d9d30"
)
DP5Q_CONTAINER_IMAGE_ID = (
    "sha256:9cca8f49223d65a6d213d33421569dbc679e05cc7643ad8a141fe58ffb299982"
)
DP5Q_EXPORTER_SHA256 = (
    "23491c4a2bbf1239c7b1a386a27d672d37305f2ad83c392c3765f13e3b5ec58c"
)
DP5Q_EXPORT_BYTES = 11071127
DP5Q_EXPORT_SHA256 = (
    "11be3f8ccca3a97ffa1550280cad76f48b84dd2979c2219509e3db55750ddb32"
)
DP5Q_ATTESTATION_BYTES = 1689
DP5Q_ATTESTATION_SHA256 = (
    "9ea8076297f493fafceadf24910734d508006def08a7157e3d94daad8175bc8b"
)
DP5Q_ROWS_BY_SPLIT = {"train": 12349, "valid": 5000, "test": 5000}


class NMRSolverReviewedDataError(ValueError):
    """Raised when a pinned source, parser invariant or release binding fails."""


@dataclass(frozen=True)
class DP5qOverlapReference:
    """Hash-bound, safe JSON export of upstream DP5q structures."""

    canonical_smiles: frozenset[str]
    inchi_keys: frozenset[str]
    connectivity_keys: frozenset[str]
    parent_canonical_smiles: frozenset[str]
    parent_inchi_keys: frozenset[str]
    parent_connectivity_keys: frozenset[str]
    scaffolds: frozenset[str]
    binding: Mapping[str, Any]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _md5_bytes(payload: bytes) -> str:
    return hashlib.md5(payload, usedforsecurity=False).hexdigest()


def _read_snapshot(path: str | Path, *, maximum_bytes: int) -> bytes:
    selected = Path(path).resolve()
    try:
        metadata = selected.lstat()
    except OSError as exc:
        raise NMRSolverReviewedDataError(
            f"cannot inspect input file {selected}: {exc}"
        ) from exc
    if selected.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise NMRSolverReviewedDataError(
            f"input must be a regular non-symlink file: {selected}"
        )
    if metadata.st_size > maximum_bytes:
        raise NMRSolverReviewedDataError(
            f"input exceeds the {maximum_bytes}-byte safety limit: {selected}"
        )
    try:
        payload = selected.read_bytes()
    except OSError as exc:
        raise NMRSolverReviewedDataError(
            f"cannot read input file {selected}: {exc}"
        ) from exc
    if len(payload) != metadata.st_size:
        raise NMRSolverReviewedDataError(
            f"input changed while it was being read: {selected}"
        )
    return payload


def _expect(value: Any, expected: Any, field: str) -> None:
    if value != expected:
        raise NMRSolverReviewedDataError(
            f"fixed source field {field} changed: expected {expected!r}"
        )


def load_source_manifest(path: str | Path) -> dict[str, Any]:
    """Load and strictly validate the committed source and safety manifest."""

    payload = _read_snapshot(path, maximum_bytes=MAX_MANIFEST_BYTES)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise NMRSolverReviewedDataError(
            f"source manifest is not valid UTF-8 JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise NMRSolverReviewedDataError("source manifest root must be an object")
    _validate_source_manifest(value)
    return value


def _validate_source_manifest(manifest: Mapping[str, Any]) -> None:
    _expect(
        manifest.get("schema_version"),
        SOURCE_SCHEMA_VERSION,
        "schema_version",
    )
    source = manifest.get("source")
    archive = manifest.get("archive")
    policy = manifest.get("extraction_policy")
    expected = manifest.get("expected_dataset")
    dp5q = manifest.get("dp5q_upstream_overlap_audit")
    release_policy = manifest.get("release_policy")
    if not all(
        isinstance(item, Mapping)
        for item in (source, archive, policy, expected, dp5q, release_policy)
    ):
        raise NMRSolverReviewedDataError(
            "source manifest is missing a required object"
        )

    _expect(source.get("record_id"), RECORD_ID, "source.record_id")
    _expect(
        source.get("concept_record_id"),
        CONCEPT_RECORD_ID,
        "source.concept_record_id",
    )
    _expect(source.get("doi"), SOURCE_DOI, "source.doi")
    _expect(source.get("title"), SOURCE_TITLE, "source.title")
    _expect(source.get("paper_doi"), PAPER_DOI, "source.paper_doi")
    _expect(
        tuple(source.get("creators", ())),
        SOURCE_CREATORS,
        "source.creators",
    )
    license_value = source.get("license")
    if not isinstance(license_value, Mapping):
        raise NMRSolverReviewedDataError("source.license must be an object")
    _expect(
        license_value.get("spdx_id"),
        SOURCE_LICENSE,
        "source.license.spdx_id",
    )
    _expect(
        license_value.get("uri"),
        SOURCE_LICENSE_URI,
        "source.license.uri",
    )
    version_policy = source.get("version_policy")
    review = source.get("review_semantics")
    if not isinstance(version_policy, Mapping) or not isinstance(review, Mapping):
        raise NMRSolverReviewedDataError(
            "source version and review policies are required"
        )
    _expect(
        version_policy.get("concept_or_latest_resolution_allowed"),
        False,
        "source.version_policy.concept_or_latest_resolution_allowed",
    )
    _expect(
        review.get("review_kind"),
        "upstream_manually_curated_benchmark",
        "source.review_semantics.review_kind",
    )
    for field in (
        "chemapp_double_reviewed",
        "row_level_reviewer_identity_available",
        "row_level_review_timestamp_available",
        "source_document_mapping_available",
    ):
        _expect(review.get(field), False, f"source.review_semantics.{field}")

    _expect(archive.get("name"), ARCHIVE_NAME, "archive.name")
    _expect(archive.get("bytes"), ARCHIVE_BYTES, "archive.bytes")
    _expect(archive.get("md5"), ARCHIVE_MD5, "archive.md5")
    _expect(archive.get("sha256"), ARCHIVE_SHA256, "archive.sha256")
    url = archive.get("url")
    if (
        not isinstance(url, str)
        or urlparse(url).scheme != "https"
        or urlparse(url).hostname != "zenodo.org"
    ):
        raise NMRSolverReviewedDataError("archive URL is not the pinned HTTPS URL")
    members = archive.get("members")
    if not isinstance(members, list) or len(members) != len(
        EXPECTED_ARCHIVE_MEMBERS
    ):
        raise NMRSolverReviewedDataError("archive member inventory changed")
    inventory: dict[str, Mapping[str, Any]] = {}
    for item in members:
        if not isinstance(item, Mapping) or not isinstance(item.get("name"), str):
            raise NMRSolverReviewedDataError(
                "archive member inventory contains an invalid entry"
            )
        name = str(item["name"])
        if name in inventory:
            raise NMRSolverReviewedDataError(
                f"archive manifest contains duplicate member {name}"
            )
        inventory[name] = item
    if set(inventory) != set(EXPECTED_ARCHIVE_MEMBERS):
        raise NMRSolverReviewedDataError("archive member names changed")
    for name, (size, kind) in EXPECTED_ARCHIVE_MEMBERS.items():
        _expect(inventory[name].get("bytes"), size, f"archive.members[{name}].bytes")
        _expect(inventory[name].get("kind"), kind, f"archive.members[{name}].kind")
    _expect(
        inventory[ALLOWED_MEMBER].get("sha256"),
        ALLOWED_MEMBER_SHA256,
        "allowed member sha256",
    )
    _expect(
        inventory[ALLOWED_MEMBER].get("crc32"),
        ALLOWED_MEMBER_CRC32,
        "allowed member crc32",
    )

    _expect(
        policy.get("allowed_readable_members"),
        [ALLOWED_MEMBER],
        "extraction_policy.allowed_readable_members",
    )
    _expect(
        policy.get("archive_extraction_allowed"),
        False,
        "extraction_policy.archive_extraction_allowed",
    )
    _expect(
        policy.get("lmdb_open_or_deserialize_allowed"),
        False,
        "extraction_policy.lmdb_open_or_deserialize_allowed",
    )
    if (
        not isinstance(policy.get("maximum_archive_entries"), int)
        or int(policy["maximum_archive_entries"]) < len(EXPECTED_ARCHIVE_MEMBERS)
        or not isinstance(policy.get("maximum_total_expanded_bytes"), int)
        or int(policy["maximum_total_expanded_bytes"]) < sum(
            size for size, _kind in EXPECTED_ARCHIVE_MEMBERS.values()
        )
        or not isinstance(policy.get("maximum_allowed_member_bytes"), int)
        or int(policy["maximum_allowed_member_bytes"]) < ALLOWED_MEMBER_BYTES
    ):
        raise NMRSolverReviewedDataError(
            "archive extraction limits are too small or invalid"
        )

    fixed_counts = {
        "record_count": 450,
        "lines_per_record": 4,
        "h_peak_group_count": 3955,
        "h_reported_integral_total": 8957,
        "c_shift_count": 6340,
        "c_upstream_compatible_shift_count": 6350,
        "records_with_appended_heteronuclear_text": 10,
        "records_with_h_peaks": 450,
        "records_with_c_shifts": 450,
        "unique_block_payload_count": 450,
        "unique_paired_spectrum_count": 450,
        "unique_product_inchikey_count": 449,
    }
    for field, fixed in fixed_counts.items():
        _expect(expected.get(field), fixed, f"expected_dataset.{field}")
    _expect(
        expected.get("appended_heteronuclear_records_by_nucleus"),
        {"11B": 4, "19F": 1, "31P": 5},
        "expected_dataset.appended_heteronuclear_records_by_nucleus",
    )
    duplicate_groups = expected.get("duplicate_molecule_groups")
    if (
        not isinstance(duplicate_groups, list)
        or len(duplicate_groups) != 1
        or duplicate_groups[0].get("product_inchikey")
        != "RUIRKYJVQVAVGV-JTQLQIEISA-N"
        or duplicate_groups[0].get("zero_based_ordinals") != [77, 93]
    ):
        raise NMRSolverReviewedDataError(
            "expected duplicate molecule group changed"
        )

    _expect(
        dp5q.get("required_for_release"),
        True,
        "dp5q_upstream_overlap_audit.required_for_release",
    )
    _expect(
        dp5q.get("repository_commit"),
        DP5Q_COMMIT,
        "dp5q_upstream_overlap_audit.repository_commit",
    )
    _expect(
        dp5q.get("fixed_asset_manifest_sha256"),
        DP5Q_MANIFEST_SHA256,
        "dp5q_upstream_overlap_audit.fixed_asset_manifest_sha256",
    )
    export = dp5q.get("export")
    attestation = dp5q.get("attestation")
    overlap_counts = dp5q.get("expected_nmrsolver_counts")
    if not all(
        isinstance(item, Mapping)
        for item in (export, attestation, overlap_counts)
    ):
        raise NMRSolverReviewedDataError("DP5q audit bindings are incomplete")
    _expect(export.get("bytes"), DP5Q_EXPORT_BYTES, "DP5q export bytes")
    _expect(export.get("sha256"), DP5Q_EXPORT_SHA256, "DP5q export sha256")
    _expect(
        export.get("rows_by_split"),
        DP5Q_ROWS_BY_SPLIT,
        "DP5q rows_by_split",
    )
    _expect(
        attestation.get("bytes"),
        DP5Q_ATTESTATION_BYTES,
        "DP5q attestation bytes",
    )
    _expect(
        attestation.get("sha256"),
        DP5Q_ATTESTATION_SHA256,
        "DP5q attestation sha256",
    )
    _expect(
        overlap_counts,
        {
            "structure_overlap_records": 7,
            "scaffold_overlap_records": 190,
            "structure_and_scaffold_independent_records": 260,
        },
        "DP5q expected NMR-Solver counts",
    )
    for field in (
        "model_scores_included",
        "prediction_outcomes_included",
        "headline_accuracy_allowed",
    ):
        _expect(release_policy.get(field), False, f"release_policy.{field}")
    _expect(
        release_policy.get("partition_status"),
        "not_assigned",
        "release_policy.partition_status",
    )


def _safe_member_name(info: ZipInfo) -> None:
    name = info.filename
    posix = PurePosixPath(name)
    windows = PureWindowsPath(name)
    if (
        not name
        or "\\" in name
        or posix.is_absolute()
        or windows.is_absolute()
        or ".." in posix.parts
        or "\x00" in name
    ):
        raise NMRSolverReviewedDataError(
            f"archive contains an unsafe member path: {name!r}"
        )
    mode = info.external_attr >> 16
    if mode and stat.S_ISLNK(mode):
        raise NMRSolverReviewedDataError(
            f"archive contains a symbolic-link member: {name}"
        )
    if info.flag_bits & 0x1:
        raise NMRSolverReviewedDataError(
            f"archive contains an encrypted member: {name}"
        )


def read_allowlisted_experiment_member(
    archive_snapshot: bytes,
    archive_spec: Mapping[str, Any],
    extraction_policy: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Validate an archive inventory and read exactly one allowlisted member."""

    allowed = extraction_policy.get("allowed_readable_members")
    if allowed != [ALLOWED_MEMBER]:
        raise NMRSolverReviewedDataError(
            "exactly data/experiment/test.txt must be allowlisted"
        )
    maximum_entries = extraction_policy.get("maximum_archive_entries")
    maximum_expanded = extraction_policy.get("maximum_total_expanded_bytes")
    maximum_member = extraction_policy.get("maximum_allowed_member_bytes")
    if not all(
        isinstance(value, int) and value > 0
        for value in (maximum_entries, maximum_expanded, maximum_member)
    ):
        raise NMRSolverReviewedDataError("archive safety limits are invalid")
    manifest_members = archive_spec.get("members")
    if not isinstance(manifest_members, list):
        raise NMRSolverReviewedDataError("archive manifest members are invalid")
    expected_inventory = {
        str(item["name"]): item
        for item in manifest_members
        if isinstance(item, Mapping) and isinstance(item.get("name"), str)
    }
    if len(expected_inventory) != len(manifest_members):
        raise NMRSolverReviewedDataError(
            "archive manifest contains duplicate or invalid members"
        )

    try:
        with ZipFile(io.BytesIO(archive_snapshot), mode="r") as archive:
            infos = archive.infolist()
            if len(infos) > maximum_entries:
                raise NMRSolverReviewedDataError(
                    "archive contains too many entries"
                )
            actual: dict[str, ZipInfo] = {}
            expanded_bytes = 0
            for info in infos:
                _safe_member_name(info)
                if info.filename in actual:
                    raise NMRSolverReviewedDataError(
                        f"archive contains duplicate member {info.filename}"
                    )
                actual[info.filename] = info
                expanded_bytes += info.file_size
            if expanded_bytes > maximum_expanded:
                raise NMRSolverReviewedDataError(
                    "archive expanded-byte limit exceeded"
                )
            if set(actual) != set(expected_inventory):
                raise NMRSolverReviewedDataError(
                    "archive central-directory inventory changed"
                )
            for name, expected in expected_inventory.items():
                info = actual[name]
                if info.file_size != expected.get("bytes"):
                    raise NMRSolverReviewedDataError(
                        f"archive member size changed: {name}"
                    )
                should_be_directory = expected.get("kind") == "directory"
                if info.is_dir() != should_be_directory:
                    raise NMRSolverReviewedDataError(
                        f"archive member type changed: {name}"
                    )
            selected = actual[ALLOWED_MEMBER]
            if selected.file_size > maximum_member:
                raise NMRSolverReviewedDataError(
                    "allowlisted member exceeds its byte limit"
                )
            expected_crc = str(
                expected_inventory[ALLOWED_MEMBER].get("crc32", "")
            ).lower()
            if f"{selected.CRC:08x}" != expected_crc:
                raise NMRSolverReviewedDataError(
                    "allowlisted member CRC32 changed"
                )
            # This is intentionally the sole ZipFile.read call.  LMDB and all
            # other text members remain unopened.
            member = archive.read(ALLOWED_MEMBER)
    except (BadZipFile, RuntimeError, OSError) as exc:
        raise NMRSolverReviewedDataError(
            f"cannot safely inspect pinned archive: {exc}"
        ) from exc
    expected_sha256 = expected_inventory[ALLOWED_MEMBER].get("sha256")
    if (
        len(member) != expected_inventory[ALLOWED_MEMBER].get("bytes")
        or _sha256_bytes(member) != expected_sha256
    ):
        raise NMRSolverReviewedDataError(
            "allowlisted member size or SHA-256 changed"
        )
    return member, {
        "archive_entries": len(expected_inventory),
        "expanded_bytes": sum(
            int(item["bytes"]) for item in expected_inventory.values()
        ),
        "read_members": [ALLOWED_MEMBER],
        "lmdb_members_opened": False,
        "archive_extracted": False,
        "member_bytes": len(member),
        "member_sha256": _sha256_bytes(member),
    }


def verify_archive_file(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Verify transport bytes and return only the allowlisted text payload."""

    archive = manifest["archive"]
    expected_bytes = int(archive["bytes"])
    snapshot = _read_snapshot(path, maximum_bytes=expected_bytes)
    if (
        len(snapshot) != expected_bytes
        or _md5_bytes(snapshot) != archive["md5"]
        or _sha256_bytes(snapshot) != archive["sha256"]
    ):
        raise NMRSolverReviewedDataError(
            "archive bytes do not match the fixed size, MD5 and SHA-256"
        )
    member, zip_audit = read_allowlisted_experiment_member(
        snapshot,
        archive,
        manifest["extraction_policy"],
    )
    return member, {
        "archive": {
            "file": Path(path).name,
            "bytes": len(snapshot),
            "md5": _md5_bytes(snapshot),
            "sha256": _sha256_bytes(snapshot),
        },
        "zip_safety": zip_audit,
    }


def download_archive(
    manifest: Mapping[str, Any],
    destination: str | Path,
    *,
    timeout_seconds: float = 60.0,
    overwrite: bool = False,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, Any]:
    """Atomically download the pinned archive after streaming hash checks."""

    target = Path(destination).resolve()
    if target.exists() and not overwrite:
        _member, audit = verify_archive_file(target, manifest)
        return {**audit["archive"], "reused": True}
    target.parent.mkdir(parents=True, exist_ok=True)
    archive = manifest["archive"]
    url = str(archive["url"])
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".nmrsolver-download-",
            suffix=".part",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            request = Request(
                url,
                headers={"User-Agent": "ChemApp-NMRSolver-reviewed-import/1"},
            )
            with opener(request, timeout=timeout_seconds) as response:
                final_url = str(response.geturl())
                parsed = urlparse(final_url)
                if (
                    parsed.scheme != "https"
                    or parsed.hostname not in ALLOWED_RESPONSE_HOSTS
                ):
                    raise NMRSolverReviewedDataError(
                        f"archive download redirected off allowlist: {final_url}"
                    )
                digest = hashlib.sha256()
                md5_digest = hashlib.md5(usedforsecurity=False)
                total = 0
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > int(archive["bytes"]):
                        raise NMRSolverReviewedDataError(
                            "archive download exceeded fixed byte count"
                        )
                    digest.update(chunk)
                    md5_digest.update(chunk)
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())
        if (
            total != archive["bytes"]
            or digest.hexdigest() != archive["sha256"]
            or md5_digest.hexdigest() != archive["md5"]
        ):
            raise NMRSolverReviewedDataError(
                "downloaded archive failed fixed size/MD5/SHA-256 checks"
            )
        os.replace(temporary, target)
        temporary = None
    except NMRSolverReviewedDataError:
        raise
    except (OSError, TimeoutError) as exc:
        raise NMRSolverReviewedDataError(
            f"archive download failed: {exc}"
        ) from exc
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    _member, audit = verify_archive_file(target, manifest)
    return {**audit["archive"], "reused": False}


def load_dp5q_overlap_reference(
    export_path: str | Path,
    attestation_path: str | Path,
    manifest: Mapping[str, Any],
) -> DP5qOverlapReference:
    """Load only the attested JSON export; never touch upstream pickle files."""

    audit_spec = manifest["dp5q_upstream_overlap_audit"]
    export_spec = audit_spec["export"]
    attestation_spec = audit_spec["attestation"]
    export_bytes = _read_snapshot(
        export_path,
        maximum_bytes=int(export_spec["bytes"]),
    )
    attestation_bytes = _read_snapshot(
        attestation_path,
        maximum_bytes=int(attestation_spec["bytes"]),
    )
    if (
        len(export_bytes) != export_spec["bytes"]
        or _sha256_bytes(export_bytes) != export_spec["sha256"]
    ):
        raise NMRSolverReviewedDataError(
            "DP5q safe JSONL export failed its fixed byte/hash binding"
        )
    if (
        len(attestation_bytes) != attestation_spec["bytes"]
        or _sha256_bytes(attestation_bytes) != attestation_spec["sha256"]
    ):
        raise NMRSolverReviewedDataError(
            "DP5q attestation failed its fixed byte/hash binding"
        )
    try:
        attestation = json.loads(attestation_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise NMRSolverReviewedDataError(
            f"DP5q attestation is invalid UTF-8 JSON: {exc}"
        ) from exc
    required_attestation = {
        "schema_version": DP5Q_ATTESTATION_SCHEMA_VERSION,
        "repository_commit": DP5Q_COMMIT,
        "manifest_sha256": DP5Q_MANIFEST_SHA256,
        "container_image_id": DP5Q_CONTAINER_IMAGE_ID,
        "exporter_sha256": DP5Q_EXPORTER_SHA256,
        "export_sha256": DP5Q_EXPORT_SHA256,
        "rows_by_split": DP5Q_ROWS_BY_SPLIT,
    }
    if not isinstance(attestation, Mapping):
        raise NMRSolverReviewedDataError("DP5q attestation root is not an object")
    for field, expected in required_attestation.items():
        _expect(attestation.get(field), expected, f"DP5q attestation.{field}")
    isolation = attestation.get("isolation")
    if (
        not isinstance(isolation, Mapping)
        or isolation.get("network") != "none"
        or isolation.get("root_filesystem") != "read-only"
        or isolation.get("source_mount") != "read-only"
        or isolation.get("capabilities") != "none"
        or isolation.get("no_new_privileges") is not True
    ):
        raise NMRSolverReviewedDataError(
            "DP5q export isolation attestation is incomplete"
        )

    canonical: set[str] = set()
    inchi_keys: set[str] = set()
    connectivity: set[str] = set()
    parent_canonical: set[str] = set()
    parent_inchi: set[str] = set()
    parent_connectivity: set[str] = set()
    scaffolds: set[str] = set()
    rows_by_split: Counter[str] = Counter()
    seen_ordinals: set[tuple[str, int]] = set()
    try:
        lines = export_bytes.decode("utf-8").splitlines()
    except UnicodeError as exc:
        raise NMRSolverReviewedDataError(
            f"DP5q export is not UTF-8: {exc}"
        ) from exc
    if len(lines) != sum(DP5Q_ROWS_BY_SPLIT.values()):
        raise NMRSolverReviewedDataError("DP5q export row count changed")
    required_strings = (
        "canonical_isomeric_smiles",
        "inchi_key",
        "connectivity_key",
        "parent_canonical_isomeric_smiles",
        "parent_inchi_key",
        "parent_connectivity_key",
        "murcko_scaffold_smiles",
    )
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise NMRSolverReviewedDataError(
                f"DP5q export row {line_number} is invalid JSON: {exc}"
            ) from exc
        if (
            not isinstance(row, Mapping)
            or row.get("schema_version") != DP5Q_EXPORT_ROW_SCHEMA_VERSION
            or row.get("source_commit") != DP5Q_COMMIT
            or row.get("split") not in DP5Q_ROWS_BY_SPLIT
            or not isinstance(row.get("source_row_ordinal"), int)
            or any(not isinstance(row.get(field), str) for field in required_strings)
        ):
            raise NMRSolverReviewedDataError(
                f"DP5q export row {line_number} failed its schema"
            )
        split = str(row["split"])
        ordinal = int(row["source_row_ordinal"])
        if (split, ordinal) in seen_ordinals:
            raise NMRSolverReviewedDataError(
                f"DP5q export duplicated row identity {split}:{ordinal}"
            )
        seen_ordinals.add((split, ordinal))
        rows_by_split[split] += 1
        canonical.add(str(row["canonical_isomeric_smiles"]))
        inchi_keys.add(str(row["inchi_key"]))
        connectivity.add(str(row["connectivity_key"]))
        parent_canonical.add(str(row["parent_canonical_isomeric_smiles"]))
        parent_inchi.add(str(row["parent_inchi_key"]))
        parent_connectivity.add(str(row["parent_connectivity_key"]))
        scaffolds.add(str(row["murcko_scaffold_smiles"]) or "acyclic")
    if dict(rows_by_split) != DP5Q_ROWS_BY_SPLIT:
        raise NMRSolverReviewedDataError("DP5q export split counts changed")

    return DP5qOverlapReference(
        canonical_smiles=frozenset(canonical),
        inchi_keys=frozenset(inchi_keys),
        connectivity_keys=frozenset(connectivity),
        parent_canonical_smiles=frozenset(parent_canonical),
        parent_inchi_keys=frozenset(parent_inchi),
        parent_connectivity_keys=frozenset(parent_connectivity),
        scaffolds=frozenset(scaffolds),
        binding={
            "schema_version": DP5Q_EXPORT_SCHEMA_VERSION,
            "repository_commit": DP5Q_COMMIT,
            "fixed_asset_manifest_sha256": DP5Q_MANIFEST_SHA256,
            "container_image_id": DP5Q_CONTAINER_IMAGE_ID,
            "exporter_sha256": DP5Q_EXPORTER_SHA256,
            "export": {
                "file": Path(export_path).name,
                "bytes": len(export_bytes),
                "sha256": _sha256_bytes(export_bytes),
                "rows_by_split": dict(rows_by_split),
            },
            "attestation": {
                "file": Path(attestation_path).name,
                "bytes": len(attestation_bytes),
                "sha256": _sha256_bytes(attestation_bytes),
            },
            "host_deserialized_pickle": False,
            "limitations": list(audit_spec["limitations"]),
        },
    )


_H_PATTERN = re.compile(
    r"(?P<shift_left_1>-?\d+\.\d+)\s*(?:–|-)\s*"
    r"(?P<shift_right_1>-?\d+\.\d+)\s*\("
    r"(?P<type_1>[a-zA-Z\s\.,]*?)?,?\s*"
    r"(?:(?:J\s*=\s*(?P<hz_1>[\d\.\s,]+)\s*Hz,\s*)?"
    r"(?P<count_1>\d+(?:\.\d+)?)\s*H"
    r"(?:,\s*(?P<annotation_1>[^)]+))?|"
    r"(?P<count_alt_1>\d+(?:\.\d+)?)\s*H,\s*J\s*=\s*"
    r"(?P<hz_alt_1>[\d\.\s,]+)\s*Hz"
    r"(?:,\s*(?P<annotation_alt_1>[^)]+))?)\)"
    r"|(?P<shift_2>-?\d+\.\d+)\s*\("
    r"(?P<type_2>[a-zA-Z\s\.,]*?)?,?\s*"
    r"(?:(?:J\s*=\s*(?P<hz_2>[\d\.\s,]+)\s*Hz,\s*)?"
    r"(?P<count_2>\d+(?:\.\d+)?)\s*H"
    r"(?:,\s*(?P<annotation_2>[^)]+))?|"
    r"(?P<count_alt_2>\d+(?:\.\d+)?)\s*H,\s*J\s*=\s*"
    r"(?P<hz_alt_2>[\d\.\s,]+)\s*Hz"
    r"(?:,\s*(?P<annotation_alt_2>[^)]+))?)\)"
)
_FLOAT_RE = re.compile(r"-?\d+\.\d+")
_FREQUENCY_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*MHz", re.IGNORECASE)
_NUCLEUS_HEADER_RE = re.compile(
    r"(?P<nucleus>\d{1,2}[A-Z][a-z]?)\s*(?:\{[^}]*\}\s*)?NMR",
    re.IGNORECASE,
)


def _number(value: str) -> int | float:
    parsed = float(value)
    return int(parsed) if parsed.is_integer() else parsed


def parse_h_peaks(raw_text: str) -> list[dict[str, Any]]:
    """Parse the published proton groups using the upstream parser semantics."""

    peaks: list[dict[str, Any]] = []
    for ordinal, match in enumerate(_H_PATTERN.finditer(raw_text)):
        if match.group("shift_left_1") is not None:
            left = float(match.group("shift_left_1"))
            right = float(match.group("shift_right_1"))
            multiplicity = match.group("type_1")
            couplings = match.group("hz_1") or match.group("hz_alt_1")
            integral = match.group("count_1") or match.group("count_alt_1")
            annotation = (
                match.group("annotation_1")
                or match.group("annotation_alt_1")
            )
        else:
            left = right = float(match.group("shift_2"))
            multiplicity = match.group("type_2")
            couplings = match.group("hz_2") or match.group("hz_alt_2")
            integral = match.group("count_2") or match.group("count_alt_2")
            annotation = (
                match.group("annotation_2")
                or match.group("annotation_alt_2")
            )
        if integral is None:
            raise NMRSolverReviewedDataError(
                "proton parser matched a group without an integral"
            )
        range_ppm = sorted((left, right))
        peaks.append(
            {
                "ordinal": ordinal,
                "shift_ppm": sum(range_ppm) / 2.0,
                "range_ppm": range_ppm,
                "multiplicity": (
                    multiplicity.strip(" ,.") if multiplicity else None
                ),
                "couplings_hz": (
                    [float(item) for item in _FLOAT_RE.findall(couplings)]
                    if couplings
                    else []
                ),
                "reported_integral": _number(integral),
                "annotation": annotation.strip() if annotation else None,
            }
        )
    return peaks


def parse_c_shifts(raw_text: str) -> list[float]:
    """Parse and sort carbon shifts with the published exclusion semantics."""

    shifts: list[float] = []
    for token in _FLOAT_RE.findall(raw_text):
        escaped = re.escape(token)
        if re.search(rf"J\s*=\s*{escaped}", raw_text):
            continue
        if re.search(rf"{escaped}\s*Hz", raw_text):
            continue
        shifts.append(float(token))
    return sorted(shifts)


def _split_primary_c_text(
    raw_text: str,
) -> tuple[str, str | None, str | None]:
    headers = list(_NUCLEUS_HEADER_RE.finditer(raw_text))
    if not headers or headers[0].group("nucleus").upper() != "13C":
        raise NMRSolverReviewedDataError(
            "carbon text does not begin with a 13C NMR header"
    )
    if len(headers) == 1:
        return raw_text, None, None
    boundary = headers[1].start()
    nucleus = headers[1].group("nucleus").upper()
    return raw_text[:boundary].rstrip(), raw_text[boundary:].strip(), nucleus


def _parenthesized_spans(text: str) -> list[tuple[int, int]]:
    stack: list[int] = []
    spans: list[tuple[int, int]] = []
    for index, character in enumerate(text):
        if character == "(":
            stack.append(index)
        elif character == ")" and stack:
            spans.append((stack.pop(), index))
    return spans


_SOLVENT_ALIASES = {
    "cdcl3": "chloroform-d",
    "chloroform-d": "chloroform-d",
    "chloroform-d1": "chloroform-d",
    "d-chloroform": "chloroform-d",
    "dmso": "dmso-d6",
    "dmso-d6": "dmso-d6",
    "dmso-d6 dmso-d6": "dmso-d6",
    "dmsod6": "dmso-d6",
    "d6-dmso": "dmso-d6",
    "(cd3)2so": "dmso-d6",
    "c6d6": "benzene-d6",
    "cd2cl2": "dichloromethane-d2",
    "methylene chloride-d2": "dichloromethane-d2",
    "cd3cn": "acetonitrile-d3",
    "cd3od": "methanol-d4",
    "meod": "methanol-d4",
    "methanol-d4": "methanol-d4",
    "d2o": "d2o",
    "d2o+naoh": "d2o+naoh",
    "acetone-d6": "acetone-d6",
    "acetone": "acetone",
    "pyridine": "pyridine",
}


def parse_nmr_header(raw_text: str, nucleus: str) -> dict[str, Any]:
    """Parse frequency and solvent without assuming their header order."""

    if nucleus not in {"1H", "13C"} or not raw_text.lstrip().startswith(
        f"{nucleus} NMR"
    ):
        raise NMRSolverReviewedDataError(
            f"{nucleus} spectrum does not start with its expected header"
        )
    matches = list(_FREQUENCY_RE.finditer(raw_text))
    if not matches:
        raise NMRSolverReviewedDataError(
            f"{nucleus} header does not contain an MHz field"
        )
    # Some 13C lines append a 11B, 19F or 31P report.  The first MHz field
    # belongs to the leading nucleus named above; later fields are not headers
    # for this spectrum and must not make metadata parsing ambiguous.
    frequency = matches[0]
    containing = [
        span
        for span in _parenthesized_spans(raw_text)
        if span[0] < frequency.start() and span[1] >= frequency.end()
    ]
    if not containing:
        raise NMRSolverReviewedDataError(
            f"{nucleus} MHz field is not inside a balanced header"
        )
    start, end = min(containing, key=lambda span: span[1] - span[0])
    header_body = raw_text[start + 1 : end]
    solvent_parts = [
        part.strip(" :")
        for part in header_body.split(",")
        if "mhz" not in part.lower()
    ]
    solvent_raw = ", ".join(part for part in solvent_parts if part)
    if not solvent_raw:
        raise NMRSolverReviewedDataError(
            f"{nucleus} header does not contain a solvent token"
        )
    value = float(frequency.group("value"))
    return {
        "field_mhz": int(value) if value.is_integer() else value,
        "solvent_raw": solvent_raw,
        "solvent": _SOLVENT_ALIASES.get(solvent_raw.casefold()),
    }


def _structure_identity(smiles: str) -> tuple[dict[str, Any], Chem.Mol]:
    blocker = rdBase.BlockLogs()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise NMRSolverReviewedDataError(
                f"RDKit could not parse product SMILES: {smiles}"
            )
        if len(Chem.GetMolFrags(mol)) != 1:
            raise NMRSolverReviewedDataError(
                f"product is not a single fragment: {smiles}"
            )
        reported_canonical = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
        reported_inchi_key = str(Chem.MolToInchiKey(mol) or "").upper()
        reported_scaffold = (
            MurckoScaffold.MurckoScaffoldSmiles(mol=mol) or "acyclic"
        )
        cleaned = rdMolStandardize.Cleanup(Chem.Mol(mol))
        parent = rdMolStandardize.FragmentParent(
            cleaned,
            skipStandardize=True,
        )
        parent = rdMolStandardize.Uncharger().uncharge(parent)
        Chem.SanitizeMol(parent)
        if len(Chem.GetMolFrags(parent)) != 1 or not any(
            atom.GetAtomicNum() == 6 for atom in parent.GetAtoms()
        ):
            raise NMRSolverReviewedDataError(
                f"product has no eligible single organic parent: {smiles}"
            )
        canonical = Chem.MolToSmiles(
            parent,
            canonical=True,
            isomericSmiles=True,
        )
        inchi_key = str(Chem.MolToInchiKey(parent) or "").upper()
        if not INCHI_KEY_RE.fullmatch(inchi_key):
            raise NMRSolverReviewedDataError(
                f"RDKit did not produce a stable product InChIKey: {smiles}"
            )
        parent_scaffold = (
            MurckoScaffold.MurckoScaffoldSmiles(mol=parent) or "acyclic"
        )
        carbon_count = sum(
            atom.GetAtomicNum() == 6 for atom in parent.GetAtoms()
        )
        return (
            {
                "standardization_version": PARENT_STANDARDIZATION_VERSION,
                "reported_smiles": smiles,
                "reported_canonical_smiles": reported_canonical,
                "reported_inchi_key": reported_inchi_key,
                "canonical_smiles": canonical,
                "inchi_key": inchi_key,
                "connectivity_key": inchi_key[:14],
                "formula": rdMolDescriptors.CalcMolFormula(parent),
                "reported_scaffold_smiles": reported_scaffold,
                # The attested DP5q exporter computes Murcko on the
                # Cleanup/FragmentParent/Uncharger standardized parent.
                "scaffold_smiles": parent_scaffold,
                "carbon_count": carbon_count,
            },
            parent,
        )
    except NMRSolverReviewedDataError:
        raise
    except Exception as exc:
        raise NMRSolverReviewedDataError(
            f"RDKit product standardization failed for {smiles}: {exc}"
        ) from exc
    finally:
        del blocker


def _reactant_identity(smiles: str) -> dict[str, Any]:
    blocker = rdBase.BlockLogs()
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise NMRSolverReviewedDataError(
                f"RDKit could not parse reactant SMILES: {smiles}"
            )
        return {
            "reported_smiles": smiles,
            "canonical_smiles": Chem.MolToSmiles(
                mol,
                canonical=True,
                isomericSmiles=True,
            ),
        }
    finally:
        del blocker


def _split_four_line_blocks(member_bytes: bytes) -> list[list[str]]:
    try:
        text = member_bytes.decode("utf-8")
    except UnicodeError as exc:
        raise NMRSolverReviewedDataError(
            f"allowlisted test.txt is not valid UTF-8: {exc}"
        ) from exc
    if "\r" in text or "\x00" in text:
        raise NMRSolverReviewedDataError(
            "allowlisted test.txt contains forbidden CR or NUL bytes"
        )
    blocks: list[list[str]] = []
    current: list[str] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if line == "":
            if not current:
                raise NMRSolverReviewedDataError(
                    f"empty or repeated block separator at line {line_number}"
                )
            blocks.append(current)
            current = []
        else:
            current.append(line)
    if current:
        blocks.append(current)
    for ordinal, block in enumerate(blocks):
        if len(block) != 4 or any(not line.strip() for line in block):
            raise NMRSolverReviewedDataError(
                f"record {ordinal} must contain exactly four nonempty lines"
            )
    return blocks


def _matches_dp5q(
    structure: Mapping[str, Any],
    reference: DP5qOverlapReference,
) -> tuple[bool, bool]:
    structure_overlap = (
        structure["canonical_smiles"] in reference.canonical_smiles
        or structure["canonical_smiles"] in reference.parent_canonical_smiles
        or structure["reported_canonical_smiles"] in reference.canonical_smiles
        or structure["inchi_key"] in reference.inchi_keys
        or structure["inchi_key"] in reference.parent_inchi_keys
        or structure["reported_inchi_key"] in reference.inchi_keys
        or structure["connectivity_key"] in reference.connectivity_keys
        or structure["connectivity_key"] in reference.parent_connectivity_keys
    )
    scaffold_overlap = structure["scaffold_smiles"] in reference.scaffolds
    return structure_overlap, scaffold_overlap


def parse_experiment_text(
    member_bytes: bytes,
    *,
    expected: Mapping[str, Any],
    dp5q_reference: DP5qOverlapReference,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Parse paired spectra, enforce counts and retain duplicate molecules."""

    blocks = _split_four_line_blocks(member_bytes)
    expected_records = expected.get("record_count")
    if not isinstance(expected_records, int) or len(blocks) != expected_records:
        raise NMRSolverReviewedDataError(
            f"expected {expected_records} four-line records, got {len(blocks)}"
        )
    records: list[dict[str, Any]] = []
    h_peak_count = 0
    h_integral_total: int | float = 0
    c_shift_count = 0
    c_upstream_compatible_shift_count = 0
    records_with_appended_heteronuclear_text = 0
    appended_nuclei: Counter[str] = Counter()
    h_nonempty = 0
    c_nonempty = 0
    block_hashes: set[str] = set()
    spectrum_hashes: set[str] = set()
    product_keys: list[str] = []
    molecule_groups: defaultdict[str, list[int]] = defaultdict(list)

    for ordinal, block in enumerate(blocks):
        reactants_line, product_smiles, h_raw, c_raw = block
        structure, _parent = _structure_identity(product_smiles)
        reactant_tokens = reactants_line.split(", ")
        if not reactant_tokens or any(not token for token in reactant_tokens):
            raise NMRSolverReviewedDataError(
                f"record {ordinal} has an invalid reactant list"
            )
        reactants = [_reactant_identity(token) for token in reactant_tokens]
        h_header = parse_nmr_header(h_raw, "1H")
        c_header = parse_nmr_header(c_raw, "13C")
        h_peaks = parse_h_peaks(h_raw)
        (
            primary_c_raw,
            appended_heteronuclear_raw,
            appended_heteronuclear_nucleus,
        ) = _split_primary_c_text(c_raw)
        c_shifts = parse_c_shifts(primary_c_raw)
        upstream_compatible_c_shifts = parse_c_shifts(c_raw)
        if h_peaks:
            h_nonempty += 1
        if c_shifts:
            c_nonempty += 1
        h_peak_count += len(h_peaks)
        h_integral_total += sum(
            peak["reported_integral"] for peak in h_peaks
        )
        c_shift_count += len(c_shifts)
        c_upstream_compatible_shift_count += len(
            upstream_compatible_c_shifts
        )
        if appended_heteronuclear_raw is not None:
            records_with_appended_heteronuclear_text += 1
            if appended_heteronuclear_nucleus is None:
                raise NMRSolverReviewedDataError(
                    f"record {ordinal} has untyped appended NMR text"
                )
            appended_nuclei[appended_heteronuclear_nucleus] += 1
        expanded_h_shifts: list[float] = []
        for peak in h_peaks:
            integral = peak["reported_integral"]
            if not isinstance(integral, int):
                raise NMRSolverReviewedDataError(
                    f"record {ordinal} has a non-integral proton count"
                )
            expanded_h_shifts.extend([peak["shift_ppm"]] * integral)
        block_bytes = "\n".join(block).encode("utf-8")
        block_sha256 = _sha256_bytes(block_bytes)
        spectrum_fingerprint = _sha256_bytes(
            _canonical_json({"1H": h_raw, "13C": c_raw}).encode("utf-8")
        )
        block_hashes.add(block_sha256)
        spectrum_hashes.add(spectrum_fingerprint)
        product_keys.append(str(structure["inchi_key"]))
        group_id = f"inchi-connectivity:{structure['connectivity_key']}"
        molecule_groups[group_id].append(ordinal)
        dp5_structure, dp5_scaffold = _matches_dp5q(
            structure,
            dp5q_reference,
        )
        records.append(
            {
                "schema_version": DERIVED_SCHEMA_VERSION,
                "record_id": f"nmrsolver:{block_sha256[:24]}",
                "source": {
                    "dataset_id": "nmrsolver-experimental-test-zenodo-16952024",
                    "record_id": RECORD_ID,
                    "doi": SOURCE_DOI,
                    "member": ALLOWED_MEMBER,
                    "member_sha256": ALLOWED_MEMBER_SHA256,
                    "ordinal": ordinal,
                    "ordinal_base": 0,
                    "block_sha256": block_sha256,
                },
                "reaction": {
                    "reactants_raw": reactants_line,
                    "reactants": reactants,
                },
                "structure": structure,
                "spectra": {
                    "1H": {
                        "raw_text": h_raw,
                        "raw_text_sha256": _sha256_bytes(
                            h_raw.encode("utf-8")
                        ),
                        **h_header,
                        "peak_groups": h_peaks,
                        "shifts_ppm": sorted(expanded_h_shifts),
                    },
                    "13C": {
                        "raw_text": c_raw,
                        "raw_text_sha256": _sha256_bytes(
                            c_raw.encode("utf-8")
                        ),
                        **c_header,
                        "shifts_ppm": c_shifts,
                        "upstream_compatible_shifts_ppm": (
                            upstream_compatible_c_shifts
                        ),
                        "appended_heteronuclear_raw_text": (
                            appended_heteronuclear_raw
                        ),
                        "appended_heteronuclear_nucleus": (
                            appended_heteronuclear_nucleus
                        ),
                        "peaks": [
                            {"ordinal": index, "shift_ppm": shift}
                            for index, shift in enumerate(c_shifts)
                        ],
                    },
                },
                "spectrum_fingerprint_sha256": spectrum_fingerprint,
                "molecule_group_id": group_id,
                "duplicate_molecule_group": None,
                "dp5_upstream_structure_overlap": dp5_structure,
                "dp5_upstream_scaffold_overlap": dp5_scaffold,
                "review": {
                    "review_kind": "upstream_manually_curated_benchmark",
                    "upstream_manually_curated": True,
                    "chemapp_double_reviewed": False,
                    "row_level_reviewer_identity_available": False,
                    "row_level_review_timestamp_available": False,
                    "source_document_mapping_available": False,
                },
                "partition": {"status": "not_assigned"},
            }
        )

    for group_id, ordinals in molecule_groups.items():
        for ordinal in ordinals:
            records[ordinal]["duplicate_molecule_group"] = {
                "molecule_group_id": group_id,
                "is_duplicate_molecule": len(ordinals) > 1,
                "size": len(ordinals),
                "zero_based_ordinals": ordinals,
                "retention_policy": (
                    "retain_distinct_spectrum_records_grouped_by_molecule"
                ),
            }

    observed = {
        "record_count": len(records),
        "h_peak_group_count": h_peak_count,
        "h_reported_integral_total": h_integral_total,
        "c_shift_count": c_shift_count,
        "c_upstream_compatible_shift_count": (
            c_upstream_compatible_shift_count
        ),
        "records_with_appended_heteronuclear_text": (
            records_with_appended_heteronuclear_text
        ),
        "appended_heteronuclear_records_by_nucleus": dict(
            sorted(appended_nuclei.items())
        ),
        "records_with_h_peaks": h_nonempty,
        "records_with_c_shifts": c_nonempty,
        "unique_block_payload_count": len(block_hashes),
        "unique_paired_spectrum_count": len(spectrum_hashes),
        "unique_product_inchikey_count": len(set(product_keys)),
    }
    for field, value in observed.items():
        if field in expected and value != expected[field]:
            raise NMRSolverReviewedDataError(
                f"parsed {field} changed: expected {expected[field]}, got {value}"
            )
    duplicate_observed = [
        {
            "product_inchikey": records[ordinals[0]]["structure"]["inchi_key"],
            "zero_based_ordinals": ordinals,
        }
        for ordinals in molecule_groups.values()
        if len(ordinals) > 1
    ]
    duplicate_expected = [
        {
            "product_inchikey": item["product_inchikey"],
            "zero_based_ordinals": item["zero_based_ordinals"],
        }
        for item in expected.get("duplicate_molecule_groups", [])
    ]
    if duplicate_observed != duplicate_expected:
        raise NMRSolverReviewedDataError(
            "standardized duplicate molecule groups changed"
        )

    structure_overlap = sum(
        bool(record["dp5_upstream_structure_overlap"]) for record in records
    )
    scaffold_overlap = sum(
        bool(record["dp5_upstream_scaffold_overlap"]) for record in records
    )
    independent = sum(
        not record["dp5_upstream_structure_overlap"]
        and not record["dp5_upstream_scaffold_overlap"]
        for record in records
    )
    summary = {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "source": {
            "title": SOURCE_TITLE,
            "creators": list(SOURCE_CREATORS),
            "record_id": RECORD_ID,
            "concept_record_id": CONCEPT_RECORD_ID,
            "doi": SOURCE_DOI,
            "paper_doi": PAPER_DOI,
            "license_spdx": SOURCE_LICENSE,
            "license_uri": SOURCE_LICENSE_URI,
            "allowed_member": ALLOWED_MEMBER,
            "allowed_member_sha256": ALLOWED_MEMBER_SHA256,
        },
        "counts": {
            **observed,
            "unique_molecule_groups": len(molecule_groups),
            "duplicate_molecule_group_count": len(duplicate_observed),
        },
        "duplicate_molecule_groups": duplicate_observed,
        "review_semantics": {
            "review_kind": "upstream_manually_curated_benchmark",
            "chemapp_double_reviewed": False,
            "row_level_reviewer_identity_available": False,
            "source_document_mapping_available": False,
            "warning": (
                "Upstream manually curated does not mean ChemApp dual review."
            ),
        },
        "dp5q_upstream_overlap_audit": {
            **dict(dp5q_reference.binding),
            "structure_overlap_records": structure_overlap,
            "scaffold_overlap_records": scaffold_overlap,
            "structure_and_scaffold_independent_records": independent,
            "selection_rule_for_future_partition": (
                "dp5_upstream_structure_overlap == false AND "
                "dp5_upstream_scaffold_overlap == false"
            ),
        },
        "release_scope": {
            "partition_status": "not_assigned",
            "prediction_artifacts_included": False,
            "headline_accuracy_allowed": False,
            "base_index_overlap_audited_in_this_release": False,
            "independence_warning": (
                "The 260 DP5q structure/scaffold-independent records have not "
                "yet been filtered against the separate local base index; the "
                "previously observed 259 count requires that additional audit."
            ),
        },
    }
    return records, summary


def derive_reviewed_dataset(
    archive_path: str | Path,
    manifest: Mapping[str, Any],
    *,
    dp5q_export_path: str | Path,
    dp5q_attestation_path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify all frozen inputs and derive the complete 450-record release."""

    member, transport = verify_archive_file(archive_path, manifest)
    dp5q = load_dp5q_overlap_reference(
        dp5q_export_path,
        dp5q_attestation_path,
        manifest,
    )
    records, summary = parse_experiment_text(
        member,
        expected=manifest["expected_dataset"],
        dp5q_reference=dp5q,
    )
    expected_overlap = manifest["dp5q_upstream_overlap_audit"][
        "expected_nmrsolver_counts"
    ]
    audit = summary["dp5q_upstream_overlap_audit"]
    for field, expected in expected_overlap.items():
        if audit.get(field) != expected:
            raise NMRSolverReviewedDataError(
                f"DP5q overlap audit {field} changed: "
                f"expected {expected}, got {audit.get(field)}"
            )
    summary["transport"] = transport
    return records, summary


def source_inventory_payload(
    manifest: Mapping[str, Any],
    archive_path: str | Path,
) -> dict[str, Any]:
    _member, transport = verify_archive_file(archive_path, manifest)
    return {
        "schema_version": "chemapp.nmrsolver-source-inventory.v1",
        "record_id": RECORD_ID,
        "doi": SOURCE_DOI,
        "license_spdx": SOURCE_LICENSE,
        "archive": transport["archive"],
        "zip_safety": transport["zip_safety"],
    }


def _atomic_immutable_bytes(
    path: Path,
    payload: bytes,
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            existing = path.read_bytes()
        except OSError as exc:
            raise NMRSolverReviewedDataError(
                f"cannot inspect existing file {path}: {exc}"
            ) from exc
        if existing == payload:
            return
        if not overwrite:
            raise NMRSolverReviewedDataError(
                f"existing immutable file differs: {path}"
            )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{path.name}.",
            suffix=".part",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_source_inventory(
    source_directory: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    target = Path(source_directory).resolve() / "inventory.json"
    rendered = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_immutable_bytes(target, rendered, overwrite=overwrite)
    return target


def _attribution_payload() -> dict[str, Any]:
    return {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "dataset_title": SOURCE_TITLE,
        "creators": list(SOURCE_CREATORS),
        "record_doi": SOURCE_DOI,
        "paper_doi": PAPER_DOI,
        "license": {
            "spdx_id": SOURCE_LICENSE,
            "uri": SOURCE_LICENSE_URI,
        },
        "changes": (
            "ChemApp read only the pinned experimental test.txt member, "
            "standardized product structures with RDKit, parsed the raw paired "
            "1H/13C report text, grouped duplicate molecules without dropping "
            "their distinct spectra, and attached a hash-bound DP5q upstream "
            "structure/scaffold overlap audit."
        ),
        "review_warning": (
            "The upstream paper describes manual curation. These records were "
            "not independently dual-reviewed by ChemApp."
        ),
        "rights_limitation": (
            "Per-row JACS Supporting Information sources and rights were not "
            "mapped or individually audited."
        ),
    }


def write_derived_release(
    destination: str | Path,
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Path | str]:
    """Publish immutable artifacts, then atomically switch ``CURRENT.json``."""

    root = Path(destination).resolve()
    releases_root = root / "releases"
    releases_root.mkdir(parents=True, exist_ok=True)
    records_bytes = "".join(
        _canonical_json(record) + "\n" for record in records
    ).encode("utf-8")
    records_sha256 = _sha256_bytes(records_bytes)
    attribution_bytes = (
        json.dumps(
            _attribution_payload(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    attribution_sha256 = _sha256_bytes(attribution_bytes)
    summary_value = json.loads(_canonical_json(summary))
    summary_value["publication"] = {
        "schema_version": DERIVED_RELEASE_SCHEMA_VERSION,
        "commit_protocol": (
            "immutable release directory committed by atomic CURRENT.json replace"
        ),
        "records": {
            "file": "records.jsonl",
            "sha256": records_sha256,
            "bytes": len(records_bytes),
            "count": len(records),
        },
        "attribution": {
            "file": "ATTRIBUTION.json",
            "sha256": attribution_sha256,
            "bytes": len(attribution_bytes),
            "required_for_redistribution": True,
        },
    }
    summary_core_sha256 = _sha256_bytes(
        _canonical_json(summary_value).encode("utf-8")
    )
    release_id = _sha256_bytes(
        _canonical_json(
            {
                "schema_version": DERIVED_RELEASE_SCHEMA_VERSION,
                "summary_core_sha256": summary_core_sha256,
                "records_sha256": records_sha256,
                "attribution_sha256": attribution_sha256,
            }
        ).encode("utf-8")
    )
    summary_value["publication"].update(
        {
            "release_id": release_id,
            "summary_core_sha256": summary_core_sha256,
        }
    )
    summary_bytes = (
        json.dumps(
            summary_value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    summary_sha256 = _sha256_bytes(summary_bytes)
    release_directory = releases_root / release_id
    expected_files = {
        "records.jsonl": records_bytes,
        "summary.json": summary_bytes,
        "ATTRIBUTION.json": attribution_bytes,
    }
    if release_directory.exists():
        for name, expected in expected_files.items():
            path = release_directory / name
            if not path.is_file() or path.read_bytes() != expected:
                raise NMRSolverReviewedDataError(
                    f"immutable release collision or corruption: {path}"
                )
    else:
        staging = Path(
            tempfile.mkdtemp(
                prefix=f".{release_id}.",
                suffix=".part",
                dir=releases_root,
            )
        )
        try:
            for name, payload in expected_files.items():
                with (staging / name).open("wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            for name, payload in expected_files.items():
                if (staging / name).read_bytes() != payload:
                    raise NMRSolverReviewedDataError(
                        f"staged release verification failed: {name}"
                    )
            os.replace(staging, release_directory)
        finally:
            if staging.exists() and staging.parent == releases_root.resolve():
                shutil.rmtree(staging)
    pointer = {
        "schema_version": DERIVED_RELEASE_SCHEMA_VERSION,
        "release_id": release_id,
        "release_directory": f"releases/{release_id}",
        "records_sha256": records_sha256,
        "summary_sha256": summary_sha256,
        "attribution_sha256": attribution_sha256,
    }
    current = root / "CURRENT.json"
    pointer_bytes = (
        json.dumps(
            pointer,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    _atomic_immutable_bytes(current, pointer_bytes, overwrite=overwrite)
    return {
        "release_id": release_id,
        "current": current,
        "release": release_directory,
        "records": release_directory / "records.jsonl",
        "summary": release_directory / "summary.json",
        "attribution": release_directory / "ATTRIBUTION.json",
    }


def load_derived_release(
    path: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load records only after verifying CURRENT and all bound file hashes."""

    selected = Path(path).resolve()
    pointer_path = selected / "CURRENT.json" if selected.is_dir() else selected
    pointer_bytes = _read_snapshot(pointer_path, maximum_bytes=1024 * 1024)
    try:
        pointer = json.loads(pointer_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise NMRSolverReviewedDataError(
            f"derived release pointer is invalid: {exc}"
        ) from exc
    if (
        not isinstance(pointer, Mapping)
        or pointer.get("schema_version") != DERIVED_RELEASE_SCHEMA_VERSION
        or not isinstance(pointer.get("release_id"), str)
        or not SHA256_RE.fullmatch(str(pointer["release_id"]))
    ):
        raise NMRSolverReviewedDataError("derived release pointer schema changed")
    release_id = str(pointer["release_id"])
    if pointer.get("release_directory") != f"releases/{release_id}":
        raise NMRSolverReviewedDataError(
            "derived release pointer directory changed"
        )
    root = pointer_path.parent
    release = (root / "releases" / release_id).resolve()
    if release.parent != (root / "releases").resolve():
        raise NMRSolverReviewedDataError("release pointer escaped its root")
    payloads = {
        "records": _read_snapshot(
            release / "records.jsonl",
            maximum_bytes=64 * 1024 * 1024,
        ),
        "summary": _read_snapshot(
            release / "summary.json",
            maximum_bytes=8 * 1024 * 1024,
        ),
        "attribution": _read_snapshot(
            release / "ATTRIBUTION.json",
            maximum_bytes=2 * 1024 * 1024,
        ),
    }
    for name, payload in payloads.items():
        expected_hash = pointer.get(f"{name}_sha256")
        if _sha256_bytes(payload) != expected_hash:
            raise NMRSolverReviewedDataError(
                f"derived release {name} hash mismatch"
            )
    try:
        summary = json.loads(payloads["summary"].decode("utf-8"))
        attribution = json.loads(payloads["attribution"].decode("utf-8"))
        records = [
            json.loads(line)
            for line in payloads["records"].decode("utf-8").splitlines()
        ]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise NMRSolverReviewedDataError(
            f"derived release contains invalid UTF-8 JSON: {exc}"
        ) from exc
    publication = summary.get("publication") if isinstance(summary, Mapping) else None
    if (
        not isinstance(publication, Mapping)
        or publication.get("release_id") != release_id
        or publication.get("records", {}).get("sha256")
        != pointer["records_sha256"]
        or publication.get("records", {}).get("count") != len(records)
        or attribution.get("schema_version") != ATTRIBUTION_SCHEMA_VERSION
    ):
        raise NMRSolverReviewedDataError(
            "derived release summary/attribution binding changed"
        )
    return records, dict(summary)
