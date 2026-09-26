"""Frozen acquisition and conservative import of human-reviewed NMRexp rows.

Only the six small ``*_checked.csv`` files from the immutable Zenodo record
17296666 are in scope.  The multi-gigabyte automatically extracted corpus is
deliberately excluded.  The checked files contain spectrum-level literature
peak annotations, not raw FIDs or dense intensity traces.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import tempfile
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Protocol
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

from rdkit import Chem, rdBase
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from app.ml.nmr_data_v2 import connect_readonly, schema_version


CATALOG_SCHEMA_VERSION = 1
DERIVED_SCHEMA_VERSION = "nmrexp-human-reviewed-derived-v2"
DERIVED_RELEASE_SCHEMA_VERSION = "chemapp.nmrexp-derived-release.v2"
ATTRIBUTION_SCHEMA_VERSION = "chemapp.third-party-attribution.v1"
PARENT_STANDARDIZATION_VERSION = (
    "rdkit-cleanup-fragment-parent-uncharger-single-organic-v1"
)
RECORD_ID = 17296666
RECORD_REVISION = 6
CONCEPT_RECORD_ID = 16809533
SOURCE_DOI = "10.5281/zenodo.17296666"
CONCEPT_DOI = "10.5281/zenodo.16809533"
SOURCE_LICENSE = "CC-BY-4.0"
SOURCE_TITLE = "NMRexp: A database of 3.37 million experimental NMR spectra"
SOURCE_CREATORS = ("Wang, Jun-Jie", "Zhu, Rong")
SOURCE_LICENSE_URI = "https://creativecommons.org/licenses/by/4.0/legalcode"
DERIVATION_CHANGE_NOTICE = (
    "ChemApp selected the six pinned human-reviewed CSV files, verified their "
    "bytes, deduplicated exact row aliases, parsed reviewed peak annotations, "
    "removed source prose from redistributed records, standardized eligible "
    "single-fragment organic parent structures with RDKit, quarantined "
    "multi-fragment structures, and audited parent overlap against a frozen "
    "nmrshiftdb2-derived index."
)
SOURCE_PROSE_KEYS = frozenset(
    {
        "NMR_shift_text",
        "NMR_note",
        "text_in_pdf",
        "reported_shift_text",
        "reported_note",
        # These two names were briefly used by a pre-release v2 draft.  They
        # are forbidden so a digest object can never be mistaken for prose.
        "source_shift_text",
        "source_note",
    }
)
RECORD_HOST = "zenodo.org"
ALLOWED_RESPONSE_HOSTS = {"zenodo.org", "files.zenodo.org"}
DEFAULT_CHUNK_SIZE = 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_CELL_CHARS = 128 * 1024
MAX_CHECKED_FILE_BYTES = 2 * 1024 * 1024
HTML_PREFIXES = (b"<!doctype html", b"<html")
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
FREQUENCY_RE = re.compile(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))")

CSV_FIELDS = (
    "Filename",
    "SMILES",
    "Page_in_file_mol",
    "Page_in_file_para",
    "Location_in_page_mol",
    "Location_in_page_para",
    "NMR_type",
    "NMR_frequency",
    "NMR_solvent",
    "NMR_shift_text",
    "NMR_note",
    "NMR_processed",
    "Atom_number",
    "Atom_number_diff_env",
    "Atom_number_abstract",
    "smiles_actual",
    "text_in_pdf",
    "nmr_frequency_right",
    "nmr_solvent_right",
    "nmr_processed_right",
    "is_same_molecule",
    "is_same_skeleton",
    "num_chiral_centers",
)
CRITICAL_ASCII_FIELDS = (
    "Filename",
    "SMILES",
    "NMR_type",
    "NMR_frequency",
    "NMR_solvent",
    "NMR_processed",
    "smiles_actual",
    "nmr_frequency_right",
    "nmr_solvent_right",
    "nmr_processed_right",
    "is_same_molecule",
    "is_same_skeleton",
)
ALLOWED_NMR_TYPES = {
    "1H NMR": "1H",
    "13C NMR": "13C",
    "19F NMR": "19F",
    "31P NMR": "31P",
    "29Si NMR": "29Si",
    "11B NMR": "11B",
}
CURRENT_MODEL_NUCLEI = {"1H", "13C"}
ALLOWED_REVIEW_LABELS = {"right", "wrong", "SIdon'thave"}
EXPECTED_NUCLEUS_COUNTS = {
    "1H": 143,
    "13C": 138,
    "19F": 68,
    "31P": 51,
    "29Si": 50,
    "11B": 50,
}
EXPECTED_FILE_METADATA: dict[str, dict[str, Any]] = {
    "B_50_checked.csv": {
        "bytes": 17758,
        "md5": "29f849f72aa82a28f03021db2e084ffd",
        "sha256": ("1bda4066e572e49c9e2a55e41788d2a963f015bf2d86f0bc2c6f891863b0c4e9"),
        "rows": 50,
    },
    "F_50_checked.csv": {
        "bytes": 20152,
        "md5": "2215638aa53c5ac10a3ce10afadfdf09",
        "sha256": ("4bb547d8a8bf68c45e971ff2184a7704a05bad028e42c5f231928f0856a71c70"),
        "rows": 50,
    },
    "P_50_checked.csv": {
        "bytes": 18746,
        "md5": "a2765d3a2457dd5a7d486d01688a8517",
        "sha256": ("af4030aea25136867f1f8f7a2a97d2d20f4bd4c522bd3d2e1193549cb584aa4e"),
        "rows": 50,
    },
    "Si_50_checked.csv": {
        "bytes": 18343,
        "md5": "705548706c9e66c141a3be5e9d04d509",
        "sha256": ("c5e7139413d305b10356c1c634c4f0d78d8a68914b0eccabf46c2b28759557e5"),
        "rows": 50,
    },
    "hetero_200_checked.csv": {
        "bytes": 73922,
        "md5": "612f248f288b59b52a96806cde4d3c7c",
        "sha256": ("9af1fc1b77123b53308553747ca9536aa097d3144ea7902103fc25486f3ae002"),
        "rows": 200,
    },
    "test_300_checked.csv": {
        "bytes": 254418,
        "md5": "57acad1d418599194ecacd3d4dd7bcf8",
        "sha256": ("7a6ebe20fc8dbcfe6cf3d823b790e0c3f1b96571b5d73f02f90be12a16e6b4ee"),
        "rows": 300,
    },
}
EXPECTED_INVENTORY_SHA256 = (
    "dc740de50e8bab94d89135bd4101a6945a898e84f10a54ff5051b0118b21d3c6"
)
EXPECTED_SELECTED_KEYS = {
    "id",
    "record_id",
    "record_revision",
    "concept_record_id",
    "doi",
    "concept_doi",
    "title",
    "publication_date",
    "record_uri",
    "api_uri",
    "paper_doi",
    "creators",
    "access_right",
    "license",
    "version_policy",
    "scope",
    "encoding_policy",
    "inventory_sha256",
    "inventory_hash_basis",
    "files",
}
EXPECTED_FILE_KEYS = {
    "name",
    "bytes",
    "md5",
    "sha256",
    "rows",
    "url",
    "media_type",
}


class IndependentNMRDataError(ValueError):
    """Raised when source identity, bytes, rows or derived semantics fail."""


class ResponseLike(Protocol):
    """Small urllib-compatible surface used by the atomic downloader."""

    headers: Mapping[str, str]

    def read(self, size: int = -1) -> bytes: ...

    def geturl(self) -> str: ...

    def __enter__(self) -> ResponseLike: ...

    def __exit__(self, *args: object) -> object: ...


Opener = Callable[..., ResponseLike]


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DEFAULT_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DEFAULT_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_filename(value: object) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value:
        return False
    return (
        PurePosixPath(value).name == value
        and PureWindowsPath(value).name == value
        and "/" not in value
        and "\\" not in value
        and value not in {".", ".."}
    )


def _canonical_inventory(files: Iterable[Mapping[str, Any]]) -> bytes:
    selected = [
        {
            "name": str(item["name"]),
            "bytes": int(item["bytes"]),
            "md5": str(item["md5"]),
        }
        for item in files
    ]
    selected.sort(key=lambda item: item["name"])
    return _canonical_json(selected).encode("utf-8")


def load_source_catalog(path: str | Path) -> dict[str, Any]:
    """Load the committed catalog and fail closed on selected-source drift."""

    catalog_path = Path(path).resolve()
    try:
        value = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndependentNMRDataError(f"cannot load source catalog: {exc}") from exc
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "catalog_id",
        "reviewed_at",
        "selected_source",
        "evaluated_sources",
    }:
        raise IndependentNMRDataError("source catalog fields changed")
    if value.get("schema_version") != CATALOG_SCHEMA_VERSION:
        raise IndependentNMRDataError("source catalog schema version changed")
    if value.get("catalog_id") != "chemapp-independent-nmr-sources-v1":
        raise IndependentNMRDataError("source catalog identity changed")

    source = value.get("selected_source")
    if not isinstance(source, dict) or set(source) != EXPECTED_SELECTED_KEYS:
        raise IndependentNMRDataError("selected source fields changed")
    expected_scalars = {
        "id": "nmrexp-human-reviewed-zenodo-17296666",
        "record_id": RECORD_ID,
        "record_revision": RECORD_REVISION,
        "concept_record_id": CONCEPT_RECORD_ID,
        "doi": SOURCE_DOI,
        "concept_doi": CONCEPT_DOI,
        "title": SOURCE_TITLE,
        "publication_date": "2025-10-09",
        "record_uri": f"https://zenodo.org/records/{RECORD_ID}",
        "api_uri": f"https://zenodo.org/api/records/{RECORD_ID}",
        "paper_doi": "10.1038/s41597-025-06245-5",
        "access_right": "open",
        "inventory_sha256": EXPECTED_INVENTORY_SHA256,
    }
    for field, expected in expected_scalars.items():
        if source.get(field) != expected:
            raise IndependentNMRDataError(f"selected source {field} changed")
    if source.get("creators") != list(SOURCE_CREATORS):
        raise IndependentNMRDataError("selected source creators changed")
    if source.get("license") != {
        "spdx_id": SOURCE_LICENSE,
        "record_value": "cc-by-4.0",
        "uri": SOURCE_LICENSE_URI,
    }:
        raise IndependentNMRDataError("selected source license changed")

    version_policy = source.get("version_policy")
    if (
        not isinstance(version_policy, dict)
        or version_policy.get("mode") != "specific_record_only"
        or version_policy.get("concept_or_latest_resolution_allowed") is not False
        or version_policy.get("supersedes_record_id") != 16809534
    ):
        raise IndependentNMRDataError("selected source version policy changed")
    scope = source.get("scope")
    expected_scope = {
        "selected_file_count": 6,
        "download_bytes": 403339,
        "surface_rows": 700,
        "expected_unique_content_rows": 500,
        "expected_duplicate_alias_rows": 200,
        "current_model_nuclei": ["1H", "13C"],
        "current_model_scope_rows": 281,
        "future_nucleus_rows": 219,
    }
    if not isinstance(scope, dict):
        raise IndependentNMRDataError("selected source scope changed")
    for field, expected in expected_scope.items():
        if scope.get(field) != expected:
            raise IndependentNMRDataError(f"selected source scope {field} changed")
    if (
        scope.get("raw_fid_or_dense_trace") is not False
        or scope.get("atom_level_peak_assignments") is not False
        or scope.get("headline_accuracy_allowed_without_split_and_overlap_audit")
        is not False
    ):
        raise IndependentNMRDataError("selected source scientific scope changed")

    files = source.get("files")
    if not isinstance(files, list) or len(files) != len(EXPECTED_FILE_METADATA):
        raise IndependentNMRDataError("selected checked-file inventory changed")
    names: set[str] = set()
    total_bytes = 0
    for record in files:
        if (
            not isinstance(record, dict)
            or set(record) != EXPECTED_FILE_KEYS
            or not _safe_filename(record.get("name"))
        ):
            raise IndependentNMRDataError("selected file fields or name changed")
        name = str(record["name"])
        if name in names:
            raise IndependentNMRDataError(f"duplicate selected filename: {name}")
        names.add(name)
        expected_file = EXPECTED_FILE_METADATA.get(name)
        if expected_file is None:
            raise IndependentNMRDataError(f"unexpected selected filename: {name}")
        expected_record = {
            "name": name,
            **expected_file,
            "url": (f"https://zenodo.org/api/records/{RECORD_ID}/files/{name}/content"),
            "media_type": "text/csv",
        }
        if _canonical_json(record) != _canonical_json(expected_record):
            raise IndependentNMRDataError(f"frozen metadata changed for {name}")
        total_bytes += int(record["bytes"])
        if not MD5_RE.fullmatch(str(record["md5"])):
            raise IndependentNMRDataError(f"invalid MD5 for {name}")
        if not SHA256_RE.fullmatch(str(record["sha256"])):
            raise IndependentNMRDataError(f"invalid SHA-256 for {name}")
        parsed = urlparse(str(record["url"]))
        expected_path = f"/api/records/{RECORD_ID}/files/{name}/content"
        if (
            parsed.scheme != "https"
            or parsed.hostname != RECORD_HOST
            or unquote(parsed.path) != expected_path
            or parsed.query
            or parsed.fragment
        ):
            raise IndependentNMRDataError(f"unsafe selected URL for {name}")
    if names != set(EXPECTED_FILE_METADATA):
        raise IndependentNMRDataError("selected checked-file names changed")
    if total_bytes != scope["download_bytes"]:
        raise IndependentNMRDataError("selected checked-file byte total changed")
    inventory = hashlib.sha256(_canonical_inventory(files)).hexdigest()
    if inventory != source["inventory_sha256"]:
        raise IndependentNMRDataError("selected checked-file inventory hash changed")
    evaluations = value.get("evaluated_sources")
    if not isinstance(evaluations, list) or not evaluations:
        raise IndependentNMRDataError("evaluated source catalog is empty")
    return value


def _parse_checked_csv_snapshot(
    data: bytes,
    *,
    display_name: str,
    expected_rows: int | None,
) -> list[dict[str, str]]:
    """Parse one already-frozen byte snapshot.

    Latin-1 is a one-byte-to-one-code-point transport decoder. It is not a
    claim about the natural-language encoding of embedded PDF snippets.
    """

    try:
        with io.TextIOWrapper(
            io.BytesIO(data),
            encoding="latin-1",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise IndependentNMRDataError(
                    f"{display_name} checked CSV columns changed"
                )
            rows: list[dict[str, str]] = []
            for line_number, raw in enumerate(reader, start=2):
                if None in raw or set(raw) != set(CSV_FIELDS):
                    raise IndependentNMRDataError(
                        f"{display_name}:{line_number} has a malformed CSV row"
                    )
                row = {field: str(raw[field]) for field in CSV_FIELDS}
                for field, cell in row.items():
                    if "\x00" in cell or len(cell) > MAX_CELL_CHARS:
                        raise IndependentNMRDataError(
                            f"{display_name}:{line_number} has an unsafe {field} cell"
                        )
                for field in CRITICAL_ASCII_FIELDS:
                    cell = row[field]
                    if not cell or not cell.isascii():
                        raise IndependentNMRDataError(
                            f"{display_name}:{line_number} has a non-ASCII or empty "
                            f"critical {field} field"
                        )
                if row["NMR_type"] not in ALLOWED_NMR_TYPES:
                    raise IndependentNMRDataError(
                        f"{display_name}:{line_number} has an unsupported NMR type"
                    )
                for field in (
                    "nmr_frequency_right",
                    "nmr_solvent_right",
                    "nmr_processed_right",
                ):
                    if row[field] not in ALLOWED_REVIEW_LABELS:
                        raise IndependentNMRDataError(
                            f"{display_name}:{line_number} has an invalid {field}"
                        )
                for field in ("is_same_molecule", "is_same_skeleton"):
                    if row[field].upper() not in {"TRUE", "FALSE"}:
                        raise IndependentNMRDataError(
                            f"{display_name}:{line_number} has an invalid {field}"
                        )
                rows.append(row)
    except IndependentNMRDataError:
        raise
    except (csv.Error, UnicodeError) as exc:
        raise IndependentNMRDataError(
            f"cannot parse checked CSV {display_name}: {exc}"
        ) from exc
    if expected_rows is not None and len(rows) != expected_rows:
        raise IndependentNMRDataError(
            f"{display_name} row-count mismatch: expected {expected_rows}, "
            f"got {len(rows)}"
        )
    return rows


def _read_immutable_file_snapshot(
    path: Path,
    *,
    maximum_bytes: int,
) -> bytes:
    """Read one regular file once and reject mutation during that open handle."""

    if maximum_bytes < 1:
        raise ValueError("maximum_bytes must be positive")
    try:
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise IndependentNMRDataError(
                    f"checked file is not a regular file: {path}"
                )
            if before.st_size < 1 or before.st_size > maximum_bytes:
                raise IndependentNMRDataError(
                    f"checked file size is outside the snapshot bound: {path}"
                )
            data = handle.read(maximum_bytes + 1)
            after = os.fstat(handle.fileno())
    except IndependentNMRDataError:
        raise
    except OSError as exc:
        raise IndependentNMRDataError(
            f"cannot snapshot checked file {path}: {exc}"
        ) from exc
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    )
    if identity_before != identity_after or len(data) != before.st_size:
        raise IndependentNMRDataError(
            f"checked file changed while its immutable snapshot was read: {path}"
        )
    if len(data) > maximum_bytes:
        raise IndependentNMRDataError(
            f"checked file exceeded the snapshot byte limit: {path}"
        )
    return data


def read_checked_rows(
    path: str | Path,
    *,
    expected_rows: int | None = None,
) -> list[dict[str, str]]:
    """Read and parse one immutable mixed-encoding checked-CSV snapshot."""

    csv_path = Path(path)
    data = _read_immutable_file_snapshot(
        csv_path,
        maximum_bytes=MAX_CHECKED_FILE_BYTES,
    )
    return _parse_checked_csv_snapshot(
        data,
        display_name=csv_path.name,
        expected_rows=expected_rows,
    )


def load_checked_file_snapshot(
    path: str | Path,
    record: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Verify hashes and parse rows from exactly the same immutable bytes."""

    candidate = Path(path)
    if not candidate.is_file():
        raise IndependentNMRDataError(f"checked file is missing: {candidate}")
    expected_bytes = int(record["bytes"])
    data = _read_immutable_file_snapshot(
        candidate,
        maximum_bytes=max(expected_bytes, 1),
    )
    byte_size = len(data)
    if byte_size != expected_bytes:
        raise IndependentNMRDataError(
            f"{record['name']} size mismatch: expected {expected_bytes}, "
            f"got {byte_size}"
        )
    prefix = data[:512].lstrip().lower()
    if prefix.startswith(HTML_PREFIXES) or b"<html" in prefix:
        raise IndependentNMRDataError(
            f"{record['name']} is HTML, not checked CSV content"
        )
    md5 = hashlib.md5(data, usedforsecurity=False).hexdigest()
    if md5 != record["md5"]:
        raise IndependentNMRDataError(
            f"{record['name']} MD5 mismatch: expected {record['md5']}, got {md5}"
        )
    sha256 = hashlib.sha256(data).hexdigest()
    if sha256 != record["sha256"]:
        raise IndependentNMRDataError(
            f"{record['name']} SHA-256 mismatch: expected "
            f"{record['sha256']}, got {sha256}"
        )
    rows = _parse_checked_csv_snapshot(
        data,
        display_name=str(record["name"]),
        expected_rows=int(record["rows"]),
    )
    metadata = {
        "name": str(record["name"]),
        "bytes": byte_size,
        "md5": md5,
        "sha256": sha256,
        "rows": len(rows),
        "format_audit": {
            "csv_valid": True,
            "columns_exact": True,
            "critical_fields_ascii": True,
            "transport_decoder": "latin-1-byte-preserving",
            "single_immutable_snapshot": True,
            "size_hashes_and_parse_same_bytes": True,
        },
    }
    return metadata, rows


def verify_checked_file(
    path: str | Path,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify and parse one file from a single immutable byte snapshot."""

    metadata, _rows = load_checked_file_snapshot(path, record)
    return metadata


def _validate_response_url(response: ResponseLike, expected_url: str) -> None:
    final_url = response.geturl() or expected_url
    parsed = urlparse(final_url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_RESPONSE_HOSTS:
        raise IndependentNMRDataError(
            f"download redirected to an untrusted URL: {final_url}"
        )


def _stream_response(
    response: ResponseLike,
    handle: BinaryIO,
    *,
    expected_bytes: int,
    chunk_size: int,
) -> None:
    content_length = response.headers.get("Content-Length")
    if content_length:
        try:
            declared = int(content_length)
        except ValueError as exc:
            raise IndependentNMRDataError("invalid Content-Length header") from exc
        if declared != expected_bytes:
            raise IndependentNMRDataError(
                f"server Content-Length mismatch: expected {expected_bytes}, "
                f"got {declared}"
            )
    content_type = str(response.headers.get("Content-Type", "")).casefold()
    if "text/html" in content_type:
        raise IndependentNMRDataError(
            "server returned HTML instead of checked CSV content"
        )

    total = 0
    first = True
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        if first:
            first = False
            prefix = chunk[:512].lstrip().lower()
            if prefix.startswith(HTML_PREFIXES) or b"<html" in prefix:
                raise IndependentNMRDataError(
                    "download body is HTML, not checked CSV content"
                )
        total += len(chunk)
        if total > expected_bytes:
            raise IndependentNMRDataError(
                "download exceeded the frozen checked-file byte count"
            )
        handle.write(chunk)
    if total != expected_bytes:
        raise IndependentNMRDataError(
            f"download ended at {total} bytes; expected {expected_bytes}"
        )


def download_checked_file(
    record: Mapping[str, Any],
    destination: str | Path,
    *,
    opener: Opener = urlopen,
    timeout_seconds: float = 60.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Download one pinned checked file atomically and never replace bad bytes."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not _safe_filename(record.get("name")):
        raise IndependentNMRDataError("unsafe checked-file destination name")
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root / str(record["name"])).resolve()
    if target.parent != root:
        raise IndependentNMRDataError("checked-file destination escaped its root")
    if target.exists():
        result = verify_checked_file(target, record)
        result["reused"] = True
        return result

    request = Request(
        str(record["url"]),
        headers={
            "Accept": "text/csv,application/octet-stream;q=0.9,*/*;q=0.1",
            "Referer": f"https://zenodo.org/records/{RECORD_ID}",
            "User-Agent": "ChemApp-independent-nmr-fetch/1",
        },
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{record['name']}.",
            suffix=".part",
            dir=root,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            try:
                with opener(request, timeout=timeout_seconds) as response:
                    _validate_response_url(response, str(record["url"]))
                    _stream_response(
                        response,
                        handle,
                        expected_bytes=int(record["bytes"]),
                        chunk_size=chunk_size,
                    )
            except IndependentNMRDataError:
                raise
            except (OSError, TimeoutError) as exc:
                raise IndependentNMRDataError(
                    f"network retrieval failed without publishing partial bytes: {exc}"
                ) from exc
            handle.flush()
            os.fsync(handle.fileno())
        result = verify_checked_file(temporary_path, record)
        os.replace(temporary_path, target)
        temporary_path = None
        result["reused"] = False
        return result
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def verify_source_files(
    source: Mapping[str, Any],
    destination: str | Path,
) -> list[dict[str, Any]]:
    """Verify every selected file from one immutable snapshot per file."""

    verified, _rows_by_file = load_source_file_snapshots(source, destination)
    return verified


def load_source_file_snapshots(
    source: Mapping[str, Any],
    destination: str | Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    """Verify and parse all selected files without reopening any source path."""

    root = Path(destination).resolve()
    verified: list[dict[str, Any]] = []
    rows_by_file: dict[str, list[dict[str, str]]] = {}
    for record in sorted(source["files"], key=lambda item: str(item["name"])):
        name = str(record["name"])
        metadata, rows = load_checked_file_snapshot(root / name, record)
        verified.append(metadata)
        rows_by_file[name] = rows
    return verified, rows_by_file


def _read_bounded_json_response(
    response: ResponseLike,
    *,
    maximum_bytes: int = MAX_METADATA_BYTES,
) -> dict[str, Any]:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.read(min(DEFAULT_CHUNK_SIZE, maximum_bytes + 1 - total))
        if not chunk:
            break
        total += len(chunk)
        if total > maximum_bytes:
            raise IndependentNMRDataError("upstream record metadata is too large")
        chunks.append(chunk)
    try:
        value = json.loads(b"".join(chunks).decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IndependentNMRDataError(
            f"upstream record metadata is not valid JSON: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise IndependentNMRDataError("upstream record metadata root is not an object")
    return value


def verify_upstream_record(
    source: Mapping[str, Any],
    *,
    opener: Opener = urlopen,
    timeout_seconds: float = 60.0,
) -> dict[str, Any]:
    """Check current API metadata without resolving a concept/latest version."""

    api_uri = str(source["api_uri"])
    request = Request(
        api_uri,
        headers={
            "Accept": "application/json",
            "User-Agent": "ChemApp-independent-nmr-metadata-check/1",
        },
    )
    try:
        with opener(request, timeout=timeout_seconds) as response:
            _validate_response_url(response, api_uri)
            value = _read_bounded_json_response(response)
    except IndependentNMRDataError:
        raise
    except (OSError, TimeoutError) as exc:
        raise IndependentNMRDataError(
            f"could not verify pinned upstream record: {exc}"
        ) from exc

    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise IndependentNMRDataError("upstream record has no metadata object")
    observed = {
        "record_id": value.get("id"),
        "record_revision": value.get("revision"),
        "concept_record_id": int(value.get("conceptrecid", -1)),
        "doi": value.get("doi"),
        "concept_doi": value.get("conceptdoi"),
        "title": metadata.get("title"),
        "publication_date": metadata.get("publication_date"),
        "access_right": metadata.get("access_right"),
        "license": (metadata.get("license") or {}).get("id"),
    }
    expected = {
        "record_id": RECORD_ID,
        "record_revision": RECORD_REVISION,
        "concept_record_id": CONCEPT_RECORD_ID,
        "doi": SOURCE_DOI,
        "concept_doi": CONCEPT_DOI,
        "title": SOURCE_TITLE,
        "publication_date": "2025-10-09",
        "access_right": "open",
        "license": "cc-by-4.0",
    }
    if observed != expected:
        raise IndependentNMRDataError(
            "pinned upstream record metadata changed from the frozen catalog"
        )
    upstream_files = {
        str(item.get("key")): item
        for item in value.get("files", [])
        if isinstance(item, dict)
    }
    for record in source["files"]:
        upstream = upstream_files.get(str(record["name"]))
        checksum = str((upstream or {}).get("checksum") or "")
        if (
            upstream is None
            or int(upstream.get("size", -1)) != int(record["bytes"])
            or checksum != f"md5:{record['md5']}"
        ):
            raise IndependentNMRDataError(
                f"pinned upstream checked-file metadata changed for {record['name']}"
            )
    return {
        "status": "matched",
        **observed,
        "selected_files": len(source["files"]),
        "version_resolution": "specific_record_only",
    }


def _row_content_hash(row: Mapping[str, str]) -> str:
    payload = {field: row[field] for field in CSV_FIELDS}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def deduplicate_checked_rows(
    rows_by_file: Mapping[str, Sequence[Mapping[str, str]]],
) -> list[dict[str, Any]]:
    """Deduplicate byte-preserved row content while retaining all aliases."""

    groups: dict[str, dict[str, Any]] = {}
    for filename in sorted(rows_by_file):
        for data_row, row_value in enumerate(rows_by_file[filename], start=1):
            row = {field: str(row_value[field]) for field in CSV_FIELDS}
            content_hash = _row_content_hash(row)
            alias = {"file": filename, "data_row": data_row}
            existing = groups.get(content_hash)
            if existing is None:
                groups[content_hash] = {
                    "source_content_sha256": content_hash,
                    "row": row,
                    "source_aliases": [alias],
                }
            else:
                if _canonical_json(existing["row"]) != _canonical_json(row):
                    raise IndependentNMRDataError(
                        "SHA-256 collision while deduplicating checked rows"
                    )
                existing["source_aliases"].append(alias)
    records = list(groups.values())
    records.sort(key=lambda item: str(item["source_content_sha256"]))
    for record in records:
        record["source_aliases"].sort(
            key=lambda item: (str(item["file"]), int(item["data_row"]))
        )
    return records


def _literal_to_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise IndependentNMRDataError("NMR_processed contains a non-finite value")
        return value
    if isinstance(value, (tuple, list)):
        return [_literal_to_json(item) for item in value]
    raise IndependentNMRDataError(
        f"NMR_processed contains unsupported literal type {type(value).__name__}"
    )


def _finite_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise IndependentNMRDataError(f"{field} is not numeric") from exc
    if not math.isfinite(parsed):
        raise IndependentNMRDataError(f"{field} is not finite")
    return parsed


def _shift_values(value: Any) -> list[float]:
    raw_values = value if isinstance(value, (list, tuple)) else [value]
    values = [_finite_float(item, field="NMR shift") for item in raw_values]
    if not values or len(values) > 32:
        raise IndependentNMRDataError("NMR shift range has an unsafe length")
    return values


def parse_processed_peaks(raw: str, nucleus: str) -> list[dict[str, Any]]:
    """Parse the reviewed Python-literal peak list without executing code."""

    if len(raw) > MAX_CELL_CHARS:
        raise IndependentNMRDataError("NMR_processed is too large")
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise IndependentNMRDataError(
            f"NMR_processed is not a safe literal: {exc}"
        ) from exc
    if not isinstance(parsed, list) or not parsed or len(parsed) > 2048:
        raise IndependentNMRDataError("NMR_processed must be a non-empty, bounded list")

    peaks: list[dict[str, Any]] = []
    for ordinal, item in enumerate(parsed):
        if not isinstance(item, tuple):
            raise IndependentNMRDataError("NMR_processed entries must be tuples")
        if nucleus == "1H":
            if len(item) != 5:
                raise IndependentNMRDataError(
                    "reviewed 1H peak entries must contain five fields"
                )
            multiplicity, couplings, integral, first, second = item
            values = [
                _finite_float(first, field="1H range endpoint"),
                _finite_float(second, field="1H range endpoint"),
            ]
            lower, upper = min(values), max(values)
            peak = {
                "ordinal": ordinal,
                "shift_ppm": round(sum(values) / 2.0, 8),
                "range_ppm": [lower, upper],
                "multiplicity": None if multiplicity is None else str(multiplicity),
                "couplings": _literal_to_json(couplings),
                "reported_integral": None if integral is None else str(integral),
            }
        else:
            if len(item) != 3:
                raise IndependentNMRDataError(
                    "reviewed non-1H peak entries must contain three fields"
                )
            shift_value, multiplicity, couplings = item
            values = _shift_values(shift_value)
            peak = {
                "ordinal": ordinal,
                "shift_ppm": round(sum(values) / len(values), 8),
                "range_ppm": [min(values), max(values)],
                "multiplicity": None if multiplicity is None else str(multiplicity),
                "couplings": _literal_to_json(couplings),
                "reported_integral": None,
            }
        peaks.append(peak)
    return peaks


def _empty_structure_identity(
    status: str,
    *,
    fragment_count: int | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "canonical_smiles": None,
        "inchi_key": None,
        "molecule_key": None,
        "formula": None,
        "scaffold_key": None,
        "training_smiles": None,
        "parent_eligible": False,
        "parent_status": status,
        "parent_standardization_version": PARENT_STANDARDIZATION_VERSION,
        "fragment_count": fragment_count,
        "multi_fragment_or_mixture": (
            fragment_count is not None and fragment_count > 1
        ),
        "reported_canonical_smiles": None,
        "reported_inchi_key": None,
        "reported_molecule_key": None,
        "reported_formula": None,
        "reported_scaffold_key": None,
        "parent_is_reported_structure": None,
    }


def _structure_identity(
    smiles: str,
    *,
    quarantine_multifragment: bool = True,
) -> dict[str, Any]:
    """Return a standardized organic-parent identity plus reported identity.

    Human multi-fragment values are never silently reduced to one component.
    Base-index structures may opt into fragment-parent normalization solely to
    make overlap detection conservative.
    """

    blocker = rdBase.BlockLogs()
    try:
        return _structure_identity_with_blocked_logs(
            smiles,
            quarantine_multifragment=quarantine_multifragment,
        )
    finally:
        del blocker


def _structure_identity_with_blocked_logs(
    smiles: str,
    *,
    quarantine_multifragment: bool,
) -> dict[str, Any]:
    """Implementation kept under one RDKit log block for expected bad inputs."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _empty_structure_identity("human_truth_not_machine_parseable")
    fragment_count = len(Chem.GetMolFrags(mol))
    try:
        reported_canonical_smiles = Chem.MolToSmiles(
            mol,
            canonical=True,
            isomericSmiles=True,
        )
        reported_inchi_key = str(Chem.MolToInchiKey(mol) or "").upper()
        reported_formula = rdMolDescriptors.CalcMolFormula(mol)
        reported_scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
    except Exception as exc:  # pragma: no cover - defensive around RDKit builds
        raise IndependentNMRDataError(
            f"RDKit could not derive reported structure identity: {exc}"
        ) from exc
    if not reported_inchi_key or len(reported_inchi_key) < 14:
        raise IndependentNMRDataError("RDKit did not produce a stable InChIKey")

    reported = {
        "reported_canonical_smiles": reported_canonical_smiles,
        "reported_inchi_key": reported_inchi_key,
        "reported_molecule_key": reported_inchi_key[:14],
        "reported_formula": reported_formula,
        "reported_scaffold_key": reported_scaffold or "acyclic",
    }
    if fragment_count > 1 and quarantine_multifragment:
        return {
            **_empty_structure_identity(
                "multi_fragment_quarantined",
                fragment_count=fragment_count,
            ),
            **reported,
        }

    try:
        cleaned = rdMolStandardize.Cleanup(Chem.Mol(mol))
        parent = rdMolStandardize.FragmentParent(
            cleaned,
            skipStandardize=True,
        )
        parent = rdMolStandardize.Uncharger().uncharge(parent)
        Chem.SanitizeMol(parent)
        parent_fragments = len(Chem.GetMolFrags(parent))
        has_carbon = any(atom.GetAtomicNum() == 6 for atom in parent.GetAtoms())
        if parent_fragments != 1 or not has_carbon:
            status = (
                "standardized_parent_not_single_fragment"
                if parent_fragments != 1
                else "standardized_parent_not_organic"
            )
            return {
                **_empty_structure_identity(
                    status,
                    fragment_count=fragment_count,
                ),
                **reported,
            }
        canonical_smiles = Chem.MolToSmiles(
            parent,
            canonical=True,
            isomericSmiles=True,
        )
        inchi_key = str(Chem.MolToInchiKey(parent) or "").upper()
        formula = rdMolDescriptors.CalcMolFormula(parent)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent)
    except Exception as exc:  # pragma: no cover - defensive around RDKit builds
        raise IndependentNMRDataError(
            f"RDKit could not standardize organic parent identity: {exc}"
        ) from exc
    if not inchi_key or len(inchi_key) < 14:
        raise IndependentNMRDataError(
            "RDKit did not produce a stable standardized parent InChIKey"
        )
    return {
        "status": "human_corrected_standardized_parent",
        "canonical_smiles": canonical_smiles,
        "inchi_key": inchi_key,
        "molecule_key": inchi_key[:14],
        "formula": formula,
        "scaffold_key": scaffold or "acyclic",
        "training_smiles": canonical_smiles,
        "parent_eligible": True,
        "parent_status": "eligible_single_organic_parent",
        "parent_standardization_version": PARENT_STANDARDIZATION_VERSION,
        "fragment_count": fragment_count,
        "multi_fragment_or_mixture": fragment_count > 1,
        **reported,
        "parent_is_reported_structure": inchi_key == reported_inchi_key,
    }


def _document_doi(source_key: str) -> str | None:
    candidate = source_key.replace("_", "/", 1)
    return candidate if DOI_RE.fullmatch(candidate) else None


def _parse_frequency_mhz(value: str) -> float | None:
    match = FREQUENCY_RE.search(value)
    if match is None:
        return None
    parsed = _finite_float(match.group(1), field="NMR frequency")
    return parsed if parsed > 0 else None


def _as_review_bool(value: str) -> bool:
    upper = value.upper()
    if upper not in {"TRUE", "FALSE"}:
        raise IndependentNMRDataError("invalid reviewed boolean")
    return upper == "TRUE"


def _text_digest(value: str) -> dict[str, Any]:
    return {
        "sha256": hashlib.sha256(value.encode("latin-1")).hexdigest(),
        "chars": len(value),
        "redistributed": False,
    }


def _normalise_reviewed_record(item: Mapping[str, Any]) -> dict[str, Any]:
    row = item["row"]
    nucleus = ALLOWED_NMR_TYPES[row["NMR_type"]]
    actual = _structure_identity(row["smiles_actual"])
    extracted = _structure_identity(row["SMILES"])
    peaks = parse_processed_peaks(row["NMR_processed"], nucleus)
    frequency_label = row["nmr_frequency_right"]
    solvent_label = row["nmr_solvent_right"]
    processed_label = row["nmr_processed_right"]
    labels_all_right = (
        frequency_label == "right"
        and solvent_label == "right"
        and processed_label == "right"
    )
    return {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "record_id": f"nmrexp:{item['source_content_sha256'][:24]}",
        "source_content_sha256": item["source_content_sha256"],
        "source_aliases": item["source_aliases"],
        "source": {
            "dataset": "NMRexp",
            "zenodo_record_id": RECORD_ID,
            "zenodo_doi": SOURCE_DOI,
            "license_spdx": SOURCE_LICENSE,
            "attribution_manifest_required": True,
            "document_key": row["Filename"],
            "document_doi": _document_doi(row["Filename"]),
            "molecule_page": int(row["Page_in_file_mol"]),
            "spectrum_text_page": int(row["Page_in_file_para"]),
            "reported_text_sha256": hashlib.sha256(
                row["text_in_pdf"].encode("latin-1")
            ).hexdigest(),
            "reported_text_chars": len(row["text_in_pdf"]),
            "reported_text_redistributed": False,
            "transport_text_decoder": "latin-1-byte-preserving",
        },
        "structure": {
            "ground_truth_field": "smiles_actual",
            "human_corrected_smiles": row["smiles_actual"],
            **actual,
            "extracted_field": "SMILES",
            "extracted_smiles": row["SMILES"],
            "extracted_canonical_smiles": extracted["canonical_smiles"],
            "extracted_inchi_key": extracted["inchi_key"],
            "extracted_parent_status": extracted["parent_status"],
            "extracted_reported_canonical_smiles": extracted[
                "reported_canonical_smiles"
            ],
            "extracted_reported_inchi_key": extracted["reported_inchi_key"],
            "extracted_matches_actual": _as_review_bool(row["is_same_molecule"]),
            "extracted_skeleton_matches_actual": _as_review_bool(
                row["is_same_skeleton"]
            ),
        },
        "spectrum": {
            "representation": "literature_full_spectrum_peak_annotations",
            "nucleus": nucleus,
            "frequency_text": row["NMR_frequency"],
            "frequency_mhz": _parse_frequency_mhz(row["NMR_frequency"]),
            "solvent": row["NMR_solvent"],
            "source_shift_text_digest": _text_digest(row["NMR_shift_text"]),
            "source_note_digest": (
                _text_digest(row["NMR_note"]) if row["NMR_note"] else None
            ),
            "processed_peak_count": len(peaks),
            "processed_peaks": peaks,
            "raw_fid_or_dense_trace": False,
            "atom_level_peak_assignments": False,
        },
        "human_review": {
            "frequency_extraction": frequency_label,
            "solvent_extraction": solvent_label,
            "processed_peak_extraction": processed_label,
            "critical_spectrum_fields_all_right": labels_all_right,
            "structure_exact_match": _as_review_bool(row["is_same_molecule"]),
            "structure_skeleton_match": _as_review_bool(row["is_same_skeleton"]),
            "semantics": (
                "right/wrong fields grade automatic extraction against the source "
                "PDF; they are not chemical class labels or model probabilities"
            ),
        },
        "calibration": {
            "model_nucleus_supported": nucleus in CURRENT_MODEL_NUCLEI,
            "base_index_overlap_status": "not_checked",
            "base_index_exact_inchi_key_overlap": None,
            "base_index_connectivity_overlap": None,
            "base_index_exact_parent_inchi_key_overlap": None,
            "base_index_parent_connectivity_overlap": None,
            "parent_standardization_version": PARENT_STANDARDIZATION_VERSION,
            "parent_eligible": actual["parent_eligible"],
            "strict_independent_eligible": False,
            "exclusion_reasons": ["base index overlap has not been checked"],
        },
    }


def _audit_base_index(
    records: Sequence[Mapping[str, Any]],
    index_path: str | Path | None,
) -> tuple[dict[str, Any], set[str], set[str]]:
    if index_path is None:
        return (
            {
                "status": "not_checked",
                "base_index_sha256": None,
                "base_index_schema_version": None,
                "base_source_snapshots": [],
                "parent_standardization_version": PARENT_STANDARDIZATION_VERSION,
                "valid_external_parent_structures": sum(
                    record["structure"]["parent_eligible"] is True for record in records
                ),
                "exact_parent_inchi_key_overlaps": None,
                "parent_connectivity_overlaps": None,
                "exact_inchi_key_overlaps": None,
                "connectivity_overlaps": None,
            },
            set(),
            set(),
        )
    path = Path(index_path).expanduser().resolve()
    if not path.is_file():
        raise IndependentNMRDataError(f"base NMR index does not exist: {path}")
    before_sha256 = _sha256_file(path)
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
        base_rows = conn.execute(
            """
            SELECT id, smiles
            FROM molecules
            WHERE smiles IS NOT NULL AND TRIM(smiles) != ''
            ORDER BY id
            """
        ).fetchall()
        base_keys: set[str] = set()
        invalid_base_structures = 0
        base_multifragment_structures = 0
        for row in base_rows:
            try:
                identity = _structure_identity(
                    str(row["smiles"]),
                    quarantine_multifragment=False,
                )
            except IndependentNMRDataError:
                invalid_base_structures += 1
                continue
            parent_key = identity["inchi_key"]
            if not identity["parent_eligible"] or not parent_key:
                invalid_base_structures += 1
                continue
            base_multifragment_structures += int(identity["multi_fragment_or_mixture"])
            base_keys.add(str(parent_key).upper())
    except Exception as exc:
        raise IndependentNMRDataError(
            f"cannot audit the base NMR index: {exc}"
        ) from exc
    finally:
        conn.close()
    after_sha256 = _sha256_file(path)
    if before_sha256 != after_sha256:
        raise IndependentNMRDataError(
            "base NMR index changed while overlap was being audited"
        )
    base_molecule_keys = {key[:14] for key in base_keys}
    external_keys = {
        str(record["structure"]["inchi_key"]).upper()
        for record in records
        if record["structure"]["parent_eligible"] and record["structure"]["inchi_key"]
    }
    exact_matches = external_keys & base_keys
    connectivity_matches = {
        key[:14] for key in external_keys if key[:14] in base_molecule_keys
    }
    return (
        {
            "status": "checked",
            "base_index": "base-index.sqlite",
            "base_index_sha256": after_sha256,
            "base_index_schema_version": version,
            "base_source_snapshots": snapshots,
            "parent_standardization_version": PARENT_STANDARDIZATION_VERSION,
            "base_structure_rows_considered": len(base_rows),
            "base_standardized_parent_structures": len(base_keys),
            "base_multifragment_structures_parent_normalized": (
                base_multifragment_structures
            ),
            "invalid_base_structures": invalid_base_structures,
            "valid_external_parent_structures": len(external_keys),
            "quarantined_external_multifragment_structures": sum(
                record["structure"]["multi_fragment_or_mixture"] for record in records
            ),
            "exact_parent_inchi_key_overlaps": len(exact_matches),
            "parent_connectivity_overlaps": len(connectivity_matches),
            # Compatibility aliases now have explicit standardized-parent
            # semantics; v2 consumers should use the fields above.
            "exact_inchi_key_overlaps": len(exact_matches),
            "connectivity_overlaps": len(connectivity_matches),
            "parent_connectivity_overlap_rate": (
                len(connectivity_matches) / max(len(external_keys), 1)
            ),
            "connectivity_overlap_rate": (
                len(connectivity_matches) / max(len(external_keys), 1)
            ),
            "overlapping_parent_molecule_keys": sorted(connectivity_matches),
            "overlapping_molecule_keys": sorted(connectivity_matches),
        },
        exact_matches,
        connectivity_matches,
    )


def _apply_calibration_eligibility(
    record: dict[str, Any],
    *,
    overlap_checked: bool,
    exact_matches: set[str],
    connectivity_matches: set[str],
) -> None:
    nucleus_supported = bool(record["calibration"]["model_nucleus_supported"])
    structure = record["structure"]
    inchi_key = structure["inchi_key"]
    parent_eligible = structure.get("parent_eligible") is True
    structure_valid = bool(inchi_key) and parent_eligible
    critical_right = bool(record["human_review"]["critical_spectrum_fields_all_right"])
    exact_overlap = (
        str(inchi_key).upper() in exact_matches
        if overlap_checked and inchi_key
        else False
    )
    connectivity_overlap = (
        str(inchi_key)[:14].upper() in connectivity_matches
        if overlap_checked and inchi_key
        else False
    )
    reasons: list[str] = []
    if not nucleus_supported:
        reasons.append("nucleus is outside the current 1H/13C model scope")
    if not structure_valid:
        if structure.get("multi_fragment_or_mixture") is True:
            reasons.append(
                "smiles_actual is multi-fragment or a mixture and is quarantined"
            )
        else:
            reasons.append(
                "smiles_actual has no eligible standardized single organic parent"
            )
    if not critical_right:
        reasons.append("one or more reviewed spectrum metadata fields are not right")
    if not overlap_checked:
        reasons.append("base index overlap has not been checked")
    elif connectivity_overlap:
        reasons.append(
            "standardized parent connectivity overlaps the base nmrshiftdb2 index"
        )
    eligible = (
        nucleus_supported
        and structure_valid
        and critical_right
        and overlap_checked
        and not connectivity_overlap
    )
    record["calibration"] = {
        "model_nucleus_supported": nucleus_supported,
        "base_index_overlap_status": "checked" if overlap_checked else "not_checked",
        "base_index_exact_inchi_key_overlap": (
            exact_overlap if overlap_checked and structure_valid else None
        ),
        "base_index_connectivity_overlap": (
            connectivity_overlap if overlap_checked and structure_valid else None
        ),
        "base_index_exact_parent_inchi_key_overlap": (
            exact_overlap if overlap_checked and structure_valid else None
        ),
        "base_index_parent_connectivity_overlap": (
            connectivity_overlap if overlap_checked and structure_valid else None
        ),
        "parent_standardization_version": PARENT_STANDARDIZATION_VERSION,
        "parent_eligible": parent_eligible,
        "strict_independent_eligible": eligible,
        "exclusion_reasons": reasons,
    }


def derive_reviewed_dataset(
    source: Mapping[str, Any],
    source_directory: str | Path,
    *,
    base_index_path: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Verify, deduplicate, normalize and overlap-audit the six checked CSVs."""

    if (
        source.get("record_id") != RECORD_ID
        or source.get("record_revision") != RECORD_REVISION
        or source.get("doi") != SOURCE_DOI
        or (source.get("license") or {}).get("spdx_id") != SOURCE_LICENSE
        or (source.get("version_policy") or {}).get("mode") != "specific_record_only"
    ):
        raise IndependentNMRDataError("unexpected selected-source identity or license")
    verified, rows_by_file = load_source_file_snapshots(
        source,
        source_directory,
    )
    deduplicated = deduplicate_checked_rows(rows_by_file)
    scope = source["scope"]
    surface_rows = sum(len(rows) for rows in rows_by_file.values())
    alias_rows = sum(len(item["source_aliases"]) - 1 for item in deduplicated)
    if surface_rows != int(scope["surface_rows"]):
        raise IndependentNMRDataError("checked source surface-row count changed")
    if len(deduplicated) != int(scope["expected_unique_content_rows"]):
        raise IndependentNMRDataError("checked source unique-row count changed")
    if alias_rows != int(scope["expected_duplicate_alias_rows"]):
        raise IndependentNMRDataError("checked source alias-duplicate count changed")
    duplicate_groups = sum(len(item["source_aliases"]) > 1 for item in deduplicated)
    if duplicate_groups != 200 or any(
        len(item["source_aliases"]) > 2 for item in deduplicated
    ):
        raise IndependentNMRDataError("checked source duplicate topology changed")

    records = [_normalise_reviewed_record(item) for item in deduplicated]
    nucleus_counts = Counter(record["spectrum"]["nucleus"] for record in records)
    if dict(nucleus_counts) != EXPECTED_NUCLEUS_COUNTS:
        raise IndependentNMRDataError("checked source nucleus distribution changed")
    overlap, exact_matches, connectivity_matches = _audit_base_index(
        records,
        base_index_path,
    )
    overlap_checked = overlap["status"] == "checked"
    for record in records:
        _apply_calibration_eligibility(
            record,
            overlap_checked=overlap_checked,
            exact_matches=exact_matches,
            connectivity_matches=connectivity_matches,
        )

    valid_reported_structures = [
        record
        for record in records
        if record["structure"]["reported_inchi_key"] is not None
    ]
    valid_parent_structures = [
        record for record in records if record["structure"]["parent_eligible"] is True
    ]
    current_scope = [
        record
        for record in records
        if record["spectrum"]["nucleus"] in CURRENT_MODEL_NUCLEI
    ]
    future_scope = [
        record
        for record in records
        if record["spectrum"]["nucleus"] not in CURRENT_MODEL_NUCLEI
    ]
    strict_metadata = [
        record
        for record in current_scope
        if record["human_review"]["critical_spectrum_fields_all_right"]
        and record["structure"]["parent_eligible"] is True
    ]
    nonoverlap_current = [
        record
        for record in current_scope
        if record["structure"]["parent_eligible"] is True
        and record["calibration"]["base_index_connectivity_overlap"] is False
    ]
    strict_independent = [
        record
        for record in current_scope
        if record["calibration"]["strict_independent_eligible"]
    ]
    source_documents = {record["source"]["document_key"] for record in records}
    source_dois = {
        record["source"]["document_doi"]
        for record in records
        if record["source"]["document_doi"]
    }
    summary = {
        "schema_version": DERIVED_SCHEMA_VERSION,
        "source": {
            "title": SOURCE_TITLE,
            "creators": list(SOURCE_CREATORS),
            "record_id": RECORD_ID,
            "record_revision": RECORD_REVISION,
            "concept_record_id": CONCEPT_RECORD_ID,
            "doi": SOURCE_DOI,
            "concept_doi": CONCEPT_DOI,
            "paper_doi": source["paper_doi"],
            "license_spdx": SOURCE_LICENSE,
            "license_uri": SOURCE_LICENSE_URI,
            "version_resolution": "specific_record_only",
            "selected_file_inventory_sha256": source["inventory_sha256"],
            "modification_notice": DERIVATION_CHANGE_NOTICE,
            "attribution_manifest": "ATTRIBUTION.json",
        },
        "migration": {
            "from_schema_version": "nmrexp-human-reviewed-derived-v1",
            "to_schema_version": DERIVED_SCHEMA_VERSION,
            "breaking": True,
            "change_notice": (
                "v2 uses standardized single-organic-parent identities for "
                "overlap and eligibility, quarantines all multi-fragment human "
                "truth values, removes redistributed shift/note prose, and "
                "publishes records plus summary through an atomic release pointer."
            ),
        },
        "transport": {
            "selected_files": len(verified),
            "selected_bytes": sum(int(item["bytes"]) for item in verified),
            "files": verified,
            "large_automatic_corpus_downloaded": False,
        },
        "deduplication": {
            "surface_rows": surface_rows,
            "content_unique_rows": len(records),
            "duplicate_alias_rows": alias_rows,
            "duplicate_content_groups": duplicate_groups,
            "maximum_aliases_per_record": max(
                len(record["source_aliases"]) for record in records
            ),
            "hash_basis": (
                "SHA-256 of compact, key-sorted JSON over all 23 byte-preserved "
                "CSV fields"
            ),
        },
        "content": {
            "unique_source_documents": len(source_documents),
            "normalized_source_dois": len(source_dois),
            "nucleus_counts": dict(sorted(nucleus_counts.items())),
            "parsed_peak_entries": sum(
                int(record["spectrum"]["processed_peak_count"]) for record in records
            ),
            "human_actual_structure_values": len(
                {record["structure"]["human_corrected_smiles"] for record in records}
            ),
            "machine_parseable_actual_structures": len(valid_reported_structures),
            "machine_parseable_reported_actual_structures": len(
                valid_reported_structures
            ),
            "eligible_standardized_parent_structures": len(valid_parent_structures),
            "quarantined_multifragment_structures": sum(
                record["structure"]["multi_fragment_or_mixture"] for record in records
            ),
            "unparseable_actual_structure_notes": len(records)
            - len(valid_reported_structures),
            "extracted_exact_structure_matches": sum(
                bool(record["human_review"]["structure_exact_match"])
                for record in records
            ),
            "extracted_skeleton_matches": sum(
                bool(record["human_review"]["structure_skeleton_match"])
                for record in records
            ),
            "raw_fid_or_dense_trace_records": 0,
            "atom_level_assignment_records": 0,
            "source_shift_or_note_prose_redistributed": False,
            "source_shift_text_hashes": len(records),
            "source_note_hashes": sum(
                record["spectrum"]["source_note_digest"] is not None
                for record in records
            ),
        },
        "overlap": overlap,
        "calibration_scope": {
            "current_1h_13c_rows": len(current_scope),
            "future_nucleus_rows": len(future_scope),
            "current_rows_with_valid_actual_structure": sum(
                record["structure"]["parent_eligible"] is True
                for record in current_scope
            ),
            "current_rows_with_all_critical_review_labels_right": len(strict_metadata),
            "current_rows_without_base_connectivity_overlap": (
                len(nonoverlap_current) if overlap_checked else None
            ),
            "strict_independent_eligible_rows": (
                len(strict_independent) if overlap_checked else 0
            ),
            "strict_independent_by_nucleus": dict(
                sorted(
                    Counter(
                        record["spectrum"]["nucleus"] for record in strict_independent
                    ).items()
                )
            ),
            "calibrated_probability_claim_allowed": False,
            "reason": (
                "These rows can support held-out score calibration after a "
                "source/scaffold split. They do not by themselves establish "
                "open-world identification accuracy, and they contain no raw traces."
            ),
        },
        "semantic_contract": {
            "ground_truth_structure": "smiles_actual only",
            "training_structure": (
                "RDKit-standardized single-fragment organic parent derived "
                "only from smiles_actual"
            ),
            "multifragment_policy": (
                "quarantine; never select one component as training truth"
            ),
            "parent_standardization_version": PARENT_STANDARDIZATION_VERSION,
            "automatic_structure_output": "SMILES (provenance/QA only)",
            "right_wrong_labels": (
                "manual grades of automatic frequency, solvent and peak-list "
                "extraction against the source PDF"
            ),
            "structure_match_labels": (
                "manual comparison of extracted SMILES with smiles_actual"
            ),
            "prohibited_shortcut": (
                "Never replace missing or invalid smiles_actual with extracted SMILES"
            ),
            "source_prose_policy": (
                "NMR_shift_text, NMR_note and text_in_pdf are represented only "
                "by SHA-256 and character count in redistributed records"
            ),
        },
    }
    if len(current_scope) != int(scope["current_model_scope_rows"]):
        raise IndependentNMRDataError("current model calibration scope changed")
    if len(future_scope) != int(scope["future_nucleus_rows"]):
        raise IndependentNMRDataError("future nucleus scope changed")
    if (
        len(valid_reported_structures) != 499
        or len(valid_parent_structures) != 494
        or len(source_documents) != 486
    ):
        raise IndependentNMRDataError(
            "reviewed structure/source-document audit changed"
        )
    records.sort(key=lambda record: str(record["record_id"]))
    return records, summary


def source_inventory_payload(
    source: Mapping[str, Any],
    verified_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    records = [
        {
            "name": item["name"],
            "bytes": item["bytes"],
            "md5": item["md5"],
            "sha256": item["sha256"],
            "rows": item["rows"],
            "format_audit": item["format_audit"],
        }
        for item in verified_files
    ]
    records.sort(key=lambda item: str(item["name"]))
    expected_names = sorted(str(item["name"]) for item in source["files"])
    if [str(item["name"]) for item in records] != expected_names:
        raise IndependentNMRDataError("source inventory requires all six checked files")
    return {
        "schema_version": 2,
        "title": SOURCE_TITLE,
        "creators": list(SOURCE_CREATORS),
        "record_id": RECORD_ID,
        "record_revision": RECORD_REVISION,
        "concept_record_id": CONCEPT_RECORD_ID,
        "doi": SOURCE_DOI,
        "concept_doi": CONCEPT_DOI,
        "license_spdx": SOURCE_LICENSE,
        "license_uri": SOURCE_LICENSE_URI,
        "paper_doi": source["paper_doi"],
        "modification_notice": DERIVATION_CHANGE_NOTICE,
        "attribution": _attribution_payload(),
        "version_resolution": "specific_record_only",
        "selected_inventory_sha256": source["inventory_sha256"],
        "files": records,
    }


def _attribution_payload() -> dict[str, Any]:
    return {
        "schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "artifact": "ChemApp NMRexp human-reviewed derived spectrum records",
        "source": {
            "title": SOURCE_TITLE,
            "creators": list(SOURCE_CREATORS),
            "record_doi": SOURCE_DOI,
            "concept_doi": CONCEPT_DOI,
            "paper_doi": "10.1038/s41597-025-06245-5",
            "record_uri": f"https://zenodo.org/records/{RECORD_ID}",
        },
        "license": {
            "spdx_id": SOURCE_LICENSE,
            "uri": SOURCE_LICENSE_URI,
        },
        "modifications": {
            "modified": True,
            "notice": DERIVATION_CHANGE_NOTICE,
            "source_prose_redistributed": False,
            "standardized_parent_version": PARENT_STANDARDIZATION_VERSION,
        },
        "attribution_instruction": (
            "Retain this manifest with any redistributed derived records."
        ),
    }


def _assert_redistribution_safe(
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Fail closed before publishing a third-party derived-data release."""

    source = summary.get("source")
    deduplication = summary.get("deduplication")
    if (
        summary.get("schema_version") != DERIVED_SCHEMA_VERSION
        or not isinstance(source, Mapping)
        or source.get("title") != SOURCE_TITLE
        or source.get("creators") != list(SOURCE_CREATORS)
        or source.get("record_id") != RECORD_ID
        or source.get("doi") != SOURCE_DOI
        or source.get("license_spdx") != SOURCE_LICENSE
        or source.get("license_uri") != SOURCE_LICENSE_URI
        or not isinstance(deduplication, Mapping)
        or deduplication.get("content_unique_rows") != len(records)
    ):
        raise IndependentNMRDataError(
            "derived summary identity, attribution, or record count changed"
        )

    def visit(value: Any, *, location: str) -> None:
        if isinstance(value, Mapping):
            for raw_key, item in value.items():
                key = str(raw_key)
                if key in SOURCE_PROSE_KEYS:
                    raise IndependentNMRDataError(
                        f"source prose field {key} is forbidden in {location}"
                    )
                visit(item, location=f"{location}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                visit(item, location=f"{location}[{index}]")

    for index, record in enumerate(records):
        record_source = record.get("source")
        if (
            record.get("schema_version") != DERIVED_SCHEMA_VERSION
            or not isinstance(record_source, Mapping)
            or record_source.get("attribution_manifest_required") is not True
            or record_source.get("license_spdx") != SOURCE_LICENSE
        ):
            raise IndependentNMRDataError(
                f"derived record {index} identity or attribution contract changed"
            )
        visit(record, location=f"records[{index}]")
    visit(summary, location="summary")


def _write_immutable_text(
    path: Path,
    rendered: str,
    *,
    overwrite: bool,
) -> None:
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise IndependentNMRDataError(
                f"cannot inspect existing derived file {path.name}: {exc}"
            ) from exc
        if existing == rendered:
            return
        if not overwrite:
            raise IndependentNMRDataError(
                f"existing {path.name} differs; pass --overwrite only after review"
            )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=f".{path.name}.",
            suffix=".part",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_source_inventory(
    source_directory: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    root = Path(source_directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "inventory.json"
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    _write_immutable_text(target, rendered + "\n", overwrite=overwrite)
    return target


def write_derived_dataset(
    destination: str | Path,
    records: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Publish an immutable release and atomically switch ``CURRENT.json``.

    ``CURRENT.json`` is the sole commit point. A crash can leave an unreferenced
    immutable release directory, but can never expose new records with an old
    summary (or the reverse).
    """

    root = Path(destination).resolve()
    releases_root = root / "releases"
    releases_root.mkdir(parents=True, exist_ok=True)
    _assert_redistribution_safe(records, summary)
    records_rendered = "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for record in records
    )
    records_bytes = records_rendered.encode("utf-8")
    records_sha256 = hashlib.sha256(records_bytes).hexdigest()
    attribution = _attribution_payload()
    attribution_rendered = (
        json.dumps(
            attribution,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    attribution_bytes = attribution_rendered.encode("utf-8")
    attribution_sha256 = hashlib.sha256(attribution_bytes).hexdigest()
    summary_value = json.loads(_canonical_json(dict(summary)))
    summary_value["publication"] = {
        "schema_version": DERIVED_RELEASE_SCHEMA_VERSION,
        "commit_protocol": (
            "immutable version directory committed by atomic CURRENT.json replace"
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
    summary_core_sha256 = hashlib.sha256(
        _canonical_json(summary_value).encode("utf-8")
    ).hexdigest()
    release_id = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": DERIVED_RELEASE_SCHEMA_VERSION,
                "summary_core_sha256": summary_core_sha256,
                "records_sha256": records_sha256,
                "attribution_sha256": attribution_sha256,
            }
        ).encode("utf-8")
    ).hexdigest()
    summary_value["publication"].update(
        {
            "release_id": release_id,
            "summary_core_sha256": summary_core_sha256,
        }
    )
    summary_rendered = (
        json.dumps(
            summary_value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    summary_bytes = summary_rendered.encode("utf-8")
    summary_sha256 = hashlib.sha256(summary_bytes).hexdigest()

    release_directory = releases_root / release_id
    if release_directory.exists():
        expected = {
            "records.jsonl": records_bytes,
            "summary.json": summary_bytes,
            "ATTRIBUTION.json": attribution_bytes,
        }
        for name, expected_bytes in expected.items():
            path = release_directory / name
            if not path.is_file() or path.read_bytes() != expected_bytes:
                raise IndependentNMRDataError(
                    f"immutable derived release collision or corruption: {path}"
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
            for name, payload in (
                ("records.jsonl", records_bytes),
                ("summary.json", summary_bytes),
                ("ATTRIBUTION.json", attribution_bytes),
            ):
                path = staging / name
                with path.open("wb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            if (
                hashlib.sha256((staging / "records.jsonl").read_bytes()).hexdigest()
                != records_sha256
                or hashlib.sha256((staging / "summary.json").read_bytes()).hexdigest()
                != summary_sha256
                or hashlib.sha256(
                    (staging / "ATTRIBUTION.json").read_bytes()
                ).hexdigest()
                != attribution_sha256
            ):
                raise IndependentNMRDataError(
                    "derived staging release failed its publication hash audit"
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
    current_path = root / "CURRENT.json"
    _write_immutable_text(
        current_path,
        json.dumps(
            pointer,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        overwrite=overwrite,
    )
    return {
        "current": current_path,
        "release": release_directory,
        "records": release_directory / "records.jsonl",
        "summary": release_directory / "summary.json",
        "attribution": release_directory / "ATTRIBUTION.json",
    }


def load_derived_summary(path: str | Path) -> dict[str, Any]:
    """Load a release summary only after verifying its bound artifact set."""

    selected = Path(path).resolve()
    pointer: Mapping[str, Any] | None = None
    if selected.is_dir():
        selected = selected / "CURRENT.json"
    if selected.name == "CURRENT.json":
        try:
            pointer_bytes = _read_immutable_file_snapshot(
                selected,
                maximum_bytes=1024 * 1024,
            )
            pointer_value = json.loads(pointer_bytes.decode("utf-8"))
        except (
            IndependentNMRDataError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            raise IndependentNMRDataError(
                f"cannot load derived release pointer: {exc}"
            ) from exc
        if (
            not isinstance(pointer_value, dict)
            or pointer_value.get("schema_version") != DERIVED_RELEASE_SCHEMA_VERSION
        ):
            raise IndependentNMRDataError("derived release pointer schema changed")
        release_id = pointer_value.get("release_id")
        if not isinstance(release_id, str) or not SHA256_RE.fullmatch(release_id):
            raise IndependentNMRDataError("derived release pointer ID is invalid")
        expected_relative = f"releases/{release_id}"
        if pointer_value.get("release_directory") != expected_relative:
            raise IndependentNMRDataError("derived release pointer directory changed")
        release_directory = selected.parent / "releases" / release_id
        if release_directory.parent != (selected.parent / "releases").resolve():
            raise IndependentNMRDataError("derived release pointer escaped its root")
        selected = release_directory / "summary.json"
        pointer = pointer_value
    summary_path = selected
    try:
        summary_bytes = _read_immutable_file_snapshot(
            summary_path,
            maximum_bytes=16 * 1024 * 1024,
        )
        value = json.loads(summary_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IndependentNMRDataError(
            f"cannot load reviewed-data summary: {exc}"
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != DERIVED_SCHEMA_VERSION
        or (value.get("source") or {}).get("record_id") != RECORD_ID
        or (value.get("source") or {}).get("doi") != SOURCE_DOI
        or (value.get("source") or {}).get("license_spdx") != SOURCE_LICENSE
        or (value.get("deduplication") or {}).get("content_unique_rows") != 500
    ):
        raise IndependentNMRDataError("reviewed-data summary identity or scope changed")
    publication = value.get("publication")
    if (
        not isinstance(publication, dict)
        or publication.get("schema_version") != DERIVED_RELEASE_SCHEMA_VERSION
    ):
        raise IndependentNMRDataError(
            "reviewed-data publication manifest is missing or changed"
        )
    records_binding = publication.get("records")
    attribution_binding = publication.get("attribution")
    if not isinstance(records_binding, dict) or not isinstance(
        attribution_binding,
        dict,
    ):
        raise IndependentNMRDataError("derived artifact bindings are missing")
    release_directory = summary_path.parent
    records_path = release_directory / "records.jsonl"
    attribution_path = release_directory / "ATTRIBUTION.json"
    try:
        records_bytes = _read_immutable_file_snapshot(
            records_path,
            maximum_bytes=128 * 1024 * 1024,
        )
        attribution_bytes = _read_immutable_file_snapshot(
            attribution_path,
            maximum_bytes=1024 * 1024,
        )
    except IndependentNMRDataError:
        raise
    if (
        records_binding.get("file") != "records.jsonl"
        or records_binding.get("sha256") != hashlib.sha256(records_bytes).hexdigest()
        or records_binding.get("bytes") != len(records_bytes)
    ):
        raise IndependentNMRDataError("derived records binding failed")
    record_lines = [line for line in records_bytes.splitlines() if line.strip()]
    if records_binding.get("count") != len(record_lines):
        raise IndependentNMRDataError("derived records row-count binding failed")
    for line_number, line in enumerate(record_lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise IndependentNMRDataError(
                f"derived records line {line_number} is invalid JSON"
            ) from exc
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != DERIVED_SCHEMA_VERSION
        ):
            raise IndependentNMRDataError(
                f"derived records line {line_number} schema changed"
            )
    if (
        attribution_binding.get("file") != "ATTRIBUTION.json"
        or attribution_binding.get("sha256")
        != hashlib.sha256(attribution_bytes).hexdigest()
        or attribution_binding.get("bytes") != len(attribution_bytes)
        or attribution_binding.get("required_for_redistribution") is not True
    ):
        raise IndependentNMRDataError("derived attribution binding failed")
    try:
        attribution = json.loads(attribution_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise IndependentNMRDataError("derived attribution is invalid") from exc
    if not isinstance(attribution, dict) or _canonical_json(
        attribution
    ) != _canonical_json(_attribution_payload()):
        raise IndependentNMRDataError("derived attribution contract changed")
    release_id = publication.get("release_id")
    core_sha = publication.get("summary_core_sha256")
    if (
        not isinstance(release_id, str)
        or not SHA256_RE.fullmatch(release_id)
        or not isinstance(core_sha, str)
        or not SHA256_RE.fullmatch(core_sha)
    ):
        raise IndependentNMRDataError("derived release identity is invalid")
    core = json.loads(_canonical_json(value))
    core["publication"].pop("release_id", None)
    core["publication"].pop("summary_core_sha256", None)
    actual_core_sha = hashlib.sha256(_canonical_json(core).encode("utf-8")).hexdigest()
    expected_release_id = hashlib.sha256(
        _canonical_json(
            {
                "schema_version": DERIVED_RELEASE_SCHEMA_VERSION,
                "summary_core_sha256": actual_core_sha,
                "records_sha256": records_binding["sha256"],
                "attribution_sha256": attribution_binding["sha256"],
            }
        ).encode("utf-8")
    ).hexdigest()
    if actual_core_sha != core_sha or expected_release_id != release_id:
        raise IndependentNMRDataError("derived release identity binding failed")
    if release_directory.name != release_id:
        raise IndependentNMRDataError(
            "derived summary is not inside its immutable release directory"
        )
    if pointer is not None:
        if (
            pointer.get("summary_sha256") != hashlib.sha256(summary_bytes).hexdigest()
            or pointer.get("records_sha256") != records_binding["sha256"]
            or pointer.get("attribution_sha256") != attribution_binding["sha256"]
        ):
            raise IndependentNMRDataError("CURRENT.json artifact binding failed")
    return value
