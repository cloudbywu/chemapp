"""Safe, immutable retrieval for the pinned six-sample external smoke set."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Protocol
from urllib.parse import unquote, urlparse
from urllib.request import Request, urlopen

MANIFEST_SCHEMA_VERSION = 1
RECORD_ID = 16881130
RECORD_REVISION = 4
RECORD_HOST = "zenodo.org"
ALLOWED_RESPONSE_HOSTS = {"zenodo.org", "files.zenodo.org"}
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
HTML_PREFIXES = (b"<!doctype html", b"<html")
DEFAULT_CHUNK_SIZE = 1024 * 1024
EXPECTED_RECORD_METADATA: dict[str, Any] = {
    "schema_version": MANIFEST_SCHEMA_VERSION,
    "record_id": RECORD_ID,
    "record_revision": RECORD_REVISION,
    "doi": "10.5281/zenodo.16881130",
    "title": (
        "Proof-of-Concept Dataset of Chemical Structures with NMR, IR and MS "
        "Spectra."
    ),
    "publication_date": "2025-08-15",
    "record_uri": f"https://zenodo.org/records/{RECORD_ID}",
    "api_uri": f"https://zenodo.org/api/records/{RECORD_ID}",
    "license": {
        "spdx_id": "CC0-1.0",
        "record_value": "cc-zero",
        "uri": "https://creativecommons.org/publicdomain/zero/1.0/legalcode",
    },
    "creators": [
        "Henzelin, Kenan",
        "Risse, Lucas",
        "Patiny, Luc",
    ],
    "scope": {
        "sample_count": 6,
        "role": "proof_of_concept_external_smoke_only",
        "headline_benchmark_allowed": False,
        "reason": (
            "Six samples cannot support a general model-accuracy or "
            "generalization claim."
        ),
    },
    "archive_policy": {
        "extract_by_default": False,
        "max_entries_per_zip": 10000,
        "max_uncompressed_bytes_per_zip": 2147483648,
        "max_compression_ratio": 100.0,
        "require_crc_check": True,
        "reject_absolute_or_parent_paths": True,
    },
    "inventory_sha256": (
        "e29cf96b2b210371e6df592cb3cc1736006c9b40af6777f1c8f37dd5080c4a94"
    ),
    "inventory_hash_basis": (
        "SHA-256 of compact JSON for the filename-sorted array of name, bytes "
        "and md5 fields"
    ),
    "total_bytes": 97728550,
}
EXPECTED_FILE_METADATA: dict[str, dict[str, Any]] = {
    "1.zip": {
        "bytes": 25909010,
        "md5": "15235791209c0e5ace35eba254487a72",
        "sha256": (
            "93a2c44f19e73aa8816d87ab1ee9d0ae9e27638f3632aaf9075fc7218206571f"
        ),
        "media_type": "application/zip",
    },
    "2.zip": {
        "bytes": 13094888,
        "md5": "955974fa106ea0a95e1e1674ec98bea9",
        "sha256": (
            "ec9e4e72a9a624fae55f9e3f73ad327c49556bd13673f8fd2319de4a94cc872a"
        ),
        "media_type": "application/zip",
    },
    "3.zip": {
        "bytes": 12081362,
        "md5": "6b862b0dc3908023a3385f7b6bd04bea",
        "sha256": (
            "396620fc65d29c51abf47fc80a400c9c9af76eb3771fe5a24e44334469a7e80f"
        ),
        "media_type": "application/zip",
    },
    "4.zip": {
        "bytes": 209431,
        "md5": "6045d4664a07883039e6ff7738002f82",
        "sha256": (
            "6411d139185d88c5738f15aa68bb5b97d53e4fa8a777d236770533e6ce8b238f"
        ),
        "media_type": "application/zip",
    },
    "5.zip": {
        "bytes": 12557395,
        "md5": "6741417eb75edb11c4e87e7046050df5",
        "sha256": (
            "b39d63221cd6d0820cce22649e42d07db73823d744990d1bfdff8235954b85bf"
        ),
        "media_type": "application/zip",
    },
    "6.zip": {
        "bytes": 31440624,
        "md5": "2df5f56a764063d7fa3013e1d0ae928f",
        "sha256": (
            "3f17caa6f907c432a5f68c0f95dd9221d82ac8b2cd968f32798271cc16899059"
        ),
        "media_type": "application/zip",
    },
    "README.md": {
        "bytes": 2893,
        "md5": "f51498877e99439e82759997f971c772",
        "sha256": (
            "effa446dbd2a6424c8a1e0d98709e2b0c74457cf8b9c0c3bec47adc66e430a49"
        ),
        "media_type": "text/markdown",
    },
    "toc.json": {
        "bytes": 2432947,
        "md5": "8fab67e538e8f76beef2b9aabe3a7d6b",
        "sha256": (
            "f8db0bd095ba952e606f7cb6a0bbc6cd47469757e985467c7424a7adae44fd9f"
        ),
        "media_type": "application/json",
    },
}
EXPECTED_MANIFEST_KEYS = frozenset({*EXPECTED_RECORD_METADATA, "files"})
EXPECTED_FILE_KEYS = frozenset(
    {"name", "bytes", "md5", "sha256", "url", "media_type"}
)


class ExternalDatasetError(ValueError):
    """Raised when the fixed manifest or downloaded bytes fail validation."""


class ResponseLike(Protocol):
    """Subset of urllib response behavior used by the downloader."""

    headers: Mapping[str, str]

    def read(self, size: int = -1) -> bytes: ...

    def geturl(self) -> str: ...

    def __enter__(self) -> ResponseLike: ...

    def __exit__(self, *args: object) -> object: ...


Opener = Callable[..., ResponseLike]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DEFAULT_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _md5_file(path: Path) -> str:
    # MD5 is required only to match Zenodo's published transport checksum.
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
    return json.dumps(
        selected,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _canonical_json(value: Any) -> str:
    """Return a type-sensitive, order-independent JSON representation."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def load_fixed_manifest(path: str | Path) -> dict[str, Any]:
    """Load and fail closed on any change to the fixed Zenodo record."""

    manifest_path = Path(path).resolve()
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalDatasetError(f"cannot load fixed manifest: {exc}") from exc
    if not isinstance(value, dict):
        raise ExternalDatasetError("fixed manifest root must be an object")
    if set(value) != EXPECTED_MANIFEST_KEYS:
        raise ExternalDatasetError("fixed manifest fields changed")
    for field, expected in EXPECTED_RECORD_METADATA.items():
        if _canonical_json(value.get(field)) != _canonical_json(expected):
            raise ExternalDatasetError(f"fixed manifest {field} changed")

    files = value.get("files")
    if not isinstance(files, list) or len(files) != 8:
        raise ExternalDatasetError("the fixed record must contain exactly eight files")

    names: set[str] = set()
    total_bytes = 0
    for item in files:
        if not isinstance(item, dict) or not _safe_filename(item.get("name")):
            raise ExternalDatasetError("manifest contains an unsafe filename")
        if set(item) != EXPECTED_FILE_KEYS:
            raise ExternalDatasetError("fixed file manifest fields changed")
        name = str(item["name"])
        if name in names:
            raise ExternalDatasetError(f"duplicate manifest filename: {name}")
        names.add(name)
        expected_file = EXPECTED_FILE_METADATA.get(name)
        if expected_file is None:
            raise ExternalDatasetError(f"unexpected fixed record filename: {name}")
        expected_record = {
            "name": name,
            **expected_file,
            "url": (
                f"https://zenodo.org/records/{RECORD_ID}/files/{name}?download=1"
            ),
        }
        if _canonical_json(item) != _canonical_json(expected_record):
            raise ExternalDatasetError(f"fixed metadata changed for {name}")
        byte_size = item.get("bytes")
        if not isinstance(byte_size, int) or byte_size <= 0:
            raise ExternalDatasetError(f"invalid byte size for {name}")
        total_bytes += byte_size
        if not isinstance(item.get("md5"), str) or not MD5_RE.fullmatch(item["md5"]):
            raise ExternalDatasetError(f"invalid Zenodo MD5 for {name}")
        if not isinstance(item.get("sha256"), str) or not SHA256_RE.fullmatch(
            item["sha256"]
        ):
            raise ExternalDatasetError(f"invalid frozen SHA-256 for {name}")
        parsed = urlparse(str(item.get("url", "")))
        expected_path = f"/records/{RECORD_ID}/files/{name}"
        if (
            parsed.scheme != "https"
            or parsed.hostname != RECORD_HOST
            or unquote(parsed.path) != expected_path
            or parsed.query != "download=1"
            or parsed.fragment
        ):
            raise ExternalDatasetError(f"unsafe or unexpected download URL for {name}")

    if names != set(EXPECTED_FILE_METADATA):
        raise ExternalDatasetError("fixed record file set changed")
    if total_bytes != value.get("total_bytes"):
        raise ExternalDatasetError("fixed record total byte count changed")
    inventory_digest = hashlib.sha256(_canonical_inventory(files)).hexdigest()
    if inventory_digest != value.get("inventory_sha256"):
        raise ExternalDatasetError("fixed record inventory digest changed")
    return value


def _safe_member_name(name: str) -> bool:
    if not name or "\x00" in name:
        return False
    normalized = name.replace("\\", "/")
    posix = PurePosixPath(normalized)
    windows = PureWindowsPath(name)
    return (
        not posix.is_absolute()
        and not windows.is_absolute()
        and not windows.drive
        and ".." not in posix.parts
        and ".." not in windows.parts
    )


def inspect_zip(
    path: str | Path,
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate central-directory limits, paths and CRC without extraction."""

    archive_path = Path(path)
    try:
        with zipfile.ZipFile(archive_path) as archive:
            infos = archive.infolist()
            max_entries = int(policy["max_entries_per_zip"])
            if len(infos) > max_entries:
                raise ExternalDatasetError(
                    f"ZIP has {len(infos)} entries; maximum is {max_entries}"
                )
            unsafe = [info.filename for info in infos if not _safe_member_name(info.filename)]
            if unsafe:
                raise ExternalDatasetError(
                    f"ZIP contains unsafe member path: {unsafe[0]!r}"
                )
            total_uncompressed = sum(info.file_size for info in infos)
            max_uncompressed = int(policy["max_uncompressed_bytes_per_zip"])
            if total_uncompressed > max_uncompressed:
                raise ExternalDatasetError(
                    "ZIP declared uncompressed size exceeds the fixed safety limit"
                )
            total_compressed = sum(info.compress_size for info in infos)
            compression_ratio = total_uncompressed / max(total_compressed, 1)
            max_ratio = float(policy["max_compression_ratio"])
            if compression_ratio > max_ratio:
                raise ExternalDatasetError(
                    f"ZIP compression ratio {compression_ratio:.1f} exceeds {max_ratio:.1f}"
                )
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ExternalDatasetError(
                    f"ZIP CRC check failed for {bad_member!r}"
                )
            return {
                "entries": len(infos),
                "uncompressed_bytes": total_uncompressed,
                "compressed_bytes": total_compressed,
                "compression_ratio": round(compression_ratio, 6),
                "crc_ok": True,
                "paths_safe": True,
            }
    except (KeyError, TypeError, ValueError, zipfile.BadZipFile) as exc:
        if isinstance(exc, ExternalDatasetError):
            raise
        raise ExternalDatasetError(f"invalid ZIP archive: {exc}") from exc


def verify_file(
    path: str | Path,
    record: Mapping[str, Any],
    *,
    archive_policy: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify one existing file against byte count, MD5, SHA-256 and format."""

    candidate = Path(path)
    if not candidate.is_file():
        raise ExternalDatasetError(f"dataset file is missing: {candidate}")
    byte_size = candidate.stat().st_size
    if byte_size != record["bytes"]:
        raise ExternalDatasetError(
            f"{record['name']} size mismatch: expected {record['bytes']}, got {byte_size}"
        )
    md5 = _md5_file(candidate)
    if md5 != record["md5"]:
        raise ExternalDatasetError(
            f"{record['name']} MD5 mismatch: expected {record['md5']}, got {md5}"
        )
    sha256 = _sha256_file(candidate)
    if sha256 != record["sha256"]:
        raise ExternalDatasetError(
            f"{record['name']} SHA-256 mismatch: expected {record['sha256']}, got {sha256}"
        )
    with candidate.open("rb") as handle:
        prefix = handle.read(512).lstrip().lower()
    if prefix.startswith(HTML_PREFIXES) or b"<html" in prefix:
        raise ExternalDatasetError(f"{record['name']} is HTML, not dataset content")

    format_audit: dict[str, Any]
    if str(record["name"]).endswith(".zip"):
        format_audit = inspect_zip(candidate, archive_policy)
    elif record["name"] == "toc.json":
        try:
            parsed = json.loads(candidate.read_text(encoding="utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ExternalDatasetError(f"toc.json is invalid JSON: {exc}") from exc
        if not isinstance(parsed, (dict, list)):
            raise ExternalDatasetError("toc.json root must be an object or array")
        format_audit = {"json_valid": True}
    else:
        format_audit = {"nonempty_text": bool(candidate.read_text(encoding="utf-8").strip())}
        if not format_audit["nonempty_text"]:
            raise ExternalDatasetError(f"{record['name']} is empty text")
    return {
        "name": record["name"],
        "bytes": byte_size,
        "md5": md5,
        "sha256": sha256,
        "format_audit": format_audit,
    }


def _validate_response_url(response: ResponseLike, expected_url: str) -> None:
    final_url = response.geturl() or expected_url
    parsed = urlparse(final_url)
    if parsed.scheme != "https" or parsed.hostname not in ALLOWED_RESPONSE_HOSTS:
        raise ExternalDatasetError(f"download redirected to an untrusted URL: {final_url}")


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
            raise ExternalDatasetError("invalid Content-Length header") from exc
        if declared != expected_bytes:
            raise ExternalDatasetError(
                f"server Content-Length mismatch: expected {expected_bytes}, got {declared}"
            )
    content_type = str(response.headers.get("Content-Type", "")).casefold()
    if "text/html" in content_type:
        raise ExternalDatasetError("server returned HTML instead of dataset content")

    total = 0
    first = True
    while True:
        chunk = response.read(chunk_size)
        if not chunk:
            break
        if first:
            first = False
            lowered = chunk[:512].lstrip().lower()
            if lowered.startswith(HTML_PREFIXES) or b"<html" in lowered:
                raise ExternalDatasetError("download body is HTML, not dataset content")
        total += len(chunk)
        if total > expected_bytes:
            raise ExternalDatasetError("download exceeded the frozen byte count")
        handle.write(chunk)
    if total != expected_bytes:
        raise ExternalDatasetError(
            f"download ended at {total} bytes; expected {expected_bytes}"
        )


def download_file(
    record: Mapping[str, Any],
    destination: str | Path,
    *,
    archive_policy: Mapping[str, Any],
    opener: Opener = urlopen,
    timeout_seconds: float = 60.0,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Download one fixed record atomically; never overwrite invalid bytes."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not _safe_filename(record.get("name")):
        raise ExternalDatasetError("unsafe destination filename")
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root / str(record["name"])).resolve()
    if target.parent != root:
        raise ExternalDatasetError("destination escaped the dataset directory")
    if target.exists():
        result = verify_file(target, record, archive_policy=archive_policy)
        result["reused"] = True
        return result

    request = Request(
        str(record["url"]),
        headers={
            "Accept": "*/*",
            "Referer": f"https://zenodo.org/records/{RECORD_ID}",
            "User-Agent": "ChemApp-external-smoke-fetch/1",
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
            except ExternalDatasetError:
                raise
            except (OSError, TimeoutError) as exc:
                raise ExternalDatasetError(
                    f"network retrieval failed without publishing partial bytes: {exc}"
                ) from exc
            handle.flush()
            os.fsync(handle.fileno())
        result = verify_file(
            temporary_path,
            record,
            archive_policy=archive_policy,
        )
        os.replace(temporary_path, target)
        temporary_path = None
        result["reused"] = False
        return result
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def import_local_file(
    record: Mapping[str, Any],
    source_directory: str | Path,
    destination: str | Path,
    *,
    archive_policy: Mapping[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Import a previously downloaded file through the same full validation."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not _safe_filename(record.get("name")):
        raise ExternalDatasetError("unsafe source filename")
    source_root = Path(source_directory).resolve()
    source = (source_root / str(record["name"])).resolve()
    if source.parent != source_root:
        raise ExternalDatasetError("source escaped the supplied source directory")
    source_result = verify_file(
        source,
        record,
        archive_policy=archive_policy,
    )

    target_root = Path(destination).resolve()
    target_root.mkdir(parents=True, exist_ok=True)
    target = (target_root / str(record["name"])).resolve()
    if target.parent != target_root:
        raise ExternalDatasetError("destination escaped the dataset directory")
    if target.exists():
        result = verify_file(
            target,
            record,
            archive_policy=archive_policy,
        )
        result["reused"] = True
        result["source"] = "local_import"
        return result

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{record['name']}.",
            suffix=".part",
            dir=target_root,
            delete=False,
        ) as target_handle:
            temporary_path = Path(target_handle.name)
            copied = 0
            with source.open("rb") as source_handle:
                while True:
                    chunk = source_handle.read(chunk_size)
                    if not chunk:
                        break
                    copied += len(chunk)
                    if copied > int(record["bytes"]):
                        raise ExternalDatasetError(
                            "local source grew beyond the frozen byte count"
                        )
                    target_handle.write(chunk)
            if copied != int(record["bytes"]):
                raise ExternalDatasetError(
                    "local source changed while it was being imported"
                )
            target_handle.flush()
            os.fsync(target_handle.fileno())
        result = verify_file(
            temporary_path,
            record,
            archive_policy=archive_policy,
        )
        if (
            result["sha256"] != source_result["sha256"]
            or result["md5"] != source_result["md5"]
        ):
            raise ExternalDatasetError(
                "local source changed while it was being imported"
            )
        os.replace(temporary_path, target)
        temporary_path = None
        result["reused"] = False
        result["source"] = "local_import"
        return result
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def verify_dataset(
    manifest: Mapping[str, Any],
    destination: str | Path,
    *,
    selected_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Verify selected local files without any network access."""

    selected = selected_names or {str(item["name"]) for item in manifest["files"]}
    known = {str(item["name"]): item for item in manifest["files"]}
    unknown = selected - known.keys()
    if unknown:
        raise ExternalDatasetError(f"unknown requested file(s): {sorted(unknown)}")
    root = Path(destination).resolve()
    return [
        verify_file(
            root / name,
            known[name],
            archive_policy=manifest["archive_policy"],
        )
        for name in sorted(selected)
    ]


def immutable_inventory_payload(
    manifest: Mapping[str, Any],
    verified_files: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Create a deterministic local inventory without timestamps or local paths."""

    records = [
        {
            "name": item["name"],
            "bytes": item["bytes"],
            "md5": item["md5"],
            "sha256": item["sha256"],
            "format_audit": item["format_audit"],
        }
        for item in verified_files
    ]
    records.sort(key=lambda item: str(item["name"]))
    expected_names = sorted(str(item["name"]) for item in manifest["files"])
    if [str(item["name"]) for item in records] != expected_names:
        raise ExternalDatasetError(
            "immutable inventory requires all eight fixed record files"
        )
    return {
        "schema_version": 1,
        "record_id": manifest["record_id"],
        "record_revision": manifest["record_revision"],
        "doi": manifest["doi"],
        "license_spdx": manifest["license"]["spdx_id"],
        "source_inventory_sha256": manifest["inventory_sha256"],
        "scope": manifest["scope"],
        "files": records,
    }


def write_immutable_inventory(
    destination: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    """Write inventory.json atomically, refusing to mutate an existing record."""

    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "inventory.json"
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if target.exists():
        if target.read_text(encoding="utf-8") != rendered:
            raise ExternalDatasetError(
                "existing inventory.json differs; immutable inventory was not replaced"
            )
        return target
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            prefix=".inventory.",
            suffix=".part",
            dir=root,
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
        return target
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
