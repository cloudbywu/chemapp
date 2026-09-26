from __future__ import annotations

import logging
import os
import stat
import tempfile
import zipfile
from functools import partial
from pathlib import Path

import anyio
from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.deps import get_store
from app.parsers import ParserRegistry

router = APIRouter(prefix="/api/upload", tags=["upload"])
logger = logging.getLogger(__name__)

_UPLOAD_CHUNK_SIZE = 1024 * 1024
_MAX_UPLOAD_BYTES = int(os.environ.get("CHEMAPP_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))
_MAX_ZIP_FILES = int(os.environ.get("CHEMAPP_MAX_ZIP_FILES", "2000"))
_MAX_ZIP_UNCOMPRESSED_BYTES = int(
    os.environ.get("CHEMAPP_MAX_ZIP_UNCOMPRESSED_BYTES", str(200 * 1024 * 1024))
)
_MAX_ZIP_COMPRESSION_RATIO = float(os.environ.get("CHEMAPP_MAX_ZIP_COMPRESSION_RATIO", "200"))
_MAX_ZIP_MEMBER_NAME = int(os.environ.get("CHEMAPP_MAX_ZIP_MEMBER_NAME", "512"))


async def _stream_upload_to_file(file: UploadFile, suffix: str) -> Path:
    """Stream an upload to a temporary file, enforcing the size limit."""

    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = Path(handle.name)
    handle.close()
    total = 0
    try:
        with path.open("wb") as destination:
            while True:
                chunk = await file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        413,
                        f"Uploaded file exceeds the "
                        f"{_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit",
                    )
                destination.write(chunk)
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return path


def _safe_extract_zip(zf: zipfile.ZipFile, extract_dir: Path) -> None:
    base_dir = extract_dir.resolve()
    infos = zf.infolist()
    if len(infos) > _MAX_ZIP_FILES:
        raise HTTPException(400, f"ZIP contains too many files; limit is {_MAX_ZIP_FILES}")

    total_size = 0
    normalized_names: set[str] = set()
    targets: list[tuple[zipfile.ZipInfo, Path]] = []
    for info in infos:
        filename = info.filename.replace("\\", "/")
        if not filename or len(filename) > _MAX_ZIP_MEMBER_NAME:
            raise HTTPException(400, "ZIP contains an empty or excessively long member name")
        if filename.startswith("/") or ".." in Path(filename).parts:
            raise HTTPException(400, f"Unsafe path in ZIP: {info.filename}")
        if any(":" in part for part in Path(filename).parts):
            raise HTTPException(400, f"Windows alternate-stream paths are not allowed: {info.filename}")
        normalized = filename.rstrip("/").casefold()
        if normalized in normalized_names:
            raise HTTPException(400, f"Duplicate path in ZIP: {info.filename}")
        normalized_names.add(normalized)
        if info.flag_bits & 0x1:
            raise HTTPException(400, f"Encrypted ZIP members are not supported: {info.filename}")

        mode = (info.external_attr >> 16) & 0o170000
        if mode not in {0, stat.S_IFREG, stat.S_IFDIR}:
            raise HTTPException(400, f"Special files are not allowed in ZIP files: {info.filename}")

        total_size += info.file_size
        if total_size > _MAX_ZIP_UNCOMPRESSED_BYTES:
            raise HTTPException(
                413,
                "ZIP uncompressed size exceeds "
                f"{_MAX_ZIP_UNCOMPRESSED_BYTES // (1024 * 1024)} MB",
            )
        if info.file_size and (
            info.compress_size <= 0
            or info.file_size / max(info.compress_size, 1) > _MAX_ZIP_COMPRESSION_RATIO
        ):
            raise HTTPException(413, f"Suspicious ZIP compression ratio: {info.filename}")

        target = (base_dir / filename).resolve()
        if target != base_dir and base_dir not in target.parents:
            raise HTTPException(400, f"Unsafe path in ZIP: {info.filename}")
        targets.append((info, target))

    # Extract explicitly and enforce the declared total again while streaming.
    extracted_size = 0
    for info, target in targets:
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        member_size = 0
        with zf.open(info, "r") as source, target.open("xb") as destination:
            while True:
                chunk = source.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                member_size += len(chunk)
                extracted_size += len(chunk)
                if member_size > info.file_size or extracted_size > _MAX_ZIP_UNCOMPRESSED_BYTES:
                    raise HTTPException(413, "ZIP expanded beyond its declared safe size")
                destination.write(chunk)


def _find_nmr_dir(extract_dir: Path) -> Path | None:
    if (extract_dir / "acqu").exists() and (extract_dir / "fid").exists():
        return extract_dir
    if (extract_dir / "acqu").exists():
        return extract_dir
    for root, dirs, _ in os.walk(str(extract_dir)):
        for d in dirs:
            candidate = Path(root) / d
            if (candidate / "acqu").exists() and (candidate / "fid").exists():
                return candidate
    for root, dirs, _ in os.walk(str(extract_dir)):
        for d in dirs:
            candidate = Path(root) / d
            if (candidate / "acqu").exists():
                return candidate
    return None


def _find_hplc_bundle_files(extract_dir: Path) -> list[Path]:
    """Find HPLC chromatograms whose sibling result files were preserved."""

    candidates: list[Path] = []
    for path in extract_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".dx":
            continue
        siblings = {
            (item.stem.casefold(), item.suffix.lower())
            for item in path.parent.iterdir()
            if item.is_file()
        }
        key = path.stem.casefold()
        if (key, ".rx") in siblings or (key, ".acaml") in siblings:
            candidates.append(path)
    return sorted(candidates, key=lambda item: str(item).casefold())


def _parse_and_store(file_path: str | Path, source_name: str | None = None) -> list[dict]:
    store = get_store()
    technique_name = ParserRegistry.detect(str(file_path))
    if technique_name is None:
        raise HTTPException(400, f"Unrecognized file format: {Path(file_path).name}")

    parser_cls = ParserRegistry.get(technique_name)
    parser = parser_cls()
    result = parser.parse(str(file_path))

    if isinstance(result, list):
        results = []
        for spectrum in result:
            if source_name:
                spectrum.source_file = source_name
            stored = store.add(spectrum)
            results.append({
                "id": stored.id,
                "technique": spectrum.technique.value,
                "points": spectrum.num_points,
                "summary": repr(spectrum),
                "spectrum_revision": stored.spectrum_revision,
                "result_revision": stored.result_revision,
            })
        return results

    if source_name:
        result.source_file = source_name
    stored = store.add(result)
    return [{
        "id": stored.id,
        "technique": result.technique.value,
        "points": result.num_points,
        "summary": repr(result),
        "spectrum_revision": stored.spectrum_revision,
        "result_revision": stored.result_revision,
    }]


@router.post("")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "No filename provided")

    suffix = Path(file.filename).suffix.lower()
    tmp_path = await _stream_upload_to_file(file, suffix or ".bin")

    try:
        if suffix == ".zip":
            with tempfile.TemporaryDirectory() as extract_dir:
                try:
                    with zipfile.ZipFile(tmp_path) as zf:
                        await anyio.to_thread.run_sync(
                            _safe_extract_zip, zf, Path(extract_dir)
                        )
                except zipfile.BadZipFile as exc:
                    raise HTTPException(400, "Invalid ZIP file") from exc

                extract_path = Path(extract_dir)
                nmr_dir = _find_nmr_dir(extract_path)
                hplc_files = (
                    []
                    if nmr_dir is not None
                    else _find_hplc_bundle_files(extract_path)
                )
                if nmr_dir is None and not hplc_files:
                    # The message matches _find_nmr_dir's loose acceptance:
                    # a Bruker folder qualifies when it contains an acqu file
                    # (fid optional).
                    raise HTTPException(
                        400,
                        "No supported instrument bundle found in ZIP. Include either "
                        "a Bruker NMR directory containing at least an acqu file "
                        "(add the fid file when available) or HPLC .dx files together "
                        "with their .rx/.acaml sidecars.",
                    )
                try:
                    if nmr_dir is not None:
                        results = await anyio.to_thread.run_sync(
                            partial(
                                _parse_and_store,
                                nmr_dir,
                                source_name=file.filename,
                            )
                        )
                    else:
                        results = []
                        for hplc_file in hplc_files:
                            relative_name = hplc_file.relative_to(
                                extract_path
                            ).as_posix()
                            results.extend(
                                await anyio.to_thread.run_sync(
                                    partial(
                                        _parse_and_store,
                                        hplc_file,
                                        source_name=f"{file.filename}:{relative_name}",
                                    )
                                )
                            )
                    for r in results:
                        r["name"] = file.filename
                    return results[0] if len(results) == 1 else results
                except HTTPException:
                    raise
                except Exception as exc:
                    logger.exception("Instrument bundle parsing failed")
                    raise HTTPException(
                        422,
                        "The instrument bundle could not be parsed; verify its "
                        "format and required sidecar files.",
                    ) from exc

        if suffix == "":
            raise HTTPException(
                400,
                "Directory upload not supported. "
                "For Bruker NMR data, please ZIP the folder first and upload the .zip file.",
            )

        results = await anyio.to_thread.run_sync(
            partial(_parse_and_store, tmp_path, source_name=file.filename)
        )
        for r in results:
            r["name"] = file.filename
        return results[0] if len(results) == 1 else results
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Instrument file parsing failed")
        raise HTTPException(
            422,
            "The instrument file could not be parsed; verify its format and integrity.",
        ) from exc
    finally:
        tmp_path.unlink(missing_ok=True)
