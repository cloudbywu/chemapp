"""Fetch and verify the vendored CSP5 model weights for clean builds.

The ``*.pt`` files are intentionally excluded from git (about 73 MB).  This
script restores them from the pinned PyPI source distribution and verifies
every file against ``weights-manifest.json`` before they can be used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path
from typing import Sequence
from urllib.request import urlopen

PYPI_SDIST_URL = (
    "https://pypi.org/packages/source/c/csp5/csp5-0.2.18.tar.gz"
)
PYPI_SDIST_SHA256 = (
    "5e8ca5f1c4c7bcb6146ffaa6d9a4cab5212e29aaa753ea4875096cd92e92444a"
)
SDIST_TOP = "csp5-0.2.18"
MODELS_REL = "src/csp5/models"
MANIFEST_NAME = "weights-manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(target: Path) -> dict:
    path = target / MANIFEST_NAME
    return json.loads(path.read_text(encoding="utf-8"))


def verify_weights(target: Path, manifest: dict) -> list[str]:
    problems: list[str] = []
    for entry in manifest.get("files", []):
        rel = Path(str(entry.get("path") or "").replace("\\", "/"))
        path = target / rel
        if not path.is_file():
            problems.append(f"missing: {rel.as_posix()}")
            continue
        actual = _sha256_file(path)
        expected = str(entry.get("sha256") or "")
        if actual != expected:
            problems.append(
                f"hash mismatch: {rel.as_posix()} "
                f"expected {expected}, got {actual}"
            )
    return problems


def _extract_sdist(sdist: Path, target: Path, manifest: dict) -> None:
    prefix = f"{SDIST_TOP}/{MODELS_REL}/"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        with tarfile.open(sdist, "r:gz") as archive:
            members = [
                member
                for member in archive.getmembers()
                if member.name.startswith(prefix) and member.isfile()
            ]
            archive.extractall(tmp_root, members=members)
        src_models = tmp_root / SDIST_TOP / MODELS_REL
        for entry in manifest.get("files", []):
            rel = Path(str(entry.get("path") or "").replace("\\", "/"))
            source_rel = (
                Path(*rel.parts[1:])
                if rel.parts and rel.parts[0] == "models"
                else rel
            )
            source = src_models / source_rel
            if not source.is_file():
                raise RuntimeError(
                    f"source distribution is missing {rel.as_posix()}"
                )
            destination = target / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        for model_dir in (
            "CSP5-13C",
            "CSP5-1H",
            "CSP5q-13C",
            "CSP5q-1H",
        ):
            metadata = src_models / model_dir / "model_metadata.json"
            if metadata.is_file():
                shutil.copy2(
                    metadata,
                    target / "models" / model_dir / "model_metadata.json",
                )


def fetch(
    target: Path,
    *,
    url: str = PYPI_SDIST_URL,
    sdist_sha256: str = PYPI_SDIST_SHA256,
    force: bool = False,
) -> None:
    target.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(target)
    problems = verify_weights(target, manifest)
    if not problems and not force:
        print("CSP5 weights already present and verified; nothing to do.")
        return
    if problems and not force:
        print(
            "CSP5 weights missing or invalid; restoring from pinned PyPI "
            "source distribution."
        )
    with tempfile.TemporaryDirectory() as tmp:
        archive_path = Path(tmp) / "csp5-0.2.18.tar.gz"
        print(f"Downloading {url}")
        with urlopen(url, timeout=120) as response:
            with archive_path.open("wb") as out:
                shutil.copyfileobj(response, out)
        actual = _sha256_file(archive_path)
        if actual != sdist_sha256:
            raise RuntimeError(
                f"pinned source distribution hash mismatch: "
                f"expected {sdist_sha256}, got {actual}"
            )
        _extract_sdist(archive_path, target, manifest)
    remaining = verify_weights(target, manifest)
    if remaining:
        raise RuntimeError(
            "CSP5 weights verification failed after extraction:\n"
            + "\n".join(remaining)
        )
    print(
        "CSP5 weights verified: "
        f"{len(manifest.get('files', []))} files match weights-manifest.json"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Restore and verify CSP5 model weights from PyPI."
    )
    parser.add_argument(
        "--target",
        default=str(
            Path(__file__).resolve().parents[1] / "vendor" / "csp5"
        ),
        help="Path to the vendored csp5 directory containing weights-manifest.json.",
    )
    parser.add_argument("--url", default=PYPI_SDIST_URL)
    parser.add_argument("--sha256", default=PYPI_SDIST_SHA256)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even when weights already verify.",
    )
    args = parser.parse_args(argv)
    fetch(
        Path(args.target).resolve(),
        url=args.url,
        sdist_sha256=args.sha256,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
