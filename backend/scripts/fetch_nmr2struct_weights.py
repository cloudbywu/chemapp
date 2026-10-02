"""Fetch and verify the NMR2Struct model weights for clean builds.

The checkpoint files (backend/vendor/nmr2struct/checkpoints/) are excluded
from git (*.pt / *.ckpt).  This script restores them from the pinned GitHub
release ``assets-nmr2struct-v1`` and verifies every file against the SHA-256
values recorded at upload time.  NMR2Struct is MIT-licensed (see NOTICE).

Usage:
    python scripts/fetch_nmr2struct_weights.py              # all four files
    python scripts/fetch_nmr2struct_weights.py --variant cnmr_only
    python scripts/fetch_nmr2struct_weights.py --check      # verify only
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen

RELEASE_BASE = (
    "https://github.com/cloudbywu/chemapp/releases/download/assets-nmr2struct-v1"
)

FILES = {
    "cnmr_only_checkpoint.pt": (
        "1c14362a0b24c951641b14341c075457d9f1a2020ae0eabc0db2f51d358e3d40"
    ),
    "hnmr_only_checkpoint.pt": (
        "4a9a0c6e114e296bd06b8fb3c722b7628f66c0c21109dc71b62db2ddfea121cf"
    ),
    "multitask_checkpoint.pt": (
        "da28998d7993eded4d7298969a61a8b4952191ca7fdcf63b3f8182ae81bdf630"
    ),
    "transformer_checkpoint.ckpt": (
        "0204992f44f9934e7a276f9fcc2b14f61fac3aead3a14954291ef3df8d677469"
    ),
}

TARGET = Path(__file__).resolve().parents[1] / "vendor" / "nmr2struct" / "checkpoints"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: Path) -> None:
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        tmp_path = Path(tmp) / dest.name
        with urlopen(url, timeout=600) as response, tmp_path.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        shutil.move(str(tmp_path), dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        default="all",
        help="fetch only one checkpoint: all | cnmr_only | hnmr_only | multitask "
        "| transformer (full file names also accepted; default: all)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify hashes of files already present, download nothing",
    )
    args = parser.parse_args()

    alias = {
        "cnmr_only": "cnmr_only_checkpoint.pt",
        "hnmr_only": "hnmr_only_checkpoint.pt",
        "multitask": "multitask_checkpoint.pt",
        "transformer": "transformer_checkpoint.ckpt",
    }
    selected = alias.get(args.variant, args.variant)
    if selected != "all" and selected not in FILES:
        parser.error(f"unknown variant: {args.variant}")
    names = list(FILES) if selected == "all" else [selected]
    TARGET.mkdir(parents=True, exist_ok=True)
    problems: list[str] = []
    for name in names:
        path = TARGET / name
        expected = FILES[name]
        if path.is_file() and _sha256_file(path) == expected:
            print(f"ok: {name} already present and verified")
            continue
        if args.check:
            problems.append(f"missing or mismatch: {name}")
            continue
        url = f"{RELEASE_BASE}/{name}"
        print(f"downloading {url}")
        _download(url, path)
        actual = _sha256_file(path)
        if actual != expected:
            path.unlink(missing_ok=True)
            problems.append(f"hash mismatch after download: {name}")
        else:
            print(f"verified: {name}")
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
