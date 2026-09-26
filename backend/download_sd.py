"""Safely download and validate nmrshiftdb2 SD snapshots.

Downloads are written to a temporary file and become visible at the requested
destination only after HTTP, content, size, SD-marker, and optional SHA-256
validation all succeed.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import requests

from app.ml.nmr_data_v2 import SourceValidationError, validate_sd_source


FILES = {
    "nmrshiftdb2withsignals.sd": (
        158.6,
        "SD file with assigned NMR signals",
    ),
    "nmrshiftdb2.nmredata.sd": (
        284.4,
        "NMReDATA SD file with assignments",
    ),
    "nmrshiftdb2withsignals_3d.sd": (
        94.0,
        "3D SD file with signals",
    ),
    "nmrshiftdb2.sd": (
        138.5,
        "Basic SD file",
    ),
}
BASE_URL = "https://sourceforge.net/projects/nmrshiftdb2/files/data"
USER_AGENT = "ChemApp-NMR-data-import/2"


def download_file(
    filename: str,
    destination_directory: str | Path,
    *,
    expected_sha256: str | None = None,
    timeout_s: float = 300.0,
) -> dict:
    if filename not in FILES:
        raise ValueError(f"unsupported nmrshiftdb2 snapshot: {filename}")
    expected_mb, _description = FILES[filename]
    destination_directory = Path(destination_directory).resolve()
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / filename
    url = f"{BASE_URL}/{filename}/download"

    handle, temporary_name = tempfile.mkstemp(
        dir=destination_directory,
        prefix=f".{filename}.",
        suffix=".part",
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        with requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            stream=True,
            allow_redirects=True,
            timeout=(30.0, timeout_s),
        ) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").casefold()
            if "text/html" in content_type or "application/xhtml" in content_type:
                raise SourceValidationError(
                    f"server returned {content_type or 'HTML'} instead of SD data"
                )
            with temporary.open("wb") as stream:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        stream.write(chunk)
                stream.flush()
                os.fsync(stream.fileno())

        # The advertised sizes are approximate. Requiring at least half keeps
        # a mirror revision possible while rejecting short error responses.
        minimum_bytes = max(1024, int(expected_mb * 1_000_000 * 0.5))
        validated = validate_sd_source(
            temporary,
            expected_sha256=expected_sha256,
            min_bytes=minimum_bytes,
        )
        os.replace(temporary, destination)
        return {
            "status": "ok",
            "path": str(destination),
            "bytes": validated.byte_size,
            "sha256": validated.sha256,
            "source_url": url,
        }
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "filename",
        choices=sorted(FILES),
        help="Snapshot filename to download",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument(
        "--expected-sha256",
        help="Pinned upstream digest; strongly recommended for reproducible builds",
    )
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing validated destination",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    destination = args.destination.resolve() / args.filename
    if destination.exists() and not args.replace:
        validated = validate_sd_source(
            destination,
            expected_sha256=args.expected_sha256,
            min_bytes=1024,
        )
        print(
            {
                "status": "already_valid",
                "path": str(destination),
                "bytes": validated.byte_size,
                "sha256": validated.sha256,
            }
        )
        return
    result = download_file(
        args.filename,
        args.destination,
        expected_sha256=args.expected_sha256,
        timeout_s=args.timeout_s,
    )
    print(result)


if __name__ == "__main__":
    main()
