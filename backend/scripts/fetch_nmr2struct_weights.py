"""Install supported NMR2Struct checkpoints directly from the official repository.

Uses the same pinned, atomic, checksum-verified downloader as Settings > Models.
Run from any working directory. Set CHEMAPP_NMR2STRUCT_WEIGHTS_DIR to override
per-user model storage. No weights, code, or telemetry are uploaded.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.nmr2struct_weights import ASSETS, DownloadBusy, DownloadFailure, manager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["all", *ASSETS], default="all")
    parser.add_argument("--check", action="store_true", help="verify installed files without downloading")
    args = parser.parse_args()
    selected = list(ASSETS) if args.variant == "all" else [args.variant]
    if args.check:
        inventory = manager.inventory()
        print(f"Storage: {inventory['storage_dir']}")
        results = [asset for asset in inventory["assets"] if asset["id"] in selected]
        for asset in results:
            print(f"{asset['id']}: {asset['status']}")
        return 0 if all(asset["status"] == "ready" for asset in results) else 1
    for asset_id in selected:
        try:
            print(f"Official source: {ASSETS[asset_id].url}", flush=True)
            job = manager.start(asset_id)
            while job["status"] in {"queued", "downloading", "verifying", "cancelling"}:
                time.sleep(0.25)
                job = manager.get(job["id"])
            print(f"{asset_id}: {job['status']}" + (f" ({job['error']})" if job["error"] else ""))
            if job["status"] != "completed":
                return 1
        except (DownloadBusy, DownloadFailure) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            if "job" in locals():
                manager.cancel(job["id"])
            return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
