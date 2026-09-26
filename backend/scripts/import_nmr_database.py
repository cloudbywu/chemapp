from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.nmr_structure_elucidation import import_nmrshiftdb_sd, import_retrieval_db, index_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Import NMR spectral databases into ChemApp index.")
    parser.add_argument(
        "--data-dir",
        default="",
        help="External NMR data directory; defaults to repo backend/data.",
    )
    parser.add_argument("--index", default=str(Path("data") / "nmr_spectral_index.sqlite"))
    parser.add_argument("--source", choices=["retrieval", "nmrshiftdb2"], default="nmrshiftdb2")
    parser.add_argument("--path", default="")
    parser.add_argument("--limit", type=int, default=-1, help="-1 means full import")
    parser.add_argument("--time-budget-s", type=float, default=7200)
    parser.add_argument("--progress-every", type=int, default=1000)
    parser.add_argument("--commit-every", type=int, default=1000)
    parser.add_argument("--show-rdkit-warnings", action="store_true")
    args = parser.parse_args()

    if args.data_dir:
        os.environ["CHEMAPP_NMR_DATA_DIR"] = args.data_dir
    os.environ["CHEMAPP_NMR_INDEX"] = args.index
    limit = None if args.limit < 0 else args.limit
    path = args.path or None

    if args.source == "retrieval":
        result = import_retrieval_db(path=path, limit=limit)
    else:
        result = import_nmrshiftdb_sd(
            path=path,
            limit=limit,
            time_budget_s=args.time_budget_s,
            progress_every=args.progress_every,
            commit_every=args.commit_every,
            quiet_rdkit=not args.show_rdkit_warnings,
        )
    print("IMPORT_RESULT", result, flush=True)
    print("INDEX_STATUS", index_status(), flush=True)


if __name__ == "__main__":
    main()
