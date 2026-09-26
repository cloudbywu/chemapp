"""CLI for shift matching utilities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

from .matching import match_shifts


def _read_float_lines(path: Path) -> List[float]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    values: List[float] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = line.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError(f"No numeric values found in {path}")
    return values


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSP5 shift matching")
    parser.add_argument("--predicted-file", type=Path, required=True, help="One predicted shift per line.")
    parser.add_argument("--experimental-file", type=Path, required=True, help="One experimental shift per line.")
    parser.add_argument("--solver", choices=["dp", "scipy", "murty"], default="dp")
    parser.add_argument("--k-best", type=int, default=1)
    parser.add_argument(
        "--k-best-policy",
        choices=["strict", "clip"],
        default="clip",
        help=(
            "strict: fail if k_best exceeds available unique Murty assignments; "
            "clip: return all available assignments instead."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--mae-delta-threshold", type=float, default=0.2)
    parser.add_argument("--dummy-cost", type=float, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    ns = _parse_args(sys.argv[1:] if argv is None else argv)
    predicted = _read_float_lines(ns.predicted_file)
    experimental = _read_float_lines(ns.experimental_file)

    result = match_shifts(
        predicted,
        experimental,
        solver=ns.solver,
        k_best=ns.k_best,
        k_best_policy=ns.k_best_policy,
        temperature=ns.temperature,
        mae_delta_threshold=ns.mae_delta_threshold,
        dummy_cost=ns.dummy_cost,
    )
    payload = {
        "solver": result.solver,
        "k_best_requested": int(ns.k_best),
        "k_best_effective": int(len(result.ranked_assignments)),
        "k_best_policy": str(ns.k_best_policy),
        "dummy_cost": result.dummy_cost,
        "best_assignment": result.best_assignment,
        "best_total_cost": result.best_total_cost,
        "best_mean_abs_error": result.best_mean_abs_error,
        "assignment_entropy": result.assignment_entropy,
        "num_competing_assignments": result.num_competing_assignments,
        "matching_count": result.matching_count,
        "ranked_assignments": [
            {
                "rank": item.rank,
                "assignment": item.assignment,
                "total_cost": item.total_cost,
                "mean_abs_error": item.mean_abs_error,
            }
            for item in result.ranked_assignments
        ],
    }
    if ns.output_json is not None:
        ns.output_json.parent.mkdir(parents=True, exist_ok=True)
        ns.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
