"""Command line interface for CSP5 prediction."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, List, Mapping, TypeVar


NUCLEI = ("13C", "1H")


def _parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CSP5 NMR shift predictor")
    parser.add_argument("--smiles", action="append", default=[], help="SMILES string (repeatable).")
    parser.add_argument("--smiles-file", type=Path, default=None, help="Text file with one SMILES per line.")
    parser.add_argument(
        "--structures-path",
        type=Path,
        default=None,
        help="Precomputed-geometry parquet dataset with columns: smiles, molblock.",
    )
    parser.add_argument(
        "--molecule-file",
        type=Path,
        default=None,
        help="Molfile or SDF input. Uses embedded coordinates unless --regenerate-geometry is set.",
    )
    parser.add_argument("--conformer-rank", type=int, default=0, help="Conformer rank filter for structures mode.")
    parser.add_argument(
        "--use-all-conformers",
        action="store_true",
        help="Use all conformers in structures mode instead of filtering by --conformer-rank.",
    )
    parser.add_argument("--structures-limit", type=int, default=0, help="Optional structures-mode molecule limit.")
    parser.add_argument("--nucleus", choices=["13C", "1H", "both"], default="13C")
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Explicit model name (for example CSP5-1H-dmso-d6).",
    )
    parser.add_argument(
        "--solvent",
        type=str,
        default=None,
        help="Solvent key for clean solvent-specific checkpoint selection.",
    )
    parser.add_argument("--device", default="auto", help="Torch device: auto/cuda/mps/cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-conformers", type=int, default=1)
    parser.add_argument(
        "--boltzmann-temperature-k",
        type=float,
        default=298.15,
        help="Boltzmann averaging temperature in K for multi-conformer SMILES or structures runs.",
    )
    parser.add_argument("--max-embed-tries", type=int, default=20)
    parser.add_argument("--ff-max-iters", type=int, default=200)
    parser.add_argument("--prune-rms-thresh", type=float, default=0.0)
    parser.add_argument(
        "--regenerate-geometry",
        action="store_true",
        help="For --molecule-file, discard embedded coordinates and generate new geometry while preserving atom order.",
    )
    parser.add_argument("--adaptive-conformers", action="store_true")
    parser.add_argument("--skip-heavy-atoms-gt", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build prediction rows without loading model weights or running inference.",
    )
    parser.add_argument(
        "--no-status",
        action="store_true",
        help="Disable human-readable status lines on stderr.",
    )
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--output-conformers-json",
        type=Path,
        default=None,
        help="Optional JSON output path for individual conformer predictions.",
    )
    parser.add_argument(
        "--output-conformers-sdf",
        type=Path,
        default=None,
        help="Optional SDF output path for exact conformer geometries used for prediction.",
    )
    parser.add_argument("--output-svg", type=Path, default=None, help="Optional annotated SVG drawing path.")
    parser.add_argument("--svg-width", type=int, default=None, help="Optional fixed SVG drawing width in pixels.")
    parser.add_argument("--svg-height", type=int, default=None, help="Optional fixed SVG drawing height in pixels.")
    parser.add_argument("--svg-bond-length", type=int, default=64, help="Target SVG bond length in pixels.")
    parser.add_argument("--svg-atom-font-size", type=int, default=12, help="SVG atom-label font size.")
    parser.add_argument(
        "--svg-shift-font-scale",
        dest="svg_shift_font_scale",
        type=float,
        default=None,
        help="SVG shift annotation font scale.",
    )
    parser.add_argument(
        "--svg-note-font-scale",
        dest="svg_note_font_scale",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--svg-padding", type=float, default=0.06, help="SVG drawing padding fraction.")
    parser.add_argument("--failures-tsv", type=Path, default=None, help="Optional failures TSV output path.")
    args = parser.parse_args(argv)

    has_smiles_input = bool(args.smiles) or args.smiles_file is not None
    has_structures_input = args.structures_path is not None
    has_molecule_file_input = args.molecule_file is not None
    input_count = sum([has_smiles_input, has_structures_input, has_molecule_file_input])
    if input_count > 1:
        raise ValueError("Provide only one input source: SMILES, --structures-path, or --molecule-file.")
    if input_count == 0:
        raise ValueError("Provide SMILES input (--smiles/--smiles-file), --structures-path, or --molecule-file.")
    if args.regenerate_geometry and not has_molecule_file_input:
        raise ValueError("--regenerate-geometry can only be used with --molecule-file.")
    if has_molecule_file_input and int(args.num_conformers) != 1:
        raise ValueError("--num-conformers is only supported for SMILES input, not --molecule-file.")
    if args.model_name and args.solvent:
        raise ValueError("Use either --model-name or --solvent, not both.")
    if args.nucleus == "both" and args.model_name:
        raise ValueError("--model-name can only be used with a single nucleus.")
    if args.nucleus == "both" and args.output_svg is not None:
        raise ValueError("--output-svg requires a single nucleus; use --nucleus 13C or --nucleus 1H.")
    if float(args.boltzmann_temperature_k) <= 0.0:
        raise ValueError("--boltzmann-temperature-k must be positive.")
    if args.svg_shift_font_scale is None and args.svg_note_font_scale is None:
        args.svg_shift_font_scale = 1.00
    elif args.svg_shift_font_scale is None:
        args.svg_shift_font_scale = args.svg_note_font_scale
    elif args.svg_note_font_scale is not None and args.svg_shift_font_scale != args.svg_note_font_scale:
        raise ValueError("Use either --svg-shift-font-scale or --svg-note-font-scale, not conflicting values for both.")
    if (args.svg_width is None) != (args.svg_height is None):
        raise ValueError("Provide both --svg-width and --svg-height, or omit both to auto-size the SVG canvas.")
    return args


def _collect_smiles(args: argparse.Namespace) -> List[str]:
    smiles = [str(s).strip() for s in args.smiles if str(s).strip()]
    if args.smiles_file is not None:
        if not args.smiles_file.exists():
            raise FileNotFoundError(f"SMILES file not found: {args.smiles_file}")
        with args.smiles_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                value = line.strip()
                if value:
                    smiles.append(value)
    if not smiles:
        raise ValueError("No non-empty SMILES found in input.")
    return smiles


def _status_enabled(args: argparse.Namespace) -> bool:
    if bool(args.no_status):
        return False
    return bool(sys.stderr.isatty())


def _status(message: str, *, enabled: bool) -> None:
    if not enabled:
        return
    print(f"[csp5] {message}", file=sys.stderr, flush=True)


T = TypeVar("T")


def _run_with_heartbeat(work: Callable[[], T], *, enabled: bool, interval_s: float = 10.0) -> T:
    if not enabled:
        return work()

    stop_event = threading.Event()
    started_at = time.perf_counter()
    warned_slow = False
    slow_warning_after_s = 8.0
    next_heartbeat_at_s = float(interval_s)

    def _heartbeat() -> None:
        nonlocal warned_slow, next_heartbeat_at_s
        while not stop_event.wait(1.0):
            elapsed = time.perf_counter() - started_at
            if (not warned_slow) and elapsed >= slow_warning_after_s:
                _status(
                    "This run is taking a while. First invocation can be slow while dependencies and model weights initialize.",
                    enabled=True,
                )
                warned_slow = True
            if elapsed >= next_heartbeat_at_s:
                _status(f"Still working... {elapsed:.0f}s elapsed.", enabled=True)
                next_heartbeat_at_s += float(interval_s)

    thread = threading.Thread(target=_heartbeat, name="csp5-heartbeat", daemon=True)
    thread.start()
    try:
        return work()
    finally:
        stop_event.set()
        thread.join(timeout=0.2)


def _molecule_identity(molecule: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return (
        molecule.get("molecule_id"),
        molecule.get("smiles"),
        molecule.get("mapped_smiles_explicit_h"),
    )


def _combine_conformers_by_nucleus(molecules_by_nucleus: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    conformers_by_nucleus = {
        nucleus: molecule.get("conformers", [])
        for nucleus, molecule in molecules_by_nucleus.items()
    }
    if not any(conformers_by_nucleus.values()):
        return []
    expected_len = len(next(iter(conformers_by_nucleus.values())))
    if any(len(rows) != expected_len for rows in conformers_by_nucleus.values()):
        raise ValueError("Cannot combine nuclei with mismatched conformer counts")

    combined: list[dict[str, Any]] = []
    metadata_keys = ("conformer_rank", "conformer_id", "conformer_energy", "conformer_energy_method")
    for idx in range(expected_len):
        first = next(iter(conformers_by_nucleus.values()))[idx]
        item = {key: first[key] for key in metadata_keys if key in first}
        item["predictions"] = {}
        for nucleus, conformers in conformers_by_nucleus.items():
            conformer = conformers[idx]
            for key in metadata_keys:
                if key in item and conformer.get(key) != item[key]:
                    raise ValueError("Cannot combine nuclei with mismatched conformer metadata")
            item["predictions"][nucleus] = conformer.get("predictions", [])
        combined.append(item)
    return combined


def _combine_nucleus_payloads(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(payloads) != set(NUCLEI):
        raise ValueError("Combined output requires both 13C and 1H payloads")

    molecule_counts = {nucleus: len(payload.get("molecules", [])) for nucleus, payload in payloads.items()}
    if len(set(molecule_counts.values())) != 1:
        raise ValueError(f"Cannot combine nuclei with mismatched molecule counts: {molecule_counts}")

    combined_molecules: list[dict[str, Any]] = []
    for idx in range(next(iter(molecule_counts.values()))):
        molecules_by_nucleus = {
            nucleus: payload["molecules"][idx]
            for nucleus, payload in payloads.items()
        }
        identities = {nucleus: _molecule_identity(molecule) for nucleus, molecule in molecules_by_nucleus.items()}
        if len(set(identities.values())) != 1:
            raise ValueError(f"Cannot combine nuclei with mismatched molecule identities: {identities}")

        first = next(iter(molecules_by_nucleus.values()))
        molecule_payload: dict[str, Any] = {
            "molecule_id": first["molecule_id"],
            "smiles": first["smiles"],
            "mapped_smiles_explicit_h": first["mapped_smiles_explicit_h"],
            "predictions": {
                nucleus: molecule.get("predictions", [])
                for nucleus, molecule in molecules_by_nucleus.items()
            },
        }
        for key, value in first.items():
            if key not in {"molecule_id", "smiles", "mapped_smiles_explicit_h", "predictions", "conformers"}:
                molecule_payload[key] = value
        conformers = _combine_conformers_by_nucleus(molecules_by_nucleus)
        if conformers:
            molecule_payload["conformers"] = conformers
        combined_molecules.append(molecule_payload)

    return {
        "models": {
            nucleus: payload["model"]
            for nucleus, payload in payloads.items()
        },
        "molecules": combined_molecules,
        "failures": {
            nucleus: list(payload.get("failures", []))
            for nucleus, payload in payloads.items()
        },
    }


def _combine_nucleus_conformer_payloads(payloads: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if set(payloads) != set(NUCLEI):
        raise ValueError("Combined conformer output requires both 13C and 1H payloads")

    molecule_counts = {nucleus: len(payload.get("molecules", [])) for nucleus, payload in payloads.items()}
    if len(set(molecule_counts.values())) != 1:
        raise ValueError(f"Cannot combine conformer nuclei with mismatched molecule counts: {molecule_counts}")

    combined_molecules: list[dict[str, Any]] = []
    for idx in range(next(iter(molecule_counts.values()))):
        molecules_by_nucleus = {
            nucleus: payload["molecules"][idx]
            for nucleus, payload in payloads.items()
        }
        identities = {nucleus: _molecule_identity(molecule) for nucleus, molecule in molecules_by_nucleus.items()}
        if len(set(identities.values())) != 1:
            raise ValueError(f"Cannot combine conformer nuclei with mismatched molecule identities: {identities}")

        first = next(iter(molecules_by_nucleus.values()))
        molecule_payload: dict[str, Any] = {
            "molecule_id": first["molecule_id"],
            "smiles": first["smiles"],
            "mapped_smiles_explicit_h": first["mapped_smiles_explicit_h"],
        }
        for key, value in first.items():
            if key not in {"molecule_id", "smiles", "mapped_smiles_explicit_h", "conformers"}:
                molecule_payload[key] = value
        molecule_payload["conformers"] = _combine_conformers_by_nucleus(molecules_by_nucleus)
        combined_molecules.append(molecule_payload)

    return {
        "models": {
            nucleus: payload["model"]
            for nucleus, payload in payloads.items()
        },
        "molecules": combined_molecules,
        "failures": {
            nucleus: list(payload.get("failures", []))
            for nucleus, payload in payloads.items()
        },
    }


def main(argv: List[str] | None = None) -> int:
    ns = _parse_args(sys.argv[1:] if argv is None else argv)
    show_status = _status_enabled(ns)
    started_at = time.perf_counter()
    nuclei = list(NUCLEI) if ns.nucleus == "both" else [ns.nucleus]

    if ns.structures_path is not None:
        _status(
            "Starting structures prediction.",
            enabled=show_status,
        )
        if not ns.structures_path.exists():
            raise FileNotFoundError(f"structures-path not found: {ns.structures_path}")

        def _work_structures():
            from .api import predict_structures

            return {
                nucleus: predict_structures(
                    ns.structures_path,
                    nucleus=nucleus,
                    model_name=ns.model_name,
                    solvent=ns.solvent,
                    device=ns.device,
                    batch_size=ns.batch_size,
                    conformer_rank=ns.conformer_rank,
                    use_all_conformers=ns.use_all_conformers,
                    limit=ns.structures_limit,
                    boltzmann_temperature_k=ns.boltzmann_temperature_k,
                    dry_run=ns.dry_run,
                )
                for nucleus in nuclei
            }

        results = _run_with_heartbeat(
            _work_structures,
            enabled=show_status,
        )
    elif ns.molecule_file is not None:
        if not ns.molecule_file.exists():
            raise FileNotFoundError(f"molecule-file not found: {ns.molecule_file}")
        suffix = ns.molecule_file.suffix.lower()
        if suffix not in {".mol", ".sdf", ".sd"}:
            raise ValueError("--molecule-file expects a .mol, .sdf, or .sd file.")
        if ns.regenerate_geometry:
            _status(
                f"Starting molecule-file prediction; regenerating geometry from {ns.molecule_file} while preserving input atom order.",
                enabled=show_status,
            )
        else:
            _status(
                f"Starting molecule-file prediction; using embedded coordinates from {ns.molecule_file}.",
                enabled=show_status,
            )

        def _work_molecule_file():
            from .api import predict_molecule_file

            return {
                nucleus: predict_molecule_file(
                    ns.molecule_file,
                    nucleus=nucleus,
                    model_name=ns.model_name,
                    solvent=ns.solvent,
                    device=ns.device,
                    batch_size=ns.batch_size,
                    regenerate_geometry=ns.regenerate_geometry,
                    max_embed_tries=ns.max_embed_tries,
                    prune_rms_thresh=ns.prune_rms_thresh,
                    ff_max_iters=ns.ff_max_iters,
                    dry_run=ns.dry_run,
                )
                for nucleus in nuclei
            }

        results = _run_with_heartbeat(
            _work_molecule_file,
            enabled=show_status,
        )
    else:
        _status(
            "Starting SMILES prediction.",
            enabled=show_status,
        )
        smiles = _collect_smiles(ns)

        def _work_smiles():
            from .api import predict_smiles

            return {
                nucleus: predict_smiles(
                    smiles,
                    nucleus=nucleus,
                    model_name=ns.model_name,
                    solvent=ns.solvent,
                    device=ns.device,
                    batch_size=ns.batch_size,
                    max_embed_tries=ns.max_embed_tries,
                    num_conformers=ns.num_conformers,
                    boltzmann_temperature_k=ns.boltzmann_temperature_k,
                    prune_rms_thresh=ns.prune_rms_thresh,
                    ff_max_iters=ns.ff_max_iters,
                    adaptive_conformers=ns.adaptive_conformers,
                    skip_heavy_atoms_gt=ns.skip_heavy_atoms_gt,
                    num_workers=ns.num_workers,
                    dry_run=ns.dry_run,
                )
                for nucleus in nuclei
            }

        results = _run_with_heartbeat(
            _work_smiles,
            enabled=show_status,
        )
    elapsed = time.perf_counter() - started_at
    total_predictions = sum(len(result.predictions) for result in results.values())
    total_failures = sum(len(result.failures) for result in results.values())
    _status(
        f"Done in {elapsed:.1f}s ({total_predictions} predictions, {total_failures} failures).",
        enabled=show_status,
    )

    payloads = {
        nucleus: result.to_compact_mapped_payload()
        for nucleus, result in results.items()
    }
    payload = _combine_nucleus_payloads(payloads) if ns.nucleus == "both" else next(iter(payloads.values()))
    conformer_payloads = {
        nucleus: result.to_conformer_mapped_payload()
        for nucleus, result in results.items()
    }
    conformer_payload = (
        _combine_nucleus_conformer_payloads(conformer_payloads)
        if ns.nucleus == "both"
        else next(iter(conformer_payloads.values()))
    )

    if ns.output_json is not None:
        ns.output_json.parent.mkdir(parents=True, exist_ok=True)
        ns.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))

    if ns.output_conformers_json is not None:
        ns.output_conformers_json.parent.mkdir(parents=True, exist_ok=True)
        ns.output_conformers_json.write_text(json.dumps(conformer_payload, indent=2), encoding="utf-8")

    if ns.output_conformers_sdf is not None:
        sdf_result = next(
            (
                result
                for result in results.values()
                if result.conformer_mol_records or result.molecule_records
            ),
            next(iter(results.values())),
        )
        sdf_result.write_conformers_sdf(ns.output_conformers_sdf)

    if ns.output_svg is not None:
        from .drawing import write_compact_prediction_svg

        write_compact_prediction_svg(
            payload,
            ns.output_svg,
            width=ns.svg_width,
            height=ns.svg_height,
            bond_length=ns.svg_bond_length,
            atom_font_size=ns.svg_atom_font_size,
            shift_font_scale=ns.svg_shift_font_scale,
            padding=ns.svg_padding,
        )

    if ns.failures_tsv is not None:
        ns.failures_tsv.parent.mkdir(parents=True, exist_ok=True)
        with ns.failures_tsv.open("w", encoding="utf-8") as handle:
            for nucleus, result in results.items():
                for row in result.failures:
                    if ns.nucleus == "both":
                        handle.write(f"{nucleus}\t{row}\n")
                    else:
                        handle.write(f"{row}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
