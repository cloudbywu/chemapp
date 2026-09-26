#!/usr/bin/env python
"""Predict CASCADE-2.0 13C shifts for NMRexp sc_carbon_le_1 (one conformer per molecule)."""
from __future__ import annotations

import argparse
import json
import os
import sys
import pickle
import re
import random
import math
import shutil
import multiprocessing as mp
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Lipinski, rdMolDescriptors
from rdkit.Chem.rdchem import Mol
from tqdm import tqdm


RDLogger.DisableLog("rdApp.*")
os.environ.setdefault("NFP_NO_KERAS", "1")

# CASCADE occasionally produces non-finite / absurd values for some out-of-domain
# molecules (e.g. salts with disconnected fragments). We treat these as failures
# rather than writing them to the predictions dataset.
SHIFT_PPM_MIN = -50.0
SHIFT_PPM_MAX = 400.0
LEGACY_OUTPUT_SCALE = 50.484337
LEGACY_OUTPUT_BIAS = 99.798111
Q90_ZSCORE = 1.2815515655446004
QUANTILE_99_LEVELS = tuple(float(i) / 100.0 for i in range(1, 100))
MODEL_METADATA_FILENAMES = (
    "model_metadata.json",
    "metadata.json",
)


def _quantile_column_name(level: float) -> str:
    return f"shift_q{int(round(float(level) * 100.0)):02d}_ppm"


def _repo_root() -> Path:
    root = Path(__file__).resolve()
    while root != root.parent and not (root / "src").exists():
        root = root.parent
    return root


REPO_ROOT = _repo_root()
CASCADE_MODEL_CODE_DIR = Path(__file__).resolve().parent / "Predict_SMILES_FF"


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict CASCADE-2.0 13C shifts for NMRexp sc_carbon_le_1 molecules."
    )
    parser.add_argument(
        "--input-parquet",
        type=Path,
        required=True,
        help="Input NMRexp parquet path.",
    )
    parser.add_argument(
        "--smiles-col",
        default="SMILES",
        help="SMILES column name in the input parquet (default: SMILES).",
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("results/cascade_predicted_shifts_13c.parquet"),
        help="Output parquet dataset path for CASCADE predictions.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=CASCADE_MODEL_CODE_DIR,
        help="CASCADE Predict_SMILES_FF model/artifact directory (default: bundled csp5/_runtime/Predict_SMILES_FF).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Path to CASCADE PyTorch weights (default: <model-dir>/best_model.pt).",
    )
    parser.add_argument(
        "--preprocessor",
        type=Path,
        default=None,
        help="Path to CASCADE preprocessor pickle (default: <model-dir>/preprocessor_orig.p).",
    )
    parser.add_argument(
        "--output-scale",
        type=float,
        default=None,
        help=(
            "Optional output scale to convert model outputs to ppm. "
            "If unset, resolve from model metadata with legacy fallback."
        ),
    )
    parser.add_argument(
        "--output-bias",
        type=float,
        default=None,
        help=(
            "Optional output bias to convert model outputs to ppm. "
            "If unset, resolve from model metadata with legacy fallback."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device (cuda/mps/cpu/auto). auto uses CUDA, then MPS, then CPU.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size for the CASCADE model (default: 32).",
    )
    parser.add_argument(
        "--smiles-batch-size",
        type=int,
        default=2000,
        help="Number of SMILES to embed/predict per chunk (default: 2000).",
    )
    parser.add_argument(
        "--max-embed-tries",
        type=int,
        default=20,
        help="Maximum embedding attempts per molecule (default: 20).",
    )
    parser.add_argument(
        "--num-conformers",
        type=int,
        default=1,
        help="Number of conformers to generate per molecule (default: 1).",
    )
    parser.add_argument(
        "--prune-rms-thresh",
        type=float,
        default=0.0,
        help="Prune RMS threshold for conformer embedding (default: 0.0 = no pruning).",
    )
    parser.add_argument(
        "--ff-max-iters",
        type=int,
        default=200,
        help="Maximum force-field optimization iterations per conformer (default: 200).",
    )
    parser.add_argument(
        "--adaptive-conformers",
        action="store_true",
        help=(
            "Adapt the number of conformers per molecule based on rotatable bonds/size "
            "(saves time on rigid/very large molecules)."
        ),
    )
    parser.add_argument(
        "--skip-heavy-atoms-gt",
        type=int,
        default=0,
        help="Skip molecules with more than this many heavy atoms (0 = disable).",
    )
    parser.add_argument(
        "--min-conformer-rank",
        type=int,
        default=0,
        help="When writing structures, only keep conformers with rank >= this (default: 0).",
    )
    parser.add_argument(
        "--embed-stall-timeout-s",
        type=float,
        default=120.0,
        help=(
            "If using multiprocessing embedding, abort the current SMILES batch if no "
            "conformer result is produced for this many seconds (0 = disable)."
        ),
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Multiprocessing workers for embedding (0 = auto, 1 = disable).",
    )
    parser.add_argument(
        "--mp-context",
        default="spawn",
        choices=["spawn", "fork", "forkserver"],
        help="Multiprocessing start method for embedding (default: spawn).",
    )
    parser.add_argument(
        "--mp-chunksize",
        type=int,
        default=20,
        help="Multiprocessing chunk size for embedding tasks (default: 20).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of unique SMILES for testing (0 = no limit).",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Total number of shards for distributed runs (default: 1).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Shard index (0-based) for distributed runs (default: 0).",
    )
    parser.add_argument(
        "--structures-only",
        action="store_true",
        help="Only build conformers and write a structures dataset (no predictions).",
    )
    parser.add_argument(
        "--structures-out",
        type=Path,
        default=None,
        help="Output path for structures dataset (directory). "
        "Default: hpc_runs/<timestamp>_cascade_conformers/structures.parquet when --structures-only.",
    )
    parser.add_argument(
        "--structures-in",
        type=Path,
        default=None,
        help="Input structures dataset (directory) to use for predictions (skips embedding).",
    )
    parser.add_argument(
        "--conformer-rank",
        type=int,
        default=0,
        help="Conformer rank to use from structures dataset (default: 0 = lowest energy).",
    )
    parser.add_argument(
        "--use-all-conformers",
        action="store_true",
        help="Use all conformers from structures dataset instead of filtering by rank.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output parquet if it exists.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output parquet dataset by skipping already predicted SMILES.",
    )
    parser.add_argument(
        "--skip-existing-from",
        type=Path,
        default=None,
        help="Skip SMILES already present in this parquet dataset (useful when writing to a new output).",
    )
    parser.add_argument(
        "--failed-smiles-path",
        type=Path,
        default=None,
        help="Write failed SMILES with reasons to this TSV (default depends on mode).",
    )
    parser.add_argument(
        "--no-write-failures",
        action="store_true",
        help="Disable writing failed SMILES file.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log progress every N batches (default: 10).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load/embed SMILES but skip model inference (for quick smoke tests).",
    )
    return parser.parse_args(argv)


def _append_module_path(model_dir: Path) -> None:
    code_base = CASCADE_MODEL_CODE_DIR.resolve()
    local_modules = code_base / "modules"
    model_base = model_dir.resolve()
    model_modules = model_base / "modules"

    for path in (code_base, local_modules, model_base, model_modules):
        if not path.exists():
            continue
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _import_torch_dependencies(model_dir: Path):
    _append_module_path(model_dir)
    import torch  # noqa: WPS433
    from torch_model import PaiNNConfig, PaiNNModel  # noqa: WPS433

    return {"torch": torch, "PaiNNConfig": PaiNNConfig, "PaiNNModel": PaiNNModel}


def _compute_stacked_offsets(sizes, repeats):
    return np.repeat(np.cumsum(np.hstack([0, sizes[:-1]])), repeats)


def ragged_const(inp_arr):
    raise RuntimeError("ragged_const is no longer used; switch to torch batching.")


def atomic_number_tokenizer(atom):
    return atom.GetAtomicNum()


def Mol_iter(df):
    for _, row in df.iterrows():
        yield row["Mol"], row["atom_index"]


def _collate_graphs(graphs: Sequence[dict]) -> dict:
    node_attributes = []
    node_coordinates = []
    edge_indices = []
    atom_index = []
    node_offset = 0
    target_offset = 0

    for graph in graphs:
        n_atom = int(graph["n_atom"])
        n_pro = int(graph["n_pro"])

        node_attributes.append(graph["node_attributes"])
        node_coordinates.append(graph["node_coordinates"])

        edges = graph["edge_indices"].astype(np.int64, copy=False) + node_offset
        edge_indices.append(edges)

        atom_idx = graph["atom_index"].astype(np.int64, copy=True)
        mask = atom_idx >= 0
        atom_idx[mask] += target_offset
        atom_index.append(atom_idx)

        node_offset += n_atom
        target_offset += n_pro

    return {
        "node_attributes": np.concatenate(node_attributes, axis=0),
        "node_coordinates": np.concatenate(node_coordinates, axis=0),
        "edge_indices": np.concatenate(edge_indices, axis=0),
        "atom_index": np.concatenate(atom_index, axis=0),
        "num_targets": target_offset,
    }


def _batch_inputs(inputs: Sequence[dict], batch_size: int) -> Iterable[dict]:
    for start in range(0, len(inputs), batch_size):
        yield _collate_graphs(inputs[start : start + batch_size])


def _mol_with_single_conformer(mol: Mol, conf_id: int) -> Mol:
    new_mol = Chem.Mol(mol)
    conf = mol.GetConformer(conf_id)
    new_mol.RemoveAllConformers()
    new_mol.AddConformer(Chem.Conformer(conf), assignId=True)
    new_mol.SetProp("ConfId", str(conf_id))
    return new_mol


def _rank_conformer_records(records: List[dict]) -> List[dict]:
    def sort_key(rec: dict) -> Tuple[int, float]:
        method_rank = 0 if rec.get("energy_method") == "MMFF" else 1
        energy = rec.get("energy")
        return (method_rank, float("inf") if energy is None else float(energy))

    ranked = sorted(records, key=sort_key)
    for rank, rec in enumerate(ranked):
        rec["conformer_rank"] = rank
    return ranked


def _should_use_2d_coords_fallback(mol: Mol) -> bool:
    """Heuristic for molecules that can make RDKit distance-geometry embedding hang.

    Some large fused polycyclic aromatic systems (many rings, no rotatable bonds,
    almost entirely aromatic) are extremely slow to embed with ETKDG/EmbedMolecule.

    For these, generating 2D coordinates and then optimizing with MMFF/UFF is a
    pragmatic alternative that finishes quickly and produces a usable 3D geometry.
    """
    try:
        heavy = mol.GetNumHeavyAtoms()
        if heavy <= 0:
            return False
        num_rings = int(rdMolDescriptors.CalcNumRings(mol))
        if num_rings < 10:
            return False
        if int(Lipinski.NumRotatableBonds(mol)) != 0:
            return False
        aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        aromatic_frac = aromatic / heavy
        return aromatic_frac >= 0.9
    except Exception:
        return False


def _effective_num_conformers(mol_no_h: Mol, requested: int, adaptive: bool) -> int:
    if requested <= 1:
        return 1
    if not adaptive:
        return int(requested)
    try:
        heavy = int(mol_no_h.GetNumHeavyAtoms())
        rot = int(Lipinski.NumRotatableBonds(mol_no_h))
    except Exception:
        return int(requested)

    if rot <= 0:
        return 1
    # Modest caps keep throughput high without sacrificing much ensemble diversity.
    if heavy >= 80:
        return min(int(requested), 5)
    if rot <= 2:
        return min(int(requested), 5)
    if heavy >= 60:
        return min(int(requested), 10)
    return int(requested)


def _add_2d_conformers_with_z_jitter(mol: Mol, smi: str, num_conformers: int) -> List[int]:
    """Create initial conformers from 2D coords and tiny z-jitter (deterministic)."""
    AllChem.Compute2DCoords(mol)
    base = mol.GetConformer()
    coords = [base.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    mol.RemoveAllConformers()

    # Deterministic seed per SMILES for reproducible structures under resume/reruns.
    seed = int(hashlib.md5(smi.encode("utf-8")).hexdigest()[:8], 16) ^ 0xF00D
    rng = random.Random(seed)

    conf_ids: List[int] = []
    for _ in range(max(1, int(num_conformers))):
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, p in enumerate(coords):
            # Keep xy fixed (2D) but add tiny z jitter to make it 3D.
            conf.SetAtomPosition(i, (float(p.x), float(p.y), (rng.random() - 0.5) * 0.1))
        conf_id = mol.AddConformer(conf, assignId=True)
        conf_ids.append(int(conf_id))
    return conf_ids


def _embed_worker(task: Tuple[int, str, int, int, float, int, bool, int]) -> Tuple[str, str, List[dict]]:
    input_index, smi, max_embed_tries, num_conformers, prune_rms_thresh, ff_max_iters, adaptive, skip_heavy_atoms_gt = task
    RDLogger.DisableLog("rdApp.*")
    try:
        mol_no_h = Chem.MolFromSmiles(smi)
        if mol_no_h is None:
            return "parse", smi, []

        heavy_atoms = int(mol_no_h.GetNumHeavyAtoms())
        if skip_heavy_atoms_gt and skip_heavy_atoms_gt > 0 and heavy_atoms > int(skip_heavy_atoms_gt):
            return "skip_heavy", smi, []

        target_confs = _effective_num_conformers(
            mol_no_h, requested=int(num_conformers), adaptive=bool(adaptive)
        )
        use_2d_fallback = _should_use_2d_coords_fallback(mol_no_h)

        mol = Chem.AddHs(mol_no_h)
        if mol is None:
            return "parse", smi, []

        conf_ids: List[int] = []
        if use_2d_fallback:
            conf_ids = _add_2d_conformers_with_z_jitter(mol, smi, num_conformers=target_confs)
        else:
            params = AllChem.ETKDGv3()
            if prune_rms_thresh and prune_rms_thresh > 0:
                params.pruneRmsThresh = prune_rms_thresh

            # Deterministic seed per SMILES, but vary across embed retries so retries
            # are meaningful (the previous version used a fixed seed for every try).
            base_seed = int(hashlib.md5(smi.encode("utf-8")).hexdigest()[:8], 16) ^ 0xF00D
            for attempt in range(max(1, int(max_embed_tries))):
                mol.RemoveAllConformers()
                params.randomSeed = int((base_seed + attempt) & 0x7FFFFFFF)
                params.useRandomCoords = attempt > 0

                if target_confs <= 1:
                    if AllChem.EmbedMolecule(mol, params=params) == 0:
                        conf_ids = [0]
                        break
                else:
                    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=target_confs, params=params))
                    if ids:
                        conf_ids = [int(cid) for cid in ids]
                        break

        if not conf_ids:
            return "embed", smi, []

        mmff_has = AllChem.MMFFHasAllMoleculeParams(mol)
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol) if mmff_has else None
        uff_has = True
        if hasattr(AllChem, "UFFHasAllMoleculeParams"):
            uff_has = AllChem.UFFHasAllMoleculeParams(mol)

        records: List[dict] = []
        for conf_id in conf_ids:
            energy = None
            method = None
            optimized = False

            if mmff_has:
                try:
                    mmff_status = AllChem.MMFFOptimizeMolecule(
                        mol, maxIters=int(ff_max_iters), confId=conf_id
                    )
                except Exception:
                    mmff_status = -1
                if mmff_status in (0, 1):
                    try:
                        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
                        energy = float(ff.CalcEnergy())
                    except Exception:
                        energy = None
                    method = "MMFF"
                    optimized = True

            if not optimized and uff_has:
                try:
                    uff_status = AllChem.UFFOptimizeMolecule(
                        mol, maxIters=int(ff_max_iters), confId=conf_id
                    )
                except Exception:
                    uff_status = -1
                if uff_status in (0, 1):
                    try:
                        ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                        energy = float(ff.CalcEnergy())
                    except Exception:
                        energy = None
                    method = "UFF"
                    optimized = True

            if optimized:
                single = _mol_with_single_conformer(mol, conf_id)
                records.append(
                    {
                        "smiles": smi,
                        "input_index": int(input_index),
                        "mol": single,
                        "conformer_id": int(conf_id),
                        "energy": energy,
                        "energy_method": method,
                    }
                )

        if not records:
            return "opt", smi, []

        records = _rank_conformer_records(records)
        return "ok", smi, records
    except Exception as exc:
        return f"embed_exc:{type(exc).__name__}", smi, []


def _resolve_num_workers(num_workers: int) -> int:
    if num_workers <= 0:
        count = os.cpu_count() or 1
        return max(1, count - 1)
    return num_workers


def embed_smiles_conformers(
    smiles_list: Sequence[str],
    max_embed_tries: int = 20,
    num_conformers: int = 1,
    prune_rms_thresh: float = 0.0,
    ff_max_iters: int = 200,
    adaptive_conformers: bool = False,
    skip_heavy_atoms_gt: int = 0,
    embed_stall_timeout_s: float = 120.0,
    num_workers: int = 1,
    mp_context: str = "spawn",
    mp_chunksize: int = 20,
) -> Tuple[List[dict], List[str]]:
    records: List[dict] = []
    failures: List[str] = []

    resolved_workers = _resolve_num_workers(num_workers)
    tasks = [
        (
            idx,
            smi,
            max_embed_tries,
            num_conformers,
            prune_rms_thresh,
            ff_max_iters,
            adaptive_conformers,
            skip_heavy_atoms_gt,
        )
        for idx, smi in enumerate(smiles_list)
    ]
    if resolved_workers <= 1 or len(tasks) == 1:
        iterator = map(_embed_worker, tasks)
    else:
        ctx = mp.get_context(mp_context)
        remaining = set(smiles_list)
        with ctx.Pool(processes=resolved_workers) as pool:
            # Unordered reduces head-of-line blocking, but we also need a watchdog:
            # a single pathological molecule can stall completion of a whole batch.
            iterator = pool.imap_unordered(_embed_worker, tasks, chunksize=mp_chunksize)
            timed_out = False
            for _ in range(len(tasks)):
                try:
                    if embed_stall_timeout_s and embed_stall_timeout_s > 0:
                        status, smi, recs = iterator.next(timeout=float(embed_stall_timeout_s))
                    else:
                        status, smi, recs = iterator.next()
                except mp.TimeoutError:
                    timed_out = True
                    pool.terminate()
                    break

                remaining.discard(smi)
                if status == "ok":
                    records.extend(recs)
                else:
                    failures.append(f"{status}\t{smi}")

        if timed_out and remaining:
            for smi in remaining:
                failures.append(f"stall_timeout\t{smi}")
        return records, failures

    for status, smi, recs in iterator:
        if status == "ok":
            records.extend(recs)
        else:
            failures.append(f"{status}\t{smi}")

    return records, failures


def embed_smiles(
    smiles_list: Sequence[str],
    max_embed_tries: int = 20,
    num_conformers: int = 1,
    prune_rms_thresh: float = 0.0,
    ff_max_iters: int = 200,
    adaptive_conformers: bool = False,
    skip_heavy_atoms_gt: int = 0,
    embed_stall_timeout_s: float = 120.0,
    num_workers: int = 1,
    mp_context: str = "spawn",
    mp_chunksize: int = 20,
) -> Tuple[List[Mol], List[str], List[str]]:
    records, failures = embed_smiles_conformers(
        smiles_list,
        max_embed_tries=max_embed_tries,
        num_conformers=num_conformers,
        prune_rms_thresh=prune_rms_thresh,
        ff_max_iters=ff_max_iters,
        adaptive_conformers=adaptive_conformers,
        skip_heavy_atoms_gt=skip_heavy_atoms_gt,
        embed_stall_timeout_s=embed_stall_timeout_s,
        num_workers=num_workers,
        mp_context=mp_context,
        mp_chunksize=mp_chunksize,
    )
    if not records:
        return [], [], failures

    by_input_index: dict[int, List[dict]] = {}
    for rec in records:
        by_input_index.setdefault(int(rec["input_index"]), []).append(rec)

    mols: List[Mol] = []
    clean_smiles: List[str] = []
    for input_index, smi in enumerate(smiles_list):
        recs = by_input_index.get(int(input_index))
        if not recs:
            continue
        # Choose lowest-rank conformer (rank assigned in worker).
        recs_sorted = sorted(recs, key=lambda r: r.get("conformer_rank", 0))
        rec = recs_sorted[0]
        mol = rec["mol"]
        mol.SetProp("_Name", f"Molecule_{len(mols) + 1}")
        if not mol.HasProp("ConfId"):
            mol.SetProp("ConfId", str(rec.get("conformer_id", 0)))
        mols.append(mol)
        clean_smiles.append(smi)

    return mols, clean_smiles, failures


def _default_structures_out() -> Path:
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return REPO_ROOT / "hpc_runs" / f"{stamp}_cascade_conformers" / "structures.parquet"


def mol_to_molblock(mol: Mol) -> str:
    return Chem.MolToMolBlock(mol)


def mol_from_molblock(block: str) -> Mol | None:
    try:
        mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=False)
    except Exception:
        return None
    if mol is None:
        return None
    if not mol.HasProp("ConfId"):
        mol.SetProp("ConfId", "0")
    return mol


def get_carbon_indices(mol: Mol) -> np.ndarray:
    return np.array([atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6], dtype=int)


@dataclass
class ModelArtifacts:
    preprocessor: object
    model: object
    device: str
    torch: object
    output_dim: int
    output_scale: float
    output_bias: float
    output_affine_source: str
    quantile_metadata: dict | None = None
    output_quantile_scale: float = 1.0


def _to_float_or_none(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _extract_output_affine(payload: dict) -> Tuple[Optional[float], Optional[float]]:
    scale = _to_float_or_none(payload.get("output_scale"))
    bias = _to_float_or_none(payload.get("output_bias"))
    if scale is not None and bias is not None:
        return scale, bias

    output_units = str(payload.get("output_units", "")).strip().lower()
    if output_units == "ppm":
        return 1.0, 0.0
    if output_units == "normalized":
        mean = None
        std = None
        for key in ("target_mean_ppm", "mean_ppm", "target_mean", "train_y_mean"):
            mean = _to_float_or_none(payload.get(key))
            if mean is not None:
                break
        for key in ("target_std_ppm", "std_ppm", "target_std", "train_y_std"):
            std = _to_float_or_none(payload.get(key))
            if std is not None:
                break
        if mean is not None and std is not None and std > 0:
            return std, mean
    return None, None


def _unwrap_state_dict(payload) -> dict:
    if isinstance(payload, dict):
        model_state = payload.get("model_state")
        if isinstance(model_state, dict):
            return model_state
        return payload
    raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")


def _infer_painn_config_overrides(state_dict: dict) -> dict:
    overrides: dict = {}

    out_w = state_dict.get("mlp.2.weight")
    if out_w is not None and hasattr(out_w, "shape") and len(out_w.shape) == 2:
        overrides["output_dim"] = int(out_w.shape[0])

    emb = state_dict.get("solvent_embedding.weight")
    if emb is not None and hasattr(emb, "shape") and len(emb.shape) == 2:
        overrides["solvent_vocab_size"] = int(emb.shape[0])
        overrides["solvent_emb_dim"] = int(emb.shape[1])

    bias = state_dict.get("solvent_bias.weight")
    if bias is not None and hasattr(bias, "shape") and len(bias.shape) >= 1:
        overrides["solvent_use_bias"] = True
        overrides.setdefault("solvent_vocab_size", int(bias.shape[0]))

    adapter0 = state_dict.get("solvent_adapter.0.weight")
    if adapter0 is not None and hasattr(adapter0, "shape") and len(adapter0.shape) == 2:
        overrides["solvent_adapter_hidden_dim"] = int(adapter0.shape[0])
        in_dim = int(adapter0.shape[1])
        # PaiNN pooled representation width defaults to 256 in this codebase.
        if "solvent_emb_dim" not in overrides and in_dim > 256:
            overrides["solvent_emb_dim"] = int(in_dim - 256)

    return overrides


def resolve_output_affine(
    model_dir: Path,
    weights_path: Path,
    output_scale: Optional[float],
    output_bias: Optional[float],
) -> Tuple[float, float, str]:
    if (output_scale is None) ^ (output_bias is None):
        raise ValueError("Provide both --output-scale and --output-bias, or neither.")
    if output_scale is not None and output_bias is not None:
        return float(output_scale), float(output_bias), "cli"

    candidate_dirs = [weights_path.parent, model_dir, weights_path.parent.parent]
    seen = set()
    for directory in candidate_dirs:
        for name in MODEL_METADATA_FILENAMES:
            path = (directory / name).resolve()
            if path in seen or not path.exists():
                continue
            seen.add(path)
            try:
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            scale, bias = _extract_output_affine(payload)
            if scale is not None and bias is not None:
                return float(scale), float(bias), f"metadata:{path}"

    return float(LEGACY_OUTPUT_SCALE), float(LEGACY_OUTPUT_BIAS), "legacy_default"


def _load_model_metadata(model_dir: Path, weights_path: Path) -> dict | None:
    candidate_dirs = [weights_path.parent, model_dir, weights_path.parent.parent]
    seen = set()
    for directory in candidate_dirs:
        for name in MODEL_METADATA_FILENAMES:
            path = (directory / name).resolve()
            if path in seen or not path.exists():
                continue
            seen.add(path)
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if not isinstance(payload, dict):
                raise RuntimeError(f"Model metadata is not a JSON object: {path}")
            return payload
    return None


def _apply_quantile_parameterization(raw, metadata: dict, torch, device: str):
    mode = str(metadata.get("quantile_parameterization", ""))
    if not mode:
        raise RuntimeError("99-output quantile checkpoints require quantile_parameterization metadata")
    if mode == "direct":
        return raw
    if mode != "ordered-gaps":
        raise ValueError(f"Unknown quantile parameterization in metadata: {mode!r}")

    if raw.ndim != 2:
        raise RuntimeError(f"Ordered-gap raw quantile prediction must be 2D, got shape={tuple(raw.shape)}")
    quantiles = np.asarray(metadata.get("quantiles"), dtype=np.float64)
    if quantiles.ndim != 1 or int(quantiles.shape[0]) != int(raw.shape[1]):
        raise RuntimeError(f"Quantile metadata mismatch: quantiles={quantiles.shape}, raw={tuple(raw.shape)}")
    median_idx = int(metadata.get("median_quantile_index", int(np.argmin(np.abs(quantiles - 0.5)))))
    if median_idx <= 0 or median_idx >= int(raw.shape[1]) - 1:
        raise RuntimeError(f"Invalid ordered-gap quantile shape={tuple(raw.shape)} median_idx={median_idx}")
    gap_scale = float(metadata.get("quantile_gap_scale", 0.05))
    if gap_scale <= 0.0:
        raise ValueError(f"quantile_gap_scale must be positive, got {gap_scale}")

    median = raw[:, median_idx : median_idx + 1]
    gap_floor = torch.as_tensor(1.0e-4, device=device, dtype=raw.dtype)
    gap_scale_t = torch.as_tensor(gap_scale, device=device, dtype=raw.dtype)
    left_gaps = torch.nn.functional.softplus(raw[:, :median_idx]) * gap_scale_t + gap_floor
    right_gaps = torch.nn.functional.softplus(raw[:, median_idx + 1 :]) * gap_scale_t + gap_floor
    left_offsets = torch.flip(torch.cumsum(torch.flip(left_gaps, dims=[1]), dim=1), dims=[1])
    right_offsets = torch.cumsum(right_gaps, dim=1)
    return torch.cat([median - left_offsets, median, median + right_offsets], dim=1)


def _scale_quantiles_around_median(quantiles_ppm: np.ndarray, *, median_idx: int, scale: float) -> np.ndarray:
    scale = float(scale)
    if scale <= 0.0:
        raise ValueError(f"output_quantile_scale must be positive, got {scale}")
    if scale == 1.0:
        return quantiles_ppm
    if quantiles_ppm.ndim != 2 or median_idx < 0 or median_idx >= int(quantiles_ppm.shape[1]):
        raise RuntimeError(f"Invalid quantile array shape={quantiles_ppm.shape} median_idx={median_idx}")
    median = quantiles_ppm[:, median_idx : median_idx + 1]
    return median + scale * (quantiles_ppm - median)


def load_artifacts(
    model_dir: Path,
    weights_path: Path,
    preprocessor_path: Path,
    device: str,
    *,
    output_scale: Optional[float] = None,
    output_bias: Optional[float] = None,
    output_affine_source: str = "legacy_default",
    output_quantile_scale: float = 1.0,
) -> ModelArtifacts:
    deps = _import_torch_dependencies(model_dir)
    torch = deps["torch"]
    PaiNNConfig = deps["PaiNNConfig"]
    PaiNNModel = deps["PaiNNModel"]

    globals().setdefault("atomic_number_tokenizer", atomic_number_tokenizer)
    globals().setdefault("Mol_iter", Mol_iter)
    globals().setdefault("_compute_stacked_offsets", _compute_stacked_offsets)
    globals().setdefault("ragged_const", ragged_const)

    import types

    main_module = sys.modules.get("__main__")
    if main_module is None or not isinstance(main_module, types.ModuleType):
        main_module = types.ModuleType("__main__")
        sys.modules["__main__"] = main_module
    for name in (
        "atomic_number_tokenizer",
        "Mol_iter",
        "_compute_stacked_offsets",
        "ragged_const",
    ):
        setattr(main_module, name, globals()[name])

    with open(preprocessor_path, "rb") as fh:
        preprocessor_bundle = pickle.load(fh)  # noqa: S301
    preprocessor = preprocessor_bundle["preprocessor"]

    state_payload = torch.load(weights_path, map_location=device)
    state_dict = _unwrap_state_dict(state_payload)
    config_overrides = _infer_painn_config_overrides(state_dict)
    model = PaiNNModel(PaiNNConfig(**config_overrides))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    output_dim = int(getattr(getattr(model, "config", None), "output_dim", config_overrides.get("output_dim", 1)))
    metadata_payload = _load_model_metadata(model_dir, weights_path)
    if output_dim == 99:
        if metadata_payload is None:
            raise RuntimeError(f"99-output quantile checkpoint requires metadata next to weights: {weights_path}")
        quantiles = np.asarray(metadata_payload.get("quantiles"), dtype=np.float64)
        if quantiles.ndim != 1 or int(quantiles.shape[0]) != 99:
            raise RuntimeError(f"99-output quantile checkpoint has invalid quantile metadata: {weights_path}")

    if output_scale is None or output_bias is None:
        auto_scale, auto_bias, auto_source = resolve_output_affine(
            model_dir,
            weights_path,
            None,
            None,
        )
        resolved_scale = float(auto_scale)
        resolved_bias = float(auto_bias)
        if output_affine_source == "legacy_default":
            output_affine_source = auto_source
    else:
        resolved_scale = float(output_scale)
        resolved_bias = float(output_bias)

    return ModelArtifacts(
        preprocessor=preprocessor,
        model=model,
        device=device,
        torch=torch,
        output_dim=output_dim,
        output_scale=resolved_scale,
        output_bias=resolved_bias,
        output_affine_source=str(output_affine_source),
        quantile_metadata=metadata_payload,
        output_quantile_scale=float(output_quantile_scale),
    )


def _expand_by_atoms(values: Sequence, atom_indices: List[np.ndarray]) -> List:
    expanded: List = []
    for value, indices in zip(values, atom_indices):
        expanded.extend([value] * len(indices))
    return expanded


def _normalize_extra_cols(extra_cols: dict | None, n_rows: int) -> dict[str, List]:
    if not extra_cols:
        return {}
    normalized: dict[str, List] = {}
    for key, values in extra_cols.items():
        values_list = list(values)
        if len(values_list) != int(n_rows):
            raise ValueError(
                f"extra_cols[{key!r}] length mismatch: got {len(values_list)}, expected {n_rows}"
            )
        normalized[str(key)] = values_list
    return normalized


def _predict_from_mols_ready(
    mols_ready: List[Mol],
    atom_indices: List[np.ndarray],
    mols_smiles: List[str],
    artifacts: ModelArtifacts | None,
    batch_size: int,
    dry_run: bool,
    extra_cols: dict | None = None,
    solvent_ids_per_mol: Sequence[int] | None = None,
    molecule_ids_per_mol: Sequence[int] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    if not atom_indices:
        return pd.DataFrame(), []
    extra_cols_per_mol = _normalize_extra_cols(extra_cols, len(mols_ready))
    if molecule_ids_per_mol is None:
        molecule_ids = list(range(len(mols_ready)))
    else:
        molecule_ids = [int(x) for x in molecule_ids_per_mol]
        if len(molecule_ids) != int(len(mols_ready)):
            raise ValueError(
                f"molecule_ids_per_mol length mismatch: got {len(molecule_ids)}, expected {len(mols_ready)}"
            )
    if solvent_ids_per_mol is not None:
        solvent_ids = [int(x) for x in solvent_ids_per_mol]
        if len(solvent_ids) != int(len(mols_ready)):
            raise ValueError(
                f"solvent_ids_per_mol length mismatch: got {len(solvent_ids)}, expected {len(mols_ready)}"
            )
    else:
        solvent_ids = None

    if dry_run:
        smiles_expanded: List[str] = []
        for smi, indices in zip(mols_smiles, atom_indices):
            smiles_expanded.extend([smi] * len(indices))
        dry_output_dim = int(getattr(artifacts, "output_dim", 1)) if artifacts is not None else 1
        df = pd.DataFrame(
            {
                "molecule_id": _expand_by_atoms(molecule_ids, atom_indices),
                "smiles": smiles_expanded,
                "atom_index": np.concatenate(atom_indices),
                "shift_ppm": np.nan,
            }
        )
        if dry_output_dim == 2:
            df["shift_std_ppm"] = np.nan
        elif dry_output_dim == 3:
            df["shift_q10_ppm"] = np.nan
            df["shift_q90_ppm"] = np.nan
            df["shift_std_ppm"] = np.nan
        elif dry_output_dim == 99:
            quantile_cols = {
                _quantile_column_name(level): np.full(len(df), np.nan)
                for level in QUANTILE_99_LEVELS
            }
            df = pd.concat([df, pd.DataFrame(quantile_cols)], axis=1)
            df["shift_std_ppm"] = np.nan
        if extra_cols_per_mol:
            for key, values in extra_cols_per_mol.items():
                df[key] = _expand_by_atoms(values, atom_indices)
        return df, []

    if artifacts is None:
        raise RuntimeError("Model artifacts are required for prediction.")

    inputs: List[dict] = []
    preprocess_failures: List[str] = []
    try:
        inp_df = pd.DataFrame({"Mol": mols_ready, "atom_index": atom_indices})
        inputs = list(artifacts.preprocessor.predict(Mol_iter(inp_df)))
    except Exception:
        kept_mols: List[Mol] = []
        kept_indices: List[np.ndarray] = []
        kept_smiles: List[str] = []
        kept_extra_cols: dict[str, List] = {k: [] for k in extra_cols_per_mol}
        kept_solvent_ids: List[int] = []
        kept_molecule_ids: List[int] = []
        for row_idx, (mol, indices, smi) in enumerate(zip(mols_ready, atom_indices, mols_smiles)):
            try:
                inp_df = pd.DataFrame({"Mol": [mol], "atom_index": [indices]})
                inputs.extend(artifacts.preprocessor.predict(Mol_iter(inp_df)))
                kept_mols.append(mol)
                kept_indices.append(indices)
                kept_smiles.append(smi)
                for key in kept_extra_cols:
                    kept_extra_cols[key].append(extra_cols_per_mol[key][row_idx])
                kept_molecule_ids.append(int(molecule_ids[row_idx]))
                if solvent_ids is not None:
                    kept_solvent_ids.append(int(solvent_ids[row_idx]))
            except Exception:
                preprocess_failures.append(f"preprocess\t{smi}")
        mols_ready = kept_mols
        atom_indices = kept_indices
        mols_smiles = kept_smiles
        extra_cols_per_mol = kept_extra_cols
        molecule_ids = kept_molecule_ids
        if solvent_ids is not None:
            solvent_ids = kept_solvent_ids
        if not atom_indices:
            return pd.DataFrame(), preprocess_failures

    torch = artifacts.torch
    output_dim = int(getattr(artifacts, "output_dim", 1))
    if output_dim not in (1, 2, 3, 99):
        raise RuntimeError(f"Unsupported CASCADE output_dim={output_dim}; expected 1, 2, 3, or 99.")
    is_uncertainty_mode = output_dim == 2
    is_quantile_mode = output_dim in (3, 99)
    if output_dim == 3:
        q10_idx, q50_idx, q90_idx = 0, 1, 2
    elif output_dim == 99:
        q10_idx, q50_idx, q90_idx = 9, 49, 89
    else:
        q10_idx = q50_idx = q90_idx = -1

    pred_means: List[float] = []
    pred_log_vars: List[float] = []
    pred_q10: List[float] = []
    pred_q90: List[float] = []
    pred_quantile_batches: List[np.ndarray] = []
    model = artifacts.model
    with torch.no_grad():
        for start in range(0, len(inputs), int(batch_size)):
            stop = min(start + int(batch_size), len(inputs))
            batch = _collate_graphs(inputs[start:stop])
            torch_batch = {
                "node_attributes": torch.as_tensor(batch["node_attributes"], device=artifacts.device, dtype=torch.long),
                "node_coordinates": torch.as_tensor(
                    batch["node_coordinates"], device=artifacts.device, dtype=torch.float32
                ),
                "edge_indices": torch.as_tensor(batch["edge_indices"], device=artifacts.device, dtype=torch.long),
                "atom_index": torch.as_tensor(batch["atom_index"], device=artifacts.device, dtype=torch.long),
                "num_targets": batch["num_targets"],
            }
            if solvent_ids is not None:
                expanded_solvent_ids = _expand_by_atoms(solvent_ids[start:stop], atom_indices[start:stop])
                solvent_ids_t = torch.as_tensor(
                    expanded_solvent_ids,
                    device=artifacts.device,
                    dtype=torch.long,
                )
                raw_preds = model(torch_batch, solvent_ids=solvent_ids_t)
            else:
                raw_preds = model(torch_batch)
            if output_dim == 99:
                raw_preds = _apply_quantile_parameterization(
                    raw_preds,
                    artifacts.quantile_metadata,
                    torch,
                    artifacts.device,
                )
            preds = raw_preds.detach().cpu().numpy()

            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            if preds.ndim != 2 or int(preds.shape[1]) != int(output_dim):
                raise RuntimeError(
                    f"Unexpected prediction tensor shape: {tuple(preds.shape)} for output_dim={output_dim}"
                )
            if is_quantile_mode:
                pred_means.extend(preds[:, q50_idx].tolist())
                pred_q10.extend(preds[:, q10_idx].tolist())
                pred_q90.extend(preds[:, q90_idx].tolist())
                if output_dim == 99:
                    pred_quantile_batches.append(preds.astype(np.float64, copy=True))
            else:
                pred_means.extend(preds[:, 0].tolist())
            if is_uncertainty_mode:
                pred_log_vars.extend(preds[:, 1].tolist())

    smiles_expanded: List[str] = []
    for smi, indices in zip(mols_smiles, atom_indices):
        smiles_expanded.extend([smi] * len(indices))

    pred_arr = np.asarray(pred_means, dtype=np.float64)
    shift_ppm = np.round(
        pred_arr * float(artifacts.output_scale) + float(artifacts.output_bias),
        2,
    )
    shift_std_ppm = None
    shift_q10_ppm = None
    shift_q90_ppm = None
    shift_quantiles_ppm = None
    if is_uncertainty_mode:
        log_var_arr = np.asarray(pred_log_vars, dtype=np.float64)
        std_native = np.exp(0.5 * np.clip(log_var_arr, -40.0, 40.0))
        shift_std_ppm = np.round(std_native * abs(float(artifacts.output_scale)), 4)
    elif is_quantile_mode:
        if output_dim == 99:
            if not pred_quantile_batches:
                raise RuntimeError("Missing 99-quantile prediction batches")
            quantile_arr = np.concatenate(pred_quantile_batches, axis=0)
            if quantile_arr.ndim != 2 or int(quantile_arr.shape[1]) != 99:
                raise RuntimeError(f"Unexpected 99-quantile tensor shape: {tuple(quantile_arr.shape)}")
            shift_quantiles_ppm_raw = quantile_arr * float(artifacts.output_scale) + float(artifacts.output_bias)
            shift_quantiles_ppm_raw = _scale_quantiles_around_median(
                shift_quantiles_ppm_raw,
                median_idx=int(q50_idx),
                scale=float(getattr(artifacts, "output_quantile_scale", 1.0)),
            )
            shift_q10_ppm_raw = shift_quantiles_ppm_raw[:, q10_idx]
            shift_q90_ppm_raw = shift_quantiles_ppm_raw[:, q90_idx]
            shift_quantiles_ppm = np.round(shift_quantiles_ppm_raw, 2)
        else:
            q10_arr = np.asarray(pred_q10, dtype=np.float64)
            q90_arr = np.asarray(pred_q90, dtype=np.float64)
            shift_q10_ppm_raw = q10_arr * float(artifacts.output_scale) + float(artifacts.output_bias)
            shift_q90_ppm_raw = q90_arr * float(artifacts.output_scale) + float(artifacts.output_bias)
        shift_q10_ppm = np.round(shift_q10_ppm_raw, 2)
        shift_q90_ppm = np.round(shift_q90_ppm_raw, 2)
        iqr_ppm = np.abs(shift_q90_ppm_raw - shift_q10_ppm_raw)
        shift_std_ppm = np.round(np.maximum(iqr_ppm / (2.0 * Q90_ZSCORE), 1.0e-4), 4)
    expected = int(sum(len(indices) for indices in atom_indices))
    if shift_ppm.shape[0] != expected:
        raise RuntimeError(
            f"CASCADE prediction length mismatch: got {shift_ppm.shape[0]} targets, expected {expected}"
        )
    if shift_quantiles_ppm is not None and int(shift_quantiles_ppm.shape[0]) != expected:
        raise RuntimeError(
            f"CASCADE quantile prediction length mismatch: got {shift_quantiles_ppm.shape[0]} targets, expected {expected}"
        )

    # Drop entire molecules with any invalid shift value (keeps downstream parquet clean).
    keep_mask = np.ones(expected, dtype=bool)
    invalid_failures: List[str] = []
    invalid_seen: set[str] = set()
    pos = 0
    for smi, indices in zip(mols_smiles, atom_indices):
        n = len(indices)
        sl = shift_ppm[pos : pos + n]
        bad_reason = None
        if not np.isfinite(sl).all():
            bad_reason = "pred_nonfinite"
        elif (sl < SHIFT_PPM_MIN).any() or (sl > SHIFT_PPM_MAX).any():
            bad_reason = "pred_out_of_range"
        elif shift_q10_ppm is not None and shift_q90_ppm is not None:
            sl_q10 = shift_q10_ppm[pos : pos + n]
            sl_q90 = shift_q90_ppm[pos : pos + n]
            if not np.isfinite(sl_q10).all() or not np.isfinite(sl_q90).all():
                bad_reason = "pred_quantile_invalid"
            elif shift_quantiles_ppm is not None:
                sl_quantiles = shift_quantiles_ppm[pos : pos + n, :]
                if not np.isfinite(sl_quantiles).all():
                    bad_reason = "pred_quantile_invalid"
        elif shift_std_ppm is not None:
            sl_std = shift_std_ppm[pos : pos + n]
            if not np.isfinite(sl_std).all() or (sl_std <= 0).any():
                bad_reason = "pred_std_invalid"
        if bad_reason is not None:
            keep_mask[pos : pos + n] = False
            if smi not in invalid_seen:
                invalid_seen.add(smi)
                invalid_failures.append(f"{bad_reason}\t{smi}")
        pos += n

    if invalid_failures:
        preprocess_failures.extend(invalid_failures)
    df = pd.DataFrame(
        {
            "molecule_id": _expand_by_atoms(molecule_ids, atom_indices),
            "smiles": smiles_expanded,
            "atom_index": np.concatenate(atom_indices),
            "shift_ppm": shift_ppm,
        }
    )
    if shift_quantiles_ppm is None and shift_q10_ppm is not None and shift_q90_ppm is not None:
        df["shift_q10_ppm"] = shift_q10_ppm
        df["shift_q90_ppm"] = shift_q90_ppm
    if shift_quantiles_ppm is not None:
        quantile_cols = {
            _quantile_column_name(level): shift_quantiles_ppm[:, idx]
            for idx, level in enumerate(QUANTILE_99_LEVELS)
        }
        df = pd.concat([df, pd.DataFrame(quantile_cols)], axis=1)
    if shift_std_ppm is not None:
        df["shift_std_ppm"] = shift_std_ppm
    if extra_cols_per_mol:
        for key, values in extra_cols_per_mol.items():
            df[key] = _expand_by_atoms(values, atom_indices)
    if invalid_failures:
        df = df.loc[keep_mask].reset_index(drop=True)
    return df, preprocess_failures


def predict_batch(
    smiles_batch: Sequence[str],
    artifacts: ModelArtifacts | None,
    batch_size: int,
    max_embed_tries: int,
    num_conformers: int,
    prune_rms_thresh: float,
    ff_max_iters: int,
    adaptive_conformers: bool,
    skip_heavy_atoms_gt: int,
    embed_stall_timeout_s: float,
    num_workers: int,
    mp_context: str,
    mp_chunksize: int,
    dry_run: bool,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    mols_embedded, smiles_embedded, failures = embed_smiles(
        smiles_batch,
        max_embed_tries=max_embed_tries,
        num_conformers=num_conformers,
        prune_rms_thresh=prune_rms_thresh,
        ff_max_iters=ff_max_iters,
        adaptive_conformers=adaptive_conformers,
        skip_heavy_atoms_gt=skip_heavy_atoms_gt,
        embed_stall_timeout_s=embed_stall_timeout_s,
        num_workers=num_workers,
        mp_context=mp_context,
        mp_chunksize=mp_chunksize,
    )
    mols_ready: List[Mol] = []
    atom_indices: List[np.ndarray] = []
    mols_smiles: List[str] = []
    no_carbon: List[str] = []

    for smi, mol in zip(smiles_embedded, mols_embedded):
        indices = get_carbon_indices(mol)
        if len(indices) == 0:
            no_carbon.append(f"no_carbon\t{smi}")
            continue
        mols_ready.append(mol)
        atom_indices.append(indices)
        mols_smiles.append(smi)

    if not atom_indices:
        return pd.DataFrame(), failures, no_carbon

    df, preprocess_failures = _predict_from_mols_ready(
        mols_ready, atom_indices, mols_smiles, artifacts, batch_size=batch_size, dry_run=dry_run
    )
    failures.extend(preprocess_failures)
    return df, failures, no_carbon


def predict_from_mols(
    mols: Sequence[Mol],
    smiles: Sequence[str],
    artifacts: ModelArtifacts | None,
    batch_size: int,
    dry_run: bool,
    extra_cols: dict | None = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    if len(mols) != len(smiles):
        raise ValueError(
            f"mols/smiles length mismatch: got {len(mols)} mols and {len(smiles)} smiles"
        )

    extra_cols_all = _normalize_extra_cols(extra_cols, len(mols))
    extra_cols_ready: dict[str, List] = {k: [] for k in extra_cols_all}

    mols_ready: List[Mol] = []
    atom_indices: List[np.ndarray] = []
    mols_smiles: List[str] = []
    no_carbon: List[str] = []

    for row_idx, (smi, mol) in enumerate(zip(smiles, mols)):
        indices = get_carbon_indices(mol)
        if len(indices) == 0:
            no_carbon.append(f"no_carbon\t{smi}")
            continue
        mols_ready.append(mol)
        atom_indices.append(indices)
        mols_smiles.append(smi)
        for key in extra_cols_ready:
            extra_cols_ready[key].append(extra_cols_all[key][row_idx])

    if not atom_indices:
        return pd.DataFrame(), [], no_carbon

    df, preprocess_failures = _predict_from_mols_ready(
        mols_ready,
        atom_indices,
        mols_smiles,
        artifacts,
        batch_size=batch_size,
        dry_run=dry_run,
        extra_cols=extra_cols_ready,
    )
    return df, preprocess_failures, no_carbon


class ParquetDatasetWriter:
    def __init__(self, path: Path, metadata: dict, overwrite: bool = False, resume: bool = False):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            if self.path.is_file():
                if overwrite:
                    self.path.unlink()
                    self.path.mkdir(parents=True, exist_ok=True)
                elif resume:
                    self._migrate_file_to_dataset()
                else:
                    raise RuntimeError(f"Output parquet path is a file: {self.path}")
            else:
                if overwrite:
                    shutil.rmtree(self.path)
                    self.path.mkdir(parents=True, exist_ok=True)
                elif not resume:
                    raise RuntimeError(f"Output parquet directory already exists: {self.path}")
        else:
            self.path.mkdir(parents=True, exist_ok=True)

        self.metadata = {str(k): str(v) for k, v in metadata.items()}
        self.part_idx = self._next_part_index()
        self._write_metadata_file(resume=resume)

    def _migrate_file_to_dataset(self) -> None:
        tmp_path = self.path.with_suffix(self.path.suffix + ".bak")
        shutil.move(self.path, tmp_path)
        self.path.mkdir(parents=True, exist_ok=True)
        part_path = self.path / "part-000000.parquet"
        shutil.move(tmp_path, part_path)

    def _next_part_index(self) -> int:
        existing = [p for p in self.path.glob("part-*.parquet") if p.is_file()]
        if not existing:
            return 0
        max_idx = -1
        for path in existing:
            match = re.match(r"part-(\d+)\.parquet", path.name)
            if match:
                max_idx = max(max_idx, int(match.group(1)))
        return max_idx + 1

    def _write_metadata_file(self, resume: bool) -> None:
        meta_path = self.path / "_dataset_metadata.json"
        if meta_path.exists() and resume:
            return
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(self.metadata, fh, indent=2, sort_keys=True)

    def write(self, df: pd.DataFrame) -> None:
        if df.empty:
            return
        import pyarrow as pa  # noqa: WPS433
        import pyarrow.parquet as pq  # noqa: WPS433

        table = pa.Table.from_pandas(df, preserve_index=False)
        if self.metadata:
            table = table.replace_schema_metadata({k: v.encode("utf-8") for k, v in self.metadata.items()})
        out_path = self.path / f"part-{self.part_idx:06d}.parquet"
        pq.write_table(table, out_path)
        self.part_idx += 1

    def close(self) -> None:
        return


def _resolve_device(raw_device: str, model_dir: Path) -> str:
    if raw_device != "auto":
        return raw_device
    deps = _import_torch_dependencies(model_dir)
    torch = deps["torch"]
    if torch.cuda.is_available():
        return "cuda"

    mps_backend = getattr(getattr(torch, "backends", None), "mps", None)
    mps_is_available = getattr(mps_backend, "is_available", None)
    if callable(mps_is_available) and bool(mps_is_available()):
        return "mps"

    return "cpu"


def load_smiles(input_parquet: Path, smiles_col: str, limit: int) -> List[str]:
    df = pd.read_parquet(input_parquet, columns=[smiles_col])
    smiles_series = df[smiles_col].dropna().astype(str)
    smiles_series = smiles_series.drop_duplicates()
    if limit and limit > 0:
        smiles_series = smiles_series.iloc[:limit]
    return smiles_series.tolist()


def load_smiles_from_structures(structures_path: Path, limit: int) -> List[str]:
    import pyarrow.dataset as ds  # noqa: WPS433

    dataset = ds.dataset(str(structures_path), format="parquet", exclude_invalid_files=True)
    smiles: List[str] = []
    seen: set[str] = set()
    for batch in dataset.to_batches(columns=["smiles"]):
        batch_smiles = batch.column(0).to_pylist()
        for smi in batch_smiles:
            if smi is None:
                continue
            smi_str = str(smi)
            if smi_str in seen:
                continue
            seen.add(smi_str)
            smiles.append(smi_str)
            if limit and len(smiles) >= limit:
                return smiles
    return smiles


def _apply_shard(smiles: List[str], num_shards: int, shard_index: int) -> List[str]:
    if num_shards <= 1:
        return smiles
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError(f"Invalid shard index {shard_index} for num_shards={num_shards}")
    filtered: List[str] = []
    for smi in smiles:
        digest = hashlib.md5(smi.encode("utf-8")).hexdigest()
        if int(digest, 16) % num_shards == shard_index:
            filtered.append(smi)
    return filtered


def _init_failure_log(path: Path, enabled: bool, resume: bool) -> Path | None:
    if not enabled:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    if not (resume and path.exists()):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("reason\tsmiles\n")
    return path


def _append_failures(path: Path | None, rows: Sequence[str]) -> None:
    if path is None or not rows:
        return
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(f"{row}\n")


def load_existing_smiles(path: Path) -> set[str]:
    import pyarrow.dataset as ds  # noqa: WPS433
    import pyarrow.parquet as pq  # noqa: WPS433

    if not path.exists():
        return set()
    if path.is_dir():
        has_parts = any(path.glob("part-*.parquet"))
        if not has_parts:
            return set()
    if path.is_file():
        try:
            table = pq.read_table(path, columns=["smiles"])
        except Exception:
            return set()
        return set(table.column(0).to_pylist())

    dataset = ds.dataset(str(path), format="parquet", exclude_invalid_files=True)
    if "smiles" not in dataset.schema.names:
        return set()
    existing: set[str] = set()
    for batch in dataset.to_batches(columns=["smiles"]):
        existing.update(batch.column(0).to_pylist())
    return existing


def load_failed_smiles(path: Path) -> set[str]:
    if not path.exists():
        return set()
    failed: set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if line.lower().startswith("reason\t"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    failed.add(parts[1])
    except Exception:
        return set()
    return failed


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)

    weights_path = args.weights or (args.model_dir / "best_model.pt")
    preprocessor_path = args.preprocessor or (args.model_dir / "preprocessor_orig.p")
    structures_out = args.structures_out
    if args.structures_only:
        if structures_out is None:
            structures_out = _default_structures_out()
        if args.structures_in:
            print("[warn] --structures-in ignored when --structures-only is set.")

    if args.failed_smiles_path is None:
        if args.structures_only:
            target_root = structures_out.parent if structures_out else Path("hpc_runs")
            args.failed_smiles_path = target_root / "failed_smiles.tsv"
        else:
            args.failed_smiles_path = Path("results/cascade_failed_smiles.tsv")

    if not args.structures_only and not args.dry_run:
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Missing CASCADE weights at {weights_path}. Run convert_tf_to_torch.py in"
                f" {args.model_dir} to generate best_model.pt."
            )
        if not preprocessor_path.exists():
            raise FileNotFoundError(f"Missing CASCADE preprocessor at {preprocessor_path}.")

    if args.structures_in and not args.structures_only:
        if not args.structures_in.exists():
            raise FileNotFoundError(f"Structures dataset not found: {args.structures_in}")
    else:
        if not args.input_parquet.exists():
            raise FileNotFoundError(f"Input parquet not found: {args.input_parquet}")

    if args.structures_only:
        load_limit = 0 if args.num_shards > 1 else args.limit
        smiles = load_smiles(args.input_parquet, args.smiles_col, load_limit)
        total_smiles_all = len(smiles)
        smiles = _apply_shard(smiles, args.num_shards, args.shard_index)
        if args.limit and args.limit > 0:
            smiles = smiles[: args.limit]
        total_smiles_shard = len(smiles)
        if total_smiles_all == 0:
            raise RuntimeError("No SMILES loaded from input parquet.")

        skipped_smiles = 0
        if args.skip_existing_from:
            existing = load_existing_smiles(args.skip_existing_from)
            if existing:
                before = len(smiles)
                smiles = [smi for smi in smiles if smi not in existing]
                skipped_smiles += before - len(smiles)
        if args.resume and structures_out and structures_out.exists():
            existing = load_existing_smiles(structures_out)
            if existing:
                before = len(smiles)
                smiles = [smi for smi in smiles if smi not in existing]
                skipped_smiles += before - len(smiles)
        if args.resume and args.failed_smiles_path and args.failed_smiles_path.exists():
            failed = load_failed_smiles(args.failed_smiles_path)
            if failed:
                before = len(smiles)
                smiles = [smi for smi in smiles if smi not in failed]
                skipped_smiles += before - len(smiles)

        metadata = {
            "created_at": datetime.utcnow().isoformat() + "Z",
            "source": "CASCADE-2.0",
            "mode": "structures_only",
            "input_parquet": str(args.input_parquet),
            "smiles_col": args.smiles_col,
            "unique_smiles_total": total_smiles_all,
            "unique_smiles_shard_total": total_smiles_shard,
            "unique_smiles_remaining": len(smiles),
            "skipped_smiles": skipped_smiles,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "skip_existing_from": str(args.skip_existing_from) if args.skip_existing_from else "",
            "max_embed_tries": args.max_embed_tries,
            "num_conformers": args.num_conformers,
            "prune_rms_thresh": args.prune_rms_thresh,
            "ff_max_iters": args.ff_max_iters,
            "adaptive_conformers": args.adaptive_conformers,
            "skip_heavy_atoms_gt": args.skip_heavy_atoms_gt,
            "min_conformer_rank": args.min_conformer_rank,
            "embed_stall_timeout_s": args.embed_stall_timeout_s,
            "num_workers": _resolve_num_workers(args.num_workers),
            "mp_context": args.mp_context,
            "mp_chunksize": args.mp_chunksize,
            "smiles_batch_size": args.smiles_batch_size,
            "conformer": f"ETKDGv3+MMFF/UFF ({args.num_conformers} conformers)",
            "resume": args.resume,
            "output_format": "parquet_dataset",
            "structures_out": str(structures_out),
        }
        writer = ParquetDatasetWriter(
            structures_out, metadata=metadata, overwrite=args.overwrite, resume=args.resume
        )
        failure_path = _init_failure_log(args.failed_smiles_path, not args.no_write_failures, args.resume)

        total_written = 0
        total_failed = 0

        if not smiles:
            print("[INFO] No new SMILES to process after resume filtering.")
            writer.close()
            return 0

        pbar = tqdm(total=len(smiles), unit="smiles")
        interrupted = False
        try:
            for batch_idx, start in enumerate(range(0, len(smiles), args.smiles_batch_size), start=1):
                batch_smiles = smiles[start : start + args.smiles_batch_size]
                records, failures = embed_smiles_conformers(
                    batch_smiles,
                    max_embed_tries=args.max_embed_tries,
                    num_conformers=args.num_conformers,
                    prune_rms_thresh=args.prune_rms_thresh,
                    ff_max_iters=args.ff_max_iters,
                    adaptive_conformers=args.adaptive_conformers,
                    skip_heavy_atoms_gt=args.skip_heavy_atoms_gt,
                    embed_stall_timeout_s=args.embed_stall_timeout_s,
                    num_workers=args.num_workers,
                    mp_context=args.mp_context,
                    mp_chunksize=args.mp_chunksize,
                )
                if records and args.min_conformer_rank and args.min_conformer_rank > 0:
                    min_rank = int(args.min_conformer_rank)
                    records = [rec for rec in records if int(rec.get("conformer_rank", 0)) >= min_rank]
                if records:
                    molblocks = [mol_to_molblock(rec["mol"]) for rec in records]
                    df_struct = pd.DataFrame(
                        {
                            "smiles": [rec["smiles"] for rec in records],
                            "conformer_id": [rec["conformer_id"] for rec in records],
                            "conformer_rank": [rec.get("conformer_rank", 0) for rec in records],
                            "energy": [rec.get("energy") for rec in records],
                            "energy_method": [rec.get("energy_method") for rec in records],
                            "molblock": molblocks,
                            "conformer": f"ETKDGv3+MMFF/UFF ({args.num_conformers})",
                        }
                    )
                    writer.write(df_struct)
                    total_written += len(df_struct)

                total_failed += len(failures)
                _append_failures(failure_path, failures)
                pbar.update(len(batch_smiles))

                if args.log_every and batch_idx % args.log_every == 0:
                    done = min(start + args.smiles_batch_size, len(smiles))
                    pbar.write(
                        f"[INFO] Batch {batch_idx} | SMILES {done}/{len(smiles)} | "
                        f"written {total_written} | failed {total_failed}"
                    )
        except KeyboardInterrupt:
            interrupted = True
            pbar.write("[WARN] Interrupted; closing writer with partial results.")
        finally:
            pbar.close()
            writer.close()

        print(
            json.dumps(
                {
                    "unique_smiles_total": total_smiles_all,
                    "unique_smiles_shard_total": total_smiles_shard,
                    "unique_smiles_remaining": len(smiles),
                    "skipped_smiles": skipped_smiles,
                    "structures_written": total_written,
                    "failed_smiles": total_failed,
                    "structures_out": str(structures_out),
                    "resume": args.resume,
                    "interrupted": interrupted,
                    "num_conformers": args.num_conformers,
                    "prune_rms_thresh": args.prune_rms_thresh,
                    "num_workers": _resolve_num_workers(args.num_workers),
                    "mp_context": args.mp_context,
                    "mp_chunksize": args.mp_chunksize,
                    "num_shards": args.num_shards,
                    "shard_index": args.shard_index,
                },
                indent=2,
            )
        )
        return 0

    if args.structures_in:
        load_limit = 0 if args.num_shards > 1 else args.limit
        smiles = load_smiles_from_structures(args.structures_in, load_limit)
        smiles_source = "structures_dataset"
    else:
        load_limit = 0 if args.num_shards > 1 else args.limit
        smiles = load_smiles(args.input_parquet, args.smiles_col, load_limit)
        smiles_source = "input_parquet"

    total_smiles_all = len(smiles)
    smiles = _apply_shard(smiles, args.num_shards, args.shard_index)
    if args.limit and args.limit > 0:
        smiles = smiles[: args.limit]
    total_smiles_shard = len(smiles)
    if total_smiles_all == 0:
        raise RuntimeError("No SMILES loaded from input source.")

    skipped_smiles = 0
    if args.skip_existing_from:
        existing = load_existing_smiles(args.skip_existing_from)
        if existing:
            before = len(smiles)
            smiles = [smi for smi in smiles if smi not in existing]
            skipped_smiles += before - len(smiles)
    if args.resume and args.output_parquet.exists():
        existing = load_existing_smiles(args.output_parquet)
        if existing:
            before = len(smiles)
            smiles = [smi for smi in smiles if smi not in existing]
            skipped_smiles += before - len(smiles)
    if args.resume and args.failed_smiles_path and args.failed_smiles_path.exists():
        failed = load_failed_smiles(args.failed_smiles_path)
        if failed:
            before = len(smiles)
            smiles = [smi for smi in smiles if smi not in failed]
            skipped_smiles += before - len(smiles)

    device = _resolve_device(args.device, args.model_dir)
    output_scale, output_bias, output_affine_source = resolve_output_affine(
        args.model_dir,
        weights_path,
        args.output_scale,
        args.output_bias,
    )
    print(
        "output_affine "
        f"scale={output_scale:.6f} bias={output_bias:.6f} source={output_affine_source}"
    )

    artifacts = None
    if not args.dry_run:
        artifacts = load_artifacts(
            args.model_dir,
            weights_path,
            preprocessor_path,
            device,
            output_scale=output_scale,
            output_bias=output_bias,
            output_affine_source=output_affine_source,
        )
    model_output_dim = int(getattr(artifacts, "output_dim", 1)) if artifacts is not None else 1

    metadata = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "source": "CASCADE-2.0",
        "mode": "predict",
        "input_parquet": str(args.input_parquet),
        "structures_in": str(args.structures_in) if args.structures_in else "",
        "smiles_source": smiles_source,
        "smiles_col": args.smiles_col,
        "unique_smiles_total": total_smiles_all,
        "unique_smiles_shard_total": total_smiles_shard,
        "unique_smiles_remaining": len(smiles),
        "skipped_smiles": skipped_smiles,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "skip_existing_from": str(args.skip_existing_from) if args.skip_existing_from else "",
        "model_dir": str(args.model_dir),
        "weights": str(weights_path),
        "preprocessor": str(preprocessor_path),
        "output_scale": output_scale,
        "output_bias": output_bias,
        "output_affine_source": output_affine_source,
        "model_output_dim": model_output_dim,
        "max_embed_tries": args.max_embed_tries,
        "num_conformers": args.num_conformers,
        "prune_rms_thresh": args.prune_rms_thresh,
        "ff_max_iters": args.ff_max_iters,
        "adaptive_conformers": args.adaptive_conformers,
        "skip_heavy_atoms_gt": args.skip_heavy_atoms_gt,
        "embed_stall_timeout_s": args.embed_stall_timeout_s,
        "num_workers": _resolve_num_workers(args.num_workers),
        "mp_context": args.mp_context,
        "mp_chunksize": args.mp_chunksize,
        "smiles_batch_size": args.smiles_batch_size,
        "batch_size": args.batch_size,
        "device": device,
        "conformer": f"ETKDGv3+MMFF/UFF ({args.num_conformers} conformers)",
        "conformer_rank": args.conformer_rank,
        "use_all_conformers": args.use_all_conformers,
        "dry_run": args.dry_run,
        "resume": args.resume,
        "output_format": "parquet_dataset",
    }
    writer = ParquetDatasetWriter(
        args.output_parquet, metadata=metadata, overwrite=args.overwrite, resume=args.resume
    )
    failure_path = _init_failure_log(args.failed_smiles_path, not args.no_write_failures, args.resume)

    total_pred_rows = 0
    total_failed = 0
    total_no_carbon = 0

    if not smiles:
        print("[INFO] No new SMILES to process after resume filtering.")
        writer.close()
        return 0

    total_smiles = len(smiles)
    pbar = tqdm(total=total_smiles, unit="smiles")
    interrupted = False
    try:
        if args.structures_in:
            import pyarrow.dataset as ds  # noqa: WPS433

            allowed = set(smiles)
            use_all = args.use_all_conformers
            target_rank = args.conformer_rank
            batch_mols: List[Mol] = []
            batch_smiles: List[str] = []
            batch_conf_id: List[int | None] = []
            batch_conf_rank: List[int | None] = []
            batch_energy: List[float | None] = []
            batch_energy_method: List[str | None] = []
            batch_failures: List[str] = []
            batch_idx = 0
            dataset = ds.dataset(str(args.structures_in), format="parquet", exclude_invalid_files=True)
            available_cols = set(dataset.schema.names)
            cols = ["smiles", "molblock"]
            if "conformer_rank" in available_cols:
                cols.append("conformer_rank")
            if "conformer_id" in available_cols:
                cols.append("conformer_id")
            if "energy" in available_cols:
                cols.append("energy")
            if "energy_method" in available_cols:
                cols.append("energy_method")
            seen_smiles: set[str] = set()
            for batch in dataset.to_batches(columns=cols):
                data = batch.to_pydict()
                smiles_col = data.get("smiles", [])
                molblock_col = data.get("molblock", [])
                rank_col = data.get("conformer_rank")
                id_col = data.get("conformer_id")
                energy_col = data.get("energy")
                method_col = data.get("energy_method")
                for idx, (smi, block) in enumerate(zip(smiles_col, molblock_col)):
                    if smi is None or smi not in allowed:
                        continue
                    rank_val = rank_col[idx] if rank_col is not None else None
                    if not use_all:
                        if rank_col is not None:
                            if rank_val != target_rank:
                                continue
                        else:
                            if smi in seen_smiles:
                                continue
                            seen_smiles.add(smi)
                    mol = mol_from_molblock(block)
                    if mol is None:
                        batch_failures.append(f"molblock\t{smi}")
                        pbar.update(1)
                        continue
                    batch_mols.append(mol)
                    batch_smiles.append(smi)
                    batch_conf_rank.append(rank_val)
                    batch_conf_id.append(id_col[idx] if id_col is not None else None)
                    batch_energy.append(energy_col[idx] if energy_col is not None else None)
                    batch_energy_method.append(method_col[idx] if method_col is not None else None)
                    pbar.update(1)

                    if len(batch_smiles) >= args.smiles_batch_size:
                        batch_idx += 1
                        extra_cols = {
                            "conformer_rank": batch_conf_rank,
                            "conformer_id": batch_conf_id,
                            "conformer_energy": batch_energy,
                            "conformer_energy_method": batch_energy_method,
                        }
                        df_batch, failures, no_carbon = predict_from_mols(
                            batch_mols,
                            batch_smiles,
                            artifacts,
                            batch_size=args.batch_size,
                            dry_run=args.dry_run,
                            extra_cols=extra_cols,
                        )
                        if not df_batch.empty:
                            df_batch["prediction_source"] = "CASCADE-2.0"
                            if use_all:
                                df_batch["conformer"] = "structures_in:all"
                            else:
                                df_batch["conformer"] = f"structures_in:rank{target_rank}"
                            writer.write(df_batch)
                            total_pred_rows += len(df_batch)

                        total_failed += len(failures) + len(batch_failures)
                        total_no_carbon += len(no_carbon)
                        _append_failures(failure_path, batch_failures)
                        _append_failures(failure_path, failures)
                        _append_failures(failure_path, no_carbon)

                        batch_mols = []
                        batch_smiles = []
                        batch_conf_rank = []
                        batch_conf_id = []
                        batch_energy = []
                        batch_energy_method = []
                        batch_failures = []

                        if args.log_every and batch_idx % args.log_every == 0:
                            pbar.write(
                                f"[INFO] Batch {batch_idx} | SMILES {pbar.n}/{total_smiles} | "
                                f"rows {total_pred_rows} | failed {total_failed} | no_carbon {total_no_carbon}"
                            )

            if batch_smiles:
                batch_idx += 1
                extra_cols = {
                    "conformer_rank": batch_conf_rank,
                    "conformer_id": batch_conf_id,
                    "conformer_energy": batch_energy,
                    "conformer_energy_method": batch_energy_method,
                }
                df_batch, failures, no_carbon = predict_from_mols(
                    batch_mols,
                    batch_smiles,
                    artifacts,
                    batch_size=args.batch_size,
                    dry_run=args.dry_run,
                    extra_cols=extra_cols,
                )
                if not df_batch.empty:
                    df_batch["prediction_source"] = "CASCADE-2.0"
                    if use_all:
                        df_batch["conformer"] = "structures_in:all"
                    else:
                        df_batch["conformer"] = f"structures_in:rank{target_rank}"
                    writer.write(df_batch)
                    total_pred_rows += len(df_batch)

                total_failed += len(failures) + len(batch_failures)
                total_no_carbon += len(no_carbon)
                _append_failures(failure_path, batch_failures)
                _append_failures(failure_path, failures)
                _append_failures(failure_path, no_carbon)
            elif batch_failures:
                total_failed += len(batch_failures)
                _append_failures(failure_path, batch_failures)
        else:
            for batch_idx, start in enumerate(range(0, total_smiles, args.smiles_batch_size), start=1):
                batch_smiles = smiles[start : start + args.smiles_batch_size]
                df_batch, failures, no_carbon = predict_batch(
                    batch_smiles,
                    artifacts,
                    batch_size=args.batch_size,
                    max_embed_tries=args.max_embed_tries,
                    num_conformers=args.num_conformers,
                    prune_rms_thresh=args.prune_rms_thresh,
                    ff_max_iters=args.ff_max_iters,
                    adaptive_conformers=args.adaptive_conformers,
                    skip_heavy_atoms_gt=args.skip_heavy_atoms_gt,
                    embed_stall_timeout_s=args.embed_stall_timeout_s,
                    num_workers=args.num_workers,
                    mp_context=args.mp_context,
                    mp_chunksize=args.mp_chunksize,
                    dry_run=args.dry_run,
                )
                if not df_batch.empty:
                    df_batch["prediction_source"] = "CASCADE-2.0"
                    df_batch["conformer"] = f"ETKDGv3+MMFF/UFF (rank0/{args.num_conformers})"
                    writer.write(df_batch)
                    total_pred_rows += len(df_batch)

                total_failed += len(failures)
                total_no_carbon += len(no_carbon)
                _append_failures(failure_path, failures)
                _append_failures(failure_path, no_carbon)
                pbar.update(len(batch_smiles))

                if args.log_every and batch_idx % args.log_every == 0:
                    done = min(start + args.smiles_batch_size, total_smiles)
                    pbar.write(
                        f"[INFO] Batch {batch_idx} | SMILES {done}/{total_smiles} | "
                        f"rows {total_pred_rows} | failed {total_failed} | no_carbon {total_no_carbon}"
                    )
    except KeyboardInterrupt:
        interrupted = True
        pbar.write("[WARN] Interrupted; closing writer with partial results.")
    finally:
        pbar.close()
        writer.close()

    print(
        json.dumps(
            {
                "unique_smiles_total": total_smiles_all,
                "unique_smiles_shard_total": total_smiles_shard,
                "unique_smiles_remaining": total_smiles,
                "skipped_smiles": skipped_smiles,
                "predicted_rows": total_pred_rows,
                "failed_smiles": total_failed,
                "no_carbon_smiles": total_no_carbon,
                "output_parquet": str(args.output_parquet),
                "structures_in": str(args.structures_in) if args.structures_in else "",
                "device": device,
                "model_output_dim": model_output_dim,
                "dry_run": args.dry_run,
                "resume": args.resume,
                "interrupted": interrupted,
                "num_conformers": args.num_conformers,
                "prune_rms_thresh": args.prune_rms_thresh,
                "conformer_rank": args.conformer_rank,
                "use_all_conformers": args.use_all_conformers,
                "num_workers": _resolve_num_workers(args.num_workers),
                "mp_context": args.mp_context,
                "mp_chunksize": args.mp_chunksize,
                "num_shards": args.num_shards,
                "shard_index": args.shard_index,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
