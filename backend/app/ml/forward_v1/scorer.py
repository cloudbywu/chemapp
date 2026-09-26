"""ForwardScorer-compatible local scorer for the ForwardGNN-13C model."""

from __future__ import annotations

import hashlib
import logging
import math
from pathlib import Path
import threading
from typing import Any, Iterable, Mapping, Sequence

import torch
from rdkit import Chem
from scipy.optimize import linear_sum_assignment

from .gnn import ForwardGNN13C
from .graph_data import (
    GraphSample,
    SOLVENT_VOCAB,
    _mol_condition,
    batch_graphs,
    mol_to_graph,
)


DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "pretrained" / "forward_v1"
PROVIDER_ID = "local_forward_gnn_13c_v1"
logger = logging.getLogger("chemapp.ml.forward_v1")

_WEIGHT_SHA_CACHE: dict[tuple[str, int, int], str] = {}
_WEIGHT_SHA_LOCK = threading.Lock()


def _cached_sha256_file(path: Path) -> str:
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _WEIGHT_SHA_LOCK:
        cached = _WEIGHT_SHA_CACHE.get(key)
        if cached is not None:
            return cached
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with _WEIGHT_SHA_LOCK:
        _WEIGHT_SHA_CACHE[key] = digest
    return digest


def _default_checkpoint() -> Path | None:
    configured = Path(__import__("os").environ.get(
        "CHEMAPP_FORWARD_GNN_DIR", str(DEFAULT_MODEL_DIR)
    ))
    for candidate in ("forward_gnn_13c_v1.pt", "best.pt"):
        path = configured / candidate
        if path.exists():
            return path
    return None


class LocalForwardScorer:
    """Predict atom-level 13C shifts for candidate SMILES and score a query.

    所有预测输出中的 ``atom_index`` 均为 0-based RDKit 原子索引
    （``atom.GetIdx()``），与 ``nmr_candidate_scorer_v4`` /
    ``nmr_forward`` 的强校验约定一致。
    """

    def __init__(self, checkpoint: str | Path | None = None, device: str | None = None):
        path = Path(checkpoint) if checkpoint else _default_checkpoint()
        if path is None or not path.exists():
            raise FileNotFoundError(
                "ForwardGNN-13C checkpoint not found; run "
                "python -m app.ml.forward_v1.train first"
            )
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(str(path), map_location=self.device, weights_only=True)
        self.model = ForwardGNN13C(
            in_dim=ckpt["in_dim"],
            hidden=ckpt["hidden"],
            n_layers=ckpt["layers"],
            condition_dim=ckpt.get("condition_dim", 13),
        ).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.shift_mean = float(ckpt.get("shift_mean_ppm", 0.0))
        self.shift_std = float(ckpt.get("shift_std_ppm", 1.0))
        self.model_sha256 = ""
        try:
            self.model_sha256 = _cached_sha256_file(path)
        except OSError:
            pass

    @torch.no_grad()
    def predict_molecule(
        self,
        smiles: str,
        solvent: str | None = None,
        field_mhz: float | None = None,
    ) -> list[dict[str, float | int]]:
        """Predict per-carbon 13C shifts; atom_index 为 0-based RDKit 原子索引。"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid SMILES: {smiles}")
        node_feats, edge_index, edge_attr, symbols = mol_to_graph(mol)
        node_feats = node_feats.unsqueeze(0).to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.to(self.device)
        mol_ids = torch.zeros(node_feats[0].shape[0], dtype=torch.long, device=self.device)
        condition = torch.zeros(1, len(SOLVENT_VOCAB) + 1, device=self.device)
        solvent_norm = (solvent or "").strip().lower()
        index = 0 if not solvent_norm else next(
            (
                i
                for i, key in enumerate(SOLVENT_VOCAB)
                if key in solvent_norm or solvent_norm in key
            ),
            0,
        )
        condition[0, index] = 1.0
        if field_mhz:
            condition[0, -1] = min(float(field_mhz) / 800.0, 1.0)
        out = self.model(
            node_feats[0], edge_index, edge_attr, mol_ids, condition
        )  # (N, 3)
        predictions = []
        for i, symbol in enumerate(symbols):
            if symbol != "C":
                continue
            q10, q50, q90 = out[i].tolist()
            q10 = q10 * self.shift_std + self.shift_mean
            q50 = q50 * self.shift_std + self.shift_mean
            q90 = q90 * self.shift_std + self.shift_mean
            predictions.append({
                # atom_index 为 0-based RDKit 原子索引（与 atom.GetIdx()
                # 一致）；下游消费者（如 nmr_candidate_scorer_v4）强校验
                # 该集合必须等于分子碳原子索引集合。
                "atom_index": i,
                "shift_ppm": round(float(q50), 3),
                "q10_ppm": round(float(q10), 3),
                "q90_ppm": round(float(q90), 3),
            })
        return predictions

    @torch.no_grad()
    def predict_molecules_batched(
        self,
        smiles_list: Sequence[str],
        *,
        batch_size: int = 64,
        solvent: str | None = None,
        field_mhz: float | None = None,
    ) -> list[list[dict[str, float | int]]]:
        """Predict 13C shifts for many SMILES in one batched forward pass.

        每个预测条目中的 atom_index 为 0-based RDKit 原子索引
        （见类 docstring 的约定说明）。
        """
        samples: list[GraphSample] = []
        valid_indices: list[int] = []
        condition = _mol_condition(solvent, field_mhz)
        for position, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
                mol = Chem.RemoveHs(mol, sanitize=False)
            try:
                mol_h = Chem.AddHs(Chem.Mol(mol))
                Chem.SanitizeMol(mol_h)
                node_feats, edge_index, edge_attr, symbols = mol_to_graph(mol_h)
            except Exception:
                continue
            n_total = node_feats.shape[0]
            targets = torch.zeros(n_total, dtype=torch.float32)
            mask = torch.zeros(n_total, dtype=torch.bool)
            samples.append(
                GraphSample(
                    sample_id=f"batch-{position}",
                    inchikey="",
                    scaffold="",
                    canonical_smiles=Chem.MolToSmiles(mol),
                    solvent=solvent,
                    field_mhz=field_mhz,
                    atom_symbols=tuple(symbols),
                    mol_condition=condition,
                    node_feats=node_feats,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    targets=targets,
                    target_mask=mask,
                    env_labels=torch.zeros(n_total, dtype=torch.long),
                )
            )
            valid_indices.append(position)

        results: list[list[dict[str, float]]] = [[] for _ in smiles_list]
        if not samples:
            return results
        for start in range(0, len(samples), batch_size):
            chunk = samples[start : start + batch_size]
            (
                node_feats,
                edge_index,
                edge_attr,
                _targets,
                _mask,
                molecule_ids,
                mol_condition,
                _env,
            ) = batch_graphs(chunk)
            node_feats = node_feats.to(self.device)
            edge_index = edge_index.to(self.device)
            edge_attr = edge_attr.to(self.device)
            molecule_ids = molecule_ids.to(self.device)
            mol_condition = mol_condition.to(self.device)
            out = self.model(
                node_feats, edge_index, edge_attr, molecule_ids, mol_condition
            )
            out = out.cpu()
            base = 0
            for offset, sample in enumerate(chunk):
                n_heavy = int(
                    sum(1 for symbol in sample.atom_symbols if symbol != "H")
                )
                heavy_symbols = list(sample.atom_symbols)[:n_heavy]
                predictions: list[dict[str, float]] = []
                for index, symbol in enumerate(heavy_symbols):
                    if symbol != "C":
                        continue
                    q10, q50, q90 = out[base + index].tolist()
                    q10 = q10 * self.shift_std + self.shift_mean
                    q50 = q50 * self.shift_std + self.shift_mean
                    q90 = q90 * self.shift_std + self.shift_mean
                    predictions.append(
                        {
                            # 0-based RDKit 重原子索引；AddHs 将氢原子追加
                            # 在重原子之后，因此重原子索引与原始 mol 一致。
                            "atom_index": index,
                            "shift_ppm": round(float(q50), 3),
                            "q10_ppm": round(float(q10), 3),
                            "q90_ppm": round(float(q90), 3),
                        }
                    )
                results[valid_indices[start + offset]] = predictions
                base += sample.n_atoms
        return results

    def score_candidates(
        self,
        observed_13c: Iterable[float | Mapping[str, Any]] | None,
        candidates: Sequence[Mapping[str, Any]],
        *,
        formula: str | None = None,
    ) -> dict[str, Any]:
        shifts: list[float] = []
        if observed_13c is not None:
            for item in observed_13c:
                raw = item.get("shift") if isinstance(item, Mapping) else item
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    continue
                value = float(raw)
                if math.isfinite(value) and -20.0 <= value <= 300.0:
                    shifts.append(value)
        # Preserve multiplicity: each reported peak is one observation, and
        # chemically equivalent carbons legitimately share an observed shift.
        shifts = sorted(round(s, 3) for s in shifts)
        if not shifts:
            return {
                "status": "unsupported_modality",
                "required_nucleus": "13C",
                "reason": "local_forward_gnn_requires_observed_13c",
                "model_called": False,
                "calibrated_probability": False,
                "quantile_enabled": True,
                "candidates": [],
            }

        evidence = []
        failed_evidence: list[dict[str, Any]] = []
        for candidate in candidates:
            smiles = str(candidate.get("smiles") or "")
            try:
                predicted = self.predict_molecule(smiles)
            except Exception as exc:
                logger.warning(
                    "local forward prediction failed for %r: %s", smiles, exc
                )
                failed_evidence.append(
                    {
                        "candidate_id": str(
                            candidate.get("candidate_id")
                            or candidate.get("smiles")
                        ),
                        "relative_rank": 0,
                        "assignment_mode": "none_prediction_failed",
                        "diagnostic_only": True,
                        "prediction_failed": True,
                        "matched_count": 0,
                        "observed_count": len(shifts),
                        "predicted_atom_count": 0,
                        "unmatched_observed_count": len(shifts),
                        "unmatched_predicted_atom_count": 0,
                        "observed_coverage": 0.0,
                        "predicted_atom_coverage": 0.0,
                        "bidirectional_coverage": 0.0,
                        "assignment_complete": False,
                        "mae_ppm": None,
                        "rmse_ppm": None,
                        "max_abs_error_ppm": None,
                        "calibrated_probability": False,
                        "quantile_enabled": True,
                        "prediction": {
                            "provider": PROVIDER_ID,
                            "model_sha256": self.model_sha256,
                            "status": "prediction_failed",
                            "reason": "local_prediction_failed",
                            "atom_predictions": [],
                        },
                    }
                )
                continue
            predicted_shifts = [float(p["shift_ppm"]) for p in predicted]
            costs = [[abs(o - p) for p in predicted_shifts] for o in shifts]
            rows, cols = linear_sum_assignment(costs)
            errors = [costs[r][c] for r, c in zip(rows, cols)]
            mae = sum(errors) / max(len(errors), 1)
            rmse = math.sqrt(sum(e * e for e in errors) / max(len(errors), 1))
            observed_coverage = len(errors) / max(len(shifts), 1)
            predicted_coverage = len(errors) / max(len(predicted_shifts), 1)
            evidence.append({
                "candidate_id": str(candidate.get("candidate_id") or candidate.get("smiles")),
                "relative_rank": 0,
                "assignment_mode": "unassigned_hungarian_atom_level",
                "diagnostic_only": True,
                "matched_count": len(errors),
                "observed_count": len(shifts),
                "predicted_atom_count": len(predicted_shifts),
                "unmatched_observed_count": len(shifts) - len(errors),
                "unmatched_predicted_atom_count": len(predicted_shifts) - len(errors),
                "observed_coverage": round(observed_coverage, 4),
                "predicted_atom_coverage": round(predicted_coverage, 4),
                "bidirectional_coverage": round(min(observed_coverage, predicted_coverage), 4),
                "assignment_complete": (
                    len(shifts) - len(errors) == 0 and len(predicted_shifts) - len(errors) == 0
                ),
                "mae_ppm": round(mae, 3),
                "rmse_ppm": round(rmse, 3),
                "max_abs_error_ppm": round(max(errors), 3) if errors else None,
                "calibrated_probability": False,
                "quantile_enabled": True,
                "prediction": {
                    "provider": PROVIDER_ID,
                    "model_sha256": self.model_sha256,
                    "atom_predictions": predicted,
                },
            })

        ordered = sorted(
            [*evidence, *failed_evidence],
            key=lambda item: (
                bool(item.get("prediction_failed")),
                -float(item["bidirectional_coverage"]),
                int(item["unmatched_observed_count"])
                + int(item["unmatched_predicted_atom_count"]),
                float(item["mae_ppm"] or float("inf")),
                str(item["candidate_id"]),
            ),
        )
        for rank, item in enumerate(ordered, start=1):
            item["relative_rank"] = rank
        return {
            "status": "ok",
            "nucleus": "13C",
            "evidence_kind": "relative_13c_forward_evidence",
            "diagnostic_only": True,
            "rank_basis": (
                "higher_bidirectional_coverage_then_fewer_unmatched_then_"
                "lower_hungarian_mae_rmse"
            ),
            "observed_shifts_ppm": shifts,
            "calibrated_probability": False,
            "quantile_enabled": True,
            "model": {"provider": PROVIDER_ID, "sha256": self.model_sha256},
            "runtime": {"device": str(self.device)},
            "candidates": ordered,
        }
