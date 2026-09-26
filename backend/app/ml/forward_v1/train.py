"""Train the ForwardGNN-13C prototype on a scaffold-split v2 dataset.

Usage (from backend/):
    python -m app.ml.forward_v1.train --max-molecules 1200 --epochs 3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from .gnn import ForwardGNN13C
from .graph_data import (
    GraphSample,
    batch_graphs,
    build_atom_dataset,
    build_exp22k_dataset,
    scaffold_group_split,
)


class GraphListDataset(Dataset):
    def __init__(self, samples: list[GraphSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GraphSample:
        return self.samples[idx]


def _collate(batch: list[GraphSample]):
    return batch_graphs(batch)


def _restore_best_checkpoint(
    model: ForwardGNN13C, outdir: Path, device: torch.device
) -> bool:
    """Restore the best-val checkpoint written during training, if present.

    Without this, test metrics after early stopping are computed on the
    weights of the *last* epoch, which are not the best-val weights.
    """
    ckpt_path = outdir / "forward_gnn_13c_v1.pt"
    if not ckpt_path.exists():
        return False
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    return True


@torch.no_grad()
def evaluate(
    model: ForwardGNN13C,
    loader: DataLoader,
    device: torch.device,
    shift_mean: float,
    shift_std: float,
) -> dict[str, float]:
    model.eval()
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    covered = 0
    total = 0
    env_errors: dict[int, list[float]] = defaultdict(list)
    for (
        node_feats,
        edge_index,
        edge_attr,
        targets,
        mask,
        molecule_ids,
        mol_condition,
        env_labels,
    ) in loader:
        node_feats = node_feats.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        targets = targets.to(device)
        mask = mask.to(device)
        molecule_ids = molecule_ids.to(device)
        mol_condition = mol_condition.to(device)
        env_labels = env_labels.to(device)
        out = model(node_feats, edge_index, edge_attr, molecule_ids, mol_condition)
        q10, q50, q90 = out[:, 0], out[:, 1], out[:, 2]
        q10 = q10 * shift_std + shift_mean
        q50 = q50 * shift_std + shift_mean
        q90 = q90 * shift_std + shift_mean
        m = mask
        errors = (q50[m] - targets[m]).abs()
        abs_errors.extend(errors.tolist())
        sq_errors.extend(((q50[m] - targets[m]) ** 2).tolist())
        covered += int(((targets[m] >= q10[m]) & (targets[m] <= q90[m])).sum().item())
        total += int(m.sum().item())
        for env in torch.unique(env_labels[m]).tolist():
            env_mask = m & (env_labels == env)
            env_errors[env].extend(
                (q50[env_mask] - targets[env_mask]).abs().tolist()
            )
    mae = sum(abs_errors) / max(len(abs_errors), 1)
    rmse = math.sqrt(sum(sq_errors) / max(len(sq_errors), 1))
    env_names = {0: "sp3", 1: "sp2", 2: "aromatic", 3: "carbonyl"}
    return {
        "mae_ppm": round(float(mae), 4),
        "rmse_ppm": round(float(rmse), 4),
        "coverage_q10_q90": round(covered / max(total, 1), 4),
        "n_nodes": total,
        "env_mae_ppm": {
            env_names.get(env, f"env{env}"): round(
                sum(values) / max(len(values), 1), 4
            )
            for env, values in sorted(env_errors.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("data/nmr_spectral_index_v2.sqlite"))
    parser.add_argument("--max-molecules", type=int, default=1200)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("reports/forward_v1_smoke"))
    parser.add_argument(
        "--cache", type=Path,
        default=Path("data/derived/forward_v1_dataset_full.pt"),
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--exp22k-entries",
        type=Path,
        default=None,
        help="CSP5 Exp22K_13C_entries.pkl; switches dataset building to Exp22K.",
    )
    parser.add_argument(
        "--exp22k-splits",
        type=Path,
        default=None,
        help="CSP5-13C-scaffold-doi_split.json (entry-list indices).",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[forward_v1] device: {device}")

    t0 = time.time()
    if args.exp22k_entries is not None:
        if args.exp22k_splits is None:
            raise SystemExit("--exp22k-splits is required with --exp22k-entries")
        print(f"[forward_v1] building Exp22K dataset from {args.exp22k_entries} "
              f"(max_molecules={args.max_molecules or 'all'}) ...")
        samples, split_map = build_exp22k_dataset(
            args.exp22k_entries,
            args.exp22k_splits,
            max_molecules=args.max_molecules,
            seed=args.seed,
            cache_path=None if args.no_cache else args.cache,
        )
        train = [samples[i] for i in split_map["train"]]
        val = [samples[i] for i in split_map["val"]]
        test = [samples[i] for i in split_map["test"]]
        split_policy = "exp22k_scaffold_doi"
    else:
        print(f"[forward_v1] building dataset from {args.index} "
              f"(max_molecules={args.max_molecules or 'all'}) ...")
        samples = build_atom_dataset(
            args.index,
            max_molecules=args.max_molecules,
            seed=args.seed,
            cache_path=None if args.no_cache else args.cache,
        )
        train, val, test = scaffold_group_split(samples, seed=args.seed)
        split_policy = "bemis_murcko_scaffold"
    print(f"[forward_v1] {len(samples)} molecules in {time.time() - t0:.0f}s")

    print(f"[forward_v1] split train={len(train)} val={len(val)} test={len(test)} "
          f"({split_policy})")
    n_targets = sum(int(s.target_mask.sum().item()) for s in train)
    print(f"[forward_v1] train carbon labels: {n_targets}")

    train_loader = DataLoader(
        GraphListDataset(train), batch_size=args.batch_size,
        shuffle=True, collate_fn=_collate,
    )
    val_loader = DataLoader(
        GraphListDataset(val), batch_size=args.batch_size, shuffle=False, collate_fn=_collate,
    )
    test_loader = DataLoader(
        GraphListDataset(test), batch_size=args.batch_size, shuffle=False, collate_fn=_collate,
    )

    in_dim = samples[0].node_feats.shape[1]
    all_targets = torch.cat(
        [s.targets[s.target_mask] for s in train]
    )
    shift_mean = float(all_targets.mean().item())
    shift_std = max(float(all_targets.std().item()), 1e-6)
    print(f"[forward_v1] target stats mean={shift_mean:.2f} std={shift_std:.2f} ppm")
    model = ForwardGNN13C(in_dim=in_dim, hidden=args.hidden, n_layers=args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    def lr_lambda(epoch: int) -> float:
        if epoch < args.warmup:
            return (epoch + 1) / max(args.warmup, 1)
        progress = (epoch - args.warmup) / max(args.epochs - args.warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_val = float("inf")
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for (
            node_feats,
            edge_index,
            edge_attr,
            targets,
            mask,
            molecule_ids,
            mol_condition,
            _,
        ) in train_loader:
            node_feats = node_feats.to(device)
            edge_index = edge_index.to(device)
            edge_attr = edge_attr.to(device)
            targets = targets.to(device)
            mask = mask.to(device)
            molecule_ids = molecule_ids.to(device)
            mol_condition = mol_condition.to(device)
            optimizer.zero_grad()
            targets_norm = (targets - shift_mean) / shift_std
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = model.loss(
                    node_feats,
                    edge_index,
                    edge_attr,
                    targets_norm,
                    mask,
                    molecule_ids,
                    mol_condition,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, shift_mean, shift_std)
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(total_loss / max(n_batches, 1), 4),
                **val_metrics,
            }
        )
        print(f"[forward_v1] epoch {epoch}/{args.epochs} loss={total_loss / max(n_batches, 1):.4f} "
              f"val_mae={val_metrics['mae_ppm']:.3f} val_rmse={val_metrics['rmse_ppm']:.3f} "
              f"coverage={val_metrics['coverage_q10_q90']:.3f}")
        if val_metrics["mae_ppm"] < best_val:
            best_val = val_metrics["mae_ppm"]
            epochs_without_improvement = 0
            args.outdir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hidden": args.hidden,
                    "layers": args.layers,
                    "in_dim": in_dim,
                    "condition_dim": samples[0].mol_condition.shape[0],
                    "seed": args.seed,
                    "shift_mean_ppm": shift_mean,
                    "shift_std_ppm": shift_std,
                },
                args.outdir / "forward_gnn_13c_v1.pt",
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"[forward_v1] early stopping at epoch {epoch}")
                break

    restored = _restore_best_checkpoint(model, args.outdir, device)
    if restored:
        print("[forward_v1] restored best-val checkpoint for test evaluation")
    test_metrics = evaluate(model, test_loader, device, shift_mean, shift_std)
    print(f"[forward_v1] TEST mae={test_metrics['mae_ppm']:.3f} "
          f"rmse={test_metrics['rmse_ppm']:.3f} coverage={test_metrics['coverage_q10_q90']:.3f}")

    report = {
        "schema_version": "chemapp.forward-gnn-13c.smoke.v1",
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "dataset": {
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
            "n_train_carbon_labels": n_targets,
            "shift_mean_ppm": round(shift_mean, 3),
            "shift_std_ppm": round(shift_std, 3),
            "split": split_policy,
            "measurement_policy": (
                "exp22k_human_curated"
                if args.exp22k_entries is not None
                else "measured_and_inferred_measured"
            ),
            "dedupe_inchikey": args.exp22k_entries is None,
        },
        "history": history,
        "test": test_metrics,
        "best_val_mae": best_val,
        "test_evaluated_from": "best_checkpoint" if restored else "last_epoch",
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[forward_v1] report written to {args.outdir / 'report.json'}")


if __name__ == "__main__":
    main()
