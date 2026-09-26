"""Minimal pure-PyTorch message-passing GNN with quantile heads."""

from __future__ import annotations

import torch
import torch.nn as nn


def pinball_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, tau: float) -> torch.Tensor:
    """Quantile regression loss (pinball) on masked nodes."""
    pred = pred[mask]
    target = target[mask]
    error = target - pred
    return torch.mean(torch.maximum(tau * error, (tau - 1.0) * error))


class _MessagePassingLayer(nn.Module):
    """Mean-aggregation message passing with edge features (GCN-style).

    Empirically this layer learns atom-level shift targets far better than the
    earlier edge-attention variant on the v2 dataset (see design doc 12.1).
    """

    def __init__(self, hidden: int, edge_dim: int = 10, dropout: float = 0.1):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden + edge_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        messages = self.message(
            torch.cat([h[src], h[dst], edge_attr], dim=-1)
        )
        messages = self.dropout(messages)
        agg = torch.zeros(
            h.shape[0], h.shape[1], device=h.device, dtype=messages.dtype
        )
        agg.index_add_(0, dst, messages)
        counts = torch.zeros(
            h.shape[0], 1, device=h.device, dtype=messages.dtype
        )
        counts.index_add_(0, dst, torch.ones_like(messages[:, :1]))
        agg = agg / counts.clamp(min=1.0)
        updated = self.update(torch.cat([h, agg], dim=-1))
        return self.norm(h + updated)


class ForwardGNN13C(nn.Module):
    """Predict q50/q10/q90 13C shift per carbon atom."""

    def __init__(
        self,
        in_dim: int,
        hidden: int = 256,
        n_layers: int = 6,
        edge_dim: int = 10,
        condition_dim: int = 13,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.cond_proj = nn.Linear(condition_dim, hidden)
        self.layers = nn.ModuleList(
            [
                _MessagePassingLayer(hidden, edge_dim, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(3 * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        molecule_ids: torch.Tensor | None = None,
        mol_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.in_proj(node_feats)
        if mol_condition is not None:
            condition = self.cond_proj(mol_condition)
            if molecule_ids is None:
                molecule_ids = torch.zeros(
                    h.shape[0], dtype=torch.long, device=h.device
                )
            h = h + condition.index_select(0, molecule_ids)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        if molecule_ids is None:
            molecule_ids = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        global_vec = torch.zeros(
            int(molecule_ids.max().item()) + 1,
            h.shape[1],
            device=h.device,
            dtype=h.dtype,
        )
        global_vec.index_add_(0, molecule_ids, h)
        counts = torch.zeros(
            global_vec.shape[0], 1, device=h.device, dtype=h.dtype
        )
        counts.index_add_(0, molecule_ids, torch.ones_like(h[:, :1]))
        global_vec = global_vec / counts.clamp(min=1.0)
        context = global_vec[molecule_ids]
        return self.head(
            torch.cat([h, context, self.in_proj(node_feats)], dim=-1)
        )  # (N, 3): q10, q50, q90

    def loss(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
        molecule_ids: torch.Tensor | None = None,
        mol_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.forward(
            node_feats, edge_index, edge_attr, molecule_ids, mol_condition
        )
        q10, q50, q90 = out[:, 0], out[:, 1], out[:, 2]
        loss = (
            pinball_loss(q50, targets, mask, 0.5)
            + 0.5 * pinball_loss(q10, targets, mask, 0.1)
            + 0.5 * pinball_loss(q90, targets, mask, 0.9)
        )
        return loss
