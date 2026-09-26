from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class PaiNNConfig:
    input_dim: int = 256
    units: int = 256
    num_radial: int = 20
    cutoff: float = 5.0
    depth: int = 6
    output_dim: int = 1
    equivariant_dim: int = 3
    envelope_exponent: int = 5
    solvent_vocab_size: int = 0
    solvent_emb_dim: int = 0
    solvent_use_bias: bool = False
    solvent_adapter_hidden_dim: int = 0
    solvent_adapter_dropout: float = 0.0


def _scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out_shape = (dim_size,) + src.shape[1:]
    out = torch.zeros(out_shape, device=src.device, dtype=src.dtype)
    if index.numel() == 0:
        return out
    out.index_add_(0, index, src)
    return out


class BesselBasisLayer(nn.Module):
    def __init__(self, num_radial: int, cutoff: float, envelope_exponent: int = 5) -> None:
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.inv_cutoff = 1.0 / cutoff
        self.envelope_exponent = envelope_exponent
        frequencies = math.pi * torch.arange(1, num_radial + 1, dtype=torch.float32)
        self.frequencies = nn.Parameter(frequencies)

    def envelope(self, inputs: torch.Tensor) -> torch.Tensor:
        p = self.envelope_exponent + 1
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2
        env_val = 1.0 / inputs + a * inputs ** (p - 1) + b * inputs ** p + c * inputs ** (p + 1)
        return torch.where(inputs < 1, env_val, torch.zeros_like(inputs))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        d_scaled = distances * self.inv_cutoff
        d_cutoff = self.envelope(d_scaled)
        return d_cutoff * torch.sin(self.frequencies * d_scaled)


class CosCutOffEnvelope(nn.Module):
    def __init__(self, cutoff: float | None) -> None:
        super().__init__()
        self.cutoff = float(abs(cutoff)) if cutoff is not None else 1e8

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        fc = distances.clamp(-self.cutoff, self.cutoff)
        return (torch.cos(fc * math.pi / self.cutoff) + 1.0) * 0.5


class EquivariantInitialize(nn.Module):
    def __init__(self, dim: int = 3, method: str = "zeros") -> None:
        super().__init__()
        self.dim = dim
        self.method = method

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        n_nodes, n_features = inputs.shape
        if self.method == "zeros":
            value = 0.0
        elif self.method == "eps":
            value = torch.finfo(inputs.dtype).eps
        elif self.method == "ones":
            value = 1.0
        else:
            raise ValueError(f"Unsupported equivariant init method: {self.method}")
        return torch.full(
            (n_nodes, self.dim, n_features),
            value,
            device=inputs.device,
            dtype=inputs.dtype,
        )


class PaiNNConv(nn.Module):
    def __init__(self, units: int, num_radial: int, cutoff: float | None) -> None:
        super().__init__()
        self.units = units
        self.cutoff = cutoff
        self.dense_s = nn.Linear(units, units, bias=True)
        self.dense_phi = nn.Linear(units, units * 3, bias=True)
        self.dense_w = nn.Linear(num_radial, units * 3, bias=True)
        self.activation = nn.SiLU()

    def forward(
        self,
        node: torch.Tensor,
        equivariant: torch.Tensor,
        rbf: torch.Tensor,
        envelope: torch.Tensor,
        r_ij: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        n_nodes = node.shape[0]
        s = self.activation(self.dense_s(node))
        s = self.dense_phi(s)
        s = s[edge_index[:, 1]]
        w = self.dense_w(rbf)
        if self.cutoff is not None:
            w = w * envelope
        sw = s * w
        sw1, sw2, sw3 = sw.chunk(3, dim=-1)
        ds = _scatter_sum(sw1, edge_index[:, 0], n_nodes)

        vj = equivariant[edge_index[:, 1]]
        dv1 = vj * sw2.unsqueeze(1)
        dv2 = r_ij.unsqueeze(-1) * sw3.unsqueeze(1)
        dv = dv1 + dv2
        dv = _scatter_sum(dv, edge_index[:, 0], n_nodes)
        return ds, dv


class PaiNNUpdate(nn.Module):
    def __init__(self, units: int) -> None:
        super().__init__()
        self.units = units
        self.dense1 = nn.Linear(units * 2, units, bias=True)
        self.lin_u = nn.Linear(units, units, bias=False)
        self.lin_v = nn.Linear(units, units, bias=False)
        self.dense_a = nn.Linear(units, units * 3, bias=True)
        self.activation = nn.SiLU()

    def forward(self, node: torch.Tensor, equivariant: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v_v = self.lin_v(equivariant)
        v_u = self.lin_u(equivariant)
        v_prod = (v_u * v_v).sum(dim=1)
        v_norm = torch.linalg.norm(v_v, dim=1)
        a = torch.cat([node, v_norm], dim=-1)
        a = self.activation(self.dense1(a))
        a = self.dense_a(a)
        a_vv, a_sv, a_ss = a.chunk(3, dim=-1)
        dv = v_u * a_vv.unsqueeze(1)
        ds = v_prod * a_sv + a_ss
        return ds, dv


class PoolingNodes(nn.Module):
    def forward(
        self, node_features: torch.Tensor, atom_index: torch.Tensor, num_targets: int
    ) -> torch.Tensor:
        if num_targets == 0:
            return node_features.new_zeros((0, node_features.shape[-1]))
        mask = atom_index >= 0
        target_index = atom_index[mask]
        out = node_features.new_zeros((num_targets, node_features.shape[-1]))
        if target_index.numel() == 0:
            return out
        out.index_add_(0, target_index, node_features[mask])
        counts = node_features.new_zeros((num_targets, 1))
        ones = torch.ones((target_index.shape[0], 1), device=node_features.device, dtype=node_features.dtype)
        counts.index_add_(0, target_index, ones)
        return out / counts.clamp(min=1.0)


class PaiNNModel(nn.Module):
    def __init__(self, config: PaiNNConfig) -> None:
        super().__init__()
        self.config = config
        self.solvent_vocab_size = int(config.solvent_vocab_size)
        self.output_dim = int(config.output_dim)
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be >= 1, got {self.output_dim}")
        self.embedding = nn.Embedding(config.input_dim, config.units)
        self.equiv_init = EquivariantInitialize(dim=config.equivariant_dim, method="eps")
        self.bessel_basis = BesselBasisLayer(
            num_radial=config.num_radial,
            cutoff=config.cutoff,
            envelope_exponent=config.envelope_exponent,
        )
        self.cutoff = CosCutOffEnvelope(cutoff=None)
        self.convs = nn.ModuleList(
            [PaiNNConv(config.units, config.num_radial, cutoff=None) for _ in range(config.depth)]
        )
        self.updates = nn.ModuleList([PaiNNUpdate(config.units) for _ in range(config.depth)])
        self.pool = PoolingNodes()
        self.mlp = nn.Sequential(
            nn.Linear(config.units, config.units),
            nn.SiLU(),
            nn.Linear(config.units, self.output_dim),
        )
        solvent_emb_dim = int(config.solvent_emb_dim)
        solvent_hidden_dim = int(config.solvent_adapter_hidden_dim)
        self.solvent_use_bias = bool(config.solvent_use_bias and self.solvent_vocab_size > 0)
        self.solvent_embedding = None
        self.solvent_bias = None
        self.solvent_adapter = None

        if self.solvent_vocab_size > 0 and solvent_emb_dim > 0:
            self.solvent_embedding = nn.Embedding(self.solvent_vocab_size, solvent_emb_dim)
        if self.solvent_use_bias:
            self.solvent_bias = nn.Embedding(self.solvent_vocab_size, self.output_dim)
            nn.init.zeros_(self.solvent_bias.weight)
        if solvent_hidden_dim > 0:
            if self.solvent_embedding is None:
                raise ValueError(
                    "solvent_adapter_hidden_dim > 0 requires solvent_emb_dim > 0 and solvent_vocab_size > 0"
                )
            self.solvent_adapter = nn.Sequential(
                nn.Linear(config.units + solvent_emb_dim, solvent_hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(config.solvent_adapter_dropout)),
                nn.Linear(solvent_hidden_dim, self.output_dim),
            )
            # Keep warm-start behavior identical at init; adapter learns deltas during finetune.
            final_linear = self.solvent_adapter[-1]
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def _prepare_solvent_ids(
        self,
        solvent_ids: Optional[torch.Tensor],
        *,
        num_targets: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if solvent_ids is None or self.solvent_vocab_size <= 0:
            return None
        solvent_ids = solvent_ids.view(-1).to(device=device, dtype=torch.long)
        if solvent_ids.shape[0] == 1 and num_targets > 1:
            solvent_ids = solvent_ids.expand(num_targets)
        if int(solvent_ids.shape[0]) != int(num_targets):
            raise RuntimeError(
                f"solvent_ids length mismatch: got {int(solvent_ids.shape[0])}, expected {int(num_targets)}"
            )
        return solvent_ids.clamp_(min=0, max=max(0, self.solvent_vocab_size - 1))

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_features: bool = False,
        solvent_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        node_attr = batch["node_attributes"].long()
        pos = batch["node_coordinates"]
        edge_index = batch["edge_indices"].long()
        atom_index = batch["atom_index"].long()
        num_targets = int(batch["num_targets"])

        z = self.embedding(node_attr)
        v = self.equiv_init(z)

        pos_i = pos[edge_index[:, 0]]
        pos_j = pos[edge_index[:, 1]]
        rij = pos_i - pos_j
        dist = torch.linalg.norm(rij, dim=-1, keepdim=True)
        inv_dist = torch.where(dist > 0, 1.0 / dist, torch.zeros_like(dist))
        rij_norm = rij * inv_dist

        rbf = self.bessel_basis(dist)
        env = self.cutoff(dist)

        for conv, update in zip(self.convs, self.updates):
            ds, dv = conv(z, v, rbf, env, rij_norm, edge_index)
            z = z + ds
            v = v + dv
            ds, dv = update(z, v)
            z = z + ds
            v = v + dv

        pooled = self.pool(z, atom_index, num_targets)
        if return_features:
            return pooled
        pred = self.mlp(pooled)
        solvent_ids_t = self._prepare_solvent_ids(
            solvent_ids,
            num_targets=int(num_targets),
            device=pred.device,
        )
        if solvent_ids_t is not None and self.solvent_bias is not None:
            pred = pred + self.solvent_bias(solvent_ids_t)
        if solvent_ids_t is not None and self.solvent_adapter is not None:
            solvent_emb = self.solvent_embedding(solvent_ids_t)
            pred = pred + self.solvent_adapter(torch.cat([pooled, solvent_emb], dim=-1))
        return pred


class PaiNNGPModel(nn.Module):
    def __init__(self, config: PaiNNConfig, inducing_points: np.ndarray) -> None:
        super().__init__()
        import gpytorch

        self.backbone = PaiNNModel(config)
        inducing = torch.as_tensor(inducing_points, dtype=self.backbone.embedding.weight.dtype)
        class _GPHead(gpytorch.models.ApproximateGP):
            def __init__(self, points: torch.Tensor) -> None:
                num_inducing = points.shape[0]
                variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(num_inducing)
                variational_strategy = gpytorch.variational.VariationalStrategy(
                    self, points, variational_distribution, learn_inducing_locations=True
                )
                super().__init__(variational_strategy)
                self.mean_module = gpytorch.means.ZeroMean()
                self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=1.5))

            def forward(self, x: torch.Tensor):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

        self.gp = _GPHead(inducing)
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.likelihood.noise = torch.tensor(0.5, dtype=self.backbone.embedding.weight.dtype)

    def forward(self, batch: Dict[str, torch.Tensor]):
        features = self.backbone(batch, return_features=True)
        return self.gp(features)
