"""NMR spectrum encoder: peak list → embedding via RBF + CNN + Transformer.

Converts variable-length peak lists into a fixed-dimensional spectrum
embedding (z_spec) and per-peak token features (token_memory) for
conditioning graph diffusion models.

Architecture:
  - Per-peak RBF expansion of chemical shifts (separate banks for 13C/1H)
  - Learned embeddings for nucleus type and multiplicity
  - Intensity normalization per spectrum
  - 3-layer 1D CNN for local spectral context
  - 4-layer Transformer encoder for global set interactions
  - Sinusoidal positional encoding on shift-sorted peaks
  - Mean pooling over valid peaks → spectrum-level embedding
"""

import math

import torch
import torch.nn as nn
# --- Vocabulary encodings ---

NUCLEUS_VOCAB = {"13C": 0, "1H": 1}
N_NUCLEI = len(NUCLEUS_VOCAB)

MULTIPLICITY_VOCAB = {
    "S": 0, "D": 1, "T": 2, "Q": 3, "M": 4,
    "dd": 5, "dt": 6, "td": 7, "sept": 8, "unknown": 9,
}
N_MULT = len(MULTIPLICITY_VOCAB)

# --- RBF configuration ---

C13_RANGE = (-20.0, 240.0)
H1_RANGE = (-2.0, 14.0)


class NMREncoder(nn.Module):
    """Encode NMR peak lists into fixed-dimensional spectrum embeddings.

    Input: variable-length peak lists (shifts, nucleus types, multiplicities,
           intensities) with padding mask.
    Output: z_spec (B, d_model) pooled embedding + token_memory (B, P, d_model)
            per-peak features for downstream cross-attention.

    Parameters
    ----------
    d_model : int
        Model dimension throughout the network (default 256).
    n_heads : int
        Number of attention heads in the Transformer encoder (default 8).
    n_layers : int
        Number of Transformer encoder layers (default 4).
    n_rbf : int
        Number of RBF Gaussian centers per nucleus bank (default 256).
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        n_rbf: int = 256,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_rbf = n_rbf

        # ------------------------------------------------------------------
        # RBF banks — registered as buffers so they move to the correct device
        # ------------------------------------------------------------------
        self._init_rbf_banks(n_rbf)

        # ------------------------------------------------------------------
        # Learned categorical embeddings
        # ------------------------------------------------------------------
        self.nucleus_embed = nn.Embedding(N_NUCLEI, 16)
        self.multiplicity_embed = nn.Embedding(N_MULT, 16)

        # Per-peak projection: RBF(256) + nucleus(16) + multiplicity(16) + intensity(1) → d_model
        concat_dim = n_rbf + 16 + 16 + 1
        self.peak_proj = nn.Linear(concat_dim, d_model)

        # ------------------------------------------------------------------
        # 3-layer 1D CNN over the peak sequence (B, d_model, P)
        # Layers exposed individually so we can zero-mask padding after
        # each activation — prevents padded positions from influencing
        # valid peaks through the convolution neighbourhood.
        # ------------------------------------------------------------------
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(d_model)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(d_model)

        # ------------------------------------------------------------------
        # Transformer encoder
        # ------------------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=1024,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _init_rbf_banks(self, n_rbf: int) -> None:
        """Register Gaussian RBF centres and widths for 13C and 1H."""
        c_lo, c_hi = C13_RANGE
        h_lo, h_hi = H1_RANGE

        self.register_buffer(
            "c13_centers", torch.linspace(c_lo, c_hi, n_rbf)
        )
        self.register_buffer(
            "h1_centers", torch.linspace(h_lo, h_hi, n_rbf)
        )

        c13_sigma = (c_hi - c_lo) / (n_rbf - 1)
        h1_sigma = (h_hi - h_lo) / (n_rbf - 1)
        self.register_buffer(
            "c13_sigma", torch.tensor(c13_sigma)
        )
        self.register_buffer(
            "h1_sigma", torch.tensor(h1_sigma)
        )

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _rbf_encode(
        self,
        shifts: torch.Tensor,       # (B, P)
        nucleus_types: torch.Tensor,  # (B, P) long
    ) -> torch.Tensor:               # (B, P, n_rbf)
        """Expand scalar shifts into Gaussian RBF vectors.

        Performs separate expansion for 13C and 1H peaks using their
        respective center/sigma banks. Masked (padding) positions
        produce a zero RBF vector.
        """
        device = shifts.device
        B, P = shifts.shape
        rbf = torch.zeros(B, P, self.n_rbf, device=device)

        # 13C peaks
        c_mask = nucleus_types == 0
        if c_mask.any():
            c_shifts = shifts[c_mask]                     # (N_c,)
            diff = c_shifts.unsqueeze(-1) - self.c13_centers.to(device)
            rbf[c_mask] = torch.exp(
                -0.5 * (diff / self.c13_sigma.to(device)) ** 2
            )

        # 1H peaks
        h_mask = nucleus_types == 1
        if h_mask.any():
            h_shifts = shifts[h_mask]
            diff = h_shifts.unsqueeze(-1) - self.h1_centers.to(device)
            rbf[h_mask] = torch.exp(
                -0.5 * (diff / self.h1_sigma.to(device)) ** 2
            )

        return rbf

    @staticmethod
    def _normalize_intensity(
        intensities: torch.Tensor,  # (B, P)
        mask: torch.Tensor,         # (B, P) bool
    ) -> torch.Tensor:              # (B, P)
        """Min-max normalise intensities per spectrum to [0, 1].

        Uses +/-inf for masked positions so they never become the
        extremum.  Spectra with zero valid peaks produce all zeros.
        """
        if not mask.any():
            return torch.zeros_like(intensities)

        i_max = intensities.masked_fill(~mask, float("-inf"))
        i_max = i_max.max(dim=-1, keepdim=True).values        # (B, 1)
        i_min = intensities.masked_fill(~mask, float("inf"))
        i_min = i_min.min(dim=-1, keepdim=True).values        # (B, 1)

        denom = (i_max - i_min).clamp(min=1e-8)
        normalized = (intensities - i_min) / denom
        return normalized * mask.float()

    def _positional_encoding(
        self, P: int, device: torch.device
    ) -> torch.Tensor:
        """Sinusoidal positional encoding, max_len=128 (supports up to that).

        Returns (1, P, d_model) so it broadcasts across the batch dimension.
        """
        position = torch.arange(P, device=device, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device, dtype=torch.float)
            * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(P, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, P, d_model)

    # ------------------------------------------------------------------
    # Canonical ordering
    # ------------------------------------------------------------------

    @staticmethod
    def _sort_by_shift(
        shifts: torch.Tensor,
        nucleus_types: torch.Tensor,
        multiplicities: torch.Tensor,
        intensities: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        """Sort peaks by chemical shift (ascending), padding at the end.

        This establishes a canonical order that is deterministic given the
        same peak list, regardless of input ordering.  When two peaks share
        the same shift, a stable sort preserves their relative order as
        presented by the caller.

        Padding positions receive a shift of +inf so they sort to the tail
        of every batch element.
        """
        sort_key = shifts.masked_fill(~mask, float("inf"))
        sort_key, sort_indices = sort_key.sort(dim=-1, stable=True)

        B = shifts.shape[0]
        batch_idx = torch.arange(B, device=shifts.device).unsqueeze(-1)

        sorted_shifts = shifts[batch_idx, sort_indices]
        sorted_nucleus = nucleus_types[batch_idx, sort_indices]
        sorted_mult = multiplicities[batch_idx, sort_indices]
        sorted_intensity = intensities[batch_idx, sort_indices]
        sorted_mask = mask[batch_idx, sort_indices]

        return (
            sorted_shifts,
            sorted_nucleus,
            sorted_mult,
            sorted_intensity,
            sorted_mask,
        )

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        shifts: torch.Tensor,         # (B, P)
        nucleus_types: torch.Tensor,  # (B, P) long  {0, 1}
        multiplicities: torch.Tensor, # (B, P) long  [0, 9]
        intensities: torch.Tensor,    # (B, P)
        mask: torch.Tensor,           # (B, P) bool, True = valid peak
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a batch of peak lists into spectrum embeddings.

        Returns
        -------
        z_spec : (B, d_model)
            Mean-pooled spectrum embedding (permutation-invariant).
        token_memory : (B, P, d_model)
            Per-peak token features for downstream cross-attention.
        """
        B, P = shifts.shape

        # 1. Canonical order: sort by shift so CNN windows are meaningful.
        (
            shifts,
            nucleus_types,
            multiplicities,
            intensities,
            mask,
        ) = self._sort_by_shift(
            shifts, nucleus_types, multiplicities, intensities, mask
        )

        # 2. Per-peak features
        rbf = self._rbf_encode(shifts, nucleus_types)                 # (B, P, n_rbf)
        nuc_emb = self.nucleus_embed(nucleus_types)                   # (B, P, 16)
        mult_emb = self.multiplicity_embed(multiplicities)             # (B, P, 16)
        int_norm = self._normalize_intensity(intensities, mask)        # (B, P)
        int_unsq = int_norm.unsqueeze(-1)                              # (B, P, 1)

        peak_features = torch.cat([rbf, nuc_emb, mult_emb, int_unsq], dim=-1)

        # 3. Project to d_model and zero out padding positions
        x = self.peak_proj(peak_features)          # (B, P, d_model)
        x = x * mask.unsqueeze(-1).float()         # zero-out padding

        # 4. Add sinusoidal positional encoding (max_len=128)
        pe = self._positional_encoding(P, shifts.device)
        x = x + pe                                  # (B, P, d_model)
        x = x * mask.unsqueeze(-1).float()          # re-zero padding after PE

        # 5. Mask-aware 1D CNN over the peak dimension.
        #    Zero-masking before each convolution step ensures padding
        #    positions contribute nothing to valid-peak features.
        m_1d = mask.float().unsqueeze(1)    # (B, 1, P)
        x = x.transpose(1, 2)               # (B, d_model, P)
        x = x * m_1d
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = x * m_1d                         # re-zero padding after activation
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = x * m_1d
        x = self.conv3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = x * m_1d
        x = x.transpose(1, 2)               # (B, P, d_model)

        # 6. Transformer with padding mask
        padding_mask = ~mask                 # (B, P) — True = IGNORE this position
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # 7. Mean pool over valid peaks
        valid_counts = mask.sum(dim=-1, keepdim=True).clamp(min=1).float()
        z_spec = (x * mask.unsqueeze(-1).float()).sum(dim=1) / valid_counts

        return z_spec, x
