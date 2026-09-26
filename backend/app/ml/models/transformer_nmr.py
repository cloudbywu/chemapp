"""Transformer Encoder for NMR peak sequence."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1), :]


class NMRTransformerClassifier(nn.Module):
    def __init__(self, n_classes: int, d_model: int = 128, nhead: int = 4, n_layers: int = 3):
        super().__init__()

        self.input_proj = nn.Linear(4, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=20)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(d_model * 2, n_classes),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.input_proj(x)  # (B, max_len, 4) → (B, max_len, d_model)
        x = self.pos_encoder(x)
        x = self.encoder(x, src_key_padding_mask=mask)
        if mask is not None:
            # mask follows src_key_padding_mask semantics (True = padded
            # position); pool only over valid peaks, as in models/nmr_encoder.py.
            valid = (~mask).unsqueeze(-1).to(x.dtype)  # (B, max_len, 1)
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)  # (B, d_model)
        return self.classifier(x)
