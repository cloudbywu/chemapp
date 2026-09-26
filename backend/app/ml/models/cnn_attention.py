"""1D CNN + Multi-Head Attention model for NMR binned spectrum."""

import torch
import torch.nn as nn


class CNNAttentionClassifier(nn.Module):
    def __init__(self, n_classes: int, input_dim: int = 143, hidden_dim: int = 256):
        super().__init__()

        self.input_dim = input_dim

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=8, batch_first=True, dropout=0.1,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        classifier_in = 128 + input_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim) — clip to configured size
        if x.shape[1] > self.input_dim:
            x = x[:, :self.input_dim]
        elif x.shape[1] < self.input_dim:
            pad = torch.zeros(x.shape[0], self.input_dim - x.shape[1], device=x.device, dtype=x.dtype)
            x = torch.cat([x, pad], dim=1)

        x_conv = x.unsqueeze(1)  # (B, 1, input_dim)
        x_conv = self.conv(x_conv)  # (B, 128, input_dim)
        x_conv = x_conv.transpose(1, 2)  # (B, input_dim, 128)
        x_attn, _ = self.attention(x_conv, x_conv, x_conv)
        x_attn = x_attn.transpose(1, 2)  # (B, 128, input_dim)
        x_pool = self.pool(x_attn).squeeze(-1)  # (B, 128)

        combined = torch.cat([x_pool, x], dim=-1)
        return self.classifier(combined)
