"""Weighted ensemble of CNN+Attention and Transformer models.

Legacy prototype classifier (2026-08 ML review): the learnable fusion weight
below is an unregularised scalar knob trained on rule-synthesised demo data;
it carries no audit trail and must not be cited as calibrated evidence.
"""

from pathlib import Path

import torch
import torch.nn as nn

from app.ml.models.cnn_attention import CNNAttentionClassifier
from app.ml.models.transformer_nmr import NMRTransformerClassifier

# Keep the weights directory identical to app.ml.predictor and the training
# scripts: everything resolves to app/ml/pretrained (this file lives in
# app/ml/models/, so parents[1] is app/ml).
_WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "pretrained"

_DEVICE: torch.device | None = None


def get_device() -> torch.device:
    """Resolve the torch device lazily at call time.

    Resolving CUDA availability at import time bakes the import-order
    environment into the module; deferring keeps CPU-only contexts (tests,
    sidecars) from paying for a CUDA probe they never use.
    """
    global _DEVICE
    if _DEVICE is None:
        _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _DEVICE


class NMREnsembleClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super().__init__()
        self.cnn = CNNAttentionClassifier(n_classes)
        self.transformer = NMRTransformerClassifier(n_classes)
        # NOTE (audit limitation, 2026-08 ML review): this learnable fusion
        # weight is a single unregularised scalar blending two prototype
        # classifiers; it is not interpretable, not calibrated, and exists
        # only to preserve the archived demo training chain.
        self.cnn_weight = nn.Parameter(torch.tensor(0.5))
        self.to(get_device())

    def forward(self, feat: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        cnn_out = self.cnn(feat)
        tf_out = self.transformer(seq)
        w = torch.sigmoid(self.cnn_weight)
        return w * cnn_out + (1 - w) * tf_out

    def predict_top_k(self, feat: torch.Tensor, seq: torch.Tensor, k: int = 5):
        self.eval()
        with torch.no_grad():
            logits = self.forward(feat.unsqueeze(0), seq.unsqueeze(0))
            probs = torch.softmax(logits, dim=-1)
            top_k_probs, top_k_indices = torch.topk(probs, k)
            return top_k_probs.squeeze(0).tolist(), top_k_indices.squeeze(0).tolist()

    def save(self, name: str = "nmr_ensemble") -> str:
        _WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        path = _WEIGHTS_DIR / f"{name}.pt"
        torch.save(
            {
                "cnn_state": self.cnn.state_dict(),
                "transformer_state": self.transformer.state_dict(),
                "cnn_weight": self.cnn_weight,
            },
            path,
        )
        return str(path)

    def load(self, name: str = "nmr_ensemble") -> bool:
        path = _WEIGHTS_DIR / f"{name}.pt"
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=get_device(), weights_only=True)
        self.cnn.load_state_dict(ckpt["cnn_state"])
        self.transformer.load_state_dict(ckpt["transformer_state"])
        if "cnn_weight" in ckpt:
            self.cnn_weight.data = ckpt["cnn_weight"]
        return True
