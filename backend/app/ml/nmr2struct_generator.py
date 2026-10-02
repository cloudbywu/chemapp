"""NMR2Struct-based experimental candidate generator.

Wraps the vendored NMR2Struct multitask model (backend/vendor/nmr2struct,
MIT-licensed, https://github.com/MarklandGroup/NMR2Struct) as a drop-in
replacement for the experimental T5 generator, which was measured to produce
zero formula-valid candidates (0/30) while this model reaches Top-15 = 38.5%
exact-structure hits on a 200-molecule CHNO-only nmrshiftdb2 sample with
13C-only input (see backend/reports/ml_eval/ML_EVALUATION_REPORT.md).

Domain limits (enforced fail-closed):
- The model alphabet is CHNO-only: molecules containing other elements are
  ungeneratable by construction, so requests whose formula contains
  non-CHNO elements return unavailable status instead of wasting inference.
- The training domain covers molecules up to 19 heavy atoms; larger
  requests return out_of_domain.
- 13C peaks are required (the 13C-only checkpoint is the validated path;
  our peak-list-derived pseudo-1H spectra measurably degrade generation).
- The model is loaded lazily once per process (~98 MB checkpoint).

Generated candidates are unverified proposals: they only become ranking
candidates after the downstream validate_generated_smiles checks and the
hybrid forward-scoring pass (which keeps calibrated_probability=False).
"""

from __future__ import annotations

import logging
import os
import pickle
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("chemapp.nmr2struct")

_VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "nmr2struct"
_CHECKPOINTS = {
    "multitask": "multitask_checkpoint.pt",
    "cnmr_only": "cnmr_only_checkpoint.pt",
    "hnmr_only": "hnmr_only_checkpoint.pt",
}
_MAX_HEAVY_ATOMS = 19
_CHNO = {"C", "H", "N", "O"}

_MODEL = None
_MODEL_LOCK = threading.Lock()
_MODEL_VARIANT: str | None = None


def _model_args() -> dict[str, Any]:
    return {
        "model_type": "MultiTaskModel",
        "load_model": None,
        "model_args": {
            "src_embed": "ConvolutionalEmbedding",
            "src_embed_options": {
                "d_model": 128,
                "n_hnmr_features": 28000,
                "n_cnmr_features": 80,
                "use_hnmr": False,
                "use_cnmr": True,
            },
            "forward_fxn": "src_fwd_fxn_conv_embedding",
            "substructure_model": "EncoderModel",
            "substructure_model_args": {
                "d_model": 128,
                "d_out": 957,
                "dim_feedforward": 1024,
                "nhead": 4,
                "num_layers": 4,
                "output_head": "SingleLinear",
                "output_head_opts": {"d_model": 128, "d_out": 957},
                "pooler": "SeqPool",
                "pooler_opts": {"d_model": 128},
                "source_size": 958,
                "src_embed": None,
                "src_embed_options": {},
                "src_forward_function": "src_fwd_fxn_packed_tensor",
                "src_pad_token": 0,
            },
            "structure_model": "TransformerModel",
            "structure_model_args": {
                "src_embed": None,
                "src_embed_options": {},
                "tgt_embed": "nn.embed",
                "tgt_embed_options": {},
                "src_forward_function": "src_fwd_fxn_packed_tensor",
                "tgt_forward_function": "tgt_fwd_fxn_basic",
                "source_size": 958,
                "target_size": 24,
                "d_model": 128,
                "dim_feedforward": 1024,
                "src_pad_token": 0,
                "tgt_pad_token": 21,
            },
            "substructure_model_ckpt": None,
            "structure_model_ckpt": None,
        },
    }


def _variant() -> str:
    env = os.environ.get("CHEMAPP_NMR2STRUCT_VARIANT", "cnmr_only").strip()
    return env if env in _CHECKPOINTS else "cnmr_only"


def available() -> bool:
    """True when the vendored code and selected checkpoint are present."""

    return (_VENDOR_DIR / "nmr").is_dir() and (
        _VENDOR_DIR / "checkpoints" / _CHECKPOINTS[_variant()]
    ).is_file()


def _load_model():
    global _MODEL, _MODEL_VARIANT
    variant = _variant()
    with _MODEL_LOCK:
        if _MODEL is not None and _MODEL_VARIANT == variant:
            return _MODEL
        if not available():
            return None
        try:
            import torch

            if str(_VENDOR_DIR) not in sys.path:
                sys.path.insert(0, str(_VENDOR_DIR))
            from nmr.models import create_model  # type: ignore[import-not-found]

            device = torch.device("cpu")
            model, _ = create_model(_model_args(), torch.float32, device)
            ckpt = torch.load(
                _VENDOR_DIR / "checkpoints" / _CHECKPOINTS[variant],
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            _MODEL = model
            _MODEL_VARIANT = variant
            logger.info("NMR2Struct %s model loaded", variant)
        except Exception:
            logger.exception("NMR2Struct model load failed")
            _MODEL = None
            _MODEL_VARIANT = None
            return None
    return _MODEL


def _formula_elements(formula: str) -> set[str]:
    return set(re.findall(r"[A-Z][a-z]?", formula))


def _heavy_atom_count(formula: str) -> int | None:
    total = 0
    found = False
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if element == "H":
            continue
        found = True
        total += int(count) if count else 1
    return total if found else None


def generate_candidates(
    peaks_13c: list[float],
    formula: str | None = None,
    top_k: int = 15,
) -> dict[str, Any]:
    """Generate structure candidates from a 13C peak list.

    Returns a dict with status and candidates; never raises - every failure
    mode maps to an explicit non-ok status (fail-closed).
    """

    started = time.time()
    base: dict[str, Any] = {
        "generator": "nmr2struct",
        "model": {
            "name": "NMR2Struct",
            "variant": _variant(),
            "checkpoint": _CHECKPOINTS[_variant()],
            "reference": "10.1021/acscentsci.4c01132",
        },
        "calibrated_probability": False,
        "candidates": [],
        "inference_time_ms": 0,
    }
    peaks = []
    for value in peaks_13c:
        try:
            shift = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(shift):
            peaks.append(shift)
    if not peaks:
        return {**base, "status": "unavailable", "reason": "no_13c_peaks"}
    if formula:
        elements = _formula_elements(formula)
        if not elements <= _CHNO:
            return {
                **base,
                "status": "unavailable",
                "reason": "formula_outside_chno_alphabet",
            }
        heavy = _heavy_atom_count(formula)
        if heavy is not None and heavy > _MAX_HEAVY_ATOMS:
            return {
                **base,
                "status": "unavailable",
                "reason": "out_of_domain_heavy_atoms",
                "heavy_atoms": heavy,
            }
    model = _load_model()
    if model is None:
        return {**base, "status": "unavailable", "reason": "model_unavailable"}

    try:
        import contextlib
        import io

        import torch

        from nmr.inference.inference_fxns import (  # type: ignore[import-not-found]
            infer_transformer_model,
        )

        with open(_VENDOR_DIR / "example_configs" / "CNMR_shifts.p", "rb") as fh:
            c_grid = np.asarray(pickle.load(fh), dtype=float)
        c_spec = np.zeros(len(c_grid))
        values = np.sort(np.asarray(peaks, dtype=float))
        bins = np.digitize(values, c_grid)
        bins = np.where(bins == len(c_grid), len(c_grid) - 1, bins)
        c_spec[bins] = 1
        spectrum = np.concatenate([np.zeros(28000), c_spec])
        x = (torch.from_numpy(spectrum).float().unsqueeze(0), ["NULL"])
        y = (torch.zeros(1), torch.zeros(1))
        opts = {
            "num_pred_per_tgt": max(1, int(top_k)),
            "sample_val": 5,
            "tgt_start_token": 22,
            "tgt_stop_token": 23,
            "track_gradients": False,
            "alphabet": str(_VENDOR_DIR / "example_configs" / "alphabet.npy"),
            "decode": True,
            "infer_fwd_fxn": "multitask",
        }
        device = torch.device("cpu")
        # The vendored generation loop prints progress every 10 tokens;
        # keep that chatter out of the server log.
        with contextlib.redirect_stdout(io.StringIO()):
            raw = infer_transformer_model(model, (x, y), opts, device)
        decoded: list[str] = []
        for pred in raw:
            seq = pred[1] if isinstance(pred, (tuple, list)) else pred
            if isinstance(seq, str):
                decoded.append(seq)
            else:
                for piece in np.asarray(seq, dtype=object).flatten():
                    if isinstance(piece, str) and piece:
                        decoded.append(piece)
        candidates = [
            {"smiles": smi, "origin": "nmr2struct_generated"}
            for smi in dict.fromkeys(decoded)
        ]
        return {
            **base,
            "status": "ok",
            "candidates": candidates,
            "inference_time_ms": round((time.time() - started) * 1000, 1),
        }
    except Exception:
        logger.exception("NMR2Struct generation failed")
        return {
            **base,
            "status": "unavailable",
            "reason": "generation_error",
            "inference_time_ms": round((time.time() - started) * 1000, 1),
        }
