"""NMR2Struct-based experimental candidate generator.

Wraps the vendored NMR2Struct multitask architecture (backend/vendor/nmr2struct,
MIT-licensed, https://github.com/MarklandGroup/NMR2Struct). Checkpoint presence,
input compatibility and architecture tests do not establish generation accuracy.
Trained-checkpoint performance must be validated separately.

Domain limits (enforced fail-closed):
- The model alphabet is CHNO-only: molecules containing other elements are
  ungeneratable by construction, so requests whose formula contains
  non-CHNO elements return unavailable status instead of wasting inference.
- The training domain covers molecules up to 19 heavy atoms; larger
  requests return out_of_domain.
- Each checkpoint requires its actual input channels: 13C peaks, a continuous
  1H spectrum, or both. Peak-list-derived pseudo-1H spectra are never synthesized.
- CHEMAPP_NMR2STRUCT_VARIANT defaults to auto: prefer an available combined
  checkpoint for H+C, then C-only, then H-only. Explicit cnmr_only, hnmr_only
  and multitask choices consume exactly those channels and never switch silently.
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

from app.ml.nmr2struct_weights import checkpoint_path

logger = logging.getLogger("chemapp.nmr2struct")

_VENDOR_DIR = Path(__file__).resolve().parents[2] / "vendor" / "nmr2struct"
_CHECKPOINTS = {
    "multitask": "multitask_checkpoint.pt",
    "cnmr_only": "cnmr_only_checkpoint.pt",
    "hnmr_only": "hnmr_only_checkpoint.pt",
}
# Channel order is (continuous 1H, peak-list 13C).
_CHANNELS = {
    "multitask": (True, True),
    "cnmr_only": (False, True),
    "hnmr_only": (True, False),
}
_INPUT_MODES = {
    "multitask": "13c_peaks+1h_spectrum",
    "cnmr_only": "13c_peaks",
    "hnmr_only": "1h_spectrum",
}
_PROMPT_SCHEMAS = {
    "multitask": "nmr2struct-13c-peaks+1h-spectrum-v1",
    "cnmr_only": "nmr2struct-13c-peaks-v1",
    "hnmr_only": "nmr2struct-1h-spectrum-v1",
}
# The fixed model proton grid covers [-2, 12) ppm at 0.0005 ppm spacing.
_H_GRID = np.arange(-2.0, 12.0, 0.0005)
# Carbon digitization thresholds, not established chemical applicability limits.
_C_GRID = np.linspace(3.423975000000001, 231.30000000000004, 80)
_C_PPM_SANITY_RANGE = (-20.0, 300.0)  # Same input guard as the API evidence path.
_MAX_HEAVY_ATOMS = 19
_CHNO = {"C", "H", "N", "O"}

_MODELS: dict[str, Any] = {}
_MODEL_IDENTITIES: dict[str, tuple[Any, ...]] = {}
_MODEL_LOCK = threading.Lock()


def _model_args(variant: str = "cnmr_only") -> dict[str, Any]:
    use_hnmr, use_cnmr = _CHANNELS[variant]
    return {
        "model_type": "MultiTaskModel",
        "load_model": None,
        "model_args": {
            "src_embed": "ConvolutionalEmbedding",
            "src_embed_options": {
                "d_model": 128,
                "n_hnmr_features": 28000,
                "n_cnmr_features": 80,
                "use_hnmr": use_hnmr,
                "use_cnmr": use_cnmr,
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
    return os.environ.get("CHEMAPP_NMR2STRUCT_VARIANT", "auto").strip().casefold()


def available(
    peaks_13c: list[float] | None = None,
    formula: str | None = None,
    spectrum_1h: tuple[Any, Any] | None = None,
) -> bool:
    """Whether a checkpoint is present and compatible with these actual inputs.

    This is a preflight check, not a claim that its weights load successfully.
    With no spectral inputs this returns False. Use _available_variant for a
    checkpoint-file-only check, for example to mark checkpoint-dependent tests.
    """

    return resolve_input_mode(peaks_13c, formula, spectrum_1h)["status"] == "ok"


def _available_variant(variant: str) -> bool:
    return (
        variant in _CHECKPOINTS
        and (_VENDOR_DIR / "nmr").is_dir()
        and checkpoint_path(variant, _VENDOR_DIR) is not None
    )


def checkpoint_status() -> dict[str, bool]:
    """Checkpoint asset inventory only; does not imply input or load readiness."""

    return {variant: _available_variant(variant) for variant in _CHECKPOINTS}


def _load_model(variant: str):
    with _MODEL_LOCK:
        path = checkpoint_path(variant, _VENDOR_DIR)
        if path is None or not (_VENDOR_DIR / "nmr").is_dir():
            return None
        try:
            stat = path.stat()
            identity = (str(path), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            if variant in _MODELS and _MODEL_IDENTITIES.get(variant) == identity:
                return _MODELS[variant]
            import torch

            if str(_VENDOR_DIR) not in sys.path:
                sys.path.insert(0, str(_VENDOR_DIR))
            from nmr.models import create_model  # type: ignore[import-not-found]

            device = torch.device("cpu")
            model, _ = create_model(_model_args(variant), torch.float32, device)
            ckpt = torch.load(
                path,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            _MODELS[variant] = model
            _MODEL_IDENTITIES[variant] = identity
            logger.info("NMR2Struct %s model loaded", variant)
        except Exception:
            logger.exception("NMR2Struct model load failed")
            return None
    return _MODELS.get(variant)


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


def _rasterize_1h(x_ppm: Any, y_intensity: Any, h_grid: Any) -> Any:
    """Validate a continuous trace and interpolate without inventing edge signal.

    Both ascending and descending axes are accepted; scrambled, duplicate,
    mismatched and nonfinite samples fail closed instead of being repaired.
    Signal outside the measured range is zero, not an extrapolated plateau.
    """

    try:
        x = np.asarray(x_ppm, dtype=float)
        y = np.asarray(y_intensity, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid_1h_spectrum") from exc
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size or x.size < 2:
        raise ValueError("invalid_1h_spectrum")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("invalid_1h_spectrum")
    if np.all(y == y[0]):
        raise ValueError("no_usable_1h_signal")
    diffs = np.diff(x)
    if np.all(diffs < 0):
        x, y = x[::-1], y[::-1]
    elif not np.all(diffs > 0):
        raise ValueError("nonmonotonic_1h_spectrum")
    h = np.interp(h_grid, x, y, left=0.0, right=0.0)
    peak = float(h.max())
    if not np.isfinite(peak) or peak <= 0:
        raise ValueError("no_usable_1h_signal")
    # Baseline-corrected traces may contain negative noise, but must have a
    # finite positive signal inside the model's supported proton window.
    with np.errstate(over="ignore", invalid="ignore"):
        normalized = h / peak
    if not np.all(np.isfinite(normalized)):
        raise ValueError("invalid_1h_spectrum")
    return normalized


def _resolve_inputs(
    peaks_13c: list[float] | None,
    formula: str | None,
    spectrum_1h: tuple[Any, Any] | None,
) -> tuple[dict[str, Any], list[float], Any]:
    """Single implementation used by both preflight and generation."""

    requested = _variant()
    explicit = requested if requested in _CHECKPOINTS else None
    result: dict[str, Any] = {
        "status": "unavailable",
        "generator": "nmr2struct",
        "model": {
            "name": "NMR2Struct",
            "variant": explicit,
            "checkpoint": _CHECKPOINTS.get(explicit),
            "reference": "10.1021/acscentsci.4c01132",
        },
        "requested_variant": requested,
        "input_mode": _INPUT_MODES.get(explicit),
        "prompt_schema": _PROMPT_SCHEMAS.get(explicit),
        "provided_modalities": [],
        "used_modalities": [],
        "ignored_modalities": [],
        "input_warnings": [],
        "calibrated_probability": False,
    }
    peaks: list[float] = []
    h_spec = None

    def reject(reason: str):
        return {**result, "reason": reason}, peaks, h_spec

    if requested not in {*_CHECKPOINTS, "auto"}:
        return reject("unsupported_variant")
    try:
        values = np.asarray([] if peaks_13c is None else peaks_13c, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return reject("invalid_13c_peaks")
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        return reject("invalid_13c_peaks")
    peaks = values.tolist()
    if np.any(values < _C_PPM_SANITY_RANGE[0]) or np.any(
        values > _C_PPM_SANITY_RANGE[1]
    ):
        return reject("13c_shifts_outside_supported_range")
    if peaks:
        result["provided_modalities"].append("13c_peaks")
    if spectrum_1h is not None:
        result["provided_modalities"].append("1h_spectrum")
        try:
            if len(spectrum_1h) != 2:
                return reject("invalid_1h_spectrum")
            h_spec = _rasterize_1h(spectrum_1h[0], spectrum_1h[1], _H_GRID)
        except (TypeError, ValueError, IndexError, OverflowError) as exc:
            reason = str(exc)
            return reject(
                reason
                if reason
                in {
                    "invalid_1h_spectrum",
                    "nonmonotonic_1h_spectrum",
                    "no_usable_1h_signal",
                }
                else "invalid_1h_spectrum"
            )
    if formula:
        if not isinstance(formula, str) or not re.fullmatch(
            r"(?:[A-Z][a-z]?(?:[1-9]\d*)?)+", formula.strip()
        ):
            return reject("invalid_formula")
        if not _formula_elements(formula) <= _CHNO:
            return reject("formula_outside_chno_alphabet")
        try:
            heavy = _heavy_atom_count(formula)
        except ValueError:
            return reject("invalid_formula")
        if heavy is not None and heavy > _MAX_HEAVY_ATOMS:
            result["heavy_atoms"] = heavy
            return reject("out_of_domain_heavy_atoms")
    # Distinct observed carbon environments are a lower bound on carbon atoms,
    # even when the formula is missing or understates the molecular size.
    observed_carbon_lower_bound = len(set(peaks))
    result["observed_carbon_lower_bound"] = observed_carbon_lower_bound
    if observed_carbon_lower_bound > _MAX_HEAVY_ATOMS:
        return reject("out_of_domain_observed_carbons")
    if requested == "auto":
        choices = []
        if peaks and h_spec is not None:
            choices.append("multitask")
        if peaks:
            choices.append("cnmr_only")
        if h_spec is not None:
            choices.append("hnmr_only")
        if not choices:
            return reject("no_13c_peaks")
        variant = next((v for v in choices if _available_variant(v)), choices[0])
    else:
        variant = requested
    use_hnmr, use_cnmr = _CHANNELS[variant]
    result["model"]["variant"] = variant
    result["model"]["checkpoint"] = _CHECKPOINTS[variant]
    result["input_mode"] = _INPUT_MODES[variant]
    result["prompt_schema"] = _PROMPT_SCHEMAS[variant]
    if use_cnmr and not peaks:
        return reject("no_13c_peaks")
    if use_hnmr and h_spec is None:
        return reject("no_1h_spectrum")
    if use_cnmr and any(shift < _C_GRID[0] or shift >= _C_GRID[-1] for shift in peaks):
        result["input_warnings"].append("13c_shifts_use_boundary_bins")
    intended = (["13c_peaks"] if use_cnmr else []) + (
        ["1h_spectrum"] if use_hnmr else []
    )
    result["ignored_modalities"] = [
        modality
        for modality in result["provided_modalities"]
        if modality not in intended
    ]
    if not _available_variant(variant):
        return reject("model_unavailable")
    result["used_modalities"] = intended
    result["status"] = "ok"
    return result, peaks, h_spec


def resolve_input_mode(
    peaks_13c: list[float] | None = None,
    formula: str | None = None,
    spectrum_1h: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Return input-aware checkpoint selection and its explicit provenance.

    Explicit variants consume exactly their named channels. Auto prefers an
    available combined checkpoint for valid H+C input, then C-only, then H-only.
    Invalid supplied evidence and known domain violations always fail closed,
    including when that modality would otherwise be ignored. A successful
    preflight only establishes input compatibility and checkpoint presence.
    """

    return _resolve_inputs(peaks_13c, formula, spectrum_1h)[0]


def generate_candidates(
    peaks_13c: list[float],
    formula: str | None = None,
    top_k: int = 15,
    spectrum_1h: tuple[Any, Any] | None = None,
) -> dict[str, Any]:
    """Generate unverified structure proposals using compatible spectral inputs.

    A continuous 1H spectrum is required for proton channels; peak lists are
    never synthesized into 1H traces. Failures return explicit non-ok statuses.
    """

    started = time.time()
    resolution, peaks, h_spec = _resolve_inputs(peaks_13c, formula, spectrum_1h)
    base: dict[str, Any] = {
        **resolution,
        "candidates": [],
        "inference_time_ms": 0,
    }
    if resolution["status"] != "ok":
        return base
    variant = resolution["model"]["variant"]
    model = _load_model(variant)
    if model is None:
        return {
            **base,
            "status": "unavailable",
            "reason": "model_unavailable",
            "used_modalities": [],
        }

    try:
        import torch

        from nmr.inference.inference_fxns import (  # type: ignore[import-not-found]
            infer_transformer_model,
        )

        with open(_VENDOR_DIR / "example_configs" / "CNMR_shifts.p", "rb") as fh:
            c_grid = np.asarray(pickle.load(fh), dtype=float)
        with open(_VENDOR_DIR / "example_configs" / "HNMR_shifts.p", "rb") as fh:
            h_grid = np.asarray(pickle.load(fh), dtype=float)
        use_hnmr, use_cnmr = _CHANNELS[variant]
        if c_grid.shape != _C_GRID.shape or not np.allclose(c_grid, _C_GRID):
            raise ValueError("Unexpected NMR2Struct carbon grid")
        c_spec = np.zeros(len(c_grid))
        if use_cnmr:
            values = np.sort(np.asarray(peaks, dtype=float))
            bins = np.digitize(values, c_grid)
            bins = np.where(bins == len(c_grid), len(c_grid) - 1, bins)
            c_spec[bins] = 1
        if use_hnmr:
            # Refuse a changed vendor grid rather than disagreeing with preflight.
            if h_grid.shape != _H_GRID.shape or not np.allclose(h_grid, _H_GRID):
                raise ValueError("Unexpected NMR2Struct proton grid")
        else:
            h_spec = np.zeros(len(h_grid))
        spectrum = np.concatenate([h_spec, c_spec])
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
            "verbose": False,
        }
        device = torch.device("cpu")
        # Per-call verbosity avoids mutating process-global sys.stdout in the
        # server's concurrent inference workers.
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
            "used_modalities": [],
            "inference_time_ms": round((time.time() - started) * 1000, 1),
        }
