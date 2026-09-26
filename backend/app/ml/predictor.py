"""Unified prediction interface: NMR spectrum → Top-K compound candidates.

LEGACY PROTOTYPE (2026-08 ML review): this predictor serves the archived
demo ensemble trained on rule-synthesised data (see
dataset/downloader.py).  It is kept for the legacy chain only; results must
not be cited as measured-data evidence.
"""

import logging
import sqlite3
import time
from pathlib import Path

import torch

from app.ml.dataset.downloader import db_path, load_or_build_dataset
from app.ml.dataset.preprocessor import extract_features, peaks_to_sequence
from app.ml.dataset.transforms import NMRCompoundDataset
from app.ml.models.ensemble import NMREnsembleClassifier, get_device

logger = logging.getLogger(__name__)

_WEIGHTS_DIR = Path(__file__).parent / "pretrained"

_MODEL: NMREnsembleClassifier | None = None
_CLASS_NAMES: list[str] = []
_MODEL_LOADED = False

_MODEL_13C: NMREnsembleClassifier | None = None
_CLASS_NAMES_13C: list[str] = []
_MODEL_13C_LOADED = False


def reset_model():
    global _MODEL, _CLASS_NAMES, _MODEL_LOADED, _MODEL_13C, _CLASS_NAMES_13C, _MODEL_13C_LOADED
    _MODEL = None
    _CLASS_NAMES = []
    _MODEL_LOADED = False
    _MODEL_13C = None
    _CLASS_NAMES_13C = []
    _MODEL_13C_LOADED = False


def _ensure_data() -> None:
    load_or_build_dataset()


def _ensure_model() -> NMREnsembleClassifier:
    global _MODEL, _CLASS_NAMES, _MODEL_LOADED
    if _MODEL_LOADED:
        return _MODEL

    # Load checkpoint first to determine class count
    weights_path = _WEIGHTS_DIR / "nmr_ensemble.pt"
    if not weights_path.exists():
        weights_path = _WEIGHTS_DIR / "nmr_ensemble_best.pt"
    if not weights_path.exists():
        raise RuntimeError("No trained model found. Run training first.")

    ckpt = torch.load(str(weights_path), map_location=get_device(), weights_only=True)
    if "class_names" in ckpt:
        _CLASS_NAMES = ckpt["class_names"]
    else:
        _ensure_data()
        ds = NMRCompoundDataset(augment=False)
        _CLASS_NAMES = ds.class_names

    _MODEL = NMREnsembleClassifier(len(_CLASS_NAMES))
    _MODEL.cnn.load_state_dict(ckpt["cnn_state"])
    _MODEL.transformer.load_state_dict(ckpt["transformer_state"])
    if "cnn_weight" in ckpt:
        _MODEL.cnn_weight.data = ckpt["cnn_weight"]
    _MODEL_LOADED = True
    return _MODEL


# Representative SMILES per functional group for structure display
_FUNC_GROUP_SMILES = {
    "ethers": "CCOCC",
    "alkenes": "C=CC",
    "aromatic_hydrocarbons": "c1ccccc1",
    "alkanes": "CCCCC",
    "carboxylic_acids": "CC(=O)O",
    "alcohols": "CCO",
    "aldehydes": "CC=O",
    "ketones": "CC(=O)C",
    "amines": "CCN",
    "pyridines": "c1ccncc1",
    "sp3_alkyl": "CCCC(C)(C)C",
    "sp3_hetero": "CCOCC",
    "sp2_alkene": "C=CCCC",
    "aromatic_CH": "c1ccccc1",
    "aromatic_Cq": "c1ccccc1C",
    "heteroaromatic": "c1ccncc1",
    "ester_acid": "CC(=O)OC",
    "ketone_aldehyde": "CC(=O)C",
}


def predict_from_peaks(peaks: list[dict], top_k: int = 5) -> dict:
    """Predict compounds from peak list (shift, intensity dicts)."""
    model = _ensure_model()
    device = get_device()

    if not peaks:
        return {"error": "No peaks detected", "top5_candidates": []}

    feat_np = extract_features(peaks)
    seq_np = peaks_to_sequence(peaks)

    feat = torch.tensor(feat_np, dtype=torch.float32, device=device)
    seq = torch.tensor(seq_np, dtype=torch.float32, device=device)

    t0 = time.time()
    probs, indices = model.predict_top_k(feat, seq, k=top_k)
    elapsed_ms = (time.time() - t0) * 1000

    candidates = []
    db = db_path()
    if db.exists():
        with sqlite3.connect(str(db)) as conn:
            for i, idx in enumerate(indices):
                if idx < len(_CLASS_NAMES):
                    name = _CLASS_NAMES[idx]
                    row = conn.execute(
                        "SELECT smiles, formula, mw FROM compounds WHERE name=? LIMIT 1",
                        (name,),
                    ).fetchone()
                    candidates.append({
                        "rank": i + 1,
                        "compound_name": name,
                        "smiles": row[0] if row and row[0] else _FUNC_GROUP_SMILES.get(name, ""),
                        "molecular_formula": row[1] if row and len(row) > 1 and row[1] else "",
                        "molecular_weight": round(row[2], 2) if row and len(row) > 2 and row[2] else None,
                        "confidence": round(probs[i], 4),
                    })
    # conn.close() handled by context manager

    return {
        "status": "completed",
        "inference_time_ms": round(elapsed_ms, 1),
        "model_version": "ensemble_v1",
        "n_classes": len(_CLASS_NAMES),
        "top5_candidates": candidates,
        "summary": _build_summary(candidates),
    }


def _build_summary(candidates: list[dict]) -> str:
    if not candidates:
        return "No candidate compounds identified."
    top = candidates[0]
    return (
        f"Top match: {top['compound_name']} (confidence: {top['confidence']:.1%}). "
        f"{len(candidates)} candidates returned."
    )


def get_model_status() -> dict:
    """Report legacy model availability without loading or building anything."""

    weights_path = Path(__file__).parent / "pretrained" / "nmr_ensemble.pt"
    return {
        "model_loaded": _MODEL_LOADED,
        "n_classes": len(_CLASS_NAMES),
        "device": str(get_device()),
        "weights_exist": weights_path.exists(),
    }


def _ensure_model_13c():
    global _MODEL_13C, _CLASS_NAMES_13C, _MODEL_13C_LOADED
    if _MODEL_13C_LOADED:
        return _MODEL_13C

    weights_path = Path(__file__).parent / "pretrained" / "nmr_ensemble_13c.pt"
    if not weights_path.exists():
        raise RuntimeError("13C model weights not found. Run train_13c_real.py first.")

    ckpt = torch.load(weights_path, map_location=get_device(), weights_only=True)

    # Use class names from checkpoint, fall back to DB
    if "class_names" in ckpt:
        _CLASS_NAMES_13C = ckpt["class_names"]
    else:
        db_path = Path(__file__).parent / "dataset" / "compounds_13c.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            rows = conn.execute("SELECT name FROM compounds").fetchall()
            conn.close()
            _CLASS_NAMES_13C = sorted(set(r[0] for r in rows))
        else:
            raise RuntimeError("13C database not found and no class_names in checkpoint")

    _MODEL_13C = NMREnsembleClassifier(len(_CLASS_NAMES_13C))
    _MODEL_13C.cnn.load_state_dict(ckpt["cnn_state"])
    _MODEL_13C.transformer.load_state_dict(ckpt["transformer_state"])
    if "cnn_weight" in ckpt:
        _MODEL_13C.cnn_weight.data = ckpt["cnn_weight"]
    _MODEL_13C_LOADED = True
    return _MODEL_13C


def predict_dual(peaks_1h: list[dict], peaks_13c: list[dict] | None = None, top_k: int = 5) -> dict:
    """Combined 1H + 13C prediction. Falls back to 1H-only if no 13C available."""
    device = get_device()
    result = {"h1": [], "c13": [], "combined": []}

    # 1H prediction - always runs
    try:
        model_h1 = _ensure_model()
        feat_h1 = extract_features(peaks_1h)
        seq_h1 = peaks_to_sequence(peaks_1h)
        f = torch.tensor(feat_h1, dtype=torch.float32, device=device)
        s = torch.tensor(seq_h1, dtype=torch.float32, device=device)
        probs_h1, indices_h1 = model_h1.predict_top_k(f, s, k=top_k)
        result["h1"] = [{"compound_name": _CLASS_NAMES[i], "confidence": round(probs_h1[j], 4), "smiles": _FUNC_GROUP_SMILES.get(_CLASS_NAMES[i], "")} for j, i in enumerate(indices_h1)]
    except Exception as e:
        # Never stuff exception text into compound_name; report the failure
        # out-of-band and leave the candidate list empty.
        logger.warning("1H prediction unavailable: %s", e)
        result["h1"] = []
        result["h1_error"] = str(e)

    # 13C prediction - only if peaks available
    try:
        if peaks_13c and len(peaks_13c) > 0:
            model_13c = _ensure_model_13c()
            feat_13c = extract_features(peaks_13c)
            seq_13c = peaks_to_sequence(peaks_13c)
            f13 = torch.tensor(feat_13c, dtype=torch.float32, device=device)
            s13 = torch.tensor(seq_13c, dtype=torch.float32, device=device)
            probs_13c, indices_13c = model_13c.predict_top_k(f13, s13, k=top_k)
            result["c13"] = [{"compound_name": _CLASS_NAMES_13C[i], "confidence": round(probs_13c[j], 4), "smiles": _FUNC_GROUP_SMILES.get(_CLASS_NAMES_13C[i], "")} for j, i in enumerate(indices_13c)]
    except Exception as e:
        # Record the failure and mark the 13C branch unavailable instead of
        # silently passing (previous behaviour hid broken 13C models).
        logger.warning("13C prediction unavailable: %s", e)
        result["c13"] = []
        result["c13_error"] = str(e)

    # Combined
    if result["c13"]:
        h1_dict = {r["compound_name"]: r["confidence"] for r in result["h1"]}
        c13_dict = {r["compound_name"]: r["confidence"] for r in result["c13"]}
        scores = {}
        for cls in set(list(h1_dict.keys())[:3] + list(c13_dict.keys())[:3]):
            scores[cls] = h1_dict.get(cls, 0) * 0.6 + c13_dict.get(cls, 0) * 0.4
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        result["combined"] = [{"compound_name": k, "confidence": round(v, 4), "smiles": _FUNC_GROUP_SMILES.get(k, "")} for k, v in sorted_scores[:top_k]]
    else:
        result["combined"] = result["h1"][:top_k]

    return result
