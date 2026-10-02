"""Tests for the NMR2Struct experimental generator and its route dispatch."""

from __future__ import annotations

import pytest

import app.ml.nmr2struct_generator as n2s
from app.api.routes import elucidate as route


def test_unavailable_without_vendor_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(n2s, "_VENDOR_DIR", tmp_path / "missing")
    monkeypatch.setattr(n2s, "_MODEL", None)
    assert not n2s.available()
    result = n2s.generate_candidates([18.3, 58.1])
    assert result["status"] == "unavailable"
    assert result["reason"] == "model_unavailable"
    assert result["calibrated_probability"] is False


def test_requires_13c_peaks():
    result = n2s.generate_candidates([], formula="C2H6O")
    assert result["status"] == "unavailable"
    assert result["reason"] == "no_13c_peaks"


def test_non_chno_formula_rejected_without_inference():
    result = n2s.generate_candidates([18.3, 58.1], formula="C2H5Cl")
    assert result["status"] == "unavailable"
    assert result["reason"] == "formula_outside_chno_alphabet"


def test_over_19_heavy_atoms_rejected():
    result = n2s.generate_candidates([10.0] * 25, formula="C25H52")
    assert result["status"] == "unavailable"
    assert result["reason"] == "out_of_domain_heavy_atoms"
    assert result["heavy_atoms"] == 25


@pytest.mark.skipif(not n2s.available(), reason="NMR2Struct checkpoint not present")
def test_real_generation_returns_smiles():
    result = n2s.generate_candidates([18.3, 58.1], formula="C2H6O", top_k=5)
    assert result["status"] == "ok"
    assert result["candidates"], "expected at least one candidate"
    for candidate in result["candidates"]:
        assert candidate["smiles"]
        assert candidate["origin"] == "nmr2struct_generated"
    assert result["inference_time_ms"] > 0


def test_dispatch_prefers_nmr2struct_in_auto_when_available(monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "auto")
    monkeypatch.setattr(
        route,
        "validate_generated_smiles",
        lambda decoded, formula=None: (
            [{"smiles": s, "valid": True} for s in decoded],
            {"accepted": len(decoded)},
        ),
    )
    import app.ml.nmr2struct_generator as gen_mod

    monkeypatch.setattr(gen_mod, "available", lambda: True)
    monkeypatch.setattr(
        gen_mod,
        "generate_candidates",
        lambda peaks, formula=None, top_k=15: {
            "status": "ok",
            "candidates": [{"smiles": "CCO", "origin": "nmr2struct_generated"}],
            "inference_time_ms": 5,
        },
    )
    out = route._dispatch_generation(
        [{"shift": 18.3}, {"shift": 58.1}], [], "C2H6O", 5, 5
    )
    assert out["status"] == "completed"
    assert out["generator"] == "nmr2struct"
    assert out["candidates"][0]["smiles"] == "CCO"


def test_dispatch_t5_forced_uses_legacy_path(monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "t5")
    sentinel = {"status": "completed", "candidates": [], "inference_time_ms": 1}
    monkeypatch.setattr(route, "_safe_generate", lambda *a: dict(sentinel))
    out = route._dispatch_generation([{"shift": 10.0}], [], None, 5, 5)
    assert out["generator"] == "t5"


def test_dispatch_forced_nmr2struct_unavailable_is_fail_closed(monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "nmr2struct")
    import app.ml.nmr2struct_generator as gen_mod

    monkeypatch.setattr(
        gen_mod,
        "generate_candidates",
        lambda peaks, formula=None, top_k=15: {
            "status": "unavailable",
            "reason": "model_unavailable",
            "candidates": [],
            "inference_time_ms": 0,
        },
    )
    out = route._dispatch_generation([{"shift": 10.0}], [], None, 5, 5)
    assert out["status"] == "generation_unavailable"
    assert out["error_code"] == "model_unavailable"
    assert out["generator"] == "nmr2struct"


def test_dispatch_auto_falls_back_to_t5_when_unavailable(monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "auto")
    import app.ml.nmr2struct_generator as gen_mod

    monkeypatch.setattr(gen_mod, "available", lambda: False)
    sentinel = {"status": "completed", "candidates": [], "inference_time_ms": 1}
    monkeypatch.setattr(route, "_safe_generate", lambda *a: dict(sentinel))
    out = route._dispatch_generation([{"shift": 10.0}], [], None, 5, 5)
    assert out["generator"] == "t5"


def test_dispatch_invalid_env_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "bogus")
    import app.ml.nmr2struct_generator as gen_mod

    monkeypatch.setattr(gen_mod, "available", lambda: False)
    sentinel = {"status": "completed", "candidates": [], "inference_time_ms": 1}
    monkeypatch.setattr(route, "_safe_generate", lambda *a: dict(sentinel))
    out = route._dispatch_generation([{"shift": 10.0}], [], None, 5, 5)
    assert out["generator"] == "t5"
