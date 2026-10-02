"""Tests for the NMR2Struct experimental generator and its route dispatch."""

from __future__ import annotations

import numpy as np
import pytest

import app.ml.nmr2struct_generator as n2s
from app.api.routes import elucidate as route


@pytest.fixture(autouse=True)
def reset_generator_configuration(monkeypatch):
    monkeypatch.delenv("CHEMAPP_NMR2STRUCT_VARIANT", raising=False)
    monkeypatch.setattr(n2s, "_MODELS", {})


def test_unavailable_without_vendor_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(n2s, "_VENDOR_DIR", tmp_path / "missing")
    monkeypatch.setattr(n2s, "_MODELS", {})
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


@pytest.mark.skipif(
    not n2s._available_variant("cnmr_only"), reason="NMR2Struct checkpoint not present"
)
def test_real_generation_returns_smiles(monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", "cnmr_only")
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
        lambda decoded, formula=None, **kwargs: (
            [
                {**s, "valid": True}
                if isinstance(s, dict)
                else {"smiles": s, "valid": True}
                for s in decoded
            ],
            {"accepted": len(decoded)},
        ),
    )
    import app.ml.nmr2struct_generator as gen_mod

    monkeypatch.setattr(gen_mod, "available", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        gen_mod,
        "generate_candidates",
        lambda peaks, formula=None, top_k=15, **_kwargs: {
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
        lambda peaks, formula=None, top_k=15, **_kwargs: {
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

    monkeypatch.setattr(gen_mod, "_available_variant", lambda _variant: False)
    sentinel = {"status": "completed", "candidates": [], "inference_time_ms": 1}
    monkeypatch.setattr(route, "_safe_generate", lambda *a: dict(sentinel))
    out = route._dispatch_generation([{"shift": 10.0}], [], None, 5, 5)
    assert out["generator"] == "t5"


def test_dispatch_invalid_env_falls_back_to_auto(monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "bogus")
    import app.ml.nmr2struct_generator as gen_mod

    monkeypatch.setattr(gen_mod, "_available_variant", lambda _variant: False)
    sentinel = {"status": "completed", "candidates": [], "inference_time_ms": 1}
    monkeypatch.setattr(route, "_safe_generate", lambda *a: dict(sentinel))
    out = route._dispatch_generation([{"shift": 10.0}], [], None, 5, 5)
    assert out["generator"] == "t5"


def test_dispatch_forwards_spectrum_1h(monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR_GENERATOR", "nmr2struct")
    import app.ml.nmr2struct_generator as gen_mod

    received = {}

    def fake_generate(peaks, formula=None, top_k=15, spectrum_1h=None, **_kwargs):
        received["spectrum_1h"] = spectrum_1h
        return {"status": "ok", "candidates": [], "inference_time_ms": 1}

    monkeypatch.setattr(gen_mod, "generate_candidates", fake_generate)
    monkeypatch.setattr(
        route,
        "validate_generated_smiles",
        lambda decoded, formula=None, **kwargs: ([], {"accepted": 0}),
    )
    out = route._dispatch_generation(
        [{"shift": 10.0}], [], None, 5, 5, spectrum_1h=([1.0, 2.0], [0.5, 1.0])
    )
    assert out["status"] == "completed"
    assert received["spectrum_1h"] == ([1.0, 2.0], [0.5, 1.0])


@pytest.mark.skipif(
    not n2s._available_variant("multitask"),
    reason="multitask checkpoint not present",
)
def test_multitask_variant_used_when_real_1h_spectrum(monkeypatch):
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", "auto")
    import numpy as np

    x = np.linspace(0, 10, 2000)
    y = np.exp(-((x - 3.6) ** 2) / 0.01) + 0.8 * np.exp(-((x - 1.2) ** 2) / 0.01)
    result = n2s.generate_candidates(
        [18.3, 58.1], formula="C2H6O", top_k=3, spectrum_1h=(x, y)
    )
    assert result["status"] == "ok"
    assert result["model"]["variant"] == "multitask"


_VALID_H = ([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.25, 1.0, 0.25, 0.0])


@pytest.mark.parametrize(
    "variant,h,c",
    [
        ("cnmr_only", False, True),
        ("hnmr_only", True, False),
        ("multitask", True, True),
    ],
)
def test_architecture_channels_match_checkpoint(variant, h, c):
    opts = n2s._model_args(variant)["model_args"]["src_embed_options"]
    assert (opts["use_hnmr"], opts["use_cnmr"]) == (h, c)


@pytest.mark.parametrize(
    "requested,checkpoints,peaks,protons,expected",
    [
        ("auto", {"multitask"}, [18.3, 58.1], _VALID_H, "multitask"),
        (
            "auto",
            {"cnmr_only", "multitask", "hnmr_only"},
            [18.3],
            _VALID_H,
            "multitask",
        ),
        ("auto", {"cnmr_only"}, [18.3], _VALID_H, "cnmr_only"),
        ("auto", {"hnmr_only"}, [18.3], _VALID_H, "hnmr_only"),
        ("auto", {"hnmr_only"}, [], _VALID_H, "hnmr_only"),
        ("auto", {"cnmr_only"}, [18.3], None, "cnmr_only"),
        ("cnmr_only", {"cnmr_only", "multitask"}, [18.3], _VALID_H, "cnmr_only"),
        ("hnmr_only", {"hnmr_only", "multitask"}, [18.3], _VALID_H, "hnmr_only"),
        ("multitask", {"cnmr_only", "multitask"}, [18.3], _VALID_H, "multitask"),
    ],
)
def test_availability_and_generation_use_same_input_resolver(
    monkeypatch, requested, checkpoints, peaks, protons, expected
):
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", requested)
    monkeypatch.setattr(
        n2s, "_available_variant", lambda variant: variant in checkpoints
    )
    loaded = []
    monkeypatch.setattr(n2s, "_load_model", lambda variant: loaded.append(variant))
    selection = n2s.resolve_input_mode(peaks, "C2H6O", protons)
    assert selection["status"] == "ok"
    assert selection["model"]["variant"] == expected
    assert n2s.available(peaks, "C2H6O", protons)
    result = n2s.generate_candidates(peaks, "C2H6O", spectrum_1h=protons)
    assert loaded == [expected]
    assert result["reason"] == "model_unavailable"
    for key in (
        "model",
        "input_mode",
        "prompt_schema",
        "provided_modalities",
        "ignored_modalities",
    ):
        assert result[key] == selection[key]
    used = {
        "cnmr_only": ["13c_peaks"],
        "hnmr_only": ["1h_spectrum"],
        "multitask": ["13c_peaks", "1h_spectrum"],
    }[expected]
    assert selection["used_modalities"] == used
    assert selection["ignored_modalities"] == [
        value for value in selection["provided_modalities"] if value not in used
    ]


@pytest.mark.parametrize(
    "variant,peaks,protons,reason",
    [
        ("cnmr_only", [], _VALID_H, "no_13c_peaks"),
        ("hnmr_only", [18.3], None, "no_1h_spectrum"),
        ("multitask", [18.3], None, "no_1h_spectrum"),
        ("multitask", [], _VALID_H, "no_13c_peaks"),
        ("not_a_checkpoint", [18.3], _VALID_H, "unsupported_variant"),
    ],
)
def test_explicit_modes_reject_missing_channels_before_model_load(
    monkeypatch, variant, peaks, protons, reason
):
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", variant)
    monkeypatch.setattr(n2s, "_available_variant", lambda variant: True)
    monkeypatch.setattr(
        n2s, "_load_model", lambda *args: pytest.fail("must not load a model")
    )
    assert not n2s.available(peaks, spectrum_1h=protons)
    result = n2s.generate_candidates(peaks, spectrum_1h=protons)
    assert result["status"] == "unavailable"
    assert result["reason"] == reason
    assert result["candidates"] == []


@pytest.mark.parametrize("variant", ["cnmr_only", "hnmr_only", "multitask"])
def test_explicit_checkpoint_missing_does_not_switch_variants(monkeypatch, variant):
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", variant)
    monkeypatch.setattr(
        n2s, "_available_variant", lambda available: available != variant
    )
    selection = n2s.resolve_input_mode([18.3], spectrum_1h=_VALID_H)
    assert selection["status"] == "unavailable"
    assert selection["reason"] == "model_unavailable"
    assert selection["model"]["variant"] == variant


@pytest.mark.parametrize(
    "protons,reason",
    [
        (([], []), "invalid_1h_spectrum"),
        (([1.0], [1.0]), "invalid_1h_spectrum"),
        (([1.0, 2.0], [1.0]), "invalid_1h_spectrum"),
        (([1.0, 2.0], [0.0, 0.0]), "no_usable_1h_signal"),
        (([1.0, 2.0], [-1.0, -2.0]), "no_usable_1h_signal"),
        (([20.0, 21.0], [1.0, 1.0]), "no_usable_1h_signal"),
        (([1.0, 2.0], [float("nan"), 1.0]), "invalid_1h_spectrum"),
        (([1.0, float("inf")], [1.0, 1.0]), "invalid_1h_spectrum"),
        (([1.0, 3.0, 2.0], [1.0, 2.0, 1.0]), "nonmonotonic_1h_spectrum"),
        (([3.0, 1.0, 2.0], [1.0, 2.0, 1.0]), "nonmonotonic_1h_spectrum"),
        (([1.0, 1.0, 2.0], [1.0, 2.0, 1.0]), "nonmonotonic_1h_spectrum"),
        (([[1.0, 2.0]], [[1.0, 2.0]]), "invalid_1h_spectrum"),
        (([1.0, 2.0], ["invalid", 1.0]), "invalid_1h_spectrum"),
        (([1.0, 2.0],), "invalid_1h_spectrum"),
        (object(), "invalid_1h_spectrum"),
    ],
)
@pytest.mark.parametrize("variant", ["auto", "cnmr_only", "hnmr_only", "multitask"])
def test_invalid_supplied_h_fails_closed_even_for_c_only(
    monkeypatch, protons, reason, variant
):
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", variant)
    monkeypatch.setattr(n2s, "_available_variant", lambda *args: True)
    monkeypatch.setattr(
        n2s, "_load_model", lambda *args: pytest.fail("must not load a model")
    )
    assert not n2s.available([18.3], spectrum_1h=protons)
    result = n2s.generate_candidates([18.3], spectrum_1h=protons)
    assert result["status"] == "unavailable"
    assert result["reason"] == reason


def test_rasterization_accepts_descending_without_extrapolated_edge_signal():
    grid = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    ascending = n2s._rasterize_1h([0.0, 1.0, 2.0], [0.5, 2.0, 0.5], grid)
    descending = n2s._rasterize_1h([2.0, 1.0, 0.0], [0.5, 2.0, 0.5], grid)
    np.testing.assert_array_equal(ascending, descending)
    np.testing.assert_array_equal(ascending, [0.0, 0.25, 1.0, 0.25, 0.0])


@pytest.mark.parametrize("formula", [None, "C2H6O"])
def test_observed_carbon_lower_bound_rejects_without_relying_on_formula(
    monkeypatch, formula
):
    monkeypatch.setattr(n2s, "_available_variant", lambda *args: True)
    monkeypatch.setattr(
        n2s, "_load_model", lambda *args: pytest.fail("must not load a model")
    )
    result = n2s.generate_candidates(list(range(20)), formula=formula)
    assert result["status"] == "unavailable"
    assert result["reason"] == "out_of_domain_observed_carbons"
    assert result["observed_carbon_lower_bound"] == 20
    assert not n2s.available(list(range(20)), formula=formula)


def test_duplicate_carbon_shifts_do_not_inflate_lower_bound(monkeypatch):
    monkeypatch.setattr(n2s, "_available_variant", lambda *args: True)
    selection = n2s.resolve_input_mode([18.3] * 25)
    assert selection["status"] == "ok"
    assert selection["observed_carbon_lower_bound"] == 1


@pytest.mark.parametrize(
    "peaks", [[18.3, float("nan")], [float("inf")], ["bad"], [[18.3]], 18.3]
)
def test_invalid_carbon_peaks_are_not_silently_dropped(monkeypatch, peaks):
    monkeypatch.setattr(
        n2s, "_load_model", lambda *args: pytest.fail("must not load a model")
    )
    assert n2s.generate_candidates(peaks)["reason"] == "invalid_13c_peaks"


@pytest.mark.parametrize("formula", ["C0H6", "C2H6+", "C2H6 rubbish", 123])
def test_invalid_formula_fails_closed(monkeypatch, formula):
    monkeypatch.setattr(
        n2s, "_load_model", lambda *args: pytest.fail("must not load a model")
    )
    assert (
        n2s.generate_candidates([18.3], formula=formula)["reason"] == "invalid_formula"
    )


def test_checkpoint_inventory_does_not_claim_input_compatibility(tmp_path, monkeypatch):
    (tmp_path / "nmr").mkdir()
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "multitask_checkpoint.pt").touch()
    monkeypatch.setattr(n2s, "_VENDOR_DIR", tmp_path)
    assert n2s.checkpoint_status() == {
        "multitask": True,
        "cnmr_only": False,
        "hnmr_only": False,
    }
    assert not n2s.available()
    assert not n2s.available([18.3])
    assert n2s.available([18.3], spectrum_1h=_VALID_H)


@pytest.fixture
def vendored_inference(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(n2s._VENDOR_DIR))
    import nmr.inference.inference_fxns as inference

    monkeypatch.setattr(n2s, "_available_variant", lambda *args: True)
    monkeypatch.setattr(n2s, "_load_model", lambda variant: object())
    return inference


@pytest.mark.parametrize(
    "variant,peaks,expected_h,expected_c",
    [
        ("hnmr_only", [], True, False),
        ("hnmr_only", [18.3, 58.1], True, False),
        ("cnmr_only", [18.3, 58.1], False, True),
        ("multitask", [18.3, 58.1], True, True),
    ],
)
def test_generation_feeds_only_the_selected_channels(
    monkeypatch, vendored_inference, variant, peaks, expected_h, expected_c
):
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", variant)
    received = {}

    def infer(model, batch, opts, device):
        received["input"] = batch[0][0].numpy()
        received["opts"] = opts
        return [("NULL", ["CCO", "CCO"], "NULL", [-1.0, -1.0])]

    monkeypatch.setattr(vendored_inference, "infer_transformer_model", infer)
    result = n2s.generate_candidates(peaks, "C2H6O", spectrum_1h=_VALID_H)
    assert result["status"] == "ok"
    assert result["model"]["variant"] == variant
    assert result["calibrated_probability"] is False
    assert result["candidates"] == [{"smiles": "CCO", "origin": "nmr2struct_generated"}]
    assert received["input"].shape == (1, 28080)
    assert bool(np.any(received["input"][0, :28000])) is expected_h
    assert bool(np.any(received["input"][0, 28000:])) is expected_c
    assert received["opts"]["verbose"] is False


def test_concurrent_generation_never_redirects_process_stdout(
    monkeypatch, vendored_inference
):
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", "cnmr_only")
    original_stdout = sys.stdout
    first_entered, second_entered, first_finished = [
        threading.Event() for _ in range(3)
    ]
    observed = []

    def infer(model, batch, opts, device):
        observed.append(sys.stdout)
        if not first_entered.is_set():
            first_entered.set()
            assert second_entered.wait(5)
        else:
            second_entered.set()
            assert first_finished.wait(5)
        observed.append(sys.stdout)
        return [("NULL", ["CCO"], "NULL", [-1.0])]

    monkeypatch.setattr(vendored_inference, "infer_transformer_model", infer)

    def first_request():
        try:
            return n2s.generate_candidates([18.3, 58.1])
        finally:
            first_finished.set()

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(first_request)
        assert first_entered.wait(5)
        second = pool.submit(n2s.generate_candidates, [18.3, 58.1])
        assert first.result(timeout=10)["status"] == "ok"
        assert second.result(timeout=10)["status"] == "ok"
    assert all(stream is original_stdout for stream in observed)
    assert sys.stdout is original_stdout


def test_actual_vendored_sampler_with_deterministic_fixture_is_quiet(
    monkeypatch, vendored_inference, capsys
):
    """Exercise the real sampler, not trained-model accuracy or calibration."""
    import torch

    class DecodeFixture(torch.nn.Module):
        def forward(self, x, y, eval_paths):
            targets = y[0][0]
            batch_size, length = targets.shape
            logits = torch.full((batch_size, length, 24), -1000.0)
            logits[:, -1, [12, 12, 14, 23][min(length - 1, 3)]] = 1000.0
            return logits, None

    monkeypatch.setattr(n2s, "_load_model", lambda variant: DecodeFixture())
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", "cnmr_only")
    result = n2s.generate_candidates([18.3, 58.1], "C2H6O", top_k=2)
    assert result["status"] == "ok"
    assert result["candidates"] == [{"smiles": "CCO", "origin": "nmr2struct_generated"}]
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize("shift", [-20.01, 300.01, 1e9, -1e9])
def test_carbon_shift_sanity_range_matches_api(monkeypatch, shift):
    monkeypatch.setattr(
        n2s, "_load_model", lambda *args: pytest.fail("must not load a model")
    )
    assert not n2s.available([shift])
    assert (
        n2s.generate_candidates([shift])["reason"]
        == "13c_shifts_outside_supported_range"
    )


def test_carbon_boundary_bin_loss_is_disclosed(monkeypatch):
    monkeypatch.setattr(n2s, "_available_variant", lambda *args: True)
    for shift in (-20.0, 0.0, float(n2s._C_GRID[-1]), 300.0):
        selection = n2s.resolve_input_mode([shift])
        assert selection["status"] == "ok"
        assert selection["input_warnings"] == ["13c_shifts_use_boundary_bins"]
    assert n2s.resolve_input_mode([18.3])["input_warnings"] == []


def test_load_model_passes_h_only_channel_flags(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(n2s._VENDOR_DIR))
    import nmr.models

    monkeypatch.setattr(n2s, "_available_variant", lambda *args: True)

    selected = {}

    class ModelFixture:
        def load_state_dict(self, state):
            selected["state"] = state

        def eval(self):
            selected["eval"] = True

    model = ModelFixture()

    def create_model(args, dtype, device):
        selected["options"] = args["model_args"]["src_embed_options"]
        return model, args

    monkeypatch.setattr(nmr.models, "create_model", create_model)
    monkeypatch.setattr(
        torch, "load", lambda *args, **kwargs: {"model_state_dict": {"fixture": 1}}
    )
    assert n2s._load_model("hnmr_only") is model
    assert selected["options"]["use_hnmr"] is True
    assert selected["options"]["use_cnmr"] is False
    assert selected["state"] == {"fixture": 1}
    assert selected["eval"] is True


def test_real_h_only_architecture_responds_to_proton_signal(monkeypatch, capsys):
    """Random-initialized architecture check, not trained inference validation."""
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(str(n2s._VENDOR_DIR))
    from nmr.models import create_model
    from nmr.inference.inference_fxns import forward_multitask_transformer

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            model, _ = create_model(
                n2s._model_args("hnmr_only"), torch.float32, torch.device("cpu")
            )
            model.eval()
            assert (model.src_embed.use_hnmr, model.src_embed.use_cnmr) == (True, False)
            x = torch.zeros((1, 28080))
            y = torch.tensor([[22]])
            base = forward_multitask_transformer(model, x, y, False)
            changed_h = x.clone()
            changed_h[0, 1000:1500] = 1
            proton_output = forward_multitask_transformer(model, changed_h, y, False)
            changed_c = x.clone()
            changed_c[0, 28005:28010] = 1
            carbon_output = forward_multitask_transformer(model, changed_c, y, False)
            assert base.shape == (1, 1, 24)
            assert bool(torch.isfinite(proton_output).all())
            assert not torch.equal(base, proton_output)
            torch.testing.assert_close(base, carbon_output, rtol=0, atol=0)
        assert capsys.readouterr().out == ""
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("spectrum", [([0, 1], [0, 10**1000]), ([0, 10**1000], [0, 1])])
def test_h_conversion_overflow_is_fail_closed(monkeypatch, spectrum):
    monkeypatch.setattr(
        n2s, "_load_model", lambda *args: pytest.fail("must not load a model")
    )
    assert not n2s.available([18.3], spectrum_1h=spectrum)
    assert (
        n2s.generate_candidates([18.3], spectrum_1h=spectrum)["reason"]
        == "invalid_1h_spectrum"
    )


def test_constant_proton_baseline_is_not_signal(monkeypatch):
    monkeypatch.setattr(
        n2s, "_load_model", lambda *args: pytest.fail("must not load a model")
    )
    assert (
        n2s.generate_candidates([18.3], spectrum_1h=([0, 1, 2], [1, 1, 1]))["reason"]
        == "no_usable_1h_signal"
    )
