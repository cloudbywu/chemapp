"""Optional trained-checkpoint compatibility probes, not an accuracy benchmark.

Each case skips when its own checkpoint is absent. Synthetic continuous proton
traces exercise the actual trained inference path; they are not experimental
spectra and cannot establish chemical accuracy or probability calibration.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.ml import nmr2struct_generator as n2s

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def isolate_generator(monkeypatch):
    monkeypatch.delenv("CHEMAPP_NMR2STRUCT_VARIANT", raising=False)
    monkeypatch.setattr(n2s, "_MODELS", {})
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(2, previous_threads))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            yield
    finally:
        torch.set_num_threads(previous_threads)


def _synthetic_proton_trace():
    x = n2s._H_GRID.copy()
    y = np.exp(-((x - 1.2) ** 2) / 0.01)
    y += 0.8 * np.exp(-((x - 3.6) ** 2) / 0.01)
    return x, y


@pytest.mark.parametrize(
    "variant",
    [
        pytest.param(
            variant,
            marks=pytest.mark.skipif(
                not n2s._available_variant(variant),
                reason=f"NMR2Struct {variant} checkpoint not present",
            ),
        )
        for variant in ("cnmr_only", "hnmr_only", "multitask")
    ],
)
def test_trained_logits_match_grad_mode_and_selected_channels(variant):
    """Strict load plus finite, mask-compatible logits and channel sensitivity."""
    model = n2s._load_model(variant)
    assert model is not None, f"{variant} checkpoint failed strict loading"
    from nmr.inference.inference_fxns import forward_multitask_transformer

    proton_axis, proton_signal = _synthetic_proton_trace()
    h = n2s._rasterize_1h(proton_axis, proton_signal, n2s._H_GRID)
    c = np.zeros(80)
    c[np.digitize([18.3, 58.1], n2s._C_GRID)] = 1
    x = torch.from_numpy(np.concatenate([h, c])).float().unsqueeze(0)
    changed_h = x.clone()
    changed_h[0, :28000] = torch.roll(changed_h[0, :28000], 2000)
    changed_c = x.clone()
    changed_c[0, 28000:] = 0
    changed_c[0, 28020] = changed_c[0, 28050] = 1
    use_h, use_c = n2s._CHANNELS[variant]
    assert (model.src_embed.use_hnmr, model.src_embed.use_cnmr) == (use_h, use_c)

    # Start-only, short aliphatic, and longer aromatic prefixes probe attention
    # masks without relying on stochastic sampled SMILES or target accuracy.
    for prefix in ([22], [22, 12, 12], [22, 18, 4, 18, 18, 18, 18, 18]):
        y = torch.tensor([prefix])
        outputs = []
        for source in (x, changed_h, changed_c):
            grad = forward_multitask_transformer(model, source, y, True)
            no_grad = forward_multitask_transformer(model, source, y, False)
            assert grad.shape == no_grad.shape == (1, len(prefix), 24)
            assert bool(torch.isfinite(grad).all())
            assert bool(torch.isfinite(no_grad).all())
            torch.testing.assert_close(grad, no_grad, atol=1e-5, rtol=1e-5)
            outputs.append(no_grad)
        baseline, proton_output, carbon_output = outputs
        for consumed, changed in ((use_h, proton_output), (use_c, carbon_output)):
            if consumed:
                assert float((baseline - changed).abs().max()) > 1e-5
            else:
                torch.testing.assert_close(baseline, changed, atol=0, rtol=0)


@pytest.mark.skipif(
    not n2s._available_variant("hnmr_only"),
    reason="NMR2Struct hnmr_only checkpoint not present",
)
def test_trained_h_only_generation_smoke(monkeypatch):
    """Exercise actual H-only loading, preprocessing, sampling, and decoding."""
    monkeypatch.setenv("CHEMAPP_NMR2STRUCT_VARIANT", "hnmr_only")
    result = n2s.generate_candidates(
        [], formula="C2H6O", top_k=3, spectrum_1h=_synthetic_proton_trace()
    )
    assert result["status"] == "ok"
    assert result["model"]["variant"] == "hnmr_only"
    assert result["input_mode"] == "1h_spectrum"
    assert result["provided_modalities"] == ["1h_spectrum"]
    assert result["used_modalities"] == ["1h_spectrum"]
    assert result["ignored_modalities"] == []
    assert result["calibrated_probability"] is False
    assert result["candidates"], "expected at least one decoded proposal"
    assert len(result["candidates"]) <= 3
    assert all(
        candidate["smiles"] and candidate["origin"] == "nmr2struct_generated"
        for candidate in result["candidates"]
    )
    assert result["inference_time_ms"] > 0
