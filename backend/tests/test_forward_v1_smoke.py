"""Smoke tests for the ForwardGNN-13C prototype."""

from pathlib import Path

import pytest
import torch

from app.ml.forward_v1.gnn import ForwardGNN13C, pinball_loss
from app.ml.forward_v1.graph_data import GraphSample, scaffold_group_split


_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "ml"
    / "pretrained"
    / "forward_v1"
    / "forward_gnn_13c_v1.pt"
)


def _fake_samples(n: int = 30) -> list[GraphSample]:
    samples = []
    for i in range(n):
        scaffold = f"scaffold{i % 6}"
        samples.append(
            GraphSample(
                sample_id=f"s{i}",
                inchikey=f"key{i}",
                scaffold=scaffold,
                canonical_smiles=f"C{i}",
                solvent="CDCl3",
                field_mhz=100.0,
                atom_symbols=("C",),
                mol_condition=torch.cat(
                    [
                        torch.zeros(12),
                        torch.as_tensor([100.0 / 800.0]),
                    ]
                ),
                node_feats=torch.randn(2, 48),
                edge_index=torch.as_tensor([[0, 1], [1, 0]]),
                edge_attr=torch.zeros(2, 10),
                targets=torch.as_tensor([100.0, 0.0]),
                target_mask=torch.as_tensor([True, False]),
                env_labels=torch.as_tensor([1, 0]),
            )
        )
    return samples


def test_pinball_loss_masked():
    pred = torch.as_tensor([1.0, 5.0, 3.0])
    target = torch.as_tensor([2.0, 4.0, 3.0])
    mask = torch.as_tensor([True, True, False])
    loss = pinball_loss(pred, target, mask, 0.5).item()
    assert abs(loss - 0.5) < 1e-5


def test_gnn_forward_shape_and_backward():
    model = ForwardGNN13C(in_dim=48, hidden=16, n_layers=2)
    feats = torch.randn(5, 48)
    edge_index = torch.as_tensor([[0, 1, 2, 3, 4, 1, 2, 3, 4, 0], [1, 2, 3, 4, 0, 0, 1, 2, 3, 4]])
    edge_attr = torch.randn(10, 10)
    condition = torch.randn(5, 13)
    out = model(feats, edge_index, edge_attr, torch.zeros(5, dtype=torch.long), condition)
    assert out.shape == (5, 3)
    loss = model.loss(
        feats, edge_index, edge_attr,
        torch.as_tensor([50.0, 60.0, 70.0, 80.0, 90.0]),
        torch.ones(5, dtype=torch.bool),
        torch.zeros(5, dtype=torch.long),
        condition,
    )
    loss.backward()
    assert loss.item() > 0


def test_scaffold_split_never_mixes_scaffolds():
    samples = _fake_samples(30)
    train, val, test = scaffold_group_split(samples, seed=0)
    train_scaffolds = {s.scaffold for s in train}
    assert not (train_scaffolds & {s.scaffold for s in val})
    assert not (train_scaffolds & {s.scaffold for s in test})


def test_graph_sample_json_roundtrip():
    sample = _fake_samples(1)[0]
    data = sample.to_json()
    assert data["n_targets"] == 1
    assert data["n_atoms"] == 2


def test_graph_cache_hash_sidecar_verification(tmp_path):
    from app.ml.forward_v1.graph_data import (
        _verify_cache_hash,
        _write_cache_with_hash,
    )

    path = tmp_path / "cache.pt"
    _write_cache_with_hash(path, {"x": torch.tensor([1.0])})
    assert path.exists()
    assert (tmp_path / "cache.pt.sha256").exists()
    assert _verify_cache_hash(path)

    # Tampered cache must fail verification.
    torch.save({"x": torch.tensor([2.0])}, str(path))
    assert not _verify_cache_hash(path)

    # Legacy cache without a sidecar must be treated as unverified.
    plain = tmp_path / "plain.pt"
    plain.write_bytes(b"legacy")
    assert not _verify_cache_hash(plain)


@pytest.mark.skipif(
    not _CHECKPOINT.exists(),
    reason="pretrained ForwardGNN-13C checkpoint is not present",
)
def test_local_scorer_denormalises_ppm_outputs():
    from app.ml.forward_v1.scorer import LocalForwardScorer

    scorer = LocalForwardScorer(checkpoint=_CHECKPOINT, device="cpu")
    predictions = scorer.predict_molecule("CCO")
    assert len(predictions) == 2
    # atom_index 必须为 0-based int（RDKit GetIdx() 约定）。
    assert all(type(item["atom_index"]) is int for item in predictions)
    assert {item["atom_index"] for item in predictions} == {0, 1}
    by_index = {int(item["atom_index"]): item for item in predictions}
    assert 10.0 <= by_index[0]["shift_ppm"] <= 25.0
    assert 50.0 <= by_index[1]["shift_ppm"] <= 65.0
    assert by_index[0]["q10_ppm"] <= by_index[0]["shift_ppm"]
    assert by_index[0]["shift_ppm"] <= by_index[0]["q90_ppm"]


@pytest.mark.skipif(
    not _CHECKPOINT.exists(),
    reason="pretrained ForwardGNN-13C checkpoint is not present",
)
def test_local_scorer_preserves_duplicate_observed_shifts():
    from app.ml.forward_v1.scorer import LocalForwardScorer

    scorer = LocalForwardScorer(checkpoint=_CHECKPOINT, device="cpu")
    result = scorer.score_candidates(
        [18.3, 18.3, 58.1],
        [{"candidate_id": "right", "smiles": "CCO"}],
    )
    assert result["status"] == "ok"
    assert result["observed_shifts_ppm"] == [18.3, 18.3, 58.1]
    top = result["candidates"][0]
    assert top["observed_count"] == 3
    assert top["predicted_atom_count"] == 2
    assert top["matched_count"] == 2
