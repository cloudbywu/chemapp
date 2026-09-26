"""Integration contract test: forward_v1 predictions vs the v4 scorer atom_index validation.

The LocalForwardScorer provider must emit atom_index as 0-based RDKit atom
indices (atom.GetIdx()) so that its raw predictions can be fed straight into
the strict nmr_candidate_scorer_v4 contract without any +1/-1 adaptation.

Csp5ForwardScorer is different: its source file is SHA-256 pinned by the
frozen v9 development-evidence manifest
(docs/nmr-v9-development-evidence-manifest-v1.json), so it intentionally
keeps its historical 1-based float output. The csp5 test below locks the
fail-closed consequence: v4 rejects those predictions instead of silently
accepting misaligned indices. Any consumer bridging CSP5 output into the
v4-style contract must convert explicitly (subtract 1, cast to int) and is
responsible for that conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.ml.nmr_candidate_scorer_v4 import (
    NMRCandidateScorerV4InputError,
    score_candidate_evidence,
)


_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "ml"
    / "pretrained"
    / "forward_v1"
    / "forward_gnn_13c_v1.pt"
)
_CSP5_MODELS_DIR = (
    Path(__file__).resolve().parents[1]
    / "vendor"
    / "csp5"
    / "models"
    / "CSP5q-13C"
)


def _as_v4_candidate(
    candidate_id: str, smiles: str, predictions: list[dict[str, Any]]
) -> dict[str, Any]:
    """Project forward_v1 rows onto the v4 strict atom_prediction allowlist."""

    return {
        "candidate_id": candidate_id,
        "smiles": smiles,
        "atom_predictions": [
            {"atom_index": item["atom_index"], "shift_ppm": item["shift_ppm"]}
            for item in predictions
        ],
    }


def _assert_forward_predictions_are_zero_based_ints(
    predictions: list[dict[str, Any]],
) -> None:
    assert predictions, "forward scorer returned no carbon predictions"
    assert all(
        type(item["atom_index"]) is int and item["atom_index"] >= 0
        for item in predictions
    )
    indices = [item["atom_index"] for item in predictions]
    assert len(indices) == len(set(indices)), "atom_index values must be unique"


@pytest.mark.skipif(
    not _CHECKPOINT.exists(),
    reason="pretrained ForwardGNN-13C checkpoint is not present",
)
def test_local_forward_predictions_pass_v4_atom_index_validation() -> None:
    from app.ml.forward_v1.scorer import LocalForwardScorer

    scorer = LocalForwardScorer(checkpoint=_CHECKPOINT, device="cpu")
    smiles = "CCO"

    predictions = scorer.predict_molecule(smiles)
    _assert_forward_predictions_are_zero_based_ints(predictions)
    assert {item["atom_index"] for item in predictions} == {0, 1}

    candidate = _as_v4_candidate("ethanol", smiles, predictions)
    evidence = score_candidate_evidence([18.3, 58.1], candidate)
    assert evidence["candidate_id"] == "ethanol"
    assert evidence["v4"]["predicted_carbon_atom_count"] == 2

    # 批式推理路径必须与单分子路径遵循同一 atom_index 约定。
    batched = scorer.predict_molecules_batched([smiles])[0]
    _assert_forward_predictions_are_zero_based_ints(batched)
    assert {item["atom_index"] for item in batched} == {0, 1}
    batched_candidate = _as_v4_candidate("ethanol", smiles, batched)
    batched_evidence = score_candidate_evidence([18.3, 58.1], batched_candidate)
    assert batched_evidence["v4"]["predicted_carbon_atom_count"] == 2


pytest.importorskip("csp5", reason="vendored csp5 package is unavailable")

from app.ml.forward_v1.csp5_scorer import Csp5ForwardScorer  # noqa: E402


@pytest.mark.skipif(
    not (_CSP5_MODELS_DIR / "best_model.pt").exists(),
    reason="bundled CSP5 model weights are not present",
)
def test_csp5_output_is_rejected_by_v4_until_explicitly_converted() -> None:
    """CSP5 输出是 1-based float（冻结文件不可改），v4 必须 fail-closed 拒绝。

    该测试锁定已知的编号约定不一致：若未来解冻 csp5_scorer.py 并改为
    0-based int，本测试会失败，提醒同步更新为正向契约。
    """

    scorer = Csp5ForwardScorer(device="cpu")
    predictions = scorer.predict_molecule("CCO")
    assert predictions, "csp5 scorer returned no carbon predictions"

    candidate = _as_v4_candidate("ethanol", "CCO", predictions)
    with pytest.raises(NMRCandidateScorerV4InputError):
        score_candidate_evidence([18.3, 58.1], candidate)

    # 显式转换（-1 并取 int）后必须能通过 v4 校验——这是唯一合法的桥接方式。
    converted = [
        {"atom_index": int(item["atom_index"]) - 1, "shift_ppm": item["shift_ppm"]}
        for item in predictions
    ]
    evidence = score_candidate_evidence(
        [18.3, 58.1], _as_v4_candidate("ethanol", "CCO", converted)
    )
    assert evidence["v4"]["predicted_carbon_atom_count"] == 2


def test_v4_rejects_one_based_atom_indices() -> None:
    """负向控制：1-based / float 索引必须被 v4 强校验拒绝（fail-closed 不变）。"""

    one_based = {
        "candidate_id": "ethanol",
        "smiles": "CCO",
        "atom_predictions": [
            {"atom_index": 1, "shift_ppm": 18.3},
            {"atom_index": 2, "shift_ppm": 58.1},
        ],
    }
    with pytest.raises(NMRCandidateScorerV4InputError):
        score_candidate_evidence([18.3, 58.1], one_based)

    float_indices = {
        "candidate_id": "ethanol",
        "smiles": "CCO",
        "atom_predictions": [
            {"atom_index": 0.0, "shift_ppm": 18.3},
            {"atom_index": 1.0, "shift_ppm": 58.1},
        ],
    }
    with pytest.raises(NMRCandidateScorerV4InputError):
        score_candidate_evidence([18.3, 58.1], float_indices)
