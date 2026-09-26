"""Smoke tests for the research hybrid Top-1 calibration module."""

from __future__ import annotations

from app.ml.nmr_hybrid_calibration_v1 import (
    conditional_rows,
    run_hybrid_calibration,
)


def _case(
    record_id: str,
    group: str,
    *,
    correct: bool,
    n: int = 5,
) -> dict:
    top1_mae = 0.4 if correct else 6.0
    runnerup_mae = 5.0
    scores = [
        {"mae_ppm": top1_mae if i == 0 else runnerup_mae + (i % 3) * 0.5}
        for i in range(n)
    ]
    return {
        "record_id": record_id,
        "status": "ok",
        "truth_in_pool": True,
        "top1_correct": correct,
        "molecule_group_id": group,
        "calibration_features": {
            "top1_mae_ppm": top1_mae,
            "top1_rmse_ppm": top1_mae + 0.1,
            "top1_max_abs_error_ppm": top1_mae + 0.2,
            "top1_bidirectional_coverage": 1.0,
            "top1_matched_count": 5,
            "observed_count": 5,
            "runnerup_mae_ppm": runnerup_mae,
            "runnerup_bidirectional_coverage": 1.0,
            "log_candidate_count": 1.6,
            "pool_size": n,
        },
        "candidate_scores": scores,
        "gcn_top1_smiles": "A",
        "csp5_top1_smiles": "A" if correct else "B",
    }


def test_conditional_rows_and_blocked_gate_with_few_errors() -> None:
    cases = [_case(f"r{i}", f"g{i % 3}", correct=i % 5 == 0) for i in range(8)]
    rows = conditional_rows(cases)
    assert len(rows) == 8
    assert len(rows[0]["values"]) == 7
    report = run_hybrid_calibration(cases)
    assert report["status"] == "blocked"
    assert report["probability_claim_allowed"] is False
    assert any("insufficient_errors" in reason for reason in report["blocking_reasons"])
