from __future__ import annotations

import copy
import math

import pytest

from app.ml.nmr_candidate_scorer_v4 import (
    NMRCandidateScorerV4InputError,
    PROTOCOL_VERSION,
    SIGMA_13C_PPM,
    rank_candidate_evidence,
    score_candidate_evidence,
)


def _candidate(
    candidate_id: str,
    smiles: str,
    shifts_by_atom: dict[int, float],
) -> dict:
    return {
        "candidate_id": candidate_id,
        "smiles": smiles,
        "atom_predictions": [
            {"atom_index": atom_index, "shift_ppm": shift}
            for atom_index, shift in sorted(shifts_by_atom.items())
        ],
    }


def test_benzene_carbon_predictions_collapse_to_one_averaged_signal():
    candidate = _candidate(
        "benzene",
        "c1ccccc1",
        {
            0: 126.0,
            1: 127.0,
            2: 128.0,
            3: 129.0,
            4: 130.0,
            5: 131.0,
        },
    )

    result = score_candidate_evidence([128.5], candidate)

    assert result["protocol_version"] == PROTOCOL_VERSION
    assert result["calibrated_probability"] is False
    assert result["v4"]["predicted_carbon_atom_count"] == 6
    assert result["v4"]["predicted_symmetry_signal_count"] == 1
    assert result["v4"]["set_similarity"] == pytest.approx(1.0)
    assert result["v4"]["matched_mae_ppm"] == pytest.approx(0.0)
    assert result["v4"]["count_agreement"] == pytest.approx(1.0)
    assert result["v3"]["predicted_count"] == 6
    assert result["v3"]["predicted_coverage"] == pytest.approx(1 / 6)


def test_published_set_score_normalisation_penalises_signal_count_smoothly():
    propane = _candidate(
        "propane",
        "CCC",
        {0: 10.0, 1: 20.0, 2: 10.0},
    )

    result = score_candidate_evidence([10.0], propane)

    assert result["v4"]["sigma_13c_ppm"] == SIGMA_13C_PPM
    assert result["v4"]["predicted_symmetry_signal_count"] == 2
    assert result["v4"]["matched_kernel_sum"] == pytest.approx(1.0)
    assert result["v4"]["set_similarity"] == pytest.approx(1 / math.sqrt(2))
    assert result["v4"]["observed_coverage"] == pytest.approx(1.0)
    assert result["v4"]["predicted_coverage"] == pytest.approx(0.5)
    assert result["v4"]["bidirectional_coverage"] == pytest.approx(0.5)
    assert result["v4"]["count_agreement"] == pytest.approx(0.5)


def test_nontrivial_gaussian_kernel_and_error_metrics_follow_same_assignment():
    candidate = _candidate("ethanol", "CCO", {0: 10.0, 1: 22.0})

    result = score_candidate_evidence([11.0, 20.0], candidate)

    expected_kernel_sum = math.exp(-(1.0**2) / 200.0) + math.exp(-(2.0**2) / 200.0)
    assert result["v4"]["set_similarity"] == pytest.approx(expected_kernel_sum / 2)
    assert result["v4"]["matched_mae_ppm"] == pytest.approx(1.5)
    assert result["v4"]["matched_rmse_ppm"] == pytest.approx(math.sqrt(2.5))
    assert result["v4"]["matched_max_abs_error_ppm"] == pytest.approx(2.0)
    assert len(result["v4"]["matching_sha256"]) == 64


def test_rank_function_returns_frozen_v3_and_v4_orders_without_probability():
    symmetry_aware_winner = _candidate(
        "symmetry-aware",
        "CCC",
        {0: 10.0, 1: 20.0, 2: 10.0},
    )
    complete_atom_winner = _candidate(
        "complete-atom",
        "CCO",
        {0: 10.2, 1: 20.2},
    )

    result = rank_candidate_evidence(
        [10.0, 20.0],
        [symmetry_aware_winner, complete_atom_winner],
    )

    assert result["calibrated_probability"] is False
    assert result["v3_order"] == ["complete-atom", "symmetry-aware"]
    assert result["v4_order"] == ["symmetry-aware", "complete-atom"]
    assert result["v4_rank_basis"] == (
        "higher_symmetry_gaussian_set_similarity_then_lower_matched_max_"
        "error_then_lower_matched_mae_then_candidate_id"
    )
    assert {item["v3_rank"] for item in result["candidates"]} == {1, 2}
    assert {item["v4_rank"] for item in result["candidates"]} == {1, 2}
    assert set(result["runtime"]) == {
        "rdkit_version",
        "scipy_version",
        "numpy_version",
    }


def test_candidate_scores_and_pairwise_order_are_pool_and_input_order_invariant():
    first = _candidate("first", "CCO", {0: 10.0, 1: 20.0})
    second = _candidate("second", "COC", {0: 11.0, 2: 11.0})
    irrelevant = _candidate("irrelevant", "C", {0: 100.0})

    pair = rank_candidate_evidence([10.0, 20.0], [first, second])
    reversed_pair = rank_candidate_evidence([10.0, 20.0], [second, first])
    superset = rank_candidate_evidence(
        [10.0, 20.0],
        [irrelevant, second, first],
    )

    assert pair["v3_order"] == reversed_pair["v3_order"]
    assert pair["v4_order"] == reversed_pair["v4_order"]
    assert pair["candidates"] == reversed_pair["candidates"]
    pair_by_id = {item["candidate_id"]: item for item in pair["candidates"]}
    superset_by_id = {item["candidate_id"]: item for item in superset["candidates"]}
    for candidate_id in ("first", "second"):
        assert pair_by_id[candidate_id]["v3"] == superset_by_id[candidate_id]["v3"]
        assert pair_by_id[candidate_id]["v4"] == superset_by_id[candidate_id]["v4"]
    assert (
        pair["v4_order"].index("first") < pair["v4_order"].index("second")
    ) is (
        superset["v4_order"].index("first")
        < superset["v4_order"].index("second")
    )


def test_smiles_atom_renumbering_with_prediction_remap_preserves_metrics():
    forward = _candidate("forward", "CCO", {0: 10.0, 1: 20.0})
    reversed_atoms = _candidate("reversed", "OCC", {1: 20.0, 2: 10.0})

    forward_result = score_candidate_evidence([10.0, 20.0], forward)
    reverse_result = score_candidate_evidence([20.0, 10.0], reversed_atoms)

    for version in ("v3", "v4"):
        left = {
            key: value
            for key, value in forward_result[version].items()
            if key not in {"matching_sha256", "symmetry_classes_sha256"}
        }
        right = {
            key: value
            for key, value in reverse_result[version].items()
            if key not in {"matching_sha256", "symmetry_classes_sha256"}
        }
        assert left == right


def test_fixed_v4_tie_break_falls_through_to_candidate_id_for_exact_metric_tie():
    lower_max = _candidate("z-lower-max", "CCO", {0: 0.0, 1: 20.0})
    higher_max = _candidate("a-higher-max", "CCN", {0: 10.0, 1: 30.0})

    first = score_candidate_evidence([0.0, 30.0], lower_max)
    second = score_candidate_evidence([0.0, 30.0], higher_max)
    # The symmetric Gaussian sums are equal: errors are {0, 10} in both cases.
    assert first["v4"]["set_similarity"] == pytest.approx(
        second["v4"]["set_similarity"]
    )
    assert first["v4"]["matched_max_abs_error_ppm"] == pytest.approx(
        second["v4"]["matched_max_abs_error_ppm"]
    )

    ranked = rank_candidate_evidence([0.0, 30.0], [lower_max, higher_max])
    assert ranked["v4_order"] == ["a-higher-max", "z-lower-max"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update({"role": "truth"}), "strict allowlist"),
        (
            lambda row: row["atom_predictions"][0].update({"assignment": "gold"}),
            "strict allowlist",
        ),
        (
            lambda row: row["atom_predictions"][0].update({"shift_ppm": math.nan}),
            "finite",
        ),
        (
            lambda row: row["atom_predictions"].pop(),
            "prediction atom set",
        ),
        (
            lambda row: row.update({"smiles": "C.C"}),
            "exactly one fragment",
        ),
    ],
)
def test_strict_allowlist_and_numeric_structure_validation_fail_closed(
    mutation,
    message: str,
):
    candidate = _candidate("strict", "CCO", {0: 10.0, 1: 20.0})
    mutation(candidate)

    with pytest.raises(NMRCandidateScorerV4InputError, match=message):
        score_candidate_evidence([10.0, 20.0], candidate)


@pytest.mark.parametrize(
    "observed",
    [
        [],
        [math.inf],
        [True],
        "10,20",
    ],
)
def test_invalid_observed_signal_lists_fail_closed(observed):
    candidate = _candidate("valid", "CCO", {0: 10.0, 1: 20.0})

    with pytest.raises(NMRCandidateScorerV4InputError):
        score_candidate_evidence(observed, candidate)


def test_duplicate_candidate_ids_fail_closed():
    candidate = _candidate("duplicate", "CCO", {0: 10.0, 1: 20.0})

    with pytest.raises(NMRCandidateScorerV4InputError, match="must be unique"):
        rank_candidate_evidence(
            [10.0, 20.0],
            [candidate, copy.deepcopy(candidate)],
        )


def test_duplicate_canonical_structures_fail_closed():
    first = _candidate("first", "CCO", {0: 10.0, 1: 20.0})
    duplicate = _candidate("duplicate", "OCC", {1: 20.0, 2: 10.0})

    with pytest.raises(NMRCandidateScorerV4InputError, match="must be unique"):
        rank_candidate_evidence([10.0, 20.0], [first, duplicate])
