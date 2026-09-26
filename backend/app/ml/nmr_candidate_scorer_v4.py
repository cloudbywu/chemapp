"""Pure, outcome-blind candidate scoring for experimental 13C peak lists.

The v4 score follows equations 6 and 7 of the published NMR-Solver method:
topologically symmetry-equivalent carbon predictions are averaged, a Gaussian
compatibility kernel is maximised with a one-to-one Hungarian assignment, and
the matched kernel sum is normalised by the geometric mean of the two signal
counts.

This module deliberately has no access to a Gold structure, candidate origin,
source document, split, outcome, or calibrated probability.  Those bindings
must be applied by a separate evaluation layer after all candidates have been
scored.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
import math
import re
from typing import Any

import numpy as np
from rdkit import Chem, rdBase
import scipy
from scipy.optimize import linear_sum_assignment


PROTOCOL_VERSION = "chemapp.nmr.candidate-scorer.v4"
SCORE_SEMANTICS = "symmetry_collapsed_13c_gaussian_set_similarity_v1"
V3_SCORE_SEMANTICS = "atom_level_13c_hungarian_mae_v1"
SYMMETRY_METHOD = "rdkit_addhs_canonical_rank_atoms_break_ties_false_v1"
SIGMA_13C_PPM = 10.0
MAX_OBSERVED_SIGNALS = 256
MAX_CANDIDATES = 100
MAX_SMILES_LENGTH = 4096

_CANDIDATE_KEYS = frozenset({"candidate_id", "smiles", "atom_predictions"})
_ATOM_PREDICTION_KEYS = frozenset({"atom_index", "shift_ppm"})
_CANDIDATE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class NMRCandidateScorerV4InputError(ValueError):
    """Raised when outcome-blind v4 scorer input violates its strict contract."""


def _strict_keys(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != allowed:
        missing = sorted(allowed - actual)
        unexpected = sorted(actual - allowed)
        raise NMRCandidateScorerV4InputError(
            f"{context} fields do not match the strict allowlist; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _finite_shift(value: Any, *, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise NMRCandidateScorerV4InputError(f"{context} must be a finite number")
    shift = float(value)
    if not -20.0 <= shift <= 300.0:
        raise NMRCandidateScorerV4InputError(
            f"{context} must be within the supported 13C range"
        )
    return shift


def _normalise_observed(observed_13c: Iterable[Any]) -> list[float]:
    if isinstance(observed_13c, (str, bytes, Mapping)):
        raise NMRCandidateScorerV4InputError(
            "observed_13c must be a non-empty iterable of shifts"
        )
    try:
        shifts = [
            _finite_shift(value, context=f"observed_13c[{index}]")
            for index, value in enumerate(observed_13c)
        ]
    except TypeError as exc:
        raise NMRCandidateScorerV4InputError(
            "observed_13c must be a non-empty iterable of shifts"
        ) from exc
    if not shifts:
        raise NMRCandidateScorerV4InputError(
            "observed_13c must contain at least one signal"
        )
    if len(shifts) > MAX_OBSERVED_SIGNALS:
        raise NMRCandidateScorerV4InputError(
            f"observed_13c exceeds the {MAX_OBSERVED_SIGNALS}-signal limit"
        )
    return sorted(shifts)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _normalise_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise NMRCandidateScorerV4InputError("each candidate must be an object")
    _strict_keys(candidate, _CANDIDATE_KEYS, context="candidate")

    candidate_id = candidate.get("candidate_id")
    if (
        not isinstance(candidate_id, str)
        or _CANDIDATE_ID_RE.fullmatch(candidate_id) is None
    ):
        raise NMRCandidateScorerV4InputError(
            "candidate_id must satisfy the fixed opaque-ID contract"
        )
    smiles = candidate.get("smiles")
    if (
        not isinstance(smiles, str)
        or not smiles
        or smiles != smiles.strip()
        or len(smiles) > MAX_SMILES_LENGTH
    ):
        raise NMRCandidateScorerV4InputError(
            "candidate smiles must be a bounded, non-empty stripped string"
        )
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise NMRCandidateScorerV4InputError(
            f"candidate {candidate_id} has invalid SMILES"
        )
    if len(Chem.GetMolFrags(molecule)) != 1:
        raise NMRCandidateScorerV4InputError(
            f"candidate {candidate_id} must contain exactly one fragment"
        )
    carbon_indices = {
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6
    }
    if not carbon_indices:
        raise NMRCandidateScorerV4InputError(
            f"candidate {candidate_id} has no carbon atoms"
        )

    raw_predictions = candidate.get("atom_predictions")
    if not isinstance(raw_predictions, list) or not raw_predictions:
        raise NMRCandidateScorerV4InputError(
            f"candidate {candidate_id} atom_predictions must be a non-empty list"
        )
    predictions: list[dict[str, float | int]] = []
    seen_indices: set[int] = set()
    for position, raw_prediction in enumerate(raw_predictions):
        if not isinstance(raw_prediction, Mapping):
            raise NMRCandidateScorerV4InputError(
                f"candidate {candidate_id} atom_predictions[{position}] "
                "must be an object"
            )
        _strict_keys(
            raw_prediction,
            _ATOM_PREDICTION_KEYS,
            context=f"candidate {candidate_id} atom prediction",
        )
        atom_index = raw_prediction.get("atom_index")
        if (
            isinstance(atom_index, bool)
            or not isinstance(atom_index, int)
            or atom_index < 0
            or atom_index in seen_indices
        ):
            raise NMRCandidateScorerV4InputError(
                f"candidate {candidate_id} atom indices must be unique "
                "non-negative integers"
            )
        seen_indices.add(atom_index)
        predictions.append(
            {
                "atom_index": atom_index,
                "shift_ppm": _finite_shift(
                    raw_prediction.get("shift_ppm"),
                    context=f"candidate {candidate_id} atom {atom_index} shift",
                ),
            }
        )
    if seen_indices != carbon_indices:
        raise NMRCandidateScorerV4InputError(
            f"candidate {candidate_id} prediction atom set must equal its carbon set"
        )

    canonical_smiles = Chem.MolToSmiles(
        molecule,
        canonical=True,
        isomericSmiles=True,
    )
    return {
        "candidate_id": candidate_id,
        "smiles": smiles,
        "canonical_smiles": canonical_smiles,
        "molecule": molecule,
        "atom_predictions": sorted(
            predictions,
            key=lambda item: int(item["atom_index"]),
        ),
    }


def _symmetry_collapsed_signals(candidate: Mapping[str, Any]) -> dict[str, Any]:
    molecule = candidate["molecule"]
    hydrogenated = Chem.AddHs(molecule)
    symmetry_ranks = list(Chem.CanonicalRankAtoms(hydrogenated, breakTies=False))
    predictions = {
        int(item["atom_index"]): float(item["shift_ppm"])
        for item in candidate["atom_predictions"]
    }
    classes: dict[int, list[tuple[int, float]]] = {}
    for atom in molecule.GetAtoms():
        if atom.GetAtomicNum() != 6:
            continue
        atom_index = atom.GetIdx()
        classes.setdefault(symmetry_ranks[atom_index], []).append(
            (atom_index, predictions[atom_index])
        )

    normalised_classes = sorted(
        (
            {
                "atom_indices": sorted(atom_index for atom_index, _ in members),
                "mean_shift_ppm": sum(shift for _, shift in members) / len(members),
            }
            for members in classes.values()
        ),
        key=lambda item: (
            float(item["mean_shift_ppm"]),
            tuple(item["atom_indices"]),
        ),
    )
    signals = [float(item["mean_shift_ppm"]) for item in normalised_classes]
    class_binding = [
        {
            "atom_indices": item["atom_indices"],
            "mean_shift_ppm": item["mean_shift_ppm"],
        }
        for item in normalised_classes
    ]
    return {
        "signals": signals,
        "class_count": len(class_binding),
        "classes_sha256": _canonical_sha256(class_binding),
    }


def _assignment_metrics(
    observed: Sequence[float],
    predicted: Sequence[float],
    *,
    gaussian_sigma: float | None,
) -> dict[str, Any]:
    observed_array = np.asarray(observed, dtype=np.float64)
    predicted_array = np.asarray(predicted, dtype=np.float64)
    errors = np.abs(observed_array[:, np.newaxis] - predicted_array[np.newaxis, :])
    if gaussian_sigma is None:
        row_indices, column_indices = linear_sum_assignment(errors)
        kernel_values: np.ndarray | None = None
    else:
        kernel = np.exp(
            -(errors**2) / (2.0 * gaussian_sigma * gaussian_sigma),
            dtype=np.float64,
        )
        row_indices, column_indices = linear_sum_assignment(-kernel)
        kernel_values = kernel[row_indices, column_indices]

    matched_errors = errors[row_indices, column_indices]
    matched_count = int(len(matched_errors))
    observed_count = int(len(observed))
    predicted_count = int(len(predicted))
    observed_coverage = matched_count / observed_count
    predicted_coverage = matched_count / predicted_count
    matching_binding = [
        {
            "observed_index": int(row),
            "predicted_index": int(column),
            "observed_shift_ppm": float(observed[row]),
            "predicted_shift_ppm": float(predicted[column]),
            "abs_error_ppm": float(error),
        }
        for row, column, error in zip(
            row_indices,
            column_indices,
            matched_errors,
            strict=True,
        )
    ]
    output = {
        "matched_count": matched_count,
        "observed_count": observed_count,
        "predicted_count": predicted_count,
        "unmatched_observed_count": observed_count - matched_count,
        "unmatched_predicted_count": predicted_count - matched_count,
        "observed_coverage": float(observed_coverage),
        "predicted_coverage": float(predicted_coverage),
        "bidirectional_coverage": float(
            min(observed_coverage, predicted_coverage)
        ),
        "count_agreement": float(
            min(observed_count, predicted_count)
            / max(observed_count, predicted_count)
        ),
        "matched_mae_ppm": float(np.mean(matched_errors)),
        "matched_rmse_ppm": float(np.sqrt(np.mean(matched_errors**2))),
        "matched_max_abs_error_ppm": float(np.max(matched_errors)),
        "matching_sha256": _canonical_sha256(matching_binding),
    }
    if kernel_values is not None:
        output["matched_kernel_sum"] = float(np.sum(kernel_values))
        output["set_similarity"] = float(
            np.sum(kernel_values) / math.sqrt(observed_count * predicted_count)
        )
    return output


def score_candidate_evidence(
    observed_13c: Iterable[Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Score one outcome-blind candidate with both frozen v3 and v4 semantics."""

    observed = _normalise_observed(observed_13c)
    clean = _normalise_candidate(candidate)
    atom_signals = [
        float(item["shift_ppm"]) for item in clean["atom_predictions"]
    ]
    collapsed = _symmetry_collapsed_signals(clean)
    atom_metrics = _assignment_metrics(
        observed,
        atom_signals,
        gaussian_sigma=None,
    )
    set_metrics = _assignment_metrics(
        observed,
        collapsed["signals"],
        gaussian_sigma=SIGMA_13C_PPM,
    )
    set_similarity = float(set_metrics["set_similarity"])
    if not 0.0 <= set_similarity <= 1.0 + 1e-12:
        raise AssertionError("v4 set similarity escaped its mathematical range")
    set_similarity = min(set_similarity, 1.0)

    return {
        "protocol_version": PROTOCOL_VERSION,
        "candidate_id": clean["candidate_id"],
        "canonical_smiles": clean["canonical_smiles"],
        "calibrated_probability": False,
        "v3": {
            "score_semantics": V3_SCORE_SEMANTICS,
            "candidate_score": -float(atom_metrics["matched_mae_ppm"]),
            **atom_metrics,
        },
        "v4": {
            "score_semantics": SCORE_SEMANTICS,
            "candidate_score": set_similarity,
            "set_similarity": set_similarity,
            "sigma_13c_ppm": SIGMA_13C_PPM,
            "symmetry_method": SYMMETRY_METHOD,
            "predicted_carbon_atom_count": len(atom_signals),
            "predicted_symmetry_signal_count": collapsed["class_count"],
            "symmetry_classes_sha256": collapsed["classes_sha256"],
            **{
                key: value
                for key, value in set_metrics.items()
                if key != "set_similarity"
            },
        },
    }


def _v3_rank_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = item["v3"]
    return (
        -float(metrics["bidirectional_coverage"]),
        int(metrics["unmatched_observed_count"])
        + int(metrics["unmatched_predicted_count"]),
        float(metrics["matched_mae_ppm"]),
        float(metrics["matched_rmse_ppm"]),
        str(item["candidate_id"]),
    )


def _v4_rank_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    metrics = item["v4"]
    return (
        -float(metrics["set_similarity"]),
        float(metrics["matched_max_abs_error_ppm"]),
        float(metrics["matched_mae_ppm"]),
        str(item["candidate_id"]),
    )


def rank_candidate_evidence(
    observed_13c: Iterable[Any],
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Produce pool-neutral v3 and v4 rankings from one candidate evidence set."""

    observed = _normalise_observed(observed_13c)
    if isinstance(candidates, (str, bytes, Mapping)) or not isinstance(
        candidates, Sequence
    ):
        raise NMRCandidateScorerV4InputError("candidates must be a sequence")
    if not 1 <= len(candidates) <= MAX_CANDIDATES:
        raise NMRCandidateScorerV4InputError(
            f"candidates must contain between 1 and {MAX_CANDIDATES} rows"
        )

    scored = [score_candidate_evidence(observed, candidate) for candidate in candidates]
    candidate_ids = [str(item["candidate_id"]) for item in scored]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise NMRCandidateScorerV4InputError("candidate_id values must be unique")
    canonical_smiles = [str(item["canonical_smiles"]) for item in scored]
    if len(set(canonical_smiles)) != len(canonical_smiles):
        raise NMRCandidateScorerV4InputError(
            "candidate structures must be unique after canonicalisation"
        )
    v3_ordered = sorted(scored, key=_v3_rank_key)
    v4_ordered = sorted(scored, key=_v4_rank_key)
    v3_rank = {
        str(item["candidate_id"]): rank
        for rank, item in enumerate(v3_ordered, start=1)
    }
    v4_rank = {
        str(item["candidate_id"]): rank
        for rank, item in enumerate(v4_ordered, start=1)
    }
    stable_scored = [
        {
            **item,
            "v3_rank": v3_rank[str(item["candidate_id"])],
            "v4_rank": v4_rank[str(item["candidate_id"])],
        }
        for item in sorted(scored, key=lambda row: str(row["candidate_id"]))
    ]
    return {
        "protocol_version": PROTOCOL_VERSION,
        "calibrated_probability": False,
        "observed_13c": observed,
        "observed_13c_sha256": _canonical_sha256(observed),
        "candidate_count": len(scored),
        "v3_rank_basis": (
            "higher_atom_bidirectional_coverage_then_fewer_unmatched_then_"
            "lower_atom_hungarian_mae_rmse_then_candidate_id"
        ),
        "v4_rank_basis": (
            "higher_symmetry_gaussian_set_similarity_then_lower_matched_max_"
            "error_then_lower_matched_mae_then_candidate_id"
        ),
        "v3_order": [str(item["candidate_id"]) for item in v3_ordered],
        "v4_order": [str(item["candidate_id"]) for item in v4_ordered],
        "candidates": stable_scored,
        "runtime": {
            "rdkit_version": rdBase.rdkitVersion,
            "scipy_version": scipy.__version__,
            "numpy_version": np.__version__,
        },
    }


__all__ = [
    "MAX_CANDIDATES",
    "NMRCandidateScorerV4InputError",
    "PROTOCOL_VERSION",
    "SCORE_SEMANTICS",
    "SIGMA_13C_PPM",
    "SYMMETRY_METHOD",
    "V3_SCORE_SEMANTICS",
    "rank_candidate_evidence",
    "score_candidate_evidence",
]
