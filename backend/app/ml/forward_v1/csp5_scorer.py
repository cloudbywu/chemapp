"""CSP5-backed 13C forward scorer.

This scorer wraps the vendored ``csp5`` package (see ``backend/vendor/csp5``)
and its bundled ``CSP5q-13C`` quantile model.  It implements the same
``ForwardScorer``-compatible protocol as :class:`LocalForwardScorer` and
returns the same relative 13C forward-fit evidence schema, so it can be
plugged directly into ``nmr_hybrid_predictor``.

Reference: Rowlands et al., "CSP5: Large-scale Neural Chemical Shift
Prediction from 2.5 Million Experimental NMR Spectra"
(Zenodo record 10.5281/zenodo.19486118; package: PyPI csp5 0.2.18, MIT).
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import threading
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.optimize import linear_sum_assignment

_VENDOR_DIR = Path(__file__).resolve().parents[3] / "vendor"
if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from csp5.api import predict_mols  # noqa: E402
from csp5.model_registry import resolve_model_weights  # noqa: E402


PROVIDER_ID = "csp5_forward_13c_v1"
DEFAULT_MODEL_NAME = "CSP5q-13C"

# NOTE: solvent-aware CSP5 fine-tunes are not wired into this scorer; the
# bundled CSP5q-13C base model is used for every solvent.  If solvent-aware
# checkpoints are added, pass the normalized solvent into ``predict_mols``.

_WEIGHT_SHA_CACHE: dict[tuple[str, int, int], str] = {}
_WEIGHT_SHA_LOCK = threading.Lock()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_sha256_file(path: Path) -> str:
    """SHA-256 a weight file once per (path, size, mtime) triple."""

    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    with _WEIGHT_SHA_LOCK:
        cached = _WEIGHT_SHA_CACHE.get(key)
        if cached is not None:
            return cached
    digest = _sha256_file(path)
    with _WEIGHT_SHA_LOCK:
        _WEIGHT_SHA_CACHE[key] = digest
    return digest


def _finite_float(value: Any, default: float | None = None) -> float | None:
    """Return a finite float, or ``default`` when the value is unusable."""

    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _mae_sort_key(item: Mapping[str, Any]) -> float:
    """MAE sort key that treats None as infinity and keeps 0.0 as zero."""

    mae = _finite_float(item.get("mae_ppm"))
    return float("inf") if mae is None else mae


def _prediction_failed_evidence(
    candidate: Mapping[str, Any],
    *,
    observed_count: int,
    reason: str,
    model_name: str,
    weights_sha256: str,
    detail: str | None = None,
) -> dict[str, Any]:
    """Return a ranked-last evidence entry for a candidate that failed."""

    return {
        "candidate_id": str(
            candidate.get("candidate_id") or candidate.get("smiles")
        ),
        "relative_rank": 0,
        "assignment_mode": "none_prediction_failed",
        "diagnostic_only": True,
        "prediction_failed": True,
        "matched_count": 0,
        "observed_count": observed_count,
        "predicted_atom_count": 0,
        "unmatched_observed_count": observed_count,
        "unmatched_predicted_atom_count": 0,
        "observed_coverage": 0.0,
        "predicted_atom_coverage": 0.0,
        "bidirectional_coverage": 0.0,
        "assignment_complete": False,
        "mae_ppm": None,
        "rmse_ppm": None,
        "max_abs_error_ppm": None,
        "calibrated_probability": False,
        "quantile_enabled": True,
        "prediction": {
            "provider": PROVIDER_ID,
            "model_name": model_name,
            "model_sha256": weights_sha256,
            "status": "prediction_failed",
            "reason": reason,
            "detail": detail,
            "atom_predictions": [],
        },
    }


def _embed_smiles(smiles: str, *, seed: int | None = None) -> Chem.Mol:
    """Return a hydrogen-added RDKit mol with an optimized 3D conformer.

    Heavy atoms keep their original 0-based indices after ``AddHs``, so
    prediction rows can be mapped straight back to the input SMILES atom
    numbering.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    if any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        mol = Chem.RemoveHs(mol, sanitize=False)
    mol_h = Chem.AddHs(Chem.Mol(mol))
    if seed is None:
        digest = hashlib.sha256(smiles.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], "little") & 0x7FFFFFFF

    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    status = AllChem.EmbedMolecule(mol_h, params=params)
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(mol_h, params=params)
    if status != 0:
        raise ValueError(f"conformer embedding failed for SMILES: {smiles}")

    optimized = False
    if AllChem.MMFFHasAllMoleculeParams(mol_h):
        try:
            optimized = AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200) in (0, 1)
        except Exception:
            optimized = False
    if not optimized:
        try:
            AllChem.UFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
    return mol_h


def _target_prediction_problem(
    molecule: Chem.Mol,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, str] | None:
    """Validate that CSP5 returned one finite 13C row per carbon atom."""

    expected_indices = {
        atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 6
    }
    if not expected_indices:
        return "no_target_atoms", "candidate contains no carbon atoms"
    if not rows:
        return (
            "missing_target_predictions",
            f"expected {len(expected_indices)} carbon predictions, received 0",
        )

    returned_indices: list[int] = []
    for row in rows:
        raw_index = _finite_float(row.get("atom_index"))
        shift = _finite_float(row.get("shift_ppm"))
        if raw_index is None or not raw_index.is_integer() or shift is None:
            return (
                "invalid_target_predictions",
                "a prediction has a non-integer atom index or non-finite shift",
            )
        returned_indices.append(int(raw_index))
        for key in ("shift_q10_ppm", "shift_q90_ppm", "shift_std_ppm"):
            if key in row and _finite_float(row.get(key)) is None:
                return (
                    "invalid_target_predictions",
                    f"a prediction has a non-finite {key}",
                )

    returned_set = set(returned_indices)
    if len(returned_indices) != len(returned_set):
        return "invalid_target_predictions", "duplicate carbon atom predictions"
    if returned_set != expected_indices:
        missing = sorted(expected_indices - returned_set)
        unexpected = sorted(returned_set - expected_indices)
        return (
            "incomplete_target_predictions",
            f"missing atom indices {missing}; unexpected atom indices {unexpected}",
        )
    return None


class Csp5PredictionIntegrityError(RuntimeError):
    """Raised when CSP5 returns an incomplete or invalid target-nucleus result."""


class Csp5ForwardScorer:
    """ForwardScorer-compatible scorer using the CSP5q-13C quantile model."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
        batch_size: int = 32,
        weights_sha256: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = int(batch_size)
        self.device = device or os.environ.get("CHEMAPP_CSP5_DEVICE", "auto")
        self.weights_sha256 = weights_sha256 or ""
        if not self.weights_sha256:
            try:
                from csp5.model_registry import get_model_spec

                spec = get_model_spec("13C", model_name=model_name)
                if not spec.weights_path.exists() and os.environ.get(
                    "CSP5_ALLOW_DOWNLOAD", "0"
                ).strip().casefold() not in {"1", "on", "true", "yes"}:
                    raise FileNotFoundError(
                        f"bundled CSP5 weights missing at {spec.weights_path}; "
                        "restore them from the csp5 sdist/Zenodo record or set "
                        "CSP5_ALLOW_DOWNLOAD=1 for on-demand fetch"
                    )
                path = resolve_model_weights(spec)
                self.weights_sha256 = _cached_sha256_file(path)
                self.model_sha256 = self.weights_sha256
            except FileNotFoundError:
                raise
            except Exception:
                self.weights_sha256 = ""
                self.model_sha256 = ""

    def predict_molecule(
        self,
        smiles: str,
        solvent: str | None = None,
        field_mhz: float | None = None,
    ) -> list[dict[str, float]]:
        mol_h = _embed_smiles(smiles)
        result = predict_mols(
            [mol_h],
            smiles=[Chem.MolToSmiles(Chem.RemoveHs(mol_h))],
            nucleus="13C",
            model_name=self.model_name,
            device=self.device,
            batch_size=self.batch_size,
        )
        if result.failures:
            raise RuntimeError(f"CSP5 prediction failed: {result.failures[0]}")
        rows = result.predictions
        raw_predictions = [row.to_dict() for _, row in rows.iterrows()]
        prediction_problem = _target_prediction_problem(mol_h, raw_predictions)
        if prediction_problem is not None:
            reason, detail = prediction_problem
            raise Csp5PredictionIntegrityError(
                f"CSP5 13C prediction integrity failure ({reason}): {detail}"
            )
        predictions: list[dict[str, float]] = []
        for row in raw_predictions:
            atom_index = int(row["atom_index"])
            entry: dict[str, float] = {
                "atom_index": float(atom_index + 1),
                "shift_ppm": round(float(row["shift_ppm"]), 3),
            }
            for key, target in (
                ("shift_q10_ppm", "q10_ppm"),
                ("shift_q90_ppm", "q90_ppm"),
                ("shift_std_ppm", "shift_std_ppm"),
            ):
                if key in row:
                    entry[target] = round(float(row[key]), 3)
            predictions.append(entry)
        predictions.sort(key=lambda item: float(item["atom_index"]))
        return predictions

    def score_candidates(
        self,
        observed_13c: Iterable[float | Mapping[str, Any]] | None,
        candidates: Sequence[Mapping[str, Any]],
        *,
        formula: str | None = None,
    ) -> dict[str, Any]:
        shifts: list[float] = []
        if observed_13c is not None:
            for item in observed_13c:
                raw = item.get("shift") if isinstance(item, Mapping) else item
                if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                    continue
                value = float(raw)
                if math.isfinite(value) and -20.0 <= value <= 300.0:
                    shifts.append(value)
        # Preserve multiplicity: chemically equivalent carbons legitimately
        # share an observed shift, and each reported peak is one observation.
        shifts = sorted(round(s, 3) for s in shifts)
        if not shifts:
            return {
                "status": "unsupported_modality",
                "required_nucleus": "13C",
                "reason": "csp5_forward_requires_observed_13c",
                "model_called": False,
                "calibrated_probability": False,
                "quantile_enabled": True,
                "candidates": [],
            }

        prepared: list[tuple[Mapping[str, Any], str, Chem.Mol]] = []
        failed: list[Mapping[str, Any]] = []
        for candidate in candidates:
            smiles = str(candidate.get("smiles") or "")
            if not smiles:
                failed.append(candidate)
                continue
            try:
                mol_h = _embed_smiles(smiles)
            except Exception:
                failed.append(candidate)
                continue
            prepared.append((candidate, smiles, mol_h))

        rows_by_candidate: list[list[Mapping[str, Any]]] = []
        batch_failed_indices: set[int] = set()
        batch_failure_detail: str | None = None
        if prepared:
            mols = [item[2] for item in prepared]
            smiles_list = [
                Chem.MolToSmiles(Chem.RemoveHs(item[2])) for item in prepared
            ]
            try:
                result = predict_mols(
                    mols,
                    smiles=smiles_list,
                    nucleus="13C",
                    model_name=self.model_name,
                    device=self.device,
                    batch_size=self.batch_size,
                )
                if result.failures:
                    batch_failure_detail = str(result.failures[0])
                    unmappable: list[str] = []
                    for failure in result.failures:
                        parts = str(failure).split("\t", 1)
                        failed_smiles = (
                            parts[-1].strip() if len(parts) == 2 else ""
                        )
                        matched = [
                            index
                            for index, smiles in enumerate(smiles_list)
                            if smiles == failed_smiles
                        ]
                        if matched:
                            batch_failed_indices.update(matched)
                        else:
                            unmappable.append(str(failure))
                    if unmappable:
                        batch_failed_indices = set(range(len(prepared)))
                        batch_failure_detail = unmappable[0]
                rows_by_id: dict[int, list[Mapping[str, Any]]] = {}
                for _, row in result.predictions.iterrows():
                    rows_by_id.setdefault(int(row["molecule_id"]), []).append(
                        row.to_dict()
                    )
                for index in range(len(prepared)):
                    if index in batch_failed_indices:
                        rows_by_candidate.append([])
                        continue
                    rows = rows_by_id.get(index, [])
                    rows.sort(key=lambda r: float(r["atom_index"]))
                    rows_by_candidate.append(rows)
            except Exception:
                batch_failed_indices = set(range(len(prepared)))
                batch_failure_detail = "predict_mols raised"
                rows_by_candidate = [[] for _ in prepared]

        evidence = []
        failed_evidence: list[Mapping[str, Any]] = []
        for index, ((candidate, _, molecule), pred_rows) in enumerate(
            zip(prepared, rows_by_candidate)
        ):
            if index in batch_failed_indices:
                failed_evidence.append(
                    _prediction_failed_evidence(
                        candidate,
                        observed_count=len(shifts),
                        reason="batch_prediction_failed",
                        model_name=self.model_name,
                        weights_sha256=self.weights_sha256,
                        detail=batch_failure_detail,
                    )
                )
                continue
            prediction_problem = _target_prediction_problem(molecule, pred_rows)
            if prediction_problem is not None:
                reason, detail = prediction_problem
                failed_evidence.append(
                    _prediction_failed_evidence(
                        candidate,
                        observed_count=len(shifts),
                        reason=reason,
                        model_name=self.model_name,
                        weights_sha256=self.weights_sha256,
                        detail=detail,
                    )
                )
                continue
            predicted_shifts = [float(row["shift_ppm"]) for row in pred_rows]
            costs = [[abs(o - p) for p in predicted_shifts] for o in shifts]
            if costs and costs[0]:
                assign_rows, assign_cols = linear_sum_assignment(costs)
                errors = [
                    costs[r][c] for r, c in zip(assign_rows, assign_cols)
                ]
            else:
                errors = []
            mae = sum(errors) / max(len(errors), 1)
            rmse = math.sqrt(
                sum(e * e for e in errors) / max(len(errors), 1)
            )
            observed_coverage = len(errors) / max(len(shifts), 1)
            predicted_coverage = len(errors) / max(len(predicted_shifts), 1)
            evidence.append(
                {
                    "candidate_id": str(
                        candidate.get("candidate_id") or candidate.get("smiles")
                    ),
                    "relative_rank": 0,
                    "assignment_mode": "unassigned_hungarian_atom_level",
                    "diagnostic_only": True,
                    "matched_count": len(errors),
                    "observed_count": len(shifts),
                    "predicted_atom_count": len(predicted_shifts),
                    "unmatched_observed_count": len(shifts) - len(errors),
                    "unmatched_predicted_atom_count": len(predicted_shifts)
                    - len(errors),
                    "observed_coverage": round(observed_coverage, 4),
                    "predicted_atom_coverage": round(predicted_coverage, 4),
                    "bidirectional_coverage": round(
                        min(observed_coverage, predicted_coverage), 4
                    ),
                    "assignment_complete": (
                        len(shifts) - len(errors) == 0
                        and len(predicted_shifts) - len(errors) == 0
                    ),
                    "mae_ppm": round(mae, 3),
                    "rmse_ppm": round(rmse, 3),
                    "max_abs_error_ppm": round(max(errors), 3) if errors else None,
                    "calibrated_probability": False,
                    "quantile_enabled": True,
                    "prediction": {
                        "provider": PROVIDER_ID,
                        "model_name": self.model_name,
                        "model_sha256": self.weights_sha256,
                        "atom_predictions": [
                            {
                                "atom_index": float(int(row["atom_index"]) + 1),
                                "shift_ppm": round(float(row["shift_ppm"]), 3),
                                "q10_ppm": round(
                                    float(row["shift_q10_ppm"]), 3
                                )
                                if "shift_q10_ppm" in row
                                else None,
                                "q90_ppm": round(
                                    float(row["shift_q90_ppm"]), 3
                                )
                                if "shift_q90_ppm" in row
                                else None,
                                "shift_std_ppm": round(
                                    float(row["shift_std_ppm"]), 3
                                )
                                if "shift_std_ppm" in row
                                else None,
                            }
                            for row in pred_rows
                        ],
                    },
                }
            )

        for candidate in failed:
            failed_evidence.append(
                _prediction_failed_evidence(
                    candidate,
                    observed_count=len(shifts),
                    reason="smiles_embedding_failed",
                    model_name=self.model_name,
                    weights_sha256=self.weights_sha256,
                )
            )

        ordered = sorted(
            [*evidence, *failed_evidence],
            key=lambda item: (
                bool(item.get("prediction_failed")),
                -float(item["bidirectional_coverage"]),
                int(item["unmatched_observed_count"])
                + int(item["unmatched_predicted_atom_count"]),
                _mae_sort_key(item),
                str(item["candidate_id"]),
            ),
        )
        for rank, item in enumerate(ordered, start=1):
            item["relative_rank"] = rank
        failed_candidate_count = sum(
            1 for item in ordered if item.get("prediction_failed") is True
        )
        if ordered and failed_candidate_count == len(ordered):
            status = "prediction_failed"
        elif failed_candidate_count:
            status = "partial_failure"
        else:
            status = "ok"
        return {
            "status": status,
            "nucleus": "13C",
            "evidence_kind": "relative_13c_forward_evidence",
            "diagnostic_only": True,
            "candidate_count": len(ordered),
            "failed_candidate_count": failed_candidate_count,
            "rank_basis": (
                "higher_bidirectional_coverage_then_fewer_unmatched_then_"
                "lower_hungarian_mae_rmse"
            ),
            "observed_shifts_ppm": shifts,
            "calibrated_probability": False,
            "quantile_enabled": True,
            "model": {
                "provider": PROVIDER_ID,
                "model_name": self.model_name,
                "sha256": self.weights_sha256,
            },
            "runtime": {"device": self.device},
            "candidates": ordered,
        }


__all__ = ["Csp5ForwardScorer", "PROVIDER_ID"]
