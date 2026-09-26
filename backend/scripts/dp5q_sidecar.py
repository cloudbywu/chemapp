#!/usr/bin/env python
"""Persistent JSON-lines sidecar for the pinned DP5q 13C mean model.

This process intentionally exposes only SMILES and caller-defined candidate
IDs.  It never accepts serialized Python/RDKit objects, experimental spectra,
paths, model choices, or conformer parameters from a request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any, Mapping


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

PROTOCOL_VERSION = 1
REPOSITORY_COMMIT = "b79968cf63cb282e8871d5595ea6cef5b4dc0d49"
MEAN_MODEL_SHA256 = "2d453b9c340a45b7e3c8d789a0c6071167e677ccbfe972eec5a5e9c26033d095"
PREPROCESSOR_SHA256 = "6d143a468595797a05434a32da76cdcf57cb8b0cc929bfe27181acc297fde1b0"
MODEL_RELATIVE_PATH = Path(
    "dp5/neural_net/NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5"
)
PREPROCESSOR_RELATIVE_PATH = Path("dp5/neural_net/mean_model_preprocessor.p")

ALLOWED_ATOMIC_NUMBERS = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35})
MAX_CANDIDATES = 20
MAX_CANDIDATE_ID_LENGTH = 256
MAX_SMILES_LENGTH = 1024
MAX_HEAVY_ATOMS = 80
MAX_TOTAL_ATOMS = 200
MAX_ROTATABLE_BONDS = 20
MAX_REQUEST_BYTES = 1024 * 1024
CONFORMER_COUNT = 20
ETKDG_RANDOM_SEED = 0xC0FFEE
RMS_PRUNE_ANGSTROM = 0.5
ENERGY_WINDOW_KJ_MOL = 10.0
TEMPERATURE_K = 298.15
GAS_CONSTANT_KJ_MOL_K = 0.00831446261815324
MMFF_MAX_ITERATIONS = 1000
CONFORMER_PROTOCOL_VERSION = "chemapp.dp5q-conformer.v2"
CONFORMER_PREFLIGHT_PROTOCOL_VERSION = "chemapp.dp5q-conformer-preflight.v1"
_PREFLIGHT_REJECTION_CODES = frozenset(
    {
        "invalid_smiles",
        "multiple_fragments",
        "molecule_too_large",
        "molecule_too_flexible",
        "unsupported_element",
        "no_carbon",
        "mmff_parameters_unavailable",
        "conformer_generation_failed",
        "mmff_optimisation_failed",
        "conformer_filter_failed",
    }
)
_CONFORMER_ATTEMPTS: tuple[dict[str, Any], ...] = (
    {
        "attempt_id": "etkdgv3_standard",
        "use_random_coords": False,
        "ignore_smoothing_failures": False,
        "use_basic_knowledge": True,
        "use_experimental_torsions": True,
        "enforce_chirality": True,
        "eligibility": "all_supported_candidates",
    },
    {
        "attempt_id": "etkdgv3_random_coordinates",
        "use_random_coords": True,
        "ignore_smoothing_failures": False,
        "use_basic_knowledge": True,
        "use_experimental_torsions": True,
        "enforce_chirality": True,
        "eligibility": "all_supported_candidates",
    },
    {
        "attempt_id": "etkdgv3_relaxed_topology",
        "use_random_coords": False,
        "ignore_smoothing_failures": True,
        "use_basic_knowledge": False,
        "use_experimental_torsions": False,
        "enforce_chirality": False,
        "eligibility": "no_defined_atom_or_bond_stereochemistry",
    },
)
_LABEL_PATTERN = re.compile(r"^C(\d+)$")


class SidecarRequestError(ValueError):
    """A safe request error that can be returned without a traceback."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_commit(repository: Path) -> str:
    git = shutil.which("git")
    if git and (repository / ".git").exists():
        try:
            result = subprocess.run(
                [git, "-C", str(repository), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            )
            return result.stdout.strip().lower()
        except (OSError, subprocess.SubprocessError):
            pass
    marker = repository / ".chemapp-dp5q-commit"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip().lower()
    raise RuntimeError(
        "Cannot verify repository commit (.git or .chemapp-dp5q-commit required)."
    )


def _verify_repository(repository: Path) -> None:
    if not repository.is_dir():
        raise RuntimeError(f"DP5q repository does not exist: {repository}")
    commit = _repository_commit(repository)
    if commit != REPOSITORY_COMMIT:
        raise RuntimeError(
            f"DP5q commit mismatch: expected {REPOSITORY_COMMIT}, got {commit}."
        )
    assets = (
        (repository / MODEL_RELATIVE_PATH, MEAN_MODEL_SHA256, "mean model"),
        (
            repository / PREPROCESSOR_RELATIVE_PATH,
            PREPROCESSOR_SHA256,
            "preprocessor",
        ),
    )
    for path, expected, label in assets:
        if not path.is_file():
            raise RuntimeError(f"DP5q {label} is missing: {path}")
        actual = _sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"DP5q {label} SHA-256 mismatch: expected {expected}, got {actual}."
            )


def _handshake() -> dict[str, Any]:
    from rdkit import rdBase

    return {
        "type": "handshake",
        "protocol_version": PROTOCOL_VERSION,
        "status": "ready",
        "repository_commit": REPOSITORY_COMMIT,
        "sidecar": {
            "code_sha256": _sha256(Path(__file__).resolve()),
        },
        "assets": {
            "mean_model_sha256": MEAN_MODEL_SHA256,
            "preprocessor_sha256": PREPROCESSOR_SHA256,
        },
        "model": {
            "name": "DP5q-CASCADE-mean",
            "nucleus": "13C",
            "output": "boltzmann_weighted_mean_shift_ppm",
        },
        "capabilities": {
            "accepted_atomic_numbers": sorted(ALLOWED_ATOMIC_NUMBERS),
            "quantile_enabled": False,
            "calibrated_probability": False,
            "operations": [
                "conformer_preflight",
                "predict_13c_mean",
            ],
        },
        "runtime": {
            "rdkit_version": rdBase.rdkitVersion,
        },
        "conformer_generation": _conformer_policy(),
        "conformer_preflight": {
            "protocol_version": CONFORMER_PREFLIGHT_PROTOCOL_VERSION,
            "candidate_failures_are_results": True,
            "operational_failures_abort_request": True,
            "uses_same_prepare_candidate_path_as_prediction": True,
        },
    }


def _conformer_policy() -> dict[str, Any]:
    """Return the complete deterministic conformer and weighting contract."""

    return {
        "protocol_version": CONFORMER_PROTOCOL_VERSION,
        "requested_conformers": CONFORMER_COUNT,
        "random_seed": ETKDG_RANDOM_SEED,
        "num_threads": 1,
        "rms_prune_angstrom": RMS_PRUNE_ANGSTROM,
        "attempts": [dict(attempt) for attempt in _CONFORMER_ATTEMPTS],
        "optimiser": "MMFF94s",
        "mmff_max_iterations": MMFF_MAX_ITERATIONS,
        "energy_window_kj_mol": ENERGY_WINDOW_KJ_MOL,
        "temperature_k": TEMPERATURE_K,
        "gas_constant_kj_mol_k": GAS_CONSTANT_KJ_MOL_K,
        "population_weighting": "boltzmann",
        "terminal_failure": "fail_closed_no_uff",
    }


def _emit(value: Mapping[str, Any]) -> None:
    rendered = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    )
    sys.stdout.write(rendered + "\n")
    sys.stdout.flush()


def _strict_object(
    value: Any,
    keys: set[str],
    *,
    context: str,
) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise SidecarRequestError("invalid_schema", f"{context} must be a JSON object.")
    if set(value) != keys:
        raise SidecarRequestError(
            "invalid_schema",
            f"{context} must contain exactly: {', '.join(sorted(keys))}.",
        )
    return value


def _parse_request(raw: bytes) -> dict[str, Any]:
    if len(raw) > MAX_REQUEST_BYTES:
        raise SidecarRequestError("request_too_large", "Request is too large.")
    try:
        text = raw.decode("utf-8")

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite number {value}")

        value = json.loads(text, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise SidecarRequestError(
            "invalid_json", "Request must be finite UTF-8 JSON."
        ) from exc
    request = _strict_object(
        value,
        {"protocol_version", "op", "request_id", "candidates"},
        context="request",
    )
    if request["protocol_version"] != PROTOCOL_VERSION:
        raise SidecarRequestError("protocol_mismatch", "Unsupported protocol_version.")
    request_id = request["request_id"]
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise SidecarRequestError(
            "invalid_request_id", "request_id must be a non-empty short string."
        )
    if request["op"] not in {
        "conformer_preflight",
        "predict_13c_mean",
        "shutdown",
    }:
        raise SidecarRequestError("invalid_operation", "Unsupported operation.")
    candidates = request["candidates"]
    if not isinstance(candidates, list):
        raise SidecarRequestError("invalid_schema", "candidates must be a JSON list.")
    if request["op"] == "shutdown":
        if candidates:
            raise SidecarRequestError(
                "invalid_schema", "shutdown candidates must be empty."
            )
        return dict(request)
    if not candidates or len(candidates) > MAX_CANDIDATES:
        raise SidecarRequestError(
            "invalid_candidate_count",
            f"Candidate count must be between 1 and {MAX_CANDIDATES}.",
        )
    clean_candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for position, raw_candidate in enumerate(candidates):
        candidate = _strict_object(
            raw_candidate,
            {"candidate_id", "smiles"},
            context=f"candidate {position}",
        )
        candidate_id = candidate["candidate_id"]
        smiles = candidate["smiles"]
        if (
            not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or len(candidate_id) > MAX_CANDIDATE_ID_LENGTH
            or candidate_id in seen_ids
        ):
            raise SidecarRequestError(
                "invalid_candidate_id",
                f"candidate {position} has an invalid or duplicate candidate_id.",
            )
        if (
            not isinstance(smiles, str)
            or not smiles.strip()
            or len(smiles) > MAX_SMILES_LENGTH
        ):
            raise SidecarRequestError(
                "invalid_smiles",
                f"candidate {candidate_id} has an invalid SMILES string.",
            )
        seen_ids.add(candidate_id)
        clean_candidates.append(
            {"candidate_id": candidate_id, "smiles": smiles.strip()}
        )
    return {
        "protocol_version": PROTOCOL_VERSION,
        "op": request["op"],
        "request_id": request_id,
        "candidates": clean_candidates,
    }


class DP5qMeanRuntime:
    """Load the trusted mean model once and serve deterministic predictions."""

    def __init__(self, repository: Path):
        sys.path.insert(0, str(repository))
        try:
            import numpy as np
            from rdkit import Chem
            from rdkit.Chem import AllChem, rdMolDescriptors
            from dp5.neural_net.CNN_model import (
                load_NMR_prediction_model,
                mols_to_df,
                predict_shifts,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not import the pinned DP5q mean-model runtime."
            ) from exc

        self.np = np
        self.Chem = Chem
        self.AllChem = AllChem
        self.rdMolDescriptors = rdMolDescriptors
        self.mols_to_df = mols_to_df
        self.predict_shifts = predict_shifts
        self.model = load_NMR_prediction_model(str(repository / MODEL_RELATIVE_PATH))

    def _has_defined_stereochemistry(self, molecule: Any) -> bool:
        atom_stereo = any(
            atom.GetChiralTag() != self.Chem.ChiralType.CHI_UNSPECIFIED
            for atom in molecule.GetAtoms()
        )
        bond_stereo = any(
            bond.GetStereo() != self.Chem.BondStereo.STEREONONE
            for bond in molecule.GetBonds()
        )
        return atom_stereo or bond_stereo

    def _run_conformer_attempt(
        self,
        molecule: Any,
        attempt: Mapping[str, Any],
    ) -> tuple[Any, list[int], list[tuple[int, float]], int]:
        """Embed and optimise one immutable-policy attempt on a fresh molecule."""

        attempt_molecule = self.Chem.Mol(molecule)
        attempt_molecule.RemoveAllConformers()
        parameters = self.AllChem.ETKDGv3()
        parameters.randomSeed = ETKDG_RANDOM_SEED
        parameters.pruneRmsThresh = RMS_PRUNE_ANGSTROM
        parameters.numThreads = 1
        parameters.clearConfs = True
        parameters.useRandomCoords = bool(attempt["use_random_coords"])
        parameters.ignoreSmoothingFailures = bool(attempt["ignore_smoothing_failures"])
        parameters.useBasicKnowledge = bool(attempt["use_basic_knowledge"])
        parameters.useExpTorsionAnglePrefs = bool(attempt["use_experimental_torsions"])
        parameters.enforceChirality = bool(attempt["enforce_chirality"])
        conformer_ids = [
            int(value)
            for value in self.AllChem.EmbedMultipleConfs(
                attempt_molecule,
                numConfs=CONFORMER_COUNT,
                params=parameters,
            )
        ]
        if not conformer_ids:
            return attempt_molecule, [], [], 0

        optimisation = self.AllChem.MMFFOptimizeMoleculeConfs(
            attempt_molecule,
            numThreads=1,
            maxIters=MMFF_MAX_ITERATIONS,
            mmffVariant="MMFF94s",
        )
        if len(optimisation) != len(conformer_ids):
            raise RuntimeError("MMFF94s returned the wrong conformer count.")
        converged: list[tuple[int, float]] = []
        discarded_nonconverged = 0
        for conformer_id, (status, energy_kcal_mol) in zip(
            conformer_ids, optimisation, strict=True
        ):
            energy = float(energy_kcal_mol)
            if status == 0 and math.isfinite(energy):
                converged.append((conformer_id, energy * 4.184))
            else:
                discarded_nonconverged += 1
        return (
            attempt_molecule,
            conformer_ids,
            converged,
            discarded_nonconverged,
        )

    def _generate_conformers(
        self,
        molecule: Any,
        candidate_id: str,
    ) -> tuple[Any, list[tuple[int, float]], list[str]]:
        """Run ordered deterministic attempts and reject if every attempt fails."""

        defined_stereochemistry = self._has_defined_stereochemistry(molecule)
        attempt_audit: list[str] = []
        embedded_any = False
        for attempt_index, attempt in enumerate(_CONFORMER_ATTEMPTS):
            attempt_id = str(attempt["attempt_id"])
            if (
                attempt["eligibility"] == "no_defined_atom_or_bond_stereochemistry"
                and defined_stereochemistry
            ):
                attempt_audit.append(f"{attempt_id}:ineligible_defined_stereo")
                continue
            (
                attempt_molecule,
                conformer_ids,
                converged,
                discarded_nonconverged,
            ) = self._run_conformer_attempt(molecule, attempt)
            if not conformer_ids:
                attempt_audit.append(f"{attempt_id}:no_conformers")
                continue
            embedded_any = True
            if not converged:
                attempt_audit.append(f"{attempt_id}:no_converged_mmff94s")
                continue

            warnings: list[str] = []
            if attempt_index:
                warnings.extend(
                    (
                        f"conformer_fallback_used:{attempt_id}",
                        "conformer_attempt_audit:" + ",".join(attempt_audit),
                    )
                )
            if discarded_nonconverged:
                warnings.append(
                    f"discarded_nonconverged_conformers:{discarded_nonconverged}"
                )
            return attempt_molecule, converged, warnings

        audit_suffix = ",".join(attempt_audit)
        if embedded_any:
            raise SidecarRequestError(
                "mmff_optimisation_failed",
                f"candidate {candidate_id} has no converged MMFF94s conformer "
                f"after fixed attempts ({audit_suffix}).",
            )
        raise SidecarRequestError(
            "conformer_generation_failed",
            f"candidate {candidate_id} produced no conformers after fixed "
            f"ETKDGv3 attempts ({audit_suffix}).",
        )

    def _prepare_candidate(
        self,
        candidate: Mapping[str, str],
    ) -> tuple[list[Any], list[float], str, list[str]]:
        candidate_id = candidate["candidate_id"]
        molecule = self.Chem.MolFromSmiles(candidate["smiles"])
        if molecule is None:
            raise SidecarRequestError(
                "invalid_smiles", f"candidate {candidate_id} has invalid SMILES."
            )
        if len(self.Chem.GetMolFrags(molecule)) != 1:
            raise SidecarRequestError(
                "multiple_fragments",
                f"candidate {candidate_id} contains multiple fragments.",
            )
        if molecule.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
            raise SidecarRequestError(
                "molecule_too_large",
                f"candidate {candidate_id} exceeds the heavy-atom limit.",
            )
        if self.rdMolDescriptors.CalcNumRotatableBonds(molecule) > MAX_ROTATABLE_BONDS:
            raise SidecarRequestError(
                "molecule_too_flexible",
                f"candidate {candidate_id} exceeds the rotatable-bond limit.",
            )
        atomic_numbers = {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
        unsupported = sorted(atomic_numbers - ALLOWED_ATOMIC_NUMBERS)
        if unsupported:
            raise SidecarRequestError(
                "unsupported_element",
                f"candidate {candidate_id} contains unsupported atomic "
                f"numbers: {unsupported}.",
            )
        if not any(atom.GetAtomicNum() == 6 for atom in molecule.GetAtoms()):
            raise SidecarRequestError(
                "no_carbon", f"candidate {candidate_id} contains no carbon atoms."
            )
        canonical_smiles = self.Chem.MolToSmiles(
            molecule, canonical=True, isomericSmiles=True
        )
        molecule = self.Chem.AddHs(molecule)
        if molecule.GetNumAtoms() > MAX_TOTAL_ATOMS:
            raise SidecarRequestError(
                "molecule_too_large",
                f"candidate {candidate_id} exceeds the total-atom limit.",
            )
        if not self.AllChem.MMFFHasAllMoleculeParams(molecule):
            raise SidecarRequestError(
                "mmff_parameters_unavailable",
                f"candidate {candidate_id} is outside MMFF94s coverage.",
            )

        molecule, converged, warnings = self._generate_conformers(
            molecule,
            candidate_id,
        )
        minimum_energy = min(energy for _, energy in converged)
        retained = [
            (conformer_id, energy)
            for conformer_id, energy in converged
            if energy - minimum_energy <= ENERGY_WINDOW_KJ_MOL
        ][:CONFORMER_COUNT]
        if not retained:
            raise SidecarRequestError(
                "conformer_filter_failed",
                f"candidate {candidate_id} has no conformer in the energy window.",
            )
        relative_energies = [energy - minimum_energy for _, energy in retained]
        weights = [
            math.exp(-energy / (GAS_CONSTANT_KJ_MOL_K * TEMPERATURE_K))
            for energy in relative_energies
        ]
        weight_total = sum(weights)
        populations = [weight / weight_total for weight in weights]
        conformers = [
            self.Chem.Mol(molecule, confId=conformer_id) for conformer_id, _ in retained
        ]
        if len(conformers) == 1:
            warnings.append("single_conformer_only")
        return conformers, populations, canonical_smiles, warnings

    def predict(
        self,
        candidates: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        conformer_molecules: list[list[Any]] = []
        populations: list[list[float]] = []
        canonical_smiles: list[str] = []
        warnings: list[list[str]] = []
        for candidate in candidates:
            mols, weights, canonical, candidate_warnings = self._prepare_candidate(
                candidate
            )
            conformer_molecules.append(mols)
            populations.append(weights)
            canonical_smiles.append(canonical)
            warnings.append(candidate_warnings)

        frame, labels = self.mols_to_df(conformer_molecules, "C")
        shifts = self.predict_shifts(self.model, frame, batch_size=16)
        if len(shifts) != len(candidates) or len(labels) != len(candidates):
            raise RuntimeError("DP5q returned the wrong candidate count.")

        predictions: list[dict[str, Any]] = []
        for (
            candidate,
            array,
            atom_labels,
            weights,
            canonical,
            candidate_warnings,
        ) in zip(
            candidates,
            shifts,
            labels,
            populations,
            canonical_smiles,
            warnings,
            strict=True,
        ):
            matrix = self.np.asarray(array, dtype=float)
            if (
                matrix.ndim != 2
                or matrix.shape[0] != len(weights)
                or matrix.shape[1] != len(atom_labels)
                or not self.np.isfinite(matrix).all()
            ):
                raise RuntimeError("DP5q returned a malformed shift matrix.")
            weighted = self.np.average(
                matrix,
                axis=0,
                weights=self.np.asarray(weights, dtype=float),
            )
            atom_predictions: list[dict[str, Any]] = []
            seen_atoms: set[int] = set()
            for label, shift_value in zip(atom_labels, weighted, strict=True):
                match = _LABEL_PATTERN.fullmatch(str(label))
                shift = float(shift_value)
                if match is None:
                    raise RuntimeError("DP5q returned an invalid atom label.")
                atom_index = int(match.group(1)) - 1
                if (
                    atom_index < 0
                    or atom_index in seen_atoms
                    or not math.isfinite(shift)
                    or not -20.0 <= shift <= 300.0
                ):
                    raise RuntimeError(
                        "DP5q returned an invalid atom-level prediction."
                    )
                seen_atoms.add(atom_index)
                atom_predictions.append({"atom_index": atom_index, "shift_ppm": shift})
            predictions.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "canonical_smiles": canonical,
                    "conformer_count": len(weights),
                    "atom_predictions": sorted(
                        atom_predictions, key=lambda item: item["atom_index"]
                    ),
                    "warnings": candidate_warnings,
                }
            )
        return predictions

    def preflight(
        self,
        candidates: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Evaluate deterministic conformer applicability without model output."""

        results: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = candidate["candidate_id"]
            try:
                conformers, _, canonical_smiles, warnings = self._prepare_candidate(
                    candidate
                )
            except SidecarRequestError as exc:
                if exc.code not in _PREFLIGHT_REJECTION_CODES:
                    raise
                results.append(
                    {
                        "candidate_id": candidate_id,
                        "status": "rejected",
                        "reason_code": exc.code,
                        "canonical_smiles": None,
                        "conformer_count": 0,
                        "warnings": [],
                    }
                )
                continue
            results.append(
                {
                    "candidate_id": candidate_id,
                    "status": "passed",
                    "reason_code": None,
                    "canonical_smiles": canonical_smiles,
                    "conformer_count": len(conformers),
                    "warnings": warnings,
                }
            )
        return results


def _prediction_response(
    request_id: str,
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "prediction",
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": "ok",
        "predictions": predictions,
        "evidence_semantics": {
            "kind": "relative_13c_forward_evidence",
            "calibrated_probability": False,
            "quantile_enabled": False,
        },
    }


def _preflight_response(
    request_id: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "conformer_preflight",
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": "ok",
        "results": results,
        "model_outputs_created": False,
    }


def _error_response(
    request_id: str,
    code: str,
    message: str,
) -> dict[str, Any]:
    return {
        "type": "prediction",
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "status": "error",
        "error": {"code": code, "message": message[:1000]},
    }


def _serve(runtime: DP5qMeanRuntime) -> int:
    _emit(_handshake())
    while True:
        raw = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
        if not raw:
            return 0
        request_id = ""
        try:
            request = _parse_request(raw)
            request_id = request["request_id"]
            if request["op"] == "shutdown":
                return 0
            if request["op"] == "conformer_preflight":
                _emit(
                    _preflight_response(
                        request_id,
                        runtime.preflight(request["candidates"]),
                    )
                )
            else:
                predictions = runtime.predict(request["candidates"])
                _emit(_prediction_response(request_id, predictions))
        except SidecarRequestError as exc:
            _emit(_error_response(request_id, exc.code, exc.message))
        except Exception:
            traceback.print_exc(file=sys.stderr)
            _emit(
                _error_response(
                    request_id,
                    "inference_failed",
                    "The pinned DP5q mean model could not complete inference.",
                )
            )


def _self_test(runtime: DP5qMeanRuntime) -> int:
    candidate = {"candidate_id": "self-test", "smiles": "CCO"}
    predictions = runtime.predict([candidate])
    fallback_probe = {
        "candidate_id": "fallback-probe",
        "smiles": "C#Cc1cc2ccc1CCc1ccc(cc1C#C)CC2",
    }
    _, populations, canonical, warnings = runtime._prepare_candidate(fallback_probe)
    if not any(
        warning == "conformer_fallback_used:etkdgv3_relaxed_topology"
        for warning in warnings
    ):
        raise RuntimeError("The fixed fallback probe did not exercise the fallback.")
    stereo_probe = {
        "candidate_id": "defined-stereo-rejection-probe",
        "smiles": r"C1=CC=C2/C=C3\C=CC=C/C(=C/C(=C1)C2)O3",
    }
    stereo_molecule = runtime.Chem.MolFromSmiles(stereo_probe["smiles"])
    if stereo_molecule is None:
        raise RuntimeError("The defined-stereo rejection probe is invalid.")
    stereo_snapshot = runtime.Chem.MolToSmiles(
        stereo_molecule,
        canonical=True,
        isomericSmiles=True,
    )
    stereo_result = runtime.preflight([stereo_probe])[0]
    if (
        stereo_result["status"] != "rejected"
        or stereo_result["reason_code"] != "conformer_generation_failed"
        or runtime.Chem.MolToSmiles(
            stereo_molecule,
            canonical=True,
            isomericSmiles=True,
        )
        != stereo_snapshot
    ):
        raise RuntimeError(
            "The defined-stereo probe was embedded unsafely or its identity changed."
        )
    _emit(
        {
            "handshake": _handshake(),
            "prediction": predictions[0],
            "conformer_fallback_probe": {
                "candidate_id": fallback_probe["candidate_id"],
                "canonical_smiles": canonical,
                "conformer_count": len(populations),
                "population_sum": sum(populations),
                "warnings": warnings,
            },
            "defined_stereo_rejection_probe": {
                **stereo_result,
                "canonical_isomeric_smiles": stereo_snapshot,
            },
        }
    )
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Pinned ruslankotl/DP5 checkout.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Load the model, predict ethanol, print JSON, and exit.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = args.repo.expanduser().resolve()
    try:
        _verify_repository(repository)
        runtime = DP5qMeanRuntime(repository)
        return _self_test(runtime) if args.self_test else _serve(runtime)
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
