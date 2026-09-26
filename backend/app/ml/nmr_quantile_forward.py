"""Fail-closed adapter for the isolated official DP5q 13C quantile model.

This module is deliberately additive.  It does not alter the existing
``NMRForwardAdapter`` mean-shift path or the source hashes bound into prior
calibration artefacts.

DP5q's released quantile method assumes atom-assigned experimental shifts.
``score_assigned_prediction`` implements those upstream semantics.  The
adapter's unassigned Hungarian scorer is explicitly diagnostic shadow evidence
and must not be presented as an official DP5q probability.

Status: production (diagnostic shadow path).  The fail-closed sidecar process
machinery and protocol error types are inherited from
:mod:`app.ml._core.sidecar_base`; the pinned mean-adapter candidate
normalisation, conformer preflight request, and preflight validation are
reused unchanged from the frozen ``NMRForwardAdapter``.  The quantile install
verification, handshake contract, stricter secret-free environment allowlist,
and quantile response validation remain frozen here.
"""

from __future__ import annotations

import atexit
from dataclasses import dataclass, field
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Iterable, Mapping, Sequence
import uuid

from . import dp5q_runtime_pin as runtime_pin
from ._core.sidecar_base import SidecarAdapterBase
from .nmr_forward import (
    DP5Q_CONFORMER_PROTOCOL_VERSION,
    DP5Q_PREPROCESSOR_RELATIVE_PATH,
    NMRForwardAdapter,
    NMRForwardConfig,
    NMRForwardConfigurationError,
    NMRForwardInputError,
    NMRForwardProtocolError,
    NMRForwardUnavailableError,
    PROTOCOL_VERSION,
    _ALLOWED_ATOMIC_NUMBERS,
    _repository_commit,
    _sha256,
    _strict_keys,
    dp5q_conformer_policy,
    dp5q_sidecar_is_configured,
)


DP5Q_QUANTILE_ARCHIVE_SHA256 = (
    "4a0a760343a4f7cdfdb33e65a0ca684ae2b0bad7e61551f64120934c61fb8ec4"
)
DP5Q_QUANTILE_ARCHIVE_RELATIVE_PATH = Path(
    "dp5/neural_net/NMRdb_CASCADE_99quantiles.zip"
)
DP5Q_QUANTILE_LEVELS = tuple(index / 100 for index in range(1, 100))
DP5Q_QUANTILE_TENSORFLOW_VERSION = "2.14.0"
DP5Q_QUANTILE_KERAS_VERSION = "2.14.0"
DP5Q_MAX_QUANTILE_CROSSING_PPM = 5.0
DP5Q_QUANTILE_SCORE_SEMANTICS = (
    "dp5q_assigned_13c_quantile_equation_parity_v1"
)
DP5Q_UNASSIGNED_SHADOW_SEMANTICS = "unassigned_hungarian_dp5q_shadow_v1"
DP5Q_RAW_QUANTILE_TENSOR_DIGEST = (
    "sha256-sorted-population-tensor-records-float64-le-v1"
)


def _default_quantile_sidecar_script() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "dp5q_quantile_sidecar.py"
    )


def dp5q_distribution_contract() -> dict[str, Any]:
    """Return the scoring contract implemented by the pinned upstream code."""

    return {
        "family": "normal",
        "fit": "scipy_curve_fit_normal_cdf_to_99_quantiles",
        "initial_mu": "q50",
        "initial_sigma": "half_q67_minus_q34",
        "fit_bounds": "unbounded_upstream_parity",
        "quantile_crossing_policy": "retain_raw_values_report_do_not_sort",
        "conformer_aggregation": "boltzmann_weighted_atom_tail_consistency",
        "atom_score": "1-minus-weighted-absolute-one-minus-two-cdf",
        "molecule_score": "geometric_mean_atom_score_plus_1e-6",
    }


@dataclass(frozen=True)
class NMRQuantileForwardConfig(NMRForwardConfig):
    """Configuration for the separately pinned official quantile sidecar."""

    sidecar_script: Path = field(default_factory=_default_quantile_sidecar_script)
    expected_model_sha256: str = DP5Q_QUANTILE_ARCHIVE_SHA256


def _finite_number(value: Any, *, context: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise NMRForwardProtocolError(f"{context} must be a finite number.")
    return float(value)


def _normal_cdf(value: float, mu: float, sigma: float) -> float:
    """Numerically stable normal CDF without adding a runtime dependency."""

    return 0.5 * math.erfc(-(value - mu) / (sigma * math.sqrt(2.0)))


def _normalise_assigned_shifts(
    assigned_shifts: Mapping[int, float] | Sequence[Mapping[str, Any]],
) -> dict[int, float]:
    if isinstance(assigned_shifts, Mapping):
        raw_items = list(assigned_shifts.items())
    elif isinstance(assigned_shifts, Sequence) and not isinstance(
        assigned_shifts, (str, bytes)
    ):
        raw_items = []
        for position, item in enumerate(assigned_shifts):
            if not isinstance(item, Mapping) or set(item) != {
                "atom_index",
                "shift_ppm",
            }:
                raise NMRForwardInputError(
                    f"assigned_shifts[{position}] requires atom_index and shift_ppm."
                )
            raw_items.append((item["atom_index"], item["shift_ppm"]))
    else:
        raise NMRForwardInputError(
            "assigned_shifts must be an atom-index mapping or assignment sequence."
        )
    if not raw_items:
        raise NMRForwardInputError("At least one assigned 13C shift is required.")
    clean: dict[int, float] = {}
    for raw_index, raw_shift in raw_items:
        if type(raw_index) is not int or raw_index < 0 or raw_index in clean:
            raise NMRForwardInputError(
                "Assigned atom indices must be unique non-negative integers."
            )
        if (
            isinstance(raw_shift, bool)
            or not isinstance(raw_shift, (int, float))
            or not math.isfinite(float(raw_shift))
            or not -20.0 <= float(raw_shift) <= 300.0
        ):
            raise NMRForwardInputError(
                f"Assigned shift for atom {raw_index} is outside the 13C contract."
            )
        clean[raw_index] = float(raw_shift)
    return clean


def score_assigned_prediction(
    prediction: Mapping[str, Any],
    assigned_shifts: Mapping[int, float] | Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Score atom-assigned 13C shifts with the released DP5q equations.

    This reproduces ``QuantileDP5ProbabilityCalculator`` at the scoring stage:
    each conformer contributes ``abs(1 - 2 * NormalCDF(exp; mu, sigma))``;
    atom scores are one minus the population-weighted value; the structure
    score is the geometric mean of ``atom_score + 1e-6``.
    """

    clean_assignments = _normalise_assigned_shifts(assigned_shifts)
    populations_raw = prediction.get("conformer_populations")
    atom_predictions = prediction.get("atom_predictions")
    if not isinstance(populations_raw, list) or not populations_raw:
        raise NMRForwardProtocolError(
            "Quantile prediction is missing conformer populations."
        )
    populations = [
        _finite_number(value, context="conformer population")
        for value in populations_raw
    ]
    if any(value <= 0.0 or value > 1.0 for value in populations) or not math.isclose(
        sum(populations), 1.0, rel_tol=0.0, abs_tol=1e-9
    ):
        raise NMRForwardProtocolError(
            "Conformer populations must be positive and sum to one."
        )
    if not isinstance(atom_predictions, list):
        raise NMRForwardProtocolError("Quantile atom_predictions must be a list.")
    by_atom: dict[int, Mapping[str, Any]] = {}
    for atom in atom_predictions:
        if not isinstance(atom, Mapping) or type(atom.get("atom_index")) is not int:
            raise NMRForwardProtocolError("Quantile atom prediction is malformed.")
        atom_index = int(atom["atom_index"])
        if atom_index in by_atom:
            raise NMRForwardProtocolError(
                "Quantile atom prediction contains duplicate indices."
            )
        by_atom[atom_index] = atom
    unknown = sorted(set(clean_assignments) - set(by_atom))
    if unknown:
        raise NMRForwardInputError(
            f"Assignments reference atoms absent from the prediction: {unknown}"
        )

    atom_scores: list[dict[str, Any]] = []
    for atom_index, observed in sorted(clean_assignments.items()):
        atom = by_atom[atom_index]
        mus_raw = atom.get("conformer_mu_ppm")
        sigmas_raw = atom.get("conformer_sigma_ppm")
        if (
            not isinstance(mus_raw, list)
            or not isinstance(sigmas_raw, list)
            or len(mus_raw) != len(populations)
            or len(sigmas_raw) != len(populations)
        ):
            raise NMRForwardProtocolError(
                "Quantile fitted-distribution count does not match conformers."
            )
        mus = [
            _finite_number(value, context="fitted normal mu") for value in mus_raw
        ]
        sigmas = [
            _finite_number(value, context="fitted normal sigma")
            for value in sigmas_raw
        ]
        if any(sigma <= 0.0 for sigma in sigmas):
            raise NMRForwardProtocolError(
                "Quantile fitted normal sigma must be positive."
            )
        tail_consistency = [
            abs(1.0 - 2.0 * _normal_cdf(observed, mu, sigma))
            for mu, sigma in zip(mus, sigmas, strict=True)
        ]
        weighted_tail = sum(
            population * tail
            for population, tail in zip(
                populations, tail_consistency, strict=True
            )
        )
        raw_atom_score = 1.0 - weighted_tail
        if not -1e-12 <= raw_atom_score <= 1.0 + 1e-12:
            raise NMRForwardProtocolError(
                "DP5q atom score fell outside its mathematical bounds."
            )
        # Preserve the upstream scoring value exactly. The tolerance above is
        # only a protocol sanity check, not a clipping or recalibration step.
        atom_score = raw_atom_score
        atom_scores.append(
            {
                "atom_index": atom_index,
                "observed_shift_ppm": observed,
                "dp5q_atom_score": atom_score,
                "conformer_tail_consistency": tail_consistency,
            }
        )
    structure_score = math.exp(
        sum(math.log(item["dp5q_atom_score"] + 1e-6) for item in atom_scores)
        / len(atom_scores)
    )
    return {
        "score_semantics": DP5Q_QUANTILE_SCORE_SEMANTICS,
        "assignment_mode": "explicit_atom_assignment",
        "official_assignment_semantics": True,
        "official_equation_parity": True,
        "official_workflow_parity": False,
        "conformer_protocol": DP5Q_CONFORMER_PROTOCOL_VERSION,
        "calibrated_probability": False,
        "dp5q_method_score": structure_score,
        "assigned_atom_count": len(atom_scores),
        "atom_scores": atom_scores,
    }


class NMRQuantileForwardAdapter(SidecarAdapterBase, NMRForwardAdapter):
    """Own one persistent, serialised DP5q 99-quantile sidecar process.

    The process lifecycle comes from :class:`SidecarAdapterBase`; the frozen
    mean-adapter candidate/preflight helpers and the ``conformer_preflight``
    request path come from ``NMRForwardAdapter`` untouched.  Only the pinned
    quantile contract differences are implemented here.
    """

    config: NMRQuantileForwardConfig

    def __init__(self, config: NMRQuantileForwardConfig):
        super().__init__(config)

    def _sidecar_label(self) -> str:
        return "quantile "

    def _sidecar_environment(self) -> dict[str, str]:
        inherited_names = (
            "COMSPEC",
            "HOME",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "NUMBER_OF_PROCESSORS",
            "PATH",
            "PATHEXT",
            "PROCESSOR_ARCHITECTURE",
            "SYSTEMROOT",
            "TEMP",
            "TMP",
            "WINDIR",
        )
        environment = {
            name: os.environ[name]
            for name in inherited_names
            if os.environ.get(name)
        }
        environment.update(
            {
                "CUDA_VISIBLE_DEVICES": "-1",
                "MKL_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONUNBUFFERED": "1",
                "TF_CPP_MIN_LOG_LEVEL": "3",
                "TF_NUM_INTEROP_THREADS": "1",
                "TF_NUM_INTRAOP_THREADS": "1",
            }
        )
        return environment

    def _spawn_arguments(
        self, executable: str, sidecar_script: Path, repository: Path
    ) -> list[str]:
        return [
            executable,
            "-u",
            str(sidecar_script),
            "--repo",
            str(repository),
        ]

    def _spawn_cwd(self, sidecar_script: Path) -> str:
        return tempfile.gettempdir()

    def _stdout_thread_name(self) -> str:
        return "dp5q-quantile-stdout"

    def _stderr_thread_name(self) -> str:
        return "dp5q-quantile-stderr"

    def _verify_install(self) -> tuple[str, Path, Path]:
        executable, repository, sidecar_script = self._resolved_install_paths()
        mean_contract_script = sidecar_script.parent / "dp5q_sidecar.py"
        runtime_pin_script = Path(runtime_pin.__file__).resolve()
        if not mean_contract_script.is_file():
            raise NMRForwardConfigurationError(
                "DP5q quantile sidecar is missing its bound mean contract."
            )
        if not runtime_pin_script.is_file():
            raise NMRForwardConfigurationError(
                "DP5q quantile sidecar is missing its runtime pin module."
            )
        self._mean_contract_code_sha256 = _sha256(mean_contract_script)
        self._runtime_pin_code_sha256 = _sha256(runtime_pin_script)
        if not self.config.verify_local_install:
            return executable, repository, sidecar_script

        commit = _repository_commit(repository)
        if commit != self.config.expected_commit.lower():
            raise NMRForwardConfigurationError(
                "DP5q repository commit mismatch; "
                f"expected {self.config.expected_commit}, got {commit}."
            )
        assets = (
            (
                repository / DP5Q_QUANTILE_ARCHIVE_RELATIVE_PATH,
                self.config.expected_model_sha256.lower(),
                "99-quantile archive",
            ),
            (
                repository / DP5Q_PREPROCESSOR_RELATIVE_PATH,
                self.config.expected_preprocessor_sha256.lower(),
                "preprocessor",
            ),
        )
        for path, expected_hash, label in assets:
            if not path.is_file():
                raise NMRForwardConfigurationError(f"DP5q {label} is missing: {path}")
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                raise NMRForwardConfigurationError(
                    f"DP5q {label} SHA-256 mismatch; expected "
                    f"{expected_hash}, got {actual_hash}."
                )
        try:
            runtime_pin.verified_source_bytes(repository)
        except runtime_pin.DP5qRuntimePinError as exc:
            raise NMRForwardConfigurationError(
                "DP5q executable source bundle differs from its fixed commit."
            ) from exc
        return executable, repository, sidecar_script

    def _validate_handshake(self, value: Mapping[str, Any]) -> dict[str, Any]:
        _strict_keys(
            value,
            {
                "type",
                "protocol_version",
                "status",
                "repository_commit",
                "sidecar",
                "assets",
                "model",
                "capabilities",
                "runtime",
                "conformer_generation",
                "conformer_preflight",
                "distribution_fit",
            },
            context="quantile handshake",
        )
        if (
            value["type"] != "handshake"
            or value["protocol_version"] != PROTOCOL_VERSION
            or value["status"] != "ready"
            or value["repository_commit"] != self.config.expected_commit.lower()
        ):
            raise NMRForwardProtocolError(
                "DP5q quantile sidecar handshake was not ready or pinned."
            )

        sidecar = value["sidecar"]
        if not isinstance(sidecar, dict):
            raise NMRForwardProtocolError(
                "Quantile handshake sidecar must be an object."
            )
        _strict_keys(
            sidecar,
            {
                "code_sha256",
                "mean_contract_code_sha256",
                "runtime_pin_code_sha256",
            },
            context="quantile handshake sidecar",
        )
        code_sha256 = sidecar["code_sha256"]
        if (
            not isinstance(code_sha256, str)
            or len(code_sha256) != 64
            or any(character not in "0123456789abcdef" for character in code_sha256)
            or code_sha256 != self._sidecar_code_sha256
        ):
            raise NMRForwardProtocolError(
                "DP5q quantile sidecar implementation hash mismatch."
            )
        if (
            sidecar["mean_contract_code_sha256"]
            != self._mean_contract_code_sha256
            or sidecar["runtime_pin_code_sha256"]
            != self._runtime_pin_code_sha256
        ):
            raise NMRForwardProtocolError(
                "DP5q quantile sidecar dependency-code hash mismatch."
            )

        assets = value["assets"]
        if not isinstance(assets, dict):
            raise NMRForwardProtocolError(
                "Quantile handshake assets must be an object."
            )
        _strict_keys(
            assets,
            {
                "quantile_archive_sha256",
                "preprocessor_sha256",
                "upstream_source_bundle_sha256",
            },
            context="quantile handshake assets",
        )
        if (
            assets["quantile_archive_sha256"]
            != self.config.expected_model_sha256.lower()
            or assets["preprocessor_sha256"]
            != self.config.expected_preprocessor_sha256.lower()
            or assets["upstream_source_bundle_sha256"]
            != runtime_pin.DP5Q_UPSTREAM_SOURCE_BUNDLE_SHA256
        ):
            raise NMRForwardProtocolError(
                "DP5q quantile sidecar reported unexpected asset hashes."
            )

        model = value["model"]
        if not isinstance(model, dict):
            raise NMRForwardProtocolError(
                "Quantile handshake model must be an object."
            )
        _strict_keys(
            model,
            {
                "name",
                "nucleus",
                "output",
                "quantile_levels",
                "raw_quantile_tensor_digest",
            },
            context="quantile handshake model",
        )
        if (
            model["name"] != "DP5q-CASCADE-99quantiles"
            or model["nucleus"] != "13C"
            or model["output"]
            != "per_conformer_normal_fit_and_boltzmann_quantile_summary"
            or model["quantile_levels"] != list(DP5Q_QUANTILE_LEVELS)
            or model["raw_quantile_tensor_digest"]
            != DP5Q_RAW_QUANTILE_TENSOR_DIGEST
        ):
            raise NMRForwardProtocolError(
                "DP5q quantile sidecar model contract mismatch."
            )

        capabilities = value["capabilities"]
        if not isinstance(capabilities, dict):
            raise NMRForwardProtocolError(
                "Quantile handshake capabilities must be an object."
            )
        _strict_keys(
            capabilities,
            {
                "accepted_atomic_numbers",
                "quantile_enabled",
                "calibrated_probability",
                "operations",
            },
            context="quantile handshake capabilities",
        )
        if (
            capabilities["accepted_atomic_numbers"]
            != sorted(_ALLOWED_ATOMIC_NUMBERS)
            or capabilities["quantile_enabled"] is not True
            or capabilities["calibrated_probability"] is not False
            or capabilities["operations"]
            != ["conformer_preflight", "predict_13c_quantiles"]
        ):
            raise NMRForwardProtocolError(
                "DP5q quantile sidecar capabilities are not the pinned contract."
            )

        runtime = value["runtime"]
        if not isinstance(runtime, dict):
            raise NMRForwardProtocolError(
                "Quantile handshake runtime must be an object."
            )
        _strict_keys(
            runtime,
            {
                "archive_checkpoint_path_separator",
                "keras_version",
                "numpy_version",
                "pandas_version",
                "python_version",
                "rdkit_version",
                "scikit_learn_version",
                "scipy_version",
                "tensorflow_version",
                "tqdm_version",
            },
            context="quantile handshake runtime",
        )
        if (
            runtime["archive_checkpoint_path_separator"] != "posix"
            or runtime["keras_version"] != DP5Q_QUANTILE_KERAS_VERSION
            or runtime["numpy_version"] != runtime_pin.DP5Q_NUMPY_VERSION
            or runtime["pandas_version"] != runtime_pin.DP5Q_PANDAS_VERSION
            or runtime["python_version"] != runtime_pin.DP5Q_PYTHON_VERSION
            or runtime["rdkit_version"] != runtime_pin.DP5Q_RDKIT_VERSION
            or runtime["scikit_learn_version"]
            != runtime_pin.DP5Q_SCIKIT_LEARN_VERSION
            or runtime["scipy_version"] != runtime_pin.DP5Q_SCIPY_VERSION
            or runtime["tensorflow_version"] != DP5Q_QUANTILE_TENSORFLOW_VERSION
            or runtime["tqdm_version"] != runtime_pin.DP5Q_TQDM_VERSION
        ):
            raise NMRForwardProtocolError(
                "DP5q quantile sidecar reported an unpinned runtime."
            )
        if value["conformer_generation"] != dp5q_conformer_policy():
            raise NMRForwardProtocolError(
                "DP5q quantile sidecar changed the frozen conformer policy."
            )
        if value["conformer_preflight"] != {
            "protocol_version": "chemapp.dp5q-conformer-preflight.v1",
            "candidate_failures_are_results": True,
            "operational_failures_abort_request": True,
            "uses_same_prepare_candidate_path_as_prediction": True,
        }:
            raise NMRForwardProtocolError(
                "DP5q quantile sidecar changed the conformer-preflight policy."
            )
        if value["distribution_fit"] != dp5q_distribution_contract():
            raise NMRForwardProtocolError(
                "DP5q quantile sidecar changed the upstream scoring contract."
            )
        return json.loads(json.dumps(value))

    def _request_quantiles(
        self,
        candidates: list[dict[str, str]],
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "op": "predict_13c_quantiles",
            "request_id": request_id,
            "candidates": candidates,
        }
        response = self._transact(payload)
        try:
            if response.get("status") == "error":
                _strict_keys(
                    response,
                    {
                        "type",
                        "protocol_version",
                        "request_id",
                        "status",
                        "error",
                    },
                    context="quantile error response",
                )
                error = response["error"]
                if (
                    response["type"] != "prediction"
                    or response["protocol_version"] != PROTOCOL_VERSION
                    or response["request_id"] != request_id
                    or not isinstance(error, dict)
                    or set(error) != {"code", "message"}
                    or not all(
                        isinstance(error.get(key), str)
                        for key in ("code", "message")
                    )
                ):
                    raise NMRForwardProtocolError(
                        "DP5q quantile sidecar returned an invalid error."
                    )
                raise NMRForwardUnavailableError(
                    "DP5q quantile sidecar rejected the request "
                    f"({error['code']}): {error['message']}"
                )
            _strict_keys(
                response,
                {
                    "type",
                    "protocol_version",
                    "request_id",
                    "status",
                    "predictions",
                    "evidence_semantics",
                },
                context="quantile prediction response",
            )
            if (
                response["type"] != "prediction"
                or response["protocol_version"] != PROTOCOL_VERSION
                or response["request_id"] != request_id
                or response["status"] != "ok"
                or response["evidence_semantics"]
                != {
                    "kind": "dp5q_13c_quantile_components",
                    "calibrated_probability": False,
                    "quantile_enabled": True,
                    "assigned_observations_required_for_equation_parity_score": (
                        True
                    ),
                }
            ):
                raise NMRForwardProtocolError(
                    "DP5q quantile response identity or semantics mismatch."
                )
            return response
        except NMRForwardProtocolError:
            self._abort()
            raise

    @staticmethod
    def _validate_quantile_predictions(
        value: Any,
        candidates: list[dict[str, str]],
        expected_carbon_indices: Mapping[str, set[int]],
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list) or len(value) != len(candidates):
            raise NMRForwardProtocolError(
                "DP5q quantile sidecar returned the wrong prediction count."
            )
        try:
            from rdkit import Chem
        except ImportError as exc:  # pragma: no cover
            raise NMRForwardConfigurationError(
                "RDKit is required to validate quantile structure identity."
            ) from exc
        expected_canonical: dict[str, str] = {}
        for candidate in candidates:
            molecule = Chem.MolFromSmiles(candidate["smiles"])
            assert molecule is not None
            expected_canonical[candidate["candidate_id"]] = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
            )
        by_id: dict[str, dict[str, Any]] = {}
        for prediction in value:
            if not isinstance(prediction, dict):
                raise NMRForwardProtocolError(
                    "Each DP5q quantile prediction must be an object."
                )
            _strict_keys(
                prediction,
                {
                    "candidate_id",
                    "canonical_smiles",
                        "conformer_count",
                        "conformer_populations",
                        "raw_conformer_quantiles_shape",
                        "raw_conformer_quantiles_atom_indices",
                        "raw_conformer_quantiles_sha256",
                        "atom_predictions",
                    "warnings",
                },
                context="quantile candidate prediction",
            )
            candidate_id = prediction["candidate_id"]
            if (
                not isinstance(candidate_id, str)
                or candidate_id not in expected_carbon_indices
                or candidate_id in by_id
            ):
                raise NMRForwardProtocolError(
                    "DP5q quantile sidecar returned an unknown or duplicate ID."
                )
            canonical_smiles = prediction["canonical_smiles"]
            if (
                not isinstance(canonical_smiles, str)
                or canonical_smiles != expected_canonical[candidate_id]
            ):
                raise NMRForwardProtocolError(
                    "DP5q quantile canonical structure identity changed."
                )
            conformer_count = prediction["conformer_count"]
            if type(conformer_count) is not int or not 1 <= conformer_count <= 20:
                raise NMRForwardProtocolError(
                    "DP5q quantile conformer_count is outside the fixed contract."
                )
            raw_populations = prediction["conformer_populations"]
            if (
                not isinstance(raw_populations, list)
                or len(raw_populations) != conformer_count
            ):
                raise NMRForwardProtocolError(
                    "DP5q quantile conformer populations are malformed."
                )
            populations = [
                _finite_number(value, context="conformer population")
                for value in raw_populations
            ]
            if any(
                population <= 0.0 or population > 1.0
                for population in populations
            ) or not math.isclose(
                sum(populations), 1.0, rel_tol=0.0, abs_tol=1e-9
            ):
                raise NMRForwardProtocolError(
                    "DP5q conformer populations must be positive and sum to one."
                )
            raw_tensor_shape = prediction["raw_conformer_quantiles_shape"]
            raw_tensor_atom_indices = prediction[
                "raw_conformer_quantiles_atom_indices"
            ]
            raw_tensor_sha256 = prediction[
                "raw_conformer_quantiles_sha256"
            ]
            if (
                raw_tensor_shape
                != [
                    conformer_count,
                    len(expected_carbon_indices[candidate_id]),
                    len(DP5Q_QUANTILE_LEVELS),
                ]
                or not isinstance(raw_tensor_atom_indices, list)
                or len(raw_tensor_atom_indices)
                != len(expected_carbon_indices[candidate_id])
                or any(
                    type(atom_index) is not int
                    for atom_index in raw_tensor_atom_indices
                )
                or set(raw_tensor_atom_indices)
                != expected_carbon_indices[candidate_id]
                or not isinstance(raw_tensor_sha256, str)
                or len(raw_tensor_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in raw_tensor_sha256
                )
            ):
                raise NMRForwardProtocolError(
                    "DP5q raw conformer quantile digest metadata is malformed."
                )
            warnings = prediction["warnings"]
            if (
                not isinstance(warnings, list)
                or len(warnings) > 50
                or any(
                    not isinstance(warning, str) or len(warning) > 1000
                    for warning in warnings
                )
            ):
                raise NMRForwardProtocolError(
                    "DP5q quantile warnings are malformed."
                )
            raw_atoms = prediction["atom_predictions"]
            if not isinstance(raw_atoms, list):
                raise NMRForwardProtocolError(
                    "DP5q quantile atom_predictions must be a list."
                )
            clean_atoms: list[dict[str, Any]] = []
            seen_atoms: set[int] = set()
            for atom in raw_atoms:
                if not isinstance(atom, dict):
                    raise NMRForwardProtocolError(
                        "DP5q quantile atom prediction must be an object."
                    )
                _strict_keys(
                    atom,
                    {
                        "atom_index",
                        "quantiles_ppm",
                        "conformer_mu_ppm",
                        "conformer_sigma_ppm",
                        "conformer_quantile_crossing_counts",
                        "conformer_max_quantile_crossing_ppm",
                    },
                    context="quantile atom prediction",
                )
                atom_index = atom["atom_index"]
                if (
                    type(atom_index) is not int
                    or atom_index < 0
                    or atom_index in seen_atoms
                ):
                    raise NMRForwardProtocolError(
                        "DP5q quantile atom indices must be unique."
                    )
                quantiles_raw = atom["quantiles_ppm"]
                if (
                    not isinstance(quantiles_raw, list)
                    or len(quantiles_raw) != len(DP5Q_QUANTILE_LEVELS)
                ):
                    raise NMRForwardProtocolError(
                        "DP5q atom must contain exactly 99 quantiles."
                    )
                quantiles = [
                    _finite_number(value, context="predicted quantile")
                    for value in quantiles_raw
                ]
                crossing_depths = [
                    current - following
                    for current, following in zip(quantiles, quantiles[1:])
                    if following < current
                ]
                if any(
                    not -100.0 <= value <= 400.0 for value in quantiles
                ) or (
                    crossing_depths
                    and max(crossing_depths)
                    > DP5Q_MAX_QUANTILE_CROSSING_PPM
                ):
                    raise NMRForwardProtocolError(
                        "DP5q quantiles are out of range or cross excessively."
                    )
                mus_raw = atom["conformer_mu_ppm"]
                sigmas_raw = atom["conformer_sigma_ppm"]
                crossing_counts_raw = atom[
                    "conformer_quantile_crossing_counts"
                ]
                max_crossings_raw = atom[
                    "conformer_max_quantile_crossing_ppm"
                ]
                if (
                    not isinstance(mus_raw, list)
                    or not isinstance(sigmas_raw, list)
                    or not isinstance(crossing_counts_raw, list)
                    or not isinstance(max_crossings_raw, list)
                    or len(mus_raw) != conformer_count
                    or len(sigmas_raw) != conformer_count
                    or len(crossing_counts_raw) != conformer_count
                    or len(max_crossings_raw) != conformer_count
                ):
                    raise NMRForwardProtocolError(
                        "DP5q quantile diagnostics do not match conformers."
                    )
                mus = [
                    _finite_number(value, context="fitted normal mu")
                    for value in mus_raw
                ]
                sigmas = [
                    _finite_number(value, context="fitted normal sigma")
                    for value in sigmas_raw
                ]
                if any(
                    type(value) is not int or not 0 <= value <= 98
                    for value in crossing_counts_raw
                ):
                    raise NMRForwardProtocolError(
                        "DP5q quantile crossing counts are malformed."
                    )
                max_crossings = [
                    _finite_number(value, context="maximum quantile crossing")
                    for value in max_crossings_raw
                ]
                if any(not -20.0 <= mu <= 300.0 for mu in mus) or any(
                    not 0.0 < sigma <= 100.0 for sigma in sigmas
                ) or any(
                    not 0.0
                    <= crossing
                    <= DP5Q_MAX_QUANTILE_CROSSING_PPM
                    for crossing in max_crossings
                ) or any(
                    (count == 0) != math.isclose(crossing, 0.0, abs_tol=1e-12)
                    for count, crossing in zip(
                        crossing_counts_raw, max_crossings, strict=True
                    )
                ):
                    raise NMRForwardProtocolError(
                        "DP5q fitted parameters or crossing diagnostics are invalid."
                    )
                seen_atoms.add(atom_index)
                clean_atoms.append(
                    {
                        "atom_index": atom_index,
                        "quantiles_ppm": quantiles,
                        "conformer_mu_ppm": mus,
                        "conformer_sigma_ppm": sigmas,
                        "conformer_quantile_crossing_counts": list(
                            crossing_counts_raw
                        ),
                        "conformer_max_quantile_crossing_ppm": max_crossings,
                    }
                )
            if seen_atoms != expected_carbon_indices[candidate_id]:
                raise NMRForwardProtocolError(
                    f"DP5q quantile atom set mismatch for {candidate_id}."
                )
            by_id[candidate_id] = {
                "candidate_id": candidate_id,
                "canonical_smiles": canonical_smiles,
                "conformer_count": conformer_count,
                "conformer_populations": populations,
                "raw_conformer_quantiles_shape": list(raw_tensor_shape),
                "raw_conformer_quantiles_atom_indices": list(
                    raw_tensor_atom_indices
                ),
                "raw_conformer_quantiles_sha256": raw_tensor_sha256,
                "atom_predictions": sorted(
                    clean_atoms, key=lambda item: int(item["atom_index"])
                ),
                "warnings": list(warnings),
            }
        expected_ids = [candidate["candidate_id"] for candidate in candidates]
        if set(by_id) != set(expected_ids):
            raise NMRForwardProtocolError(
                "DP5q quantile sidecar omitted one or more candidates."
            )
        return [by_id[candidate_id] for candidate_id in expected_ids]

    def predict_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        formula: str | None = None,
    ) -> dict[str, Any]:
        """Return verified 13C quantile components for safe candidates."""

        clean_candidates, carbon_indices = self._normalise_candidates(
            candidates, formula=formula
        )
        preflight_response = self._request(
            clean_candidates,
            operation="conformer_preflight",
        )
        try:
            preflight_results = self._validate_preflight_results(
                preflight_response["results"],
                clean_candidates,
            )
        except Exception:
            with self._lock:
                self._abort()
            raise
        rejected = [
            result
            for result in preflight_results
            if result["status"] == "rejected"
        ]
        if rejected:
            reasons = ", ".join(
                f"{result['candidate_id']}:{result['reason_code']}"
                for result in rejected
            )
            raise NMRForwardInputError(
                "DP5q conformer preflight rejected candidate(s): " + reasons
            )
        response = self._request_quantiles(clean_candidates)
        try:
            predictions = self._validate_quantile_predictions(
                response["predictions"], clean_candidates, carbon_indices
            )
        except Exception:
            with self._lock:
                self._abort()
            raise
        assert self._handshake is not None
        return {
            "status": "ok",
            "nucleus": "13C",
            "output": "per_conformer_normal_fit_and_boltzmann_quantile_summary",
            "evidence_kind": "dp5q_13c_quantile_components",
            "calibrated_probability": False,
            "quantile_enabled": True,
            "model": {
                "name": self._handshake["model"]["name"],
                "repository_commit": self._handshake["repository_commit"],
                **self._handshake["assets"],
            },
            "runtime": {
                "sidecar_code_sha256": self._handshake["sidecar"]["code_sha256"],
                "mean_contract_code_sha256": self._handshake["sidecar"][
                    "mean_contract_code_sha256"
                ],
                "runtime_pin_code_sha256": self._handshake["sidecar"][
                    "runtime_pin_code_sha256"
                ],
                "protocol_version": PROTOCOL_VERSION,
                "archive_checkpoint_path_separator": self._handshake["runtime"][
                    "archive_checkpoint_path_separator"
                ],
                "keras_version": self._handshake["runtime"]["keras_version"],
                "numpy_version": self._handshake["runtime"]["numpy_version"],
                "pandas_version": self._handshake["runtime"]["pandas_version"],
                "python_version": self._handshake["runtime"]["python_version"],
                "rdkit_version": self._handshake["runtime"]["rdkit_version"],
                "scikit_learn_version": self._handshake["runtime"][
                    "scikit_learn_version"
                ],
                "scipy_version": self._handshake["runtime"]["scipy_version"],
                "tensorflow_version": self._handshake["runtime"][
                    "tensorflow_version"
                ],
                "tqdm_version": self._handshake["runtime"]["tqdm_version"],
                "conformer_generation": self._handshake["conformer_generation"],
                "conformer_preflight": self._handshake[
                    "conformer_preflight"
                ],
                "distribution_fit": self._handshake["distribution_fit"],
            },
            "predictions": predictions,
        }

    def score_candidates(
        self,
        observed_13c: Iterable[float | Mapping[str, Any]] | None,
        candidates: Sequence[Mapping[str, Any]],
        *,
        formula: str | None = None,
    ) -> dict[str, Any]:
        """Return explicitly non-official unassigned Hungarian shadow scores."""

        shifts = (
            self._normalise_observed_shifts(observed_13c)
            if observed_13c is not None
            else []
        )
        if not shifts:
            return {
                "status": "unsupported_modality",
                "required_nucleus": "13C",
                "reason": "dp5q_quantile_requires_observed_13c_resonances",
                "model_called": False,
                "used_for_ranking": False,
                "calibrated_probability": False,
                "quantile_enabled": True,
                "candidates": [],
            }

        prediction_result = self.predict_candidates(candidates, formula=formula)
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:  # pragma: no cover
            raise NMRForwardConfigurationError(
                "SciPy is required for unassigned quantile shadow matching."
            ) from exc

        evidence: list[dict[str, Any]] = []
        for prediction in prediction_result["predictions"]:
            atoms = prediction["atom_predictions"]
            medians = [float(atom["quantiles_ppm"][49]) for atom in atoms]
            costs = [
                [abs(observed - median) for median in medians]
                for observed in shifts
            ]
            rows, columns = linear_sum_assignment(costs)
            assignments = {
                int(atoms[int(column)]["atom_index"]): shifts[int(row)]
                for row, column in zip(rows, columns, strict=True)
            }
            official_equations = score_assigned_prediction(
                prediction, assignments
            )
            equation_details = {
                "official_equation_parity": True,
                "official_workflow_parity": False,
                "official_equation_semantics": official_equations[
                    "score_semantics"
                ],
                "conformer_protocol": DP5Q_CONFORMER_PROTOCOL_VERSION,
                "assignment_mode": "unassigned_hungarian_q50_atom_level",
                "official_assignment_semantics": False,
                "calibrated_probability": False,
                "dp5q_method_score": official_equations[
                    "dp5q_method_score"
                ],
                "assigned_atom_count": official_equations[
                    "assigned_atom_count"
                ],
                "atom_scores": official_equations["atom_scores"],
            }
            errors = [
                costs[int(row)][int(column)]
                for row, column in zip(rows, columns, strict=True)
            ]
            observed_coverage = len(errors) / len(shifts)
            predicted_coverage = len(errors) / len(atoms)
            unmatched_observed = len(shifts) - len(errors)
            unmatched_predicted = len(atoms) - len(errors)
            evidence.append(
                {
                    "candidate_id": prediction["candidate_id"],
                    "relative_rank": 0,
                    "score_semantics": DP5Q_UNASSIGNED_SHADOW_SEMANTICS,
                    "assignment_mode": "unassigned_hungarian_q50_atom_level",
                    "official_assignment_semantics": False,
                    "official_equation_parity": True,
                    "official_workflow_parity": False,
                    "conformer_protocol": DP5Q_CONFORMER_PROTOCOL_VERSION,
                    "diagnostic_only": True,
                    "used_for_ranking": False,
                    "matched_count": len(errors),
                    "observed_count": len(shifts),
                    "predicted_atom_count": len(atoms),
                    "unmatched_observed_count": unmatched_observed,
                    "unmatched_predicted_atom_count": unmatched_predicted,
                    "observed_coverage": observed_coverage,
                    "predicted_atom_coverage": predicted_coverage,
                    "bidirectional_coverage": min(
                        observed_coverage, predicted_coverage
                    ),
                    "assignment_complete": (
                        unmatched_observed == 0 and unmatched_predicted == 0
                    ),
                    "q50_assignment_mae_ppm": sum(errors) / len(errors),
                    "dp5q_shadow_score": official_equations["dp5q_method_score"],
                    "calibrated_probability": False,
                    "quantile_enabled": True,
                    "equation_details": equation_details,
                    "prediction": prediction,
                }
            )
        ordered = sorted(
            evidence,
            key=lambda item: (
                -float(item["bidirectional_coverage"]),
                int(item["unmatched_observed_count"])
                + int(item["unmatched_predicted_atom_count"]),
                -float(item["dp5q_shadow_score"]),
                float(item["q50_assignment_mae_ppm"]),
                str(item["candidate_id"]),
            ),
        )
        for rank, item in enumerate(ordered, start=1):
            item["relative_rank"] = rank
        return {
            "status": "ok",
            "nucleus": "13C",
            "evidence_kind": DP5Q_UNASSIGNED_SHADOW_SEMANTICS,
            "diagnostic_only": True,
            "used_for_ranking": False,
            "rank_basis": (
                "higher_bidirectional_coverage_then_fewer_unmatched_then_"
                "higher_dp5q_shadow_score_then_lower_q50_assignment_mae"
            ),
            "assignment_limitation": (
                "Official DP5q requires atom-assigned 13C shifts. Here q50 "
                "predictions are matched to unassigned resonances with Hungarian "
                "assignment and symmetry-equivalent signal collapse is not modelled."
            ),
            "observed_shifts_ppm": shifts,
            "official_assignment_semantics": False,
            "official_equation_parity": True,
            "official_workflow_parity": False,
            "conformer_protocol": DP5Q_CONFORMER_PROTOCOL_VERSION,
            "calibrated_probability": False,
            "quantile_enabled": True,
            "model": prediction_result["model"],
            "runtime": prediction_result["runtime"],
            "candidates": ordered,
        }


def dp5q_quantile_sidecar_is_configured() -> bool:
    """Return whether the shared isolated-runtime variables are present."""

    return dp5q_sidecar_is_configured()


_default_quantile_adapter: NMRQuantileForwardAdapter | None = None
_default_quantile_adapter_lock = threading.Lock()


def get_nmr_quantile_forward_adapter() -> NMRQuantileForwardAdapter:
    """Return the lazy process-wide quantile adapter configured by environment."""

    global _default_quantile_adapter
    with _default_quantile_adapter_lock:
        if _default_quantile_adapter is None:
            _default_quantile_adapter = NMRQuantileForwardAdapter(
                NMRQuantileForwardConfig.from_environment()
            )
        return _default_quantile_adapter


def _close_default_quantile_adapter() -> None:
    global _default_quantile_adapter
    with _default_quantile_adapter_lock:
        if _default_quantile_adapter is not None:
            _default_quantile_adapter.close()
            _default_quantile_adapter = None


atexit.register(_close_default_quantile_adapter)


__all__ = [
    "DP5Q_QUANTILE_ARCHIVE_RELATIVE_PATH",
    "DP5Q_QUANTILE_ARCHIVE_SHA256",
    "DP5Q_QUANTILE_LEVELS",
    "DP5Q_QUANTILE_SCORE_SEMANTICS",
    "DP5Q_UNASSIGNED_SHADOW_SEMANTICS",
    "NMRQuantileForwardAdapter",
    "NMRQuantileForwardConfig",
    "dp5q_distribution_contract",
    "dp5q_quantile_sidecar_is_configured",
    "get_nmr_quantile_forward_adapter",
    "score_assigned_prediction",
]
