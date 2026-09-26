#!/usr/bin/env python
"""Isolated JSON-lines sidecar for the pinned DP5q 13C quantile model.

The released DP5q quantile archive is loaded only after its repository commit
and SHA-256 digest have been verified.  Requests contain candidate IDs and
SMILES only; experimental spectra, paths, pickles, and model choices are not
accepted.

This is intentionally separate from ``dp5q_sidecar.py``.  Keeping the released
mean-model sidecar byte-for-byte unchanged preserves the immutable bindings of
the existing v3 calibration artefacts.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.abc
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import pickle
import posixpath
import sys
import tempfile
import traceback
from typing import Any
import warnings
import zipfile


os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

_SCRIPT_DIRECTORY = str(Path(__file__).resolve().parent)
if _SCRIPT_DIRECTORY not in sys.path:
    sys.path.insert(0, _SCRIPT_DIRECTORY)
_BACKEND_DIRECTORY = str(Path(__file__).resolve().parents[1])
if _BACKEND_DIRECTORY not in sys.path:
    sys.path.insert(0, _BACKEND_DIRECTORY)

import dp5q_sidecar as mean_contract  # noqa: E402
from app.ml import dp5q_runtime_pin as runtime_pin  # noqa: E402


PROTOCOL_VERSION = 1
REPOSITORY_COMMIT = mean_contract.REPOSITORY_COMMIT
PREPROCESSOR_SHA256 = mean_contract.PREPROCESSOR_SHA256
QUANTILE_ARCHIVE_SHA256 = (
    "4a0a760343a4f7cdfdb33e65a0ca684ae2b0bad7e61551f64120934c61fb8ec4"
)
QUANTILE_ARCHIVE_RELATIVE_PATH = Path(
    "dp5/neural_net/NMRdb_CASCADE_99quantiles.zip"
)
PREPROCESSOR_RELATIVE_PATH = mean_contract.PREPROCESSOR_RELATIVE_PATH
QUANTILE_LEVELS = tuple(index / 100 for index in range(1, 100))
TENSORFLOW_VERSION = "2.14.0"
KERAS_VERSION = "2.14.0"
MAX_ARCHIVE_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_COMPRESSION_RATIO = 200.0
MAX_QUANTILE_CROSSING_PPM = 5.0
RAW_QUANTILE_TENSOR_DIGEST = (
    "sha256-sorted-population-tensor-records-float64-le-v1"
)
_ARCHIVE_MEMBERS = frozenset({"array.npy", "model.keras"})
QUANTILE_ARCHIVE_BYTES = 10_190_486
PREPROCESSOR_BYTES = 10_987


def _read_exact_asset(
    repository: Path,
    relative_path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> bytes:
    """Return the same pinned bytes that were hashed, without a second open."""

    root = repository.expanduser().resolve(strict=True)
    path = root.joinpath(*relative_path.parts)
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"DP5q {label} is missing or unsafe.")
    try:
        path.resolve(strict=True).relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"DP5q {label} escapes the repository.") from exc
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(f"DP5q {label} length mismatch.")
    payload = path.read_bytes()
    if (
        len(payload) != expected_bytes
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise RuntimeError(f"DP5q {label} SHA-256 mismatch.")
    return payload


def _verify_repository(repository: Path) -> dict[str, Any]:
    """Load all executable inputs into hash-verified immutable byte buffers."""

    if not repository.is_dir():
        raise RuntimeError(f"DP5q repository does not exist: {repository}")
    commit = mean_contract._repository_commit(repository)
    if commit != REPOSITORY_COMMIT:
        raise RuntimeError(
            f"DP5q commit mismatch: expected {REPOSITORY_COMMIT}, got {commit}."
        )
    try:
        source_files = runtime_pin.verified_source_bytes(repository)
    except runtime_pin.DP5qRuntimePinError as exc:
        raise RuntimeError("DP5q executable source bundle is not pinned.") from exc
    return {
        "source_files": source_files,
        "quantile_archive": _read_exact_asset(
            repository,
            QUANTILE_ARCHIVE_RELATIVE_PATH,
            expected_bytes=QUANTILE_ARCHIVE_BYTES,
            expected_sha256=QUANTILE_ARCHIVE_SHA256,
            label="99-quantile archive",
        ),
        "preprocessor": _read_exact_asset(
            repository,
            PREPROCESSOR_RELATIVE_PATH,
            expected_bytes=PREPROCESSOR_BYTES,
            expected_sha256=PREPROCESSOR_SHA256,
            label="preprocessor",
        ),
    }


def _module_record(
    relative_path: str,
    payload: bytes,
) -> tuple[str, bytes, bool]:
    path = relative_path.removesuffix(".py")
    is_package = path.endswith("/__init__")
    if is_package:
        path = path.removesuffix("/__init__")
    return path.replace("/", "."), payload, is_package


class _VerifiedSourceLoader(importlib.abc.Loader):
    def __init__(
        self,
        module_name: str,
        relative_path: str,
        payload: bytes,
        *,
        is_package: bool,
    ):
        self.module_name = module_name
        self.relative_path = relative_path
        self.payload = payload
        self.is_package = is_package

    def create_module(self, spec: Any) -> None:
        return None

    def exec_module(self, module: Any) -> None:
        synthetic_path = f"<verified-dp5q:{self.relative_path}>"
        module.__file__ = synthetic_path
        module.__loader__ = self
        module.__package__ = (
            self.module_name
            if self.is_package
            else self.module_name.rpartition(".")[0]
        )
        if self.is_package:
            module.__path__ = []
        code = compile(self.payload, synthetic_path, "exec")
        exec(code, module.__dict__)


class _VerifiedSourceFinder(importlib.abc.MetaPathFinder):
    """Import the pinned upstream source bytes without reopening repository paths."""

    def __init__(self, source_files: dict[str, bytes]):
        self.records: dict[str, tuple[str, bytes, bool]] = {}
        for relative_path, payload in source_files.items():
            module_name, module_payload, is_package = _module_record(
                relative_path,
                payload,
            )
            self.records[module_name] = (
                relative_path,
                module_payload,
                is_package,
            )

    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: Any = None,
    ) -> Any:
        record = self.records.get(fullname)
        if record is None:
            return None
        relative_path, payload, is_package = record
        loader = _VerifiedSourceLoader(
            fullname,
            relative_path,
            payload,
            is_package=is_package,
        )
        return importlib.util.spec_from_loader(
            fullname,
            loader,
            is_package=is_package,
        )


def _safe_archive_members(archive: zipfile.ZipFile) -> dict[str, zipfile.ZipInfo]:
    """Validate the fixed two-file archive without trusting member paths."""

    members = archive.infolist()
    names = {member.filename for member in members}
    if len(members) != 2 or names != _ARCHIVE_MEMBERS:
        raise RuntimeError(
            "DP5q quantile archive must contain exactly array.npy and model.keras."
        )
    by_name: dict[str, zipfile.ZipInfo] = {}
    for member in members:
        if member.is_dir() or member.flag_bits & 0x1:
            raise RuntimeError(
                "DP5q quantile archive contains a directory or encrypted member."
            )
        if not 0 < member.file_size <= MAX_ARCHIVE_MEMBER_BYTES:
            raise RuntimeError("DP5q quantile archive member size is unsafe.")
        compressed_size = max(member.compress_size, 1)
        if member.file_size / compressed_size > MAX_ARCHIVE_COMPRESSION_RATIO:
            raise RuntimeError(
                "DP5q quantile archive member compression ratio is unsafe."
            )
        unix_mode = (member.external_attr >> 16) & 0xFFFF
        if unix_mode and (unix_mode & 0o170000) not in {0, 0o100000}:
            raise RuntimeError(
                "DP5q quantile archive contains a non-regular member."
            )
        by_name[member.filename] = member
    return by_name


def _copy_fixed_member(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    destination: Path,
) -> None:
    """Extract a validated member to a caller-selected fixed path."""

    written = 0
    with archive.open(member, "r") as source, destination.open("xb") as target:
        while True:
            chunk = source.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > member.file_size or written > MAX_ARCHIVE_MEMBER_BYTES:
                raise RuntimeError("DP5q quantile archive expanded beyond its limit.")
            target.write(chunk)
    if written != member.file_size:
        raise RuntimeError("DP5q quantile archive member length mismatch.")


def _load_quantile_archive(archive_payload: bytes) -> tuple[Any, Any]:
    """Load the trusted model without calling ZipFile.extract/extractall."""

    try:
        with tempfile.TemporaryDirectory(prefix="chemapp-dp5q-") as directory:
            temporary = Path(directory)
            with zipfile.ZipFile(io.BytesIO(archive_payload), "r") as archive:
                members = _safe_archive_members(archive)
                array_path = temporary / "array.npy"
                model_path = temporary / "model.keras"
                _copy_fixed_member(archive, members["array.npy"], array_path)
                _copy_fixed_member(archive, members["model.keras"], model_path)

            import keras
            import keras.src.saving.saving_lib as keras_saving_lib
            import numpy as np
            import pandas as pd
            import scipy
            import sklearn
            import tensorflow as tf
            import tqdm
            from dp5.neural_net.CNN_model import QuantileLoss
            from dp5.neural_net.nfp.layers import (
                GatherAtomToBond,
                ReduceAtomToPro,
                ReduceBondToAtom,
                Squeeze,
            )
            from dp5.neural_net.nfp.models import GraphModel

            levels = np.load(array_path, allow_pickle=False)
            expected = np.asarray(QUANTILE_LEVELS, dtype=float)
            if (
                levels.shape != (99,)
                or not np.isfinite(levels).all()
                or not np.allclose(levels, expected, rtol=0.0, atol=1e-12)
            ):
                raise RuntimeError(
                    "DP5q archive contains unexpected quantile levels."
                )
            if (
                tf.__version__ != TENSORFLOW_VERSION
                or keras.__version__ != KERAS_VERSION
                or sys.version.split()[0] != runtime_pin.DP5Q_PYTHON_VERSION
                or np.__version__ != runtime_pin.DP5Q_NUMPY_VERSION
                or pd.__version__ != runtime_pin.DP5Q_PANDAS_VERSION
                or scipy.__version__ != runtime_pin.DP5Q_SCIPY_VERSION
                or sklearn.__version__
                != runtime_pin.DP5Q_SCIKIT_LEARN_VERSION
                or tqdm.__version__ != runtime_pin.DP5Q_TQDM_VERSION
            ):
                raise RuntimeError(
                    "DP5q quantile runtime differs from its exact version pins."
                )

            # The released archive was written on POSIX by Keras 2.14.  That
            # release incorrectly uses the host path separator while looking
            # up logical HDF5 checkpoint paths, so a Windows loader searches
            # for backslash-named groups and reports missing weights.  Limit
            # the compatibility shim to the model-load critical section and
            # restore the library function even when loading fails.
            original_join = keras_saving_lib.tf.io.gfile.join
            try:
                keras_saving_lib.tf.io.gfile.join = posixpath.join
                model = keras_saving_lib.load_model(
                    model_path,
                    custom_objects={
                        "GraphModel": GraphModel,
                        "Squeeze": Squeeze,
                        "GatherAtomToBond": GatherAtomToBond,
                        "ReduceBondToAtom": ReduceBondToAtom,
                        "ReduceAtomToPro": ReduceAtomToPro,
                        "_qloss": QuantileLoss(levels),
                    },
                    compile=False,
                    safe_mode=True,
                )
            finally:
                keras_saving_lib.tf.io.gfile.join = original_join
            return model, levels.copy()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise RuntimeError("Could not safely load the DP5q quantile archive.") from exc


def _distribution_contract() -> dict[str, Any]:
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


def _handshake() -> dict[str, Any]:
    from rdkit import rdBase

    return {
        "type": "handshake",
        "protocol_version": PROTOCOL_VERSION,
        "status": "ready",
        "repository_commit": REPOSITORY_COMMIT,
        "sidecar": {
            "code_sha256": mean_contract._sha256(Path(__file__).resolve()),
            "mean_contract_code_sha256": mean_contract._sha256(
                Path(mean_contract.__file__).resolve()
            ),
            "runtime_pin_code_sha256": mean_contract._sha256(
                Path(runtime_pin.__file__).resolve()
            ),
        },
        "assets": {
            "quantile_archive_sha256": QUANTILE_ARCHIVE_SHA256,
            "preprocessor_sha256": PREPROCESSOR_SHA256,
            "upstream_source_bundle_sha256": (
                runtime_pin.DP5Q_UPSTREAM_SOURCE_BUNDLE_SHA256
            ),
        },
        "model": {
            "name": "DP5q-CASCADE-99quantiles",
            "nucleus": "13C",
            "output": "per_conformer_normal_fit_and_boltzmann_quantile_summary",
            "quantile_levels": list(QUANTILE_LEVELS),
            "raw_quantile_tensor_digest": RAW_QUANTILE_TENSOR_DIGEST,
        },
        "capabilities": {
            "accepted_atomic_numbers": sorted(mean_contract.ALLOWED_ATOMIC_NUMBERS),
            "quantile_enabled": True,
            "calibrated_probability": False,
            "operations": [
                "conformer_preflight",
                "predict_13c_quantiles",
            ],
        },
        "runtime": {
            "archive_checkpoint_path_separator": "posix",
            "keras_version": KERAS_VERSION,
            "numpy_version": runtime_pin.DP5Q_NUMPY_VERSION,
            "pandas_version": runtime_pin.DP5Q_PANDAS_VERSION,
            "python_version": runtime_pin.DP5Q_PYTHON_VERSION,
            "rdkit_version": rdBase.rdkitVersion,
            "scikit_learn_version": (
                runtime_pin.DP5Q_SCIKIT_LEARN_VERSION
            ),
            "scipy_version": runtime_pin.DP5Q_SCIPY_VERSION,
            "tensorflow_version": TENSORFLOW_VERSION,
            "tqdm_version": runtime_pin.DP5Q_TQDM_VERSION,
        },
        "conformer_generation": mean_contract._conformer_policy(),
        "conformer_preflight": {
            "protocol_version": (
                mean_contract.CONFORMER_PREFLIGHT_PROTOCOL_VERSION
            ),
            "candidate_failures_are_results": True,
            "operational_failures_abort_request": True,
            "uses_same_prepare_candidate_path_as_prediction": True,
        },
        "distribution_fit": _distribution_contract(),
    }


def _parse_request(raw: bytes) -> dict[str, Any]:
    if len(raw) > mean_contract.MAX_REQUEST_BYTES:
        raise mean_contract.SidecarRequestError(
            "request_too_large", "Request is too large."
        )
    try:
        text = raw.decode("utf-8")

        def reject_constant(value: str) -> None:
            raise ValueError(f"non-finite number {value}")

        value = json.loads(text, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise mean_contract.SidecarRequestError(
            "invalid_json", "Request must be finite UTF-8 JSON."
        ) from exc
    request = mean_contract._strict_object(
        value,
        {"protocol_version", "op", "request_id", "candidates"},
        context="request",
    )
    if request["protocol_version"] != PROTOCOL_VERSION:
        raise mean_contract.SidecarRequestError(
            "protocol_mismatch", "Unsupported protocol_version."
        )
    request_id = request["request_id"]
    if not isinstance(request_id, str) or not request_id or len(request_id) > 128:
        raise mean_contract.SidecarRequestError(
            "invalid_request_id", "request_id must be a non-empty short string."
        )
    if request["op"] not in {
        "conformer_preflight",
        "predict_13c_quantiles",
        "shutdown",
    }:
        raise mean_contract.SidecarRequestError(
            "invalid_operation", "Unsupported operation."
        )
    candidates = request["candidates"]
    if not isinstance(candidates, list):
        raise mean_contract.SidecarRequestError(
            "invalid_schema", "candidates must be a JSON list."
        )
    if request["op"] == "shutdown":
        if candidates:
            raise mean_contract.SidecarRequestError(
                "invalid_schema", "shutdown candidates must be empty."
            )
        return dict(request)
    if not candidates or len(candidates) > mean_contract.MAX_CANDIDATES:
        raise mean_contract.SidecarRequestError(
            "invalid_candidate_count",
            "Candidate count must be between 1 and "
            f"{mean_contract.MAX_CANDIDATES}.",
        )

    clean_candidates: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    for position, raw_candidate in enumerate(candidates):
        candidate = mean_contract._strict_object(
            raw_candidate,
            {"candidate_id", "smiles"},
            context=f"candidate {position}",
        )
        candidate_id = candidate["candidate_id"]
        smiles = candidate["smiles"]
        if (
            not isinstance(candidate_id, str)
            or not candidate_id.strip()
            or len(candidate_id) > mean_contract.MAX_CANDIDATE_ID_LENGTH
            or candidate_id in seen_ids
        ):
            raise mean_contract.SidecarRequestError(
                "invalid_candidate_id",
                f"candidate {position} has an invalid or duplicate candidate_id.",
            )
        if (
            not isinstance(smiles, str)
            or not smiles.strip()
            or len(smiles) > mean_contract.MAX_SMILES_LENGTH
        ):
            raise mean_contract.SidecarRequestError(
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


class DP5qQuantileRuntime(mean_contract.DP5qMeanRuntime):
    """Reuse the frozen conformer policy and run official quantile inference."""

    def __init__(self, verified_inputs: dict[str, Any]):
        if any(
            name == "dp5"
            or name.startswith("dp5.")
            or name == "nfp"
            or name.startswith("nfp.")
            for name in sys.modules
        ):
            raise RuntimeError(
                "DP5q upstream modules were imported before source verification."
            )
        self._source_finder = _VerifiedSourceFinder(
            verified_inputs["source_files"]
        )
        sys.meta_path.insert(0, self._source_finder)
        try:
            import numpy as np
            from rdkit import Chem
            from rdkit import rdBase
            from rdkit.Chem import AllChem, rdMolDescriptors
            from scipy.optimize import curve_fit
            from scipy.stats import norm
            from dp5.neural_net.CNN_model import (
                CASCADE_Quantile,
                Mol_iter2,
                RBFSequence,
                mols_to_df,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not import the pinned DP5q quantile-model runtime."
            ) from exc
        for module_name, module in list(sys.modules.items()):
            prefix = "dp5.neural_net.nfp"
            if module_name == prefix or module_name.startswith(prefix + "."):
                alias = "nfp" + module_name.removeprefix(prefix)
                sys.modules.setdefault(alias, module)
        if rdBase.rdkitVersion != runtime_pin.DP5Q_RDKIT_VERSION:
            raise RuntimeError("DP5q RDKit runtime differs from its exact pin.")
        try:
            preprocessor = pickle.loads(verified_inputs["preprocessor"])
        except Exception as exc:
            raise RuntimeError(
                "Could not load the hash-pinned DP5q preprocessor bytes."
            ) from exc
        if not callable(getattr(preprocessor, "predict", None)):
            raise RuntimeError("DP5q preprocessor contract is invalid.")

        self.np = np
        self.Chem = Chem
        self.AllChem = AllChem
        self.rdMolDescriptors = rdMolDescriptors
        self.curve_fit = curve_fit
        self.norm = norm
        self.mols_to_df = mols_to_df
        self.Mol_iter2 = Mol_iter2
        self.RBFSequence = RBFSequence
        self.preprocessor = preprocessor
        model, levels = _load_quantile_archive(
            verified_inputs["quantile_archive"]
        )
        self.levels = np.asarray(levels, dtype=float)
        self.model = CASCADE_Quantile(model, self.levels)

    def _extract_representations(
        self,
        frame: Any,
        *,
        batch_size: int,
    ) -> list[Any]:
        """Run upstream preprocessing from the one verified in-memory object."""

        inputs = self.preprocessor.predict(self.Mol_iter2(frame))
        sequence = self.RBFSequence(
            inputs,
            frame.atom_index,
            batch_size,
        )
        representations: list[Any] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for batch in sequence:
                predicted = self.model(batch[0])
                indices = batch[0]["n_pro"].cumsum()[:-1]
                representations.extend(
                    self.np.split(predicted.numpy(), indices)
                )
        return representations

    def _fit_normal(self, quantiles: Any) -> tuple[float, float, int, float]:
        values = self.np.asarray(quantiles, dtype=float)
        if values.shape != (99,) or not self.np.isfinite(values).all():
            raise RuntimeError("DP5q returned malformed quantiles.")
        if self.np.any(values < -100.0) or self.np.any(values > 400.0):
            raise RuntimeError("DP5q returned quantiles outside the safety range.")
        differences = self.np.diff(values)
        crossing_depths = -differences[differences < 0.0]
        crossing_count = int(crossing_depths.size)
        max_crossing = (
            float(self.np.max(crossing_depths)) if crossing_count else 0.0
        )
        if max_crossing > MAX_QUANTILE_CROSSING_PPM:
            raise RuntimeError("DP5q quantile crossing exceeds the safety limit.")
        median = float(values[len(values) // 2])
        initial_sigma = float(
            (
                values[len(values) * 2 // 3]
                - values[len(values) // 3]
            )
            / 2
        )
        if not math.isfinite(initial_sigma) or initial_sigma <= 0.0:
            raise RuntimeError("DP5q quantiles cannot initialise a normal fit.")
        parameters = self.curve_fit(
            self.norm.cdf,
            values,
            self.levels,
            p0=[median, initial_sigma],
        )[0]
        mu, sigma = (float(parameters[0]), float(parameters[1]))
        if (
            not math.isfinite(mu)
            or not math.isfinite(sigma)
            or not -20.0 <= mu <= 300.0
            or not 0.0 < sigma <= 100.0
        ):
            raise RuntimeError("DP5q returned an invalid fitted distribution.")
        return mu, sigma, crossing_count, max_crossing

    def _raw_quantile_tensor_sha256(
        self,
        matrix: Any,
        atom_indices: list[int],
        weights: Any,
    ) -> str:
        canonical = self.np.ascontiguousarray(
            matrix,
            dtype=self.np.dtype("<f8"),
        )
        metadata = {
            "atom_indices": atom_indices,
            "digest_contract": RAW_QUANTILE_TENSOR_DIGEST,
            "shape": [int(value) for value in canonical.shape],
        }
        encoded_metadata = json.dumps(
            metadata,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest = hashlib.sha256()
        digest.update(encoded_metadata)
        digest.update(b"\0")
        canonical_weights = self.np.asarray(weights, dtype=self.np.dtype("<f8"))
        if canonical_weights.shape != (canonical.shape[0],):
            raise RuntimeError(
                "DP5q raw quantile digest weights do not match conformers."
            )
        conformer_records = sorted(
            canonical_weights[index : index + 1].tobytes(order="C")
            + canonical[index].tobytes(order="C")
            for index in range(canonical.shape[0])
        )
        for record in conformer_records:
            digest.update(record)
        return digest.hexdigest()

    def predict(
        self,
        candidates: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        conformer_molecules: list[list[Any]] = []
        populations: list[list[float]] = []
        canonical_smiles: list[str] = []
        candidate_warnings: list[list[str]] = []
        for candidate in candidates:
            mols, weights, canonical, warnings = self._prepare_candidate(candidate)
            conformer_molecules.append(mols)
            populations.append(weights)
            canonical_smiles.append(canonical)
            candidate_warnings.append(warnings)

        frame, labels = self.mols_to_df(conformer_molecules, "C")
        arrays = self._extract_representations(frame, batch_size=16)
        if len(arrays) != len(frame) or len(labels) != len(candidates):
            raise RuntimeError("DP5q returned the wrong conformer count.")
        frame = frame.copy()
        frame["quantiles"] = arrays

        predictions: list[dict[str, Any]] = []
        grouped = list(frame.groupby("mol_id", sort=False)["quantiles"])
        if len(grouped) != len(candidates):
            raise RuntimeError("DP5q returned the wrong candidate count.")
        for candidate_index, ((mol_id, rows), candidate) in enumerate(
            zip(grouped, candidates, strict=True)
        ):
            if int(mol_id) != candidate_index:
                raise RuntimeError("DP5q returned candidates out of order.")
            matrix = self.np.stack(
                [self.np.asarray(row, dtype=float) for row in rows],
                axis=0,
            )
            weights = self.np.asarray(populations[candidate_index], dtype=float)
            atom_labels = labels[candidate_index]
            if (
                matrix.ndim != 3
                or matrix.shape
                != (len(weights), len(atom_labels), len(QUANTILE_LEVELS))
                or not self.np.isfinite(matrix).all()
            ):
                raise RuntimeError("DP5q returned a malformed quantile tensor.")
            weighted_quantiles = self.np.average(matrix, axis=0, weights=weights)
            atoms: list[dict[str, Any]] = []
            seen_atoms: set[int] = set()
            raw_tensor_atom_indices: list[int] = []
            for atom_position, (label, summary) in enumerate(
                zip(atom_labels, weighted_quantiles, strict=True)
            ):
                match = mean_contract._LABEL_PATTERN.fullmatch(str(label))
                if match is None:
                    raise RuntimeError("DP5q returned an invalid atom label.")
                atom_index = int(match.group(1)) - 1
                if atom_index < 0 or atom_index in seen_atoms:
                    raise RuntimeError("DP5q returned a duplicate atom label.")
                seen_atoms.add(atom_index)
                raw_tensor_atom_indices.append(atom_index)
                fitted = [
                    self._fit_normal(matrix[conformer_index, atom_position, :])
                    for conformer_index in range(matrix.shape[0])
                ]
                atoms.append(
                    {
                        "atom_index": atom_index,
                        "quantiles_ppm": [float(value) for value in summary],
                        "conformer_mu_ppm": [item[0] for item in fitted],
                        "conformer_sigma_ppm": [item[1] for item in fitted],
                        "conformer_quantile_crossing_counts": [
                            item[2] for item in fitted
                        ],
                        "conformer_max_quantile_crossing_ppm": [
                            item[3] for item in fitted
                        ],
                    }
                )
            raw_tensor_shape = [int(value) for value in matrix.shape]
            raw_tensor_sha256 = self._raw_quantile_tensor_sha256(
                matrix,
                raw_tensor_atom_indices,
                weights,
            )
            predictions.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "canonical_smiles": canonical_smiles[candidate_index],
                    "conformer_count": len(weights),
                    "conformer_populations": [float(value) for value in weights],
                    "raw_conformer_quantiles_shape": raw_tensor_shape,
                    "raw_conformer_quantiles_atom_indices": (
                        raw_tensor_atom_indices
                    ),
                    "raw_conformer_quantiles_sha256": raw_tensor_sha256,
                    "atom_predictions": sorted(
                        atoms, key=lambda item: item["atom_index"]
                    ),
                    "warnings": candidate_warnings[candidate_index],
                }
            )
        return predictions


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
            "kind": "dp5q_13c_quantile_components",
            "calibrated_probability": False,
            "quantile_enabled": True,
            "assigned_observations_required_for_equation_parity_score": True,
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


def _serve(runtime: DP5qQuantileRuntime) -> int:
    mean_contract._emit(_handshake())
    while True:
        raw = sys.stdin.buffer.readline(mean_contract.MAX_REQUEST_BYTES + 1)
        if not raw:
            return 0
        request_id = ""
        try:
            request = _parse_request(raw)
            request_id = request["request_id"]
            if request["op"] == "shutdown":
                return 0
            if request["op"] == "conformer_preflight":
                mean_contract._emit(
                    _preflight_response(
                        request_id,
                        runtime.preflight(request["candidates"]),
                    )
                )
            else:
                mean_contract._emit(
                    _prediction_response(
                        request_id,
                        runtime.predict(request["candidates"]),
                    )
                )
        except mean_contract.SidecarRequestError as exc:
            mean_contract._emit(
                mean_contract._error_response(request_id, exc.code, exc.message)
            )
        except Exception:
            traceback.print_exc(file=sys.stderr)
            mean_contract._emit(
                mean_contract._error_response(
                    request_id,
                    "inference_failed",
                    "The pinned DP5q quantile model could not complete inference.",
                )
            )


def _self_test(runtime: DP5qQuantileRuntime) -> int:
    predictions = runtime.predict(
        [{"candidate_id": "self-test-ethanol", "smiles": "CCO"}]
    )
    if (
        len(predictions) != 1
        or len(predictions[0]["atom_predictions"]) != 2
        or predictions[0]["conformer_count"] < 1
    ):
        raise RuntimeError("DP5q quantile self-test returned malformed output.")
    sys.stderr.write("DP5q quantile sidecar self-test passed.\n")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pinned DP5q 13C quantile-model sidecar."
    )
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    repository = arguments.repo.expanduser().resolve()
    verified_inputs = _verify_repository(repository)
    runtime = DP5qQuantileRuntime(verified_inputs)
    if arguments.self_test:
        return _self_test(runtime)
    return _serve(runtime)


if __name__ == "__main__":
    raise SystemExit(main())
