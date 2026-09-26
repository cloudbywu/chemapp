"""Fail-closed adapter for the isolated DP5q 13C mean-shift sidecar.

DP5q's released TensorFlow environment is incompatible with the backend's
NumPy/Pandas stack, so the model runs in a separately pinned Python process.
The process protocol is JSON Lines and never accepts pickle data from callers.

Only the released 13C *mean* model is enabled here.  Quantile output and DP5q
probabilities remain disabled until the upstream quantile checkpoint can be
loaded reproducibly and checked against author-provided golden predictions.
"""

from __future__ import annotations

import atexit
from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
from typing import Any, Iterable, Mapping, Sequence
import uuid

from . import dp5q_runtime_pin
from .nmr_evidence import FormulaError, canonical_formula


PROTOCOL_VERSION = 1
DP5Q_REPOSITORY_COMMIT = "b79968cf63cb282e8871d5595ea6cef5b4dc0d49"
DP5Q_MEAN_MODEL_SHA256 = (
    "2d453b9c340a45b7e3c8d789a0c6071167e677ccbfe972eec5a5e9c26033d095"
)
DP5Q_PREPROCESSOR_SHA256 = (
    "6d143a468595797a05434a32da76cdcf57cb8b0cc929bfe27181acc297fde1b0"
)
DP5Q_CONFORMER_PROTOCOL_VERSION = "chemapp.dp5q-conformer.v2"
DP5Q_CONFORMER_PREFLIGHT_PROTOCOL_VERSION = "chemapp.dp5q-conformer-preflight.v1"
DP5Q_MODEL_RELATIVE_PATH = Path(
    "dp5/neural_net/NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5"
)
DP5Q_PREPROCESSOR_RELATIVE_PATH = Path("dp5/neural_net/mean_model_preprocessor.p")

DP5Q_PYTHON_ENV = "CHEMAPP_DP5Q_PYTHON"
DP5Q_REPOSITORY_ENV = "CHEMAPP_DP5Q_REPO"
DP5Q_TIMEOUT_ENV = "CHEMAPP_DP5Q_TIMEOUT_SECONDS"
DP5Q_STARTUP_TIMEOUT_ENV = "CHEMAPP_DP5Q_STARTUP_TIMEOUT_SECONDS"

_ALLOWED_ATOMIC_NUMBERS = frozenset({1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35})
_MAX_CANDIDATES = 20
_MAX_CANDIDATE_ID_LENGTH = 256
_MAX_SMILES_LENGTH = 1024
_MAX_HEAVY_ATOMS = 80
_MAX_TOTAL_ATOMS = 200
_MAX_ROTATABLE_BONDS = 20
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024
_MAX_STDERR_LINES = 40
_EOF = object()
_RUNTIME_VERSION_FIELDS = frozenset(
    {
        "python",
        "tensorflow",
        "keras",
        "numpy",
        "pandas",
        "scipy",
        "rdkit",
        "scikit_learn",
        "tqdm",
    }
)
_RUNTIME_PROBE_CODE = """\
import importlib.metadata as metadata
import json
import platform
from rdkit import rdBase
value = {
    "keras": metadata.version("keras"),
    "numpy": metadata.version("numpy"),
    "pandas": metadata.version("pandas"),
    "python": platform.python_version(),
    "rdkit": rdBase.rdkitVersion,
    "scikit_learn": metadata.version("scikit-learn"),
    "scipy": metadata.version("scipy"),
    "tensorflow": metadata.version("tensorflow"),
    "tqdm": metadata.version("tqdm"),
}
print(json.dumps(value, sort_keys=True, separators=(",", ":")))
"""
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


def dp5q_conformer_policy() -> dict[str, Any]:
    """Return the exact sidecar conformer-generation contract."""

    return {
        "protocol_version": DP5Q_CONFORMER_PROTOCOL_VERSION,
        "requested_conformers": 20,
        "random_seed": 0xC0FFEE,
        "num_threads": 1,
        "rms_prune_angstrom": 0.5,
        "attempts": [
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
        ],
        "optimiser": "MMFF94s",
        "mmff_max_iterations": 1000,
        "energy_window_kj_mol": 10.0,
        "temperature_k": 298.15,
        "gas_constant_kj_mol_k": 0.00831446261815324,
        "population_weighting": "boltzmann",
        "terminal_failure": "fail_closed_no_uff",
    }


class NMRForwardError(RuntimeError):
    """Base exception for forward-model failures."""


class NMRForwardConfigurationError(NMRForwardError):
    """The isolated model runtime or pinned assets are not configured safely."""


class NMRForwardInputError(NMRForwardError, ValueError):
    """Caller input does not satisfy the forward-model contract."""


class NMRForwardProtocolError(NMRForwardError):
    """The sidecar returned malformed or untrusted output."""


class NMRForwardTimeoutError(NMRForwardError, TimeoutError):
    """The sidecar did not answer within the configured deadline."""


class NMRForwardUnavailableError(NMRForwardError):
    """The sidecar crashed or could not be started."""


def _default_sidecar_script() -> Path:
    return Path(__file__).resolve().parents[2] / "scripts" / "dp5q_sidecar.py"


def _parse_timeout(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise NMRForwardConfigurationError(
            f"{name} must be a finite number of seconds."
        ) from exc
    if not math.isfinite(value) or not 0.05 <= value <= 300.0:
        raise NMRForwardConfigurationError(
            f"{name} must be between 0.05 and 300 seconds."
        )
    return value


@dataclass(frozen=True)
class NMRForwardConfig:
    """Configuration for a pinned external DP5q mean-model process."""

    python_executable: str | Path
    repository: Path
    sidecar_script: Path = field(default_factory=_default_sidecar_script)
    timeout_seconds: float = 20.0
    startup_timeout_seconds: float = 45.0
    expected_commit: str = DP5Q_REPOSITORY_COMMIT
    expected_model_sha256: str = DP5Q_MEAN_MODEL_SHA256
    expected_preprocessor_sha256: str = DP5Q_PREPROCESSOR_SHA256
    verify_local_install: bool = True

    @classmethod
    def from_environment(cls) -> "NMRForwardConfig":
        python_value = os.getenv(DP5Q_PYTHON_ENV, "").strip()
        repository_value = os.getenv(DP5Q_REPOSITORY_ENV, "").strip()
        if not python_value or not repository_value:
            missing = [
                name
                for name, value in (
                    (DP5Q_PYTHON_ENV, python_value),
                    (DP5Q_REPOSITORY_ENV, repository_value),
                )
                if not value
            ]
            raise NMRForwardConfigurationError(
                "DP5q sidecar is not configured; missing " + ", ".join(missing)
            )
        return cls(
            python_executable=python_value,
            repository=Path(repository_value).expanduser(),
            timeout_seconds=_parse_timeout(DP5Q_TIMEOUT_ENV, 20.0),
            startup_timeout_seconds=_parse_timeout(DP5Q_STARTUP_TIMEOUT_ENV, 45.0),
        )


def dp5q_sidecar_is_configured() -> bool:
    """Return whether both required external-runtime variables are present."""

    return bool(
        os.getenv(DP5Q_PYTHON_ENV, "").strip()
        and os.getenv(DP5Q_REPOSITORY_ENV, "").strip()
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    rendered = json.dumps(
        dict(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _expected_runtime_versions() -> dict[str, str]:
    return {
        "python": dp5q_runtime_pin.DP5Q_PYTHON_VERSION,
        "tensorflow": dp5q_runtime_pin.DP5Q_TENSORFLOW_VERSION,
        "keras": dp5q_runtime_pin.DP5Q_KERAS_VERSION,
        "numpy": dp5q_runtime_pin.DP5Q_NUMPY_VERSION,
        "pandas": dp5q_runtime_pin.DP5Q_PANDAS_VERSION,
        "scipy": dp5q_runtime_pin.DP5Q_SCIPY_VERSION,
        "rdkit": dp5q_runtime_pin.DP5Q_RDKIT_VERSION,
        "scikit_learn": dp5q_runtime_pin.DP5Q_SCIKIT_LEARN_VERSION,
        "tqdm": dp5q_runtime_pin.DP5Q_TQDM_VERSION,
    }


def _runtime_pin_code_sha256() -> str:
    module_path = Path(str(dp5q_runtime_pin.__file__)).resolve()
    if not module_path.is_file() or module_path.suffix != ".py":
        raise NMRForwardConfigurationError(
            "DP5q runtime-pin implementation is not a verifiable Python source file."
        )
    return _sha256(module_path)


_SIDECAR_ENV_ALLOWLIST = frozenset(
    {
        "APPDATA",
        "COMSPEC",
        "CONDA_PREFIX",
        "HOME",
        "HOMEDRIVE",
        "HOMEPATH",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "LOCALAPPDATA",
        "PATH",
        "PATHEXT",
        "SYSTEMDRIVE",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "TZ",
        "USERPROFILE",
        "VIRTUAL_ENV",
        "WINDIR",
    }
)


def _clean_sidecar_environment() -> dict[str, str]:
    """Build the same small, non-shadowable environment for probe and run."""

    env = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in _SIDECAR_ENV_ALLOWLIST
        or key.upper().startswith(("DP5_", "DP5Q_", "FAKE_DP5Q_"))
    }
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["TF_NUM_INTRAOP_THREADS"] = "1"
    env["TF_NUM_INTEROP_THREADS"] = "1"
    return env


def _probe_external_runtime(
    executable: str,
    *,
    timeout_seconds: float,
) -> dict[str, str]:
    """Probe the selected isolated Python and require exact canonical JSON."""

    env = _clean_sidecar_environment()
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    try:
        result = subprocess.run(
            [executable, "-I", "-c", _RUNTIME_PROBE_CODE],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=max(1.0, min(float(timeout_seconds), 30.0)),
            env=env,
            creationflags=creationflags,
        )
    except (OSError, subprocess.SubprocessError, UnicodeError) as exc:
        raise NMRForwardConfigurationError(
            "Could not probe the selected DP5q Python runtime."
        ) from exc
    if (
        result.returncode != 0
        or result.stderr != ""
        or len(result.stdout.encode("utf-8")) > 4096
    ):
        raise NMRForwardConfigurationError(
            "Selected DP5q Python runtime probe failed or emitted diagnostics."
        )
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise NMRForwardConfigurationError(
            "Selected DP5q Python runtime probe returned invalid JSON."
        ) from exc
    if (
        not isinstance(parsed, dict)
        or set(parsed) != _RUNTIME_VERSION_FIELDS
        or any(not isinstance(value, str) for value in parsed.values())
    ):
        raise NMRForwardConfigurationError(
            "Selected DP5q Python runtime probe schema changed."
        )
    canonical_line = (
        json.dumps(
            parsed,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    )
    if result.stdout != canonical_line:
        raise NMRForwardConfigurationError(
            "Selected DP5q Python runtime probe was not canonical JSON."
        )
    expected = _expected_runtime_versions()
    if parsed != expected:
        mismatches = sorted(
            key for key in expected if parsed.get(key) != expected[key]
        )
        raise NMRForwardConfigurationError(
            "Selected DP5q Python runtime version mismatch: "
            + ", ".join(mismatches)
        )
    return dict(parsed)


def _resolve_executable(value: str | Path) -> str:
    raw = str(value)
    as_path = Path(raw).expanduser()
    if as_path.is_file():
        return str(as_path.resolve())
    located = shutil.which(raw)
    if located:
        return located
    raise NMRForwardConfigurationError(f"DP5q Python executable does not exist: {raw}")


def _repository_commit(repository: Path) -> str:
    marker = repository / ".chemapp-dp5q-commit"
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
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip().lower()
    raise NMRForwardConfigurationError(
        "Cannot verify the DP5q repository commit (.git or "
        ".chemapp-dp5q-commit is required)."
    )


def _strict_keys(
    value: Mapping[str, Any],
    expected: set[str],
    *,
    context: str,
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise NMRForwardProtocolError(
            f"{context} schema mismatch; missing={missing}, unexpected={unexpected}"
        )


class NMRForwardAdapter:
    """Own one persistent, serialised DP5q sidecar process."""

    def __init__(self, config: NMRForwardConfig):
        self.config = config
        self._process: subprocess.Popen[bytes] | None = None
        self._stdout_queue: queue.Queue[Any] = queue.Queue()
        self._stderr_tail: deque[str] = deque(maxlen=_MAX_STDERR_LINES)
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._handshake: dict[str, Any] | None = None
        self._sidecar_code_sha256: str | None = None
        self._runtime_attestation: dict[str, str] | None = None
        self._closed = False

    def __enter__(self) -> "NMRForwardAdapter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @property
    def handshake(self) -> dict[str, Any] | None:
        """Return a defensive copy of the verified startup metadata."""

        return json.loads(json.dumps(self._handshake)) if self._handshake else None

    def _verify_install(self) -> tuple[str, Path, Path]:
        self._runtime_attestation = None
        executable = _resolve_executable(self.config.python_executable)
        repository = self.config.repository.expanduser().resolve()
        sidecar_script = self.config.sidecar_script.expanduser().resolve()
        if not repository.is_dir():
            raise NMRForwardConfigurationError(
                f"DP5q repository does not exist: {repository}"
            )
        if not sidecar_script.is_file():
            raise NMRForwardConfigurationError(
                f"DP5q sidecar script does not exist: {sidecar_script}"
            )
        if not self.config.verify_local_install:
            self._runtime_attestation = {
                **_expected_runtime_versions(),
                "source_bundle_sha256": (
                    dp5q_runtime_pin.DP5Q_UPSTREAM_SOURCE_BUNDLE_SHA256
                ),
                "runtime_pin_code_sha256": _runtime_pin_code_sha256(),
            }
            return executable, repository, sidecar_script

        commit = _repository_commit(repository)
        if commit != self.config.expected_commit.lower():
            raise NMRForwardConfigurationError(
                "DP5q repository commit mismatch; "
                f"expected {self.config.expected_commit}, got {commit}."
            )
        expected_assets = (
            (
                repository / DP5Q_MODEL_RELATIVE_PATH,
                self.config.expected_model_sha256.lower(),
                "mean model",
            ),
            (
                repository / DP5Q_PREPROCESSOR_RELATIVE_PATH,
                self.config.expected_preprocessor_sha256.lower(),
                "preprocessor",
            ),
        )
        for path, expected_hash, label in expected_assets:
            if not path.is_file():
                raise NMRForwardConfigurationError(f"DP5q {label} is missing: {path}")
            actual_hash = _sha256(path)
            if actual_hash != expected_hash:
                raise NMRForwardConfigurationError(
                    f"DP5q {label} SHA-256 mismatch; expected "
                    f"{expected_hash}, got {actual_hash}."
                )
        try:
            dp5q_runtime_pin.verified_source_bytes(repository)
        except dp5q_runtime_pin.DP5qRuntimePinError as exc:
            raise NMRForwardConfigurationError(
                "DP5q executable source bundle failed its exact pin."
            ) from exc
        runtime_versions = _probe_external_runtime(
            executable,
            timeout_seconds=self.config.startup_timeout_seconds,
        )
        self._runtime_attestation = {
            **runtime_versions,
            "source_bundle_sha256": (
                dp5q_runtime_pin.DP5Q_UPSTREAM_SOURCE_BUNDLE_SHA256
            ),
            "runtime_pin_code_sha256": _runtime_pin_code_sha256(),
        }
        return executable, repository, sidecar_script

    @staticmethod
    def _pump_stdout(stream: Any, output_queue: queue.Queue[Any]) -> None:
        try:
            while True:
                line = stream.readline(_MAX_RESPONSE_BYTES + 1)
                if not line:
                    output_queue.put(_EOF)
                    return
                if len(line) > _MAX_RESPONSE_BYTES:
                    output_queue.put(
                        NMRForwardProtocolError(
                            "DP5q sidecar response exceeded the size limit."
                        )
                    )
                    return
                output_queue.put(line)
        except Exception as exc:  # pragma: no cover - OS pipe edge case
            output_queue.put(exc)

    @staticmethod
    def _pump_stderr(stream: Any, stderr_tail: deque[str]) -> None:
        try:
            while True:
                line = stream.readline(4096)
                if not line:
                    return
                stderr_tail.append(line.decode("utf-8", errors="replace").rstrip())
        except Exception:  # pragma: no cover - diagnostic channel only
            return

    def _stderr_summary(self) -> str:
        summary = "\n".join(self._stderr_tail).strip()
        return summary[-2000:] if summary else "no stderr output"

    def _abort(self) -> None:
        process = self._process
        self._process = None
        self._handshake = None
        self._sidecar_code_sha256 = None
        self._runtime_attestation = None
        if process is None:
            return
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=0.75)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=0.75)
        except (OSError, subprocess.SubprocessError):
            pass
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        while True:
            try:
                self._stdout_queue.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        with self._lock:
            self._closed = True
            process = self._process
            if process and process.poll() is None and process.stdin:
                try:
                    payload = {
                        "protocol_version": PROTOCOL_VERSION,
                        "op": "shutdown",
                        "request_id": uuid.uuid4().hex,
                        "candidates": [],
                    }
                    process.stdin.write(
                        (json.dumps(payload, separators=(",", ":")) + "\n").encode(
                            "utf-8"
                        )
                    )
                    process.stdin.flush()
                except OSError:
                    pass
            self._abort()

    def _read_json(self, timeout: float) -> dict[str, Any]:
        try:
            item = self._stdout_queue.get(timeout=timeout)
        except queue.Empty as exc:
            self._abort()
            raise NMRForwardTimeoutError(
                f"DP5q sidecar exceeded the {timeout:.3g}s deadline."
            ) from exc
        if item is _EOF:
            message = self._stderr_summary()
            self._abort()
            raise NMRForwardUnavailableError(
                "DP5q sidecar exited before responding: " + message
            )
        if isinstance(item, Exception):
            self._abort()
            if isinstance(item, NMRForwardError):
                raise item
            raise NMRForwardProtocolError(
                f"Could not read DP5q sidecar output: {item}"
            ) from item
        try:
            value = json.loads(item.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._abort()
            raise NMRForwardProtocolError(
                "DP5q sidecar returned invalid JSON."
            ) from exc
        if not isinstance(value, dict):
            self._abort()
            raise NMRForwardProtocolError(
                "DP5q sidecar response must be a JSON object."
            )
        return value

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
            },
            context="handshake",
        )
        if (
            value["type"] != "handshake"
            or value["protocol_version"] != PROTOCOL_VERSION
            or value["status"] != "ready"
        ):
            raise NMRForwardProtocolError("DP5q sidecar handshake was not ready.")
        if value["repository_commit"] != self.config.expected_commit.lower():
            raise NMRForwardProtocolError(
                "DP5q sidecar reported an unexpected repository commit."
            )

        sidecar = value["sidecar"]
        if not isinstance(sidecar, dict):
            raise NMRForwardProtocolError("Handshake sidecar must be an object.")
        _strict_keys(sidecar, {"code_sha256"}, context="handshake sidecar")
        code_sha256 = sidecar["code_sha256"]
        if (
            not isinstance(code_sha256, str)
            or len(code_sha256) != 64
            or any(character not in "0123456789abcdef" for character in code_sha256)
            or code_sha256 != self._sidecar_code_sha256
        ):
            raise NMRForwardProtocolError(
                "DP5q sidecar reported an unexpected implementation hash."
            )

        assets = value["assets"]
        if not isinstance(assets, dict):
            raise NMRForwardProtocolError("Handshake assets must be an object.")
        _strict_keys(
            assets,
            {"mean_model_sha256", "preprocessor_sha256"},
            context="handshake assets",
        )
        if assets["mean_model_sha256"] != self.config.expected_model_sha256.lower():
            raise NMRForwardProtocolError(
                "DP5q sidecar reported an unexpected mean-model hash."
            )
        if (
            assets["preprocessor_sha256"]
            != self.config.expected_preprocessor_sha256.lower()
        ):
            raise NMRForwardProtocolError(
                "DP5q sidecar reported an unexpected preprocessor hash."
            )

        model = value["model"]
        if not isinstance(model, dict):
            raise NMRForwardProtocolError("Handshake model must be an object.")
        _strict_keys(
            model,
            {"name", "nucleus", "output"},
            context="handshake model",
        )
        if model != {
            "name": "DP5q-CASCADE-mean",
            "nucleus": "13C",
            "output": "boltzmann_weighted_mean_shift_ppm",
        }:
            raise NMRForwardProtocolError(
                "DP5q sidecar model capability does not match the pinned adapter."
            )

        capabilities = value["capabilities"]
        if not isinstance(capabilities, dict):
            raise NMRForwardProtocolError("Handshake capabilities must be an object.")
        _strict_keys(
            capabilities,
            {
                "accepted_atomic_numbers",
                "quantile_enabled",
                "calibrated_probability",
                "operations",
            },
            context="handshake capabilities",
        )
        atomic_numbers = capabilities["accepted_atomic_numbers"]
        if (
            not isinstance(atomic_numbers, list)
            or any(type(number) is not int for number in atomic_numbers)
            or atomic_numbers != sorted(_ALLOWED_ATOMIC_NUMBERS)
        ):
            raise NMRForwardProtocolError(
                "DP5q sidecar reported an unexpected element vocabulary."
            )
        if (
            capabilities["quantile_enabled"] is not False
            or capabilities["calibrated_probability"] is not False
            or capabilities["operations"] != ["conformer_preflight", "predict_13c_mean"]
        ):
            raise NMRForwardProtocolError(
                "Quantile or probability output must remain disabled."
            )
        runtime = value["runtime"]
        if not isinstance(runtime, dict):
            raise NMRForwardProtocolError("Handshake runtime must be an object.")
        _strict_keys(runtime, {"rdkit_version"}, context="handshake runtime")
        rdkit_version = runtime["rdkit_version"]
        if not isinstance(rdkit_version, str) or not re.fullmatch(
            r"\d{4}\.\d{2}\.\d+", rdkit_version
        ):
            raise NMRForwardProtocolError(
                "DP5q sidecar reported an invalid RDKit version."
            )
        runtime_attestation = self._runtime_attestation
        if runtime_attestation is None:
            raise NMRForwardConfigurationError(
                "DP5q runtime was not attested before sidecar startup."
            )
        if (
            set(runtime_attestation)
            != _RUNTIME_VERSION_FIELDS
            | {"source_bundle_sha256", "runtime_pin_code_sha256"}
            or runtime_attestation.get("rdkit") != rdkit_version
        ):
            raise NMRForwardProtocolError(
                "DP5q sidecar and attested runtime versions disagree."
            )
        conformer_generation = value["conformer_generation"]
        if (
            not isinstance(conformer_generation, dict)
            or conformer_generation != dp5q_conformer_policy()
        ):
            raise NMRForwardProtocolError(
                "DP5q sidecar reported an unexpected conformer-generation policy."
            )
        conformer_preflight = value["conformer_preflight"]
        if conformer_preflight != {
            "protocol_version": DP5Q_CONFORMER_PREFLIGHT_PROTOCOL_VERSION,
            "candidate_failures_are_results": True,
            "operational_failures_abort_request": True,
            "uses_same_prepare_candidate_path_as_prediction": True,
        }:
            raise NMRForwardProtocolError(
                "DP5q sidecar reported an unexpected conformer-preflight policy."
            )
        validated = json.loads(json.dumps(value))
        legacy_runtime = {
            "sidecar_code_sha256": code_sha256,
            "rdkit_version": rdkit_version,
        }
        validated["runtime"] = {
            **runtime_attestation,
            "rdkit_version": rdkit_version,
            "sidecar_code_sha256": code_sha256,
            "legacy_preflight_runtime_sha256": _canonical_sha256(
                legacy_runtime
            ),
        }
        return validated

    def _start(self) -> None:
        if self._closed:
            raise NMRForwardUnavailableError("DP5q adapter has been closed.")
        if self._process is not None and self._process.poll() is None:
            return
        self._abort()
        executable, repository, sidecar_script = self._verify_install()
        self._sidecar_code_sha256 = _sha256(sidecar_script)
        self._stdout_queue = queue.Queue()
        self._stderr_tail = deque(maxlen=_MAX_STDERR_LINES)
        env = _clean_sidecar_environment()
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            process = subprocess.Popen(
                [
                    executable,
                    "-I",
                    "-u",
                    str(sidecar_script),
                    "--repo",
                    str(repository),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=str(sidecar_script.parent),
                env=env,
                creationflags=creationflags,
            )
        except OSError as exc:
            raise NMRForwardUnavailableError(
                f"Could not start the DP5q sidecar: {exc}"
            ) from exc
        self._process = process
        assert process.stdout is not None
        assert process.stderr is not None
        self._stdout_thread = threading.Thread(
            target=self._pump_stdout,
            args=(process.stdout, self._stdout_queue),
            name="dp5q-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr,
            args=(process.stderr, self._stderr_tail),
            name="dp5q-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()
        try:
            self._handshake = self._validate_handshake(
                self._read_json(self.config.startup_timeout_seconds)
            )
        except Exception:
            self._abort()
            raise

    def _request(
        self,
        candidates: list[dict[str, str]],
        *,
        operation: str = "predict_13c_mean",
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "op": operation,
            "request_id": request_id,
            "candidates": candidates,
        }
        encoded = (
            json.dumps(payload, ensure_ascii=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        with self._lock:
            self._start()
            process = self._process
            assert process is not None
            assert process.stdin is not None
            try:
                process.stdin.write(encoded)
                process.stdin.flush()
            except OSError as exc:
                summary = self._stderr_summary()
                self._abort()
                raise NMRForwardUnavailableError(
                    "DP5q sidecar pipe failed: " + summary
                ) from exc
            response = self._read_json(self.config.timeout_seconds)
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
                        context="error response",
                    )
                    if (
                        response["type"] != "prediction"
                        or response["protocol_version"] != PROTOCOL_VERSION
                        or response["request_id"] != request_id
                        or not isinstance(response["error"], dict)
                    ):
                        raise NMRForwardProtocolError(
                            "DP5q sidecar returned an invalid error response."
                        )
                    error = response["error"]
                    if set(error) != {"code", "message"} or not all(
                        isinstance(error[key], str) for key in ("code", "message")
                    ):
                        raise NMRForwardProtocolError(
                            "DP5q sidecar error payload is malformed."
                        )
                    raise NMRForwardUnavailableError(
                        f"DP5q sidecar rejected the request "
                        f"({error['code']}): {error['message']}"
                    )
                if operation == "conformer_preflight":
                    _strict_keys(
                        response,
                        {
                            "type",
                            "protocol_version",
                            "request_id",
                            "status",
                            "results",
                            "model_outputs_created",
                        },
                        context="conformer-preflight response",
                    )
                    if (
                        response["type"] != "conformer_preflight"
                        or response["protocol_version"] != PROTOCOL_VERSION
                        or response["request_id"] != request_id
                        or response["status"] != "ok"
                        or response["model_outputs_created"] is not False
                    ):
                        raise NMRForwardProtocolError(
                            "DP5q conformer-preflight response identity/status "
                            "mismatch."
                        )
                else:
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
                        context="prediction response",
                    )
                    if (
                        response["type"] != "prediction"
                        or response["protocol_version"] != PROTOCOL_VERSION
                        or response["request_id"] != request_id
                        or response["status"] != "ok"
                    ):
                        raise NMRForwardProtocolError(
                            "DP5q sidecar response identity/status mismatch."
                        )
                    semantics = response["evidence_semantics"]
                    if semantics != {
                        "kind": "relative_13c_forward_evidence",
                        "calibrated_probability": False,
                        "quantile_enabled": False,
                    }:
                        raise NMRForwardProtocolError(
                            "DP5q sidecar attempted to change evidence semantics."
                        )
                return response
            except NMRForwardProtocolError:
                self._abort()
                raise

    @staticmethod
    def _normalise_candidates(
        candidates: Sequence[Mapping[str, Any]],
        *,
        formula: str | None,
    ) -> tuple[list[dict[str, str]], dict[str, set[int]]]:
        if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
            raise NMRForwardInputError("candidates must be a sequence of objects.")
        if not candidates:
            raise NMRForwardInputError("At least one candidate is required.")
        if len(candidates) > _MAX_CANDIDATES:
            raise NMRForwardInputError(
                f"At most {_MAX_CANDIDATES} candidates may be evaluated at once."
            )
        try:
            from rdkit import Chem
            from rdkit.Chem import rdMolDescriptors
        except ImportError as exc:  # pragma: no cover - required backend dependency
            raise NMRForwardConfigurationError(
                "RDKit is required to validate forward-model candidates."
            ) from exc

        expected_formula: str | None = None
        if formula is not None:
            if not isinstance(formula, str):
                raise NMRForwardInputError("formula must be a string when provided.")
            try:
                expected_formula = canonical_formula(formula)
            except FormulaError as exc:
                raise NMRForwardInputError(str(exc)) from exc
            if not expected_formula:
                raise NMRForwardInputError("formula must be non-empty when provided.")

        normalised: list[dict[str, str]] = []
        carbon_indices: dict[str, set[int]] = {}
        seen_ids: set[str] = set()
        for position, candidate in enumerate(candidates):
            if not isinstance(candidate, Mapping):
                raise NMRForwardInputError(f"candidate {position} must be an object.")
            keys = set(candidate)
            if keys != {"candidate_id", "smiles"}:
                raise NMRForwardInputError(
                    f"candidate {position} requires exactly candidate_id and smiles."
                )
            candidate_id = candidate["candidate_id"]
            smiles = candidate["smiles"]
            if (
                not isinstance(candidate_id, str)
                or not candidate_id.strip()
                or len(candidate_id) > _MAX_CANDIDATE_ID_LENGTH
            ):
                raise NMRForwardInputError(
                    f"candidate {position} has an invalid candidate_id."
                )
            if candidate_id in seen_ids:
                raise NMRForwardInputError(f"Duplicate candidate_id: {candidate_id}")
            seen_ids.add(candidate_id)
            if (
                not isinstance(smiles, str)
                or not smiles.strip()
                or len(smiles) > _MAX_SMILES_LENGTH
            ):
                raise NMRForwardInputError(
                    f"candidate {candidate_id} has an invalid SMILES string."
                )
            molecule = Chem.MolFromSmiles(smiles)
            if molecule is None:
                raise NMRForwardInputError(
                    f"candidate {candidate_id} has invalid SMILES."
                )
            if len(Chem.GetMolFrags(molecule)) != 1:
                raise NMRForwardInputError(
                    f"candidate {candidate_id} contains multiple fragments."
                )
            if molecule.GetNumHeavyAtoms() > _MAX_HEAVY_ATOMS:
                raise NMRForwardInputError(
                    f"candidate {candidate_id} exceeds the heavy-atom limit."
                )
            if Chem.AddHs(molecule).GetNumAtoms() > _MAX_TOTAL_ATOMS:
                raise NMRForwardInputError(
                    f"candidate {candidate_id} exceeds the total-atom limit."
                )
            if rdMolDescriptors.CalcNumRotatableBonds(molecule) > _MAX_ROTATABLE_BONDS:
                raise NMRForwardInputError(
                    f"candidate {candidate_id} exceeds the rotatable-bond limit."
                )
            atomic_numbers = {atom.GetAtomicNum() for atom in molecule.GetAtoms()}
            unsupported = sorted(atomic_numbers - _ALLOWED_ATOMIC_NUMBERS)
            if unsupported:
                raise NMRForwardInputError(
                    f"candidate {candidate_id} contains unsupported atomic "
                    f"numbers: {unsupported}"
                )
            indices = {
                atom.GetIdx()
                for atom in molecule.GetAtoms()
                if atom.GetAtomicNum() == 6
            }
            if not indices:
                raise NMRForwardInputError(
                    f"candidate {candidate_id} contains no carbon atoms."
                )
            if expected_formula is not None:
                try:
                    actual_formula = canonical_formula(
                        rdMolDescriptors.CalcMolFormula(molecule)
                    )
                except FormulaError as exc:
                    raise NMRForwardInputError(
                        f"candidate {candidate_id} has an unsupported formula."
                    ) from exc
                if actual_formula != expected_formula:
                    raise NMRForwardInputError(
                        f"candidate {candidate_id} formula mismatch: "
                        f"{actual_formula} != {expected_formula}"
                    )
            normalised.append({"candidate_id": candidate_id, "smiles": smiles.strip()})
            carbon_indices[candidate_id] = indices
        return normalised, carbon_indices

    @staticmethod
    def _validate_predictions(
        value: Any,
        candidates: list[dict[str, str]],
        expected_carbon_indices: Mapping[str, set[int]],
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list) or len(value) != len(candidates):
            raise NMRForwardProtocolError(
                "DP5q sidecar returned the wrong number of predictions."
            )
        try:
            from rdkit import Chem
        except ImportError as exc:  # pragma: no cover
            raise NMRForwardConfigurationError(
                "RDKit is required to validate prediction structure identity."
            ) from exc
        expected_canonical: dict[str, str] = {}
        for candidate in candidates:
            molecule = Chem.MolFromSmiles(candidate["smiles"])
            if molecule is None:  # already rejected by _normalise_candidates
                raise NMRForwardProtocolError(
                    "A normalized candidate became invalid before response validation."
                )
            expected_canonical[candidate["candidate_id"]] = Chem.MolToSmiles(
                molecule,
                canonical=True,
                isomericSmiles=True,
            )

        by_id: dict[str, dict[str, Any]] = {}
        for prediction in value:
            if not isinstance(prediction, dict):
                raise NMRForwardProtocolError("Each DP5q prediction must be an object.")
            _strict_keys(
                prediction,
                {
                    "candidate_id",
                    "canonical_smiles",
                    "conformer_count",
                    "atom_predictions",
                    "warnings",
                },
                context="candidate prediction",
            )
            candidate_id = prediction["candidate_id"]
            if (
                not isinstance(candidate_id, str)
                or candidate_id not in expected_carbon_indices
                or candidate_id in by_id
            ):
                raise NMRForwardProtocolError(
                    "DP5q sidecar returned a duplicate or unknown candidate_id."
                )
            if (
                not isinstance(prediction["canonical_smiles"], str)
                or prediction["canonical_smiles"]
                != expected_canonical[candidate_id]
            ):
                raise NMRForwardProtocolError(
                    "DP5q canonical_smiles does not match the requested structure."
                )
            conformer_count = prediction["conformer_count"]
            if type(conformer_count) is not int or not 1 <= conformer_count <= 20:
                raise NMRForwardProtocolError(
                    "DP5q conformer_count is outside the fixed contract."
                )
            warnings = prediction["warnings"]
            if not isinstance(warnings, list) or any(
                not isinstance(warning, str) for warning in warnings
            ):
                raise NMRForwardProtocolError(
                    "DP5q warnings must be a list of strings."
                )
            atom_predictions = prediction["atom_predictions"]
            if not isinstance(atom_predictions, list):
                raise NMRForwardProtocolError("DP5q atom_predictions must be a list.")
            seen_atoms: set[int] = set()
            clean_atoms: list[dict[str, float | int]] = []
            for atom_prediction in atom_predictions:
                if not isinstance(atom_prediction, dict):
                    raise NMRForwardProtocolError(
                        "DP5q atom prediction must be an object."
                    )
                _strict_keys(
                    atom_prediction,
                    {"atom_index", "shift_ppm"},
                    context="atom prediction",
                )
                atom_index = atom_prediction["atom_index"]
                shift = atom_prediction["shift_ppm"]
                if (
                    type(atom_index) is not int
                    or atom_index < 0
                    or atom_index in seen_atoms
                ):
                    raise NMRForwardProtocolError(
                        "DP5q atom indices must be unique non-negative integers."
                    )
                if (
                    isinstance(shift, bool)
                    or not isinstance(shift, (int, float))
                    or not math.isfinite(float(shift))
                    or not -20.0 <= float(shift) <= 300.0
                ):
                    raise NMRForwardProtocolError(
                        "DP5q shift must be finite and within the 13C range."
                    )
                seen_atoms.add(atom_index)
                clean_atoms.append(
                    {"atom_index": atom_index, "shift_ppm": float(shift)}
                )
            if seen_atoms != expected_carbon_indices[candidate_id]:
                raise NMRForwardProtocolError(
                    f"DP5q atom set mismatch for candidate {candidate_id}."
                )
            by_id[candidate_id] = {
                "candidate_id": candidate_id,
                "canonical_smiles": prediction["canonical_smiles"],
                "conformer_count": conformer_count,
                "atom_predictions": sorted(
                    clean_atoms, key=lambda item: int(item["atom_index"])
                ),
                "warnings": warnings,
            }
        expected_ids = [candidate["candidate_id"] for candidate in candidates]
        if set(by_id) != set(expected_ids):
            raise NMRForwardProtocolError(
                "DP5q sidecar omitted one or more candidates."
            )
        return [by_id[candidate_id] for candidate_id in expected_ids]

    def predict_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        formula: str | None = None,
    ) -> dict[str, Any]:
        """Predict Boltzmann-weighted 13C shifts for validated candidates."""

        clean_candidates, carbon_indices = self._normalise_candidates(
            candidates, formula=formula
        )
        response = self._request(clean_candidates)
        try:
            predictions = self._validate_predictions(
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
            "output": "boltzmann_weighted_mean_shift_ppm",
            "evidence_kind": "relative_13c_forward_evidence",
            "calibrated_probability": False,
            "quantile_enabled": False,
            "model": {
                "name": self._handshake["model"]["name"],
                "repository_commit": self._handshake["repository_commit"],
                **self._handshake["assets"],
            },
            "runtime": self._response_runtime(),
            "predictions": predictions,
        }

    def _response_runtime(self) -> dict[str, Any]:
        """Return the complete source/runtime attestation for one response."""

        assert self._handshake is not None
        return {
            **self._handshake["runtime"],
            "protocol_version": PROTOCOL_VERSION,
            "conformer_generation": self._handshake["conformer_generation"],
            "conformer_preflight": self._handshake["conformer_preflight"],
        }

    @staticmethod
    def _validate_preflight_results(
        value: Any,
        candidates: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        if not isinstance(value, list) or len(value) != len(candidates):
            raise NMRForwardProtocolError(
                "DP5q sidecar returned the wrong number of preflight results."
            )
        expected = {
            candidate["candidate_id"]: candidate["smiles"] for candidate in candidates
        }
        by_id: dict[str, dict[str, Any]] = {}
        try:
            from rdkit import Chem
        except ImportError as exc:  # pragma: no cover
            raise NMRForwardConfigurationError(
                "RDKit is required to validate conformer preflight."
            ) from exc
        for result in value:
            if not isinstance(result, dict):
                raise NMRForwardProtocolError(
                    "Each conformer-preflight result must be an object."
                )
            _strict_keys(
                result,
                {
                    "candidate_id",
                    "status",
                    "reason_code",
                    "canonical_smiles",
                    "conformer_count",
                    "warnings",
                },
                context="conformer-preflight result",
            )
            candidate_id = result["candidate_id"]
            if (
                not isinstance(candidate_id, str)
                or candidate_id not in expected
                or candidate_id in by_id
            ):
                raise NMRForwardProtocolError(
                    "Conformer preflight returned an unknown or duplicate candidate_id."
                )
            warnings = result["warnings"]
            if not isinstance(warnings, list) or any(
                not isinstance(warning, str) for warning in warnings
            ):
                raise NMRForwardProtocolError(
                    "Conformer-preflight warnings must be strings."
                )
            status = result["status"]
            reason_code = result["reason_code"]
            canonical_smiles = result["canonical_smiles"]
            conformer_count = result["conformer_count"]
            if status == "passed":
                molecule = Chem.MolFromSmiles(expected[candidate_id])
                assert molecule is not None
                expected_canonical = Chem.MolToSmiles(
                    molecule,
                    canonical=True,
                    isomericSmiles=True,
                )
                if (
                    reason_code is not None
                    or canonical_smiles != expected_canonical
                    or type(conformer_count) is not int
                    or not 1 <= conformer_count <= 20
                ):
                    raise NMRForwardProtocolError(
                        "Conformer preflight returned an invalid pass result."
                    )
            elif status == "rejected":
                if (
                    reason_code not in _PREFLIGHT_REJECTION_CODES
                    or canonical_smiles is not None
                    or conformer_count != 0
                    or warnings
                ):
                    raise NMRForwardProtocolError(
                        "Conformer preflight returned an invalid rejection."
                    )
            else:
                raise NMRForwardProtocolError(
                    "Conformer preflight returned an invalid status."
                )
            by_id[candidate_id] = {
                "candidate_id": candidate_id,
                "status": status,
                "reason_code": reason_code,
                "canonical_smiles": canonical_smiles,
                "conformer_count": conformer_count,
                "warnings": list(warnings),
            }
        return [by_id[candidate["candidate_id"]] for candidate in candidates]

    def preflight_candidates(
        self,
        candidates: Sequence[Mapping[str, Any]],
        *,
        formula: str | None = None,
    ) -> dict[str, Any]:
        """Run deterministic conformer applicability without model predictions."""

        clean_candidates, _ = self._normalise_candidates(candidates, formula=formula)
        response = self._request(
            clean_candidates,
            operation="conformer_preflight",
        )
        try:
            results = self._validate_preflight_results(
                response["results"],
                clean_candidates,
            )
        except Exception:
            with self._lock:
                self._abort()
            raise
        assert self._handshake is not None
        return {
            "status": "ok",
            "model_outputs_created": False,
            "protocol_version": PROTOCOL_VERSION,
            "runtime": self._response_runtime(),
            "results": results,
        }

    @staticmethod
    def _normalise_observed_shifts(
        observed_13c: Iterable[float | Mapping[str, Any]],
    ) -> list[float]:
        if isinstance(observed_13c, (str, bytes)):
            raise NMRForwardInputError("observed_13c must be an iterable of shifts.")
        shifts: list[float] = []
        for position, item in enumerate(observed_13c):
            raw = item.get("shift") if isinstance(item, Mapping) else item
            if (
                isinstance(raw, bool)
                or not isinstance(raw, (int, float))
                or not math.isfinite(float(raw))
                or not -20.0 <= float(raw) <= 300.0
            ):
                raise NMRForwardInputError(
                    f"observed_13c[{position}] is not a finite 13C shift."
                )
            shifts.append(float(raw))
        if len(shifts) > 256:
            raise NMRForwardInputError(
                "At most 256 observed 13C resonances are supported."
            )
        return sorted(shifts)

    def score_candidates(
        self,
        observed_13c: Iterable[float | Mapping[str, Any]] | None,
        candidates: Sequence[Mapping[str, Any]],
        *,
        formula: str | None = None,
    ) -> dict[str, Any]:
        """Return relative, uncalibrated forward-fit evidence.

        Empty 13C input deliberately does not start the model.  In particular,
        a 1H-only JDF must remain unsupported rather than receiving a fabricated
        13C or probability score.
        """

        shifts = (
            self._normalise_observed_shifts(observed_13c)
            if observed_13c is not None
            else []
        )
        if not shifts:
            return {
                "status": "unsupported_modality",
                "required_nucleus": "13C",
                "reason": "dp5q_mean_requires_observed_13c_resonances",
                "model_called": False,
                "calibrated_probability": False,
                "quantile_enabled": False,
                "candidates": [],
            }

        prediction_result = self.predict_candidates(candidates, formula=formula)
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError as exc:  # pragma: no cover - required backend dependency
            raise NMRForwardConfigurationError(
                "SciPy is required for one-to-one 13C forward matching."
            ) from exc

        evidence: list[dict[str, Any]] = []
        for prediction in prediction_result["predictions"]:
            predicted = [
                float(atom["shift_ppm"]) for atom in prediction["atom_predictions"]
            ]
            costs = [
                [abs(observed - expected) for expected in predicted]
                for observed in shifts
            ]
            rows, columns = linear_sum_assignment(costs)
            errors = [
                costs[int(row)][int(column)] for row, column in zip(rows, columns)
            ]
            mae = sum(errors) / len(errors)
            rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
            observed_coverage = len(errors) / len(shifts)
            predicted_coverage = len(errors) / len(predicted)
            unmatched_observed = len(shifts) - len(errors)
            unmatched_predicted = len(predicted) - len(errors)
            evidence.append(
                {
                    "candidate_id": prediction["candidate_id"],
                    "relative_rank": 0,
                    "assignment_mode": "unassigned_hungarian_atom_level",
                    "diagnostic_only": True,
                    "matched_count": len(errors),
                    "observed_count": len(shifts),
                    "predicted_atom_count": len(predicted),
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
                    "mae_ppm": float(mae),
                    "rmse_ppm": float(rmse),
                    "max_abs_error_ppm": float(max(errors)),
                    "calibrated_probability": False,
                    "quantile_enabled": False,
                    "prediction": prediction,
                }
            )
        ordered = sorted(
            evidence,
            key=lambda item: (
                -float(item["bidirectional_coverage"]),
                int(item["unmatched_observed_count"])
                + int(item["unmatched_predicted_atom_count"]),
                float(item["mae_ppm"]),
                float(item["rmse_ppm"]),
                str(item["candidate_id"]),
            ),
        )
        for rank, item in enumerate(ordered, start=1):
            item["relative_rank"] = rank
        return {
            "status": "ok",
            "nucleus": "13C",
            "evidence_kind": "relative_13c_forward_evidence",
            "diagnostic_only": True,
            "rank_basis": (
                "higher_bidirectional_coverage_then_fewer_unmatched_then_"
                "lower_hungarian_mae_rmse"
            ),
            "assignment_limitation": (
                "Predicted atom shifts are matched to experimental resonance "
                "groups without modelling symmetry-equivalent signal collapse."
            ),
            "observed_shifts_ppm": shifts,
            "calibrated_probability": False,
            "quantile_enabled": False,
            "model": prediction_result["model"],
            "runtime": prediction_result["runtime"],
            "candidates": ordered,
        }


_default_adapter: NMRForwardAdapter | None = None
_default_adapter_lock = threading.Lock()


def get_nmr_forward_adapter() -> NMRForwardAdapter:
    """Return the lazy process-wide adapter configured by environment."""

    global _default_adapter
    with _default_adapter_lock:
        if _default_adapter is None:
            _default_adapter = NMRForwardAdapter(NMRForwardConfig.from_environment())
        return _default_adapter


def _close_default_adapter() -> None:
    global _default_adapter
    with _default_adapter_lock:
        if _default_adapter is not None:
            _default_adapter.close()
            _default_adapter = None


atexit.register(_close_default_adapter)


__all__ = [
    "DP5Q_MEAN_MODEL_SHA256",
    "DP5Q_PREPROCESSOR_SHA256",
    "DP5Q_REPOSITORY_COMMIT",
    "DP5Q_CONFORMER_PROTOCOL_VERSION",
    "NMRForwardAdapter",
    "NMRForwardConfig",
    "NMRForwardConfigurationError",
    "NMRForwardError",
    "NMRForwardInputError",
    "NMRForwardProtocolError",
    "NMRForwardTimeoutError",
    "NMRForwardUnavailableError",
    "dp5q_sidecar_is_configured",
    "dp5q_conformer_policy",
    "get_nmr_forward_adapter",
]
