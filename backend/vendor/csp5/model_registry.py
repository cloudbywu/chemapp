"""Model registry and on-demand resolver for CSP5 checkpoints."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen


DEFAULT_ZENODO_RECORD_ID = "19486118"
ENV_ZENODO_RECORD_ID = "CSP5_ZENODO_RECORD_ID"
ENV_ZENODO_FILE_PREFIX = "CSP5_ZENODO_FILE_PREFIX"
ENV_MODEL_CACHE_DIR = "CSP5_MODEL_CACHE_DIR"
CARBON_QUANTILE_OUTPUT_SCALE = 0.7371728137296569
PROTON_QUANTILE_OUTPUT_SCALE = 2.926568208518941


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    model_name: str
    nucleus: str
    weights_path: Path
    remote_filename: str
    sha256: str
    output_scale: float
    output_bias: float
    output_dim: int = 1
    output_quantile_scale: float = 1.0
    graph_n_neighbors: int = 20
    graph_cutoff_angstrom: float = 5.0


_BASE = Path(__file__).resolve().parent
_MODELS = _BASE / "models"

# model_name, nucleus, solvent, sha256, output_dim
_MODEL_ROWS = [
    ("CASCADE-13C", "13C", "", "692d8339147cb85b997f9645db479e0eb9d013b0a0a6e53c1272261fc0cc8011", 1),
    ("CSP5-13C", "13C", "", "d3123e1a34fa98e1d485ca571d5ac0b403f60eb2ac9d890e3d31357e235e7422", 1),
    ("CSP5q-13C", "13C", "", "7664e6ddc1fb5848f14f842a41078df306f6e3b0972b80e134511e0c6033ee78", 99),
    ("CSP5-13C-c6d6", "13C", "c6d6", "7d7c99abd8f3d2c9e8e70830f496fd39ca04b6c88668dc33a267068d502ce946", 1),
    ("CSP5-13C-cd2cl2", "13C", "cd2cl2", "9a443802ba0fac07d9c1798cfbfa9f0e27446b45e35b5a8016ba397c20156573", 1),
    ("CSP5-13C-cd3cn", "13C", "cd3cn", "bb03ea052595c0abebb828953cd21be790e14576d89a4398008124b58d766dc2", 1),
    ("CSP5-13C-cd3cocd3", "13C", "cd3cocd3", "00cfe4cc56f5b1d39d6066ebed82fe39b15d6d4ff3fc904cea4902ce7d2258e6", 1),
    ("CSP5-13C-cd3od", "13C", "cd3od", "6bd7c1f0bf7cb5ff9a46f743d3f7ec56b723d2db90b1351d228a174ded7ae396", 1),
    ("CSP5-13C-d2o", "13C", "d2o", "ffd100ab55eb395403d9d18c63a086363c9cf3ce0deceb8b4546e7cd9dfaf518", 1),
    ("CSP5-13C-dmso-d6", "13C", "dmso-d6", "0d28cfe5bb4187a901364dcd54babff7e74b32116f413a094fc15a29e8887cf1", 1),
    ("CSP5-13C-not_known", "13C", "not_known", "351340f553e5759a126376314d6729a9e6b870fc6ed9bd6e0d1588c94c3039ae", 1),
    ("CASCADE-1H", "1H", "", "ce5f76784f43243c34129e1c6a7a93577c089357069eddc4a6f8b982958fe591", 1),
    ("CSP5-1H", "1H", "", "f099f8fff0434edb2b86b437fdf49314ddd700a988c214e73c563572ba17272c", 1),
    ("CSP5q-1H", "1H", "", "ddacc9b98ca71bd55820062f7f272c6ec1319ccfb35851de42f8878a31eed02c", 99),
    ("CSP5-1H-c2d2cl4", "1H", "c2d2cl4", "dc2bd673c10e3fe2bc5ea9dc7afd69a57afd8f44e95a64479c5e418a653149ab", 1),
    ("CSP5-1H-c6d6", "1H", "c6d6", "1aef07904327f62f9fc04ade0a7afbabd76196674d68e8ae2b1af032b026730a", 1),
    ("CSP5-1H-cd2cl2", "1H", "cd2cl2", "561906f3bcdbb98d699388e9e8a73163b18203e902b61c00c12b8cb85cf3a28b", 1),
    ("CSP5-1H-cd3cn", "1H", "cd3cn", "52997817c920ab3db2090d9d8774f6f8fd7e1004f374a3547b5b61fcfc3705d9", 1),
    ("CSP5-1H-cd3cocd3", "1H", "cd3cocd3", "a255cb198c46fbfee39cb6facdd64eed51747ae50cab1d6f7c713a0d42593cbf", 1),
    ("CSP5-1H-cd3od", "1H", "cd3od", "967e78aba70afe78ff7be7a4c908b8e9150402ce4d12c3fd914de30e3577545d", 1),
    ("CSP5-1H-cf3co2d", "1H", "cf3co2d", "238e79803760a7a3b5d57f6b7eaa7e7a03a63dc333d5febb1ab5c79f194e053c", 1),
    ("CSP5-1H-d2o", "1H", "d2o", "80c9e7768bb2e45dcdc5683203597ede950a4bc7aac059e560c6f9aeb5b77abd", 1),
    ("CSP5-1H-dmf-d7", "1H", "dmf-d7", "f947cb5c3e821d281af2719879d1551a33f6f096cdde991db2ed5de07c43074b", 1),
    ("CSP5-1H-dmso-d6", "1H", "dmso-d6", "ceca04380db56d151891995cb532ad272d5dfe0741f11d1d8695f87db10450dc", 1),
    ("CSP5-1H-mixed", "1H", "mixed", "89ff53be1cc70250b063f4abb9364acc03de51b63e7ca4281f7343a3c2f78ee0", 1),
    ("CSP5-1H-not_known", "1H", "not_known", "44a385bede8d51178c4394760aa7debb35f5a6052dc1f703340d633acdf3135c", 1),
    ("CSP5-1H-pyridine-d5", "1H", "pyridine-d5", "922d17c9f14673de03bdd92efe653eb3be847c370eecfcf1e249ec64644677bc", 1),
    ("CSP5-1H-thf-d8", "1H", "thf-d8", "8055185d4ff25adc2f7b87636ae54bcf7da1503cb516c0697317f3b76f6ac8d6", 1),
]

_DEFAULT_MODEL_BY_NUCLEUS = {
    "13C": "CSP5-13C",
    "1H": "CSP5-1H",
}

_MODEL_SPECS_BY_NAME: Dict[str, ModelSpec] = {}
_MODEL_SPECS_BY_ID: Dict[str, ModelSpec] = {}
_SOLVENT_LOOKUP: Dict[Tuple[str, str], str] = {}

for model_name, nucleus, solvent, sha256, output_dim in _MODEL_ROWS:
    model_id = model_name.lower()
    spec = ModelSpec(
        model_id=model_id,
        model_name=model_name,
        nucleus=nucleus,
        weights_path=_MODELS / model_name / "best_model.pt",
        remote_filename=f"{model_name}.pt",
        sha256=sha256,
        output_scale=1.0,
        output_bias=0.0,
        output_dim=int(output_dim),
        output_quantile_scale=(
            CARBON_QUANTILE_OUTPUT_SCALE
            if model_name == "CSP5q-13C"
            else PROTON_QUANTILE_OUTPUT_SCALE
            if model_name == "CSP5q-1H"
            else 1.0
        ),
    )
    _MODEL_SPECS_BY_NAME[model_name] = spec
    _MODEL_SPECS_BY_ID[model_id] = spec
    if solvent:
        _SOLVENT_LOOKUP[(nucleus, solvent.lower())] = model_name

_MODEL_NAMES_BY_LOWER = {name.lower(): name for name in _MODEL_SPECS_BY_NAME}


def normalize_nucleus(value: str) -> str:
    key = str(value).strip().upper()
    aliases = {
        "13C": "13C",
        "C13": "13C",
        "C": "13C",
        "1H": "1H",
        "H1": "1H",
        "H": "1H",
        "PROTON": "1H",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported nucleus {value!r}. Use one of: 13C, 1H")
    return aliases[key]


def _canonical_model_name(value: str) -> str:
    key = str(value).strip().lower()
    canonical = _MODEL_NAMES_BY_LOWER.get(key)
    if canonical is None:
        raise ValueError(f"Unknown model_name {value!r}")
    return canonical


def _available_solvents(nucleus: str) -> List[str]:
    solvents = sorted({solvent for nuc, solvent in _SOLVENT_LOOKUP if nuc == nucleus})
    return solvents


def get_model_spec(
    nucleus: str,
    *,
    model_name: str | None = None,
    solvent: str | None = None,
) -> ModelSpec:
    normalized_nucleus = normalize_nucleus(nucleus)
    if model_name and solvent:
        raise ValueError("Use either model_name or solvent, not both")

    if model_name is not None:
        canonical_name = _canonical_model_name(model_name)
        spec = _MODEL_SPECS_BY_NAME[canonical_name]
        if spec.nucleus != normalized_nucleus:
            raise ValueError(
                f"Model {spec.model_name} is for {spec.nucleus}, not requested nucleus {normalized_nucleus}"
            )
        return spec

    if solvent is not None:
        solvent_key = str(solvent).strip().lower()
        if not solvent_key:
            raise ValueError("solvent cannot be empty")
        model_key = (normalized_nucleus, solvent_key)
        model_name_from_solvent = _SOLVENT_LOOKUP.get(model_key)
        if model_name_from_solvent is None:
            valid = ", ".join(_available_solvents(normalized_nucleus))
            raise ValueError(
                f"No clean model mapping for nucleus={normalized_nucleus}, solvent={solvent_key}. "
                f"Available solvents: {valid}"
            )
        return _MODEL_SPECS_BY_NAME[model_name_from_solvent]

    default_name = _DEFAULT_MODEL_BY_NUCLEUS[normalized_nucleus]
    return _MODEL_SPECS_BY_NAME[default_name]


def get_model_spec_by_id(model_id: str) -> ModelSpec:
    key = str(model_id).strip().lower()
    spec = _MODEL_SPECS_BY_ID.get(key)
    if spec is None:
        raise ValueError(f"Unknown model_id {model_id!r}")
    return spec


def _model_cache_root() -> Path:
    override = os.environ.get(ENV_MODEL_CACHE_DIR, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".cache" / "csp5" / "models"


def _zenodo_file_url(remote_filename: str) -> str:
    record_id = os.environ.get(ENV_ZENODO_RECORD_ID, DEFAULT_ZENODO_RECORD_ID).strip()
    if not record_id:
        raise ValueError(f"Environment variable {ENV_ZENODO_RECORD_ID} cannot be empty")

    prefix = os.environ.get(ENV_ZENODO_FILE_PREFIX, "").strip().strip("/")
    file_key = f"{prefix}/{remote_filename}" if prefix else remote_filename
    escaped_key = quote(file_key, safe="/")
    return f"https://zenodo.org/api/records/{record_id}/files/{escaped_key}/content"


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_to_path(url: str, destination: Path) -> None:
    try:
        with urlopen(url, timeout=120) as response, destination.open("wb") as handle:
            status = getattr(response, "status", None)
            if status is not None and int(status) >= 400:
                raise RuntimeError(f"HTTP status {status} for {url}")
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
    except HTTPError as exc:
        raise RuntimeError(f"Failed to download {url}: HTTP {exc.code} {exc.reason}") from exc
    except URLError as exc:
        raise RuntimeError(f"Failed to download {url}: {exc.reason}") from exc


def resolve_model_weights(spec: ModelSpec) -> Path:
    if spec.weights_path.exists():
        bundled_sha = _sha256_file(spec.weights_path)
        if bundled_sha != spec.sha256:
            raise ValueError(
                f"Checksum mismatch for bundled model {spec.weights_path}: "
                f"expected {spec.sha256}, got {bundled_sha}"
            )
        return spec.weights_path

    if not spec.remote_filename:
        raise FileNotFoundError(f"No remote filename configured for model {spec.model_name}")

    cache_path = _model_cache_root() / spec.remote_filename
    if cache_path.exists():
        cached_sha = _sha256_file(cache_path)
        if cached_sha != spec.sha256:
            raise ValueError(
                f"Checksum mismatch for cached model {cache_path}: expected {spec.sha256}, got {cached_sha}"
            )
        return cache_path

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(
        prefix=".csp5-download-",
        suffix=".tmp",
        dir=str(cache_path.parent),
    )
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)

    try:
        remote_url = _zenodo_file_url(spec.remote_filename)
        _download_to_path(remote_url, tmp_path)
        downloaded_sha = _sha256_file(tmp_path)
        if downloaded_sha != spec.sha256:
            raise ValueError(
                f"Checksum mismatch for downloaded model {spec.model_name} from {remote_url}: "
                f"expected {spec.sha256}, got {downloaded_sha}"
            )
        os.replace(tmp_path, cache_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    if not cache_path.exists():
        raise FileNotFoundError(f"Failed to materialize model weights at {cache_path}")
    return cache_path
