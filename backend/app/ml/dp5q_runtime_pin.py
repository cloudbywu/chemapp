"""Exact executable-source and runtime pins for the DP5q quantile sidecar."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Final


DP5Q_PYTHON_VERSION: Final = "3.11.15"
DP5Q_TENSORFLOW_VERSION: Final = "2.14.0"
DP5Q_KERAS_VERSION: Final = "2.14.0"
DP5Q_NUMPY_VERSION: Final = "1.26.4"
DP5Q_PANDAS_VERSION: Final = "2.2.3"
DP5Q_SCIPY_VERSION: Final = "1.11.4"
DP5Q_RDKIT_VERSION: Final = "2026.03.4"
DP5Q_SCIKIT_LEARN_VERSION: Final = "1.3.2"
DP5Q_TQDM_VERSION: Final = "4.67.3"
DP5Q_UPSTREAM_SOURCE_BUNDLE_SHA256: Final = (
    "1859d45b52b7b8aa158acad849a543edfe6b89151ffa4d3d5948b8700687a7c5"
)

# path -> (exact byte count, SHA-256), all obtained from raw.githubusercontent
# at b79968cf63cb282e8871d5595ea6cef5b4dc0d49.
DP5Q_UPSTREAM_SOURCE_FILES: Final[dict[str, tuple[int, str]]] = {
    "dp5/__init__.py": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "dp5/neural_net/__init__.py": (
        0,
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    ),
    "dp5/neural_net/CNN_model.py": (
        20517,
        "44d3ecc3bef9146a6bf6f4f9e3d9e198c3f57b365b5dfdcd5b5a42454d280a6a",
    ),
    "dp5/neural_net/nn_utils.py": (
        1028,
        "2adc368c4bd9f6717260e5188e1922c53853cc6b3d463f5b26ee723305cf92aa",
    ),
    "dp5/neural_net/nfp/__init__.py": (
        460,
        "3e790112df128f6d3ce891156c559ac1c83675acb2eac80383602c7a688799b9",
    ),
    "dp5/neural_net/nfp/layers/__init__.py": (
        67,
        "c02366e224c91a651c3d78fde3456bc187706cbafccdf1aa4c30746e76fea86f",
    ),
    "dp5/neural_net/nfp/layers/layers.py": (
        15890,
        "adad099220a15d424148858853e8be70a6b75b05963780fcc82b9889e1e3250c",
    ),
    "dp5/neural_net/nfp/layers/utils.py": (
        2727,
        "2b10b9e81714fefc071be29517af6244d6607cc3589a25d39aa80161e5140118",
    ),
    "dp5/neural_net/nfp/layers/wrappers.py": (
        1352,
        "e7f570f8da935fce33bc75941aa2be6d74ef3d620e573bf849af39885310f226",
    ),
    "dp5/neural_net/nfp/models/__init__.py": (
        44,
        "9193de7764d9ba44a3e90b89d1cac659dc90673ab513263196462448c794c963",
    ),
    "dp5/neural_net/nfp/models/losses.py": (
        763,
        "3f7f5826f226efcc85117b9bb3fc6e2e4895ab7ba112fe333553bac927d7205c",
    ),
    "dp5/neural_net/nfp/models/models.py": (
        430,
        "680b7ffe6f3f24362125f84c7acb4d239f0c4696a298f380036ff170be7b52f8",
    ),
    "dp5/neural_net/nfp/preprocessing/__init__.py": (
        99,
        "fa4927de64990e75428b530bf664a8a67b1602afd2db27582ae661c39328be42",
    ),
    "dp5/neural_net/nfp/preprocessing/features.py": (
        3519,
        "87895eb5f88323f10b87a9e1c475b2cc0fb978a9caa5435660412e132f9fac7a",
    ),
    "dp5/neural_net/nfp/preprocessing/preprocessor.py": (
        25809,
        "72cd5b076f3972bf4b43568498e294eea39b2a095fd24dc8ecfb1502ff08d7cc",
    ),
    "dp5/neural_net/nfp/preprocessing/scaling.py": (
        1550,
        "a225b4483faad41f0bfbdca3f58a5bdf58b9c1a0c195699432320ef16960be24",
    ),
    "dp5/neural_net/nfp/preprocessing/sequence.py": (
        4318,
        "c9d4cc02648bc696aba3e91c9075a36ab67277b4ef6617ab794205b2c1badabe",
    ),
}


class DP5qRuntimePinError(RuntimeError):
    """The executable upstream source tree differs from its exact pin."""


def source_bundle_sha256(
    source_files: dict[str, bytes],
) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(source_files):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(source_files[relative_path]).digest())
    return digest.hexdigest()


def verified_source_bytes(repository: Path) -> dict[str, bytes]:
    """Read and verify every executable upstream Python source exactly once."""

    root = repository.expanduser().resolve(strict=True)
    payloads: dict[str, bytes] = {}
    for relative_path, (expected_size, expected_sha256) in (
        DP5Q_UPSTREAM_SOURCE_FILES.items()
    ):
        path = root.joinpath(*PurePosixPath(relative_path).parts)
        if path.is_symlink() or not path.is_file():
            raise DP5qRuntimePinError(
                f"Pinned DP5q source is missing or unsafe: {relative_path}"
            )
        try:
            path.resolve(strict=True).relative_to(root)
        except ValueError as exc:
            raise DP5qRuntimePinError(
                f"Pinned DP5q source escapes the repository: {relative_path}"
            ) from exc
        if path.stat().st_size != expected_size:
            raise DP5qRuntimePinError(
                f"Pinned DP5q source length mismatch: {relative_path}"
            )
        payload = path.read_bytes()
        if (
            len(payload) != expected_size
            or hashlib.sha256(payload).hexdigest() != expected_sha256
        ):
            raise DP5qRuntimePinError(
                f"Pinned DP5q source hash mismatch: {relative_path}"
            )
        payloads[relative_path] = payload
    if source_bundle_sha256(payloads) != DP5Q_UPSTREAM_SOURCE_BUNDLE_SHA256:
        raise DP5qRuntimePinError("Pinned DP5q source-bundle hash mismatch.")
    return payloads


__all__ = [
    "DP5Q_NUMPY_VERSION",
    "DP5Q_PANDAS_VERSION",
    "DP5Q_PYTHON_VERSION",
    "DP5Q_TENSORFLOW_VERSION",
    "DP5Q_KERAS_VERSION",
    "DP5Q_RDKIT_VERSION",
    "DP5Q_SCIPY_VERSION",
    "DP5Q_SCIKIT_LEARN_VERSION",
    "DP5Q_TQDM_VERSION",
    "DP5Q_UPSTREAM_SOURCE_BUNDLE_SHA256",
    "DP5Q_UPSTREAM_SOURCE_FILES",
    "DP5qRuntimePinError",
    "source_bundle_sha256",
    "verified_source_bytes",
]
