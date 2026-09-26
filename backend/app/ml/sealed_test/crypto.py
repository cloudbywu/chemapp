"""Ed25519 helpers for the sealed-test state machine.

Reuses the project canonical-JSON convention from ``app.ml.nmr_blind_challenge``
so every hash and signature is byte-compatible with the existing v6/v7
receipt machinery.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from app.ml.nmr_blind_challenge import canonical_json_bytes, sha256_bytes


class CryptoError(ValueError):
    """Raised when key material or a signature is invalid."""


def canonical_object_bytes(value: Any) -> bytes:
    """Return the project canonical JSON bytes for ``value``."""
    try:
        return canonical_json_bytes(value)
    except Exception as exc:  # NMRBlindChallengeError subclasses ValueError
        raise CryptoError(f"value is not canonical JSON: {exc}") from exc


def object_sha256(value: Any) -> str:
    return sha256_bytes(canonical_object_bytes(value))


def payload_sha256(payload: bytes) -> str:
    return sha256_bytes(payload)


def _require_crypto() -> dict[str, Any]:
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import (
            Ed25519PrivateKey,
            Ed25519PublicKey,
        )
    except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover
        raise CryptoError("Ed25519 support requires cryptography>=44.0") from exc
    return {
        "serialization": serialization,
        "private_type": Ed25519PrivateKey,
        "public_type": Ed25519PublicKey,
    }


def generate_private_key_pem() -> bytes:
    crypto = _require_crypto()
    key = crypto["private_type"].generate()
    return key.private_bytes(
        encoding=crypto["serialization"].Encoding.PEM,
        format=crypto["serialization"].PrivateFormat.PKCS8,
        encryption_algorithm=crypto["serialization"].NoEncryption(),
    )


def public_key_from_private_pem(pem: bytes) -> Any:
    crypto = _require_crypto()
    try:
        key = crypto["serialization"].load_pem_private_key(pem, password=None)
    except (TypeError, ValueError) as exc:
        raise CryptoError("private key must be an unencrypted Ed25519 PEM key") from exc
    if not isinstance(key, crypto["private_type"]):
        raise CryptoError("private key is not Ed25519")
    return key.public_key()


def public_key_pem(public_key: Any) -> bytes:
    crypto = _require_crypto()
    return public_key.public_bytes(
        encoding=crypto["serialization"].Encoding.PEM,
        format=crypto["serialization"].PublicFormat.SubjectPublicKeyInfo,
    )


def public_spki_sha256(public_key: Any) -> str:
    crypto = _require_crypto()
    der = public_key.public_bytes(
        encoding=crypto["serialization"].Encoding.DER,
        format=crypto["serialization"].PublicFormat.SubjectPublicKeyInfo,
    )
    return sha256_bytes(der)


def public_spki_base64(public_key: Any) -> str:
    crypto = _require_crypto()
    der = public_key.public_bytes(
        encoding=crypto["serialization"].Encoding.DER,
        format=crypto["serialization"].PublicFormat.SubjectPublicKeyInfo,
    )
    return base64.b64encode(der).decode("ascii")


def load_public_key(path: str | Path) -> Any:
    crypto = _require_crypto()
    key_path = Path(path)
    try:
        key = crypto["serialization"].load_pem_public_key(key_path.read_bytes())
    except (TypeError, ValueError, OSError) as exc:
        raise CryptoError("public key must be an Ed25519 PEM file") from exc
    if not isinstance(key, crypto["public_type"]):
        raise CryptoError("public key is not Ed25519")
    return key


def load_private_key(path: str | Path) -> Any:
    crypto = _require_crypto()
    key_path = Path(path)
    try:
        key = crypto["serialization"].load_pem_private_key(
            key_path.read_bytes(), password=None
        )
    except (TypeError, ValueError, OSError) as exc:
        raise CryptoError("private key must be an unencrypted Ed25519 PEM file") from exc
    if not isinstance(key, crypto["private_type"]):
        raise CryptoError("private key is not Ed25519")
    return key


def sign_object(value: Any, private_key: Any) -> str:
    payload = canonical_object_bytes(value)
    return base64.b64encode(private_key.sign(payload)).decode("ascii")


def verify_object(value: Any, signature_b64: str, public_key: Any) -> bool:
    payload = canonical_object_bytes(value)
    try:
        signature = base64.b64decode(signature_b64)
    except (ValueError, TypeError) as exc:
        raise CryptoError("signature must be base64") from exc
    try:
        public_key.verify(signature, payload)
    except Exception as exc:
        raise CryptoError("signature verification failed") from exc
    return True
