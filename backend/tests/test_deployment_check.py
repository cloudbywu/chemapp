"""Tests for the CSP5 deployment startup verification."""

from __future__ import annotations

import hashlib
import json

import pytest

from app.ml import deployment_check


def _write_manifest(tmp_path, *, content: bytes) -> None:
    digest = hashlib.sha256(content).hexdigest()
    manifest = {
        "schema_version": "chemapp.csp5.weights-manifest.v1",
        "files": [
            {
                "path": "models/CSP5q-13C/best_model.pt",
                "sha256": digest,
                "bytes": len(content),
            }
        ],
    }
    (tmp_path / "weights-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    weight = tmp_path / "models" / "CSP5q-13C"
    weight.mkdir(parents=True)
    (weight / "best_model.pt").write_bytes(content)


def test_verify_csp5_weights_passes_on_match(tmp_path) -> None:
    _write_manifest(tmp_path, content=b"fake-weights")
    result = deployment_check.verify_csp5_weights(
        tmp_path / "weights-manifest.json"
    )
    assert result["status"] == "ok"
    assert result["verified_files"][0]["sha256"] == hashlib.sha256(
        b"fake-weights"
    ).hexdigest()


def test_verify_csp5_weights_fails_on_missing_file(tmp_path) -> None:
    _write_manifest(tmp_path, content=b"fake-weights")
    (tmp_path / "models" / "CSP5q-13C" / "best_model.pt").unlink()
    with pytest.raises(RuntimeError, match="weight file missing"):
        deployment_check.verify_csp5_weights(
            tmp_path / "weights-manifest.json"
        )


def test_verify_csp5_weights_fails_on_hash_mismatch(tmp_path) -> None:
    _write_manifest(tmp_path, content=b"fake-weights")
    (tmp_path / "models" / "CSP5q-13C" / "best_model.pt").write_bytes(
        b"tampered"
    )
    with pytest.raises(RuntimeError, match="hash mismatch"):
        deployment_check.verify_csp5_weights(
            tmp_path / "weights-manifest.json"
        )


def test_verify_deployment_modes(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_manifest(tmp_path, content=b"fake-weights")
    monkeypatch.setattr(deployment_check, "_MANIFEST_PATH", tmp_path / "weights-manifest.json")

    monkeypatch.setenv("CHEMAPP_CSP5_MODE", "off")
    assert deployment_check.verify_deployment()["status"] == "disabled"

    monkeypatch.setenv("CHEMAPP_CSP5_MODE", "on")
    assert deployment_check.verify_deployment()["status"] == "ok"

    (tmp_path / "models" / "CSP5q-13C" / "best_model.pt").write_bytes(b"bad")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        deployment_check.verify_deployment()

    monkeypatch.setenv("CHEMAPP_CSP5_MODE", "auto")
    result = deployment_check.verify_deployment()
    assert result["status"] == "fallback"
    assert "hash mismatch" in result["reason"]


def test_normalise_weight_rel() -> None:
    assert (
        deployment_check._normalise_weight_rel(r"models\CSP5q-13C\best_model.pt")
        == "models/CSP5q-13C/best_model.pt"
    )
    assert (
        deployment_check._normalise_weight_rel("models/CSP5q-13C/best_model.pt")
        == "models/CSP5q-13C/best_model.pt"
    )
    assert deployment_check._normalise_weight_rel("") == ""


def test_verify_csp5_weights_normalises_windows_manifest_paths(tmp_path) -> None:
    content = b"fake-weights"
    manifest = {
        "schema_version": "chemapp.csp5.weights-manifest.v1",
        "files": [
            {
                "path": r"models\CSP5q-13C\best_model.pt",
                "sha256": hashlib.sha256(content).hexdigest(),
                "bytes": len(content),
            }
        ],
    }
    (tmp_path / "weights-manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    weight = tmp_path / "models" / "CSP5q-13C"
    weight.mkdir(parents=True)
    (weight / "best_model.pt").write_bytes(content)

    result = deployment_check.verify_csp5_weights(
        tmp_path / "weights-manifest.json"
    )

    assert result["status"] == "ok"
    assert (
        result["verified_files"][0]["path"]
        == "models/CSP5q-13C/best_model.pt"
    )
