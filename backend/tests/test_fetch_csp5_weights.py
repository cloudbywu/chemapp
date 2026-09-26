from __future__ import annotations

import hashlib
import io
import json
import tarfile

import pytest

from scripts.fetch_csp5_weights import (
    SDIST_TOP,
    MODELS_REL,
    _extract_sdist,
    fetch,
    verify_weights,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(entries: list[tuple[str, bytes]]) -> dict:
    return {
        "schema_version": "chemapp.csp5.weights-manifest.v1",
        "files": [
            {
                "path": rel,
                "bytes": len(data),
                "sha256": _sha256_bytes(data),
            }
            for rel, data in entries
        ],
    }


def _write_target(
    tmp_path,
    entries: list[tuple[str, bytes]],
) -> tuple[dict, list[tuple[str, bytes]]]:
    target = tmp_path / "vendor" / "csp5"
    target.mkdir(parents=True)
    for rel, data in entries:
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    manifest = _manifest(entries)
    (target / "weights-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return manifest, entries


def test_verify_weights_reports_missing_and_mismatch(tmp_path) -> None:
    manifest, entries = _write_target(
        tmp_path,
        [("models/A/best_model.pt", b"a" * 16)],
    )
    target = tmp_path / "vendor" / "csp5"

    assert verify_weights(target, manifest) == []

    (target / "models" / "A" / "best_model.pt").write_bytes(b"b" * 16)
    problems = verify_weights(target, manifest)
    assert len(problems) == 1
    assert "hash mismatch" in problems[0]

    (target / "models" / "A" / "best_model.pt").unlink()
    assert verify_weights(target, manifest) == ["missing: models/A/best_model.pt"]


def test_extract_sdist_restores_weights_and_metadata(tmp_path) -> None:
    entries = [
        ("models/CSP5q-13C/best_model.pt", b"model-a"),
        ("models/CSP5q-1H/best_model.pt", b"model-b"),
    ]
    manifest = _manifest(entries)
    target = tmp_path / "vendor" / "csp5"
    target.mkdir(parents=True)
    (target / "weights-manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    source = tmp_path / "source" / SDIST_TOP / MODELS_REL
    for rel, data in entries:
        source_rel = rel.removeprefix("models/")
        path = source / source_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (source / "CSP5q-13C" / "model_metadata.json").write_text(
        '{"model": "A"}',
        encoding="utf-8",
    )
    sdist = tmp_path / "csp5.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for rel, _ in entries:
            archive.add(
                source / rel.removeprefix("models/"),
                arcname=(
                    f"{SDIST_TOP}/{MODELS_REL}/"
                    f"{rel.removeprefix('models/')}"
                ),
            )
        archive.add(
            source / "CSP5q-13C" / "model_metadata.json",
            arcname=(
                f"{SDIST_TOP}/{MODELS_REL}/CSP5q-13C/"
                "model_metadata.json"
            ),
        )

    _extract_sdist(sdist, target, manifest)

    assert verify_weights(target, manifest) == []
    assert (
        target / "models" / "CSP5q-13C" / "model_metadata.json"
    ).read_text(encoding="utf-8") == '{"model": "A"}'


def test_fetch_skips_download_when_weights_verify(
    tmp_path,
    monkeypatch,
) -> None:
    entries = [("models/A/best_model.pt", b"a" * 16)]
    _write_target(tmp_path, entries)
    target = tmp_path / "vendor" / "csp5"

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("network must not be used")

    monkeypatch.setattr("scripts.fetch_csp5_weights.urlopen", fail_if_called)

    fetch(target)

    assert (target / "models" / "A" / "best_model.pt").read_bytes() == b"a" * 16


def test_fetch_rejects_wrong_sdist_hash(tmp_path, monkeypatch) -> None:
    entries = [("models/A/best_model.pt", b"a" * 16)]
    _write_target(tmp_path, entries)
    target = tmp_path / "vendor" / "csp5"

    class _Response:
        def __enter__(self):
            return io.BytesIO(b"not the pinned archive")

        def __exit__(self, *_args):
            return False

    monkeypatch.setattr(
        "scripts.fetch_csp5_weights.urlopen",
        lambda *_args, **_kwargs: _Response(),
    )

    with pytest.raises(RuntimeError, match="source distribution hash mismatch"):
        fetch(
            target,
            url="https://example.invalid/csp5.tar.gz",
            force=True,
        )
