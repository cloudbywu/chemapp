from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

import download_sd
from app.ml.nmr_data_v2 import SourceValidationError


FILENAME = "nmrshiftdb2.sd"


class _FakeResponse:
    def __init__(self, body: bytes, content_type: str) -> None:
        self._body = body
        self.headers = {"Content-Type": content_type}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, chunk_size: int):
        for start in range(0, len(self._body), chunk_size):
            yield self._body[start : start + chunk_size]


def _install_response(
    monkeypatch: pytest.MonkeyPatch,
    *,
    body: bytes,
    content_type: str,
) -> list[tuple[str, dict[str, object]]]:
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setitem(
        download_sd.FILES,
        FILENAME,
        (0.0, "small test snapshot"),
    )

    def fake_get(url: str, **kwargs: object) -> _FakeResponse:
        calls.append((url, kwargs))
        return _FakeResponse(body, content_type)

    monkeypatch.setattr(download_sd.requests, "get", fake_get)
    return calls


def _valid_sd_bytes() -> bytes:
    record = b"download safety test\n  ChemApp\n\nM  END\n$$$$\n"
    return record + (b" " * (2048 - len(record)))


def _html_bytes() -> bytes:
    page = b"<!doctype html><html><body>download page</body></html>"
    return page + (b" " * (2048 - len(page)))


@pytest.mark.parametrize(
    ("content_type", "body"),
    [
        ("text/html; charset=utf-8", b"<html><body>error</body></html>"),
        ("application/octet-stream", _html_bytes()),
    ],
    ids=("html-content-type", "html-body"),
)
def test_html_download_is_rejected_without_overwriting_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content_type: str,
    body: bytes,
) -> None:
    destination = tmp_path / FILENAME
    sentinel = b"existing validated snapshot"
    destination.write_bytes(sentinel)
    calls = _install_response(
        monkeypatch,
        body=body,
        content_type=content_type,
    )

    with pytest.raises(SourceValidationError, match=r"HTML|text/html"):
        download_sd.download_file(FILENAME, tmp_path)

    assert destination.read_bytes() == sentinel
    assert not list(tmp_path.glob(f".{FILENAME}.*.part"))
    assert calls == [
        (
            f"{download_sd.BASE_URL}/{FILENAME}/download",
            {
                "headers": {"User-Agent": download_sd.USER_AGENT},
                "stream": True,
                "allow_redirects": True,
                "timeout": (30.0, 300.0),
            },
        )
    ]


def test_valid_sd_download_atomically_replaces_destination_and_reports_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _valid_sd_bytes()
    digest = hashlib.sha256(body).hexdigest()
    destination = tmp_path / FILENAME
    sentinel = b"old snapshot"
    destination.write_bytes(sentinel)
    _install_response(
        monkeypatch,
        body=body,
        content_type="chemical/x-mdl-sdfile",
    )

    real_replace = download_sd.os.replace
    replacements: list[tuple[Path, Path, bytes]] = []

    def observe_replace(source: str | Path, target: str | Path) -> None:
        source_path = Path(source)
        target_path = Path(target)
        replacements.append(
            (source_path, target_path, target_path.read_bytes())
        )
        real_replace(source, target)

    monkeypatch.setattr(download_sd.os, "replace", observe_replace)

    result = download_sd.download_file(
        FILENAME,
        tmp_path,
        expected_sha256=digest,
    )

    assert destination.read_bytes() == body
    assert result == {
        "status": "ok",
        "path": str(destination.resolve()),
        "bytes": len(body),
        "sha256": digest,
        "source_url": f"{download_sd.BASE_URL}/{FILENAME}/download",
    }
    assert len(replacements) == 1
    temporary, target, content_before_replace = replacements[0]
    assert temporary.parent == tmp_path.resolve()
    assert temporary.name.startswith(f".{FILENAME}.")
    assert temporary.suffix == ".part"
    assert target == destination.resolve()
    assert content_before_replace == sentinel
    assert not temporary.exists()
    assert not list(tmp_path.glob(f".{FILENAME}.*.part"))


def test_hash_mismatch_cleans_temporary_file_and_preserves_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / FILENAME
    sentinel = b"old snapshot"
    destination.write_bytes(sentinel)
    _install_response(
        monkeypatch,
        body=_valid_sd_bytes(),
        content_type="application/octet-stream",
    )

    with pytest.raises(SourceValidationError, match="SHA-256 mismatch"):
        download_sd.download_file(
            FILENAME,
            tmp_path,
            expected_sha256="0" * 64,
        )

    assert destination.read_bytes() == sentinel
    assert not list(tmp_path.glob(f".{FILENAME}.*.part"))
