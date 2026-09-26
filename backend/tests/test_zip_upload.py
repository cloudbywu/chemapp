"""Test NMR zip upload + analysis end-to-end."""
import io
import json
import http.client
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

import app.api.store as store_module
from app.api.routes.upload import _find_nmr_dir, _find_hplc_bundle_files, _safe_extract_zip
from app.main import app

BOUNDARY = "----testboundary"


def test_safe_extract_zip_rejects_path_traversal(tmp_path):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("../evil.txt", "nope")

    payload.seek(0)
    with zipfile.ZipFile(payload) as zf:
        with pytest.raises(HTTPException) as exc_info:
            _safe_extract_zip(zf, tmp_path)

    assert exc_info.value.status_code == 400
    assert not (tmp_path.parent / "evil.txt").exists()


def test_safe_extract_zip_allows_normal_members(tmp_path):
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as zf:
        zf.writestr("sample/acqu", "params")
        zf.writestr("sample/fid", "data")

    payload.seek(0)
    with zipfile.ZipFile(payload) as zf:
        _safe_extract_zip(zf, tmp_path)

    assert (tmp_path / "sample" / "acqu").read_text() == "params"
    assert (tmp_path / "sample" / "fid").read_text() == "data"


def test_find_hplc_bundle_requires_sidecar(tmp_path):
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "run.dx").write_bytes(b"dx")
    assert _find_hplc_bundle_files(tmp_path) == []
    (sample / "run.acaml").write_text("<root />", encoding="utf-8")
    assert _find_hplc_bundle_files(tmp_path) == [sample / "run.dx"]


def test_hplc_bundle_zip_upload_preserves_result_sidecars(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    store_module._store = None
    source = (
        Path(__file__).resolve().parents[2]
        / "dataexample"
        / "液相色谱example"
        / "-S-001.sirslt"
    )
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for suffix in (".dx", ".rx", ".acaml"):
            path = source / f"-S-001{suffix}"
            zf.write(path, f"sample/{path.name}")

    with TestClient(app) as client:
        response = client.post(
            "/api/upload",
            files={"file": ("hplc-bundle.zip", archive.getvalue(), "application/zip")},
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["technique"] == "HPLC"
        analysis = client.post(
            f"/api/analyze/{payload['id']}", json={"expected_revision": 0}
        )
        assert analysis.status_code == 200, analysis.text
        assert analysis.json()["metrics"]["channel_peaks"]["DAD1A"]["peaks"]


def test_find_nmr_dir_accepts_acqu_only_fallback(tmp_path):
    # Loose parsing behaviour is intentional: an acqu-only Bruker folder
    # still qualifies (fid optional).
    sample = tmp_path / "sample"
    sample.mkdir()
    (sample / "acqu").write_text("params")
    assert _find_nmr_dir(tmp_path) == sample


def test_zip_without_bundle_error_matches_loose_acqu_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("CHEMAPP_DB_PATH", str(tmp_path / "chemapp.db"))
    store_module._store = None
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("notes/readme.txt", "not an instrument bundle")
    with TestClient(app) as client:
        response = client.post(
            "/api/upload",
            files={"file": ("bundle.zip", archive.getvalue(), "application/zip")},
        )
    assert response.status_code == 400
    # The hint must match the loose _find_nmr_dir acceptance (acqu alone is
    # enough for a Bruker folder to qualify).
    assert "acqu" in response.text


def upload_file_bytes(filename: str, content: bytes) -> dict:
    body = b""
    body += f"--{BOUNDARY}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    body += content
    body += f"\r\n--{BOUNDARY}--\r\n".encode()

    conn = http.client.HTTPConnection("localhost", 8000, timeout=10)
    conn.request("POST", "/api/upload", body, {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"})
    resp = conn.getresponse()
    raw = resp.read()
    print(f"STATUS: {resp.status}")
    print(f"RAW BODY ({len(raw)} bytes): {raw[:500]}")
    data = json.loads(raw) if raw else {"error": "empty body"}
    conn.close()
    return data, resp.status


def post(path: str) -> tuple[dict, int]:
    conn = http.client.HTTPConnection("localhost", 8000)
    conn.request("POST", path)
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data, resp.status


if __name__ == "__main__":
    with open("../dataexample/HNMRexample.zip", "rb") as f:
        zip_content = f.read()

    result, status = upload_file_bytes("HNMRexample.zip", zip_content)
    print(f"ZIP UPLOAD [{status}]: {json.dumps(result, indent=2)}")

    if status == 200:
        sid = result["id"]
        analysis, astatus = post(f"/api/analyze/{sid}")
        print(f"\nANALYSIS [{astatus}]: technique={analysis.get('technique')}, "
              f"peaks={len(analysis.get('peaks', []))}, "
              f"multiplets={len(analysis.get('multiplets', []))}")
