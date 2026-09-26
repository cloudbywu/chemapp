"""Quick smoke test for API endpoints."""
import json
import http.client

BOUNDARY = "----testboundary"


def upload_file(filepath: str):
    with open(filepath, "rb") as f:
        content = f.read()
    filename = filepath.split("/")[-1].split("\\")[-1]

    body = b""
    body += f"--{BOUNDARY}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    body += content
    body += f"\r\n--{BOUNDARY}--\r\n".encode()

    conn = http.client.HTTPConnection("localhost", 8000)
    conn.request(
        "POST",
        "/api/upload",
        body,
        {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"},
    )
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data


def get_spectra():
    conn = http.client.HTTPConnection("localhost", 8000)
    conn.request("GET", "/api/spectra")
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data


def analyze(sid: str):
    conn = http.client.HTTPConnection("localhost", 8000)
    conn.request(
        "POST",
        f"/api/analyze/{sid}",
        body='{"expected_revision":0}',
        headers={"Content-Type": "application/json"},
    )
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data


if __name__ == "__main__":
    result = upload_file("../dataexample/紫外example/LA.txt")
    print("UPLOAD:", json.dumps(result, indent=2))
    sid = result["id"]

    spectra = get_spectra()
    print(f"\nSPECTRA: {len(spectra)} items")

    if sid:
        analysis = analyze(sid)
        print(f"\nANALYZE: technique={analysis.get('technique')}, peaks={len(analysis.get('peaks', []))}")
