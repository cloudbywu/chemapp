"""Test inference endpoint end-to-end."""
import json
import http.client

BOUNDARY = "----testboundary"


def upload_file_bytes(filename: str, content: bytes):
    body = b""
    body += f"--{BOUNDARY}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    body += content
    body += f"\r\n--{BOUNDARY}--\r\n".encode()
    conn = http.client.HTTPConnection("localhost", 8000, timeout=10)
    conn.request("POST", "/api/upload", body, {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data.get("id"), data


def post_json(path: str, payload: dict):
    conn = http.client.HTTPConnection("localhost", 8000, timeout=10)
    body = json.dumps(payload).encode()
    conn.request("POST", path, body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data, resp.status


if __name__ == "__main__":
    ids = []

    # Upload UV-Vis LA.txt
    with open("../dataexample/紫外example/LA.txt", "rb") as f:
        uid, _ = upload_file_bytes("LA.txt", f.read())
        ids.append(uid)
        print(f"UV-Vis uploaded: {uid}")

    # Upload fluorescence em-lao
    with open("../dataexample/荧光example/em-lao(FDS).DX", "rb") as f:
        uid, _ = upload_file_bytes("em-lao.DX", f.read())
        ids.append(uid)
        print(f"Fluorescence uploaded: {uid}")

    # Upload fluorescence ex-lao
    with open("../dataexample/荧光example/ex-lao.DX", "rb") as f:
        uid, _ = upload_file_bytes("ex-lao.DX", f.read())
        ids.append(uid)
        print(f"Fluorescence uploaded: {uid}")

    # Run inference on all
    result, status = post_json("/api/inference", {"ids": ids})
    print(f"\nInference [{status}]:")
    print(f"  Consistency: {result.get('inference', {}).get('consistency_score')}")
    print(f"  Confidence: {result.get('inference', {}).get('confidence')}")
    print(f"  Anomalies: {len(result.get('inference', {}).get('anomalies', []))}")
    print(f"  Cross-validations: {len(result.get('inference', {}).get('cross_validations', []))}")
    print(f"  Techniques: {result.get('techniques')}")
    print("\n--- Report Markdown (first 500 chars) ---")
    md = result.get("report_markdown", "")
    print(md[:500])
