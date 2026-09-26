"""Test AI analysis with DeepSeek."""
import http.client
import json

BOUNDARY = "----ai_test"

def upload(filename, filepath):
    with open(filepath, "rb") as f:
        content = f.read()
    body = b""
    body += f"--{BOUNDARY}\r\n".encode()
    body += f'Content-Disposition: form-data; name="file"; filename="{filename}"\r\n'.encode()
    body += b"Content-Type: application/octet-stream\r\n\r\n"
    body += content
    body += f"\r\n--{BOUNDARY}--\r\n".encode()
    conn = http.client.HTTPConnection("localhost", 8000, timeout=30)
    conn.request("POST", "/api/upload", body, {"Content-Type": f"multipart/form-data; boundary={BOUNDARY}"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    conn.close()
    return data

cv = upload("0_1mL.txt", "../dataexample/伏安法example/0_1mL.txt")
print(f"CV uploaded: {cv.get('id')}  {cv.get('technique')}")

hplc = upload("S-001.dx", "../dataexample/液相色谱example/-S-001.sirslt/-S-001.dx")
print(f"HPLC uploaded: {hplc.get('id')}  {hplc.get('technique')}")

xrd = upload("CdS-1.asc", "../dataexample/XRDexample/CdS-1_Theta_2-Theta.asc")
print(f"XRD uploaded: {xrd.get('id')}  {xrd.get('technique')}")

print("\n--- Test 1: Single technique (CV) ---")
conn = http.client.HTTPConnection("localhost", 8000, timeout=60)
payload = json.dumps({"ids": [cv["id"]]})
conn.request("POST", "/api/ai/analyze", payload.encode(), {"Content-Type": "application/json"})
resp = conn.getresponse()
result = json.loads(resp.read())
conn.close()
print(f"Status: {resp.status}")
analysis = result.get("analysis", {})
print(f"Mode: {result.get('mode')}")
print(f"Reversibility: {analysis.get('reversibility')}")
print(f"Electrons: {analysis.get('estimated_electrons')}")
print(f"Interpretation: {str(analysis.get('interpretation', ''))[:400]}")

print("\n--- Test 2: Cross-technique (CV + HPLC + XRD) ---")
conn = http.client.HTTPConnection("localhost", 8000, timeout=90)
payload = json.dumps({"ids": [cv["id"], hplc["id"], xrd["id"]]})
conn.request("POST", "/api/ai/analyze", payload.encode(), {"Content-Type": "application/json"})
resp = conn.getresponse()
result = json.loads(resp.read())
conn.close()
print(f"Status: {resp.status}")
analysis = result.get("analysis", {})
print(f"Mode: {result.get('mode')}")
print(f"Technique corroboration: {analysis.get('technique_corroboration')}")
print(f"Suggested identity: {analysis.get('suggested_identity')}")
print(f"Confidence: {analysis.get('confidence')}")
print(f"Interpretation: {str(analysis.get('interpretation', ''))[:500]}")

print("\n--- Test 3: Question mode ---")
conn = http.client.HTTPConnection("localhost", 8000, timeout=60)
payload = json.dumps({"ids": [cv["id"]], "question": "Is this redox process reversible? What can you tell about the analyte?"})
conn.request("POST", "/api/ai/analyze", payload.encode(), {"Content-Type": "application/json"})
resp = conn.getresponse()
result = json.loads(resp.read())
conn.close()
print(f"Status: {resp.status}")
print(f"Answer: {result.get('answer', '')[:500]}")
