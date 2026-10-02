import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
import sys
import threading

mode = os.environ.get('CHEMAPP_TEST_MODE', 'ok')
if mode == 'grandchild':
    import subprocess
    from pathlib import Path
    child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
    Path(os.environ['CHEMAPP_DESKTOP_DATA_DIR'], 'grandchild.pid').write_text(str(child.pid))
    raise SystemExit(7)
if mode == 'exit':
    raise SystemExit(9)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        authenticated = self.headers.get('X-ChemApp-Desktop-Token') == os.environ['CHEMAPP_DESKTOP_TOKEN']
        self.send_response(200 if authenticated else 403)
        instance = os.environ['CHEMAPP_DESKTOP_INSTANCE'] if mode != 'wrong-instance' else 'wrong'
        self.send_header('X-ChemApp-Desktop-Instance', instance)
        self.end_headers()
        self.wfile.write(json.dumps({'status': 'ok'}).encode())

server = HTTPServer(('127.0.0.1', 0), Handler)
print('CHEMAPP_DESKTOP_PORT ' + json.dumps({'port': server.server_port, 'instance': os.environ['CHEMAPP_DESKTOP_INSTANCE']}), flush=True)

def owner():
    while sys.stdin.buffer.read(1):
        pass
    server.shutdown()

threading.Thread(target=owner, daemon=True).start()
server.serve_forever(poll_interval=0.05)
server.server_close()
