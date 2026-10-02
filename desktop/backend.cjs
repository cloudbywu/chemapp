'use strict';
const { spawn } = require('node:child_process');
const { randomBytes } = require('node:crypto');
const { EventEmitter } = require('node:events');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const { DESKTOP_HEADER } = require('./security.cjs');

function launchPaths({ packaged, resourcesPath, root = path.resolve(__dirname, '..'), pythonOverride, platform = process.platform }) {
  if (packaged) {
    const runtime = path.join(resourcesPath, 'runtime');
    return {
      python: path.join(runtime, 'python', platform === 'win32' ? 'python.exe' : 'bin/python3'),
      launcher: path.join(runtime, 'backend/desktop_server.py'),
      frontend: path.join(resourcesPath, 'frontend'),
    };
  }
  return {
    python: pythonOverride || path.join(root, 'backend/.venv', platform === 'win32' ? 'Scripts/python.exe' : 'bin/python'),
    launcher: path.join(root, 'backend/desktop_server.py'),
    frontend: path.join(root, 'frontend/dist'),
  };
}

function health(origin, secrets, instance, timeout = 1500) {
  return new Promise((resolve) => {
    const request = http.get(`${origin}/api/ready`, {
      headers: { [DESKTOP_HEADER]: secrets.desktop, 'X-ChemApp-Access-Token': secrets.access },
      timeout,
    }, (response) => {
      let body = '';
      response.on('data', (chunk) => {
        body += chunk;
        if (body.length > 16384) response.destroy();
      });
      response.on('end', () => {
        try {
          resolve(response.statusCode === 200 && response.headers['x-chemapp-desktop-instance'] === instance && JSON.parse(body).status === 'ok');
        } catch { resolve(false); }
      });
      response.on('error', () => resolve(false));
    });
    request.on('timeout', () => { request.destroy(); resolve(false); });
    request.on('error', () => resolve(false));
  });
}

class Backend extends EventEmitter {
  constructor(options) {
    super();
    this.options = options;
    this.secrets = Object.fromEntries(['desktop', 'access', 'admin'].map((key) => [key, randomBytes(32).toString('hex')]));
    this.instance = randomBytes(32).toString('hex');
    this.stopping = false;
    this.child = null;
    this.exited = false;
    this.origin = null;
  }

  async start() {
    if (this.stopping || this.child) throw new Error('Backend cannot be started twice or after shutdown.');
    const { python, launcher, frontend } = this.options.paths;
    for (const file of [python, launcher, path.join(frontend, 'index.html')]) {
      if (!fs.existsSync(file)) throw new Error(`Required desktop component is missing: ${file}`);
    }
    fs.mkdirSync(this.options.dataDir, { recursive: true });
    const env = { ...process.env, ...this.options.env,
      CHEMAPP_DESKTOP_TOKEN: this.secrets.desktop,
      CHEMAPP_DESKTOP_INSTANCE: this.instance,
      CHEMAPP_ACCESS_TOKEN: this.secrets.access,
      CHEMAPP_ADMIN_TOKEN: this.secrets.admin,
      CHEMAPP_DESKTOP_FRONTEND: frontend,
      CHEMAPP_DESKTOP_DATA_DIR: this.options.dataDir,
      PYTHONDONTWRITEBYTECODE: '1',
    };
    delete env.PYTHONHOME;
    delete env.PYTHONPATH;
    const child = spawn(python, ['-I', '-B', '-u', launcher], {
      cwd: this.options.dataDir, env, windowsHide: true,
      detached: process.platform !== 'win32', stdio: ['pipe', 'pipe', 'pipe'],
    });
    this.child = child;
    let stdout = '';
    let failure = null;
    child.stdin.on('error', () => {}); // EPIPE when startup fails is reported via exit.
    child.on('error', (error) => { failure = error; this.exited = true; });
    child.on('exit', (code, signal) => {
      this.exited = true;
      // The child was a dedicated session/process-group leader. Clean any
      // lingering descendants immediately, even when the direct child crashed.
      if (process.platform !== 'win32' && child.pid) {
        try { process.kill(-child.pid, 'SIGKILL'); } catch { /* no members remain */ }
      }
      failure = new Error(`The local backend exited (${signal || code}).`);
      if (!this.stopping) this.emit('unexpected-exit', failure);
    });
    child.stderr.on('data', (chunk) => this.options.onLog?.(String(chunk)));
    child.stdout.on('data', (chunk) => {
      stdout += chunk;
      if (stdout.length > 65536) stdout = stdout.slice(-65536);
      let newline;
      while ((newline = stdout.indexOf('\n')) !== -1) {
        const line = stdout.slice(0, newline).trim();
        stdout = stdout.slice(newline + 1);
        if (!line.startsWith('CHEMAPP_DESKTOP_PORT ')) continue;
        try {
          const info = JSON.parse(line.slice('CHEMAPP_DESKTOP_PORT '.length));
          if (info.instance === this.instance && Number.isInteger(info.port) && info.port > 0 && info.port < 65536 && !this.origin) {
            this.origin = `http://127.0.0.1:${info.port}`;
          }
        } catch { /* non-protocol stdout is never trusted */ }
      }
    });
    const deadline = Date.now() + (this.options.startupTimeout ?? 120000);
    try {
      while (Date.now() < deadline) {
        if (this.stopping) throw new Error('Desktop startup cancelled.');
        if (failure || this.exited) throw failure || new Error('The local backend stopped.');
        if (this.origin && await health(this.origin, this.secrets, this.instance)) {
          if (failure || this.exited || this.stopping) throw failure || new Error('The local backend stopped.');
          return this.origin;
        }
        await new Promise((resolve) => setTimeout(resolve, 150));
      }
      throw new Error('The local backend did not become ready within 120 seconds.');
    } catch (error) {
      await this.stop();
      throw error;
    }
  }

  async stop() {
    if (this.stopPromise) return this.stopPromise;
    this.stopping = true;
    this.stopPromise = (async () => {
      const child = this.child;
      if (!child || this.exited) return;
      child.stdin.end();
      const exited = new Promise((resolve) => child.once('exit', resolve));
      const wait = async (ms) => {
        let timer;
        await Promise.race([exited, new Promise((resolve) => { timer = setTimeout(resolve, ms); })]);
        clearTimeout(timer);
      };
      await wait(this.options.shutdownTimeout ?? 8000);
      if (this.exited || !child.pid) return;
      // Kill only the child/process group this instance created, never a port owner.
      if (process.platform === 'win32') {
        await new Promise((resolve) => {
          const killer = spawn(path.join(process.env.SystemRoot || 'C:\\Windows', 'System32/taskkill.exe'), ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
          killer.once('exit', resolve);
          killer.once('error', resolve);
        });
      } else {
        try { process.kill(-child.pid, 'SIGKILL'); } catch { /* already exited */ }
      }
      await wait(2000);
      if (!this.exited) throw new Error('Could not confirm that the owned backend stopped.');
    })();
    return this.stopPromise;
  }
}

module.exports = { Backend, health, launchPaths };
