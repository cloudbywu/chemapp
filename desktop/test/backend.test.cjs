'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { Backend, health, launchPaths } = require('../backend.cjs');
const root = path.resolve(__dirname, '../..');
const python = process.env.CHEMAPP_DESKTOP_TEST_PYTHON || path.join(root, 'backend/.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');

function fixture(mode = 'ok') {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'chemapp-desktop-test-'));
  fs.writeFileSync(path.join(dir, 'index.html'), '<!doctype html>');
  return { dir, backend: new Backend({
    paths: { python, launcher: path.join(__dirname, 'fixtures/backend.py'), frontend: dir },
    dataDir: dir, env: { CHEMAPP_TEST_MODE: mode }, startupTimeout: 1800, shutdownTimeout: 1000,
  }) };
}

test('packaged launch paths never use developer Python overrides', () => {
  const paths = launchPaths({ packaged: true, resourcesPath: '/resources', pythonOverride: '/evil', platform: 'win32' });
  assert.equal(paths.python, '/resources/runtime/python/python.exe');
  assert.equal(paths.launcher, '/resources/runtime/backend/desktop_server.py');
});
test('launches owned random-port backend, authenticates readiness, and closes on stdin EOF', async (t) => {
  const { dir, backend } = fixture();
  t.after(async () => { await backend.stop(); fs.rmSync(dir, { recursive: true }); });
  const origin = await backend.start();
  assert.match(origin, /^http:\/\/127\.0\.0\.1:\d+$/);
  assert(await health(origin, backend.secrets, backend.instance));
  assert.equal(await health(origin, {desktop: 'wrong', access: 'wrong'}, backend.instance), false);
  await assert.rejects(backend.start(), /twice/);
  await backend.stop();
  assert(backend.exited);
  assert.equal(await health(origin, backend.secrets, backend.instance), false);
});
test('does not accept a healthy response with a different launch identity', async (t) => {
  const { dir, backend } = fixture('wrong-instance');
  t.after(async () => { await backend.stop(); fs.rmSync(dir, { recursive: true }); });
  await assert.rejects(backend.start(), /did not become ready/);
  assert(backend.exited);
});
test('startup process failure is reported and cleaned up', async (t) => {
  const { dir, backend } = fixture('exit');
  t.after(async () => { await backend.stop(); fs.rmSync(dir, { recursive: true }); });
  await assert.rejects(backend.start(), /exited/);
  assert(backend.exited);
});
test('shutdown during startup prevents late child processes', async (t) => {
  const { dir, backend } = fixture();
  t.after(() => fs.rmSync(dir, { recursive: true }));
  await backend.stop();
  await assert.rejects(backend.start(), /after shutdown/);
  assert.equal(backend.child, null);
});

test('a crashing backend cannot leave an owned process-group descendant behind', {skip: process.platform === 'win32'}, async (t) => {
  const { dir, backend } = fixture('grandchild');
  t.after(async () => { await backend.stop(); fs.rmSync(dir, {recursive: true}); });
  await assert.rejects(backend.start(), /exited/);
  const pid = Number(fs.readFileSync(path.join(dir, 'grandchild.pid'), 'utf8'));
  const deadline = Date.now() + 1500;
  let running = true;
  while (running && Date.now() < deadline) {
    try {
      // In a container a killed child may briefly be an unreaped zombie.
      const status = fs.readFileSync(`/proc/${pid}/stat`, 'utf8').split(' ')[2];
      running = status !== 'Z';
    } catch { running = false; }
    if (running) await new Promise(resolve => setTimeout(resolve, 30));
  }
  assert.equal(running, false);
});
