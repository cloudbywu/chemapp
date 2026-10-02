'use strict';
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { assertManifest, assertSourceAllowlist } = require('../scripts/verify-runtime.cjs');
const manifest = (platform) => ({ schema_version: 2, platform, arch: 'x64', python: platform === 'win32' ? 'python/python.exe' : 'python/bin/python3', python_version: '3.11.16', csp5_weights: false, sources: { 'backend/app/main.py': 'hash' } });

test('native Linux and Windows runtime manifests use distinct standalone layouts', () => {
  assert.doesNotThrow(() => assertManifest(manifest('linux'), 'linux', 'x64'));
  assert.doesNotThrow(() => assertManifest(manifest('win32'), 'win32', 'x64'));
});
test('cross-platform, unsupported architecture and stale manifests fail closed', () => {
  assert.throws(() => assertManifest(manifest('linux'), 'win32', 'x64'), /match/);
  assert.throws(() => assertManifest(manifest('linux'), 'linux', 'arm64'), /native/);
  assert.throws(() => assertManifest({ ...manifest('linux'), schema_version: 1 }, 'linux', 'x64'), /match/);
  assert.throws(() => assertManifest({ ...manifest('linux'), python: '/usr/bin/python3' }, 'linux', 'x64'), /match/);
});
test('runtime metadata and source paths cannot bypass validation', () => {
  for (const source of ['../secret', '/etc/passwd', 'backend\\secret', 'backend/../secret']) {
    assert.throws(() => assertManifest({ ...manifest('linux'), sources: { [source]: 'hash' } }, 'linux', 'x64'), /Unsafe/);
  }
  assert.throws(() => assertManifest({ ...manifest('linux'), python_version: '3.12.0' }, 'linux', 'x64'), /metadata/);
  assert.throws(() => assertManifest({ ...manifest('linux'), csp5_weights: 'yes' }, 'linux', 'x64'), /metadata/);
});

test('packaged source allowlist rejects private state and unrequested model files', (t) => {
  const runtime = fs.mkdtempSync(path.join(os.tmpdir(), 'chemapp-package-allowlist-'));
  t.after(() => fs.rmSync(runtime, { recursive: true, force: true }));
  fs.mkdirSync(path.join(runtime, 'backend'), { recursive: true });
  fs.mkdirSync(path.join(runtime, 'python'));
  fs.writeFileSync(path.join(runtime, 'backend/desktop_server.py'), '# source');
  fs.writeFileSync(path.join(runtime, 'manifest.json'), '{}');
  fs.writeFileSync(path.join(runtime, 'requirements.txt'), '# locked');
  const allowed = ['backend/desktop_server.py'];
  assert.doesNotThrow(() => assertSourceAllowlist(runtime, allowed));
  for (const unexpected of ['backend/.env', 'backend/chemapp.db', 'backend/private-model.pt']) {
    fs.writeFileSync(path.join(runtime, unexpected), 'private');
    assert.throws(() => assertSourceAllowlist(runtime, allowed), /Unexpected runtime file/);
    fs.unlinkSync(path.join(runtime, unexpected));
  }
  fs.writeFileSync(path.join(runtime, 'backend/approved.pt'), 'checked weight');
  assert.doesNotThrow(() => assertSourceAllowlist(runtime, allowed, ['backend/approved.pt']));
});

test('runtime source cannot link out to user data', { skip: process.platform === 'win32' }, (t) => {
  const runtime = fs.mkdtempSync(path.join(os.tmpdir(), 'chemapp-package-link-'));
  t.after(() => fs.rmSync(runtime, { recursive: true, force: true }));
  fs.symlinkSync('/tmp', path.join(runtime, 'backend'));
  assert.throws(() => assertSourceAllowlist(runtime, []), /symlink/);
});

test('every local Electron shell module is in the explicit package file allowlist', () => {
  const desktop = path.resolve(__dirname, '..');
  const configuration = fs.readFileSync(path.join(desktop, 'electron-builder.yml'), 'utf8');
  const files = new Set([...configuration.matchAll(/^  - ([\w.-]+\.cjs)$/gm)].map((match) => match[1]));
  const visited = new Set();
  const inspect = (file) => {
    if (visited.has(file)) return;
    visited.add(file);
    assert(files.has(file), `Local shell module is missing from electron-builder files: ${file}`);
    const source = fs.readFileSync(path.join(desktop, file), 'utf8');
    for (const match of source.matchAll(/require\(['"]\.\/([\w.-]+\.cjs)['"]\)/g)) inspect(match[1]);
  };
  inspect('main.cjs');
});
