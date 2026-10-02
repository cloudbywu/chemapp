'use strict';
// Integration check of the real Python service, built UI, API gate and cleanup.
// Uses a temporary empty database and never touches the user's working database.
const { Backend, launchPaths, health } = require('../backend.cjs');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'chemapp-desktop-smoke-'));
const backend = new Backend({
  paths: launchPaths({ packaged: false, pythonOverride: process.env.CHEMAPP_DESKTOP_PYTHON }),
  dataDir: dir, onLog: (value) => process.stderr.write(value),
});
(async () => {
  try {
    const origin = await backend.start();
    const headers = {
      'X-ChemApp-Desktop-Token': backend.secrets.desktop,
      'X-ChemApp-Access-Token': backend.secrets.access,
      'X-ChemApp-Admin-Token': backend.secrets.admin,
    };
    assert.equal((await fetch(`${origin}/api/ready`)).status, 403);
    assert.equal((await fetch(`${origin}/`, {headers: {...headers, Origin: 'https://untrusted.example'}})).status, 403);
    const page = await fetch(`${origin}/`, {headers});
    assert.equal(page.status, 200);
    assert.match(page.headers.get('content-security-policy'), /script-src 'self'/);
    const html = await page.text();
    assert(!html.includes('fonts.googleapis.com'));
    const script = html.match(/src="([^\"]+\.js)"/)[1];
    assert.equal((await fetch(origin + script, {headers})).status, 200);
    const list = await fetch(`${origin}/api/spectra`, {headers});
    assert.equal(list.status, 200);
    assert.deepEqual(await list.json(), []);
    const healthResponse = await fetch(`${origin}/api/health`, {headers});
    assert.equal(healthResponse.status, 200);
    const report = await healthResponse.json();
    assert.equal(report.assets.calibration_policy.probability_claim_allowed, false);
    assert(fs.existsSync(path.join(dir, 'chemapp.db')));
    await backend.stop();
    assert(backend.exited);
    assert.equal(await health(origin, backend.secrets, backend.instance), false);
    console.log('PASS: owned real backend, readiness, static UI/assets, access/origin denial, empty database, unchanged science gate, graceful shutdown');
  } finally {
    await backend.stop();
    fs.rmSync(dir, {recursive: true, force: true});
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
