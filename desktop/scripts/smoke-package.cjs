'use strict';
// Run the actual packaged launcher and Python with a temporary database. This
// does not start Chromium or weaken its sandbox. Pass a relocated resources dir.
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const asar = require('@electron/asar');
const { verifyRuntime } = require('./verify-runtime.cjs');
const resources = path.resolve(process.argv[2] || path.join(__dirname, '../release', process.platform === 'win32' ? 'win-unpacked' : 'linux-unpacked', 'resources'));
const runtime = path.join(resources, 'runtime');
const manifest = verifyRuntime({ runtime });
const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'chemapp-package-smoke-'));
let backend;
(async () => {
  try {
    const extracted = path.join(temporary, 'app');
    asar.extractAll(path.join(resources, 'app.asar'), extracted);
    const hash = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex');
    for (const name of ['main.cjs', 'backend.cjs', 'security.cjs', 'window-state.cjs', 'capture-workspace.cjs', 'loading.html']) {
      assert.equal(hash(path.join(extracted, name)), hash(path.join(__dirname, '..', name)), `Packaged shell is stale: ${name}`);
    }
    const frontendBuild = path.resolve(__dirname, '../../frontend/dist');
    const compareFrontend = (relative = '') => {
      for (const entry of fs.readdirSync(path.join(frontendBuild, relative), { withFileTypes: true })) {
        const file = path.join(relative, entry.name);
        if (entry.isDirectory()) compareFrontend(file);
        else assert.equal(hash(path.join(resources, 'frontend', file)), hash(path.join(frontendBuild, file)), `Packaged UI is stale: ${file}`);
      }
    };
    compareFrontend();
    // Require the exact code inside the package, not the working checkout copy.
    const { Backend, launchPaths, health } = require(path.join(extracted, 'backend.cjs'));
    const paths = launchPaths({ packaged: true, resourcesPath: resources, pythonOverride: '/missing/development-python' });
    assert.equal(paths.python, path.join(runtime, manifest.python));
    backend = new Backend({
      paths, dataDir: path.join(temporary, 'user-data'),
      env: { PATH: '', PYTHONHOME: '/missing/host-python', PYTHONPATH: '/missing/site-packages', HF_HOME: path.join(temporary, 'model-cache') },
      onLog: (value) => process.stderr.write(value),
    });
    const origin = await backend.start();
    const headers = {
      'X-ChemApp-Desktop-Token': backend.secrets.desktop,
      'X-ChemApp-Access-Token': backend.secrets.access,
      'X-ChemApp-Admin-Token': backend.secrets.admin,
    };
    assert.equal((await fetch(`${origin}/api/ready`)).status, 403);
    assert.equal((await fetch(`${origin}/`, { headers: { ...headers, Origin: 'https://untrusted.example' } })).status, 403);
    const page = await fetch(`${origin}/`, { headers });
    assert.equal(page.status, 200);
    assert.match(page.headers.get('content-security-policy'), /script-src 'self'/);
    const html = await page.text();
    assert(!html.includes('fonts.googleapis.com'));
    const script = html.match(/src="([^\"]+\.js)"/)[1];
    assert.equal((await fetch(origin + script, { headers })).status, 200);
    const spectra = await fetch(`${origin}/api/spectra`, { headers });
    assert.equal(spectra.status, 200);
    assert.deepEqual(await spectra.json(), []);
    const response = await fetch(`${origin}/api/health`, { headers });
    assert.equal(response.status, 200);
    const report = await response.json();
    assert.equal(report.assets.calibration_policy.probability_claim_allowed, false);
    assert.equal(report.assets.csp5_weights.status, manifest.csp5_weights ? 'ok' : 'error');
    assert(fs.existsSync(path.join(temporary, 'user-data/chemapp.db')));
    assert(!fs.existsSync(path.join(runtime, 'backend/data/chemapp.db')));
    await backend.stop();
    assert(backend.exited);
    assert.equal(await health(origin, backend.secrets, backend.instance), false);
    console.log(JSON.stringify({ result: 'PASS', resources, python: manifest.python, csp5: report.assets.csp5_weights.status, checks: 'Packaged ASAR launcher, bundled isolated Python with empty PATH, static UI/assets, private readiness, origin denial, new user database, unchanged science gate and clean shutdown. Native Chromium UI is a separate check.' }, null, 2));
  } finally {
    if (backend) await backend.stop();
    fs.rmSync(temporary, { recursive: true, force: true });
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
