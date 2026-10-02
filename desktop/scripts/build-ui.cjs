'use strict';
const { spawnSync } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const frontend = path.resolve(__dirname, '../../frontend');
// Vite's process environment takes precedence over .env*. Desktop is always same-origin.
const result = spawnSync(process.execPath, [path.join(frontend, 'node_modules/typescript/bin/tsc'), '-b'], {
  cwd: frontend, stdio: 'inherit', env: process.env,
});
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status ?? 1);
const build = spawnSync(process.execPath, [path.join(frontend, 'node_modules/vite/bin/vite.js'), 'build'], {
  cwd: frontend, stdio: 'inherit', env: { ...process.env, VITE_API_URL: '' },
});
if (build.error) throw build.error;
if (build.status !== 0) process.exit(build.status ?? 1);
// Desktop must also work offline; do not request third-party font services.
const index = path.join(frontend, 'dist/index.html');
fs.writeFileSync(index, fs.readFileSync(index, 'utf8').replace(/^.*<link[^>]+https:\/\/fonts\.(?:googleapis|gstatic)\.com[^>]*>\s*$/gm, ''));
// Exercise the actual production Plotly bundle without Node globals. Mocked
// component tests cannot catch accidental browser use of Node's `global`.
const plotCheck = spawnSync(process.execPath, ['--experimental-vm-modules', path.join(frontend, 'scripts/verify-plotly-bundle.mjs')], {
  cwd: frontend, stdio: 'inherit', env: process.env,
});
if (plotCheck.error) throw plotCheck.error;
if (plotCheck.status !== 0) process.exit(plotCheck.status ?? 1);
