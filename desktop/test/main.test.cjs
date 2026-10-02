'use strict';
// Source-level security invariants supplement runtime policy/lifecycle tests.
// They do not replace a real Windows renderer/installer acceptance pass.
const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const main = fs.readFileSync(path.join(__dirname, '../main.cjs'), 'utf8');
test('desktop renderer retains hard security invariants and no privilege bridge', () => {
  for (const setting of ['app.enableSandbox()', 'contextIsolation: true', 'sandbox: true', 'nodeIntegration: false', 'webSecurity: true', 'webviewTag: false']) assert(main.includes(setting), setting);
  assert(!/require\(['"](?:electron\/remote|@electron\/remote)['"]\)|ipcMain|preload:|--no-sandbox|webSecurity:\s*false/.test(main));
  assert(main.includes("setWindowOpenHandler(() => ({ action: 'deny' }))"));
  assert(main.includes("setProxy({ mode: 'direct' })"));
});
