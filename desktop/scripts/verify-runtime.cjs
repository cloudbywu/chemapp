'use strict';
const fs = require('node:fs');
const path = require('node:path');
const { createHash } = require('node:crypto');
const { spawnSync } = require('node:child_process');
const root = path.resolve(__dirname, '../..');
const platforms = { win32: 'python/python.exe', linux: 'python/bin/python3' };
const sourcePaths = ['backend/app', 'backend/vendor', 'backend/scripts', 'dataexample', 'LICENSE', 'NOTICE', 'THIRD_PARTY_DATA.md'];
const hash = (file) => createHash('sha256').update(fs.readFileSync(file)).digest('hex');

function assertManifest(manifest, platform = process.platform, arch = process.arch) {
  if (!platforms[platform] || arch !== 'x64') throw new Error('Package on a native Windows or Linux x64 host.');
  if (manifest.schema_version !== 2 || manifest.platform !== platform || manifest.arch !== arch || manifest.python !== platforms[platform]) {
    throw new Error('Staged runtime does not match this platform/architecture. Restage on the target OS.');
  }
  if (!/^3\.11\.\d+$/.test(manifest.python_version) || typeof manifest.csp5_weights !== 'boolean') throw new Error('Invalid staged Python or model metadata.');
  for (const relative of Object.keys(manifest.sources || {})) {
    if (path.isAbsolute(relative) || relative.includes('\\') || relative.split('/').some((part) => ['', '.', '..'].includes(part))) throw new Error(`Unsafe staged source path: ${relative}`);
  }
}

function assertSourceAllowlist(runtime, expected, extraFiles = []) {
  // Reject accidental local state/model additions outside the isolated Python
  // distribution. Runtime source is an allowlist, never a working-tree glob.
  const allowed = new Set([...expected, 'manifest.json', 'requirements.txt', ...extraFiles]);
  const inspect = (directory) => {
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      const file = path.join(directory, entry.name);
      const relative = path.relative(runtime, file).split(path.sep).join('/');
      if (relative === 'python') {
        if (!entry.isDirectory() || entry.isSymbolicLink()) throw new Error('Standalone Python must be a real bundled directory.');
        continue;
      }
      if (entry.isSymbolicLink()) throw new Error(`Unexpected runtime source symlink: ${relative}`);
      if (entry.isDirectory()) inspect(file);
      else if (!allowed.has(relative)) throw new Error(`Unexpected runtime file (possibly private local state): ${relative}`);
    }
  };
  inspect(runtime);
}

function verifyRuntime({ runtime = path.join(root, 'desktop/build/runtime'), checkImports = true, expectedPlatform = process.platform } = {}) {
  if (expectedPlatform !== process.platform) throw new Error('Cross-platform runtime packaging is not supported.');
  const manifest = JSON.parse(fs.readFileSync(path.join(runtime, 'manifest.json'), 'utf8'));
  assertManifest(manifest);
  if (manifest.lock_sha256 !== hash(path.join(root, 'backend/uv.lock'))) throw new Error('Runtime dependencies are stale. Restage against the current uv.lock.');
  if (manifest.requirements_sha256 !== hash(path.join(runtime, 'requirements.txt'))) throw new Error('Staged locked requirements have changed. Restage the runtime.');
  const tracked = spawnSync('git', ['ls-files', '-z', ...sourcePaths], { cwd: root, encoding: 'utf8' });
  if (tracked.status !== 0) throw new Error('Cannot verify source freshness without the repository metadata.');
  const expected = [...new Set([...tracked.stdout.split('\0').filter(Boolean), 'backend/desktop_server.py'])].sort();
  if (JSON.stringify(Object.keys(manifest.sources || {}).sort()) !== JSON.stringify(expected)) throw new Error('The staged source list is stale. Restage the runtime.');
  for (const relative of expected) {
    const source = path.join(root, relative);
    const staged = path.join(runtime, relative);
    if (fs.lstatSync(source).isSymbolicLink() || !fs.existsSync(staged) || fs.lstatSync(staged).isSymbolicLink()) throw new Error(`Invalid staged source: ${relative}`);
    if (hash(source) !== manifest.sources[relative] || hash(staged) !== manifest.sources[relative]) throw new Error(`The staged backend/source is stale or changed: ${relative}. Restage the runtime.`);
  }
  for (const file of ['backend/desktop_server.py', 'backend/app/main.py', 'backend/app/ml/calibration/policy-v1.json', manifest.python]) {
    if (!fs.existsSync(path.join(runtime, file))) throw new Error(`Missing runtime file: ${file}`);
  }
  if (fs.existsSync(path.join(runtime, 'python/pyvenv.cfg'))) throw new Error('A virtual environment cannot be packaged as standalone Python.');
  const weights = JSON.parse(fs.readFileSync(path.join(runtime, 'backend/vendor/csp5/weights-manifest.json'), 'utf8'));
  for (const entry of weights.files) {
    const weight = path.join(runtime, 'backend/vendor/csp5', entry.path);
    if (manifest.csp5_weights) {
      if (!fs.existsSync(weight) || fs.lstatSync(weight).isSymbolicLink() || hash(weight) !== entry.sha256) throw new Error(`Missing or changed CSP5 weight: ${entry.path}`);
    } else if (fs.existsSync(weight)) throw new Error('Unexpected unrequested CSP5 weights in the staged runtime.');
  }
  assertSourceAllowlist(runtime, expected, manifest.csp5_weights ? weights.files.map((entry) => `backend/vendor/csp5/${entry.path}`) : []);
  const python = path.join(runtime, manifest.python);
  if (process.platform !== 'win32') fs.accessSync(python, fs.constants.X_OK);
  if (checkImports) {
    const env = { ...process.env };
    delete env.PYTHONHOME;
    delete env.PYTHONPATH;
    const check = spawnSync(python, ['-I', '-B', path.join(__dirname, 'check-python-runtime.py'), '--runtime', runtime], { stdio: 'inherit', cwd: path.dirname(runtime), env });
    if (check.error) throw check.error;
    if (check.status !== 0) throw new Error('Bundled Python isolation/import/locked-dependency check failed.');
  }
  return manifest;
}

if (require.main === module) {
  const args = process.argv.slice(2);
  const platformIndex = args.indexOf('--platform');
  const runtimeIndex = args.indexOf('--runtime');
  const manifest = verifyRuntime({
    expectedPlatform: platformIndex < 0 ? process.platform : args[platformIndex + 1],
    runtime: runtimeIndex < 0 ? undefined : path.resolve(args[runtimeIndex + 1]),
  });
  console.log(`${manifest.platform} x64 runtime sources, assets and complete locked dependencies verified. Native desktop UI testing is still required.`);
}
module.exports = { assertManifest, assertSourceAllowlist, verifyRuntime };
