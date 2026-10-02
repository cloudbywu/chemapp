// Run after `npm run build`:
// node --experimental-vm-modules scripts/verify-plotly-bundle.mjs
// This checks the actual production chunk in a browser-like realm with no
// Node globals. Mocked component tests cannot catch bundler compatibility bugs.
import assert from 'node:assert/strict';
import { readFile, readdir } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';
import { JSDOM } from 'jsdom';

const assets = fileURLToPath(new URL('../dist/assets/', import.meta.url));
const entries = (await readdir(assets)).filter((name) => /^plotlyCustom-.*\.js$/.test(name));
assert.equal(entries.length, 1, 'Build the frontend before checking its Plotly bundle');
assert.equal(typeof vm.SourceTextModule, 'function', 'Run Node with --experimental-vm-modules');
const dom = new JSDOM('<!doctype html><html><head></head><body></body></html>', { runScripts: 'outside-only' });

try {
  const context = dom.getInternalVMContext();
  assert.equal(vm.runInContext('typeof global', context), 'undefined');
  assert.equal(vm.runInContext('typeof process', context), 'undefined');
  assert.equal(vm.runInContext('typeof require', context), 'undefined');
  const modules = new Map();
  const load = async (filename) => {
    if (modules.has(filename)) return modules.get(filename);
    assert.equal(path.dirname(filename), assets.replace(/\/$/, ''), 'The chunk must only import local build assets');
    const module = new vm.SourceTextModule(await readFile(filename, 'utf8'), { context, identifier: filename });
    modules.set(filename, module);
    await module.link((specifier, referencing) => {
      assert.ok(specifier.startsWith('./'), 'Only local bundled imports are allowed');
      return load(path.resolve(path.dirname(referencing.identifier), specifier));
    });
    return module;
  };
  const module = await load(path.join(assets, entries[0]));
  await module.evaluate();
  for (const method of ['react', 'relayout', 'purge']) {
    assert.equal(typeof module.namespace.default[method], 'function', `Plotly.${method} must load without Node globals`);
  }
  assert.equal(vm.runInContext('typeof global', context), 'undefined', 'The build must not install a runtime Node global shim');
  console.log('Production Plotly bundle loads without global, process, or require');
} finally {
  dom.window.close();
}
