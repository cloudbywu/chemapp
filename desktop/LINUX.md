# Linux x64 desktop packaging

The Linux build packages the existing Electron shell, React UI, complete Python
backend, native production dependencies and tracked scientific assets. Windows
remains supported by its separate native packaging commands.

## Build requirements

- A **glibc-based Linux x86_64** build machine; Linux ARM, musl/Alpine and macOS
  cross-builds are not supported
- Node.js >=22.12, npm, uv and a Python 3.11+ command to run the staging script
- A working graphical Linux session to test Electron; ordinary OS desktop
  libraries such as GTK, NSS and GBM must be present on the destination
- Generous disk space: the current full Linux dependency lock includes CUDA
  libraries. The staged Python/runtime is over 5 GiB before Electron and the
  frontend. Staging, unpacked output, the temporary tar and final gzip coexist
  during a build, so allow **25 GiB free** after installing development dependencies

The installed app uses only its bundled standalone Python. It does not require uv,
Node, system Python, an activated virtualenv or CUDA hardware for ordinary startup.
A GPU driver is still needed for actual GPU execution. These native wheels are
not a universal Linux binary: the current selected wheels include a
`manylinux_2_34` cryptography wheel, so do not claim an older glibc baseline.
Validate each supported distro before distribution.

## Native build

Install frontend and desktop npm dependencies first as described in README.md,
then run from desktop/:

```sh
npm run install:electron
npm run stage:linux
npm run pack:linux
npm run test:package
npm run dist:linux
```

To include the existing pinned CSP5 weights, use this **instead** of the staging
command above:

```sh
npm run stage:linux -- --with-csp5
```

The optional CSP5 fetch uses the existing checksum-verifying fetcher in a fresh
staged vendor directory. It never copies arbitrary untracked models from the
checkout. Missing external T5/DP5q/retrieval assets remain diagnostic conditions;
no calibration, provenance or probability gate is changed by packaging.

`stage:linux` pins managed standalone CPython **3.11.16**. A different explicit
3.11 patch may be selected with `--python-version 3.11.N`; its exact version is
recorded and verified. `UV_PYTHON_INSTALL_DIR` or `--python-install-dir` can select
an existing uv-managed download cache. `UV_CACHE_DIR` can select the wheel cache.
On restricted builders, `ELECTRON_BUILDER_CACHE` and `TMPDIR` may also need writable
locations for archive tools and temporary tar files.
Dependencies are reinstalled from the complete exported `uv.lock` with hashes and
copy mode, rather than copied from a venv or linked to a cache. The complete
runtime is moved to a different path (including spaces) and tested before staging
finishes. Existing staging is never overwritten: move build/runtime aside before
restaging after backend or lock changes.

Outputs:

- `release/linux-unpacked/chemapp`: unpacked application and bundled resources
- `release/ChemApp-0.2.0-linux-x64.tar.gz`: relocatable archive, not a system installer
- `build/runtime/manifest.json`: platform, interpreter, lock and tracked source hashes
- `build/runtime/requirements.txt`: exact hashed production dependency export

All packaging commands use `--publish never`. Nothing is uploaded, auto-updated,
signed or installed into system directories. The tar archive does not register a
launcher or file association. Extract it into a writable location and run its
`chemapp` executable as a normal desktop user. Keep the extracted directory intact.

For an already-downloaded Electron binary, electron-builder's supported local
input avoids a second download:

```sh
npm run pack:linux -- --config.electronDist=node_modules/electron/dist
npm run dist:linux -- --config.electronDist=node_modules/electron/dist
```

For faster development archives, the builder supports an explicit compression
level, for example `ELECTRON_BUILDER_COMPRESSION_LEVEL=1 npm run dist:linux`.
Default release compression is higher and takes longer with the full native wheels.

## Verify the package after moving it

The package smoke test reads backend launch/security code from the actual
`resources/app.asar`, uses the packaged Python with an empty PATH, a temporary
empty database and unrelated working directory, and checks:

- Every active locked runtime distribution and every mandatory dependency import
- Python's resolved executable, prefix and all import paths stay inside the bundle
- Source and optional CSP5 hashes match; no unexpected local data was bundled
- Real backend readiness, offline frontend assets and authenticated static/API access
- Unauthorized requests and foreign origins are denied
- Calibration probability claims remain disabled and CSP5 diagnostics stay truthful
- Database creation stays in user data; shutdown closes the owned backend and port

To repeat against a relocated or tar-extracted package, run from desktop/:

```sh
npm run test:package -- '/new/path/ChemApp/resources'
```

The checker compares packaged sources with the current checkout and rejects stale
backend assets. It does not launch Chromium, so it can run on a headless builder;
a passing result is **not** a claim of native window/UI verification.

Additional fast checks:

```sh
npm run check
npm run test:staging
```

## Linux acceptance checklist

1. Extract the archive as a normal user in a graphical session. Run with no system
   Python/Node on PATH. Test first launch, relaunch and second-instance focus
2. Keep Chromium sandboxing enabled. If a host forbids required namespaces or
   otherwise prevents sandbox startup, report the host restriction; do not add
   `--no-sandbox`, change sandbox permissions or disable OS security to bypass it
3. Check native window rendering, resize/minimize/maximize, keyboard navigation,
   loading/error screens and application menus on the intended display server
4. Import/analyze the real examples, edit/review/undo/compare, render Plotly/SMILES,
   and export CSV/PNG/DOCX through native Save dialogs
5. Test Close/Cancel during startup and analysis; terminate the owned backend and
   Electron separately and confirm no owned processes/ports remain
6. Check user data survives an app-directory replacement and archive removal.
   Linux defaults to `~/.config/ChemApp/data/chemapp.db` (or the desktop session's
   XDG configuration directory); never store the database inside resources
7. Validate the minimum distro/library matrix, licenses and redistribution rights
   for every dependency/model, dependency scanning and the final archive hash

No changes to scientific algorithms or external-asset permissions are implied by
this packaging work. Review the security/data sections in README.md as well.
