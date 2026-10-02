# ChemApp Electron desktop / 桌面版

Linux x64 standalone bundle workflow: [LINUX.md](LINUX.md).

Linux x64 packaging is implemented and Windows 10/11 x64 has a native build workflow. Existing browser/Docker workflows
are unchanged. This directory provides the desktop source and a reproducible
packaging workflow, **not a claim that a Windows installer has been built or
verified**. Linux cloud validation covers source checks, relocated bundled-runtime integration,
the production UI build and native Electron window interactions. Windows
install/uninstall, rendering, Job Objects and OS-signing remain unverified and
still require the checklist below.

## 开发运行 / Run from source

Requirements: Node.js >=22.12, npm, uv, Python 3.11+, and the backend's complete
locked dependency set. Run from the repository root:

```powershell
cd backend
uv sync --locked --group dev
cd ../frontend
npm ci
cd ../desktop
npm ci
npm run install:electron
npm start
```

Electron 44 installs its binary with the explicit `install-electron` command;
`npm run install:electron` exposes that command. `npm start` first builds the React
UI. For repeat launches of the current build, use `npm run start:built`.
Development defaults to `backend/.venv/Scripts/python.exe` on Windows and
`backend/.venv/bin/python` elsewhere. `CHEMAPP_DESKTOP_PYTHON` may override this
only for development; installed applications always use their bundled interpreter.
A working Linux display/session is needed to launch Electron on Linux. Never use
`--no-sandbox` or disable Chromium security to work around a host restriction.

No separate Vite or uvicorn terminal is needed. Startup can take time on the first
Torch/RDKit import; the loading window remains visible. If readiness fails within
120 seconds, a native error identifies the diagnostic log and the app exits.

## Runtime architecture and security

- The Electron main process launches one Python child using an argument array,
  with no shell interpolation. Python binds one IPv4 socket to `127.0.0.1:0`,
  then passes that same socket to uvicorn, avoiding port-selection races
- The built UI is served by that backend, making `/api` URLs same-origin.
  Desktop builds override `VITE_API_URL` and remove remote font requests.
  Normal web builds keep their existing configuration
- Main injects independent random, per-launch desktop/access/admin tokens only
  into requests belonging to its private session and exact window/origin. Tokens
  are not written to URLs, command arguments, renderer globals, or disk
- An outer backend gate checks token, Host, Origin and loopback peer even for
  static files and endpoints normally exempt in web mode. Readiness requires the
  per-launch identity header and a successful database check
- Renderer sandbox, context isolation and web security stay enabled; Node,
  webviews, permission grants, remote navigation, popups and external renderer
  network requests are disabled. There is **no preload or IPC bridge**
- The local owner has ordinary desktop administrator capabilities. Independent
  reviewer credentials and review-admin subject configuration are still required;
  no two-person review, provenance, calibration or probability gate is weakened
- Backend LLM requests remain subject to the existing frontend consent and
  backend host/auth checks. The desktop shell does not grant data-sharing consent
- Closing the window closes the owned stdin pipe. A bounded graceful shutdown is
  followed by owned-process-tree cleanup. A backend watchdog handles abrupt owner
  death; Windows uses a kill-on-close Job Object and Unix uses an owned process group
- Native export save dialogs preserve the existing blob downloads. The ChemApp
  menu also exports the current workspace to a user-chosen PNG with Ctrl+Shift+S;
  only this app window is captured, with no screen-wide access or external upload. The desktop
  session is in memory, so language/non-secret UI preferences reset on relaunch;
  credentials were already memory-only. Scientific records persist in SQLite.
  Window bounds/maximized state are remembered and clamped to the current display.
  Closing/reloading with dirty edits defaults to Keep editing; backend shutdown
  happens only after the window actually closes

The token gate prevents unrelated local websites/apps from reaching the API by
just discovering its port. It is not a security boundary against malware running
as the same OS user, which can inspect local processes/files. Keep the OS secured.

## 数据目录 / Data and diagnostics

Database and training lock live below Electron `app.getPath('userData')/data`:

- Windows default: `%APPDATA%/ChemApp/data/chemapp.db`
- Linux default: `~/.config/ChemApp/data/chemapp.db`
- Diagnostic log: `desktop-backend.log` in the parent directory, capped by rotation

The desktop uses its own database. It does not silently move or overwrite the
web/development database. To migrate an existing database, quit all writers and
make a consistent SQLite backup before copying it to the desktop data directory.
Do not copy only a live WAL database's `.db` file. Uninstall does not delete user
application data. Back up before upgrades; database downgrade compatibility is
not guaranteed.

## Windows x64 packaging

Build on a Windows x64 machine. Cross-packaging Linux Python into a Windows EXE
is deliberately rejected. Run after the source-development installation above:

```powershell
cd desktop
npm run stage:win
npm run pack:win
# Test release/win-unpacked/ChemApp.exe first, then:
npm run dist:win
```

To include the existing pinned/checksum-verified CSP5 weights during staging:

```powershell
python scripts/stage-runtime.py --with-csp5
```

Staging obtains uv-managed standalone CPython 3.11 with `--no-bin --no-registry`,
copies the relocatable distribution, installs the **full** `backend/uv.lock`
dependency set with hashes, and preserves the backend/source asset directory layout.
It does not copy an activated venv or depend on the end user's Python installation.
The installer contains React assets, source, the Python runtime and native wheels.
Torch, RDKit, SciPy and Transformers make this a **large** distribution; this is not
a slim or dependency-free build. Disk/download size depends on the locked Windows
wheels. Dependency imports are checked using the copied Python interpreter.

`build/runtime/manifest.json` records the dependency lock and source hashes.
Packaging rejects stale/mutated staged sources instead of combining an old backend
with a new UI. Move the old `desktop/build/runtime` aside before restaging; the
stager refuses to overwrite it. Versioned backend/app/vendor/scripts and examples,
plus the desktop launcher, are included. User databases, `.env`, private keys and
untracked model files are not harvested automatically. New intentional runtime
source files must be included explicitly in the staging source list.

Optional scientific assets stay optional and fail closed:

- CSP5 weights are included only with `--with-csp5`; the existing fetch script
  checks its pinned release and SHA256 manifest
- ForwardGNN/retrieval checkpoints and indexes, T5 weights, imported data and
  external DP5q Python/repository assets are not bundled by default
- Existing configuration variables may point to locally provisioned, appropriately
  licensed assets. Do not redistribute model/data files without checking their terms
- Missing assets remain visible through existing health/UI diagnostics. Packaging
  does not enable disabled scientific claims or convert diagnostic scores to
  calibrated probabilities

`pack:win` creates an unpacked directory; `dist:win` builds a per-user NSIS installer.
Both use `--publish never`. No auto-updater, automatic deployment or signing secrets
are configured. Public distribution needs an appropriate code-signing certificate,
license review and Windows validation; unsigned builds may trigger SmartScreen.

## Verification

```powershell
# From desktop/ after backend dependencies and frontend build are ready
npm run check
npm run test:integration
# From backend/
uv run --no-sync pytest tests/test_desktop_server.py -q
uv run --no-sync ruff check .
# Existing regression suite remains required
uv run --no-sync pytest tests -q
# From frontend/
npm run lint
npm test
npm run build
```

Windows acceptance checklist, still required before shipping:

1. Run packaged `ChemApp.exe` with no system Python/Node on PATH; exercise first
   launch, second-instance focus, slow startup and clean relaunch
2. Import/analyze a real example, manually edit/review, undo safely, compare,
   export CSV/PNG/DOCX and select save locations; verify Plotly/SMILES under the CSP
3. Verify no external fonts or renderer network; reviewer and LLM consent stay gated
4. Close while starting and during analysis; kill Electron and separately kill the
   backend, confirming owned children/ports stop and unrelated processes remain
5. Check optional-model diagnostics with/without trusted assets, and actual licensed
   asset locations in a read-only installation directory
6. Test NSIS install, upgrade and uninstall under a non-admin Windows account;
   confirm user data survives and logs/errors contain no launch tokens
7. Review/sign the resulting binary, scan dependencies, and verify installer hash

Primary references: [Electron security](https://www.electronjs.org/docs/latest/tutorial/security),
[WebRequest](https://www.electronjs.org/docs/latest/api/web-request),
[process sandboxing](https://www.electronjs.org/docs/latest/tutorial/sandbox),
[electron-builder Windows](https://www.electron.build/docs/win/),
[uv managed Python](https://docs.astral.sh/uv/concepts/python-versions/#managed-and-system-python-installations).

### Official NMR2Struct downloads

Settings → NMR2Struct model weights installs C-only, H-only, or H+C from the
immutable official MarklandGroup/NMR2Struct source. Progress/cancel/retry are
available; downloads continue when leaving Settings. Files are size/SHA-256
verified in a temporary file before atomic replacement and stored in the desktop
user-data `models/nmr2struct` directory, separate from the read-only application.
No model files need to be included in release packages. Reopening the app after an
interrupted download permits retry; installed models are recognized automatically.
The renderer never fetches third-party weight URLs or receives arbitrary file-write
access. Existing session-token, origin, CSP, and admin protections remain in force.
