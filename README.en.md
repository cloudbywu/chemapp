# ChemApp

🌐 **Language / 语言:** [English](README.en.md) · [简体中文](README.md)

ChemApp is a local-first platform for parsing instrument data, processing
spectra, and reviewing evidence. The project includes a FastAPI backend and a
React/TypeScript frontend, with a Chinese/English interface, interactive
spectra, human review, result versioning, cross-spectrum comparison, report
export, and experimental NMR structure candidate ranking.

> Structural suggestions are candidate rankings, not compound
> identification. Final conclusions must still be confirmed with the original
> spectra, integrals, multiplicities, 2D NMR, mass spectra, reference
> standards, or other independent evidence.

## Supported data

| Technique | Input formats | Main capabilities |
| --- | --- | --- |
| ¹H/¹³C NMR | JEOL Delta `.jdf`, Bruker directory or `.zip` | Complex FID, digital-filter compensation, FFT, phase/baseline, calibration, smoothing, cropping, integration, and multiplet analysis |
| UV-Vis | `.csv`, `.txt` | Peak detection, λmax, calibration curves |
| Fluorescence | JCAMP-DX `.dx` | Excitation/emission peaks and Stokes shift |
| XRD | `.asc`, `.ras` | d-spacing and Scherrer crystallite size |
| HPLC | OpenLab `.dx`, or ZIP containing `.dx` plus `.rx`/`.acaml` | Multi-channel parsing, peak areas, retention times, batch comparison |
| Electrochemistry | CHI text | CV/EIS metrics |

The [`dataexample`](./dataexample) directory in the repository can be used for
local validation and contains JEOL JDF and Bruker NMR examples.

## Running locally

Requirements:

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- Node.js 22+ and npm

Install backend dependencies for the first time:

```powershell
cd backend
uv sync --locked --group dev
```

Start the backend:

```powershell
cd backend
uv run python dev_server.py
```

In another terminal, install and start the frontend:

```powershell
cd frontend
npm ci
npm run dev
```

Then open <http://127.0.0.1:3000>. API documentation is available at
<http://127.0.0.1:8000/docs>.

The development server listens only on the loopback interface by default.
When no token is configured, only local requests can access the API and
administrator operations.

## Configuration and security

Copy the sample configuration:

```powershell
Copy-Item .env.example .env
```

Important variables:

| Variable | Purpose |
| --- | --- |
| `CHEMAPP_DB_PATH` | SQLite database location |
| `CHEMAPP_ACCESS_TOKEN` | Regular API access token |
| `CHEMAPP_ADMIN_TOKEN` | Separate administrator token for deletion, training, and model import |
| `CHEMAPP_AI_PREVIEW_SECRET` | Shared signing key for destructive-AI preview tokens across all backend workers; set explicitly in production |
| `CHEMAPP_AI_PREVIEW_TTL_SECONDS` | AI preview authorization-token lifetime in seconds (default 300, maximum 3600) |
| `CHEMAPP_REVIEW_ADMIN_SUBJECT` | Stable administrator pseudonym for review audit chain writes; must not collide with any reviewer subject |
| `CHEMAPP_REVIEWER_TOKENS` | JSON mapping of server-side reviewer identities to separate high-entropy tokens; review writes are disabled when not configured |
| `CHEMAPP_REVIEW_ALLOWED_LICENSES` | SPDX license whitelist allowed into the Gold review workflow |
| `CHEMAPP_LOCAL_ACCESS_BYPASS` | Whether token-free local access is allowed |
| `CHEMAPP_LOCAL_ADMIN_BYPASS` | Whether token-free local privileged operations are allowed |
| `CHEMAPP_CORS_ORIGINS` | Comma-separated browser origins allowed |
| `CHEMAPP_LLM_ALLOWED_HOSTS` | Whitelist of external LLM hosts that may be connected |
| `CHEMAPP_NMR_INDEX` | NMR candidate retrieval index |
| `CHEMAPP_NMR_INDEX_V2` | Per-spectrum traceable NMR v2 audit index (read-only mount in compose; not yet used directly for production ranking) |
| `CHEMAPP_NMR_RANKER` | NMR ranker file |
| `CHEMAPP_T5_MODEL_DIR` | Experimental SMILES generation model directory |
| `CHEMAPP_DP5Q_MODE` | Optional DP5q ¹³C forward diagnostics; accepts only `off` or `shadow` |
| `CHEMAPP_DP5Q_QUANTILE_MODE` | Administrator read-only diagnostics for the official 99th-percentile model; independent switch, default `off` |
| `CHEMAPP_DP5Q_PYTHON` / `CHEMAPP_DP5Q_REPO` | Isolated Python and pinned DP5 repository paths |

The ChemApp access token, administrator token, spectrum reviewer tokens, and
external AI API keys configured in the frontend are kept only in the memory of
the current page; they must be re-entered after a refresh. External AI features
ask the user to confirm before sending data.

For remote deployment, you must:

1. Set different strong random values for the access token and administrator token.
2. Set both local-bypass variables to `0`.
3. Use HTTPS and a trusted reverse proxy.
4. Store the database, original spectra, models, and keys outside the source tree and back them up separately.
5. Point `CHEMAPP_NMR_INDEX_V2` at a mounted NMR v2 audit index (compose ships a read-only mount).

Every worker in a multi-worker deployment must receive the same stable,
high-entropy `CHEMAPP_AI_PREVIEW_SECRET`. When it is absent, the server derives
the key from the access or administrator token; rotating either source key
immediately invalidates previews that have not yet been executed. The
process-random fallback is suitable only for single-process local development
and tests.

Spectrum-level Gold human review uses two distinct server-side authenticated
subjects, immutable spectrum/result hashes, conflict adjudication, and
per-event auditing. This ledger may only serve as a calibration candidate pool;
it cannot bypass molecular/scaffold/source leakage checks to act as an
independent test set.

See [`SECURITY.md`](./SECURITY.md) and
[`THIRD_PARTY_DATA.md`](./THIRD_PARTY_DATA.md) for more details.

## ML modules and NMR structure candidate ranking

The repository ships a set of ML modules for NMR structure candidate ranking:

- `backend/app/ml/forward_v1/`: a runnable ¹³C atom-level GNN forward
  prototype (pure PyTorch, no torch_geometric dependency) plus
  `csp5_scorer.py` (vendored CSP5q-13C scorer; the `csp5` package is in
  `backend/vendor/csp5/`, MIT);
- **Production path**: `backend/app/ml/forward_v1/` (CSP5q-13C and local GNN
  forward), `nmr_hybrid_predictor.py` (hybrid ranking plus a policy-gated
  conditional-probability execution path), `nmr_candidate_generation_v1/v2.py`,
  `nmr_evidence.py`, `nmr_structure_elucidation.py`,
  `app/ml/calibration/` (frozen calibrators and policy), and
  `nmr_blind_challenge.py` (blind-test protocol toolchain);
- **Research path**: `backend/research/` (calibration training/cases,
  baselines, benchmarks, independent data) and the frozen v4/v5 research
  releases in `app/ml` (`nmr_calibration_v4/v5.py`,
  `nmr_v4_experiment.py`, `nmr_phase6_independent_test.py`, etc.).

Note: the production gates for conditional-probability calibration and
automatic structure selection stay closed (fail-closed); the associated
research-governance artifacts (evaluation reports, model cards, release
manifests, leakage audits, blind-test round records, and external-holder
handoff packages) are not part of this public repository.

Production forward model measurements (2026-08-06):

- Exp22K official scaffold-DOI test (5,188 molecules): ¹³C MAE **0.59 ppm**, RMSE 1.05, q10–q90 coverage 91.5%;
- NMRexp human-reviewed strict ¹³C subset (132 records, disjoint from the local nmrshiftdb2 index): mean spectrum-level MAE **0.75 ppm**, with 93.1% of records ≤ 2.0 ppm. The CSP5 split audit maps 130 records to the official parquet and places 49 in scaffold-DOI `train`, so this is **not independent of CSP5 forward-model training** and is only a retrospective in-domain diagnostic.

Deployment decisions (2026-08-07, confirmed and implemented):

- **CSP5 enabled by default**: Dockerfile/compose set `CHEMAPP_CSP5_MODE=on`; local default is `auto` (use weights if present, otherwise warn and fall back);
- **Weights baked into the image**: `COPY backend/vendor ./vendor` includes CSP5 weights (~73 MB), and `.dockerignore` no longer excludes `vendor/**/*.pt`; on a clean checkout the Docker build stage automatically downloads the pinned PyPI sdist and verifies SHA-256 hashes (`scripts/fetch_csp5_weights.py`), which can also be run locally to restore them;
- **GPU-first with CPU fallback**: the scorer automatically selects CUDA/CPU without an extra switch;
- **Single worker**: uvicorn `--workers 1`, with scorer caching per process;
- **Fail-closed startup hash verification**: `backend/vendor/csp5/weights-manifest.json` pins SHA-256 hashes for the 4 weights; when `CHEMAPP_CSP5_MODE=on`, verification failure makes the app refuse to start, while `auto` only warns and falls back (see `app/ml/deployment_check.py`).

Candidate generation and end-to-end (2026-08-06):

- Hybrid generation (local index + PubChem molecular formulas) improved exact coverage on the 450-record NMR-Solver retrospective set from 1.8% to **38.0%** (connectivity 46.4%);
- End-to-end ranking (v2 count-aware pre-ranking): overall Top-1 23.5%, MRR 0.246; Top-1 **87.1%** when the truth is in the pool; calibration gate ECE 0.153 exceeded and remains fail-closed.

Blind tests and closed-pool evaluation:

- The repository includes a complete blind-test toolchain: `nmr_blind_challenge.py` supports role separation, Gold commitments, Ed25519-signed release, and per-event auditing; the accompanying spectrum-level Gold human-review flow is described under "Configuration and security" above.
- Multiple rounds of retrospective closed-pool evaluation (v1–v7, each 100–500 ¹³C spectra) have been run against historical data: Top-1 is around **85–87%** when the truth is inside the candidate pool (noticeably lower in the overall metric that includes out-of-pool truths). Because every round's selection overlaps CSP5 training data and no real externally held data exists, none of these results constitutes independent-test accuracy; they demonstrate engineering stability under one retrospective protocol only.
- The calibrated-probability execution path is wired, but its production gate remains closed: the hybrid predictor/API/frontend have a conditional Top-1 field, while `probability_claim_allowed=false` forbids treating or displaying it as production calibrated confidence. Automatic structure selection also remains off until a new training-disjoint holdout is run by a genuinely independent external holder.

Prototype smoke training (CPU, about 1 minute):

```powershell
cd backend
.venv\Scripts\python.exe -m app.ml.forward_v1.train `
  --max-molecules 5000 --epochs 60 --outdir reports/forward_v1_smoke
```

## Docker Compose

Copy `.env.example` to `.env`, replace the access and administrator tokens
(and the reviewer tokens if spectrum review is enabled), then run:

```powershell
docker compose up --build
```

Open <http://127.0.0.1:3000>. Compose exposes only the frontend to the local
machine; the backend is reverse-proxied by the frontend container. Before first
start, confirm the following model files exist, or remove the corresponding
read-only mounts as needed:

- `backend/data/nmr_spectral_index.sqlite`
- `backend/data/nmr_spectral_index_v2.sqlite`
- `backend/data/nmr_joint_ranker.joblib`
- `backend/app/ml/pretrained/t5_nmr`


### Local container testing (wslc)

Container testing for this repository uses the WSL built-in `wslc` (requires WSL
2.9.3+; on this machine the binary is at `C:\Program Files\WSL\wslc.exe`)
instead of Docker Desktop. Run `wslc settings reset` once to initialize settings.

Build and start the backend:

```powershell
wslc build -t chemapp-backend:test -f backend/Dockerfile .
wslc run --rm -d --name chemapp-backend-test -p 127.0.0.1:8001:8000 `
  -e CHEMAPP_ACCESS_TOKEN=test-access -e CHEMAPP_ADMIN_TOKEN=test-admin `
  -e 'CHEMAPP_REVIEWER_TOKENS={"reviewer1":"test-reviewer-token"}' -e CHEMAPP_CSP5_MODE=on `
  -v "$PWD\backend\data\nmr_spectral_index.sqlite:/var/lib/chemapp-models/nmr_spectral_index.sqlite:ro" `
  -v "$PWD\backend\data\nmr_spectral_index_v2.sqlite:/var/lib/chemapp-models/nmr_spectral_index_v2.sqlite:ro" `
  -v "$PWD\backend\data\nmr_joint_ranker.joblib:/var/lib/chemapp-models/nmr_joint_ranker.joblib:ro" `
  -v "$PWD\backend\app\ml\pretrained\t5_nmr:/var/lib/chemapp-models/t5_nmr:ro" `
  chemapp-backend:test
```

Verify:

```powershell
curl.exe http://127.0.0.1:8001/api/ready
curl.exe -H "X-ChemApp-Access-Token: test-access" http://127.0.0.1:8001/api/health
```

- `/api/health` without a token must return 401; with the token it should report
  `csp5_weights.status == "ok"` and `nmr_index_v2.exists == true`.
- If docker.io is unreachable, pull via a mirror first and tag it locally:
  `wslc pull docker.m.daocloud.io/library/python:3.11-slim` then
  `wslc tag docker.m.daocloud.io/library/python:3.11-slim python:3.11-slim`.
- If a proxy is required, enable `networkingMode=mirrored` and `autoProxy=true`
  in `%USERPROFILE%\.wslconfig` and restart with `wsl --shutdown`.

## NMR processing principles

- Imports preserve immutable processing sources, source hashes, and complex quadrature data (when the instrument file provides them).
- "Preview" does not write to the database; "Apply" requires a matching `spectrum_revision` and produces a new revision number.
- "Reset" replays from the immutable source to avoid cumulative distortion from repeated processing.
- Phase correction requires complex quadrature data. Data historically saved with only real values explicitly refuses phase operations rather than fabricating a correction.
- Peak detection, integration, and display preserve positive and negative signals; metrics use signed data and no longer silently truncate negative peaks.
- ¹H automatic grouping first removes labeled solvent lines peak by peak and retains isolated singlets with S/N ≥ 8; automatic labels carry source and S/N audit fields and do not impersonate manual peak assignments.

## Structure candidate elucidation

Current pipeline:

1. Strictly parse and normalize molecular formulas and compute DBE.
2. Clean solvent/TMS signals, merge overly close spectral lines, and preserve integrals, multiplicities, and assignment information.
3. Filter database candidates by molecular formula first, then rank with ¹H/¹³C one-to-one peak matching.
4. Only rankers trained with Bemis–Murcko scaffold-grouped validation are enabled; incompatible older models are automatically disabled.
5. T5-generated results are listed separately as "experimental hypotheses", must pass RDKit structure and molecular formula validation, and do not add points to database candidates.

Responses distinguish `ranking_score`, evidence level, preprocessing records,
and warnings, and never misrepresent ranking scores as calibrated
probabilities. Mixture analysis abstains when there is not enough independent
evidence.

The traceable v2 audit index stores experimental conditions, source snapshots,
licenses, and import decisions per individual spectrum and does not merge
different experimental conditions for the same molecule into one pseudo-spectrum;
source hashes and snapshot bindings are verified before writing, and a mismatch
fails. Records that lack an explicit computation-method tag are separately
labeled `inferred_measured`; such records may enter exploratory evaluation only
and must not be mixed into reports with explicitly labeled `measured` test sets.

Benchmark evaluation requires models, calibrators, run configurations, and
de-indexing proofs to be pre-registered with hash binding before they enter the
evaluation flow (tooling lives in `backend/research/`). Records without explicit
human review can serve only as protocol smoke tests and exploratory baselines,
not production accuracy evidence.

### Optional DP5q ¹³C shadow evaluation

DP5q is incompatible with the main backend's NumPy/TensorFlow dependencies, so
it must run in an isolated environment: `CHEMAPP_DP5Q_PYTHON` and
`CHEMAPP_DP5Q_REPO` point at the isolated Python and the pinned DP5 repository,
with the upstream commit, models, preprocessor, and dependency versions all
pinned. That isolated environment is not an OS-level sandbox; production
deployment should still add no-network, read-only filesystem, and privilege
reduction policies. The mean path still runs only when
`CHEMAPP_DP5Q_MODE=shadow`; the default is `off`, and protocol or ranking
behavior has not changed because the quantile model was added.

The 99th-percentile model has its own switch
`CHEMAPP_DP5Q_QUANTILE_MODE=shadow` and runs only on explicit candidates via the
administrator endpoint `POST /api/ml/elucidate/dp5q/quantile-shadow`, with at
most 3 candidates by default. Because current experimental peaks do not provide
reliable atom assignments, the endpoint uses q50 Hungarian matching; this is
not official assignment semantics, and results are only marked as uncalibrated
diagnostic scores that do not enter formal ranking. The endpoint returns only a
compressed summary, not the full 99th-percentile tensor.

Research evaluation history: NMR structure candidate ranking went through
several research stages — the v3 pre-registered NMRexp run, the v4 NMR-Solver
¹³C Gaussian-ensemble similarity scorer, the v5 multi-evidence ranker, the v8
data-isolation audit and nested grouped evaluation, and the v9 applicability
signature with a research-only calibration method-comparison harness. The v8
ranker did not beat the fixed references in a development-only nested grouped
evaluation, so it is marked `rejected_development_noninferiority`, is refused
by default, and did not replace the production model; the calibration domain
stays fail-closed (zero calibration records). The full reports, release
manifests, leakage audits, and blind-test round records from these stages are
research-governance artifacts and are not part of this public repository. Data
governance and isolation checks run at import time from modules in `app/ml`;
nmrXiv intake is metadata-first and resolves the effective licence per study —
no record can enter calibration without an allowed licence, spectrum-file
hashes, and two-person review.

## Data integrity

- SQLite uses WAL, foreign keys, and busy waiting; result-version writes and current-result updates are in the same transaction.
- Spectra and results each carry revision numbers. Stale writes return HTTP 409 to prevent silent overwrites from multiple tabs.
- Results already manually confirmed are not overwritten by ordinary re-analysis; the UI requires explicit confirmation.
- ZIP, DOCX, and HPLC packages enforce checks on file count, total uncompressed size, compression ratio, paths, and XML safety.
- Exported CSV escapes cells that could trigger spreadsheet formula execution.

Back up the running database before upgrading. `backend/data/chemapp.db` in the
repository is runtime data and should not be treated as a rollbackable source
file.

## Tests

```powershell
cd backend
uv run ruff check .
uv run pytest tests -q
uv lock --check

cd ..\frontend
npm run lint
npm test
npm run build
npm audit --omit=dev
```

Continuous integration configuration is in
[`.github/workflows/ci.yml`](./.github/workflows/ci.yml).

## Main API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/live`, `/api/ready`, `/api/health` | Liveness, readiness, and detailed health status |
| `POST` | `/api/upload` | Upload instrument data |
| `GET` | `/api/spectra/{id}` | Read a spectrum; does not implicitly run analysis |
| `POST` | `/api/analyze/{id}` | Explicitly run analysis |
| `GET` | `/api/results/{id}` | Read existing analysis; does not implicitly generate results |
| `PUT` | `/api/results/{id}/manual` | Save manual review result |
| `POST` | `/api/nmr/{id}/process` | Preview or apply NMR processing |
| `POST` | `/api/nmr/{id}/reset` | Reset from the raw processing source |
| `POST` | `/api/ml/elucidate/predict` | NMR candidate ranking |
| `POST` | `/api/ml/elucidate/predict/combined` | Rank using saved ¹H/¹³C results |
| `POST` | `/api/ml/elucidate/dp5q/quantile-shadow` | Administrator read-only 99th-percentile candidate diagnostics; does not participate in ranking |
| `POST` | `/api/inference` | Cross-technique read-only inference |
| `POST` | `/api/reports/{format}` | Export report |

API input models reject unknown fields by default and validate ID counts,
titles, numeric ranges, and revision numbers.

## License

This project is released under the MIT license; see [`LICENSE`](./LICENSE) in
the repository root. Attribution requirements for third-party components and
data are in [`NOTICE`](./NOTICE) and
[`THIRD_PARTY_DATA.md`](./THIRD_PARTY_DATA.md).
