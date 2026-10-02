# Changelog

## Unreleased — 2026-10-02 (latest-main integration)

- Integrate the prior Linux desktop and data-integrity work onto main
  `31edeec2224259fe58681c637060de600c1b4698`, preserving the new vendored ML code
  and reconciling the upstream weight-fetcher/CI additions with direct official downloads.
  Both dependency lockfiles already match the prior updates and are unchanged.
- Read reference records and cached normalized spectra from one deeply immutable
  snapshot, retaining strict length checks and avoiding concurrent-import mismatch.
- Carry the selected continuous proton spectrum through combined prediction,
  and retain generator/model/variant/input provenance in validated candidates.
- Resolve NMR2Struct checkpoint and input channels consistently; reject invalid
  required inputs and observable out-of-domain inputs before inference.
- Replace request-time global stdout redirection with per-call vendor verbosity.
- Bind NMR preview/apply/reset to the result revision the client actually saw,
  closing the pre-request manual-save window as well as the in-flight race.
  Legacy requests may omit this revision only when no current result exists.
- Preserve source-bound history, atomic AI action/undo, later-manual-edit guards,
  parser budgets, async UI protections, and floating-point golden integrity tests.
- Official trained checkpoint identity, loading, and real inference have been
  exercised for C-only, H-only and combined inputs. These are runtime/input
  compatibility checks, not scientific accuracy or probability calibration.
  Source submission excludes packaged applications, model weights, runtime
  binaries, databases, caches and credentials. It does not deploy the application.


## Unreleased — 2026-10-01

This section records the source changes prepared on 2026-10-01, before
the latest-main integration and official model-download work above.

### Reliability and review workflows

- Bind analysis, prediction, comparison, inference, and list responses to the
  request and spectrum revision that produced them. Late responses cannot
  replace a newer selection or silently clear its drafts.
- Respect newly unsaved edits when file uploads or example imports finish;
  prevent duplicate imports and stop queued work after the panel closes.
- Invalidate NMR previews when their inputs or source revisions change, and
  keep mutation controls locked while a write is in progress.
- Keep manual recalculation results attached to the correct range, peak, and
  HPLC channel. Preserve custom windows when peaks are inserted or deleted.
- Reject blank peak entries instead of converting empty fields to zero.
- Preserve successful batch refreshes after navigation while suppressing
  stale panel feedback.

### Data integrity and parser safety

- Commit a multi-spectrum upload as one transaction so partial failures do not
  leave hidden records that duplicate on retry.
- Bind new historical results to their source spectrum revision. Restore and
  AI undo reject results from another revision or from unverifiable legacy
  history. Legacy rows remain available for inspection; their source is never
  guessed or backfilled.
- Protect NMR processing/reset from clearing a manual result saved while the
  operation was running.
- Atomically persist each AI mutation, its exact persisted before-snapshot,
  after-version, and action history. Atomically restore and mark actions undone.
  Independent workers recheck result/source revisions inside the write
  transaction; scientific calculations run before acquiring its write lock.
- Bound Bruker dimensions, JCAMP decoded expansion, and HPLC aggregate
  channels/points before allocating large arrays.
- Match HPLC result sidecars by filename stem, including case-insensitive
  matches, without borrowing another sample's peak table.

### Interface and accessibility

- Name comparison selectors and announce loading/error states.
- Add an in-panel AI drawer Close control, Escape dismissal, and focus return.
- Keep editable NMR rows focused during numeric edits and explain unavailable
  historical restores in both Chinese and English.
- Allow comparison controls to fit narrow layouts.
- Support Arrow/Home/End navigation for NMR tabs, with Enter/Space activation
  through the existing unsaved-edit confirmation.

### Dependencies and scientific safeguards

- Update Axios, DOMPurify, brace-expansion, undici, and urllib3 to compatible
  releases addressing the dependency audit findings.
- Require consistent, finite calibration metadata before showing a conditional
  probability. Production probability and automatic-selection gates remain
  closed; no validation data, ranking algorithm, model artifact, or scientific
  accuracy claim was changed.
- Keep exact artifact integrity and same-runtime determinism checks while
  allowing the existing strict cross-platform floating-point tolerance in the
  optimizer golden test. Even a one-ULP unhashed parameter change is rejected.

### AI undo edit protection (2026-10-02)

- Block whole-result AI undo after a later manual save, historical restore, or
  reanalysis; preserve current results, immutable versions, and action status.
- Track a server-owned non-AI edit generation in the same write transaction.
  Consecutive AI undo remains available until a non-AI edit boundary is reached.
- Add idempotent, serialized schema migrations without reconstructing legacy
  history. Old actions remain visible; unverified undo fails closed.
- Return structured HTTP 409 reasons and show localized English/Chinese
  explanations while preserving unsaved drafts and failed-action previews.
- Cover both race orderings with independent SQLite connections, migration
  preservation/concurrent startup, sequential undo, metadata-independent
  checks, API behavior, and localized error handling.

### Verification and limitations

- Regression tests cover stale responses, interrupted navigation, duplicate
  actions, parser budgets, transaction rollback, source-bound history, and
  concurrent manual saves. AI regression tests inject failures at every write
  boundary and exercise same-revision commit/undo contention between separate
  processes and store instances.
- Linux native desktop visual QA covers the Settings model panel, verified
  bundled inventory, missing-file controls, official C-only/H-only downloads,
  progress, completion and navigation. Interrupted/retry/race cases additionally
  have automated backend and component coverage. Windows GUI was not exercised.
- Docker build verification was not run because Docker is unavailable here.
- The existing large Plotly chunk warning remains.

### Next engineering work

- Review heuristic cross-technique score terminology separately; no score
  formulas or labels have been changed in this batch.

### Official NMR2Struct weight installation

- Added bilingual Settings model management with sizes, verification, progress,
  cancellation/retry and status recovery on navigation
- Pinned three supported checkpoints to the official MarklandGroup/NMR2Struct
  commit; no release mirror, redirects, arbitrary URLs or API-selected paths
- Added per-user desktop-compatible storage, process/directory concurrency guards,
  streaming size/cooperative time-budget checks, SHA-256-before-atomic-publication, failed/cancelled
  download recovery, CLI parity and model-cache identity refresh
- Added deterministic interrupted, repeated, concurrent, slow-stream, checksum,
  auth, symlink and cache tests; CI restores official C-only weights for inference
