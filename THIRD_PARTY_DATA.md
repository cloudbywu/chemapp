# Third-party data

## nmrshiftdb2

Some local spectral-index and model-building workflows can use the
nmrshiftdb2 database of organic structures and assigned NMR spectra.

Contains information from [nmrshiftdb2](https://www.nmrshiftdb.org/), which
is made available under the
[nmrshiftdb2 Database License](https://nmrshiftdb.nmr.uni-koeln.de/nmrshiftdbhtml/nmrshiftdb2datalicense.txt).

The database license applies to database copies and derivative databases.
It includes attribution, share-alike, derivative-database access, and
software-licensing conditions. In particular, section 4.5 requires software
that relies on the database for its functionality to use an
Open Source Initiative-approved license.

This notice records the upstream terms; it does not select or grant a
license for ChemApp itself. Before publicly distributing a database-backed
build, the project owner must select a compatible project license and
confirm compliance with the complete upstream license.

Generated indexes must retain their source snapshot, source URI, license URI,
version (when known), and SHA-256 digest. Do not mix datasets with incompatible
terms into a derivative index.

The operative upstream text was rechecked on 2026-07-24. The raw 26,584-byte
license response had SHA-256
`00322cc450c2e53e305c082e1ab7454f108b7df51c0dbee0a07932666b245516`
and advertised `Last-Modified: Fri, 19 Jun 2026 13:27:30 GMT`. The
machine-readable record is in
[`docs/data-governance-manifest.json`](docs/data-governance-manifest.json).
The remote hash is a drift detector, not a substitute for reading the current
complete upstream terms.

The present conservative release interpretation is:

- Section 2.4 leaves possible rights in individual database contents outside
  the database license; those record-level rights have not been cleared.
- Sections 4.2 and 4.3 require the database-license URI and, for public
  produced works, an associated nmrshiftdb2 notice. The root `NOTICE` and the
  prediction API/UI now carry that notice.
- Section 4.4 governs public use of derivative databases.
- Section 4.5 requires qualifying dependent software to use an
  OSI-approved software license.
- Section 4.7 can require a machine-readable derivative database or a complete
  alteration file/method when a derivative database or its produced work is
  publicly used.

Consequently, the current repository is valid for controlled internal
evaluation but is **not release-ready**. This is a compliance engineering
record, not legal advice.

## External proof-of-concept smoke set

An optional independently sourced interoperability set is frozen from
[Zenodo record 16881130](https://doi.org/10.5281/zenodo.16881130), published
under CC0-1.0 by Kenan Henzelin, Lucas Risse and Luc Patiny. Its eight files
(six sample ZIP archives, `README.md`, and `toc.json`) are listed with official
Zenodo MD5 values, observed SHA-256 values, exact byte counts and fixed URLs in
[`docs/external-smoke-dataset-manifest.json`](docs/external-smoke-dataset-manifest.json).

This is only a six-sample proof-of-concept smoke set. It may verify parsing,
modality handling and end-to-end reproducibility, but it is too small for a
headline accuracy, calibration or generalization claim. Its six exact
structures overlap the current v2 spectral index, so it is not an
unknown-structure or open-world benchmark.

Raw files live under the ignored
`backend/data/external/zenodo-16881130/` directory. The downloader does not
extract archives. It rejects HTML responses, size/hash mismatches, unsafe ZIP
paths, excessive entry/expanded-size/compression limits and CRC failures before
atomically publishing a file:

```powershell
cd backend
uv run python scripts/fetch_external_smoke_data.py
uv run python scripts/fetch_external_smoke_data.py --verify-only
```

Previously downloaded files can be frozen through the same checks without
network access:

```powershell
uv run python scripts/fetch_external_smoke_data.py `
  --source-dir "$env:TEMP\chemapp-external-nmr-audit"
```

After all eight files pass, the script creates a deterministic, immutable local
`inventory.json`. A conflicting existing inventory is never overwritten.

The follow-on
[`external NMR smoke v1`](docs/external-nmr-smoke-v1.md) conversion reads only
the pinned archive members, uses the common JCAMP decoder, and writes canonical
derived manifests under the ignored `backend/data/derived/` directory. Its
frozen status confirms 16/16 processed spectra decode numerically: 10
one-dimensional spectra can populate the `Spectrum` model, while 6
two-dimensional spectra remain matrix-only. This interoperability result does
not change the six-sample or 6/6 structure-overlap limitations.

```powershell
uv run python scripts/build_external_nmr_smoke_v1.py
uv run python scripts/build_external_nmr_smoke_v1.py --verify-only
```

## NMRexp human-reviewed calibration pool

An optional independently sourced, human-reviewed peak-annotation pool is
frozen from:

Wang, Jun-Jie; Zhu, Rong (2025), "NMRexp: A database of 3.37 million
experimental NMR spectra", Zenodo,
[https://doi.org/10.5281/zenodo.17296666](https://doi.org/10.5281/zenodo.17296666),
licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode).
The associated paper is
[https://doi.org/10.1038/s41597-025-06245-5](https://doi.org/10.1038/s41597-025-06245-5).

Only the six small `*_checked.csv` files are selected. Their fixed record ID,
file hashes, byte counts, license, version policy and source comparison are in
[`docs/independent-nmr-sources-v1.json`](docs/independent-nmr-sources-v1.json).
The downloader never resolves the concept DOI or a latest-version link and
does not fetch the multi-gigabyte automatically extracted corpus.

The six files contain 700 physical rows but only 500 content-unique reviewed
records because the 200 heteronuclear rows are repeated in per-nucleus files.
Derived data retain those aliases. Structure truth is always
`smiles_actual`; the automatically extracted `SMILES` field is provenance
only. The `right/wrong` fields grade extraction against the source PDF and
must not be represented as model probabilities or chemical class labels.

The pool contains literature-reported one-dimensional peak annotations, not
raw FIDs, dense traces, 2D spectra or atom-level peak assignments. Exact
nmrshiftdb2 overlap and grouped source/scaffold splitting are required before
calibration or evaluation. Full counts and commands are recorded in
[`docs/independent-nmr-data-v1.md`](docs/independent-nmr-data-v1.md).

## NMR-Solver manually curated paired-spectrum benchmark

The v4 research benchmark is frozen from:

Jin, Yongqi et al. (2025), "NMR-Solver: Automated Molecular Structure
Elucidation via Large-Scale Spectra Matching and Physics-Guided Fragment
Optimization", Zenodo,
[https://doi.org/10.5281/zenodo.16952024](https://doi.org/10.5281/zenodo.16952024),
licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode).
The associated article is
[https://doi.org/10.1038/s41467-026-71315-0](https://doi.org/10.1038/s41467-026-71315-0).

ChemApp reads only the hash-pinned `data/experiment/test.txt` member from the
fixed `data.zip` deposit. It does not extract the archive, open the bundled
LMDB databases, or deserialize their Python pickle values. The 450 records
contain paired 1H/13C literature reports and product structures. Ten lines
also append a different nucleus; the derived release retains the original
text and upstream-compatible values while exposing only the 6,340 actual 13C
shifts as scoring input.

The paper describes this collection as manually curated, but the deposit does
not provide a reviewer identity, review time, or source-document DOI for each
row. It is therefore recorded as an
`upstream_manually_curated_benchmark`, not as ChemApp double-reviewed data.
The same collection was used in the publication that introduced the v4
set-similarity equation, so it can evaluate the frozen ChemApp + DP5q
adaptation and candidate pools but cannot serve as method-independent
validation of NMR-Solver itself.

The exact source bytes, license, parsing invariants, DP5q overlap attestation,
release hashes, and limitations are recorded in
[`docs/nmrsolver-reviewed-source-v1.json`](docs/nmrsolver-reviewed-source-v1.json)
and
[`docs/nmrsolver-reviewed-data-v1.md`](docs/nmrsolver-reviewed-data-v1.md).
Redistributions must retain the generated `ATTRIBUTION.json`. The underlying
JACS Supporting Information sources were not mapped or individually
rights-audited.

## Optional DP5q runtime

The optional 13C forward-fit sidecar uses the upstream
[DP5 repository](https://github.com/ruslankotl/DP5) at commit
`b79968cf63cb282e8871d5595ea6cef5b4dc0d49`. The repository declares the MIT
License. ChemApp does not commit or redistribute that checkout or its model
files; `backend/scripts/setup_dp5q_sidecar.ps1` retrieves the pinned upstream
revision into a separate local directory and verifies the mean model,
preprocessor and 99-quantile archive hashes.

The released 13C mean model remains an optional, non-ranking shadow. The
99-quantile model now has a separate, default-off administrator diagnostic
endpoint after a real pinned-runtime golden reproduction. Unassigned peaks use
an explicitly non-official Hungarian shadow assignment, and neither path emits
a calibrated structure-correctness probability. The repository and dataset
declare MIT licenses, but there is no separate weight-specific license
declaration; model-weight redistribution and the underlying nmrshiftdb2
database/content rights remain release-review items. Exact assets and limits
are recorded in
[`docs/dp5q-upstream-assets-v1.json`](docs/dp5q-upstream-assets-v1.json) and
[`docs/dp5q-quantile-shadow-v1.md`](docs/dp5q-quantile-shadow-v1.md).

## Release gate check

项目软件许可证：**MIT**（仓库根 `LICENSE`，SPDX `MIT`，OSI 批准；决策记录
见 `docs/governance/decision-project-license-mit-v1.md`）。该许可证只覆盖
软件代码，不改变下方各数据源的独立许可义务。

Run the offline manifest check before any data-backed release:

```powershell
cd backend
uv run python scripts/check_data_governance.py --verify-assets --strict-release
```

Exit code `1` means an integrity failure. Exit code `2` means the inventory is
internally consistent but one or more known release blockers remain. The
optional `--verify-upstream-license` flag checks for byte-level drift at the
official license URI; network unavailability is reported as a warning rather
than silently changing the frozen record.

## CSP5 prediction package and Exp22K/NMRexp training data

The production ¹³C forward scorer vendors the `csp5` Python package
(version 0.2.18, MIT License; see `backend/vendor/csp5/LICENSE`) with its
bundled model weights (`CSP5-13C`, `CSP5-1H`, `CSP5q-13C`, `CSP5q-1H`).
Upstream: PyPI `csp5`, Goodman lab, "CSP5: Large-scale Neural Chemical Shift
Prediction from 2.5 Million Experimental NMR Spectra".

The underlying training data and additional checkpoints come from
[Zenodo record 19486118](https://doi.org/10.5281/zenodo.19486118)
(`CSP5_data.tar.gz`): Exp22K/DFT8K assigned entries and scaffold-DOI split
files, plus solvent fine-tuned checkpoints.  Downloaded locally under
`backend/data/external/csp5/` for evaluation and reproducibility.  The
record's reuse terms should be re-checked before any public redistribution of
the training data; the vendored package itself is MIT.

Independent external evaluation additionally used the human-reviewed NMRexp
peak-annotation pool from
[Zenodo record 17296666](https://doi.org/10.5281/zenodo.17296666)
(CC BY 4.0), already documented in the "NMRexp human-reviewed calibration
pool" section above.

## Official NMRexp full parquet (blind-test source)

The path-one blind test uses the complete official NMRexp parquet from
[Zenodo record 16809534](https://doi.org/10.5281/zenodo.16809534) (CC BY 4.0;
paper DOI 10.1038/s41597-025-06245-5):

- File: `NMRexp_10to24_1_0811.parquet`
- Bytes: 682,329,437
- SHA-256: `fc4500708e83bae9a2961d4c68f6359c24d60752402a75850731908c867b6f9b`
- Rows: 3,372,987
- Local copy: `backend/data/external/nmrexp-official/`

Usage: the blind-challenge builder
(`backend/scripts/build_nmr_blind_from_nmrexp_v1.py`) selects ¹³C records
whose normalized row keys are **not** present in the CSP5-bundled
`NMRexp_with_ids.parquet` (Zenodo 19486118), so the released challenge does
not re-use the upstream NMRexp benchmark component distributed by the CSP5
team (best effort).  See `docs/nmr-blind-nmrexp-v1.md` for the full protocol,
honesty boundary, and metrics.
