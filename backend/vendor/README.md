# Vendored third-party code

## csp5 0.2.18

The `csp5/` directory is an unmodified copy of the pure-Python source
distribution of the `csp5` package (PyPI `csp5==0.2.18`), including its
bundled model weights (`CSP5-13C`, `CSP5-1H`, `CSP5q-13C`, `CSP5q-1H`).

- Upstream source: https://pypi.org/project/csp5/
- Paper: "CSP5: Large-scale Neural Chemical Shift Prediction from 2.5 Million
  Experimental NMR Spectra" (Goodman lab; Zenodo record 19486118).
- License: MIT (see `csp5/LICENSE`).

The native C++ matching backends (`_native`) are not built on Windows and are
not needed for prediction.  The app imports only `csp5.api` /
`csp5.model_registry`; matching in the app uses SciPy's Hungarian solver.

Model weights (`*.pt`, about 73 MB) are ignored by the repository `.gitignore`
and therefore must be restored after a fresh clone.  They are part of the
`csp5==0.2.18` source distribution (PyPI) and are also on
[Zenodo record 19486118](https://doi.org/10.5281/zenodo.19486118).  The scorer
refuses to fetch weights silently; set `CSP5_ALLOW_DOWNLOAD=1` only when an
on-demand download is explicitly acceptable.

To enable solvent-specific 13C fine-tuned checkpoints, copy the matching
`CSP5-13C-*.pt` files from the Zenodo record into
`csp5/models/<model-name>/best_model.pt` or pre-populate the model cache
directory referenced by `CSP5_MODEL_CACHE_DIR`.
