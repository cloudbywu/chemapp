# CSP5

`CSP5` is a pip-installable NMR predictor package with:
- batched `13C` and `1H` prediction
- prediction from precomputed geometries
- shift matching utilities with `dp` (default), `scipy`, and `murty` (k-best)

Bundled defaults:
- 13C model: `CSP5-13C` (`model_id`: `csp5-13c`)
- 1H model: `CSP5-1H` (`model_id`: `csp5-1h`)

Bundled quantile models:
- 13C quantile model: `CSP5q-13C` (`model_id`: `csp5q-13c`)
- 1H quantile model: `CSP5q-1H` (`model_id`: `csp5q-1h`)

## Install

Requires Python 3.9 or newer.
The current release is supported on Linux. Installing from the source
distribution requires working C and C++ compilers for the native matching
backends.

```bash
pip install CSP5
```

## Prediction CLI

In interactive terminals, `csp5` prints status lines to `stderr` before
and after prediction. If a run is slow, it prints an additional note that first
invocation can take longer while dependencies and model weights initialize, plus
periodic "still working" updates during long runs. Use `--no-status` to silence
them.

### From SMILES

```bash
csp5 --smiles "CCO" --nucleus 1H
csp5 --smiles "CCO" --nucleus both
csp5 --smiles-file smiles.txt --nucleus 13C --batch-size 64
csp5 --smiles "CCO" --nucleus 13C --num-conformers 8
csp5 --smiles "CCO" --nucleus both --num-conformers 8 --output-conformers-json cco_conformers.json
csp5 --smiles "CCO" --nucleus 13C --num-conformers 8 --output-conformers-sdf cco_conformers.sdf
csp5 --smiles "CCO" --nucleus 13C --output-svg cco_13c.svg
csp5 --smiles "CCO" --nucleus 13C --output-svg cco_13c.svg --svg-bond-length 72 --svg-shift-font-scale 1.1
csp5 --smiles "CCO" --nucleus 13C --model-name CSP5q-13C
csp5 --smiles "CCO" --nucleus 1H --model-name CSP5q-1H
```

### From molecule files (molfile or SDF)

By default, molecule-file input uses the coordinates embedded in the file. Add
`--regenerate-geometry` to keep the input atom order/numbering while generating
fresh ETKDG + MMFF/UFF coordinates for prediction.

```bash
csp5 --molecule-file input.mol --nucleus 13C
csp5 --molecule-file input.sdf --nucleus 1H --regenerate-geometry
```

### From precomputed geometries (parquet structures dataset)

Input dataset requirements:
- required columns: `smiles`, `molblock`
- optional columns: `conformer_rank`, `conformer_id`, `energy`, `energy_method`

Predict only rank-0 conformers:

```bash
csp5 \
  --structures-path /path/to/structures.parquet \
  --conformer-rank 0 \
  --nucleus 1H \
  --batch-size 64
```

Predict using all conformers in the dataset:

```bash
csp5 \
  --structures-path /path/to/structures.parquet \
  --use-all-conformers \
  --nucleus 13C
```

## Prediction Python API

```python
from csp5 import draw_prediction, predict_molecule_file, predict_smiles, predict_structures, predict_sdf

# Standard SMILES mode
res = predict_smiles(["CCO", "c1ccccc1"], nucleus="1H", batch_size=32)
print(res.predictions.head())
svg = draw_prediction(res)

# Precomputed-geometry parquet mode
res2 = predict_structures(
    "/path/to/structures.parquet",
    nucleus="1H",
    conformer_rank=0,
    use_all_conformers=False,
)

# Precomputed-geometry SDF mode
res3 = predict_sdf("/path/to/embedded.sdf", nucleus="13C")

# Molfile/SDF mode with fresh generated geometry while preserving atom order
res4 = predict_molecule_file("/path/to/input.mol", nucleus="13C", regenerate_geometry=True)

# Quantile models for uncertainty quantification
res5 = predict_smiles(["CCO"], nucleus="13C", model_name="CSP5q-13C")
print(res5.predictions[["atom_index", "shift_ppm", "shift_q05_ppm", "shift_q50_ppm", "shift_q95_ppm"]])
```

## Matching CLI

`csp5-match` expects one shift per line in each file.

### Default fast path (`dp`)

```bash
csp5-match \
  --predicted-file predicted.txt \
  --experimental-file experimental.txt \
  --solver dp
```

### SciPy Hungarian option

```bash
csp5-match \
  --predicted-file predicted.txt \
  --experimental-file experimental.txt \
  --solver scipy
```

### Murty k-best option

```bash
csp5-match \
  --predicted-file predicted.txt \
  --experimental-file experimental.txt \
  --solver murty \
  --k-best-policy clip \
  --k-best 25 \
  --temperature 0.5 \
  --mae-delta-threshold 0.2
```

## Matching Python API

```python
from csp5 import match_shifts

pred = [7.35, 7.30, 1.25]
exp = [7.34, 7.31, 1.20]

# DP (default)
r1 = match_shifts(pred, exp, solver="dp")

# SciPy Hungarian
r2 = match_shifts(pred, exp, solver="scipy")

# Murty k-best
r3 = match_shifts(pred, exp, solver="murty", k_best=10, k_best_policy="clip")
print(r3.assignment_entropy, r3.num_competing_assignments)
```

## Solver Notes

- `dp` is the default and is intended for the standard 1D shift objective.
- `scipy` uses Hungarian assignment on the full padded cost matrix.
- `murty` is the k-best solver; use this when you need assignment ambiguity analysis.
- For `murty`, `k_best_policy="clip"` (default) returns all feasible unique assignments
  when `k_best` is larger than what exists. Use `k_best_policy="strict"` to fail instead.
- `dp` and `scipy` are top-1 only (`k_best` must be `1`).

## Output Notes

- Prediction failures are returned explicitly (`failures`) with reason tags.
- Prediction output always includes `nucleus`, `model_id`, and `model_name`.
- Quantile models are selected explicitly with `model_name="CSP5q-13C"` or
  `model_name="CSP5q-1H"`. Their `shift_ppm` value is the median prediction
  (`shift_q50_ppm`) and the output also includes `shift_q01_ppm` through
  `shift_q99_ppm`, plus `shift_std_ppm` estimated from the calibrated q10-q90
  interval. The bundled scale is calibrated on held-out NMRexp-like data.
  Coverage is not guaranteed for rare or out-of-distribution chemistry, so
  these intervals should not be interpreted as a general OOD safety bound.
- For structures-mode predictions, conformer metadata columns are propagated when available.
- CLI JSON is molecule-oriented, with top-level model metadata, per-molecule
  prediction lists, and atom-map numbers matching `mapped_smiles_explicit_h`.
- Use `--nucleus both` to write 13C and 1H predictions in one JSON, grouped by
  nucleus under each molecule's `predictions`.
- In SMILES mode, `--num-conformers N` predicts generated conformers and returns
  Boltzmann-averaged shifts at 298.15 K (`--boltzmann-temperature-k` changes the
  temperature). The default remains one conformer.
- In structures mode, `--use-all-conformers` also returns Boltzmann-averaged
  shifts. Use `--output-conformers-json` to save individual conformer
  predictions separately.
- Use `--output-conformers-sdf` to save the exact conformer geometry or
  geometries used for prediction.
- Use `--molecule-file path.mol` or `--molecule-file path.sdf` for molfile/SDF
  input. Add `--regenerate-geometry` to discard embedded coordinates and create
  fresh geometry without changing the input atom order used for atom maps.
- Use `--output-svg path.svg` or `draw_prediction(result)` to create an
  RDKit-native SVG drawing with atom labels (`C4`, `H9`) and shift notes. SVGs
  auto-size by default. Use both `--svg-width` and `--svg-height` to force a
  fixed canvas; tune with `--svg-bond-length`, `--svg-atom-font-size`,
  `--svg-shift-font-scale`, and `--svg-padding`.
- When drawing quantile-model predictions, each shift note includes the median
  followed by `shift_std_ppm`, for example `18.35 +/- 1.24`.
