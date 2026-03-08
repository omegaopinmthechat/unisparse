# unisparse

This repository contains MATLAB implementations of UniSparse estimators (UniLASSO, UniMCP, UniSCAD) and supporting utilities for simulation, cross-validation, and optimization (RMPSH).

## Quick start

1. Open MATLAB and change to the project root folder:

```matlab
cd 'path/to/project/root'
```

2. Initialize project paths (adds helper folders and propagates to workers):

```matlab
init_project
```

3. Run the demo script:

```matlab
function_test
```

## API / Functions

### `unisparse(X, y, lambda_range, nfolds, method, rmps_lb, rmps_ub, rmps_x0, rmps_options, a, gamma)`

Main entry point to run cross-validated UniSparse estimators. The function centralizes path setup so callers (e.g. `function_test`) can remain minimal.

Parameters
- `X` (numeric matrix): n x p design matrix.
- `y` (numeric vector): n x 1 response vector.
- `lambda_range` (numeric vector or 2-element vector): either a list of lambda values to evaluate or a two-element [min,max] range (default [1e-5,1e5], 50 log-spaced values).
- `nfolds` (scalar): number of CV folds. If set to 1, an 80/20 train-test split is used. Default: 2.
- `method` (string or cell array): `'unilasso'`, `'unimcp'`, `'uniscad'`, or `'all'` (default `'all'`). Can be a cell array to sweep multiple methods.
- `rmps_lb` (numeric vector): RMPS lower bounds for optimization variables (default: -100 for intercept, 0 for slopes).
- `rmps_ub` (numeric vector): RMPS upper bounds (default: 100 for all variables).
- `rmps_x0` (numeric vector): RMPS starting point. If empty, sensible defaults are computed inside `unisparse`.
- `rmps_options` (struct): Options passed to `RMPSH` optimizer. Defaults are set inside `unisparse` (DisplayUpdate=0, PrintSolution=0, MaxRuns=5, TolFun2=1e-6, cutoff=1e-6).
- `a` (numeric): SCAD concavity parameter (default 3.7).
- `gamma` (numeric): MCP concavity parameter (default 3.0).

Returns
- `results` (struct): contains fields for each method run: `lambda_grid`, `best_lambda`, `beta_hat` (final [beta0; beta]), `rmps_options`, and performance metrics. If a sweep was requested, `results.SWEEP` contains per-run outputs.

Example

```matlab
% generate data
[X, y, ~, ~, ~] = Generate_data_scenario_homecourt(120, 10);

% run unisparse cross-validation
results = unisparse(X, y, [1e-3, 1e1], 5, 'all');
```

### `init_project`

Convenience script that adds the project folders to MATLAB path and propagates them to any parallel workers. Use it once per session (or include it in startup).

Usage

```matlab
init_project
```

### `function_test`

Minimal demo script demonstrating how to call `unisparse`. It is intentionally kept short to show usage.

Usage

```matlab
function_test
```

### Support functions

- `supp funs/split_data.m`: builds index-based train/test splits (K-fold or 80/20).
- `Unisparse/unisparse_univreg.m`: internal helper for per-fold univariate regressions.
- `other methods/*`: alternative fitters and penalty functions (LASSO, MCP, SCAD implementations).
- `RMPSH/*`: optimization routines used by `unisparse`.

## Parallel & GPU notes

- `unisparse` will start a parallel pool if none exists (using the 'Processes' profile). If you prefer manual control, start a pool before calling `unisparse`.
- Helper folders are propagated to workers via `pctRunOnAll` in `unisparse` and `init_project`. If you start a pool before running `init_project`, restart or call `pctRunOnAll` manually.
- If a compatible GPU is present, `unisparse` will attempt to use `gpuArray` for prediction steps. Numerical results are computed on CPU for optimization.

## Troubleshooting

- Error: `Undefined function 'split_data'` — ensure you ran `init_project` or that `supp funs` is on your MATLAB path. If using parallel workers, restart the pool after adding paths.
- If you see parallel-worker path errors, run:

```matlab
delete(gcp('nocreate'));
init_project;
parpool('Processes');
```

Last updated: 2026-03-08
