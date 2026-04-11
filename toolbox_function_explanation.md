# UniSparse Toolbox Function Reference

This document provides a concise API-style reference for all public functions listed in the toolbox inventory.

## Conventions

- `X`: design matrix (`n x p`)
- `y`: response vector (`n x 1`)
- Coefficient vectors are usually ordered as `[beta0; beta]` (intercept first)
- `lambda`: regularization strength

## Quick Start

```matlab
% Example workflow
[X, y, beta0_true, beta_true, Sigma] = Generate_data_scenario_homecourt(120, 10);

results = unisparse(X, y, [1e-3, 1e1], 5, 'all', [], [], [], [], 3.7, 3.0);

disp(results.UNILASSO.beta);
disp(results.UNIMCP.beta);
disp(results.UNISCAD.beta);
```

---

## 1) Core UniSparse (folder: `Unisparse`)

### `unisparse`

**Signature**
```matlab
results = unisparse(X, y, lambda_grid, nfolds, method, x0, rmps_lb, rmps_ub, rmps_options, a, gamma)
```

**Purpose**
Runs cross-validated UniLASSO / UniMCP / UniSCAD and returns fitted coefficients.

**Parameters**
- `X`, `y`: training data.
- `lambda_grid`: lambda vector, or `[min max]` range (log-spaced grid generated internally).
- `nfolds` (default `2`): CV folds (`1` gives an 80/20 split via `split_data`).
- `method` (default `'all'`): `'unilasso'`, `'unimcp'`, `'uniscad'`, or `'all'`.
- `x0`: RMPS initial point (`[theta0; theta]`).
- `rmps_lb`, `rmps_ub`: optimizer bounds.
- `rmps_options`: options struct passed to `RMPSH`.
- `a` (default `3.7`): SCAD concavity.
- `gamma` (default `3.0`): MCP concavity.

**Returns**
- `results`: struct with fields such as `UNILASSO`, `UNIMCP`, `UNISCAD`.
- Each method field includes `lambda` and `beta`; MCP/SCAD also include `gamma`/`a`.

**Example**
```matlab
results = unisparse(X, y, logspace(-4, 2, 20), 5, 'unimcp', [], [], [], [], 3.7, 3.0);
beta_unimcp = results.UNIMCP.beta;
```

### `unisparse_univreg`

**Signature**
```matlab
[beta0, beta, beta0_loo, beta_loo, eta_loo] = unisparse_univreg(X, y)
```

**Purpose**
Computes per-feature univariate regressions and efficient leave-one-out (LOO) estimates.

**Returns**
- `beta0`, `beta`: full-sample univariate intercept/slope vectors (`p x 1`).
- `beta0_loo`, `beta_loo`, `eta_loo`: LOO matrices (`n x p`).

**Example**
```matlab
[b0, b, ~, ~, eta_loo] = unisparse_univreg(X, y);
```

### `unilasso_objective_given_eta_loo`

**Signature**
```matlab
obj = unilasso_objective_given_eta_loo(theta, eta_loo, y, lambda)
```

**Purpose**
UniLASSO objective: mean squared error plus L1 penalty on slope components of `theta`.

**Parameters**
- `theta`: `[theta0; theta_j]`.
- `eta_loo`: LOO design-like matrix from `unisparse_univreg`.
- `y`: response.
- `lambda`: L1 penalty level.

**Example**
```matlab
objFun = @(psi) unilasso_objective_given_eta_loo(psi, eta_loo, y, 0.1);
```

### `uniMCP_objective_given_eta_loo`

**Signature**
```matlab
obj = uniMCP_objective_given_eta_loo(theta, eta_loo, y, lambda, gamma)
```

**Purpose**
UniMCP objective: mean squared error plus MCP penalty.

**Parameters**
- `gamma`: MCP concavity (`> 1`).

**Example**
```matlab
objFun = @(psi) uniMCP_objective_given_eta_loo(psi, eta_loo, y, 0.1, 3.0);
```

### `uniSCAD_objective_given_eta_loo`

**Signature**
```matlab
obj = uniSCAD_objective_given_eta_loo(theta, eta_loo, y, lambda, a)
```

**Purpose**
UniSCAD objective: mean squared error plus SCAD penalty.

**Parameters**
- `a`: SCAD concavity (`> 2`, common default `3.7`).

**Example**
```matlab
objFun = @(psi) uniSCAD_objective_given_eta_loo(psi, eta_loo, y, 0.1, 3.7);
```

### `unibridge_objective_given_eta_loo`

**Signature**
```matlab
obj = unibridge_objective_given_eta_loo(theta, eta_loo, y, lambda, q)
```

**Purpose**
UniBridge objective: mean squared error plus bridge penalty $\sum |\theta_j|^q$.

**Parameters**
- `q`: bridge exponent (`0 < q <= 1`).

**Example**
```matlab
objFun = @(psi) unibridge_objective_given_eta_loo(psi, eta_loo, y, 0.1, 0.5);
```

---

## 2) Utilities (folder: `supp funs`)

### `split_data`

**Signature**
```matlab
data = split_data(X, y, nfolds)
```

**Purpose**
Builds train/test indices for CV.

**Behavior**
- `nfolds == 1`: 80/20 split.
- `nfolds > 1`: balanced K-fold split.

**Returns**
- `data.train_idx{f}`, `data.test_idx{f}`.
- Backward-compatible data copies in `data.train{f}` and `data.test{f}`.

**Example**
```matlab
data = split_data(X, y, 5);
train_idx = data.train_idx{1};
```

### `compute_sparse_metrics`

**Signature**
```matlab
metrics = compute_sparse_metrics(beta_hat_whole, beta_true_whole, yhat_full, yfull, tol)
```

**Purpose**
Computes sparse-recovery and prediction metrics.

**Returns**
- Row vector:
`[TPR, FPR, MCC, Beta_RMSE, Beta_MAD, Full_MSE, FDR]`

**Example**
```matlab
metrics = compute_sparse_metrics(beta_hat, beta_true, yhat, y, 1e-4);
```

---

## 3) Optimizer (folder: `RMPSH`)

### `RMPSH`

**Signature**
```matlab
[x_opt, fval, comp_time] = RMPSH(objFun, x0, lb, ub, options)
```

**Purpose**
Bound-constrained derivative-free optimization (Recursive Modified Pattern Search on Hyper-Rectangles).

**Key options fields**
- `MaxTime`, `MaxRuns`, `MaxIter`
- `sInitial`, `rho`, `rho2`, `TolFun1`, `TolFun2`, `phi`, `cutoff`
- `DisplayUpdate`, `DisplayEvery`, `PrintSolution`

**Example**
```matlab
objFun = @(x) sum((x - 1).^2);
x0 = zeros(3,1); lb = -5*ones(3,1); ub = 5*ones(3,1);
opts.DisplayUpdate = 0; opts.PrintSolution = 0;
[x_opt, fval, comp_time] = RMPSH(objFun, x0, lb, ub, opts);
```

---

## 4) Data Generation (folder: `Data generation`)

### `Generate_data_scenario_1`

**Signature**
```matlab
[X, y, beta0_true, beta_true, Sigma] = Generate_data_scenario_1(n, p, true_p)
```

**Purpose**
Generates sparse linear data with correlated Gaussian design and multi-scale signals.

**Example**
```matlab
[X, y, b0, b, Sigma] = Generate_data_scenario_1(300, 1000, 20);
```

### `Generate_data_scenario_counterexample`

**Signature**
```matlab
[X, y, beta0_true, beta_true] = Generate_data_scenario_counterexample(n, p)
```

**Purpose**
Generates the counterexample setup where `x2` is correlated with `x1` and only first two coefficients are active.

**Example**
```matlab
[X, y, b0, b] = Generate_data_scenario_counterexample(100, 20);
```

### `Generate_data_scenario_homecourt`

**Signature**
```matlab
[X, y, beta0_true, beta_true_final, Sigma] = Generate_data_scenario_homecourt(n, p)
```

**Purpose**
Implements the Homecourt-style two-stage generation with AR(1) covariance and SNR-controlled responses.

**Example**
```matlab
[X, y, b0, b, Sigma] = Generate_data_scenario_homecourt(200, 100);
```

---

## 5) Other Methods (folder: `other methods`)

### `fit_lasso`

**Signature**
```matlab
[beta0_hat, beta_hat] = fit_lasso(Xtr, ytr, lambda)
```

**Purpose**
Fits standard LASSO using MATLAB `lasso`.

**Example**
```matlab
[b0, b] = fit_lasso(X, y, 0.05);
```

### `fit_adap_lasso`

**Signature**
```matlab
[beta0_hat, beta_hat] = fit_adap_lasso(Xtr, ytr, lambda)
```

**Purpose**
Fits adaptive LASSO using OLS-based weights then weighted LASSO.

**Example**
```matlab
[b0, b] = fit_adap_lasso(X, y, 0.05);
```

### `fit_mcp`

**Signature**
```matlab
[beta0_hat, beta_hat] = fit_mcp(Xtr, ytr, lambda)
```

**Purpose**
Fits MCP-penalized regression by minimizing `mcp_objective` (internally uses `gamma_MCP = 3`).

**Example**
```matlab
[b0, b] = fit_mcp(X, y, 0.05);
```

### `fit_mcp_LLA`

**Signature**
```matlab
[beta0_hat, beta_hat] = fit_mcp_LLA(X, y, lambda)
```

**Purpose**
Fits MCP via Local Linear Approximation (iterative weighted LASSO).

**Example**
```matlab
[b0, b] = fit_mcp_LLA(X, y, 0.05);
```

### `fit_scad`

**Signature**
```matlab
[beta0_hat, beta_hat] = fit_scad(Xtr, ytr, lambda)
```

**Purpose**
Fits SCAD-penalized regression by minimizing `scad_objective`.

**Example**
```matlab
[b0, b] = fit_scad(X, y, 0.05);
```

### `fit_scad_LLA`

**Signature**
```matlab
[beta0_hat, beta_hat] = fit_scad_LLA(X, y, lambda)
```

**Purpose**
Fits SCAD via Local Linear Approximation (iterative weighted LASSO).

**Example**
```matlab
[b0, b] = fit_scad_LLA(X, y, 0.05);
```

### `mcp_penalty`

**Signature**
```matlab
p = mcp_penalty(beta, lambda, gamma)
```

**Purpose**
Computes total MCP penalty value for coefficient vector `beta`.

**Example**
```matlab
p = mcp_penalty(beta, 0.05, 3.0);
```

### `mcp_derivative`

**Signature**
```matlab
w = mcp_derivative(t, lambda, gamma)
```

**Purpose**
Computes MCP derivative weights for nonnegative magnitudes `t`.

**Example**
```matlab
w = mcp_derivative(abs(beta), 0.05, 3.0);
```

### `mcp_objective`

**Signature**
```matlab
obj = mcp_objective(beta, X, y, lambda, gamma_MCP)
```

**Purpose**
MCP objective used by `fit_mcp`: MSE + MCP penalty.

**Example**
```matlab
obj = mcp_objective(beta, X, y, 0.05, 3.0);
```

### `scad_penalty`

**Signature**
```matlab
p = scad_penalty(beta, lambda, a)
```

**Purpose**
Computes total SCAD penalty value for coefficient vector `beta`.

**Example**
```matlab
p = scad_penalty(beta, 0.05, 3.7);
```

### `scad_derivative`

**Signature**
```matlab
w = scad_derivative(t, lambda, a)
```

**Purpose**
Computes SCAD derivative weights for nonnegative magnitudes `t`.

**Example**
```matlab
w = scad_derivative(abs(beta), 0.05, 3.7);
```

### `scad_objective`

**Signature**
```matlab
obj = scad_objective(beta, X, y, lambda, a)
```

**Purpose**
SCAD objective used by `fit_scad`: squared loss + SCAD penalty.

**Example**
```matlab
obj = scad_objective(beta, X, y, 0.05, 3.7);
```

---

## 6) Internal Helper Functions (not public API)

These are local/helper functions inside script or function files and are usually not called directly.

- In `Unisparse/unisparse.m`: `cv_for_objective`, `unisparse_cv_single`, `cv_run_local`.
- In `scenarios/scenario_1.m`: `generate_scenario1_data`, `summarize_unisparse_methods`.

---

## 7) Notes

- Some functions rely on MATLAB toolboxes (for example, `lasso` and parallel features).
- For package users, install the `.mltbx` and call `unisparse` directly.
- For source-tree users, run `init_project.m` once per MATLAB session if needed.