# unisparse

## What this function does
Runs cross-validated UniLASSO / UniMCP / UniSCAD and returns fitted coefficients.

## Signature
~~~matlab
results = unisparse(X, y, lambda_grid, nfolds, method, x0, rmps_lb, rmps_ub, rmps_options, a, gamma)
~~~

## Typical use case
- Run end-to-end model fitting with one call.
- Compare UniLASSO, UniMCP, and UniSCAD under one CV setup.

## Mathematical form
$$
\hat{\theta}(\lambda) = \arg\min_{\theta_0,\theta}
\frac{1}{n}\sum_{i=1}^{n}\left(y_i - (\theta_0 + \eta_{i,:}^{(-i)}\theta)\right)^2 + P_{\lambda}(\theta)
$$

Penalty by method:
- UniLASSO: $P_{\lambda}(\theta) = \lambda\sum_j|\theta_j|$
- UniMCP: $P_{\lambda}(\theta) = \sum_j p_{MCP}(|\theta_j|;\lambda,\gamma)$
- UniSCAD: $P_{\lambda}(\theta) = \sum_j p_{SCAD}(|\theta_j|;\lambda,a)$

## Parameters
- `X`, `y`: training data.
- `lambda_grid`: lambda vector, or `[min max]` range (log-spaced grid generated internally).
- `nfolds` (default `2`): CV folds (`1` gives an 80/20 split via `split_data`).
- `method` (default `'all'`): `'unilasso'`, `'unimcp'`, `'uniscad'`, or `'all'`.
- `x0`: RMPS initial point (`[theta0; theta]`).
- `rmps_lb`, `rmps_ub`: optimizer bounds.
- `rmps_options`: options struct passed to `RMPSH`.
- `a` (default `3.7`): SCAD concavity.
- `gamma` (default `3.0`): MCP concavity.

## Returns
- `results`: struct with fields such as `UNILASSO`, `UNIMCP`, `UNISCAD`.
- Each method field includes `lambda` and `beta`; MCP/SCAD also include `gamma`/`a`.

## Example
~~~matlab
results = unisparse(X, y, logspace(-4, 2, 20), 5, 'unimcp', [], [], [], [], 3.7, 3.0);
beta_unimcp = results.UNIMCP.beta;
~~~

## Practical notes
- If lambda_grid has only [min, max], it expands to a log-spaced grid.
- Tuning RMPS bounds/options can significantly change runtime.
