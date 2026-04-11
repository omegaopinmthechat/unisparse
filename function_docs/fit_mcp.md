# fit_mcp

## What this function does
Fits MCP-penalized regression by minimizing `mcp_objective` (internally uses `gamma_MCP = 3`).

## Signature
~~~matlab
[beta0_hat, beta_hat] = fit_mcp(Xtr, ytr, lambda)
~~~

## Typical use case
- Train a baseline sparse model for comparison.
- Use as a benchmark against UniSparse methods.

## Mathematical form
$$
\hat{\beta}=\arg\min_{\beta}\ \frac{1}{n}\|y-X\beta\|_2^2 + \sum_j p_{MCP}(|\beta_j|;\lambda,\gamma)
$$

## Parameters
- `Xtr`: Design matrix of size n x p.
- `ytr`: Response vector of size n x 1.
- `lambda`: Regularization value or lambda grid.

## Returns
- `beta0_hat`: Estimated intercept.
- `beta_hat`: Estimated slopes.

## Example
~~~matlab
[b0, b] = fit_mcp(X, y, 0.05);
~~~

## Practical notes
- These are baseline methods and may need separate lambda tuning.
- Standardization behavior can differ across implementations.
