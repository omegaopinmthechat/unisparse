# fit_mcp_LLA

## What this function does
Fits MCP via Local Linear Approximation (iterative weighted LASSO).

## Signature
~~~matlab
[beta0_hat, beta_hat] = fit_mcp_LLA(X, y, lambda)
~~~

## Typical use case
- Train a baseline sparse model for comparison.
- Use as a benchmark against UniSparse methods.

## Mathematical form
LLA iteration for MCP:
$$
\lambda_j^{(k)} = p'_{MCP}(|\beta_j^{(k)}|),\quad
\beta^{(k+1)} = \arg\min_{\beta}\ \frac{1}{n}\|y-X\beta\|_2^2 + \sum_j \lambda_j^{(k)}|\beta_j|
$$

## Parameters
- `X`: Design matrix of size n x p.
- `y`: Response vector of size n x 1.
- `lambda`: Regularization value or lambda grid.

## Returns
- `beta0_hat`: Estimated intercept.
- `beta_hat`: Estimated slopes.

## Example
~~~matlab
[b0, b] = fit_mcp_LLA(X, y, 0.05);
~~~

## Practical notes
- These are baseline methods and may need separate lambda tuning.
- Standardization behavior can differ across implementations.
