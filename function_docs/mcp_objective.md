# mcp_objective

## What this function does
MCP objective used by `fit_mcp`: MSE + MCP penalty.

## Signature
~~~matlab
obj = mcp_objective(beta, X, y, lambda, gamma_MCP)
~~~

## Typical use case
- Inspect penalty behavior directly.
- Reuse in custom optimization pipelines.

## Mathematical form
$$
\mathcal{L}(\beta)=\frac{1}{n}\|y-X\beta\|_2^2 + \sum_j p_{MCP}(|\beta_j|;\lambda,\gamma)
$$

## Parameters
- `beta`: Coefficient vector.
- `X`: Design matrix of size n x p.
- `y`: Response vector of size n x 1.
- `lambda`: Regularization value or lambda grid.
- `gamma_MCP`: MCP concavity parameter.

## Returns
- `obj`: Objective value.

## Example
~~~matlab
obj = mcp_objective(beta, X, y, 0.05, 3.0);
~~~

## Practical notes
- Check dimensions before running large jobs.
- Validate hyperparameters early in the workflow.
