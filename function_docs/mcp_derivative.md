# mcp_derivative

## What this function does
Computes MCP derivative weights for nonnegative magnitudes `t`.

## Signature
~~~matlab
w = mcp_derivative(t, lambda, gamma)
~~~

## Typical use case
- Inspect penalty behavior directly.
- Reuse in custom optimization pipelines.

## Mathematical form
For $t\ge 0$:
$$
p'_{MCP}(t)=\lambda\max\left(1-\frac{t}{\gamma\lambda},0\right)
$$

## Parameters
- `t`: Nonnegative magnitude input, often abs(beta).
- `lambda`: Regularization value or lambda grid.
- `gamma`: MCP concavity parameter.

## Returns
- `w`: Derivative or weight vector.

## Example
~~~matlab
w = mcp_derivative(abs(beta), 0.05, 3.0);
~~~

## Practical notes
- Check dimensions before running large jobs.
- Validate hyperparameters early in the workflow.
