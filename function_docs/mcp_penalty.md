# mcp_penalty

## What this function does
Computes total MCP penalty value for coefficient vector `beta`.

## Signature
~~~matlab
p = mcp_penalty(beta, lambda, gamma)
~~~

## Typical use case
- Inspect penalty behavior directly.
- Reuse in custom optimization pipelines.

## Mathematical form
$$
p_{MCP}(t)=
\begin{cases}
\lambda t - \frac{t^2}{2\gamma}, & t \le \gamma\lambda \\
\frac{1}{2}\gamma\lambda^2, & t > \gamma\lambda
\end{cases}
$$
This function returns $\sum_j p_{MCP}(|\beta_j|)$.

## Parameters
- `beta`: Coefficient vector.
- `lambda`: Regularization value or lambda grid.
- `gamma`: MCP concavity parameter.

## Returns
- `p`: Penalty value.

## Example
~~~matlab
p = mcp_penalty(beta, 0.05, 3.0);
~~~

## Practical notes
- Check dimensions before running large jobs.
- Validate hyperparameters early in the workflow.
