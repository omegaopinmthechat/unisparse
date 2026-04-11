# uniMCP_objective_given_eta_loo

## What this function does
UniMCP objective: mean squared error plus MCP penalty.

## Signature
~~~matlab
obj = uniMCP_objective_given_eta_loo(theta, eta_loo, y, lambda, gamma)
~~~

## Typical use case
- Pass as objective handle to RMPSH.
- Evaluate objective values for candidate vectors.

## Mathematical form
$$
\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^{n}\left(y_i-(\theta_0+\eta_{i,:}^{(-i)}\theta)\right)^2 + \sum_j p_{MCP}(|\theta_j|;\lambda,\gamma)
$$
$$
p_{MCP}(t)=
\begin{cases}
\lambda t - \frac{t^2}{2\gamma}, & t \le \gamma\lambda \\
\frac{1}{2}\gamma\lambda^2, & t > \gamma\lambda
\end{cases}
$$

## Parameters
- `gamma`: MCP concavity (`> 1`).

## Returns
- `obj`: Objective value.

## Example
~~~matlab
objFun = @(psi) uniMCP_objective_given_eta_loo(psi, eta_loo, y, 0.1, 3.0);
~~~

## Practical notes
- Check dimensions before running large jobs.
- Validate hyperparameters early in the workflow.
