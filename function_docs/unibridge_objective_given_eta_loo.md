# unibridge_objective_given_eta_loo

## What this function does
UniBridge objective: mean squared error plus bridge penalty $\sum |\theta_j|^q$.

## Signature
~~~matlab
obj = unibridge_objective_given_eta_loo(theta, eta_loo, y, lambda, q)
~~~

## Typical use case
- Pass as objective handle to RMPSH.
- Evaluate objective values for candidate vectors.

## Mathematical form
$$
\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^{n}\left(y_i-(\theta_0+\eta_{i,:}^{(-i)}\theta)\right)^2 + \lambda\sum_j |\theta_j|^q,
\quad 0 < q \le 1
$$

## Parameters
- `q`: bridge exponent (`0 < q <= 1`).

## Returns
- `obj`: Objective value.

## Example
~~~matlab
objFun = @(psi) unibridge_objective_given_eta_loo(psi, eta_loo, y, 0.1, 0.5);
~~~

## Practical notes
- Check dimensions before running large jobs.
- Validate hyperparameters early in the workflow.
