# unilasso_objective_given_eta_loo

## What this function does
UniLASSO objective: mean squared error plus L1 penalty on slope components of `theta`.

## Signature
~~~matlab
obj = unilasso_objective_given_eta_loo(theta, eta_loo, y, lambda)
~~~

## Typical use case
- Pass as objective handle to RMPSH.
- Evaluate objective values for candidate vectors.

## Mathematical form
$$
\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^{n}\left(y_i-(\theta_0+\eta_{i,:}^{(-i)}\theta)\right)^2 + \lambda\sum_j |\theta_j|
$$

## Parameters
- `theta`: `[theta0; theta_j]`.
- `eta_loo`: LOO design-like matrix from `unisparse_univreg`.
- `y`: response.
- `lambda`: L1 penalty level.

## Returns
- `obj`: Objective value.

## Example
~~~matlab
objFun = @(psi) unilasso_objective_given_eta_loo(psi, eta_loo, y, 0.1);
~~~

## Practical notes
- Check dimensions before running large jobs.
- Validate hyperparameters early in the workflow.
