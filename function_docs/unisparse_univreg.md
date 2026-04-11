# unisparse_univreg

## What this function does
Computes per-feature univariate regressions and efficient leave-one-out (LOO) estimates.

## Signature
~~~matlab
[beta0, beta, beta0_loo, beta_loo, eta_loo] = unisparse_univreg(X, y)
~~~

## Typical use case
- Prepare LOO ingredients before solving UniSparse objectives.
- Inspect univariate feature effects quickly.

## Mathematical form
For each feature $j$:
$$
\hat{\beta}_j = \frac{\sum_i (x_{ij}-\bar{x}_j)(y_i-\bar{y})}{\sum_i (x_{ij}-\bar{x}_j)^2},
\quad
\hat{\beta}_{0j} = \bar{y} - \bar{x}_j\hat{\beta}_j
$$
$$
\hat{\eta}_{ij}^{(-i)} = \hat{\beta}_{0j}^{(-i)} + \hat{\beta}_j^{(-i)}x_{ij}
$$

## Parameters
- `X`: Design matrix of size n x p.
- `y`: Response vector of size n x 1.

## Returns
- `beta0`, `beta`: full-sample univariate intercept/slope vectors (`p x 1`).
- `beta0_loo`, `beta_loo`, `eta_loo`: LOO matrices (`n x p`).

## Example
~~~matlab
[b0, b, ~, ~, eta_loo] = unisparse_univreg(X, y);
~~~

## Practical notes
- Check dimensions before running large jobs.
- Validate hyperparameters early in the workflow.
