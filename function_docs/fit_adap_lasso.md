# fit_adap_lasso

## What this function does
Fits adaptive LASSO using OLS-based weights then weighted LASSO.

## Signature
~~~matlab
[beta0_hat, beta_hat] = fit_adap_lasso(Xtr, ytr, lambda)
~~~

## Typical use case
- Train a baseline sparse model for comparison.
- Use as a benchmark against UniSparse methods.

## Mathematical form
$$
\hat{\beta} = \arg\min_{\beta}\ \frac{1}{n}\|y-X\beta\|_2^2 + \lambda\sum_j w_j|\beta_j|,
\quad w_j = \frac{1}{|\beta_j^{OLS}|^{\gamma}}
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
[b0, b] = fit_adap_lasso(X, y, 0.05);
~~~

## Practical notes
- These are baseline methods and may need separate lambda tuning.
- Standardization behavior can differ across implementations.
