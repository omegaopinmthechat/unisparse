# Generate_data_scenario_1

## What this function does
Generates sparse linear data with correlated Gaussian design and multi-scale signals.

## Signature
~~~matlab
[X, y, beta0_true, beta_true, Sigma] = Generate_data_scenario_1(n, p, true_p)
~~~

## Typical use case
- Build simulation datasets with known truth.
- Stress-test estimators under controlled signal/correlation settings.

## Mathematical form
$$
X \sim \mathcal{N}(0,\Sigma),\quad y = \beta_0 + X\beta + \epsilon
$$
The generator uses sparse multi-scale coefficients and Gaussian noise with target SNR.

## Parameters
- `n`: Sample size.
- `p`: Number of features.
- `true_p`: Number of true nonzero coefficients.

## Returns
- `X`: Generated design matrix.
- `y`: Generated response vector.
- `beta0_true`: True intercept used for generation.
- `beta_true`: True coefficient vector used for generation.
- `Sigma`: Feature covariance matrix.

## Example
~~~matlab
[X, y, b0, b, Sigma] = Generate_data_scenario_1(300, 1000, 20);
~~~

## Practical notes
- Seed RNG before generation to reproduce experiments.
- Validate empirical SNR when comparing simulation tables.
