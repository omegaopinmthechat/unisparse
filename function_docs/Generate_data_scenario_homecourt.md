# Generate_data_scenario_homecourt

## What this function does
Implements the Homecourt-style two-stage generation with AR(1) covariance and SNR-controlled responses.

## Signature
~~~matlab
[X, y, beta0_true, beta_true_final, Sigma] = Generate_data_scenario_homecourt(n, p)
~~~

## Typical use case
- Build simulation datasets with known truth.
- Stress-test estimators under controlled signal/correlation settings.

## Mathematical form
AR(1) covariance design:
$$
\Sigma_{jk}=\rho^{|j-k|},\quad \rho=0.8
$$
Two-stage signal generation is used to mimic the homecourt scenario.

## Parameters
- `n`: Sample size.
- `p`: Number of features.

## Returns
- `X`: Generated design matrix.
- `y`: Generated response vector.
- `beta0_true`: True intercept used for generation.
- `beta_true_final`: True coefficient vector used for generation.
- `Sigma`: Feature covariance matrix.

## Example
~~~matlab
[X, y, b0, b, Sigma] = Generate_data_scenario_homecourt(200, 100);
~~~

## Practical notes
- Seed RNG before generation to reproduce experiments.
- Validate empirical SNR when comparing simulation tables.
