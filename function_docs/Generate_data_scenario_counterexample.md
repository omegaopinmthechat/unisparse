# Generate_data_scenario_counterexample

## What this function does
Generates the counterexample setup where `x2` is correlated with `x1` and only first two coefficients are active.

## Signature
~~~matlab
[X, y, beta0_true, beta_true] = Generate_data_scenario_counterexample(n, p)
~~~

## Typical use case
- Build simulation datasets with known truth.
- Stress-test estimators under controlled signal/correlation settings.

## Mathematical form
$$
x_1\sim\mathcal{N}(0,1),\quad x_2 = x_1 + z,\ z\sim\mathcal{N}(0,1)
$$
$$
\beta=(1,-0.5,0,\ldots,0),\quad y = X\beta + \epsilon,\ \epsilon\sim\mathcal{N}(0,0.5^2)
$$

## Parameters
- `n`: Sample size.
- `p`: Number of features.

## Returns
- `X`: Generated design matrix.
- `y`: Generated response vector.
- `beta0_true`: True intercept used for generation.
- `beta_true`: True coefficient vector used for generation.

## Example
~~~matlab
[X, y, b0, b] = Generate_data_scenario_counterexample(100, 20);
~~~

## Practical notes
- Seed RNG before generation to reproduce experiments.
- Validate empirical SNR when comparing simulation tables.
