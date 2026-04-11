# scad_objective

## What this function does
SCAD objective used by `fit_scad`: squared loss + SCAD penalty.

## Signature
~~~matlab
obj = scad_objective(beta, X, y, lambda, a)
~~~

## Typical use case
- Inspect penalty behavior directly.
- Reuse in custom optimization pipelines.

## Mathematical form
$$
\mathcal{L}(\beta)=\frac{1}{2}\|y-X\beta\|_2^2 + \sum_j p_{SCAD}(|\beta_j|;\lambda,a)
$$

## Parameters
- `beta`: Coefficient vector.
- `X`: Design matrix of size n x p.
- `y`: Response vector of size n x 1.
- `lambda`: Regularization value or lambda grid.
- `a`: SCAD concavity parameter.

## Returns
- `obj`: Objective value.

## Example
~~~matlab
obj = scad_objective(beta, X, y, 0.05, 3.7);
~~~

## Practical notes
- Check dimensions before running large jobs.
- Validate hyperparameters early in the workflow.
