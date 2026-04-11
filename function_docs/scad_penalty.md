# scad_penalty

## What this function does
Computes total SCAD penalty value for coefficient vector `beta`.

## Signature
~~~matlab
p = scad_penalty(beta, lambda, a)
~~~

## Typical use case
- Inspect penalty behavior directly.
- Reuse in custom optimization pipelines.

## Mathematical form
$$
p_{SCAD}(t)=
\begin{cases}
\lambda t, & t\le\lambda \\
\frac{2a\lambda t - t^2 - \lambda^2}{2(a-1)}, & \lambda<t\le a\lambda \\
\frac{(a+1)\lambda^2}{2}, & t>a\lambda
\end{cases}
$$
This function returns $\sum_j p_{SCAD}(|\beta_j|)$.

## Parameters
- `beta`: Coefficient vector.
- `lambda`: Regularization value or lambda grid.
- `a`: SCAD concavity parameter.

## Returns
- `p`: Penalty value.

## Example
~~~matlab
p = scad_penalty(beta, 0.05, 3.7);
~~~

## Practical notes
- Check dimensions before running large jobs.
- Validate hyperparameters early in the workflow.
