# scad_derivative

## What this function does
Computes SCAD derivative weights for nonnegative magnitudes `t`.

## Signature
~~~matlab
w = scad_derivative(t, lambda, a)
~~~

## Typical use case
- Inspect penalty behavior directly.
- Reuse in custom optimization pipelines.

## Mathematical form
For $t\ge 0$:
$$
p'_{SCAD}(t)=
\begin{cases}
\lambda, & t\le\lambda \\
\frac{a\lambda-t}{a-1}, & \lambda<t\le a\lambda \\
0, & t>a\lambda
\end{cases}
$$

## Parameters
- `t`: Nonnegative magnitude input, often abs(beta).
- `lambda`: Regularization value or lambda grid.
- `a`: SCAD concavity parameter.

## Returns
- `w`: Derivative or weight vector.

## Example
~~~matlab
w = scad_derivative(abs(beta), 0.05, 3.7);
~~~

## Practical notes
- Check dimensions before running large jobs.
- Validate hyperparameters early in the workflow.
