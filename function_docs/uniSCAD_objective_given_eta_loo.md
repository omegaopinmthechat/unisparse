# uniSCAD_objective_given_eta_loo

## What this function does
UniSCAD objective: mean squared error plus SCAD penalty.

## Signature
~~~matlab
obj = uniSCAD_objective_given_eta_loo(theta, eta_loo, y, lambda, a)
~~~

## Typical use case
- Pass as objective handle to RMPSH.
- Evaluate objective values for candidate vectors.

## Mathematical form
$$
\mathcal{L}(\theta) = \frac{1}{n}\sum_{i=1}^{n}\left(y_i-(\theta_0+\eta_{i,:}^{(-i)}\theta)\right)^2 + \sum_j p_{SCAD}(|\theta_j|;\lambda,a)
$$
$$
p_{SCAD}(t)=
\begin{cases}
\lambda t, & t \le \lambda \\
\frac{2a\lambda t - t^2 - \lambda^2}{2(a-1)}, & \lambda < t \le a\lambda \\
\frac{(a+1)\lambda^2}{2}, & t > a\lambda
\end{cases}
$$

## Parameters
- `a`: SCAD concavity (`> 2`, common default `3.7`).

## Returns
- `obj`: Objective value.

## Example
~~~matlab
objFun = @(psi) uniSCAD_objective_given_eta_loo(psi, eta_loo, y, 0.1, 3.7);
~~~

## Practical notes
- Check dimensions before running large jobs.
- Validate hyperparameters early in the workflow.
