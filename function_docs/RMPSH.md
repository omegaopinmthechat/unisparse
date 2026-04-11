# RMPSH

## What this function does
Bound-constrained derivative-free optimization (Recursive Modified Pattern Search on Hyper-Rectangles).

## Signature
~~~matlab
[x_opt, fval, comp_time] = RMPSH(objFun, x0, lb, ub, options)
~~~

## Typical use case
- Optimize non-convex objectives with simple bounds.
- Use when gradients are unavailable or expensive.

## Mathematical form
RMPSH performs bounded pattern search on transformed coordinates:
$$
\theta^{(k+1)} = \arg\min_{u\in\mathcal{N}(\theta^{(k)},\epsilon_k)} f(u)
$$
with adaptive step-size reduction when improvement is small.

## Parameters
- `objFun`: Objective function handle.
- `x0`: Initial point for optimization.
- `lb`: Lower bounds vector.
- `ub`: Upper bounds vector.
- `options`: Solver options struct.

## Returns
- `x_opt`: Best solution vector found by RMPSH.
- `fval`: Objective value at x_opt.
- `comp_time`: Computation time in seconds.

## Example
~~~matlab
objFun = @(x) sum((x - 1).^2);
x0 = zeros(3,1); lb = -5*ones(3,1); ub = 5*ones(3,1);
opts.DisplayUpdate = 0; opts.PrintSolution = 0;
[x_opt, fval, comp_time] = RMPSH(objFun, x0, lb, ub, opts);
~~~

## Practical notes
- Keep x0 inside [lb, ub] for faster convergence.
- Use smaller MaxRuns/MaxIter for quick prototyping.
