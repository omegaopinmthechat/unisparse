% ============================================================
% DEMONSTRATION: How to use uniSCAD_objective_given_eta_loo
% ============================================================
%
% This script demonstrates how to use the SCAD-penalized
% UniSparse objective function, following the same pattern
% as UniMCP but with SCAD penalty instead.
%
% SCAD (Smoothly Clipped Absolute Deviation) - Fan & Li (2001)
%   Three-region penalty:
%     1) |t| <= lambda       : lambda * |t|            (LASSO region)
%     2) lambda<|t|<=a*lambda: quadratic smoothing     (transition)
%     3) |t| > a*lambda      : (a+1)*lambda^2/2        (constant/unbiased)
%
% KEY PARAMETER: a = 3.7 (Fan & Li default)
% ============================================================

clc; clear;
addpath('./Unisparse/');
addpath('./supp funs/');
addpath('./RMPSH/');
addpath('./Data generation/');
addpath('./other methods/');

% -------- 1. Generate or load data --------
n   = 200;
p   = 10;
tol = 1e-4;
rng(42);
[X, y, beta0_true, beta_true, ~] = Generate_data_scenario_homecourt(n, p);
beta_true_whole = [beta0_true; beta_true];

% -------- 2. Univariate regressions + LOO --------
% Identical step for all UniSparse variants
[b0, b, ~, ~, eta_loo] = unisparse_univreg(X, y);

% -------- 3. Set hyperparameters --------
% UniSCAD needs BOTH lambda AND a
lambda = 0.1;   % Regularization parameter
a      = 3.7;   % SCAD concavity parameter (> 2, Fan & Li default = 3.7)

% -------- 4. Initial values --------
theta_init = ones(p, 1);
psi0 = [mean(b0); theta_init];

% -------- 5. Bounds for optimization --------
ub_lb_factor = 1e2;
lb = [-ub_lb_factor; zeros(p,1)];
ub = ub_lb_factor * ones(p+1, 1);

% -------- 6. Define SCAD objective function --------
% KEY DIFFERENCE from UniMCP: uses uniSCAD_objective_given_eta_loo
objFun_SCAD = @(psi) uniSCAD_objective_given_eta_loo( ...
                        psi, eta_loo, y, lambda, a);

% -------- 7. RMPSH optimizer options --------
options.DisplayUpdate = 0;
options.PrintSolution = 0;
options.MaxRuns = 5;
options.TolFun2 = 1e-6;
options.cutoff = 1e-6;

% -------- 8. Optimize --------
tic;
[x_opt_SCAD, fval, exitflag] = RMPSH(objFun_SCAD, psi0, lb, ub, options);
time_SCAD = toc;

% -------- 9. Extract optimal parameters --------
theta0_hat_SCAD = x_opt_SCAD(1);
theta_hat_SCAD  = x_opt_SCAD(2:end);

% -------- 10. Recover final coefficients --------
gamma_hat_SCAD  = b(:) .* theta_hat_SCAD(:);
gamma0_hat_SCAD = theta0_hat_SCAD + sum(b0(:) .* theta_hat_SCAD(:));
beta_hat_whole_uniSCAD = [gamma0_hat_SCAD; gamma_hat_SCAD];

% -------- 11. Predictions and metrics --------
yhat_SCAD = gamma0_hat_SCAD + X * gamma_hat_SCAD;
metrics = compute_sparse_metrics(beta_hat_whole_uniSCAD, beta_true_whole, ...
                                  yhat_SCAD, y, tol);

% -------- 12. Display results --------
fprintf('\n========== UniSCAD Results (lambda=%.3f, a=%.1f) ==========\n', ...
        lambda, a);
fprintf('Objective value : %.6f\n', fval);
fprintf('Exit flag       : %d\n', exitflag);
fprintf('Time            : %.4f sec\n', time_SCAD);
fprintf('MSE             : %.6f\n', metrics(6));
fprintf('Beta RMSE       : %.6f\n', metrics(4));
fprintf('Beta MAD        : %.6f\n', metrics(5));
fprintf('TPR             : %.4f\n', metrics(1));
fprintf('FPR             : %.4f\n', metrics(2));
fprintf('MCC             : %.4f\n', metrics(3));
fprintf('Non-zero coefs  : %d\n',   sum(abs(gamma_hat_SCAD) > tol));

disp(' ');
disp('True coefficients:');
disp([beta0_true; beta_true]');
disp('Estimated coefficients (UniSCAD):');
disp(beta_hat_whole_uniSCAD');

% ============================================================
% MATHEMATICAL SUMMARY: UniLASSO vs UniMCP vs UniSCAD
% ============================================================
%
% UniLASSO Penalty (L1):
%   P(t) = lambda * |t|                    (always proportional, biased)
%
% UniMCP Penalty (Zhang 2010):
%   P(t) = lambda*|t| - t^2/(2*gamma)     if |t| <= gamma*lambda
%        = 0.5*gamma*lambda^2             if |t| > gamma*lambda
%
% UniSCAD Penalty (Fan & Li 2001):
%   P(t) = lambda * |t|                    if |t| <= lambda
%        = -(|t|^2-2*a*lambda*|t|+lambda^2)/(2*(a-1))
%                                          if lambda < |t| <= a*lambda
%        = (a+1)*lambda^2/2               if |t| > a*lambda
%
% ADVANTAGES vs LASSO:
%   - Both MCP/SCAD: unbiased for large |t|, oracle properties
% SCAD-specific:
%   - Smooth transition at |t|=lambda (no kink at origin after region 1)
%   - Derivative: lambda for |t|<=lambda, linearly decays, 0 after a*lambda
%   - Recommended default a=3.7 (Fan & Li, JRSS-B 2001)
%
% ============================================================
