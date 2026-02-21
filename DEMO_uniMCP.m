% ============================================================
% DEMONSTRATION: How to use uniMCP_objective_given_eta_loo
% ============================================================
%
% This script demonstrates how to use the new MCP-penalized
% UniSparse objective function, following the same pattern
% as UniLASSO but with MCP penalty instead of L1.
%
% ============================================================

clc; clear;
addpath('./Unisparse/');
addpath('./supp funs/');
addpath('./RMPSH/');
addpath('./Data generation/');

% -------- 1. Generate or load data --------
n = 200;
p = 10;
rng(42);
[X, y, beta0_true, beta_true, ~] = Generate_data_scenario_homecourt(n, p);

% -------- 2. Univariate regressions + LOO --------
% This step is identical for UniLASSO and UniMCP
[b0, b, ~, ~, eta_loo] = unisparse_univreg(X, y);

% -------- 3. Set hyperparameters --------
% For UniMCP, we need BOTH lambda and gamma
lambda = 0.1;   % Regularization parameter
gamma = 3.0;    % MCP concavity parameter (must be > 1, typically 2-4)

% -------- 4. Initial values --------
theta_init = ones(p, 1);
psi0 = [mean(b0); theta_init];

% -------- 5. Bounds for optimization --------
ub_lb_factor = 1e2;
lb = [-ub_lb_factor; zeros(p,1)];
ub = ub_lb_factor * ones(p+1, 1);

% -------- 6. Define MCP objective function --------
% KEY DIFFERENCE: Pass gamma as additional parameter
objFun_MCP = @(psi) uniMCP_objective_given_eta_loo( ...
                       psi, eta_loo, y, lambda, gamma);

% -------- 7. RMPSH optimizer options --------
options.DisplayUpdate = 0;
options.PrintSolution = 0;
options.MaxRuns = 5;
options.TolFun2 = 1e-6;
options.cutoff = 1e-6;

% -------- 8. Optimize --------
[x_opt_MCP, fval, exitflag] = RMPSH(objFun_MCP, psi0, lb, ub, options);

% -------- 9. Extract optimal parameters --------
theta0_hat_MCP = x_opt_MCP(1);
theta_hat_MCP  = x_opt_MCP(2:end);

% -------- 10. Recover final coefficients --------
gamma_hat_MCP  = b(:) .* theta_hat_MCP(:);
gamma0_hat_MCP = theta0_hat_MCP + sum(b0(:) .* theta_hat_MCP(:));

beta_hat_whole_uniMCP = [gamma0_hat_MCP; gamma_hat_MCP];

% -------- 11. Predictions --------
yhat_MCP = gamma0_hat_MCP + X * gamma_hat_MCP;
mse_MCP = mean((y - yhat_MCP).^2);

% -------- 12. Display results --------
fprintf('\n========== UniMCP Results (lambda=%.3f, gamma=%.1f) ==========\n', ...
        lambda, gamma);
fprintf('Objective value: %.6f\n', fval);
fprintf('Exit flag: %d\n', exitflag);
fprintf('MSE: %.6f\n', mse_MCP);
fprintf('Number of non-zero coefficients: %d\n', sum(abs(gamma_hat_MCP) > 1e-6));

disp(' ');
disp('True coefficients:');
disp([beta0_true; beta_true]');
disp('Estimated coefficients (UniMCP):');
disp(beta_hat_whole_uniMCP');

% ============================================================
% MATHEMATICAL DIFFERENCES: UniLASSO vs UniMCP
% ============================================================
%
% UniLASSO Penalty (L1):
%   P_L1(θ_j) = λ|θ_j|
%
% UniMCP Penalty:
%   P_MCP(θ_j; λ, γ) = λ|θ_j| - θ_j²/(2γ)    if |θ_j| ≤ γλ
%                    = (1/2)γλ²              if |θ_j| > γλ
%
% KEY PROPERTIES:
% 1. MCP is continuous but non-convex (unlike L1)
% 2. MCP applies constant penalty for large |θ_j| (reduces bias)
% 3. MCP still encourages sparsity for small |θ_j|
% 4. γ controls concavity:
%    - γ → ∞ recovers L1 penalty
%    - γ ≈ 2-4 is typical in practice
% 5. MCP derivative:
%    ∂P_MCP/∂θ_j = λ·sign(θ_j)·max(1 - |θ_j|/(γλ), 0)
%
% COMPARISON TO L1 (LASSO):
% - L1: Always penalizes proportional to |θ_j| (introduces bias)
% - MCP: Stops penalizing after |θ_j| > γλ (less bias for large coefs)
%
% ADVANTAGES OF UniMCP:
% - Nearly unbiased estimates for large coefficients
% - Maintains variable selection properties
% - Often achieves oracle properties under regularity conditions
%
% ============================================================
