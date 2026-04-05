% ============================================================
% UNIFIED COMPARISON: UniLASSO vs UniMCP vs UniSCAD
% 
% This script compares all three unified sparse methods:
%   - UniLASSO (L1 penalty)
%   - UniMCP   (MCP penalty)
%   - UniSCAD  (SCAD penalty)
%
% % Metrics: TPR, FPR, MCC, Beta RMSE, Beta MAD, MSE, FDR
% ============================================================

clc; clear;
addpath('./Unisparse/');
addpath('./supp funs/');
addpath('./RMPSH/');
addpath('./Data generation/');
addpath('./other methods/');

% -------- Data Generation --------
n = 200;
p = 10;
true_p = 5;
tol = 1e-4;

rng(2);
[X, y, beta0_true, beta_true, ~] = Generate_data_scenario_homecourt(n, p);
beta_true_whole = [beta0_true; beta_true];

% -------- Hyperparameters --------
lambda = 0.05;       % Regularization parameter (same for all)
gamma_mcp = 3.0;     % MCP concavity parameter (Zhang default)
a_scad = 3.7;        % SCAD concavity parameter (Fan & Li default)

% -------- Bounds and Options --------
ub_lb_factor = 1e2;
lb = [-ub_lb_factor; zeros(p,1)];
ub = ub_lb_factor * ones(p+1, 1);

options.DisplayUpdate = 0;
options.PrintSolution = 0;
options.MaxRuns = 5;
options.TolFun2 = 1e-6;
options.cutoff = 1e-6;

% -------- Step 1: Univariate Regressions (same for all methods) --------
fprintf('Computing univariate regressions and LOO estimates...\n');
[b0, b, ~, ~, eta_loo] = unisparse_univreg(X, y);

% Initial values
theta_init = ones(p, 1);
psi0 = [mean(b0); theta_init];

% ============================================================
% Method 1: UniLASSO (L1 penalty)
% ============================================================
fprintf('\n--- Fitting UniLASSO (L1 penalty) ---\n');

% Define L1 objective
objFun_L1 = @(psi) unilasso_objective_given_eta_loo( ...
                      psi, eta_loo, y, lambda);

% Optimize
tic;
[x_opt_L1, fval_L1, exitflag_L1] = RMPSH(objFun_L1, psi0, lb, ub, options);
time_L1 = toc;

% Extract results
theta0_hat_L1 = x_opt_L1(1);
theta_hat_L1  = x_opt_L1(2:end);

% Recover coefficients
gamma_hat_L1  = b(:) .* theta_hat_L1(:);
gamma0_hat_L1 = theta0_hat_L1 + sum(b0(:) .* theta_hat_L1(:));
beta_hat_whole_L1 = [gamma0_hat_L1; gamma_hat_L1];

% Predictions and metrics
yhat_L1 = gamma0_hat_L1 + X * gamma_hat_L1;
metrics_L1 = compute_sparse_metrics(beta_hat_whole_L1, beta_true_whole, ...
                                     yhat_L1, y, tol);

fprintf('  Completed in %.4f sec | TPR=%.3f, FPR=%.3f, FDR=%.3f, MSE=%.6f\n', ...
        time_L1, metrics_L1(1), metrics_L1(2), metrics_L1(7), metrics_L1(6));

% ============================================================
% Method 2: UniMCP (MCP penalty)
% ============================================================
fprintf('\n--- Fitting UniMCP (MCP penalty with gamma=%.1f) ---\n', gamma_mcp);

% Define MCP objective
objFun_MCP = @(psi) uniMCP_objective_given_eta_loo( ...
                       psi, eta_loo, y, lambda, gamma_mcp);

% Optimize
tic;
[x_opt_MCP, fval_MCP, exitflag_MCP] = RMPSH(objFun_MCP, psi0, lb, ub, options);
time_MCP = toc;

% Extract results
theta0_hat_MCP = x_opt_MCP(1);
theta_hat_MCP  = x_opt_MCP(2:end);

% Recover coefficients
gamma_hat_MCP  = b(:) .* theta_hat_MCP(:);
gamma0_hat_MCP = theta0_hat_MCP + sum(b0(:) .* theta_hat_MCP(:));
beta_hat_whole_MCP = [gamma0_hat_MCP; gamma_hat_MCP];

% Predictions and metrics
yhat_MCP = gamma0_hat_MCP + X * gamma_hat_MCP;
metrics_MCP = compute_sparse_metrics(beta_hat_whole_MCP, beta_true_whole, ...
                                      yhat_MCP, y, tol);

fprintf('  Completed in %.4f sec | TPR=%.3f, FPR=%.3f, FDR=%.3f, MSE=%.6f\n', ...
        time_MCP, metrics_MCP(1), metrics_MCP(2), metrics_MCP(7), metrics_MCP(6));

% ============================================================
% Method 3: UniSCAD (SCAD penalty)
% ============================================================
fprintf('\n--- Fitting UniSCAD (SCAD penalty with a=%.1f) ---\n', a_scad);

% Define SCAD objective
objFun_SCAD = @(psi) uniSCAD_objective_given_eta_loo( ...
                        psi, eta_loo, y, lambda, a_scad);

% Optimize
tic;
[x_opt_SCAD, fval_SCAD, exitflag_SCAD] = RMPSH(objFun_SCAD, psi0, lb, ub, options);
time_SCAD = toc;

% Extract results
theta0_hat_SCAD = x_opt_SCAD(1);
theta_hat_SCAD  = x_opt_SCAD(2:end);

% Recover coefficients
gamma_hat_SCAD  = b(:) .* theta_hat_SCAD(:);
gamma0_hat_SCAD = theta0_hat_SCAD + sum(b0(:) .* theta_hat_SCAD(:));
beta_hat_whole_SCAD = [gamma0_hat_SCAD; gamma_hat_SCAD];

% Predictions and metrics
yhat_SCAD = gamma0_hat_SCAD + X * gamma_hat_SCAD;
metrics_SCAD = compute_sparse_metrics(beta_hat_whole_SCAD, beta_true_whole, ...
                                      yhat_SCAD, y, tol);

fprintf('  Completed in %.4f sec | TPR=%.3f, FPR=%.3f, FDR=%.3f, MSE=%.6f\n', ...
        time_SCAD, metrics_SCAD(1), metrics_SCAD(2), metrics_SCAD(7), metrics_SCAD(6));

% ============================================================
% UNIFIED COMPARISON TABLE
% ============================================================
Method = ["UniLASSO"; "UniMCP"; "UniSCAD"];

Lambda = [lambda; lambda; lambda];
Penalty_Param = [NaN; gamma_mcp; a_scad];

TPR = [metrics_L1(1); metrics_MCP(1); metrics_SCAD(1)];
FPR = [metrics_L1(2); metrics_MCP(2); metrics_SCAD(2)];
FDR = [metrics_L1(7); metrics_MCP(7); metrics_SCAD(7)];
MCC = [metrics_L1(3); metrics_MCP(3); metrics_SCAD(3)];

Beta_RMSE = [metrics_L1(4); metrics_MCP(4); metrics_SCAD(4)];
Beta_MAD = [metrics_L1(5); metrics_MCP(5); metrics_SCAD(5)];

MSE = [metrics_L1(6); metrics_MCP(6); metrics_SCAD(6)];
Time_sec = [time_L1; time_MCP; time_SCAD];

Obj_Value = [fval_L1; fval_MCP; fval_SCAD];

Results_Table = table(Method, Lambda, Penalty_Param, TPR, FPR, FDR, MCC, ...
                      Beta_RMSE, Beta_MAD, MSE, Time_sec, Obj_Value);

fprintf('\n');
disp('============================================================');
disp('     UNIFIED COMPARISON: UniLASSO vs UniMCP vs UniSCAD');
disp('============================================================');
disp(Results_Table);

% -------- Key Metrics Focus --------
fprintf('\n');
disp('============================================================');
disp('               KEY METRICS SUMMARY');
disp('============================================================');
Key_Metrics = table(Method, TPR, FPR, FDR, MSE, Time_sec, ...
                    'VariableNames', {'Method', 'TPR', 'FPR', 'FDR', 'MSE', 'Time_sec'});
disp(Key_Metrics);

% -------- Coefficient Comparison --------
fprintf('\n');
disp('============================================================');
disp('             COEFFICIENT ESTIMATES');
disp('============================================================');

Coef_Table = table((0:p)', beta_true_whole, beta_hat_whole_L1, ...
                   beta_hat_whole_MCP, beta_hat_whole_SCAD, ...
                   'VariableNames', {'Index', 'True', 'UniLASSO', 'UniMCP', 'UniSCAD'});
disp(Coef_Table);

% -------- Summary --------
fprintf('\n');
disp('============================================================');
disp('                     SUMMARY');
disp('============================================================');
fprintf('Dataset: n=%d, p=%d, true_p=%d\n', n, p, true_p);
fprintf('Tolerance for sparsity: %.1e\n', tol);
fprintf('\nHyperparameters:\n');
fprintf('  - Lambda (all methods): %.4f\n', lambda);
fprintf('  - MCP gamma: %.2f\n', gamma_mcp);
fprintf('  - SCAD a: %.2f\n\n', a_scad);

fprintf('Performance Ranking by MSE:\n');
[~, idx] = sort(MSE);
for i = 1:length(idx)
    fprintf('  %d. %s: MSE=%.6f, TPR=%.3f, FPR=%.3f\n', ...
            i, Method{idx(i)}, MSE(idx(i)), TPR(idx(i)), FPR(idx(i)));
end

fprintf('\nPerformance Ranking by TPR (True Positive Rate):\n');
[~, idx] = sort(TPR, 'descend');
for i = 1:length(idx)
    fprintf('  %d. %s: TPR=%.3f, FPR=%.3f, MSE=%.6f\n', ...
            i, Method{idx(i)}, TPR(idx(i)), FPR(idx(i)), MSE(idx(i)));
end
