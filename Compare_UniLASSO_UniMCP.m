% ============================================================
% COMPARISON: UniLASSO vs UniMCP
% 
% This script compares UniLASSO (L1 penalty) with 
% UniMCP (MCP penalty) on the same dataset.
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
lambda = 0.05;     % Regularization parameter (same for both)
gamma_mcp = 3.0;   % MCP concavity parameter (only for MCP)

% -------- Bounds and Options --------
ub_lb_factor = 1e2;
lb = [-ub_lb_factor; zeros(p,1)];
ub = ub_lb_factor * ones(p+1, 1);

options.DisplayUpdate = 0;
options.PrintSolution = 0;
options.MaxRuns = 5;
options.TolFun2 = 1e-6;
options.cutoff = 1e-6;

% -------- Step 1: Univariate Regressions (same for both methods) --------
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

% Predictions
yhat_L1 = gamma0_hat_L1 + X * gamma_hat_L1;
metrics_L1 = compute_sparse_metrics(beta_hat_whole_L1, beta_true_whole, ...
                                     yhat_L1, y, tol);

% ============================================================
% Method 2: UniMCP (MCP penalty)
% ============================================================
fprintf('\n--- Fitting UniMCP (MCP penalty with gamma=%.1f) ---\n', gamma_mcp);

% Define MCP objective (with gamma parameter)
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

% Predictions
yhat_MCP = gamma0_hat_MCP + X * gamma_hat_MCP;
metrics_MCP = compute_sparse_metrics(beta_hat_whole_MCP, beta_true_whole, ...
                                      yhat_MCP, y, tol);

% ============================================================
% COMPARISON TABLE
% ============================================================
Method = ["UniLASSO (L1)"; "UniMCP"];

Lambda = [lambda; lambda];
Gamma = [NaN; gamma_mcp];

TPR = [metrics_L1.TPR; metrics_MCP.TPR];
FPR = [metrics_L1.FPR; metrics_MCP.FPR];
MCC = [metrics_L1.MCC; metrics_MCP.MCC];

Beta_RMSE = [metrics_L1.beta_rmse; metrics_MCP.beta_rmse];
Beta_MAD = [metrics_L1.beta_mad; metrics_MCP.beta_mad];

MSE = [metrics_L1.mse; metrics_MCP.mse];
Time_sec = [time_L1; time_MCP];

Obj_Value = [fval_L1; fval_MCP];

Results_Table = table(Method, Lambda, Gamma, TPR, FPR, MCC, ...
                      Beta_RMSE, Beta_MAD, MSE, Time_sec, Obj_Value);

fprintf('\n');
disp('============================================================');
disp('           UniLASSO vs UniMCP COMPARISON');
disp('============================================================');
disp(Results_Table);

% -------- Coefficient Comparison --------
fprintf('\n');
disp('============================================================');
disp('             COEFFICIENT ESTIMATES');
disp('============================================================');

Coef_Table = table((0:p)', beta_true_whole, beta_hat_whole_L1, beta_hat_whole_MCP, ...
                   'VariableNames', {'Index', 'True', 'UniLASSO', 'UniMCP'});
disp(Coef_Table);

% -------- Visualize Coefficients --------
figure('Position', [100 100 1000 500]);

subplot(1,2,1);
bar([beta_true_whole, beta_hat_whole_L1, beta_hat_whole_MCP]);
xlabel('Coefficient Index', 'FontSize', 11);
ylabel('Coefficient Value', 'FontSize', 11);
title('Coefficient Comparison', 'FontSize', 12);
legend('True', 'UniLASSO', 'UniMCP', 'Location', 'best');
grid on;

subplot(1,2,2);
hold on;
plot(beta_true_whole, beta_hat_whole_L1, 'o', 'MarkerSize', 8, ...
     'DisplayName', 'UniLASSO');
plot(beta_true_whole, beta_hat_whole_MCP, 's', 'MarkerSize', 8, ...
     'DisplayName', 'UniMCP');
plot([-5 5], [-5 5], 'k--', 'LineWidth', 1.5, 'DisplayName', 'Perfect Fit');
xlabel('True Coefficients', 'FontSize', 11);
ylabel('Estimated Coefficients', 'FontSize', 11);
title('Estimation Accuracy', 'FontSize', 12);
legend('Location', 'best');
grid on;
axis equal;
hold off;

% -------- Summary --------
fprintf('\n');
disp('============================================================');
disp('                     SUMMARY');
disp('============================================================');
fprintf('Dataset: n=%d, p=%d, true_p=%d\n', n, p, true_p);
fprintf('Lambda: %.4f\n', lambda);
fprintf('MCP Gamma: %.2f\n\n', gamma_mcp);

fprintf('UniLASSO Performance:\n');
fprintf('  - MSE: %.6f\n', metrics_L1.mse);
fprintf('  - Beta RMSE: %.6f\n', metrics_L1.beta_rmse);
fprintf('  - MCC: %.4f\n', metrics_L1.MCC);
fprintf('  - Time: %.4f sec\n\n', time_L1);

fprintf('UniMCP Performance:\n');
fprintf('  - MSE: %.6f\n', metrics_MCP.mse);
fprintf('  - Beta RMSE: %.6f\n', metrics_MCP.beta_rmse);
fprintf('  - MCC: %.4f\n', metrics_MCP.MCC);
fprintf('  - Time: %.4f sec\n\n', time_MCP);

if metrics_MCP.mse < metrics_L1.mse
    fprintf('Winner: UniMCP (%.2f%% MSE reduction)\n', ...
            100*(metrics_L1.mse - metrics_MCP.mse)/metrics_L1.mse);
else
    fprintf('Winner: UniLASSO (%.2f%% MSE reduction)\n', ...
            100*(metrics_MCP.mse - metrics_L1.mse)/metrics_MCP.mse);
end
disp('============================================================');
