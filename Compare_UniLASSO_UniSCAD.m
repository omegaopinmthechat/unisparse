% ============================================================
% COMPARISON: UniLASSO vs UniSCAD
%
% This script compares UniLASSO (L1 penalty) with
% UniSCAD (SCAD penalty) on the same dataset.
%
% SCAD - Fan & Li (2001), default concavity parameter a = 3.7
% ============================================================

clc; clear;
addpath('./Unisparse/');
addpath('./supp funs/');
addpath('./RMPSH/');
addpath('./Data generation/');
addpath('./other methods/');

% -------- Data Generation --------
n      = 200;
p      = 10;
true_p = 5;
tol    = 1e-4;

rng(2);
[X, y, beta0_true, beta_true, ~] = Generate_data_scenario_homecourt(n, p);
beta_true_whole = [beta0_true; beta_true];

% -------- Hyperparameters --------
lambda  = 0.05;   % Regularization parameter (same for both)
a_scad  = 3.7;    % SCAD concavity parameter (Fan & Li default)

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

objFun_L1 = @(psi) unilasso_objective_given_eta_loo( ...
                      psi, eta_loo, y, lambda);

tic;
[x_opt_L1, fval_L1, exitflag_L1] = RMPSH(objFun_L1, psi0, lb, ub, options);
time_L1 = toc;

theta0_hat_L1 = x_opt_L1(1);
theta_hat_L1  = x_opt_L1(2:end);

gamma_hat_L1  = b(:) .* theta_hat_L1(:);
gamma0_hat_L1 = theta0_hat_L1 + sum(b0(:) .* theta_hat_L1(:));
beta_hat_whole_L1 = [gamma0_hat_L1; gamma_hat_L1];

yhat_L1   = gamma0_hat_L1 + X * gamma_hat_L1;
metrics_L1 = compute_sparse_metrics(beta_hat_whole_L1, beta_true_whole, ...
                                    yhat_L1, y, tol);

% ============================================================
% Method 2: UniSCAD (SCAD penalty)
% ============================================================
fprintf('\n--- Fitting UniSCAD (SCAD penalty with a=%.1f) ---\n', a_scad);

objFun_SCAD = @(psi) uniSCAD_objective_given_eta_loo( ...
                        psi, eta_loo, y, lambda, a_scad);

tic;
[x_opt_SCAD, fval_SCAD, exitflag_SCAD] = RMPSH(objFun_SCAD, psi0, lb, ub, options);
time_SCAD = toc;

theta0_hat_SCAD = x_opt_SCAD(1);
theta_hat_SCAD  = x_opt_SCAD(2:end);

gamma_hat_SCAD  = b(:) .* theta_hat_SCAD(:);
gamma0_hat_SCAD = theta0_hat_SCAD + sum(b0(:) .* theta_hat_SCAD(:));
beta_hat_whole_SCAD = [gamma0_hat_SCAD; gamma_hat_SCAD];

yhat_SCAD   = gamma0_hat_SCAD + X * gamma_hat_SCAD;
metrics_SCAD = compute_sparse_metrics(beta_hat_whole_SCAD, beta_true_whole, ...
                                      yhat_SCAD, y, tol);

% ============================================================
% COMPARISON TABLE
% ============================================================
Method = ["UniLASSO (L1)"; "UniSCAD"];

Lambda    = [lambda;  lambda];
A_param   = [NaN;     a_scad];

TPR = [metrics_L1(1); metrics_SCAD(1)];
FPR = [metrics_L1(2); metrics_SCAD(2)];
MCC = [metrics_L1(3); metrics_SCAD(3)];

Beta_RMSE = [metrics_L1(4); metrics_SCAD(4)];
Beta_MAD  = [metrics_L1(5); metrics_SCAD(5)];

MSE      = [metrics_L1(6); metrics_SCAD(6)];
Time_sec = [time_L1;        time_SCAD];
Obj_Value = [fval_L1;       fval_SCAD];

Results_Table = table(Method, Lambda, A_param, TPR, FPR, MCC, ...
                      Beta_RMSE, Beta_MAD, MSE, Time_sec, Obj_Value);

fprintf('\n');
disp('============================================================');
disp('           UniLASSO vs UniSCAD COMPARISON');
disp('============================================================');
disp(Results_Table);

% -------- Coefficient Comparison --------
fprintf('\n');
disp('============================================================');
disp('             COEFFICIENT ESTIMATES');
disp('============================================================');

Coef_Table = table((0:p)', beta_true_whole, beta_hat_whole_L1, beta_hat_whole_SCAD, ...
                   'VariableNames', {'Index', 'True', 'UniLASSO', 'UniSCAD'});
disp(Coef_Table);

% -------- Summary --------
fprintf('\n');
disp('============================================================');
disp('                     SUMMARY');
disp('============================================================');
fprintf('Dataset: n=%d, p=%d, true_p=%d\n', n, p, true_p);
fprintf('Lambda: %.4f\n', lambda);
fprintf('SCAD a: %.2f\n\n', a_scad);

fprintf('UniLASSO Performance:\n');
fprintf('  - TPR: %.4f\n',      metrics_L1(1));
fprintf('  - FPR: %.4f\n',      metrics_L1(2));
fprintf('  - MCC: %.4f\n',      metrics_L1(3));
fprintf('  - MSE: %.6f\n',      metrics_L1(6));
fprintf('  - Beta RMSE: %.6f\n',metrics_L1(4));
fprintf('  - Time: %.4f sec\n\n', time_L1);

fprintf('UniSCAD Performance:\n');
fprintf('  - TPR: %.4f\n',      metrics_SCAD(1));
fprintf('  - FPR: %.4f\n',      metrics_SCAD(2));
fprintf('  - MCC: %.4f\n',      metrics_SCAD(3));
fprintf('  - MSE: %.6f\n',      metrics_SCAD(6));
fprintf('  - Beta RMSE: %.6f\n',metrics_SCAD(4));
fprintf('  - Time: %.4f sec\n\n', time_SCAD);

% -------- MSE-based winner --------
mse_diff     = abs(metrics_SCAD(6) - metrics_L1(6));
mse_pct_diff = 100 * mse_diff / max(metrics_L1(6), metrics_SCAD(6));

if mse_pct_diff < 0.01
    fprintf('MSE Result: Essentially equivalent (MSE difference < 0.01%%)\n');
    fprintf('            MSE difference: %.6f\n', mse_diff);
elseif metrics_SCAD(6) < metrics_L1(6)
    fprintf('MSE  Winner: UniSCAD (%.4f%% MSE reduction)\n', ...
            100*(metrics_L1(6) - metrics_SCAD(6))/metrics_L1(6));
else
    fprintf('MSE  Winner: UniLASSO (%.4f%% MSE reduction)\n', ...
            100*(metrics_SCAD(6) - metrics_L1(6))/metrics_SCAD(6));
end

% -------- MCC-based winner --------
if metrics_SCAD(3) > metrics_L1(3)
    fprintf('MCC  Winner: UniSCAD  (%.4f vs %.4f)\n', metrics_SCAD(3), metrics_L1(3));
elseif metrics_SCAD(3) < metrics_L1(3)
    fprintf('MCC  Winner: UniLASSO (%.4f vs %.4f)\n', metrics_L1(3), metrics_SCAD(3));
else
    fprintf('MCC: Tie (%.4f each)\n', metrics_L1(3));
end

disp('============================================================');
