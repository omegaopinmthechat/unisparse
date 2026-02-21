clc; clear;
addpath('./Unisparse/');
addpath('./supp funs/');
addpath('./RMPSH/');
addpath('./Data generation/');
addpath('./other methods/');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                FIXED SIMULATION BLOCK                             %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n      = 200;
p      = 10;
true_p = 5;
nfolds = 2;
tol = 1e-4;

rng(2)
[X, y, beta0_true, beta_true, Sigma] = Generate_data_scenario_homecourt(n, p);
beta_true_whole = [beta0_true; beta_true];
data = split_data(X, y, nfolds);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ---------------- HYPERPARAMETER GRID ----------------
lambda_max = max(abs(X' * y)) / size(X,1);
lambda_min = lambda_max * 1e-4;
lambda_grid = logspace(log10(lambda_max), log10(lambda_min), 10);

nl = length(lambda_grid);

% ---------------- RMPSH OPTIONS ----------------
options.DisplayUpdate = 0;
options.PrintSolution = 0;
options.MaxRuns = 5;
options.TolFun2 = 1e-6;
%options.phi = 1e-6;
options.cutoff = 1e-6;
% ---------------- BOUNDS ----------------
ub_lb_factor = 1e2;
lb = [-ub_lb_factor; zeros(p,1)];
ub = ub_lb_factor*ones(p+1,1);

% ---------------- MCP PARAMETER ----------------
gamma_mcp = 3.0;  % MCP concavity parameter (typical: 2-4)

% ---------------- STORAGE ----------------
Results = zeros(nl, 3 + p);

best_test_mse = inf;
best_lambda   = NaN;

best_test_mse_unilasso = inf;
best_lambda_unilasso   = NaN;

best_test_mse_unimcp = inf;
best_lambda_unimcp   = NaN;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                    CV LOOP ONLY      (UNILASSO)                   %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

q = 1; % for unilasso
for il = 1:nl
    lambda = lambda_grid(il);
    
    fold_train_mse = zeros(nfolds,1);
    fold_test_mse  = zeros(nfolds,1);
    
    for f = 1:nfolds
        fprintf('UniLass CV | lambda = %.2e | fold %d/%d\n',...
            lambda, f, nfolds);
        
        Xtr = data.train{f}.X;
        ytr = data.train{f}.y;
        Xte = data.test{f}.X;
        yte = data.test{f}.y;
        
        [b0, b, ~, ~, eta_loo] = unisparse_univreg(Xtr, ytr);
        
        theta_init = ones(p,1);
        psi0 = [0; theta_init];
        
        objFun = @(psi) unilasso_objective_given_eta_loo( ...
            psi, eta_loo, ytr, lambda);
        
        [x_opt, ~, ~] = RMPSH(objFun, psi0, lb, ub, options);
        
        theta0_hat = x_opt(1);
        theta_hat  = x_opt(2:end);
        
        gamma_hat  = b(:) .* theta_hat(:);
        gamma0_hat = theta0_hat + sum(b0(:).*theta_hat(:));
        
        yhat_tr = gamma0_hat + Xtr * gamma_hat;
        yhat_te = gamma0_hat + Xte * gamma_hat;
        
        fold_train_mse(f) = mean((ytr - yhat_tr).^2);
        fold_test_mse(f)  = mean((yte - yhat_te).^2);
    end
    
    train_mse = mean(fold_train_mse);
    test_mse  = mean(fold_test_mse);
    
    Results(il,:) = [lambda train_mse test_mse zeros(1,p)];
    
    
    % ---- UNILASSO (q = 1) ----
    if test_mse < best_test_mse_unilasso
        best_test_mse_unilasso = test_mse;
        best_lambda_unilasso   = lambda;
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                    UniMCP CV LOOP                                %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\n========== Starting UniMCP Cross-Validation ==========\n');
for il = 1:nl
    lambda = lambda_grid(il);
    
    fold_train_mse = zeros(nfolds,1);
    fold_test_mse  = zeros(nfolds,1);
    
    for f = 1:nfolds
        fprintf('UniMCP CV | lambda = %.2e | fold %d/%d\n',...
            lambda, f, nfolds);
        
        Xtr = data.train{f}.X;
        ytr = data.train{f}.y;
        Xte = data.test{f}.X;
        yte = data.test{f}.y;
        
        [b0, b, ~, ~, eta_loo] = unisparse_univreg(Xtr, ytr);
        
        theta_init = ones(p,1);
        psi0 = [0; theta_init];
        
        % UniMCP objective with gamma parameter
        objFun = @(psi) uniMCP_objective_given_eta_loo( ...
            psi, eta_loo, ytr, lambda, gamma_mcp);
        
        [x_opt, ~, ~] = RMPSH(objFun, psi0, lb, ub, options);
        
        theta0_hat = x_opt(1);
        theta_hat  = x_opt(2:end);
        
        gamma_hat  = b(:) .* theta_hat(:);
        gamma0_hat = theta0_hat + sum(b0(:).*theta_hat(:));
        
        yhat_tr = gamma0_hat + Xtr * gamma_hat;
        yhat_te = gamma0_hat + Xte * gamma_hat;
        
        fold_train_mse(f) = mean((ytr - yhat_tr).^2);
        fold_test_mse(f)  = mean((yte - yhat_te).^2);
    end
    
    train_mse = mean(fold_train_mse);
    test_mse  = mean(fold_test_mse);
    
    % Track best lambda for UniMCP
    if test_mse < best_test_mse_unimcp
        best_test_mse_unimcp = test_mse;
        best_lambda_unimcp   = lambda;
    end
end

    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                     UNILASSO FINAL REFIT                         %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nRefitting UniLASSO on full data (lambda=%.2e)\n', ...
        best_lambda_unilasso);

% ---- 1. Univariate regressions + LOO on full data ----
[b0, b, ~, ~, eta_loo] = unisparse_univreg(X, y);

% ---- 2. Initial values ----
theta_init = ones(p,1);
psi0 = [mean(b0); theta_init];

% ---- 3. Objective function ----
objFun_L1 = @(psi) unilasso_objective_given_eta_loo( ...
                      psi, eta_loo, y, best_lambda_unilasso);

% ---- 4. Optimize ----
[x_opt_L1, ~, ~] = RMPSH(objFun_L1, psi0, lb, ub, options);

theta0_hat_L1 = x_opt_L1(1);
theta_hat_L1  = x_opt_L1(2:end);

% ---- 5. Recover final UniLASSO coefficients ----
gamma_hat_L1  = b(:) .* theta_hat_L1(:);
gamma0_hat_L1 = theta0_hat_L1 + sum(b0(:).*theta_hat_L1(:));

beta_hat_whole_unilasso = [gamma0_hat_L1; gamma_hat_L1];

% ---- 6. Predictions & metrics ----
yhat_full_unilasso = gamma0_hat_L1 + X * gamma_hat_L1;

metrics_row_unilasso = compute_sparse_metrics( ...
    beta_hat_whole_unilasso, beta_true_whole, ...
    yhat_full_unilasso, y, tol);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                     UniMCP FINAL REFIT                           %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('\nRefitting UniMCP on full data (lambda=%.2e, gamma=%.1f)\n', ...
        best_lambda_unimcp, gamma_mcp);

% ---- 1. Univariate regressions + LOO on full data ----
[b0, b, ~, ~, eta_loo] = unisparse_univreg(X, y);

% ---- 2. Initial values ----
theta_init = ones(p,1);
psi0 = [mean(b0); theta_init];

% ---- 3. Objective function with MCP penalty ----
objFun_MCP = @(psi) uniMCP_objective_given_eta_loo( ...
                      psi, eta_loo, y, best_lambda_unimcp, gamma_mcp);

% ---- 4. Optimize ----
[x_opt_MCP, ~, ~] = RMPSH(objFun_MCP, psi0, lb, ub, options);

theta0_hat_MCP = x_opt_MCP(1);
theta_hat_MCP  = x_opt_MCP(2:end);

% ---- 5. Recover final UniMCP coefficients ----
gamma_hat_MCP  = b(:) .* theta_hat_MCP(:);
gamma0_hat_MCP = theta0_hat_MCP + sum(b0(:).*theta_hat_MCP(:));

beta_hat_whole_unimcp = [gamma0_hat_MCP; gamma_hat_MCP];

% ---- 6. Predictions & metrics ----
yhat_full_unimcp = gamma0_hat_MCP + X * gamma_hat_MCP;

metrics_row_unimcp = compute_sparse_metrics( ...
    beta_hat_whole_unimcp, beta_true_whole, ...
    yhat_full_unimcp, y, tol);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                      BASELINE METHODS                           %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Methods = {'LASSO','AdapLASSO','SCAD','MCP'};
nMethods = numel(Methods);
Results_baseline = struct();

for m = 1:nMethods
    method_name = Methods{m};

    best_test_mse = inf;
    best_lambda   = NaN;

    for il = 1:nl
        lambda = lambda_grid(il);

        fold_test_mse = zeros(nfolds,1);

        for f = 1:nfolds
            Xtr = data.train{f}.X;
            ytr = data.train{f}.y;
            Xte = data.test{f}.X;
            yte = data.test{f}.y;

            switch method_name
                case 'LASSO'
                    [b0, b] = fit_lasso(Xtr, ytr, lambda);
                case 'AdapLASSO'
                    [b0, b] = fit_adap_lasso(Xtr, ytr, lambda);
                case 'SCAD'
                    [b0, b] = fit_scad_LLA(Xtr, ytr, lambda);
                case 'MCP'
                    [b0, b] = fit_mcp_LLA(Xtr, ytr, lambda);
            end

            yhat_te = b0 + Xte*b;
            fold_test_mse(f) = mean((yte - yhat_te).^2);
        end

        test_mse = mean(fold_test_mse);

        if test_mse < best_test_mse
            best_test_mse = test_mse;
            best_lambda   = lambda;
        end
    end

    % ---- FULL DATA REFIT ----
    switch method_name
        case 'LASSO'
            [b0, b] = fit_lasso(X, y, best_lambda);
        case 'AdapLASSO'
            [b0, b] = fit_adap_lasso(X, y, best_lambda);
        case 'SCAD'
            [b0, b] = fit_scad_LLA(X, y, best_lambda);
        case 'MCP'
            [b0, b] = fit_mcp_LLA(X, y, best_lambda);
    end

    beta_hat_whole = [b0; b];
    yhat_full = b0 + X*b;

    metrics = compute_sparse_metrics( ...
        beta_hat_whole, beta_true_whole, ...
        yhat_full, y, tol);

    Results_baseline.(method_name).lambda  = best_lambda;
    Results_baseline.(method_name).metrics = metrics;
    Results_baseline.(method_name).beta    = beta_hat_whole;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%                 FINAL COMPARISON TABLE                         %%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Method = ["UniLASSO";"UniMCP";"LASSO";"AdapLASSO";"SCAD";"MCP"];

Lambda = [
    best_lambda_unilasso
    best_lambda_unimcp
    Results_baseline.LASSO.lambda
    Results_baseline.AdapLASSO.lambda
    Results_baseline.SCAD.lambda
    Results_baseline.MCP.lambda
];

Gamma = [
    nan
    gamma_mcp
    nan
    nan
    nan
    nan
];

q = [
    1
    nan
    nan
    nan
    nan
    nan
];

TPR = [
    metrics_row_unilasso(1)
    metrics_row_unimcp(1)
    Results_baseline.LASSO.metrics(1)
    Results_baseline.AdapLASSO.metrics(1)
    Results_baseline.SCAD.metrics(1)
    Results_baseline.MCP.metrics(1)
];

FPR = [
    metrics_row_unilasso(2)
    metrics_row_unimcp(2)
    Results_baseline.LASSO.metrics(2)
    Results_baseline.AdapLASSO.metrics(2)
    Results_baseline.SCAD.metrics(2)
    Results_baseline.MCP.metrics(2)
];

MCC = [
    metrics_row_unilasso(3)
    metrics_row_unimcp(3)
    Results_baseline.LASSO.metrics(3)
    Results_baseline.AdapLASSO.metrics(3)
    Results_baseline.SCAD.metrics(3)
    Results_baseline.MCP.metrics(3)
];

Beta_RMSE = [
    metrics_row_unilasso(4)
    metrics_row_unimcp(4)
    Results_baseline.LASSO.metrics(4)
    Results_baseline.AdapLASSO.metrics(4)
    Results_baseline.SCAD.metrics(4)
    Results_baseline.MCP.metrics(4)
];

Beta_MAD = [
    metrics_row_unilasso(5)
    metrics_row_unimcp(5)
    Results_baseline.LASSO.metrics(5)
    Results_baseline.AdapLASSO.metrics(5)
    Results_baseline.SCAD.metrics(5)
    Results_baseline.MCP.metrics(5)
];

Full_MSE = [
    metrics_row_unilasso(6)
    metrics_row_unimcp(6)
    Results_baseline.LASSO.metrics(6)
    Results_baseline.AdapLASSO.metrics(6)
    Results_baseline.SCAD.metrics(6)
    Results_baseline.MCP.metrics(6)
];

Results_All = table(Method, Lambda, Gamma, TPR, FPR, MCC, ...
                    Beta_RMSE, Beta_MAD, Full_MSE);

disp('================ FINAL METHOD COMPARISON ================');
disp(Results_All);


%% ===================== COEFFICIENT DISPLAY =====================

disp('======================================================');
disp(' TRUE COEFFICIENT VECTOR  [beta0_true ; beta_true]');
disp('======================================================');
disp(beta_true_whole');

disp('======================================================');
disp(' UNILASSO ESTIMATED COEFFICIENTS  [beta0_hat ; beta_hat]');
disp('======================================================');
disp(beta_hat_whole_unilasso');

disp('======================================================');
disp(' UniMCP ESTIMATED COEFFICIENTS  [beta0_hat ; beta_hat]');
disp('======================================================');
disp(beta_hat_whole_unimcp');

disp('======================================================');
disp(' LASSO ESTIMATED COEFFICIENTS  [beta0_hat ; beta_hat]');
disp('======================================================');
disp(Results_baseline.LASSO.beta');

disp('======================================================');
disp(' ADAPTIVE LASSO ESTIMATED COEFFICIENTS  [beta0_hat ; beta_hat]');
disp('======================================================');
disp(Results_baseline.AdapLASSO.beta');

disp('======================================================');
disp(' SCAD ESTIMATED COEFFICIENTS  [beta0_hat ; beta_hat]');
disp('======================================================');
disp(Results_baseline.SCAD.beta');

disp('======================================================');
disp(' MCP ESTIMATED COEFFICIENTS  [beta0_hat ; beta_hat]');
disp('======================================================');
disp(Results_baseline.MCP.beta');

