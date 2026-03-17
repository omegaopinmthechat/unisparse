function results = unisparse(X, y, lambda_range, nfolds, method, rmps_lb, rmps_ub, rmps_x0, rmps_options, a, gamma)
% UNISPARSE_CV  Cross-validated UniSparse estimators (UniLASSO / UniMCP / UniSCAD)
%
% Usage:
%   results = unisparse_cv(X, y)
%   results = unisparse_cv(X, y, lambda_range, nfolds, method, rmps_lb, rmps_ub, rmps_x0, rmps_options, a, gamma)
%
% Inputs (most are optional - sensible defaults are used):
%   X             - n x p design matrix
%   y             - n x 1 response
%   lambda_range  - either a vector of lambda values or two-element [min,max]
%                   (default: [1e-5,1e5] -> 50 log-spaced values)
%   nfolds        - number of CV folds (default: 2)
%   method        - 'unilasso', 'unimcp', 'uniscad' or 'all' (default: 'all')
%   rmps_lb       - RMPS lower bounds vector (default: -100 for intercept, 0 for slopes)
%   rmps_ub       - RMPS upper bounds vector (default: 100 for all)
%   rmps_x0       - RMPS starting point (default: [mean(b0); ones(p,1)])
%   rmps_options  - struct of RMPS options (defaults set inside RMPSH)
%   a             - SCAD concavity parameter (default: 3.7)
%   gamma         - MCP concavity parameter (default: 3.0)

% Output:
%   results - struct containing fields for each method run: lambda_grid, best_lambda,
%             beta_hat (final [beta0; beta]), rmps_options used, and metrics.

if nargin < 3 || isempty(lambda_range)
    lambda_range = [1e-5, 1e5];
end
if nargin < 4 || isempty(nfolds)
    nfolds = 2;
end
if nargin < 5 || isempty(method)
    method = 'all';
end
if nargin < 6
    rmps_lb = [];
end
if nargin < 7
    rmps_ub = [];
end
if nargin < 8
    rmps_x0 = [];
end
if nargin < 9
    rmps_options = [];
end
if nargin < 10 || isempty(a)
    a = 3.7;
end
if nargin < 11 || isempty(gamma)
    gamma = 3.0;
end

% --- Prepare lambda grid ---
if ~iscell(lambda_range)
    if numel(lambda_range) == 2
        lambda_grid = logspace(log10(lambda_range(1)), log10(lambda_range(2)), 50);
    else
        lambda_grid = lambda_range(:)';
    end
    nl = length(lambda_grid);
else
    lambda_grid = {};
    nl = 0;
end

[n, p] = size(X);

% Ensure helper folders are on the path so dependent functions are available.
try
    thisFile = mfilename('fullpath');
    repoRoot = fileparts(fileparts(thisFile));
    addpath(fullfile(repoRoot, 'supp funs'));
    addpath(fullfile(repoRoot, 'other methods'));
    if exist(fullfile(repoRoot,'RMPSH'),'dir')
        addpath(fullfile(repoRoot,'RMPSH'));
    end

    % Propagate to workers if a parallel pool exists
    pool_for_path = gcp('nocreate');
    if ~isempty(pool_for_path)
        pctRunOnAll(sprintf('addpath(''%s'')', fullfile(repoRoot, 'supp funs')));
        pctRunOnAll(sprintf('addpath(''%s'')', fullfile(repoRoot, 'other methods')));
        if exist(fullfile(repoRoot,'RMPSH'),'dir')
            pctRunOnAll(sprintf('addpath(''%s'')', fullfile(repoRoot, 'RMPSH')));
        end
    end
catch
    % non-fatal if path propagation fails
end

% --- Default RMPS bounds / options ---
ub_lb_factor = 1e2;
if isempty(rmps_lb)
    rmps_lb = [-ub_lb_factor; zeros(p,1)];
end
if isempty(rmps_ub)
    rmps_ub = ub_lb_factor * ones(p+1,1);
end

% RMPS options default merges inside RMPSH; override a few to be quiet by default
if isempty(rmps_options)
    rmps_options.DisplayUpdate = 0;
    rmps_options.PrintSolution = 0;
    rmps_options.MaxRuns = 5;
    rmps_options.TolFun2 = 1e-6;
    rmps_options.cutoff = 1e-6;
end

% Methods to run
if ischar(method)
    method = lower(method);
    if strcmp(method, 'all')
        run_unilasso = true; run_unimcp = true; run_uniscad = true;
    else
        run_unilasso = strcmp(method,'unilasso');
        run_unimcp   = strcmp(method,'unimcp');
        run_uniscad  = strcmp(method,'uniscad');
    end
else
    run_unilasso = any(strcmpi(method,'unilasso'));
    run_unimcp   = any(strcmpi(method,'unimcp'));
    run_uniscad  = any(strcmpi(method,'uniscad'));
end

results = struct();

% Ensure a parallel pool exists (mandatory multi-core CPU parallelism)
try
    pool = gcp('nocreate');
    if isempty(pool)
        parpool('Processes');
    end
catch
    % If starting a pool fails, continue serially
end

% Report parallel pool status
try
    pool_info = gcp('nocreate');
    if ~isempty(pool_info)
        fprintf('Parallel pool: active with %d workers.\n', pool_info.NumWorkers);
    else
        fprintf('Parallel pool: none (running serially).\n');
    end
catch
    fprintf('Parallel pool: status unknown.\n');
end

% Decide if we should run a parameter sweep
sweep_flag = false;
if iscell(method) && numel(method) > 1
    sweep_flag = true;
end
if iscell(lambda_range) && numel(lambda_range) > 1
    sweep_flag = true;
end
if numel(nfolds) > 1
    sweep_flag = true;
end
if numel(a) > 1
    sweep_flag = true;
end
if numel(gamma) > 1
    sweep_flag = true;
end
if iscell(rmps_lb) && numel(rmps_lb) > 1
    sweep_flag = true;
end
if iscell(rmps_x0) && numel(rmps_x0) > 1
    sweep_flag = true;
end

if sweep_flag
    % Build all parameter lists
    if iscell(method),       methods_list  = method;        else, methods_list  = {method};        end
    if iscell(lambda_range), lambda_list   = lambda_range;  else, lambda_list   = {lambda_range};  end
    if iscell(rmps_lb),      rmps_lb_list  = rmps_lb;       else, rmps_lb_list  = {rmps_lb};       end
    if iscell(rmps_ub),      rmps_ub_list  = rmps_ub;       else, rmps_ub_list  = {rmps_ub};       end
    if iscell(rmps_x0),      rmps_x0_list  = rmps_x0;       else, rmps_x0_list  = {rmps_x0};       end
    nfolds_list = nfolds(:)';
    a_list      = a(:)';
    gamma_list  = gamma(:)';

    % Flat cartesian product via ndgrid
    [IM, IL, IF, IA, IG, ILB, IUB, IX0] = ndgrid( ...
        1:numel(methods_list),  1:numel(lambda_list), ...
        1:numel(nfolds_list),   1:numel(a_list), ...
        1:numel(gamma_list),    1:numel(rmps_lb_list), ...
        1:numel(rmps_ub_list),  1:numel(rmps_x0_list));

    combos = [IM(:), IL(:), IF(:), IA(:), IG(:), ILB(:), IUB(:), IX0(:)];
    total_runs = size(combos, 1);
    fprintf('Running parameter sweep: %d runs\n', total_runs);

    results_all = struct();

    for run_idx = 1:total_runs
        c = combos(run_idx, :);
        cur_method   = methods_list{c(1)};
        cur_lambda   = lambda_list{c(2)};
        cur_nfolds   = nfolds_list(c(3));
        cur_a        = a_list(c(4));
        cur_gamma    = gamma_list(c(5));
        cur_rmps_lb  = rmps_lb_list{c(6)};
        cur_rmps_ub  = rmps_ub_list{c(7)};
        cur_rmps_x0  = rmps_x0_list{c(8)};

        fprintf('[%d/%d] method=%s folds=%d lambda=[%g,%g] a=%.2f gamma=%.2f\n', ...
            run_idx, total_runs, cur_method, cur_nfolds, cur_lambda(1), cur_lambda(end), cur_a, cur_gamma);

        try
            out = unisparse_cv_single(X, y, cur_lambda, cur_nfolds, cur_method, ...
                      cur_rmps_lb, cur_rmps_ub, cur_rmps_x0, rmps_options, cur_a, cur_gamma);
            results_all(run_idx).success = true;
            results_all(run_idx).output  = out;
            results_all(run_idx).error   = [];
        catch ME
            results_all(run_idx).success = false;
            results_all(run_idx).output  = [];
            results_all(run_idx).error.message = ME.message;
            results_all(run_idx).error.stack   = ME.stack;
            fprintf('  -> Error: %s\n', ME.message);
        end

        results_all(run_idx).params = struct('method', cur_method, 'lambda_range', cur_lambda, ...
            'nfolds', cur_nfolds, 'a', cur_a, 'gamma', cur_gamma, ...
            'rmps_lb', cur_rmps_lb, 'rmps_x0', cur_rmps_x0);

        if mod(run_idx, 10) == 0 || run_idx == total_runs
            save('unisparse_cv_sweep_results.mat', 'results_all', '-v7.3');
        end
    end

    results.SWEEP = results_all;
    save('unisparse_cv_sweep_results.mat', 'results_all', '-v7.3');
    return;
end

% Helper to perform CV for a given objective wrapper
if ~sweep_flag
    data = split_data(X, y, nfolds);
end

function best_lambda = cv_for_objective(obj_fun_builder)
    best_mse = inf;
    best_lambda = NaN;

    train_idx_cell = cell(nfolds,1);
    test_idx_cell  = cell(nfolds,1);
    B_mat  = zeros(p, nfolds);
    B0_mat = zeros(p, nfolds);
    ETA    = cell(nfolds,1);
    PSI0   = cell(nfolds,1);

    for f = 1:nfolds
        train_idx = data.train_idx{f};
        test_idx  = data.test_idx{f};
        train_idx_cell{f} = train_idx;
        test_idx_cell{f}  = test_idx;

        Xtr = X(train_idx, :);
        ytr = y(train_idx);
        [b0_fold, b_fold, ~, ~, eta_loo_fold] = unisparse_univreg(Xtr, ytr);

        B_mat(:,f)  = b_fold(:);
        B0_mat(:,f) = b0_fold(:);
        ETA{f}      = eta_loo_fold;

        if isempty(rmps_x0)
            PSI0{f} = [mean(b0_fold); ones(p,1)];
        else
            PSI0{f} = rmps_x0;
        end
    end

    use_parallel = license('test','Distrib_Computing_Toolbox');

    for il = 1:nl
        lambda = lambda_grid(il);
        fold_test_mse = zeros(nfolds,1);

        if use_parallel
            parfor f = 1:nfolds
                Xte     = X(test_idx_cell{f}, :);
                yte     = y(test_idx_cell{f});
                b0      = B0_mat(:,f);
                b       = B_mat(:,f);
                eta_loo = ETA{f};
                psi0    = PSI0{f};

                objFun = obj_fun_builder(psi0, eta_loo, y(train_idx_cell{f}), lambda);
                [x_opt, ~, ~] = RMPSH(objFun, psi0, rmps_lb, rmps_ub, rmps_options);

                theta0_hat = x_opt(1);
                theta_hat  = x_opt(2:end);
                gamma_hat  = b(:) .* theta_hat(:);
                gamma0_hat = theta0_hat + sum(b0(:) .* theta_hat(:));

                yhat_te = gamma0_hat + Xte * gamma_hat;
                fold_test_mse(f) = mean((yte - yhat_te).^2);
            end
        else
            for f = 1:nfolds
                Xte     = X(test_idx_cell{f}, :);
                yte     = y(test_idx_cell{f});
                b0      = B0_mat(:,f);
                b       = B_mat(:,f);
                eta_loo = ETA{f};
                psi0    = PSI0{f};

                objFun = obj_fun_builder(psi0, eta_loo, y(train_idx_cell{f}), lambda);
                [x_opt, ~, ~] = RMPSH(objFun, psi0, rmps_lb, rmps_ub, rmps_options);

                theta0_hat = x_opt(1);
                theta_hat  = x_opt(2:end);
                gamma_hat  = b(:) .* theta_hat(:);
                gamma0_hat = theta0_hat + sum(b0(:) .* theta_hat(:));

                yhat_te = gamma0_hat + Xte * gamma_hat;
                fold_test_mse(f) = mean((yte - yhat_te).^2);
            end
        end

        test_mse = mean(fold_test_mse);
        if test_mse < best_mse
            best_mse    = test_mse;
            best_lambda = lambda;
        end
    end
end

% --- Run UniLASSO CV and final fit ---
if run_unilasso
    fprintf('========== UniLASSO CV (nfolds=%d) ==========\n', nfolds);
    obj_builder = @(psi0, eta_loo, ytr, lambda) (@(psi) unilasso_objective_given_eta_loo(psi, eta_loo, ytr, lambda));
    best_lambda_unilasso = cv_for_objective(obj_builder);

    [b0, b, ~, ~, eta_loo] = unisparse_univreg(X, y);
    if isempty(rmps_x0)
        psi0 = [mean(b0); ones(p,1)];
    else
        psi0 = rmps_x0;
    end
    objFun_L1 = @(psi) unilasso_objective_given_eta_loo(psi, eta_loo, y, best_lambda_unilasso);
    [x_opt_L1, ~, ~] = RMPSH(objFun_L1, psi0, rmps_lb, rmps_ub, rmps_options);
    theta0_hat_L1 = x_opt_L1(1);
    theta_hat_L1  = x_opt_L1(2:end);
    gamma_hat_L1  = b(:) .* theta_hat_L1(:);
    gamma0_hat_L1 = theta0_hat_L1 + sum(b0(:) .* theta_hat_L1(:));

    results.UNILASSO.lambda       = best_lambda_unilasso;
    results.UNILASSO.beta         = [gamma0_hat_L1; gamma_hat_L1];
    results.UNILASSO.rmps_options = rmps_options;
end

% --- Run UniMCP CV and final fit ---
if run_unimcp
    fprintf('========== UniMCP CV (nfolds=%d) ==========\n', nfolds);
    obj_builder = @(psi0, eta_loo, ytr, lambda) (@(psi) uniMCP_objective_given_eta_loo(psi, eta_loo, ytr, lambda, gamma));
    best_lambda_unimcp = cv_for_objective(obj_builder);

    [b0, b, ~, ~, eta_loo] = unisparse_univreg(X, y);
    if isempty(rmps_x0)
        psi0 = [mean(b0); ones(p,1)];
    else
        psi0 = rmps_x0;
    end
    objFun_MCP = @(psi) uniMCP_objective_given_eta_loo(psi, eta_loo, y, best_lambda_unimcp, gamma);
    [x_opt_MCP, ~, ~] = RMPSH(objFun_MCP, psi0, rmps_lb, rmps_ub, rmps_options);
    theta0_hat_MCP = x_opt_MCP(1);
    theta_hat_MCP  = x_opt_MCP(2:end);
    gamma_hat_MCP  = b(:) .* theta_hat_MCP(:);
    gamma0_hat_MCP = theta0_hat_MCP + sum(b0(:) .* theta_hat_MCP(:));

    results.UNIMCP.lambda       = best_lambda_unimcp;
    results.UNIMCP.beta         = [gamma0_hat_MCP; gamma_hat_MCP];
    results.UNIMCP.gamma        = gamma;
    results.UNIMCP.rmps_options = rmps_options;
end

% --- Run UniSCAD CV and final fit ---
if run_uniscad
    fprintf('========== UniSCAD CV (nfolds=%d) ==========\n', nfolds);
    obj_builder = @(psi0, eta_loo, ytr, lambda) (@(psi) uniSCAD_objective_given_eta_loo(psi, eta_loo, ytr, lambda, a));
    best_lambda_uniscad = cv_for_objective(obj_builder);

    [b0, b, ~, ~, eta_loo] = unisparse_univreg(X, y);
    if isempty(rmps_x0)
        psi0 = [mean(b0); ones(p,1)];
    else
        psi0 = rmps_x0;
    end
    objFun_SCAD = @(psi) uniSCAD_objective_given_eta_loo(psi, eta_loo, y, best_lambda_uniscad, a);
    [x_opt_SCAD, ~, ~] = RMPSH(objFun_SCAD, psi0, rmps_lb, rmps_ub, rmps_options);
    theta0_hat_SCAD = x_opt_SCAD(1);
    theta_hat_SCAD  = x_opt_SCAD(2:end);
    gamma_hat_SCAD  = b(:) .* theta_hat_SCAD(:);
    gamma0_hat_SCAD = theta0_hat_SCAD + sum(b0(:) .* theta_hat_SCAD(:));

    results.UNISCAD.lambda       = best_lambda_uniscad;
    results.UNISCAD.beta         = [gamma0_hat_SCAD; gamma_hat_SCAD];
    results.UNISCAD.a            = a;
    results.UNISCAD.rmps_options = rmps_options;
end

% --- Display results ---
if isfield(results,'UNILASSO')
    disp('======================================================');
    disp(' UNILASSO ESTIMATED COEFFICIENTS  [beta0_hat ; beta_hat]');
    disp('======================================================');
    disp(results.UNILASSO.beta');
end
if isfield(results,'UNIMCP')
    disp('======================================================');
    disp(' UNIMCP ESTIMATED COEFFICIENTS  [beta0_hat ; beta_hat]');
    disp('======================================================');
    disp(results.UNIMCP.beta');
end
if isfield(results,'UNISCAD')
    disp('======================================================');
    disp(' UNISCAD ESTIMATED COEFFICIENTS  [beta0_hat ; beta_hat]');
    disp('======================================================');
    disp(results.UNISCAD.beta');
end

end

% -------------------------------------------------------------------------
function out = unisparse_cv_single(X, y, lambda_range, nfolds, method, rmps_lb, rmps_ub, rmps_x0, rmps_options, a, gamma)
% Runs a single configuration (no sweeping) and returns results struct
    if nargin < 3 || isempty(lambda_range)
        lambda_range = [1e-5,1e5];
    end
    if nargin < 4 || isempty(nfolds)
        nfolds = 2;
    end
    if nargin < 5 || isempty(method)
        method = 'all';
    end
    if nargin < 8
        rmps_x0 = [];
    end
    if nargin < 9
        rmps_options = [];
    end
    if nargin < 10 || isempty(a)
        a = 3.7;
    end
    if nargin < 11 || isempty(gamma)
        gamma = 3.0;
    end

    results_local = struct();

    % Prepare lambda grid
    if numel(lambda_range) == 2
        lambda_grid = logspace(log10(lambda_range(1)), log10(lambda_range(2)), 50);
    else
        lambda_grid = lambda_range(:)';
    end

    % Data split
    data_local = split_data(X, y, nfolds);

    [~, ploc] = size(X);

    % Default RMPS bounds if empty
    ub_lb_factor = 1e2;
    if isempty(rmps_lb)
        rmps_lb = [-ub_lb_factor; zeros(ploc,1)];
    end
    if isempty(rmps_ub)
        rmps_ub = ub_lb_factor * ones(ploc+1,1);
    end
    if isempty(rmps_options)
        rmps_options.DisplayUpdate = 0;
        rmps_options.PrintSolution = 0;
        rmps_options.MaxRuns = 5;
        rmps_options.TolFun2 = 1e-6;
        rmps_options.cutoff = 1e-6;
    end

    % Determine which methods to run
    if ischar(method)
        method = lower(method);
        run_unil = strcmp(method,'all') || strcmp(method,'unilasso');
        run_mcp  = strcmp(method,'all') || strcmp(method,'unimcp');
        run_scad = strcmp(method,'all') || strcmp(method,'uniscad');
    else
        run_unil = any(strcmpi(method,'unilasso'));
        run_mcp  = any(strcmpi(method,'unimcp'));
        run_scad = any(strcmpi(method,'uniscad'));
    end

    function best_lambda = cv_run_local(obj_builder)
        best_mse = inf; best_lambda = NaN;

        train_idx_cell = cell(nfolds,1);
        test_idx_cell  = cell(nfolds,1);
        B_mat  = zeros(ploc, nfolds);
        B0_mat = zeros(ploc, nfolds);
        ETA    = cell(nfolds,1);
        PSI0   = cell(nfolds,1);

        for f = 1:nfolds
            train_idx = data_local.train_idx{f};
            test_idx  = data_local.test_idx{f};
            train_idx_cell{f} = train_idx;
            test_idx_cell{f}  = test_idx;

            Xtr = X(train_idx, :);
            ytr = y(train_idx);
            [b0_fold, b_fold, ~, ~, eta_loo_fold] = unisparse_univreg(Xtr, ytr);

            B_mat(:,f)  = b_fold(:);
            B0_mat(:,f) = b0_fold(:);
            ETA{f}      = eta_loo_fold;

            if isempty(rmps_x0)
                PSI0{f} = [mean(b0_fold); ones(ploc,1)];
            else
                PSI0{f} = rmps_x0;
            end
        end

        use_parallel = license('test','Distrib_Computing_Toolbox');

        for il = 1:length(lambda_grid)
            lambda = lambda_grid(il);
            fold_test_mse = zeros(nfolds,1);

            if use_parallel
                parfor f = 1:nfolds
                    Xte     = X(test_idx_cell{f}, :);
                    yte     = y(test_idx_cell{f});
                    b0      = B0_mat(:,f);
                    b       = B_mat(:,f);
                    eta_loo = ETA{f};
                    psi0    = PSI0{f};

                    objFun = obj_builder(psi0, eta_loo, y(train_idx_cell{f}), lambda);
                    [x_opt, ~, ~] = RMPSH(objFun, psi0, rmps_lb, rmps_ub, rmps_options);

                    theta0_hat = x_opt(1);
                    theta_hat  = x_opt(2:end);
                    gamma_hat  = b(:) .* theta_hat(:);
                    gamma0_hat = theta0_hat + sum(b0(:) .* theta_hat(:));

                    yhat_te = gamma0_hat + Xte * gamma_hat;
                    fold_test_mse(f) = mean((yte - yhat_te).^2);
                end
            else
                for f = 1:nfolds
                    Xte     = X(test_idx_cell{f}, :);
                    yte     = y(test_idx_cell{f});
                    b0      = B0_mat(:,f);
                    b       = B_mat(:,f);
                    eta_loo = ETA{f};
                    psi0    = PSI0{f};

                    objFun = obj_builder(psi0, eta_loo, y(train_idx_cell{f}), lambda);
                    [x_opt, ~, ~] = RMPSH(objFun, psi0, rmps_lb, rmps_ub, rmps_options);

                    theta0_hat = x_opt(1);
                    theta_hat  = x_opt(2:end);
                    gamma_hat  = b(:) .* theta_hat(:);
                    gamma0_hat = theta0_hat + sum(b0(:) .* theta_hat(:));

                    yhat_te = gamma0_hat + Xte * gamma_hat;
                    fold_test_mse(f) = mean((yte - yhat_te).^2);
                end
            end

            test_mse = mean(fold_test_mse);
            if test_mse < best_mse
                best_mse    = test_mse;
                best_lambda = lambda;
            end
        end
    end

    % Run each requested method and final refit
    if run_unil
        obj_builder = @(psi0, eta_loo, ytr, lambda) (@(psi) unilasso_objective_given_eta_loo(psi, eta_loo, ytr, lambda));
        best_lambda_unil = cv_run_local(obj_builder);
        [b0, b, ~, ~, eta_loo] = unisparse_univreg(X, y);
        if isempty(rmps_x0)
            psi0 = [mean(b0); ones(ploc,1)];
        else
            psi0 = rmps_x0;
        end
        objFun_L1 = @(psi) unilasso_objective_given_eta_loo(psi, eta_loo, y, best_lambda_unil);
        [x_opt_L1, ~, ~] = RMPSH(objFun_L1, psi0, rmps_lb, rmps_ub, rmps_options);
        theta0_hat_L1 = x_opt_L1(1);
        theta_hat_L1  = x_opt_L1(2:end);
        gamma_hat_L1  = b(:) .* theta_hat_L1(:);
        gamma0_hat_L1 = theta0_hat_L1 + sum(b0(:) .* theta_hat_L1(:));
        results_local.UNILASSO.lambda = best_lambda_unil;
        results_local.UNILASSO.beta   = [gamma0_hat_L1; gamma_hat_L1];
    end

    if run_mcp
        obj_builder = @(psi0, eta_loo, ytr, lambda) (@(psi) uniMCP_objective_given_eta_loo(psi, eta_loo, ytr, lambda, gamma));
        best_lambda_mcp = cv_run_local(obj_builder);
        [b0, b, ~, ~, eta_loo] = unisparse_univreg(X, y);
        if isempty(rmps_x0)
            psi0 = [mean(b0); ones(ploc,1)];
        else
            psi0 = rmps_x0;
        end
        objFun_MCP = @(psi) uniMCP_objective_given_eta_loo(psi, eta_loo, y, best_lambda_mcp, gamma);
        [x_opt_MCP, ~, ~] = RMPSH(objFun_MCP, psi0, rmps_lb, rmps_ub, rmps_options);
        theta0_hat_MCP = x_opt_MCP(1);
        theta_hat_MCP  = x_opt_MCP(2:end);
        gamma_hat_MCP  = b(:) .* theta_hat_MCP(:);
        gamma0_hat_MCP = theta0_hat_MCP + sum(b0(:) .* theta_hat_MCP(:));
        results_local.UNIMCP.lambda = best_lambda_mcp;
        results_local.UNIMCP.beta   = [gamma0_hat_MCP; gamma_hat_MCP];
        results_local.UNIMCP.gamma  = gamma;
    end

    if run_scad
        obj_builder = @(psi0, eta_loo, ytr, lambda) (@(psi) uniSCAD_objective_given_eta_loo(psi, eta_loo, ytr, lambda, a));
        best_lambda_scad = cv_run_local(obj_builder);
        [b0, b, ~, ~, eta_loo] = unisparse_univreg(X, y);
        if isempty(rmps_x0)
            psi0 = [mean(b0); ones(ploc,1)];
        else
            psi0 = rmps_x0;
        end
        objFun_SCAD = @(psi) uniSCAD_objective_given_eta_loo(psi, eta_loo, y, best_lambda_scad, a);
        [x_opt_SCAD, ~, ~] = RMPSH(objFun_SCAD, psi0, rmps_lb, rmps_ub, rmps_options);
        theta0_hat_SCAD = x_opt_SCAD(1);
        theta_hat_SCAD  = x_opt_SCAD(2:end);
        gamma_hat_SCAD  = b(:) .* theta_hat_SCAD(:);
        gamma0_hat_SCAD = theta0_hat_SCAD + sum(b0(:) .* theta_hat_SCAD(:));
        results_local.UNISCAD.lambda = best_lambda_scad;
        results_local.UNISCAD.beta   = [gamma0_hat_SCAD; gamma_hat_SCAD];
        results_local.UNISCAD.a      = a;
    end

    out = results_local;
end