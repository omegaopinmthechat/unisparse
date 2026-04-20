clc; clear;
rng(2026);

if exist('unisparse','file') ~= 2
    error('unisparse not found. Install Unisparse.mltbx first.');
end
addpath('Data generation');

% Problem dimensions
n         = 20;
p         = 30;
n_shifted = 20;
rho       = 0.8;
shift     = 0.5;

% UniSparse settings (keep modest for local runs)
lambda_grid = logspace(-4, 4, 10);
nfolds      = 3;
a_scad      = 3.7;
gamma_mcp   = 3.0;
tol         = 1e-4;

fprintf('\n=== Two-Class Problem: n=%d, p=%d, rho=%.1f, shift=%.1f ===\n', ...
        n, p, rho, shift);

[X, y, true_support] = Generate_data_twoclass(n, p, n_shifted, rho, shift);

% Fit all three methods
fit = unisparse(X, y, lambda_grid, nfolds, 'all', [], [], [], [], a_scad, gamma_mcp);

% Evaluate
metrics = summarize_twoclass(fit, true_support, X, y, tol);
disp(metrics);

function metrics_table = summarize_twoclass(fit, true_support, X, y, tol)
% Classification-aware metrics for the two-class problem.

    method_keys   = {'UNILASSO', 'UNIMCP', 'UNISCAD'};
    method_labels = {'UniLASSO', 'UniMCP', 'UniSCAD'};
    n_methods     = numel(method_keys);

    lambda_vals = nan(n_methods,1);
    tpr_vals    = nan(n_methods,1);
    fpr_vals    = nan(n_methods,1);
    fdr_vals    = nan(n_methods,1);
    nnz_vals    = nan(n_methods,1);
    acc_vals    = nan(n_methods,1);

    for k = 1:n_methods
        key = method_keys{k};
        if ~isfield(fit, key), continue; end

        beta_hat = fit.(key).beta(:);
        b0_hat   = beta_hat(1);
        b_hat    = beta_hat(2:end);

        % Linear score -> sigmoid -> binary prediction
        scores  = b0_hat + X * b_hat;
        prob1   = 1 ./ (1 + exp(-scores));
        y_pred  = double(prob1 >= 0.5);

        est_support = abs(b_hat) > tol;

        tp = sum(est_support &  true_support);
        fp = sum(est_support & ~true_support);
        fn = sum(~est_support & true_support);
        tn = sum(~est_support & ~true_support);

        tpr = tp / max(tp + fn, 1);
        fpr = fp / max(fp + tn, 1);
        fdr = fp / max(tp + fp, 1);
        acc = mean(y_pred == y);

        lambda_vals(k) = fit.(key).lambda;
        tpr_vals(k)    = tpr;
        fpr_vals(k)    = fpr;
        fdr_vals(k)    = fdr;
        nnz_vals(k)    = sum(est_support);
        acc_vals(k)    = acc;
    end

    metrics_table = table(string(method_labels(:)), lambda_vals, ...
                          tpr_vals, fpr_vals, fdr_vals, nnz_vals, acc_vals, ...
                          'VariableNames', {'Method','Lambda','TPR','FPR', ...
                                            'FDR','NNZ','Accuracy'});
end