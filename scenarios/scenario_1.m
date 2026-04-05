% Scenario 1: low, medium, and high SNR (n=300, p=1000)
% We use Gaussian features with pairwise correlation rho = 0.5.
% The nonzero coefficients are sampled from N(0,1).
% Noise is Gaussian and sigma is chosen for low (<1), medium (~1), and high (>2) SNR.
% The script calls unisparse with method='all' to fit UniLASSO, UniMCP, and UniSCAD.

% Quick summary:
% 300 samples (n = 300)
% 1000 features (p = 1000) → very high-dimensional
% Only 20 features actually matter (sparse signal)
% Features are correlated (0.5)


clc;
clear;

% Reproducibility for data generation and CV splits.
rng(2026);

% Path setup
% Add project folders relative to this script so it can be run from anywhere.
this_file = mfilename('fullpath');
repo_root = fileparts(fileparts(this_file));
addpath(fullfile(repo_root, 'Unisparse'));
addpath(fullfile(repo_root, 'supp funs'));
addpath(fullfile(repo_root, 'RMPSH'));
addpath(fullfile(repo_root, 'other methods'));

% Scenario settings
n      = 300;   
p      = 1000; % My laptop testing is done on 200
true_p = 20;   % Number of true nonzero coefficients (sparsity level)
rho    = 0.5;  % Pairwise feature correlation

% Targets chosen to match: low < 1, medium ~ 1, high > 2.
snr_labels  = {'low', 'medium', 'high'};
snr_targets = [0.5, 1.0, 2.5];

% UniSparse settings. Keep a modest grid here to make runs practical.
% lambda_grid = logspace(-4, 1, 20); 
lambda_grid = logspace(-4, 4, 10); % Using this as my PC can not handle the last lesser coarse lambda_grid
nfolds      = 3; % use 5 (again my pc :( )
a_scad      = 3.7;
gamma_mcp   = 3.0;
tol         = 1e-4;

% Store all outputs in a single struct for later analysis/saving.
scenario_results = struct();

fprintf('\n===============================================================\n');
fprintf('Scenario 1: n=%d, p=%d, true_p=%d, rho=%.2f\n', n, p, true_p, rho);
fprintf('Running UniLASSO, UniMCP, UniSCAD via unisparse(method=''all'')\n');
fprintf('===============================================================\n');

for i = 1:numel(snr_targets)
	target_snr = snr_targets(i);
	label      = snr_labels{i};

	% Generate one dataset at this SNR level.
	[X, y, beta0_true, beta_true, sigma_eps, snr_empirical] = ...
		generate_scenario1_data(n, p, true_p, rho, target_snr);

	fprintf('\n--- SNR level: %s (target=%.2f, empirical=%.3f, sigma=%.4f) ---\n', ...
		upper(label), target_snr, snr_empirical, sigma_eps);

	% One call fits all three methods: UNILASSO, UNIMCP, UNISCAD.
	fit = unisparse(X, y, lambda_grid, nfolds, 'all', [], [], [], [], a_scad, gamma_mcp);

	% Build a clear metrics table from the fitted coefficients.
	metrics_table = summarize_unisparse_methods(fit, beta0_true, beta_true, X, y, tol);
	disp(metrics_table);

	% Save everything useful for this SNR condition.
	scenario_results(i).snr_label      = label;
	scenario_results(i).snr_target     = target_snr;
	scenario_results(i).snr_empirical  = snr_empirical;
	scenario_results(i).sigma_eps      = sigma_eps;
	scenario_results(i).beta0_true     = beta0_true;
	scenario_results(i).beta_true      = beta_true;
	scenario_results(i).fit            = fit;
	scenario_results(i).metrics_table  = metrics_table;
end

fprintf('\nDone. Results are available in variable: scenario_results\n');


function [X, y, beta0_true, beta_true, sigma_eps, snr_empirical] = ...
	generate_scenario1_data(n, p, true_p, rho, target_snr)
% Generate one Scenario 1 dataset with controlled SNR.

	if true_p > p
		error('true_p must be <= p.');
	end
	if target_snr <= 0
		error('target_snr must be positive.');
	end

	% Compound-symmetry covariance gives pairwise corr(X_j, X_k) = rho.
	Sigma = (1 - rho) * eye(p) + rho * ones(p);

	% Small ridge term improves numerical stability in Cholesky.
	R = chol(Sigma + 1e-10 * eye(p), 'lower');
	X = randn(n, p) * R';

	% Standardize features (column-wise).
	X = X - mean(X, 1);
	s = std(X, 0, 1);
	s(s == 0) = 1;
	X = X ./ s;

	% Sparse true coefficients: first true_p are active and sampled from N(0,1).
	beta0_true = 0;
	beta_true = zeros(p, 1);
    % making the non-zero more random for realistic scenarios
	idx = randperm(p, true_p);
    beta_true(idx) = randn(true_p, 1);

	signal = beta0_true + X * beta_true;
	var_signal = var(signal, 1);

	% SNR = Var(signal) / Var(noise). Choose sigma to hit target_snr.
	sigma_eps = sqrt(var_signal / target_snr);
	eps = sigma_eps * randn(n, 1);

	y = signal + eps;
	snr_empirical = var(signal, 1) / var(eps, 1);
end


function metrics_table = summarize_unisparse_methods(fit, beta0_true, beta_true, X, y, tol)
% Compute and format metrics for UNILASSO, UNIMCP, and UNISCAD fits.

	method_keys   = {'UNILASSO', 'UNIMCP', 'UNISCAD'};
	method_labels = {'UniLASSO', 'UniMCP', 'UniSCAD'};
	beta_true_whole = [beta0_true; beta_true];

	n_methods = numel(method_keys);
	lambda_vals = nan(n_methods, 1);
	tpr_vals    = nan(n_methods, 1);
	fpr_vals    = nan(n_methods, 1);
	fdr_vals    = nan(n_methods, 1);
	mcc_vals    = nan(n_methods, 1);
	mse_vals    = nan(n_methods, 1);
	nnz_vals    = nan(n_methods, 1);

	for k = 1:n_methods
		key = method_keys{k};

		if ~isfield(fit, key)
			continue;
		end

		beta_hat = fit.(key).beta(:);
		yhat = beta_hat(1) + X * beta_hat(2:end);
		m = compute_sparse_metrics(beta_hat, beta_true_whole, yhat, y, tol);

		lambda_vals(k) = fit.(key).lambda;
		tpr_vals(k)    = m(1);
		fpr_vals(k)    = m(2);
		mcc_vals(k)    = m(3);
		mse_vals(k)    = m(6);
		fdr_vals(k)    = m(7);
		nnz_vals(k)    = sum(abs(beta_hat(2:end)) > tol);
	end

	metrics_table = table(string(method_labels(:)), lambda_vals, tpr_vals, fpr_vals, ...
						  fdr_vals, mcc_vals, mse_vals, nnz_vals, ...
						  'VariableNames', {'Method', 'Lambda', 'TPR', 'FPR', ...
											'FDR', 'MCC', 'MSE', 'NNZ'});
end

