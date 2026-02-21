function [X, y, beta0_true, beta_true, Sigma] = Generate_data_scenario_1(n, p, true_p)
% Generate_data_scenario_1
% ----------------------------------------------------------
% Data generator for sparse linear regression where Uni-LASSO
% is expected to outperform standard LASSO due to
% multi-scale heterogeneous signal strengths.
%
% Model:
%   y = beta0_true + X * beta_true + eps
%
% Inputs:
%   n       - sample size
%   p       - ambient dimension
%   true_p  - number of true nonzero coefficients (first true_p active)
%
% Outputs:
%   X           - n x p design matrix (standardized)
%   y           - n x 1 response
%   beta0_true  - scalar true intercept
%   beta_true   - p x 1 true coefficient vector (multi-scale)
%   Sigma       - p x p covariance of X
%
% ----------------------------------------------------------

% ------------------- Safety checks -------------------
if true_p > p
    error('true_p must be <= p');
end
if n <= 0 || p <= 0
    error('n and p must be positive integers');
end

% ------------------- Covariance of X -------------------
rho = 0.1;
Sigma = rho .^ abs((1:p)' - (1:p));

% ------------------- Generate Gaussian design -------------------
Z = randn(n, p);
X = Z * chol(Sigma);

% ------------------- Standardize columns -------------------
X = X - mean(X,1);
X = X ./ std(X,0,1);

% ------------------- TRUE INTERCEPT -------------------
beta0_true = 1.5;   % fixed nonzero intercept (important!)

% ------------------- True sparse beta (MULTI-SCALE) -------------------
beta_true = zeros(p,1);

strong_level = 5;
weak_level   = 0.4;

beta_levels = logspace(log10(strong_level), ...
                        log10(weak_level), true_p);

beta_true(1:true_p) = beta_levels(:);

% Optional random sign
sign_vec = sign(randn(true_p,1));
beta_true(1:true_p) = beta_true(1:true_p) .* sign_vec;

% ------------------- Noise (moderate SNR) -------------------
signal = beta0_true + X * beta_true;

snr_target = 2.5;
sigma_eps  = std(signal) / snr_target;

eps = sigma_eps * randn(n,1);

% ------------------- Response -------------------
y = signal + eps;

end
