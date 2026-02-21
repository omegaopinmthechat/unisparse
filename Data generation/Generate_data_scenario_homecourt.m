function [X, y, beta0_true, beta_true_final, Sigma] = ...
            Generate_data_scenario_homecourt(n, p)
% Generate_data_scenario_homecourt
% -------------------------------------------------------------
% Implements the Homecourt example in Section 2.3 of the
% uniLasso paper (Chatterjee, Hastie, Tibshirani).
%
% MODEL SUMMARY:
%   • X: AR(1) correlated Gaussian features, rho = 0.8
%   • β (Stage 1): 20% nonzero, nonnegative, moderately large magnitudes
%   • Stage 1:  y' = X * β + σ' z, SNR = 1
%   • Compute univariate OLS slopes β_uni(j)
%   • Stage 2:  y  = X * ( β .* β_uni ) + σ z, SNR = 1
%
% OUTPUTS:
%   X               : n × p feature matrix
%   y               : final Stage-2 response
%   beta0_true      : intercept (0 in this example)
%   beta_true_final : final coefficients β ⊙ β_uni
%   Sigma           : AR(1) covariance matrix
% -------------------------------------------------------------

%% ---------------- AR(1) Covariance for X --------------------
rho = 0.8;
Sigma = rho .^ abs((1:p)' - (1:p));

%% ---------------- Generate X ~ N(0, Σ_AR1) -------------------
Z = randn(n, p);
X = Z * chol(Sigma);

% Standardize to marginal N(0,1) without destroying AR(1) correlation
X = X - mean(X,1);
X = X ./ std(X,0,1);

%% ---------------- Stage-1 TRUE β: 20% sparse, nonnegative -----
s = ceil(0.20 * p);           % number of active coefficients
active_idx = randperm(p, s); % randomly chosen active variables

beta_stage1 = zeros(p,1);

% Use moderately-sized nonnegative signals (better SNR, matches intention)
beta_stage1(active_idx) = 1 + 0.5*rand(s,1);   % values between 1 and 5

beta0_true = 0; % Homecourt uses no intercept anywhere

%% ---------------- Stage 1: y' with SNR = 1 --------------------
signal1 = X * beta_stage1;
sigma1  = sqrt(var(signal1));       % Ensures SNR = 1
y_stage1 = signal1 + sigma1 * randn(n,1);

%% ---------------- Univariate OLS (slope-only) -----------------
beta_uni = zeros(p,1);

for j = 1:p
    bj = regress(y_stage1, [ones(n,1), X(:,j)]);  % [intercept; slope]
    beta_uni(j) = bj(2);                          % slope only
end

%% ---------------- Stage 2: final β_j = β_j * β_uni_j ----------
beta_true_final = beta_stage1 .* beta_uni;

signal2 = X * beta_true_final;
sigma2  = sqrt(var(signal2));       % Again enforce SNR = 1
y = signal2 + sigma2 * randn(n,1);

end
