function [X, y, beta0_true, beta_true] = Generate_data_scenario_counterexample(n, p)
% Generate_data_scenario_counterexample
% ----------------------------------------------------------
% Counterexample data generator as specified:
%
%   n = 100, p = 20 (default)
%   x1 ~ N(0,1)
%   x2 = x1 + N(0,1)
%   remaining p-2 features: i.i.d. N(0,1)
%
%   beta = (1, -0.5, 0, ..., 0)
%   error SD = 0.5
%
% Model:
%   y = beta0_true + X * beta_true + eps
%
% ----------------------------------------------------------

% -------------- safety checks --------------
if p < 2
    error('p must be at least 2 for x1 and x2 construction.');
end

% -------------- True coefficients --------------
beta0_true = 0;                % no intercept mentioned in LaTeX
beta_true  = zeros(p,1);
beta_true(1) = 1;              % beta_1 = 1
beta_true(2) = -0.5;           % beta_2 = -0.5

% -------------- Generate X -------------------
X = zeros(n,p);

% x1 ~ N(0,1)
X(:,1) = randn(n,1);

% x2 = x1 + N(0,1)
X(:,2) = X(:,1) + randn(n,1);

% remaining features: i.i.d. N(0,1)
if p > 2
    X(:,3:p) = randn(n, p-2);
end

% -------------- Generate noise ----------------
sigma_eps = 0.5;
eps = sigma_eps * randn(n,1);

% -------------- Generate response -------------
y = beta0_true + X * beta_true + eps;

end
