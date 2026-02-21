function [beta0_hat, beta_hat] = fit_adap_lasso(Xtr, ytr, lambda)

% Initial OLS
beta_ols = Xtr \ ytr;

% Weights
gamma = 1;
w = 1 ./ max(abs(beta_ols), 1e-6).^gamma;

% Weighted LASSO transform
Xw = Xtr ./ w';

[B, FitInfo] = lasso(Xw, ytr, 'Lambda', lambda, 'Standardize', false);

beta_hat  = B(:,1) ./ w;
beta0_hat = FitInfo.Intercept(1);

end
