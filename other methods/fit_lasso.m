function [beta0_hat, beta_hat] = fit_lasso(Xtr, ytr, lambda)

[B, FitInfo] = lasso(Xtr, ytr, 'Lambda', lambda, 'Standardize', false);

beta_hat  = B(:,1);
beta0_hat = FitInfo.Intercept(1);

end
