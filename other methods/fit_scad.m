% What this function does:

% Standardizes the features (so all variables are on the same scale).
% Centers the target variable (so intercept is handled separately).
% Uses Ridge regression for a good starting point (since SCAD is non-convex).
% Minimizes the SCAD objective using fminsearch.
% Rescales coefficients back to original units.
% Computes the intercept correctly.

function [beta0_hat, beta_hat] = fit_scad(Xtr, ytr, lambda)

a = 3.7;                      % Fan & Li default
[n, p] = size(Xtr);

muX = mean(Xtr,1);
sdX = std(Xtr,0,1);
sdX(sdX == 0) = 1;

Xs = (Xtr - muX) ./ sdX;

ybar = mean(ytr);
yc   = ytr - ybar;

beta_init = (Xs' * Xs + lambda * eye(p)) \ (Xs' * yc);

opts = optimset('Display','off', ...
                'MaxIter', 5000, ...
                'MaxFunEvals', 1e5);

beta_hat_scaled = fminsearch( ...
    @(b) scad_objective(b, Xs, yc, lambda, a), ...
    beta_init, ...
    opts);

beta_hat = beta_hat_scaled ./ sdX(:);

beta0_hat = ybar - muX * beta_hat;

end
