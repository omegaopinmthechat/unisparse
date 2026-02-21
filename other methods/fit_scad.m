function [beta0_hat, beta_hat] = fit_scad(Xtr, ytr, lambda)

a = 3.7;                      % Fan & Li default
[n, p] = size(Xtr);

% ---------- STANDARDIZE DESIGN ----------
muX = mean(Xtr,1);
sdX = std(Xtr,0,1);
sdX(sdX == 0) = 1;

Xs = (Xtr - muX) ./ sdX;

ybar = mean(ytr);
yc   = ytr - ybar;

% ---------- GOOD INITIALIZATION (RIDGE) ----------
beta_init = (Xs' * Xs + lambda * eye(p)) \ (Xs' * yc);

% ---------- OPTIMIZATION (ROBUST SETTINGS) ----------
opts = optimset('Display','off', ...
                'MaxIter', 5000, ...
                'MaxFunEvals', 1e5);

beta_hat_scaled = fminsearch( ...
    @(b) scad_objective(b, Xs, yc, lambda, a), ...
    beta_init, ...
    opts);

% ---------- UNSTANDARDIZE ----------
beta_hat = beta_hat_scaled ./ sdX(:);

% ---------- INTERCEPT ----------
beta0_hat = ybar - muX * beta_hat;

end
