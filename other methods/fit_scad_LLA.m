function [beta0_hat, beta_hat] = fit_scad_LLA(X, y, lambda)
% fit_scad_LLA
% ---------------------------------------------------------
% SCAD-penalized least squares via Local Linear Approximation (LLA)
% using weighted LASSO implemented via column scaling + MATLAB lasso.
%
% Model: y = beta0 + X * beta + eps   (WITH INTERCEPT)
%
% INPUTS:
%   X      : n x p design
%   y      : n x 1 response
%   lambda : SCAD tuning parameter (scalar)
%
% OUTPUTS:
%   beta0_hat : scalar intercept
%   beta_hat  : p x 1 SCAD estimate (original scale)
% ---------------------------------------------------------

    [n,p] = size(X);

    % ---------- STANDARDIZE DESIGN ----------
    muX = mean(X,1);
    sdX = std(X,0,1);
    sdX(sdX == 0) = 1;   % avoid zero-variance columns

    Xs = (X - muX) ./ sdX;

    % ---------- CENTER RESPONSE (INTERCEPT REMOVED INTERNALLY) ----------
    ybar = mean(y);
    yc   = y - ybar;

    % ---------- INITIAL LASSO (NO INTERCEPT) ----------
    beta = lasso(Xs, yc, ...
                  'Lambda', lambda, ...
                  'Intercept', false, ...
                  'Standardize', false);
    beta = beta(:);

    % ---------- SCAD PARAMETERS ----------
    a        = 3.7;     % Fan & Li default
    maxIter  = 25;
    tol      = 1e-6;

    for iter = 1:maxIter

        ab = abs(beta);

        % ----- SCAD derivative p'(|beta|) -----
        lam_j = zeros(p,1);

        idx1 = (ab <= lambda);
        lam_j(idx1) = lambda;

        idx2 = (ab > lambda & ab <= a*lambda);
        lam_j(idx2) = (a*lambda - ab(idx2)) / (a - 1);

        % Numerical stability
        lam_floor = lambda * 1e-3;
        lam_j = max(lam_j, lam_floor);

        % ----- Weighted LASSO via column scaling -----
        Z = Xs .* (1 ./ lam_j(:))';

        theta = lasso(Z, yc, ...
                       'Lambda', 1, ...
                       'Intercept', false, ...
                       'Standardize', false);
        theta = theta(:);

        beta_new = theta ./ lam_j;

        % ----- Convergence check -----
        if norm(beta_new - beta, Inf) < tol
            beta = beta_new;
            break;
        end

        beta = beta_new;
    end

    % ---------- UNSTANDARDIZE COEFFICIENTS ----------
    beta_hat = beta ./ sdX(:);

    % ---------- RECOVER INTERCEPT ----------
    beta0_hat = ybar - muX * beta_hat;

end
