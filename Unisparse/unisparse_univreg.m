function [beta0, beta, beta0_loo, beta_loo, eta_loo] = unisparse_univreg(X, y)
% ============================================================
% Step 1 of uniSparse
%
% INPUT:
%   X : n x p design matrix
%   y : n x 1 response
%
% OUTPUT:
%   beta0      : p x 1   univariate intercepts  (β̂_0j)
%   beta       : p x 1   univariate slopes      (β̂_j)
%   beta0_loo  : n x p   LOO intercepts         (β̂_{0j}^{-i})
%   beta_loo   : n x p   LOO slopes             (β̂_j^{-i})
%   eta_loo    : n x p   LOO fitted values      (β̂_{0j}^{-i} + β̂_j^{-i} x_ij)
%
% ============================================================

[n, p] = size(X);

% ---------- Full-sample univariate estimates ----------
xbar = mean(X, 1);        % 1 x p
ybar = mean(y);           % scalar

XC = X - xbar;            % centered X
YC = y - ybar;            % centered y

Sxx = sum(XC.^2, 1);      % 1 x p
Sxy = sum(XC .* YC, 1);   % 1 x p

beta  = Sxy ./ Sxx;                     % slopes  β̂_j
beta0 = ybar - xbar .* beta;           % intercepts β̂_0j

% ---------- Preallocate LOO quantities ----------
beta_loo  = zeros(n, p);
beta0_loo = zeros(n, p);
eta_loo   = zeros(n, p);

% ---------- Efficient LOO using closed-form updates ----------
for i = 1:n
    
    xi = X(i, :);
    yi = y(i);
    
    % Leave-one-out means
    xbar_i = (n*xbar - xi) / (n-1);
    ybar_i = (n*ybar - yi) / (n-1);
    
    % Leave-one-out Sxx and Sxy
    Sxx_i = Sxx - (xi - xbar).^2 * (n/(n-1));
    Sxy_i = Sxy - (xi - xbar) .* (yi - ybar) * (n/(n-1));
    
    % LOO slopes and intercepts
    beta_loo(i,:)  = Sxy_i ./ Sxx_i;
    beta0_loo(i,:) = ybar_i - xbar_i .* beta_loo(i,:);
    
    % LOO fitted values η̂^{-i}_{ij}
    eta_loo(i,:) = beta0_loo(i,:) + beta_loo(i,:) .* xi;
end

end
