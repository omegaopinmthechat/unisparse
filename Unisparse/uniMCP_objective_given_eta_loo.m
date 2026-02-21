function obj = uniMCP_objective_given_eta_loo(theta, eta_loo, y, lambda, gamma)
% ============================================================
% Objective function for UniMCP with MCP penalty
%
% This function computes the UniMCP objective:
%   obj = MSE loss + MCP penalty on theta
%
% INPUT:
%   theta    : (p+1) x 1  parameter vector [theta0; theta_1; ...; theta_p]
%   eta_loo  : n x p      LOO fitted values from univariate regressions
%   y        : n x 1      response vector
%   lambda   : scalar     regularization parameter (λ)
%   gamma    : scalar     MCP concavity parameter (γ > 1)
%
% OUTPUT:
%   obj      : scalar     objective function value
%
% MCP PENALTY DEFINITION:
%   For each θ_j:
%     MCP(θ_j; λ, γ) = λ|θ_j| - θ_j²/(2γ)     if |θ_j| ≤ γλ
%                    = (1/2)γλ²                if |θ_j| > γλ
%
% ============================================================

% Validate gamma
if gamma <= 1
    error('MCP concavity parameter gamma must be > 1');
end

% Extract intercept and slope parameters
theta0 = theta(1);
thetaj = theta(2:end);

% Ensure column vectors
thetaj = thetaj(:);
y      = y(:);

% -------- 1. Compute residuals and MSE loss --------
resid = y - (theta0 + eta_loo * thetaj);
loss = mean(resid.^2);

% -------- 2. Compute MCP penalty --------
% MCP(θ_j; λ, γ) = λ|θ_j| - θ_j²/(2γ)     if |θ_j| ≤ γλ
%                = (1/2)γλ²                if |θ_j| > γλ

abs_thetaj = abs(thetaj);
threshold = gamma * lambda;

% Preallocate penalty vector
mcp_penalty = zeros(size(thetaj));

% Case 1: |θ_j| <= γλ  (concave region)
idx_small = abs_thetaj <= threshold;
mcp_penalty(idx_small) = lambda * abs_thetaj(idx_small) - ...
                          (abs_thetaj(idx_small).^2) / (2 * gamma);

% Case 2: |θ_j| > γλ   (constant region - no further penalization)
idx_large = abs_thetaj > threshold;
mcp_penalty(idx_large) = 0.5 * gamma * lambda^2;

% Total penalty
pen = sum(mcp_penalty);

% -------- 3. Combine loss and penalty --------
obj = loss + pen;

end
