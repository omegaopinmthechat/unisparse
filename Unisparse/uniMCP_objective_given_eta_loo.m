function obj = uniMCP_objective_given_eta_loo(theta, eta_loo, y, lambda, gamma)

if gamma <= 1
    error('MCP concavity parameter gamma must be > 1');
end


theta0 = theta(1);
thetaj = theta(2:end);

thetaj = thetaj(:);
y      = y(:);

resid = y - (theta0 + eta_loo * thetaj);
loss = mean(resid.^2);

abs_thetaj = abs(thetaj);
threshold = gamma * lambda;

mcp_penalty = zeros(size(thetaj));

idx_small = abs_thetaj <= threshold;
mcp_penalty(idx_small) = lambda * abs_thetaj(idx_small) - ...
                          (abs_thetaj(idx_small).^2) / (2 * gamma);


idx_large = abs_thetaj > threshold;
mcp_penalty(idx_large) = 0.5 * gamma * lambda^2;

% Total penalty
pen = sum(mcp_penalty);


obj = loss + pen;

end
