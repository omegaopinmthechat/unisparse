function [beta0_hat, beta_hat] = fit_mcp(Xtr, ytr, lambda)

gamma_MCP = 3;   % Zhang default
p = size(Xtr,2);

ybar = mean(ytr);
Xc   = Xtr - mean(Xtr,1);
yc   = ytr - ybar;

beta0 = zeros(p,1);
beta_hat = fminsearch(@(b) mcp_objective(b, Xc, yc, lambda, gamma_MCP), beta0);

beta0_hat = ybar - mean(Xtr,1)*beta_hat;

end
