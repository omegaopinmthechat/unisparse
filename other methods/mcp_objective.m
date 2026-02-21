function obj = mcp_objective(beta, X, y, lambda, gamma_MCP)

res  = y - X * beta;
loss = mean(res.^2);
pen  = mcp_penalty(beta, lambda, gamma_MCP);

obj = loss + pen;

end
