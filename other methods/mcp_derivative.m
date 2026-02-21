function w = mcp_derivative(t, lambda, gamma)

w = lambda * max(1 - t ./ (gamma * lambda), 0);

end
