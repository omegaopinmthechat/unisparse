function p = mcp_penalty(beta, lambda, gamma)

ab = abs(beta);
p = sum( ...
    (lambda*ab - ab.^2./(2*gamma)) .* (ab <= gamma*lambda) + ...
    (gamma*lambda^2/2) .* (ab > gamma*lambda) );

end
