function obj = unibridge_objective_given_eta_loo(theta, eta_loo, y, lambda, q)

if q <= 0 || q > 1
    error('q must satisfy 0 < q <= 1');
end

theta0 = theta(1);
thetaj = theta(2:end);

thetaj = thetaj(:);
y      = y(:);

resid = y - (theta0 + eta_loo * thetaj);

loss = mean(resid.^2);
pen  = lambda * sum(abs(thetaj).^q);

obj = loss + pen;

end
