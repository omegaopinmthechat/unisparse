function obj = unilasso_objective_given_eta_loo(theta, eta_loo, y, lambda)

theta0 = theta(1);
thetaj = theta(2:end);

thetaj = thetaj(:);
y      = y(:);

resid = y - (theta0 + eta_loo * thetaj);

loss = mean(resid.^2);
pen  = lambda * sum(abs(thetaj));

obj = loss + pen;

end
