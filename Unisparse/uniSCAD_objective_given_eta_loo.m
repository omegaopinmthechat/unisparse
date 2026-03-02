function obj = uniSCAD_objective_given_eta_loo(theta, eta_loo, y, lambda, a)

if a <= 2
    error('SCAD concavity parameter a must be > 2 (Fan & Li default: 3.7).');
end

theta0 = theta(1);
thetaj = theta(2:end);
thetaj = thetaj(:);
y      = y(:);

resid = y - (theta0 + eta_loo * thetaj);
loss  = mean(resid.^2);

%SCAD penalty on slopes 
abs_tj = abs(thetaj);
pen_vec = zeros(size(thetaj));

%Region 1: |t| <= lambda  (LASSO region)
idx1 = abs_tj <= lambda;
pen_vec(idx1) = lambda .* abs_tj(idx1);

%Region 2: lambda < |t| <= a*lambda  (quadratic smoothing region)
idx2 = (abs_tj > lambda) & (abs_tj <= a * lambda);
pen_vec(idx2) = (2*a*lambda*abs_tj(idx2) - abs_tj(idx2).^2 - lambda^2) ...
                / (2 * (a - 1));

%Region 3: |t| > a*lambda  (constant / unbiased region)
idx3 = abs_tj > a * lambda;
pen_vec(idx3) = (a + 1) * lambda^2 / 2;

pen = sum(pen_vec);

obj = loss + pen;

end
