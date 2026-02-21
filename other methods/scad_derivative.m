function w = scad_derivative(t, lambda, a)

w = zeros(size(t));

w(t <= lambda) = lambda;

idx = (t > lambda & t <= a*lambda);
w(idx) = (a*lambda - t(idx)) / (a-1);

w(t > a*lambda) = 0;

end
