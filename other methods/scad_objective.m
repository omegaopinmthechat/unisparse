function obj = scad_objective(beta, X, y, lambda, a)

res  = y - X * beta;
loss = 0.5 * sum(res.^2);    

pen  = scad_penalty(beta, lambda, a);

obj = loss + pen;

end
