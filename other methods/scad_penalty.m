function p = scad_penalty(beta, lambda, a)

ab = abs(beta);

term1 = lambda * ab .* (ab <= lambda);

term2 = ((2*a*lambda*ab - ab.^2 - lambda^2) ./ (2*(a-1))) ...
        .* (ab > lambda & ab <= a*lambda);

term3 = ((a+1)*lambda^2 / 2) .* (ab > a*lambda);

p = sum(term1 + term2 + term3);

end
