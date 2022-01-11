%gaussian cut 

function g = JF_gaussian_cut(x, a, x0, sigma, xcut)
     g = a .* exp(-(x - x0) .^ 2 / (2 * sigma .^ 2));
     g(x < xcut) = 0;
end