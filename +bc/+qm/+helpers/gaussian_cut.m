%gaussian cut 

function g = JF_gaussian_cut(x, a, x0, sigma, xcut)
     g = a .* exp(-(x - x0) .^ 2 / (2 * sigma .^ 2));
     g(x < xcut) = 0;
%      if diff(g(1:2))<0%wierd shape
%          if length(g(1:find(g==max(g)))) < length(g(end:-1:find(g==max(g))-1))
%             g(1:find(g==max(g))) = g(find(g==max(g))-1+length(g(1:find(g==max(g))))-1:-1:find(g==max(g))-1);
%          else
%             g(find(g==max(g))-length(g(end:-1:find(g==max(g))-1)):find(g==max(g))) = g(find(g==max(g)):-1:find(g==max(g))-1);
%          end
%          
%      end
end