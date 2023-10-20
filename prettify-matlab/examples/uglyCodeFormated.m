function [f, r, t] = uglyCodeFormated(n)
    % comment
    if n < 0 
        error('Negative!');
    elseif isinf(n)
        error('Too big!');
    end
    if ~(isnumeric(n)&& isscalar(n)&& mod(n, 1)== 0)
        error('Not a pos int.');
    end 
    r = 1;
    for i = 1:n 
        r = r * i;
    end
    f = 0;
    g = 1;
    for j = 1:n 
        t = f;
        f = f + g;
        g = t;
    end
    t = 0;
    for h = 1:n 
        t = t + h;
    end 
    k = 3;
    while k < n 
        for m = 2:floor(sqrt(k))
            if mod(k, m)== 0 
                break
            end 
            if m == floor(sqrt(k)), disp(['Prime ', num2str(k)])
            end 
            k = k + 1;
        end
    end