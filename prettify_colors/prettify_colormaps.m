function map = prettify_colormaps(N, scheme)
    % A mix of the brewermap and crameri (https://www.nature.com/articles/s41467-020-19160-7, https://www.fabiocrameri.ch/colourmaps/) colormaps
    switch lower(scheme)
        % sequential colorschemes 
        case 'blues'
            colors = [247,251,255; 222,235,247; 198,219,239; 158,202,225; 107,174,214; 66,146,198; 33,113,181; 8,81,156; 8,48,107];
        case 'reds'
            colors = [255,245,240; 254,224,210; 252,187,161; 252,146,114; 251,106,74; 239,59,44; 203,24,29; 165,15,21; 103,0,13];
        case 'greens'
            colors = [247,252,245; 229,245,224; 199,233,192; 161,217,155; 116,196,118; 65,171,93; 35,139,69; 0,109,44; 0,68,27];
        case 'greys'
            colors = [255,255,255; 240,240,240; 217,217,217; 189,189,189; 150,150,150; 115,115,115; 82,82,82; 37,37,37; 0,0,0];
       
        % diverging colorschemes 
        case 'spectral'
            colors = [158,1,66; 213,62,79; 244,109,67; 253,174,97; 254,224,139; 255,255,191; 230,245,152; 171,221,164; 102,194,165; 50,136,189; 94,79,162];
        case 'rdylbu'
            colors = [215,25,28; 253,174,97; 255,255,191; 171,217,233; 44,123,182];
        case 'rdbu'
            colors = [202,0,32; 244,165,130; 247,247,247; 146,197,222; 5,113,176];
        case 'piyg'
            colors = [208,28,139; 241,182,218; 247,247,247; 184,225,134; 77,172,38];
        case 'prgn'
            colors = [123,50,148; 194,165,207; 247,247,247; 166,219,160; 0,136,55];

        % qualitative colorschemes 
        
        % Add additional color schemes here
        % ...
        otherwise
            error('Unsupported color scheme. Please add the scheme to the function.');
    end
    
    % If only one color is required, return the middle one
    if N == 1
        map = colors(ceil(size(colors, 1) / 2), :) / 255;
        return;
    end

    % Interpolate to find N colors
    xp = linspace(1, size(colors, 1), N);
    r = interp1(colors(:,1), xp);
    g = interp1(colors(:,2), xp);
    b = interp1(colors(:,3), xp);
    
    % Combine the channels and normalize to [0, 1]
    map = [r(:), g(:), b(:)] / 255;
end
