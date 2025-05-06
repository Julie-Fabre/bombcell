function colorMtx = colors(nColors, backgroundCol, type)
% JF, get distinguishable color-blind friendly colors
% ------
% Inputs
% ------
% nColors: number of colors to return (default: 15)
% backgroundCol: background color ('w' for white, 'b' for black, default: 'w')
% type: colormap type ('standard', 'wang', 'ibm', 'roma', default: 'standard')
% ------
% Outputs
% ------
% colorMtx: matrix of RGB colors

% Set default values
if nargin < 1 || isempty(nColors)
    nColors = 15;
end

if nargin < 2 || isempty(backgroundCol)
    backgroundCol = 'w';
end

if nargin < 3 || isempty(type)
    type = 'standard';
end

% Define colormaps
colormaps = struct();

% Standard colors
colormaps.standard = [
    255, 255, 255; ... %1 white
    0, 0, 0; ... %2 black
    37, 37, 37; ... %3 dark gray
    103, 103, 103; ... %4 light gray
    23, 23, 35; ... %5 aubergine
    0, 73, 73; ... %6 dark teal
    0, 153, 153; ... %7 dark turquoise
    34, 207, 34; ... %8 lime green
    73, 0, 146; ... %9 indigo
    0, 109, 219; ... %10 light royal blue
    182, 109, 255; ... %11 amethyst
    255, 109, 182; ... %12 Baker-Miller pink
    146, 0, 0; ... %13 ruby
    143, 78, 0; ... %14 saddle brown
    219, 109, 0; ... %15 butternut orange
    255, 223, 77 ... %16 daffodil yellow
] / 255;

% Wang colors
colormaps.wang = [
    0, 0, 0; ... % black
    230, 159, 0; ... % orange
    86, 180, 233; ... % sky blue
    0, 158, 115; ... % bluish green
    240, 228, 66; ... % yellow
    0, 114, 178; ... % blue
    213, 94, 0; ... % vermillion
    204, 121, 167 ... % reddish purple
] / 255;

% IBM colors
colormaps.ibm = [
    100, 143, 255;
    120, 94, 240;
    220, 38, 127;
    254, 97, 0;
    255, 176, 0
] / 255;

% ROMA colors (hardcoded)
colormaps.roma = [
    0.451373874666344, 0.223458709918417, 0.341870799965347;
    0.462508758064081, 0.220345584246605, 0.319346410115490;
    0.473519265726396, 0.219623882296950, 0.298217773730138;
    0.484529958777781, 0.221302490266283, 0.278502330945246;
    0.495670101286812, 0.225363081959894, 0.260159132801034;
    0.507065485813229, 0.231880162811605, 0.243219851754147;
    0.518841911844592, 0.240822744063106, 0.227787051712419;
    0.531082226579023, 0.252306972415458, 0.213873324035087;
    0.543885785011561, 0.266280741940019, 0.201584797958454;
    0.557305138435603, 0.282734966244876, 0.191089012591656;
    0.571339570837261, 0.301567625153758, 0.182482555580965;
    0.586016684749346, 0.322749176207851, 0.175966004432052;
    0.601285208641269, 0.346064887320984, 0.171791071025377;
    0.617133052637388, 0.371431700016994, 0.170196472439744;
    0.633515835346981, 0.398691613665266, 0.171480298326735;
    0.650407311180225, 0.427708212561651, 0.175999703086171;
    0.667797548648492, 0.458344002465640, 0.184158836750910;
    0.685655046143849, 0.490513841767484, 0.196242123648254;
    0.703943799444585, 0.524059133087570, 0.212508158076656;
    0.722570748796306, 0.558811919993631, 0.233180387444497;
    0.741359662805532, 0.594511436020260, 0.258373361840245;
    0.760006263757777, 0.630745190728584, 0.287961592854003;
    0.778063513547249, 0.666936481733587, 0.321653420457785;
    0.794941521481620, 0.702400073437457, 0.358880664372762;
    0.809388073505355, 0.880358839111006, 0.679643693683488;
    0.820303020497873, 0.875308817916059, 0.656170514312734;
    0.828642731538215, 0.867996924203077, 0.630853456571219;
    0.834353490535377, 0.858352935600571, 0.603824386219959;
    0.837435056914653, 0.846348410663529, 0.575250092483893;
    0.837882936293263, 0.841823729780831, 0.565416871642003
];

% Select the appropriate colormap
if ~isfield(colormaps, type)
    error('Invalid colormap type. Choose from: standard, wang, ibm, or roma.');
end

selectedColormap = colormaps.(type);

% Remove background color from the colormap
if strcmpi(backgroundCol, 'w')
    % If background is white, replace white (1,1,1) with black (0,0,0)
    selectedColormap(all(selectedColormap == 1, 2), :) = 0;
elseif strcmpi(backgroundCol, 'k')
    % If background is black, replace black (0,0,0) with white (1,1,1)
    selectedColormap(all(selectedColormap == 0, 2), :) = 1;
end
% Select colors based on the colormap type and nColors
switch type
    case 'standard'
        if nColors <= 2
            colorMtx = selectedColormap([9, 12], :);
        elseif nColors <= 4
            colorMtx = selectedColormap([9, 12, 14, 6], :);
        else
            colorIndices = size(selectedColormap, 1) - min(nColors, size(selectedColormap, 1) - 1) + 1 : size(selectedColormap, 1);
            colorMtx = selectedColormap(colorIndices, :);
        end
    case 'roma'
        % Interpolate ROMA colormap
        originalLength = size(selectedColormap, 1);
        x = linspace(1, originalLength, originalLength);
        xq = linspace(1, originalLength, nColors);
        colorMtx = interp1(x, selectedColormap, xq);
    otherwise
        % For other colormaps, cycle through colors if more are requested than available
        maxColors = size(selectedColormap, 1);
        colorIndices = mod(0:nColors-1, maxColors) + 1;
        colorMtx = selectedColormap(colorIndices, :);
end

% Ensure we have the correct number of colors
colorMtx = colorMtx(1:min(nColors, size(colorMtx, 1)), :);

end