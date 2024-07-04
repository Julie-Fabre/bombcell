function upSetPlot(data, labels, figH)
% More info on UpSet plots in the original publication:
% Alexander Lex, Nils Gehlenborg, Hendrik Strobelt, Romain Vuillemot,
% Hanspeter Pfister. UpSet: Visualization of Intersecting Sets
% IEEE Transactions on Visualization and Computer Graphics (InfoVis),
% 20(12): 1983--1992, doi:10.1109/TVCG.2014.2346248, 2014.
%
% this MATLAB code inspired from this FEX code:
% https://uk.mathworks.com/matlabcentral/fileexchange/123695-upset-plot,
% written by Zhaoxu Liu / slandarer

if nargin < 3 || isempty(figH)
    figH = figure('Units', 'normalized', 'Position', [.3, .2, .5, .63], 'Color', [1, 1, 1]);

end
% set colors 
intxColor = [66, 182, 195] ./ 255;
setSizeColor = viz.colors(size(data,2));
lineColor = [61, 58, 61] ./ 255;


% get probabilities & groups 
pBool = abs(dec2bin((1:(2^size(data, 2) - 1))')) - 48;
[pPos, ~] = find(((pBool * (1 - data')) | ((1 - pBool) * data')) == 0);
sPPos = sort(pPos);
dPPos = find([diff(sPPos); 1]);
pType = sPPos(dPPos);
pCount = diff([0; dPPos]);
[pCount, pInd] = sort(pCount, 'descend');
pType = pType(pInd);
sCount = sum(data, 1);
[sCount, sInd] = sort(sCount, 'descend');
sType = 1:size(data, 2);
sType = sType(sInd);

%% create figure and subplots
axI = axes('Parent', figH);
hold on;
set(axI, 'Position', [.33, .35, .655, .61], 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', ...
    'FontName', 'Arial', 'FontSize', 12, 'XTick', [], 'XLim', [0, length(pType) + 1])
axI.YLabel.String = 'Intersection Size';
axI.YLabel.FontSize = 16;
%
axS = axes('Parent', figH);
hold on;
set(axS, 'Position', [.01, .08, .245, .26], 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', ...
    'FontName', 'Arial', 'FontSize', 12, 'YColor', 'none', 'YLim', [.5, size(data, 2) + .5], ...
    'YAxisLocation', 'right', 'XDir', 'reverse', 'YTick', [])
axS.XLabel.String = 'Set Size';
axS.XLabel.FontSize = 16;
%
axL = axes('Parent', figH);
hold on;
set(axL, 'Position', [.33, .08, .655, .26], 'YColor', 'none', 'YLim', [.5, size(data, 2) + .5], 'XColor', 'none', 'XLim', axI.XLim)

%% plot interaction bar plots 
barHdlI = bar(axI, pCount);
barHdlI.EdgeColor = 'none';
if size(intxColor, 1) == 1
    intxColor = [intxColor; intxColor];
end
tx = linspace(0, 1, size(intxColor, 1))';
ty1 = intxColor(:, 1);
ty2 = intxColor(:, 2);
ty3 = intxColor(:, 3);
tX = linspace(0, 1, length(pType))';
intxColor = [interp1(tx, ty1, tX, 'pchip'), interp1(tx, ty2, tX, 'pchip'), interp1(tx, ty3, tX, 'pchip')];
barHdlI.FaceColor = 'flat';
for i = 1:length(pType)
    barHdlI.CData(i, :) = intxColor(i, :);
end
text(axI, 1:length(pType), pCount, string(pCount), 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom', 'FontName', 'Arial', 'FontSize', 12, 'Color', [61, 58, 61]./255)

%% plot set sizes 
barHdlS = barh(axS, sCount, 'BarWidth', .6);
barHdlS.EdgeColor = 'none';
barHdlS.BaseLine.Color = 'none';
for i = 1:size(data, 2)
    annotation('textbox', [(axS.Position(1) + axS.Position(3) + axI.Position(1)) / 2 - .02, ...
        axS.Position(2) + axS.Position(4) ./ size(data, 2) .* (i - .5) - .02, .04, .04], ...
        'String', labels{sInd(i)}, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
        'FitBoxToText', 'on', 'LineStyle', 'none', 'FontName', 'Arial', 'FontSize', 13)
end
if size(setSizeColor, 1) == 1
    setSizeColor = [setSizeColor; setSizeColor];
end
tx = linspace(0, 1, size(setSizeColor, 1))';
ty1 = setSizeColor(:, 1);
ty2 = setSizeColor(:, 2);
ty3 = setSizeColor(:, 3);
tX = linspace(0, 1, size(data, 2))';
setSizeColor = [interp1(tx, ty1, tX, 'pchip'), interp1(tx, ty2, tX, 'pchip'), interp1(tx, ty3, tX, 'pchip')];
barHdlS.FaceColor = 'flat';
sstr{size(data, 2)} = '';
for i = 1:size(data, 2)
    barHdlS.CData(i, :) = setSizeColor(i, :);
    sstr{i} = [num2str(sCount(i)), ' '];
end
text(axS, sCount, 1:size(data, 2), sstr, 'HorizontalAlignment', 'right', ...
    'VerticalAlignment', 'middle', 'FontName', 'Arial', 'FontSize', 12, 'Color', [61, 58, 61]./255)

%% plot interaction details 
patchColor = [248, 246, 249; 255, 254, 255] ./ 255;
for i = 1:size(data, 2)
    fill(axL, axI.XLim([1, 2, 2, 1]), [-.5, -.5, .5, .5]+i, patchColor(mod(i+1, 2)+1, :), 'EdgeColor', 'none')
end
[tX, tY] = meshgrid(1:length(pType), 1:size(data, 2));
plot(axL, tX(:), tY(:), 'o', 'Color', [233, 233, 233]./255, ...
    'MarkerFaceColor', [233, 233, 233]./255, 'MarkerSize', 10);
for i = 1:length(pType)
    tY = find(pBool(pType(i), :));
    oY = zeros(size(tY));
    for j = 1:length(tY)
        oY(j) = find(sType == tY(j));
    end
    tX = i .* ones(size(tY));
    plot(axL, tX(:), oY(:), '-o', 'Color', lineColor(1, :), 'MarkerEdgeColor', 'none', ...
        'MarkerFaceColor', lineColor(1, :), 'MarkerSize', 10, 'LineWidth', 2);
end
end