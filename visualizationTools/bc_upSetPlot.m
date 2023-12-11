function bc_upSetPlot(setNames, Data, barColors)
    % bc_upSetPlot creates an UpSet plot for visualizing set intersections
    % and their sizes.
    %
    % Arguments:
    %   setNames - Cell array of strings representing names of sets
    %   Data - Binary matrix representing set memberships
    %   barColors - Matrix of colors for bars
    
    % Define default colors if not provided
    if nargin < 3 || isempty(barColors)
        barColors = [66, 182, 195] ./ 255; % Default bar color
    end
    lineColor = [61, 58, 61] ./ 255; % Color for lines connecting sets

    % Generate all combinations of sets
    pBool = abs(dec2bin((1:(2^size(Data, 2) - 1))') - 48);

    % Find positions for the UpSet plot
    [pPos, ~] = find(((pBool * (1 - Data')) | ((1 - pBool) * Data')) == 0);
    sPPos = sort(pPos);
    dPPos = find([diff(sPPos); 1]);
    pType = sPPos(dPPos);
    pCount = diff([0; dPPos]);
    [pCount, pInd] = sort(pCount, 'descend');
    pType = pType(pInd);
    
    % Calculate the sum and sort the set counts
    sCount = sum(Data, 1);
    [sCount, sInd] = sort(sCount, 'descend');
    sType = 1:size(Data, 2);
    sType = sType(sInd);

    % Create figure and axes
    fig = figure('Units', 'normalized', 'Position', [.3, .2, .5, .63], 'Color', [1, 1, 1]);
    % Axes for intersection plot
    axI = setupAxes(fig, [.33, .35, .655, .61], 'Intersection Size', []);
    % Axes for set size plot
    axS = setupAxes(fig, [.01, .08, .245, .26], 'Set Size', 1:size(Data, 2));
    % Axes for lines
    axL = setupAxes(fig, [.33, .08, .655, .26], '', []);
    
    % Plot intersection sizes
    plotBarGraph(axI, pCount, barColors);
    % Plot set sizes
    plotBarGraph(axS, sCount, barColors, setNames, sInd);
    % Draw lines connecting sets and intersections
    drawLines(axL, pBool, pType, sType, lineColor);
    
end

function ax = setupAxes(parent, position, label, yticks)
    % Helper function to setup axes properties
    ax = axes('Parent', parent, 'Position', position);
    hold on;
    set(ax, 'LineWidth', 1.2, 'Box', 'off', 'TickDir', 'out', ...
        'FontName', 'Times New Roman', 'FontSize', 12, 'YColor', 'none', 'YTick', yticks);
    ax.XLabel.String = label;
    ax.XLabel.FontSize = 16;
end

function plotBarGraph(ax, data, colors, setNames, indices)
    % Helper function to plot bar graphs
    barHdl = bar(ax, data, 'BarWidth', .6, 'FaceColor', 'flat');
    barHdl.EdgeColor = 'none';
    barHdl.BaseLine.Color = 'none';
    for i = 1:length(data)
        barHdl.CData(i, :) = colors(mod(i, size(colors, 1)) + 1, :);
        if nargin > 3
            % Add set names as text boxes
            text(ax, data(i), i, setNames{indices(i)}, ...
                'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                'FontName', 'Times New Roman', 'FontSize', 12);
        end
    end
end

function drawLines(ax, pBool, pType, sType, color)
    % Helper function to draw lines between bars and sets
    [tX, tY] = meshgrid(1:length(pType), 1:size(pBool, 2));
    for i = 1:length(pType)
        tY = find(pBool(pType(i), :));
        oY = arrayfun(@(y) find(sType == y), tY);
        tX = i .* ones(size(tY));
        plot(ax, tX(:), oY(:), '-o', 'Color', color, ...
            'MarkerEdgeColor', 'none', 'MarkerFaceColor', color, ...
            'MarkerSize', 10, 'LineWidth', 2);
    end
end
