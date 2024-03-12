function prettify_plot_old(sameXLimits, sameYLimits, figureColor, legendAsTxt, titleFontSize, labelFontSize, ...
    generalFontSize, pointSize, lineThickness, textColor)
% make current figure pretty
% ------
% Inputs
% ------
% - sameXLimits: string. Either:
%       - 'none': don't change any of the xlimits
%       - 'all': set all xlimits to the same values
%       - 'row': set all xlimits to the same values for each subplot row
%       - 'col': set all xlimits to the same values for each subplot col
% - sameYLimits: string. Either:
%       - 'none': don't change any of the xlimits
%       - 'all': set all xlimits to the same values
%       - 'row': set all xlimits to the same values for each subplot row
%       - 'col': set all xlimits to the same values for each subplot col
% - figureColor: string (e.g. 'w', 'k', ..) or RGB value defining the plots
%       background color.
% - legendAsTxt: 1 if you want the legend box to be replace by text
%       directly plotted on the figure, next to the each subplot's line/point
% - titleFontSize: double
% - labelFontSize: double
% - generalFontSize: double
% - pointSize: double
% - lineThickness: double
% - textColor: double
% ------
% to do:
% - option to adjust vertical and horiz. lines
% - padding
% - fit data to plot (adjust lims)
% - font
% - padding / suptitles
% ------
% Julie M. J. Fabre

% Set default parameter values
if nargin < 1 || isempty(sameXLimits)
    sameXLimits = 'none';
end
if nargin < 2 || isempty(sameYLimits)
    sameYLimits = 'none';
end
if nargin < 3 || isempty(figureColor)
    figureColor = 'w';
end
if nargin < 4 || isempty(legendAsTxt)
    legendAsTxt = 0;
end
if nargin < 5 || isempty(titleFontSize)
    titleFontSize = 15;
end
if nargin < 6 || isempty(labelFontSize)
    labelFontSize = 15;
end
if nargin < 7 || isempty(generalFontSize)
    generalFontSize = 15;
end
if nargin < 8 || isempty(pointSize)
    pointSize = 15;
end
if nargin < 9 || isempty(lineThickness)
    lineThickness = 2;
end
if nargin < 10 || isempty(textColor)
    % Set default font color based on the input color
    switch figureColor
        case 'k'
            textColor = 'w';
        case 'none'
            textColor = [0.7, 0.7, 0.7]; % Gray
        otherwise
            textColor = 'k';
    end
end

% Get handles for current figure and axis
currFig = gcf;


% Set color properties for figure and axis
set(currFig, 'color', figureColor);

% get axes children
all_axes = find(arrayfun(@(x) contains(currFig.Children(x).Type, 'axes'), 1:size(currFig.Children, 1)));

for iAx = 1:size(all_axes, 2)
    thisAx = all_axes(iAx);
    currAx = currFig.Children(thisAx);
    set(currAx, 'color', figureColor);
    if ~isempty(currAx)
        % Set font properties for the axis
        try
            set(currAx.XLabel, 'FontSize', labelFontSize, 'Color', textColor);
            if strcmp(currAx.YAxisLocation, 'left') % if there is both a left and right yaxis, keep the colors
                set(currAx.YLabel, 'FontSize', labelFontSize);
            else
                set(currAx.YLabel, 'FontSize', labelFontSize, 'Color', textColor);
            end
            set(currAx.Title, 'FontSize', titleFontSize, 'Color', textColor);
            set(currAx, 'FontSize', generalFontSize, 'GridColor', textColor, ...
                'YColor', textColor, 'XColor', textColor, ...
                'MinorGridColor', textColor);
            if ~isempty(currAx.Legend)
                set(currAx.Legend, 'Color', figureColor, 'TextColor', textColor)
            end

            % Adjust properties of line children within the plot
            childLines = findall(currAx, 'Type', 'line');
            for thisLine = childLines'
                if strcmp('.', get(thisLine, 'Marker'))
                    set(thisLine, 'MarkerSize', pointSize);
                end
                if strcmp('-', get(thisLine, 'LineStyle'))
                    set(thisLine, 'LineWidth', lineThickness);
                end
            end

            % Adjust properties of errorbars children within the plot
            childErrBars = findall(currAx, 'Type', 'ErrorBar');
            for thisErrBar = childErrBars'
                if strcmp('.', get(thisErrBar, 'Marker'))
                    set(thisErrBar, 'MarkerSize', pointSize);
                end
                if strcmp('-', get(thisErrBar, 'LineStyle'))
                    set(thisErrBar, 'LineWidth', lineThickness);
                end
            end

            % Get x and y limits
            xlims_subplot(iAx, :) = currAx.XLim;
            ylims_subplot(iAx, :) = currAx.YLim;

            % Get plot position
            pos_subplot(iAx, :) = currAx.Position(1:2); % [left bottom width height
            if ~isempty(currAx.Legend)
                if legendAsTxt == 1
                    prettify_legend(currAx)
                else
                    legend('Location', 'best')
                end
            end
        catch
        end
    end
end


% make x and y lims the same
if ismember(sameXLimits, {'all', 'row', 'col'}) || ismember(sameYLimits, {'all', 'row', 'col'})
    % get rows and cols
    col_subplots = unique(pos_subplot(:, 1));
    row_subplots = unique(pos_subplot(:, 2));

    col_xlims = arrayfun(@(x) [min(min(xlims_subplot(pos_subplot(:, 1) == col_subplots(x), :))), ...
        max(max(xlims_subplot(pos_subplot(:, 1) == col_subplots(x), :)))], 1:size(col_subplots, 1), 'UniformOutput', false);
    row_xlims = arrayfun(@(x) [min(min(xlims_subplot(pos_subplot(:, 2) == row_subplots(x), :))), ...
        max(max(xlims_subplot(pos_subplot(:, 2) == row_subplots(x), :)))], 1:size(row_subplots, 1), 'UniformOutput', false);
    col_ylims = arrayfun(@(x) [min(min(ylims_subplot(pos_subplot(:, 1) == col_subplots(x), :))), ...
        max(max(ylims_subplot(pos_subplot(:, 1) == col_subplots(x), :)))], 1:size(col_subplots, 1), 'UniformOutput', false);
    row_ylims = arrayfun(@(x) [min(min(ylims_subplot(pos_subplot(:, 2) == row_subplots(x), :))), ...
        max(max(ylims_subplot(pos_subplot(:, 2) == row_subplots(x), :)))], 1:size(row_subplots, 1), 'UniformOutput', false);

    for iAx = 1:size(all_axes, 2)
        thisAx = all_axes(iAx);
        currAx = currFig.Children(thisAx);
        if ~isempty(currAx)
            %try
                if ismember(sameXLimits, {'all'})
                    set(currAx, 'Xlim', [min(min(xlims_subplot)), max(max(xlims_subplot))]);
                end
                if ismember(sameYLimits, {'all'})
                    set(currAx, 'Ylim', [min(min(ylims_subplot)), max(max(ylims_subplot))]);
                end
                if ismember(sameXLimits, {'row'})
                    set(currAx, 'Xlim', row_xlims{pos_subplot(iAx, 2) == row_subplots});
                end
                if ismember(sameYLimits, {'row'})
                    set(currAx, 'Ylim', row_ylims{pos_subplot(iAx, 2) == row_subplots});
                end
                if ismember(sameXLimits, {'col'})
                    set(currAx, 'Xlim', col_xlims{pos_subplot(iAx, 1) == col_subplots});
                end
                if ismember(sameYLimits, {'col'})
                    set(currAx, 'Ylim', col_ylims{pos_subplot(iAx, 1) == col_subplots});
                end
            %catch
            %end
        end
        % set legend position
        if ~isempty(currAx.Legend)
            if legendAsTxt == 1
                prettify_legend(currAx)
            else
                legend('Location', 'best')
            end
        end
    end
end
end
