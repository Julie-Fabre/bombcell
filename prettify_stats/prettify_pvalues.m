function prettify_pvalues(ax, x1, x2, pvals, varargin)
% add_pvalues: Draw lines and add text for multiple p-values on a plot.
%
% Usage:
%   add_pvalues(ax, [x1_1, x1_2, ...], [x2_1, x2_2, ...], [pval1, pval2, ...])
%   add_pvalues(ax, [x1_1, x1_2, ...], [x2_1, x2_2, ...], [pval1, pval2, ...], 'param', value, ...)
%
% Inputs:
%   - ax: Axes handle on which to plot
%   - x1, x2: Vectors of X-coordinates for the start and end of the lines
%   - pvals: Vector of P-values to be displayed
%   - Optional parameters (see below for default values)

% QQ add:
% - handling uposide down/sideways axis
% - options to plot below certain amount, NaNs or not
% - options to just have stars

% (Parse optional parameters)
p = inputParser;
% Visual parameters
addParameter(p, 'TextRotation', 0, @isnumeric);
addParameter(p, 'TextFontSize', 10, @isnumeric);
addParameter(p, 'LineColor', 'k');
addParameter(p, 'LineWidth', 1.5, @isnumeric);
addParameter(p, 'LineMargin', 0.001, @isnumeric);
addParameter(p, 'TickLength', 0.01, @isnumeric);
addParameter(p, 'TextMargin', 0.05, @isnumeric);
% Plot type paramaters 
% addParameter(p, 'AxisForPvalues', 'Y'); % Y axis by default. Change to X
% or Z if necessary %QQ to add 
% P value display parameters 
addParameter(p, 'PlotNonSignif', true); % whether to plot the non-significant values or not
addParameter(p, 'NaNCutoff', 0.05, @isnumeric); % any p values above this 
    % will be plotted as 'n.s.'. Set to 1 or Inf to disable this feature
addParameter(p, 'FullDisplayCutoff', 0.001, @isnumeric); % any p values 
    % below this will be plotted as 'p < thisValue'. Set to 0 or -Inf to disable this feature
addParameter(p, 'OnlyStars', false); % set to true to only plot stars rather
    % than full p values 
addParameter(p, 'StarsLevel_1', 0.050, @isnumeric); % *
addParameter(p, 'StarsLevel_2', 0.010, @isnumeric); % **
addParameter(p, 'StarsLevel_3', 0.001, @isnumeric); % ***

parse(p, varargin{:});
params = p.Results;

% remove any values we don't want to plot 
if params.PlotNonSignif == false
    valuesToPlot = pvals < params.NaNCutoff & ~isnan(pvals);
    pvals =  pvals(valuesToPlot);
    x1 = x1(valuesToPlot);
    x2 = x2(valuesToPlot);
end

% Hold onto the current plot
hold(ax, 'on');

% Calculate a consistent tick length adjusted by the number of p-value lines
baseTickLength = params.TickLength; % Default base tick length
tickLength = baseTickLength * length(pvals) * diff(ax.YLim); % Adjust tick length based on number of p-values

baseTextMargin = params.TextMargin;
textMargin = baseTextMargin * length(pvals) * diff(ax.YLim);

baseLineMargin = params.LineMargin;
LineMargin = baseLineMargin * length(pvals) * diff(ax.YLim);

% Calculate the y limits based on the bars involved in the comparisons
yLimits = arrayfun(@(x) ylim(ax), 1:length(pvals), 'UniformOutput', false);

% Determine the order to plot the p-values to minimize overlaps
[~, sortIdx] = sort(cellfun(@max, yLimits), 'descend');
x1_sorted = x1(sortIdx);
x2_sorted = x2(sortIdx);
pvals_sorted = pvals(sortIdx);

% Initialize the highest level already used for placing a p-value line
highestYLevel = 0;
y_offset = range(ylim) * LineMargin; % 5 % offset

for i = 1:length(pvals_sorted)
    % Find the y-values of bars involved in the comparison (including in-between bars)

    maxYValue = minY_involvedBars([x1_sorted(i), x2_sorted(i)]) + y_offset;

    %    maxYValue = max(involvedBars);

    % Determine the y position for the line
    y_line = maxYValue;

    % Ensure no overlap with previous lines
    if y_line <= highestYLevel
        y_line = highestYLevel + (maxYValue * params.TextMargin);
    end

    % Update the highest level used
    highestYLevel = y_line;

    % y position for the text
    y_text = y_line + textMargin;

    % Draw line for each p-value comparison
    line(ax, [x1(i), x2(i)], [y_line, y_line], 'Color', params.LineColor, 'LineWidth', params.LineWidth);
    line(ax, [x1(i), x1(i)], [y_line , y_line - tickLength], 'Color', params.LineColor, 'LineWidth', params.LineWidth);
    line(ax, [x2(i), x2(i)], [y_line , y_line - tickLength], 'Color', params.LineColor, 'LineWidth', params.LineWidth);

    % Format p-value text
    if pvals_sorted(i) >= params.NaNCutoff || isnan(pvals_sorted(i)) % plot non significant values as n.s.
        pval_text = 'n.s.';
    elseif params.OnlyStars % only display stars
        if pvals_sorted(i) < params.StarsLevel_1
           pval_text = '*';
        elseif pvals_sorted(i) < params.StarsLevel_2
           pval_text = '**';
        elseif pvals_sorted(i) < params.StarsLevel_3
           pval_text = '***';
        end
    else
        if pvals_sorted(i) < params.FullDisplayCutoff
            pval_text = ['p < ', num2str(params.FullDisplayCutoff)];
        else
            pval_text = sprintf('p = %0.3f', pvals_sorted(i));
        end
    end

    % Add text for p-value
    text(mean([x1_sorted(i), x2_sorted(i)]), y_text, pval_text, 'HorizontalAlignment', 'center', 'Rotation', params.TextRotation, 'FontSize', params.TextFontSize, 'Parent', ax);
end

% Adjust the ylim to accommodate the highest p-value line
current_ylim = ylim(ax);
if highestYLevel > current_ylim(2) + y_offset + textMargin
    ylim(ax, [current_ylim(1), highestYLevel + y_offset + textMargin]);
end

% Release the plot hold
hold(ax, 'off');
end

function Y = minY_involvedBars(x)
% hacky way of getting the maximum y value for the bars at the
% locations x (and in between)

% store the original axis limits
original_XLim = get(gca, 'XLim');
original_YLim = get(gca, 'YLim');

% add a little padding to ensure the function behaves nicely
x(1) = x(1) - 0.1;
x(2) = x(2) + 0.1;

% artifically set the xlimits to only include the bars at locations x
% and extract the new xlims from this
axis(gca, 'tight')
set(gca, 'xlim', x)
yLim = get(gca, 'YLim');
Y = max(yLim);

% set back the axis to the original values
axis(gca, 'normal')
set(gca, 'XLim', original_XLim, 'YLim', original_YLim) 

end
