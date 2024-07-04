function makepretty(figureColor, titleFontSize, labelFontSize, generalFontSize, pointSize, lineThickness, textColor)
% make current figure pretty
% ------
% Inputs
% ------
% - figureColor: string (e.g. 'w', 'k', ..) or RGB value defining the plots
%       background color. 
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
    if nargin < 1 || isempty(figureColor)
        figureColor = 'w';
    end
    if nargin < 2 || isempty(titleFontSize)
        titleFontSize = 18;
    end
    if nargin < 3 || isempty(labelFontSize)
        labelFontSize = 15;
    end
    if nargin < 4 || isempty(generalFontSize)
        generalFontSize = 13;
    end
    if nargin < 5 || isempty(pointSize)
        pointSize = 15;
    end
    if nargin < 6 || isempty(lineThickness)
        lineThickness = 2;
    end
    if nargin < 7 || isempty(textColor)
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
    all_axes = find(arrayfun(@(x) contains(currFig.Children(x).Type, 'axes'), 1:size(currFig.Children,1)));

    for iAx = 1:size(all_axes,2)
        thisAx = all_axes(iAx);
        currAx = currFig.Children(thisAx);
        set(currAx, 'color', figureColor);
        if ~isempty(currAx)
            % Set font properties for the axis
            set(currAx.XLabel, 'FontSize', labelFontSize, 'Color', textColor);
            set(currAx.YLabel, 'FontSize', labelFontSize, 'Color', textColor);
            set(currAx.Title, 'FontSize', titleFontSize, 'Color', textColor);
            set(currAx, 'FontSize', generalFontSize, 'GridColor', textColor, ...
                        'YColor', textColor, 'XColor', textColor, ...
                        'MinorGridColor', textColor);
            
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
        end
    end
end


