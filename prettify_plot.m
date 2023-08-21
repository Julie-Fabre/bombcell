function prettify_plot(inputColor, titleFontSize, labelFontSize, generalFontSize, pointSize, lineThickness, fontColor)
    
    % Set default parameter values
    if nargin < 1 || isempty(inputColor)
        inputColor = 'w';
    end
    if nargin < 2 || isempty(titleFontSize)
        titleFontSize = 20;
    end
    if nargin < 3 || isempty(labelFontSize)
        labelFontSize = 17;
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
    if nargin < 7 || isempty(fontColor)
        % Set default font color based on the input color
        switch inputColor
            case 'k'
                fontColor = 'w';
            case 'none'
                fontColor = [0.7, 0.7, 0.7]; % Gray
            otherwise
                fontColor = 'k';
        end
    end
    
    % Get handles for current figure and axis
    currFig = gcf;
    currAx = gca;
    
    % Set color properties for figure and axis
    set(currFig, 'color', inputColor);
    set(currAx, 'color', inputColor);
    
    % Set font properties for the axis
    set(currAx.XLabel, 'FontSize', labelFontSize, 'Color', fontColor);
    set(currAx.YLabel, 'FontSize', labelFontSize, 'Color', fontColor);
    set(currAx.Title, 'FontSize', titleFontSize, 'Color', fontColor);
    set(currAx, 'FontSize', generalFontSize, 'GridColor', fontColor, ...
                'YColor', fontColor, 'XColor', fontColor, ...
                'MinorGridColor', fontColor);
    
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
