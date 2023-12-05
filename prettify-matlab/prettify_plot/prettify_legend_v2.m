function prettify_legend(ax, LegendReplace, LegendLocation, LegendBox)
    
    if ~LegendReplace
        set(ax.Legend, 'Location', LegendLocation)
        set(ax.Legend, 'Box', LegendBox)
        
        % Extract handles and labels
        handles = findall(ax.Children);
        labels = {};
        
        % Create a structure to store label positions and colors
        labelInfo = struct('x', [], 'y', [], 'color', []);
        
        for iHandle = 1:length(handles)
            h = handles(iHandle);
            % Extract color and display name
            if strcmp(get(h, 'Type'), 'line')
                color = h.Color;
            elseif strcmp(get(h, 'Type'), 'scatter')
                color = h.CData;
            end
            name = h.DisplayName;
            labels{end + 1} = name;
            
            % Calculate label positions
            ydata = h.YData;
            xdata = h.XData;
            labelX = xdata(end); % Adjust as needed, e.g., xdata(1) for left alignment
            
            % Use binary search to find the optimal labelY position
            labelY = findOptimalLabelY(ax, xdata, ydata, labelX);
            
            % Store label information
            labelInfo(end + 1).x = labelX;
            labelInfo(end).y = labelY;
            labelInfo(end).color = color;
            
            % Place the text with a transparent background
            text(labelX, labelY, name, 'Color', color, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'BackgroundColor', 'none');
        end
        
        % Sort the labels based on their x-positions
        [~, order] = sort(cellfun(@(x) find(strcmp(labels, x)), labels));
        legend(ax, handles(order));
        
    else
        yRange = ax.YLim(2) - ax.YLim(1);
        offset = yRange * 0.05;  % Adjust offset as needed for better spacing
        
        handles = findall(ax.Children);
        
        lines = handles(ishandle(handles) & strcmp(get(handles, 'Type'), 'line'));
        points = handles(ishandle(handles) & strcmp(get(handles, 'Type'), 'scatter'));
        
        % Remove any single points from legend
        lines = lines(arrayfun(@(x) numel(x.XData) > 1, lines));
        
        for h = lines'
            % Extract color and display name
            color = h.Color;
            name = h.DisplayName;
            
            % Calculate label position based on average y-value
            ydata = h.YData;
            yavg = mean(ydata);
            
            % Use binary search to find the optimal labelX position
            labelX = findOptimalLabelX(ax, h.XData, ydata, yavg);
            
            % Place the text with a transparent background
            text(labelX, yavg + offset, name, 'Color', color, 'FontWeight', 'bold', 'HorizontalAlignment', 'right', 'BackgroundColor', 'none');
        end
        
        for h = points'
            % Extract color and display name
            color = h.CData;
            name = h.DisplayName;
            
            % Calculate label position based on average y-value
            ydata = h.YData;
            yavg = mean(ydata);
            
            % Use binary search to find the optimal labelX position
            labelX = findOptimalLabelX(ax, h.XData, ydata, yavg);
            
            % Place the text with a transparent background
            text(labelX, yavg + offset, name, 'Color', color, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'BackgroundColor', 'none');
        end
        
        % Remove legend
        ax.Legend.Visible = 'off';
    end
end

function labelY = findOptimalLabelY(ax, xdata, ydata, labelX)
    % Binary search to find the optimal labelY position
    lowerY = min(ydata);
    upperY = max(ydata);
    epsilon = 0.001; % Small tolerance
    
    while upperY - lowerY > epsilon
        midY = (lowerY + upperY) / 2;
        if ~checkOverlap(ax, xdata, ydata, labelX, midY)
            lowerY = midY;
        else
            upperY = midY;
        end
    end
    
    labelY = lowerY;
end

function labelX = findOptimalLabelX(ax, xdata, ydata, labelY)
    % Binary search to find the optimal labelX position
    lowerX = min(xdata);
    upperX = max(xdata);
    epsilon = 0.001; % Small tolerance
    
    while upperX - lowerX > epsilon
        midX = (lowerX + upperX) / 2;
        if ~checkOverlap(ax, xdata, ydata, midX, labelY)
            lowerX = midX;
        else
            upperX = midX;
        end
    end
    
    labelX = lowerX;
end

function overlap = checkOverlap(ax, xdata, ydata, labelX, labelY)
    % Check if the label overlaps with any lines or points
    tolerance = 0.001; % Adjust as needed
    
    for i = 1:length(xdata)
        if abs(xdata(i) - labelX) < tolerance && abs(ydata(i) - labelY) < tolerance
            overlap = true;
            return;
        end
    end
    
    overlap = false;
end
