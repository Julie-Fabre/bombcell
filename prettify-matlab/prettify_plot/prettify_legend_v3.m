function prettify_legend(ax, LegendReplace, LegendLocation, LegendBox)
    
    if ~LegendReplace
        set(ax.Legend, 'Location', LegendLocation)
        set(ax.Legend, 'Box', LegendBox)
        
        handles = findall(ax.Children);
        labels = {};

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
            xdata = h.XData;
            ydata = h.YData;
            labelX = xdata(end); % Adjust as needed, e.g., xdata(1) for left alignment

            % Place the text with a transparent background
            Pos = tscan(ax, wdt, hgt, tol); % Call the modified tscan function
            text(ax, Pos(1), Pos(2), name, 'Color', color, 'FontWeight', 'bold', 'HorizontalAlignment', 'left', 'BackgroundColor', 'none');
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

            % Place the text with a transparent background
            Pos = tscan(ax, wdt, hgt, tol); % Call the modified tscan function
            text(ax, Pos(1), Pos(2) + offset, name, 'Color', color, 'FontWeight', 'bold', 'HorizontalAlignment', 'right', 'BackgroundColor', 'none');
        end

        for h = points'
            % Extract color and display name
            color = h.CData;
            name = h.DisplayName;

            % Calculate label position based on average y-value
            ydata = h.YData;
            yavg = mean(ydata);

            % Place the text with a transparent background
            Pos = tscan(ax, wdt, hgt, tol); % Call the modified tscan function
            text(ax, Pos(1), Pos(2) + offset, name, 'Color', color, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'BackgroundColor', 'none');
        end

        % Remove legend
        ax.Legend.Visible = 'off';
    end
end
