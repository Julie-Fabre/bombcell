function prettify_legend(ax)
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

        % Place the text
        text(ax, max(h.XData), yavg + offset, name, 'Color', color, 'FontWeight', 'bold', 'HorizontalAlignment', 'right');
    end

    for h = points'
        % Extract color and display name
        color = h.CData;
        name = h.DisplayName;

        % Calculate label position based on average y-value
        ydata = h.YData;
        yavg = mean(ydata);

        % Place the text
        text(h.XData, yavg + offset, name, 'Color', color, 'FontWeight', 'bold', 'HorizontalAlignment', 'center');
    end

    % Remove legend (if needed)
    ax.Legend.Visible = 'off';
end
