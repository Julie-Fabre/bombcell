% Get all figure handles
fig_handles = findobj('Type', 'figure');

% Initialize minimum width and height as large numbers
min_width = inf;
min_height = inf;

% Loop through figures to find the minimum width and height of subplots
for fig_idx = 1:length(fig_handles)
    ax_handles = findobj(fig_handles(fig_idx), 'Type', 'axes');
    for ax_idx = 1:length(ax_handles)
        ax_pos = get(ax_handles(ax_idx), 'Position');
        min_width = min(min_width, ax_pos(3));
        min_height = min(min_height, ax_pos(4));
    end
end

% Now resize the subplots and adjust the figure size
for fig_idx = 1:length(fig_handles)
    figure(fig_handles(fig_idx)); % Activate figure

    % Get subplot handles
    ax_handles = findobj(fig_handles(fig_idx), 'Type', 'axes');

    % Find out how many rows and columns of subplots there are
    nRows = max(arrayfun(@(h)round(1 - h.Position(2) - h.Position(4)), ax_handles)); % bottom to top for rows
    nCols = max(arrayfun(@(h)round(h.Position(1)), ax_handles)); % left to right for columns

    % Calculate total spacing required for subplots
    total_vert_spacing = (nRows - 1) * (min_height * 0.1); % 10% of subplot height used for vertical spacing
    total_horz_spacing = (nCols - 1) * (min_width * 0.1); % 10% of subplot width used for horizontal spacing

    % Calculate new figure size
    new_fig_width = nCols * min_width + total_horz_spacing;
    new_fig_height = nRows * min_height + total_vert_spacing;

    % Resize figure
    set(fig_handles(fig_idx), 'Units', 'Pixels');
    fig_pos = get(fig_handles(fig_idx), 'Position');
    set(fig_handles(fig_idx), 'Position', [fig_pos(1), fig_pos(2), new_fig_width, new_fig_height]);

    % Adjust subplots within figure
    for ax_idx = 1:length(ax_handles)
        ax = ax_handles(ax_idx);
        % Calculate new axes position based on its grid position
        row = round(1 - ax.Position(2) - ax.Position(4));  % Bottom to top for rows
        col = round(ax.Position(1));  % Left to right for columns
        new_ax_pos = [(col - 1) * (min_width + min_width * 0.1), ...
                      (nRows - row) * (min_height + min_height * 0.1), ...
                      min_width, min_height];
        % Set new position
        set(ax, 'Position', new_ax_pos);
    end
end
