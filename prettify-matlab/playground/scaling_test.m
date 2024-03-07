% Gather all subplot dimensions and find the minimum width and height
subplot_dims = [];  % Store [width, height] of each subplot
fig_handles = findobj('Type', 'figure');  % Get all figure handles

for fig_idx = 1:length(fig_handles)
    ax_handles = findobj(fig_handles(fig_idx), 'Type', 'axes');
    for ax_idx = 1:length(ax_handles)
        ax_pos = get(ax_handles(ax_idx), 'Position');
        subplot_dims = [subplot_dims; ax_pos(3:4)];  % Only width and height
    end
end

% Determine the minimum width and height to be used for all subplots
min_width = min(subplot_dims(:,1));
min_height = min(subplot_dims(:,2));



% Call the function to resize subplots in all figures
resize_subplots(fig_handles, min_width, min_height);
% Function to resize subplots and optionally adjust figure size
function resize_subplots(fig_handles, min_width, min_height)
    for fig_idx = 1:length(fig_handles)
        fig = fig_handles(fig_idx);
        ax_handles = findobj(fig, 'Type', 'axes');

        % Resize subplots
        for ax_idx = 1:length(ax_handles)
            ax = ax_handles(ax_idx);
            ax_pos = get(ax, 'Position');
            % Scale subplot width and height independently to maintain aspect ratio
            scale_w = min_width / ax_pos(3);
            scale_h = min_height / ax_pos(4);
            new_pos = [ax_pos(1), ax_pos(2) + (ax_pos(4) - min_height * scale_h),...
                       ax_pos(3) * scale_w, ax_pos(4) * scale_h];
            set(ax, 'Position', new_pos);
        end

        % Optionally, adjust figure size here if needed
        % For example, to make room for suptitle or other figure-level annotations.
    end
end