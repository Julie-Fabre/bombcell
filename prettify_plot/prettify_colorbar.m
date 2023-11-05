QQ nee to find axes / colorbar pairs, and adjust colormap there. 

colorbarProperties = struct;
for iColorBar = 1:length(colorbars)
    currColorbar = colorbars(iColorBar);
    %colorbarProperties = table;

    % set limits in smart way + choose colormap.
    if ismember(options.CLimits, {'all', 'row', 'col'})
        % get rows and cols

        if ismember(options.CLimits, {'all'})
            currColorbar.Limits = [min(min(clims)), max(max(clims))];
        end
        if ismember(options.CLimits, {'col'})
            currColorbar.Limits = col_clims{col_pos(iColorBar, 1) == col_cols};
        end
        if ismember(options.CLimits, {'row'})
            currColorbar.Limits = row_clims{col_pos(iColorBar, 2) == row_cols};
        end

    else
        currColorbar.Limits = clims(iColorBar, :);
    end
    colorbarProperties(iColorBar).Limits = currColorbar.Limits;
    colorbarProperties(iColorBar).Parent = currColorbar.Parent;
    % get label
    if ~isempty(currColorbar.Label.String) % label string
        label = currColorbar.Label.String;
    elseif ~isempty(currColorbar.Title.String) % title
        label = currColorbar.Title.String;
    elseif ~isempty(currColorbar.XLabel.String) %x label string
        label = currColorbar.Xlabel.String;
    else
        label = '';
    end

    colorbarProperties(iColorBar).Label = label;
    currColorbar.Units = 'Points'; %get old, add padding.
    colorbarProperties(iColorBar).Position_ori = currColorbar.Position;

    % remove colorbar
    % get parent axis size
    parentPosition_ori = colorbarProperties(iColorBar).Parent.Position; % [left bottom width height]


    % get parent axis size
    parentPosition = colorbarProperties(iColorBar).Parent.Position; % [left bottom width height]
    padding = 5;
    colorbarProperties(iColorBar).Position = [parentPosition(3) + padding, ...
        padding, colorbarProperties(iColorBar).Position_ori(3), parentPosition(4) - 2 * padding];

    % add colorbar back

    currColorbar.Parent = colorbarProperties(iColorBar).Parent;
    currColorbar.Units = 'Points';
    currColorbar.Position = [400, colorbarProperties(iColorBar).Position_ori(2:4)]; % QQ hardcoded
    currColorbar.Label.String = colorbarProperties(iColorBar).Label;%QQ error
    %findall(colorbarProperties(iColorBar).Parent,'type','axes');
    %clim(colorbarProperties(iColorBar).Parent, colorbarProperties(iColorBar).Limits)
    currColorbar.Limits = colorbarProperties(iColorBar).Limits;
    currColorbar.Units = 'Normalized'; % set back to normalized so it scales with figure

    % add colorbar limits above and below
    currColorbar.Title.String = num2str(colorbarProperties(iColorBar).Limits(2));
    set(currColorbar.XLabel, {'String', 'Rotation', 'Position'}, {num2str(colorbarProperties(iColorBar).Limits(1)), ...
        0, [0.5 - 0.01, colorbarProperties(iColorBar).Limits(1) - 1]})
    currColorbar.TickLabels = {};


end