%QQ nee to find axes / colorbar pairs, and adjust colormap there.

%'YTickLabel'

function prettify_colorbar(colorbars, colorsToReplace, mainColor, ChangeColormaps, DivergingColormap, ...
    SequentialColormap)
colorbarProperties = struct;


for iColorBar = 1:length(colorbars)
    currColorbar = colorbars(iColorBar);

    %text color
    colorbarProperties(iColorBar).Color = mainColor;

    % check/change colormap
    currColorbar.Limits = colorbars(iColorBar).Limits;
    if ChangeColormaps && currColorbar.Limits(1) < 0 && currColorbar.Limits(2) > 0 % equalize, and use a diverging colormap
        colormap(currColorbar.Parent, brewermap(100, DivergingColormap))
    elseif ChangeColormaps %use a sequential colormap
        colormap(currColorbar.Parent, brewermap(100, SequentialColormap))
    end
    colorbarProperties(iColorBar).Limits = currColorbar.Limits;
    colorbarProperties(iColorBar).Parent = currColorbar.Parent;

    % get label
    if ~isempty(currColorbar.Label.String) % label string
        label = currColorbar.Label.String;
        currColorbar.Label.String = '';
    elseif ~isempty(currColorbar.Title.String) % title
        label = currColorbar.Title.String;
        currColorbar.Title.String = '';
    elseif ~isempty(currColorbar.XLabel.String) %x label string
        label = currColorbar.Xlabel.String;
        currColorbar.Xlabel.String = '';
    else
        label = '';
    end

    colorbarProperties(iColorBar).Label = label;
    currColorbar.Units = 'Points'; %get old, add padding.
    colorbarProperties(iColorBar).Position_ori = currColorbar.Position;

    currColorbar.Parent = colorbarProperties(iColorBar).Parent;
    %currColorbar.Units = 'Points';
    %colorbarProperties(iColorBar).Parent.Position
    %currColorbar.Position = [colorbarProperties(iColorBar).Position_ori(1)+35, colorbarProperties(iColorBar).Position_ori(2:4)]; % QQ hardcoded
    currColorbar.Label.String = colorbarProperties(iColorBar).Label; %QQ error - this doesn't work with XLabel.
    currColorbar.Label.Rotation = 270; % Rotate the label to be horizontal
    currColorbar.Label.Position = [3, 0.5, 0]; % You might need to adjust these values

    currColorbar.Limits = colorbarProperties(iColorBar).Limits;
    %currColorbar.Units = 'Normalized'; % set back to normalized so it scales with figure

    % add colorbar limits above and below
    % currColorbar.Title.String = num2str(colorbarProperties(iColorBar).Limits(2));
    % set(currColorbar.XLabel, {'String', 'Rotation', 'Position'}, {num2str(colorbarProperties(iColorBar).Limits(1)), ...
    %    0, [0.5 - 0.01, colorbarProperties(iColorBar).Limits(1) - 1]})
    currColorbar.Ticks = [currColorbar.Limits(1), currColorbar.Limits(2)];
    currColorbar.TickLabels = {num2str(currColorbar.Limits(1)), num2str(currColorbar.Limits(2))};
    currColorbar.Color = colorbarProperties(iColorBar).Color;


end
