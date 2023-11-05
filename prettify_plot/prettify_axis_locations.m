%Adjust axis position
set(currAx, 'Units', options.AxisUnits); % Set units to pixels
if ~strcmp(options.AxisWidth, 'keep') && ~strcmp(options.AxisHeight, 'keep')
    AxisHeight = str2num(options.AxisHeight);
    AxisWidth = str2num(options.AxisWidth);
    ax_pos_ori = get(currAx, 'Position');
    % Scale subplot width and height independently to maintain aspect ratio
    scale_w = AxisWidth / ax_pos_ori(3);
    scale_h = AxisHeight / ax_pos_ori(4);
    new_pos = [ax_pos_ori(1), ax_pos_ori(2) + (ax_pos_ori(4) - AxisHeight * scale_h),...
               ax_pos_ori(3) * scale_w, ax_pos_ori(4) * scale_h];
    set(currAx, 'Position', new_pos);
    ax_pos(iAx,:) = new_pos;
elseif ~strcmp(options.AxisWidth, 'keep')

    AxisWidth = str2num(options.AxisWidth);
    ax_pos_ori = get(currAx, 'Position');
    % Scale subplot width and height independently to maintain aspect ratio
    scale_w = AxisWidth / ax_pos_ori(3);
    scale_h = 1;
    new_pos = [ax_pos_ori(1), ax_pos_ori(2) + ax_pos_ori(4),...
               ax_pos_ori(3) * scale_w, ax_pos_ori(4)];
    set(currAx, 'Position', new_pos);
    ax_pos(iAx,:) = new_pos;
elseif ~strcmp(options.AxisHeight, 'keep')
     AxisHeight = str2num(options.AxisHeight);
    ax_pos_ori = get(currAx, 'Position');
    % Scale subplot width and height independently to maintain aspect ratio
    scale_w = 1;
    scale_h = AxisHeight / ax_pos_ori(4);
    new_pos = [ax_pos_ori(1), ax_pos_ori(2) + (ax_pos_ori(4) - AxisHeight * scale_h),...
               ax_pos_ori(3) * scale_w, ax_pos_ori(4) * scale_h];
    set(currAx, 'Position', new_pos);
    ax_pos(iAx,:) = new_pos;
else
    ax_pos = get(currAx, 'Position');
end