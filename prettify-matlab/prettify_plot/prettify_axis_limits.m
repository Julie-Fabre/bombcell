
function prettify_axis_limits(all_axes, currFig_children, ax_pos, xlims_subplot, ylims_subplot, clims_subplot, ...
    XLimits, YLimits, CLimits, LimitsRound, SymmetricalCLimits)

if ~isnan(LimitsRound) % round up all limits to the nearest LimitsRound decimal place
    xlims_subplot = arrayfun(@(x) prettify_roundUpNatural(x, LimitsRound), xlims_subplot);
    ylims_subplot = arrayfun(@(x) prettify_roundUpNatural(x, LimitsRound), ylims_subplot);
    clims_subplot = arrayfun(@(x) prettify_roundUpNatural(x, LimitsRound), clims_subplot);
end

% homogenize x, y, and climits across rows/columns of plots.
col_subplots = unique(ax_pos(:, 1));
row_subplots = unique(ax_pos(:, 2));

% Set Xlimits
if isnumeric(XLimits)
    for iAx = 1:size(all_axes, 2)
        thisAx = all_axes(iAx);
        currAx = currFig_children(thisAx);
        set(currAx, 'XLim', [XLimits(1), XLimits(2)]);
    end
elseif ismember(XLimits, {'all', 'rows', 'cols'})
    col_xlims = arrayfun(@(x) [min(min(xlims_subplot(ax_pos(:, 1) == col_subplots(x), :))), ...
        max(max(xlims_subplot(ax_pos(:, 1) == col_subplots(x), :)))], 1:size(col_subplots, 1), 'UniformOutput', false);
    row_xlims = arrayfun(@(x) [min(min(xlims_subplot(ax_pos(:, 2) == row_subplots(x), :))), ...
        max(max(xlims_subplot(ax_pos(:, 2) == row_subplots(x), :)))], 1:size(row_subplots, 1), 'UniformOutput', false);
    for iAx = 1:size(all_axes, 2)
        thisAx = all_axes(iAx);
        currAx = currFig_children(thisAx);
        if ~isempty(currAx) %(currAx, limits, limitRows, limitCols, axPos, limitIdx_row, limitIdx_col, limitType)
            setNewXYLimits(currAx, xlims_subplot, row_xlims, col_xlims, ax_pos, row_subplots, ...
                col_subplots, XLimits, 'Xlim', iAx) %set x limits
        end

    end
end

% Set Ylimits
if isnumeric(YLimits)
    for iAx = 1:size(all_axes, 2)
        thisAx = all_axes(iAx);
        currAx = currFig_children(thisAx);
        set(currAx, 'YLim', [YLimits(1), YLimits(2)]);
    end
elseif ismember(YLimits, {'all', 'rows', 'cols'})
    col_ylims = arrayfun(@(x) [min(min(ylims_subplot(ax_pos(:, 1) == col_subplots(x), :))), ...
        max(max(ylims_subplot(ax_pos(:, 1) == col_subplots(x), :)))], 1:size(col_subplots, 1), 'UniformOutput', false);
    row_ylims = arrayfun(@(x) [min(min(ylims_subplot(ax_pos(:, 2) == row_subplots(x), :))), ...
        max(max(ylims_subplot(ax_pos(:, 2) == row_subplots(x), :)))], 1:size(row_subplots, 1), 'UniformOutput', false);
    for iAx = 1:size(all_axes, 2)
        thisAx = all_axes(iAx);
        currAx = currFig_children(thisAx);
        if ~isempty(currAx) %(currAx, limits, limitRows, limitCols, axPos, limitIdx_row, limitIdx_col, limitType)
            setNewXYLimits(currAx, ylims_subplot, row_ylims, col_ylims, ax_pos, row_subplots, ...
                col_subplots, YLimits, 'Ylim', iAx) %set y limits
        end
    end
end

% Set Climits
if isnumeric(CLimits)
    for iAx = 1:size(all_axes, 2)
        thisAx = all_axes(iAx);
        currAx = currFig_children(thisAx);
        set(currAx, 'CLim', [CLimits(1), CLimits(2)]);
    end
elseif ismember(CLimits, {'all', 'rows', 'cols'})
    % get rows and cols
    col_subplots = unique(ax_pos(:, 1));
    row_subplots = unique(ax_pos(:, 2));
    col_clims = arrayfun(@(x) [min(min(clims_subplot(ax_pos(:, 1) == col_subplots(x), :))), ...
        max(max(clims_subplot(ax_pos(:, 1) == col_subplots(x), :)))], 1:size(col_subplots, 1), 'UniformOutput', false);
    row_clims = arrayfun(@(x) [min(min(clims_subplot(ax_pos(:, 2) == row_subplots(x), :))), ...
        max(max(clims_subplot(ax_pos(:, 2) == row_subplots(x), :)))], 1:size(row_subplots, 1), 'UniformOutput', false);
    for iAx = 1:size(all_axes, 2)
        thisAx = all_axes(iAx);
        currAx = currFig_children(thisAx);
        if ~isempty(currAx) %(currAx, limits, limitRows, limitCols, axPos, limitIdx_row, limitIdx_col, limitType)
            if ismember(CLimits, {'all'})
                theseCLims = clims_subplot;
            elseif ismember(CLimits, {'cols'})
                theseCLims = col_clims{ax_pos(iAx, 1) == col_subplots};
            elseif ismember(CLimits, {'rows'})
                theseCLims = row_clims{ax_pos(iAx, 2) == row_subplots};
            else
                theseCLims = clims_subplot(iAx, :);
            end
            setNewCLimits(currAx, theseCLims, SymmetricalCLimits)
        end
    end


elseif SymmetricalCLimits
    for iAx = 1:size(all_axes, 2)
        thisAx = all_axes(iAx);
        currAx = currFig_children(thisAx);
        theseCLims = currAx.CLim;
        setNewCLimits(currAx, theseCLims, SymmetricalCLimits)
    end
end


    function setNewXYLimits(currAx, limits, limitRows, limitCols, axPos, limitIdx_row, limitIdx_col, limitType, textLim, iAx)
        if ismember(limitType, {'all'})
            set(currAx, textLim, [min(min(limits)), max(max(limits))]);
        elseif ismember(limitType, {'rows'})
            set(currAx, textLim, limitRows{limitIdx_row == axPos(iAx, 2)});
        elseif ismember(limitType, {'cols'})
            set(currAx, textLim, limitCols{limitIdx_col == axPos(iAx, 1)});
        end
    end

    function setNewCLimits(currAx, theseCLims, SymmetricalCLimits)
        if any(any(~isnan(theseCLims)))
            if SymmetricalCLimits && any(any(theseCLims < 0)) && any(any(theseCLims > 0)) % diverging
                set(currAx, 'Clim', [-nanmax(nanmax(abs(theseCLims))), nanmax(nanmax(abs(theseCLims)))]);
            elseif SymmetricalCLimits && any(any(theseCLims < 0))
                set(currAx, 'Clim', [-nanmax(nanmax(abs(theseCLims))), 0]);
            elseif SymmetricalCLimits && any(any(theseCLims > 0))
                set(currAx, 'Clim', [0, nanmax(nanmax(abs(theseCLims)))]);
            else
                set(currAx, 'Clim', [nanmin(nanmin(theseCLims)), nanmax(nanmax(theseCLims))]);
            end
        end
    end
end
