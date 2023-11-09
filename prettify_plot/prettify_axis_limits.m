
function prettify_axis_limits(all_axes, currFig_children, ax_pos, xlims_subplot, ylims_subplot, clims_subplot, ...
    XLimits, YLimits, CLimits, LimitsRound, SymmetricalCLimits)

if ~isnan(LimitsRound) % round up all limits to the nearest LimitsRound decimal place
    xlims_subplot = arrayfun(@(x) prettify_roundUpNatural(x, LimitsRound), xlims_subplot);
    ylims_subplot = arrayfun(@(x) prettify_roundUpNatural(x, LimitsRound), ylims_subplot);
    clims_subplot = arrayfun(@(x) prettify_roundUpNatural(x, LimitsRound), clims_subplot);
end

% homogenize x, y, and climits across rows/columns of plots.
if ismember(XLimits, {'all', 'row', 'col'}) || ismember(YLimits, {'all', 'row', 'col'}) || ismember(CLimits, {'all', 'row', 'col'})
    % get rows and cols
    col_subplots = unique(ax_pos(:, 1));
    row_subplots = unique(ax_pos(:, 2));

    col_xlims = arrayfun(@(x) [min(min(xlims_subplot(ax_pos(:, 1) == col_subplots(x), :))), ...
        max(max(xlims_subplot(ax_pos(:, 1) == col_subplots(x), :)))], 1:size(col_subplots, 1), 'UniformOutput', false);
    row_xlims = arrayfun(@(x) [min(min(xlims_subplot(ax_pos(:, 2) == row_subplots(x), :))), ...
        max(max(xlims_subplot(ax_pos(:, 2) == row_subplots(x), :)))], 1:size(row_subplots, 1), 'UniformOutput', false);
    col_ylims = arrayfun(@(x) [min(min(ylims_subplot(ax_pos(:, 1) == col_subplots(x), :))), ...
        max(max(ylims_subplot(ax_pos(:, 1) == col_subplots(x), :)))], 1:size(col_subplots, 1), 'UniformOutput', false);
    row_ylims = arrayfun(@(x) [min(min(ylims_subplot(ax_pos(:, 2) == row_subplots(x), :))), ...
        max(max(ylims_subplot(ax_pos(:, 2) == row_subplots(x), :)))], 1:size(row_subplots, 1), 'UniformOutput', false);
    col_clims = arrayfun(@(x) [min(min(clims_subplot(ax_pos(:, 1) == col_subplots(x), :))), ...
        max(max(clims_subplot(ax_pos(:, 1) == col_subplots(x), :)))], 1:size(col_subplots, 1), 'UniformOutput', false);
    row_clims = arrayfun(@(x) [min(min(clims_subplot(ax_pos(:, 2) == row_subplots(x), :))), ...
        max(max(clims_subplot(ax_pos(:, 2) == row_subplots(x), :)))], 1:size(row_subplots, 1), 'UniformOutput', false);


    for iAx = 1:size(all_axes, 2)
        thisAx = all_axes(iAx);
        currAx = currFig_children(thisAx);
        if ~isempty(currAx) %(currAx, limits, limitRows, limitCols, axPos, limitIdx_row, limitIdx_col, limitType)
            setNewXYLimits(currAx, xlims_subplot, row_xlims, col_xlims, ax_pos, row_subplots, col_subplots, XLimits, 'Xlim') %set x limits
            setNewXYLimits(currAx, ylims_subplot, row_ylims, col_ylims, ax_pos, row_subplots, col_subplots, YLimits, 'Ylim') %set y limits


            if ismember(CLimits, {'all'})
                theseCLims = clims_subplot;
            elseif ismember(CLimits, {'col'})
                theseCLims = col_clims{ax_pos(iAx, 1) == col_subplots};
            elseif ismember(CLimits, {'row'})
                theseCLims = row_clims{ax_pos(iAx, 2) == row_subplots};
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


    function setNewXYLimits(currAx, limits, limitRows, limitCols, axPos, limitIdx_row, limitIdx_col, limitType, textLim)
        if ismember(limitType, {'all'})
            set(currAx, textLim, [min(min(limits)), max(max(limits))]);
        elseif ismember(limitType, {'row'})
            set(currAx, textLim, limitRows{axPos(:, 2) == limitIdx_row});
        elseif ismember(limitType, {'col'})
            set(currAx, textLim, limitCols{axPos(:, 1) == limitIdx_col});
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
