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
        labels{end+1} = name;

        % Calculate label positions
        if strcmp(h.Type, 'histogram')
            xdata = h.BinCounts;
            ydata = h.BinEdges;
        else
            xdata = h.XData;
            ydata = h.YData;
        end
        labelX = xdata(end); % Adjust as needed, e.g., xdata(1) for left alignment

    end

    % Sort the labels based on their x-positions
    labels_empty = cellfun(@isempty, labels);
    [~, order] = sort(cellfun(@(x) find(strcmp(labels(~labels_empty), x)), labels(~labels_empty)));
    nonEmptyHandles = handles(~labels_empty);
    legend(ax, nonEmptyHandles (order));

else
    % Define wdt and hgt
    wdt = 0.1; % Adjust the width as needed
    hgt = 0.02; % Adjust the height as needed
    tol = 0.01;

    yRange = ax.YLim(2) - ax.YLim(1);
    offset = yRange * 0.05; % Adjust offset as needed for better spacing

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
        text(ax, Pos(1), Pos(2)+offset, name, 'Color', color, 'FontWeight', 'bold', 'HorizontalAlignment', 'right', 'BackgroundColor', 'none');
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
        text(ax, Pos(1), Pos(2)+offset, name, 'Color', color, 'FontWeight', 'bold', 'HorizontalAlignment', 'center', 'BackgroundColor', 'none');
    end

    % Remove legend
    ax.Legend.Visible = 'off';
end


%-------MODIFIED VERSION OF LSCAN FOLLOWS------
function [Pos] = tscan(ha, wdt, hgt, tol, stickytol, hl)
%TSCAN  Scan for good text location.
%   TSCAN is used by TEXTBP to determine a "good" place for
%   the text to appear. TSCAN returns either the
%   position (in figure normalized units) the text should
%   appear, or a -1 if no "good" place is found.
%
%   TSCAN searches for the best place on the graph according
%   to the following rules.
%       1. Text must obscure as few data points as possible.
%          Number of data points the text may cover before plot
%          is "squeezed" can be set with TOL. The default is a
%          large number so to enable squeezing, TOL must
%          be set. A negative TOL will force squeezing.
%       2. Regions with neighboring empty space are better.
%       3. Bottom and Left are better than Top and Right.
%      x 4. If a legend already exists and has been manually placed,
%      x    then try to put new legend "close" to old one.
%
%   TSCAN(HA,WDT,HGT,TOL,STICKYTOL,HL) returns a 2 element
%   position vector. WDT and HGT are the Width and Height of
%   the legend object in figure normalized units. TOL
%   and STICKYTOL are tolerances for covering up data points.
%   HL is the handle of the current legend or -1 if none exist.
%
%   TSCAN(HA,WDT,HGT,TOL) allows up to TOL data
%   points to be covered when selecting the best
%   text location.
%
%changes from LSCAN
%1. existing text bracketing fixed
%2. sticky references removed for clarity
%3. returns position in plot normalized units, not figure
%normalized (0,0 is LL axis, not LL of figure)
% data point extraction for-loop modified to handle histograms
% properly (Peter Mao, 6/21/11).
% modified from LSCAN by Peter Mao 6/16/06.
%   Drea Thomas     5/7/93
%   Copyright 1984-2005 The MathWorks, Inc.
%   $Revision: 1.1 $  $Date: 2006/06/19 21:08:48 $
%   $Revision: 1.2 $  $Date: 2011/06/21 09:41 $
% Defaults
debug = 0;
if debug > 1
    holdstatOFF = ~ishold; %%
    hold on; %%
end
% Calculate tile size
% save old units

%% this part makes text walk across screen with repeated use!!
%axoldunits = get(ha,'units');
%set(ha,'units','normalized')
%cap=get(ha,'Position'); %[fig]
%set(ha,'units',axoldunits);
cap = [0, 0, 1, 1]; %'position' in [norm] units
xlim = get(ha, 'Xlim'); %[data]
ylim = get(ha, 'Ylim'); %[data]
H = ylim(2) - ylim(1);
W = xlim(2) - xlim(1);
dh = .03 * H;
dw = .03 * W; % Scale so legend is away from edge of plot
H = .94 * H;
W = .94 * W;
xlim = xlim + [dw, -dw];
ylim = ylim + [dh, -dh];
Hgt = hgt / cap(4) * H; %[data]
Wdt = wdt / cap(3) * W;
Thgt = H / round(-.5+H/Hgt);
Twdt = W / round(-.5+W/Wdt);
% Get data, points and text
% legend, not included here, is a child of gcf with 'tag' 'legend'
Kids = get(ha, 'children');
Xdata = [];
Ydata = [];
for i = 1:size(Kids),
    Xtemp = [];
    Ytemp = [];
    if strcmp(get(Kids(i), 'type'), 'line'),
        Xtemp = get(Kids(i), 'Xdata');
        Ytemp = get(Kids(i), 'Ydata');
    elseif strcmp(get(Kids(i), 'type'), 'patch'), % for histograms
        % X/Ydata from patch are LL,UL,UR,LR for each bar.
        % loop below fills in the edges of the bar with fake data
        Xtemp0 = get(Kids(i), 'Xdata');
        Ytemp0 = get(Kids(i), 'Ydata');
        Xstart = Xtemp0(1, :);
        Yend = Ytemp0(2, :);
        for jj = 1:length(Xstart)
            thisY = Yend(jj):-Thgt:0;
            thisX = repmat(Xstart(jj), 1, length(thisY));
            Xtemp = [Xtemp, thisX];
            Ytemp = [Ytemp, thisY];
        end
    elseif strcmp(get(Kids(i), 'type'), 'text'),
        tmpunits = get(Kids(i), 'units');
        set(Kids(i), 'units', 'data')
        %        tmp=get(Kids(i),'Position');
        ext = get(Kids(i), 'Extent');
        set(Kids(i), 'units', tmpunits);
        %        Xdata=[Xdata,[tmp(1) tmp(1)+ext(3)]];
        %        Ydata=[Ydata,[tmp(2) tmp(2)+ext(4)]];
        Xtemp = [ext(1), ext(1), ext(1) + ext(3), ext(1) + ext(3) * .5, ext(1) + ext(3)];
        Ytemp = [ext(2), ext(2) + ext(4), ext(2), ext(2) + ext(4) * .5, ext(2) + ext(4)];
    end
    Xdata = [Xdata; Xtemp(:)];
    Ydata = [Ydata; Ytemp(:)];
end
if debug > 1, plot(Xdata, Ydata, 'r.'); end
%   Determine # of data points under each "tile"
i = 1;
j = 1;
for yp = ylim(1):Thgt / 2:(ylim(2) - Thgt),
    i = 1;
    for xp = xlim(1):Twdt / 2:(xlim(2) - Twdt),
        pop(j, i) = ...
            sum(sum((Xdata >= xp).*(Xdata <= xp + Twdt).*(Ydata >= yp).*(Ydata <= yp + Thgt)));
        %       line([xp xp],[ylim(1) ylim(2)]);
        i = i + 1;
    end
    %    line([xlim(1) xlim(2)],[yp yp]);
    j = j + 1;
end
% Cover up fewest points.
minpop = min(min(pop));
if debug, disp(sprintf('minimally covering tile convers %d points', minpop)); end
if minpop > tol,
    Pos = -1;
    warning('Raise TOL in calling function to %d', minpop);
    return
end
%%%%%%%%%%%%%%%%%%%%%%
%                    %
%sticky stuff removed%
%                    %
%%%%%%%%%%%%%%%%%%%%%%
popmin = pop == min(min(pop));
if sum(sum(popmin)) > 1, % Multiple minima in # of points
    [a, b] = size(pop);
    if min(a, b) > 1, %check over all tiles

        % Look at adjacent tiles and see if they are empty
        % adds in h/v nearest neighbors, double add if on an edge
        pop = [pop(2, :)', pop(1:(a - 1), :)']' + [pop(2:(a), :)', pop((a - 1), :)']' + ...
            [pop(:, 2), pop(:, 1:(b - 1))] + [pop(:, 2:b), pop(:, (b - 1))] + pop;
        % LSCAN had two calls to the line above w/o the trailing "+ pop"
        popx = popmin .* (pop == min(pop(popmin)));
        if sum(sum(popx)) > 1, % prefer bottom left to top right
            flag = 1;
            i = 1;
            j = 1;
            while flag,
                if flag == 2,
                    if popx(i, j) == 1,
                        popx = popx * 0;
                        popx(i, j) = 1;
                        flag = 0;
                    else
                        popx = popx * 0;
                        popx(i, j+1) = 1;
                        flag = 0;
                    end
                else
                    if popx(i, j) == 1,
                        flag = 2;
                        popx = popx * 0;
                        popx(i, j) = 1;
                    else
                        j = j + 1;
                        if j == b + 1,
                            j = 1;
                            i = i + 1;
                        end
                        if i == a + 1, % my add'n
                            i = 1;
                            flag = 2;
                        end
                    end
                end
            end
        end
    else % only one tile
        popx = popmin * 0;
        popx(1, 1) = 1;
    end
else % Only 1 minima in # covered points
    popx = popmin;
end
%recover i,j location that we want to use
i = find(max(popx));
i = i(1);
j = find(max(popx'));
j = j(1);
Pos = [((i - 1) / (W / Twdt * 2 / .94) + .03) * cap(3) + cap(1), ((j - 1) / (H / Thgt * 2 / .94) + .03) * cap(4) + cap(2)];
if debug, disp(sprintf('(i,j) = (%d,%d)', i, j)); end
if debug > 1
    if holdstatOFF
        hold off
    end
end
