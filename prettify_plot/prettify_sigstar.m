function varargout = prettify_sigstar(groups, stats, nosort, fullPValue)
% sigstar - Add significance stars to bar charts, boxplots, line charts, etc,
%
% H = sigstar(groups,stats,nsort)
%
% Purpose
% Add stars and lines highlighting significant differences between pairs of groups.
% The user specifies the groups and associated p-values. The function handles much of
% the placement and drawing of the highlighting. Stars are drawn according to:
%   * represents p<=0.05
%  ** represents p<=1E-2
% *** represents p<=1E-3
%
%
% Inputs
% groups - a cell array defining the pairs of groups to compare. Groups defined
%          either as pairs of scalars indicating locations along the X axis or as
%          strings corresponding to X-tick labels. Groups can be a mixture of both
%          definition types.
% stats -  a vector of p-values the same length as groups. If empty or missing it's
%          assumed to be a vector of 0.05s the same length as groups. Nans are treated
%          as indicating non-significance.
% nsort -  optional, 0 by default. If 1, then significance markers are plotted in
%          the order found in groups. If 0, then they're sorted by the length of the
%          bar.
%
% Outputs
% H - optionally return handles for significance highlights. Each row is a different
%     highlight bar. The first column is the line. The second column is the text (stars).
%
%
% Examples
% 1.
% bar([5,2,1.5])
% sigstar({[1,2], [1,3]})
%
% 2.
% bar([5,2,1.5])
% sigstar({[2,3],[1,2], [1,3]},[nan,0.05,0.05])
%
% 3.  **DOESN'T WORK IN 2014b**
% R=randn(30,2);
% R(:,1)=R(:,1)+3;
% boxplot(R)
% set(gca,'XTick',1:2,'XTickLabel',{'A','B'})
% H=sigstar({{'A','B'}},0.01);
% ylim([-3,6.5])
% set(H,'color','r')
%
% 4. Note the difference in the order with which we define the groups in the
%    following two cases.
% x=[1,2,3,2,1];
% subplot(1,2,1)
% bar(x)
% sigstar({[1,2], [2,3], [4,5]})
% subplot(1,2,2)
% bar(x)
% sigstar({[2,3],[1,2], [4,5]})
%
% ALSO SEE: demo_sigstar
%
% KNOWN ISSUES:
% 1. Algorithm for identifying whether significance bar will overlap with
%    existing plot elements may not work in some cases (see line 277)
% 2. Bars may not look good on exported graphics with small page sizes.
%    Simply increasing the width and height of the graph with the
%    PaperPosition property of the current figure should fix things.
%
% Rob Campbell - CSHL 2013
% Modified 2023/11/05 Julie Fabre


%Input argument error checking

%If the user entered just one group pair and forgot to wrap it in a cell array
%then we'll go easy on them and wrap it here rather then generate an error

%% sanitize inputs 
if ~iscell(groups) && length(groups) == 2
    groups = {groups};
end

if nargin < 2 || isempty(stats)
    stats = repmat(0.05, 1, length(groups));
end
if nargin < 3 || isempty(nosort)
    nosort = 0;
end
if nargin < 4 || isempty(fullPValue)
    fullPValue = false;
end

%Check the inputs are of the right sort
if ~iscell(groups)
    error('groups must be a cell array')
end

if ~isvector(stats)
    error('stats must be a vector')
end

if length(stats) ~= length(groups)
    error('groups and stats must be the same length')
end

%% get indices for pairs
%Each member of the cell array groups may be one of three things:
%1. A pair of indices.
%2. A pair of strings (in cell array) referring to X-Tick labels
%3. A cell array containing one index and one string
%
% For our function to run, we will need to convert all of these into pairs of
% indices. Here we loop through groups and do this.

xlocs = nan(length(groups), 2); %matrix that will store the indices
xtl = get(gca, 'XTickLabel');

for ii = 1:length(groups)
    grp = groups{ii};

    if isnumeric(grp)
        xlocs(ii, :) = grp; %Just store the indices if they're the right format already

    elseif iscell(grp) %Handle string pairs or string/index pairs

        if isstr(grp{1})
            a = strmatch(grp{1}, xtl);
        elseif isnumeric(grp{1})
            a = grp{1};
        end
        if isstr(grp{2})
            b = strmatch(grp{2}, xtl);
        elseif isnumeric(grp{2})
            b = grp{2};
        end

        xlocs(ii, :) = [a, b];
    end

    %Ensure that the first column is always smaller number than the second
    xlocs(ii, :) = sort(xlocs(ii, :));

end

%If there are any NaNs we have messed up.
if any(isnan(xlocs(:)))
    error('Some groups were not found')
end


%Optionally sort sig bars from shortest to longest so we plot the shorter ones first
%in the loop below. Usually this will result in the neatest plot. If we waned to
%optimise the order the sig bars are plotted to produce the neatest plot, then this
%is where we'd do it. Not really worth the effort, though, as few plots are complicated
%enough to need this and the user can define the order very easily at the command line.
if ~nosort
    [~, ind] = sort(xlocs(:, 2)-xlocs(:, 1), 'ascend');
    xlocs = xlocs(ind, :);
    groups = groups(ind);
    stats = stats(ind);
end


%-----------------------------------------------------
%Add the sig bar lines and asterisks
holdstate = ishold;
hold on

H = ones(length(groups), 2); %The handles will be stored here

y = ylim;
yd = myRange(y) * 0.05; %separate sig bars vertically by 5%

for ii = 1:length(groups)
    thisY = findMinY(xlocs(ii, :)) + yd;
    H(ii, :) = makeSignificanceBar(xlocs(ii, :), thisY, stats(ii));
end
%-----------------------------------------------------


%Now we can add the little downward ticks on the ends of each line. We are
%being extra cautious and leaving this it to the end just in case the y limits
%of the graph have changed as we add the highlights. The ticks are set as a
%proportion of the y axis range and we want them all to be the same the same
%for all bars.
yd = myRange(ylim) * 0.01; %Ticks are 1% of the y axis range
for ii = 1:length(groups)
    y = get(H(ii, 1), 'YData');
    y(1) = y(1) - yd;
    y(4) = y(4) - yd;
    set(H(ii, 1), 'YData', y)
end


%Be neat and return hold state to whatever it was before we started
if ~holdstate
    hold off
elseif holdstate
    hold on
end


%Optionally return the handles to the plotted significance bars (first column of H)
%and asterisks (second column of H).
if nargout > 0
    varargout{1} = H;
end


end %close sigstar


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Internal functions

function H = makeSignificanceBar(x, y, p)
%makeSignificanceBar produces the bar and defines how many asterisks we get for a
%given p-value


if p <= 1E-3
    stars = '***';
elseif p <= 1E-2
    stars = '**';
elseif p <= 0.05
    stars = '*';
elseif isnan(p)
    stars = 'n.s.';
else
    stars = '';
end

x = repmat(x, 2, 1);
y = repmat(y, 4, 1);

H(1) = plot(x(:), y, '-k', 'LineWidth', 1.5, 'Tag', 'sigstar_bar');

%Increase offset between line and text if we will print "n.s."
%instead of a star.
if ~isnan(p)
    offset = 0.005;
else
    offset = 0.02;
end

starY = mean(y) + myRange(ylim) * offset;
H(2) = text(mean(x(:)), starY, stars, ...
    'HorizontalAlignment', 'Center', ...
    'BackGroundColor', 'none', ...
    'Tag', 'sigstar_stars');

Y = ylim;
if Y(2) < starY
    ylim([Y(1), starY + myRange(Y) * 0.05])
end


end %close makeSignificanceBar


function Y = findMinY(x)
% The significance bar needs to be plotted a reasonable distance above all the data points
% found over a particular range of X values. So we need to find these data and calculat the
% the minimum y value needed to clear all the plotted data present over this given range of
% x values.
%
% This version of the function is a fix from Evan Remington
oldXLim = get(gca, 'XLim');
oldYLim = get(gca, 'YLim');

axis(gca, 'tight')

%increase range of x values by 0.1 to ensure correct y max is used
x(1) = x(1) - 0.1;
x(2) = x(2) + 0.1;

set(gca, 'xlim', x) %Matlab automatically re-tightens y-axis

yLim = get(gca, 'YLim'); %Now have max y value of all elements within range.
Y = max(yLim);

axis(gca, 'normal')
set(gca, 'XLim', oldXLim, 'YLim', oldYLim)

end %close findMinY


function rng = myRange(x)
%replacement for stats toolbox range function
rng = max(x) - min(x);
end %close myRange
