

function makepretty(color,  titleSize, labelSize, textSize, markerSize, lineWidth, textColor)

if nargin < 7
    if nargin < 1 
        textColor = 'k';
    elseif strcmp(color, 'k')
        textColor = 'w';
    elseif strcmp(color, 'none')
        textColor = [0.7, 0.7, 0.7]; % gray, will will work on most backgrounds
    end
end
if nargin < 6
    lineWidth = 2;
end
if nargin < 5
    markerSize = 15;
end
if nargin < 4
    textSize = 13;
end
if nargin < 3
    labelSize = 17;
end
if nargin < 2
    titleSize = 20;
end
if nargin < 1
    color = 'w';
end


% set some graphical attributes of the current axis
set(gcf,'color',color);
set(gca,'color',color);

set(get(gca, 'XLabel'), 'FontSize', labelSize);
set(get(gca, 'YLabel'), 'FontSize', labelSize);
set(gca, 'FontSize', textSize);
set(get(gca, 'Title'), 'FontSize', titleSize);

set(get(gca, 'XLabel'), 'Color', textColor);
set(get(gca, 'YLabel'), 'Color', textColor);
set(gca, 'GridColor', textColor);
set(gca, 'YColor', textColor);
set(gca, 'XColor', textColor);
set(gca, 'MinorGridColor', textColor);
set(get(gca, 'Title'), 'FontSize', titleSize);


ch = get(gca, 'Children');
axis tight;
for c = 1:length(ch)
    thisChild = ch(c);
    if strcmp('line', get(thisChild, 'Type')) 
        if strcmp('.', get(thisChild, 'Marker'))
            set(thisChild, 'MarkerSize', markerSize);
        end
        if strcmp('-', get(thisChild, 'LineStyle'))
            set(thisChild, 'LineWidth', lineWidth);
        end
    end
end
