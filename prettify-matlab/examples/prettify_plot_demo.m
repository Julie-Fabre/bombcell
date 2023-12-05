
% generate an uglyPlot
uglyPlot;

% Make it pretty with default settings 
prettify_plot;

% Change the figure color, fonts, fontsizes, ticks 
prettify_plot('FigureColor', 'k')
prettify_plot('FigureColor', 'none', 'Font', 'Times','TitleFontSize', 12, 'LabelFontSize', 14,'GeneralFontSize', 12,...
    'AxisTicks', 'in', 'TickLength', 0.01, 'TickWidth', 1)

% let's go back to the defaults
prettify_plot;

% Homogenize the x, y and climits by rows of subplots (other options include by columns
% ('col') and for all aubplots ('all')
prettify_plot('YLimits', 'cols', 'XLimits', 'all')

% Change colormaps, make symmetric 
prettify_plot('CLimits', 'all', 'SymmetricalCLimits', true, 'ChangeColormaps', true)

% Change the legends
prettify_plot('LegendReplace', true)

% Replace axis by scale bars 
prettify_addScaleBars

% Add p-values 