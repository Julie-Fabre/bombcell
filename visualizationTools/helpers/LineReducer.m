classdef LineReducer < handle
    
% LinePlotReducer
%
% Manages the information in a standard MATLAB plot so that only the
% necessary number of data points are shown. For instance, if the width of
% the axis in the plot is only 500 pixels, there's no reason to have more
% than 1000 data points along the width. This tool selects which data
% points to show so that, for each pixel, all of the data mapping to that
% pixel is crushed down to just two points, a minimum and a maximum. Since
% all of the data is between the minimum and maximum, the user will not see
% any difference in the reduced plot compared to the full plot. Further, as
% the user zooms in or changes the figure size, this tool will create a new
% map of reduced points for the new axes limits automatically (it requires
% no further user input).
% 
% Using this tool, users can plot huge amounts of data without their 
% machines becoming unresponsive, and yet they will still "see" all of the 
% data that they would if they had plotted every single point.
%
% To keep things simple, the interface allows a user to pass in arguments
% in the same way those arguments would be passed directly to most line
% plot commands. For instance:
%
% plot(t, x);
%
% Becomes:
%
% LinePlotReducer(t, x);
%
% More arguments work as well.
%
% plot(t, x, 'r:', t, y, 'b', 'LineWidth', 3);
%
% Becomes:
%
% LinePlotReducer(t, x, 'r:', t, y, 'b', 'LineWidth', 3);
%
% Note that LinePlotReducer returns a LinePlotReducer object as output.
%
% lpr = LinePlotReducer(t, x);
%
% Another function, reduce_plot, takes exactly the same arguments as
% LinePlotReducer, but returns the plot handles instead of a
% LinePlotReducer object.
%
% h_plots = reduce_plot(t, x);
%
% One can use reduce_plot or LinePlotReducer according to one's comfort
% with using objects in MATLAB. By using reduce_plot, one does not need to
% use objects if one doesn't want to.
%
% The plot handles are also available as a public property of
% LinePlotReducer called h_plot. These handles would allow one to, e.g.,
% change a line color or marker.
%
% By default 'plot' is the function used to display the data, however,
% other functions can be used as well. For instance, to use 'stairs':
%
% LinePlotReducer(@stairs, t, x);
%
% Alternately, if one already has an existing plot, but wants a
% LinePlotReducer to manage it, one can simply pass the plot handles to a
% new LinePlotReducer, such as:
%
% h = plot(t, x, 'r:');
% LinePlotReducer(h);
%
% Finally, one can also set up a plot with a "small" set of data, then pass
% the plot handle and full x and y data to LinePlotReducer. This allows a
% user to create a detailed custom plot, but still use the LinePlotReducer
% without ever having to plot all of the data. For instance:
%
% h = plot(t([1 end]), x([1 end]), 'rd--', t([1 end]), y([1 end]), 'bs');
% LinePlotReducer(h, t, x, t, y);
%
% One can still use normal zooming and panning tools, whether in the figure
% window or from the command line, and the LinePlotReducer will still
% notice that the axes limits or size are changing and will automatically
% create a new, reduced data set to fit the current size.
%
% LinePlotReducer looks best on continuous lines. When plotting points
% only (with no connecting line), it might be noticeable that only the 
% minimum and maximum are showing up in a plot. A user can still explore
% the data quickly, and details will always be filled in when the user
% zooms (all the way down to the raw data).
%
% Finally, for those who need to zoom and pan frequently, a utility is
% included to make this a little faster. When a LinePlotExplorer is applied
% to a figure, it allows the user to zoom in and out with the scroll wheel
% and pan by clicking and dragging. Left and right bounds can also be
% passed to LinePlotExplorer.
%
% lpe = LinePlotExplorer(gcf(), 0, 5);
%
% The LinePlotExplorer is not strictly related to the LinePlotReducer.
% However, frequent zooming and handling large data so frequently occur at
% the same time that this class was included for convenience.
%
% --- Change Log ---
%
% 2015-05-31: Reduced long-term memory usage by making a LinePlotReducer
% delete its listener for events from the figure. Without explicitly
% deleting this listener, the LinePlotReducer remains registered and so is
% not deleted. As new LinePlotReducers are used in the same figure, memory
% usage grows. Thanks to Jack for pointing this out. Changes copyright 2015
% Tucker McClure.
%
% 2014-06-04: Now allows multiple LinePlotReducer objects in the same axes.
% Changes copyright 2014 Tucker McClure.
% 
% 2014-01-15: Now allows input as combination of rows and columns, just 
% like the regular plot functions. Changes copyright 2014 Tucker McClure.
%
% 2014-01-06: Fixed a bug when "taking over" a plot with only a single
% line. Also fixed a bug with the final line spec being ignored. Changes 
% copyright 2014 Tucker McClure.
%
% 2013-03-15: Original. Copyright 2013, The MathWorks, Inc.
%
% ---
%
% Copyright 2015, The MathWorks, Inc. and Tucker McClure


        
        % Create a ReductiveViewer for the x and y variables.
        function o = LinePlotReducer(varargin)
            
            % We're busy. Ignore resizing and things.
            o.busy = true;

            % If the user is just passing in an array of plot handles,
            % we'll take over managing the data shown in the plot.
            taking_over_existing_plot =    nargin >= 1 ...
                                        && isvector(varargin{1}) ...
                                        && all(ishandle(varargin{1}));
            if taking_over_existing_plot

                % Record the handles.
                o.h_plot   = varargin{1};
                o.h_axes   = get(o.h_plot(1), 'Parent');
                o.h_figure = get(o.h_axes,    'Parent');
                 
                % Get the original data either from the plot or from input
                % arguments.
                if nargin == 1
                    
                    o.x = get(o.h_plot, 'XData');
                    o.y = get(o.h_plot, 'YData');
                    o.y_to_x_map = 1:size(o.y, 1);
                    
                    % If there are multiple lines, o.x will be a cell
                    % array, which is how we use it from here on. If 
                    % there's only one line, we need to make it a cell
                    % array.
                    if ~iscell(o.x)
                        o.x = {o.x};
                    end
                    if ~iscell(o.y)
                        o.y = {o.y};
                    end
                    
                    % Format the data as columns.
                    for k = 1:length(o.x)
                        o.x{k} = o.x{k}(:);
                        o.y{k} = o.y{k}(:);
                    end
                    
                end
                
                start = 2;
                axes_specified = false;
                
            % Otherwise, we need to plot the data.
            else
                
                % The first argument might be a function handle or it might
                % just be the start of the data. 'next' will represent the
                % index we need to examine next.
                start = 1;

                % If the first input is a function handle, use it to plot.
                % Otherwise, use the normal @plot function.
                if isa(varargin{start}, 'function_handle')
                    plot_fcn = varargin{1};
                    start = start + 1;
                else
                    plot_fcn = @plot;
                end

                % Check for an axes input.
                if    isscalar(varargin{start}) ...
                   && ishandle(varargin{start}) ...
                   && strcmp(get(varargin{start}, 'Type'), 'axes')
                
                    % User provided the axes. Keep 'em.
                    o.h_axes = varargin{start};
                    
                    % Get the figure.
                    o.h_figure = get(o.h_axes, 'Parent');
                    
                    % Make them active.
                    set(0, 'CurrentFigure', o.h_figure);
                    set(o.h_figure, 'CurrentAxes', o.h_axes);
                    
                    % Move the start.
                    start = start + 1;
                    
                    axes_specified = true;
                    
                else
                    
                    % Record the handles.
                    o.h_figure   = gcf();
                    o.h_axes     = gca();
                    
                    axes_specified = false;
                    
                end
                
            end

            % Function to check if something's a line spec
            expr = '[^rgbcmykw\-\:\.\+o\*xsd\^v\>\<ph]';
            is_line_spec = @(s)    ischar(s) ...
                && isempty(regexp(s, expr, 'once'));

            % A place to store the linespecs as we find them.
            linespecs = {};

            % Loop through all of the inputs.
            km1_was_x = false;
            ym = [];
            for k = start:nargin+1

                % If it's a bunch of numbers...
                if k <= nargin && isnumeric(varargin{k})

                    % If we already have an x, then this must be y.
                    if km1_was_x

                        % Rename for simplicity.
                        ym = varargin{k};
                        xm = varargin{k-1};
                        
                        % We can accept data in rows or columns. If this is
                        % 1-by-n -> 1 series from columns
                        % m-by-n -> n series from columns
                        % m-by-1 -> 1 series from rows (transpose)
                        if size(xm, 1) == 1
                            xm = xm.';
                        end
                        if size(ym, 1) == 1
                            ym = ym.';
                        end

                        % Transpose if necessary.
                        if size(xm, 1) ~= size(ym, 1)
                            ym = ym';
                        end

                        % Store y, x, and a map from y index to x
                        % index.
                        for c = 1:size(ym, 2)
                            if c <= size(xm, 2)
                                o.x{end+1} = xm(:, c);
                            end
                            o.y{end+1} = ym(:, c);
                            o.y_to_x_map(end+1) = length(o.x);
                        end

                        % We've now matched this x.
                        km1_was_x = false;

                    % If we don't have an x, this must be x.
                    else
                        km1_was_x = true;
                    end

                % It's not numeric.
                else

                    % If we had an x and were looking for a y, it
                    % probably was actually a y with an implied x.
                    if km1_was_x

                        % Rename for simplicity.
                        ym = varargin{k-1};
                        
                        % We can accept data in rows or columns. If this is
                        % 1-by-n -> 1 series from columns
                        % m-by-n -> n series from columns
                        % m-by-1 -> 1 series from rows (transpose)
                        if size(ym, 1) == 1
                            ym = ym.';
                        end
                        
                        % Make the implied x explicit.
                        o.x{end+1} = (1:size(ym, 1))';

                        % Store y, x, and a map from y index to x
                        % index.
                        for c = 1:size(ym, 2)
                            o.y{end+1} = ym(:, c);
                            o.y_to_x_map(end+1) = length(o.x);
                        end
                        
                        km1_was_x = false;

                    end

                    % Maybe a line spec?
                    if k <= nargin && is_line_spec(varargin{k})

                        linespecs(length(o.y)+1 - (1:size(ym, 2))) =...
                            varargin(k); %#ok<AGROW>

                    % If it's neither numbers nor a line spec, stop.
                    else
                        break;
                    end

                end

            end

            % We've now parsed up to k.
            start = k;

            % Create cell arrays for the reduced data.
            x_r = cell(1, length(o.y));
            y_r = cell(1, length(o.y));

            % Get the axes width once.
            width = get_axes_width(o.h_axes);
            o.last_width = width;
            o.last_lims  = [-inf inf];

            % Reduce the data!
            for k = 1:length(o.y)
                [x_r{k}, y_r{k}] = reduce_to_width(...
                    o.x{o.y_to_x_map(k)}(:), ...
                    o.y{k}(:), ...
                    width, ...
                    [-inf inf]);
            end

            % If taking over a plot, just update it. Otherwise, plot it.
            if taking_over_existing_plot
                o.RefreshData();
                
            % Otherwise, we need to make a new plot.
            else
                
                % Make the plot arguments.
                plot_args = {};
                
                % Add the axes handle if the user supplied it.
                if axes_specified
                    plot_args{end+1} = o.h_axes;
                end
                
                % Add the lines.
                for k = 1:length(o.y)
                    plot_args{end+1} = x_r{k}; %#ok<AGROW>
                    plot_args{end+1} = y_r{k}; %#ok<AGROW>
                    if k <= length(linespecs) && ~isempty(linespecs{k})
                        plot_args{end+1} = linespecs{k}; %#ok<AGROW>
                    end
                end
                
                % Add any other arguments.
                plot_args = [plot_args, varargin(start:end)];
                
                % Plot it!
                try
                    
                    % plotyy
                    if isequal(plot_fcn, @plotyy)
                        
                        [o.h_axes, h1, h2] = plot_fcn(plot_args{:});
                        o.h_plot = [h1 h2];
                        
                    % stairs
                    elseif isequal(plot_fcn, @stairs) && length(o.y) > 1
                        
                        error(['Function ''stairs'' cannot plot ' ...
                               'multiple lines at once using ' ...
                               'LinePlotReducer. Try using ''hold on'' '...
                               'and calling LinePlotReducer once for ' ...
                               'each line.']);
                           
                    % All other lineseries functions.
                    else
                        o.h_plot = plot_fcn(plot_args{:});
                    end
                    
                catch err
                    fprintf(['LinePlotReducer had trouble managing the '...
                             '%s function. Perhaps the arguments are ' ...
                             'incorrect. The error is below.\n'], ...
                            func2str(plot_fcn));
                    rethrow(err);
                end
                
            end
            
            % Listen for changes to the x limits of the axes.
            if verLessThan('matlab', '8.4')
                size_cb = {'Position', 'PostSet'};
            else
                size_cb = {'SizeChanged'};
            end
            
            % Listen for changes on the axes.
            linkaxes(o.h_axes, 'x');
            for k = 1:length(o.h_axes)
                addlistener(o.h_axes(k), 'Units', 'PreSet', ...
                            @(~,~) o.UnitsPreSet);
                addlistener(o.h_axes(k), 'XLim',  'PostSet', ...
                            @(~,~) o.RefreshData);
                addlistener(o.h_axes(k), size_cb{:}, ...
                            @(~,~) o.RefreshData);
            end
            
            % Listen for changes on the figure itself.
            o.figure_listener = addlistener(o.h_figure, size_cb{:}, ...
                @(~,~) o.RefreshData);
            
            % Define DeletePlot as Nested Function, so the figure can be deleted 
            % even if LinePlotReducer.m is not on Matlab's search path anymore.
            function DeletePlot(o,k)
                o.deleted_plots(k) = true;
                if all(o.deleted_plots)
                    delete(o.figure_listener);
                end
            end
            
            % When all of our managed plots are deleted, we need to erase
            % ourselves, so we'll keep track when each is deleted.
            for k = 1:length(o.h_plot)
                set(o.h_plot(k), 'DeleteFcn', @(~,~) DeletePlot(o,k));
            end
            o.deleted_plots = false(1, length(o.h_plot));
            
            % Force the drawing to happen now.
            drawnow();

            % No longer busy.
            o.busy = false;
            
        end
                
    end
    
    methods
        
        % Redraw all of the data.
        function RefreshData(o)

            % When we set the axes units to 'pixels' and back, it will
            % trigger a callback each time for *both* 'Position' and
            % 'Units' (and in that order). Since we've set up callbacks to
            % trigger after the value is set, we can therefore set up a
            % PreSet callback for 'Units' to tell us to ignore a call.
            if o.calls_to_ignore > 0
                o.calls_to_ignore = o.calls_to_ignore - 1;
                return;
            end
            
            % We can do many things here that trigger additional callbacks,
            % so ignore them until we're done.
            if o.busy || ~all(ishandle(o.h_plot))
                return;
            end

            % We're busy now.
            o.busy = true;
            
            % Get the new limits. Sometimes there are multiple axes stacked
            % on top of each other. Just grab the first. This is really
            % just for plotyy.
            lims = get(o.h_axes(1), 'XLim');

            % Get axes width in pixels.
            width = get_axes_width(o.h_axes(1));
            
            % Just in case...
            if width < 0
                error(['The axes object reported a negative width. ' ...
                       'This is unexpected.']);
            end
            
            % Return if there's nothing to do.
            if width == o.last_width && all(lims == o.last_lims)
                o.busy = false;
                return;
            end
            
            % Record the last values for which we resized the data so we
            % can skip inconsequential updates later.
            o.last_width = width;
            o.last_lims  = lims;
            
            % For all data we manage...
            for k = 1:length(o.h_plot)
                
                % Reduce the data.
                [x_r, y_r] = reduce_to_width(o.x{o.y_to_x_map(k)}(:), ...
                                             o.y{k}(:), ...
                                             width, lims);
                
                % Update the plot.
                set(o.h_plot(k), 'XData', x_r, 'YData', y_r);
                
            end

            % We're no longer busy.
            o.busy = false;
            
        end

        % Setting the units (which we do to change them to 'pixels' and
        % back when getting the axes width) also triggers callbacks for
        % both 'Position' and 'Units' (in that order). We'll want to make
        % sure we ignore 1 request to refresh per call to 'Units'.
        function UnitsPreSet(o, ~, ~)
            
            % Note: In MATLAB 2014b, changing units won't trigger a change
            % in size, so we don't need to do this.
            if verLessThan('matlab', '8.4')
                o.calls_to_ignore = o.calls_to_ignore + 1;
            end
            
        end
                
    end
    
end
