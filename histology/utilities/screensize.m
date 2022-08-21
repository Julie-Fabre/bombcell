function ss_out = screensize(screen_number)
%screensize: return screen coordinates of multiple monitors.
    % Version: 1.0, 26 June 2008 Author: Douglas M. Schwarz
    % Version: 1.1, 21 July 2017 Author: Wolfie 
    persistent myss
    if isempty(myss)
        % Get Screen Devices array.
        sd = java.awt.GraphicsEnvironment.getLocalGraphicsEnvironment.getScreenDevices;
        % Initialize screensize array.
        num_screens = length(sd);
        myss = zeros(num_screens,4);
        % Loop over all Screen Devices.
        for ii = 1:num_screens
            bounds = sd(ii).getDefaultConfiguration.getBounds;
            myss(ii,:) = [bounds.x, bounds.y, bounds.width, bounds.height];
        end
    end
    num_screens = size(myss,1);
    if nargin == 0
        screen_number = 1:num_screens;
    end
    screen_index = min(screen_number,num_screens);
    ss_out = myss(screen_index,:);
end
