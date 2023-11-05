%% uglyPlot

% Define x values
x = linspace(0, 2*pi, 1000);

% Define y values for various functions
y1 = sin(x);
y2 = cos(x);
y3 = tan(x);
y4 = log(x + 1); % Adding 1 to avoid negative values and log(0)
y_cot = cot(x); % Cotangent values

% Create a new figure
figure(2); clf;


%% line plots 
% First subplot: sine curve
subplot(2, 3, 1);
plot(x, y1, 'r'); 
hold on; % Keep the current plot and add new plots to it
plot(x, 0.5*ones(size(x)), 'k--'); % Dashed line at y = 0.5
title('Sine Curve');
xlabel('x');
ylabel('sin(x)');
legend('sin(x)', 'y = 0.5');

% Second subplot: cosine curve
subplot(2, 3, 4);
plot(x, y2, 'b'); 
hold on;
% Sample points
x_points = [pi/4, pi/2, 3*pi/4];
y_points = cos(x_points);
plot(x_points, y_points, 'ro'); % red circles
title('Cosine Curve');
xlabel('x');
ylabel('cos(x)');

% scatter plots

numDots = 20;
subplot(2, 3, 2);
x = rand(numDots, 1); % Random numbers between 0 and 1 for the x-axis
y = rand(numDots, 1); % Random numbers between 0 and 1 for the y-axis
scatter(x, y);
xlabel('X-axis');
ylabel('Y-axis');
title('Scatter Plot of Random Dots');



numDots = 25;
subplot(2, 3, 5);
x = rand(numDots, 1); % Random numbers between 0 and 1 for the x-axis
y = rand(numDots, 1); % Random numbers between 0 and 1 for the y-axis
scatter(x, y);
xlabel('X-axis');
ylabel('Y-axis');
title('Scatter Plot of Random Dots');

%% images 
time = linspace(0, 24, 100); % Time from 0 to 24 hours
activity = cumsum(randn(15, 100)); % Random walk for activity
subplot(2, 3, 3);
uglyColors = [1 0 1; 0 1 0; 0 0 1; 1 1 0; 1 0.5 0.2];
colormap(uglyColors);
imagesc(time, [], activity);
ylabel('Activity');
xlabel('Time (ms)');
c = colorbar;
c.Title.String = 'Zscore';

% Activity plot 
time = linspace(0, 24, 100); % Time from 0 to 24 hours
activity = cumsum(randn(15, 100)).*2; % Random walk for activity
subplot(2, 3, 6);
uglyColors = [1 0 1; 0 1 0; 0 0 1; 1 1 0; 1 0.5 0.2];
colormap(uglyColors);
% Plot the activity data
imagesc(time, [], activity);
ylabel('Activity');
xlabel('Time (ms)');
c2 = colorbar;
c2.Title.String = 'Zscore';

prettify_plot;
