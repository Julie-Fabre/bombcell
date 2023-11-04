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

% First subplot: sine curve
subplot(2, 2, 1);
plot(x, y1, 'r'); 
hold on; % Keep the current plot and add new plots to it
plot(x, 0.5*ones(size(x)), 'k--'); % Dashed line at y = 0.5
title('Sine Curve');
xlabel('x');
ylabel('sin(x)');
legend('sin(x)', 'y = 0.5');

% Second subplot: cosine curve
subplot(2, 2, 3);
plot(x, y2, 'b'); 
hold on;
% Sample points
x_points = [pi/4, pi/2, 3*pi/4];
y_points = cos(x_points);
plot(x_points, y_points, 'ro'); % red circles
title('Cosine Curve');
xlabel('x');
ylabel('cos(x)');

% ac
time = linspace(0, 24, 100); % Time from 0 to 24 hours
activity = cumsum(randn(15, 100)); % Random walk for activity
subplot(2, 2, 2);
uglyColors = [1 0 1; 0 1 0; 0 0 1; 1 1 0; 1 0.5 0.2];
colormap(uglyColors);
imagesc(time, [1], activity);
ylabel('Activity');
xlabel('Time (ms)');
colorbar;

% Activity plot 
time = linspace(0, 24, 100); % Time from 0 to 24 hours
activity = cumsum(randn(15, 100)); % Random walk for activity
subplot(2, 2, 4);
uglyColors = [1 0 1; 0 1 0; 0 0 1; 1 1 0; 1 0.5 0.2];
colormap(uglyColors);
% Plot the activity data
imagesc(time, [1], activity);
ylabel('Activity');
xlabel('Time (ms)');
colorbar;


