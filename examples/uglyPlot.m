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
figure;

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
subplot(2, 2, 2);
plot(x, y2, 'b'); 
hold on;
% Sample points
x_points = [pi/4, pi/2, 3*pi/4];
y_points = cos(x_points);
plot(x_points, y_points, 'ro'); % red circles
title('Cosine Curve');
xlabel('x');
ylabel('cos(x)');
legend('cos(x)', 'Sample Points');

% Third subplot: tangent and cotangent curves
subplot(2, 2, 3);
plot(x, y3, 'g'); 
hold on;
plot(x, y_cot, 'c'); % Cyan color for cotangent
ylim([-10, 10]);
title('Tangent and Cotangent Curves');
xlabel('x');
ylabel('Value');
legend('tan(x)', 'cot(x)');

% Fourth subplot: logarithm curve
subplot(2, 2, 4);
plot(x, y4, 'm');
hold on;
% Point at the maximum value of the logarithm curve
[max_y4, idx_max] = max(y4);
plot(x(idx_max), max_y4, 'bo'); % blue circle
title('Logarithm Curve');
xlabel('x');
ylabel('log(x+1)');
legend('log(x+1)', 'Max Point');

