function prettify_colorblind_simulator()

% Get current figure's content
f = gcf;
all_axes = findall(f, 'Type', 'axes');

% Define transformation matrices for color blindness
% These are rough approximations

protanopia = [0.567, 0.433, 0; 0.558, 0.442, 0; 0, 0.242, 0.758];
deuteranopia = [0.625, 0.375, 0; 0.7, 0.3, 0; 0, 0.3, 0.7];
tritanopia = [0.95, 0.05, 0; 0, 0.433, 0.567; 0, 0.475, 0.525];
protanomaly = (eye(3) + protanopia) / 2;
deuteranomaly = (eye(3) + deuteranopia) / 2;
tritanomaly = (eye(3) + tritanopia) / 2;
achromatomaly = repmat(mean(eye(3), 2)', 3, 1);
achromatopsia = repmat(mean(achromatomaly, 2)', 3, 1);

transforms = {protanopia, deuteranopia, tritanopia, protanomaly, deuteranomaly, tritanomaly, achromatomaly, achromatopsia};
typeNames = {'Protanopia', 'Deuteranopia', 'Tritanopia', 'Protanomaly', 'Deuteranomaly', 'Tritanomaly', 'Achromatomaly', 'Achromatopsia'};

% Create a new figure for colorblind visualizations
figure;

for j = 1:length(all_axes)
    ax = all_axes(j);
    img = getframe(ax);
    imgData = img.cdata;
    imgData = im2double(imgData); % Convert to double for matrix multiplication
    
    for i = 1:length(transforms)
        % Simulate the colorblind vision
        transformed = reshape(imgData, [], 3) * transforms{i}';
        transformed = reshape(transformed, size(imgData));
        transformed = min(max(transformed, 0), 1); % Clamp values between 0 and 1
        
        % Display in a subplot
        subplot(length(all_axes), length(transforms), (j-1)*length(transforms) + i);
        imshow(transformed);
        title(typeNames{i});
    end
end
end