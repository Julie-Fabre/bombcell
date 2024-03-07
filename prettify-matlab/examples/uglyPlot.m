% Create a 3x4 subplot
figure;

% Parameters for simulation
time = linspace(0, 2, 2000);  % Time vector
stimulus_onset = 0.5;  % Time of stimulus onset
stimulus_duration = 0.2;  % Duration of the stimulus
response_delay = 0.3;  % Response delay after stimulus onset
response_duration = 0.25;  % Duration of the response
noise_amplitude = 0.5;  % Amplitude of noise

% Simulate stimulus
stimulus = zeros(size(time));
stimulus((time >= stimulus_onset) & (time < stimulus_onset + stimulus_duration)) = 1;

% Simulate neurons
neuron1_response = 12 * exp(-(time - (stimulus_onset + response_delay)) / response_duration) .* (time >= (stimulus_onset + response_delay)) +...
     noise_amplitude * randn(size(time));
neuron2_response = 0.5 * exp(-(time - (stimulus_onset + response_delay)) / response_duration) .* (time >= (stimulus_onset + response_delay)) +...
    + noise_amplitude * randn(size(time));
neuron3_response = 22 * -exp(-(time - (stimulus_onset + response_delay)) / response_duration) .* (time >= (stimulus_onset + response_delay))+...
    noise_amplitude * randn(size(time));

% Plot the simulated responses
subplot(3, 3, 1);
plot(time, neuron1_response);
xlim([0 1])
xlabel('Time');
ylabel('Response');

subplot(3, 3, 4);
plot(time, neuron2_response);
xlabel('Time');
ylabel('Response');

subplot(3, 3, 7);
plot(time, neuron3_response);
xlabel('Time');
ylabel('Response');

% simulate PSTH 
% Define a function to generate neuron responses
generate_neuron_response = @(amplitude) amplitude * exp(-(time - (stimulus_onset + response_delay)) / response_duration) .* ...
    (time >= (stimulus_onset + response_delay)) + amplitude * randn(size(time));

noise_amplitude = 0.1; 
% Use arrayfun to generate responses for all neurons
neurons1_amplitudes = [-2:0.5:12];
neurons_1 = arrayfun(generate_neuron_response, neurons1_amplitudes, 'UniformOutput', false);
neurons_1_matrix = cell2mat(neurons_1');

neurons2_amplitudes = [-0.5:0.01:0.5];
neurons_2 = arrayfun(generate_neuron_response, neurons2_amplitudes, 'UniformOutput', false);
neurons_2_matrix = cell2mat(neurons_2');

neurons3_amplitudes = [-22:1:1];
neurons_3 = arrayfun(generate_neuron_response, neurons3_amplitudes, 'UniformOutput', false);
neurons_3_matrix = cell2mat(neurons_3');


% plot PSTHs
subplot(3, 3, 2);
imagesc(time, [], neurons_1_matrix)
colormap("jet")
cb = colorbar;
cb.Title.String = 'Activity';

subplot(3, 3, 5);
imagesc(time, [], neurons_2_matrix)
colormap("jet")
colorbar;

subplot(3, 3, 8);
imagesc(time, [], neurons_3_matrix)
colormap("jet")
colorbar;

% plot some lines 
average_psth1 = smoothdata(mean(neurons_1_matrix, 1),'gaussian', [0, 100]);
average_psth2 =  smoothdata(mean(neurons_2_matrix, 1), 'gaussian',[0, 100]);
average_psth3 =  smoothdata(mean(neurons_3_matrix, 1), 'gaussian',[0, 100]);


subplot(3, 3, [3,6,9]); hold on;
plot(time, average_psth2);
plot(time, average_psth3);
plot(time, average_psth1);


legend('Type 1', 'Type 2', 'Type 3')






% Adjust the layout of the subplots
sgtitle('Demo');
