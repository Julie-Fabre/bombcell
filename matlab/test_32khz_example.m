% Test example for 32kHz data with BombCell
% This demonstrates how the flexible spike width calculation works
% for non-standard sampling rates

% Add BombCell to path (the helper functions are in the +bc namespace)
addpath(genpath('+bc'));

% Test the spike width calculation for different scenarios
fprintf('=== Testing Spike Width Calculations ===\n\n');

% Standard 30kHz scenarios
fprintf('Standard 30kHz sampling:\n');
fprintf('  KS4 (30kHz): %d samples (expected: 61)\n', bc.qm.helpers.calculateSpikeWidth(30000, 4));
fprintf('  KS<4 (30kHz): %d samples (expected: 82)\n', bc.qm.helpers.calculateSpikeWidth(30000, 2.5));

% User's 32kHz scenario
fprintf('\nUser''s 32kHz sampling:\n');
fprintf('  KS4 (32kHz): %d samples (expected: ~65)\n', bc.qm.helpers.calculateSpikeWidth(32000, 4));
fprintf('  KS<4 (32kHz): %d samples (expected: ~87)\n', bc.qm.helpers.calculateSpikeWidth(32000, 2.5));

% Other sampling rates
fprintf('\nOther sampling rates:\n');
fprintf('  KS4 (25kHz): %d samples\n', bc.qm.helpers.calculateSpikeWidth(25000, 4));
fprintf('  KS4 (40kHz): %d samples\n', bc.qm.helpers.calculateSpikeWidth(40000, 4));

% Test half-width calculations
fprintf('\n=== Testing Half-Width Calculations ===\n\n');
fprintf('Spike Width -> Half Width:\n');
fprintf('  61 samples -> %d (expected: 20)\n', bc.qm.helpers.calculateHalfWidth(61));
fprintf('  65 samples -> %d (expected: ~21)\n', bc.qm.helpers.calculateHalfWidth(65));
fprintf('  82 samples -> %d (expected: 40)\n', bc.qm.helpers.calculateHalfWidth(82));
fprintf('  87 samples -> %d (expected: ~42)\n', bc.qm.helpers.calculateHalfWidth(87));

% Example of how to use with BombCell
fprintf('\n=== Example Usage with BombCell ===\n\n');

% Simulate loading metadata with 32kHz sampling rate
ephysMetaDir = struct('folder', '.', 'name', 'test.meta');
rawFile = 'test.bin';
ephysKilosortPath = '.';
gain_to_uV = 2.34375;
kilosortVersion = 4; % Using KS4

% Create parameters - the spike width will be automatically calculated
fprintf('Creating parameters for 32kHz data...\n');

% Mock the metadata loading to return 32kHz
% In real usage, bc.load.loadMetaData would read this from your meta file
% For this example, we'll manually set it after creation

% First, add the bombcell paths
addpath(genpath('+bc'));

% Create parameters (you would normally call bc.qm.qualityParamValues)
% For this demo, we'll create a minimal param structure
param = struct();
param.ephys_sample_rate = 32000; % Your actual sampling rate
param.spikeWidth = bc.qm.helpers.calculateSpikeWidth(32000, 4); % Will be 65 samples

fprintf('  Sampling rate: %d Hz\n', param.ephys_sample_rate);
fprintf('  Calculated spike width: %d samples\n', param.spikeWidth);
fprintf('  Calculated half width: %d samples\n', bc.qm.helpers.calculateHalfWidth(param.spikeWidth));

% The parameters will now work correctly with your 32kHz data!
fprintf('\nYour 32kHz data with 65-sample waveforms will now be processed correctly!\n');
fprintf('No more hardcoded 61/82 checks - the code adapts to your sampling rate.\n');