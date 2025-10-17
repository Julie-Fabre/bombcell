function spikeWidth = calculateSpikeWidth(samplingRate, kilosortVersion)
% CALCULATESPIKEWIDTH Calculate spike width based on sampling rate and Kilosort version
%
% This function provides flexible spike width calculation to support
% different sampling rates (not just 30kHz).
%
% Inputs:
%   samplingRate    - Sampling rate in Hz (e.g., 30000)
%   kilosortVersion - Kilosort version number (e.g., 2.5, 3, 4)
%
% Output:
%   spikeWidth      - Number of samples for the spike waveform
%
% The spike width is calculated to maintain consistent time windows:
%   - Kilosort 4: ~2.03ms window (61 samples at 30kHz)
%   - Kilosort <4: ~2.73ms window (82 samples at 30kHz)

if nargin < 2
    % Default to Kilosort 2.5 behavior if version not specified
    kilosortVersion = 2.5;
end

% Define time windows in milliseconds
if kilosortVersion >= 4
    % Kilosort 4 uses a shorter window
    timeWindow_ms = 2.033; % 61 samples at 30kHz = 2.033ms
else
    % Earlier versions use a longer window
    timeWindow_ms = 2.733; % 82 samples at 30kHz = 2.733ms
end

% Calculate spike width based on sampling rate
% Round to nearest odd number to maintain symmetry around peak
spikeWidth = round(timeWindow_ms * samplingRate / 1000);

% Ensure spike width is odd for symmetry
if mod(spikeWidth, 2) == 0
    spikeWidth = spikeWidth + 1;
end

% Display info if sampling rate is non-standard
standardRates = [30000];
if ~ismember(samplingRate, standardRates)
    fprintf('Note: Using non-standard sampling rate of %d Hz\n', samplingRate);
    fprintf('Calculated spike width: %d samples (%.3f ms)\n', spikeWidth, timeWindow_ms);
end

end