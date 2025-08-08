function halfWidth = calculateHalfWidth(spikeWidth)
% CALCULATEHALFWIDTH Calculate half-width for spike extraction
%
% This function calculates the appropriate half-width (samples before/after peak)
% based on the total spike width.
%
% Input:
%   spikeWidth - Total number of samples in the spike waveform
%
% Output:
%   halfWidth  - Number of samples before/after the peak
%
% The half-width is calculated to maintain the same proportions as the
% standard configurations:
%   - For 61 samples: halfWidth = 20 (20 before, 1 peak, 40 after)
%   - For 82 samples: halfWidth = 40 (40 before, 1 peak, 41 after)

% Calculate proportional half-width based on standard configurations
if spikeWidth <= 70
    % For shorter waveforms (like KS4), use ~33% before peak
    % This matches the 20/61 = 0.328 ratio
    halfWidth = round(spikeWidth * 0.328);
else
    % For longer waveforms, use ~49% before peak
    % This matches the 40/82 = 0.488 ratio
    halfWidth = round(spikeWidth * 0.488);
end

% Ensure halfWidth doesn't exceed what's possible
maxHalfWidth = floor((spikeWidth - 1) / 2);
halfWidth = min(halfWidth, maxHalfWidth);

% Ensure we have at least some samples before the peak
halfWidth = max(halfWidth, 10);

end