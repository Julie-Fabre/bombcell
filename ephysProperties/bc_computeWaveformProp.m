function [waveformDuration_peakTrough, halfWidth, peakTroughRatio, firstPeakTroughRatio,...
    nPeaks, nTroughs, isSomatic] = bc_computeWaveformProp(templateWaveforms, ...
    thisUnit, maxChannel, ephys_sample_rate, channelPositions, minThreshDetectPeaksTroughs)

% all waveform metrics based on template and not mean raw waveform for now 

% get waveform, waveform duration and peak/trough locations 
plotThis = false;
baselineThresh = NaN;
waveformBaselineWindow = NaN;
[nPeaks, nTroughs, isSomatic, peakLocs, troughLocs, waveformDuration_peakTrough, ...
    ~, ~, ~, thisWaveform] = bc_waveformShape(templateWaveforms, ...
    thisUnit, maxChannel, ephys_sample_rate, channelPositions, baselineThresh, ...
    waveformBaselineWindow, minThreshDetectPeaksTroughs, plotThis);

% time 
wvTime = 1e3 * ((0:size(thisWaveform, 2) - 1) / ephys_sample_rate);

% Compute Half-Width
troughAmplitude = thisWaveform(troughLocs(1));
halfAmplitude = troughAmplitude / 2;
aboveHalfIndices = find(thisWaveform >= halfAmplitude);
halfWidthStartIndex = aboveHalfIndices(find(aboveHalfIndices < peakLocs(end), 1, 'last'));
halfWidthEndIndex = aboveHalfIndices(find(aboveHalfIndices > peakLocs(end), 1));
halfWidth = wvTime(halfWidthEndIndex) - wvTime(halfWidthStartIndex);

% peak to trough ratio
peakAmplitude = thisWaveform(peakLocs(end));
peakTroughRatio = abs(peakAmplitude/troughAmplitude);

% 1rst peak to trough ratio
firstPeakAmplitude = max(thisWaveform(1:troughLocs(1)));
firstPeakTroughRatio = abs(firstPeakAmplitude/troughAmplitude);

% % Compute Rise Time
% riseTime = time(peakIndex) - time(halfWidthStartIndex);
% 
% % Compute Decay Time
% decayTime = time(halfWidthEndIndex) - time(peakIndex);
% 
% % Compute Rise Slope (Max Slope during the Rising Phase)
% riseSlope = max(diff(thisWaveform(halfWidthStartIndex:peakIndex)) ./ diff(time(halfWidthStartIndex:peakIndex)));
% 
% % Compute Decay Slope (Max Slope during the Falling Phase)
% decaySlope = min(diff(thisWaveform(peakIndex:halfWidthEndIndex)) ./ diff(time(peakIndex:halfWidthEndIndex)));
end