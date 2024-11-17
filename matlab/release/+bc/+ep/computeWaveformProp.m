function [waveformDuration_peakTrough, halfWidth, peakTroughRatio, firstPeakTroughRatio,...
    nPeaks, nTroughs, isSomatic] = computeWaveformProp(templateWaveforms, ...
    thisUnit, maxChannel, param, channelPositions)

% all waveform metrics based on template and not mean raw waveform for now 

% get waveform, waveform duration and peak/trough locations 
param.plotThis = false;
param.baselineThresh = NaN;
waveformBaselineWindow = NaN;
param.computeSpatialDecay = false;
[nPeaks, nTroughs, mainPeak_before_size, mainPeak_after_size, mainTrough_size,...
    mainPeak_before_width, mainPeak_after_width, mainTrough_width, peakLocs, troughLocs, waveformDuration_peakTrough, ...
    spatialDecayPoints, spatialDecaySlope, waveformBaseline, thisWaveform] = bc.qm.waveformShape(templateWaveforms, ...
    thisUnit, maxChannel, param, channelPositions, ...
    waveformBaselineWindow);

if ~isnan(nPeaks) && ~isnan(nTroughs)

% time 
wvTime = 1e3 * ((0:size(thisWaveform, 2) - 1) / param.ephys_sample_rate);

% Compute Half-Width
troughAmplitude = thisWaveform(troughLocs(1));

halfAmplitude = troughAmplitude / 2;
aboveHalfIndices = find(thisWaveform >= halfAmplitude);
halfWidthStartIndex = aboveHalfIndices(find(aboveHalfIndices < peakLocs(end), 1, 'last'));
halfWidthEndIndex = aboveHalfIndices(find(aboveHalfIndices > peakLocs(end), 1));
halfWidth = wvTime(halfWidthEndIndex) - wvTime(halfWidthStartIndex);
if isempty(halfWidth)
    halfWidth = NaN;
end
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
else
    halfWidth = NaN;
    peakTroughRatio = NaN;
    firstPeakTroughRatio = NaN;
end
end