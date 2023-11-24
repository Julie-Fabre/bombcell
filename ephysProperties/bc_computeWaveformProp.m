function bc_computeWaveformProp(templateWaveforms, ...
    thisUnit, maxChannel, ephys_sample_rate, channelPositions, baselineThresh, ...
    waveformBaselineWindow, minThreshDetectPeaksTroughs)

% get waveform, waveform duration and peak/trough locations 
plotThis = false;
[nPeaks, nTroughs, isSomatic, peakLocs, troughLocs, waveformDuration_peakTrough, ...
    ~, ~, ~, thisWaveform] = bc_waveformShape(templateWaveforms, ...
    thisUnit, maxChannel, ephys_sample_rate, channelPositions, baselineThresh, ...
    waveformBaselineWindow, minThreshDetectPeaksTroughs, plotThis);


% get full width half max

% rise time

% asymetry 
end