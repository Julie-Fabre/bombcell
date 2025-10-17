function [waveformDuration_peakTrough, peakLoc_forDuration, troughLoc_forDuration] = ...
    computeWaveformDuration_peakTrough(thisWaveform, peakLocs, troughLocs, PKS, TRS, ephys_sample_rate)
% COMPUTEWAVEFORMDURATION_PEAKTROUGH Compute waveform duration from peak to trough
% This function implements the sophisticated algorithm from waveformShape.m
% to ensure consistent duration calculations across qualityMetrics and ephysProperties
%
% Inputs:
%   thisWaveform - 1xN waveform vector
%   peakLocs - locations of detected peaks
%   troughLocs - locations of detected troughs  
%   PKS - peak amplitudes
%   TRS - trough amplitudes (absolute values)
%   ephys_sample_rate - sampling rate in Hz
%
% Outputs:
%   waveformDuration_peakTrough - duration in microseconds
%   peakLoc_forDuration - peak location used for duration
%   troughLoc_forDuration - trough location used for duration

% Check for empty inputs
if isempty(peakLocs) || isempty(troughLocs) || all(isnan(thisWaveform))
    waveformDuration_peakTrough = NaN;
    peakLoc_forDuration = NaN;
    troughLoc_forDuration = NaN;
    return;
end

% First assess which peak/trough locations to use
max_waveform_abs_value = max(abs(thisWaveform));
if length(max_waveform_abs_value) > 1
    max_waveform_abs_value = max_waveform_abs_value(1);
end
max_waveform_location = find(abs(thisWaveform) == max_waveform_abs_value);
if length(max_waveform_location) > 1
    max_waveform_location = max_waveform_location(1);
end
max_waveform_value = thisWaveform(max_waveform_location);

% Determine if waveform is peak-first or trough-first
if max_waveform_value(end) > 0
    % Peak-first waveform: find highest peak, then search for trough after it
    peakLoc_forDuration = peakLocs(PKS == max(PKS));
    if length(peakLoc_forDuration) > 1
        peakLoc_forDuration = peakLoc_forDuration(1);
    end
    [~, troughLoc_forDuration] = min(thisWaveform(peakLoc_forDuration:end));
    troughLoc_forDuration = troughLoc_forDuration + peakLoc_forDuration - 1;
else
    % Trough-first waveform: find highest trough, then search for peak after it
    troughLoc_forDuration = troughLocs(TRS == max(TRS));
    if length(troughLoc_forDuration) > 1
        troughLoc_forDuration = troughLoc_forDuration(1);
    end
    [~, peakLoc_forDuration] = max(thisWaveform(troughLoc_forDuration:end));
    peakLoc_forDuration = peakLoc_forDuration + troughLoc_forDuration - 1;
end

% Calculate waveform duration in microseconds
if ~isempty(troughLoc_forDuration) && ~isempty(peakLoc_forDuration)
    waveformDuration_peakTrough = 1e6 * abs(troughLoc_forDuration - peakLoc_forDuration) / ephys_sample_rate;
else
    waveformDuration_peakTrough = NaN;
    peakLoc_forDuration = NaN;
    troughLoc_forDuration = NaN;
end

end