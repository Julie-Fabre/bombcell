function [nPeaks, nTroughs, mainPeak_before_size, mainPeak_after_size, mainTrough_size, ...
    width_before, width_after, widthTrough, peakLocs, troughLocs, PKS, TRS, troughLoc] = getWaveformPeakProperties(thisWaveform, param)
if size(thisWaveform, 2) == 82
    % < KS4 waveforms, remove the zero start values that can create
    % articifical peaks/troughs
    thisWaveform(1:24) = NaN;
else
    thisWaveform(1:4) = NaN;

end

% Set minimum threshold for peak/trough detection
minProminence = param.minThreshDetectPeaksTroughs * max(abs(squeeze(thisWaveform)));

% Detect trough
[TRS, troughLocs, widthTrough, prominence] = findpeaks(squeeze(thisWaveform)*-1, 'MinPeakProminence', minProminence);

if length(widthTrough) > 1
    maxPeak = find(prominence == max(prominence));
    widthTrough = widthTrough(maxPeak);
end

% If no trough detected, find the minimum
if isempty(TRS)
    [TRS, troughLocs] = min(squeeze(thisWaveform));
    nTroughs = 1;
    widthTrough = NaN;
else
    nTroughs = numel(TRS);
end

% Get the main trough location
[mainTrough, mainTroughIdx] = max(TRS);
% If there are two troughs with the exact same value, choose the first
% one
if numel(mainTrough) > 1
    mainTrough = mainTrough(1);
    mainTroughIdx = mainTroughIdx(1);
end
troughLoc = troughLocs(mainTroughIdx);

% Find peaks before and after the trough
if troughLoc > 3 % need at least 3 samples
    [PKS_before, peakLocs_before, width_before, prominence] = findpeaks(squeeze(thisWaveform(1:troughLoc)), 'MinPeakProminence', minProminence);
    if length(width_before) > 1
        maxPeak = find(prominence == max(prominence));
        maxPeak = maxPeak(1);
        width_before = width_before(maxPeak);
    end
else
    PKS_before = '';
end
if size(thisWaveform, 2) - troughLoc > 3
    [PKS_after, peakLocs_after, width_after, prominence] = findpeaks(squeeze(thisWaveform(troughLoc:end)), 'MinPeakProminence', minProminence);
    peakLocs_after = peakLocs_after + troughLoc - 1;
    if length(width_after) > 1
        maxPeak = find(prominence == max(prominence));
        maxPeak = maxPeak(1);
        width_after = width_after(maxPeak);
    end
else
    PKS_after = '';
end

% If no peaks detected, find the maximum values
usedMaxBefore = 0;
if isempty(PKS_before)
    if troughLoc > 3
        [PKS_before, peakLocs_before, width_before, prominence] = findpeaks(squeeze(thisWaveform(1:troughLoc)), 'MinPeakProminence', 0.01*max(abs(squeeze(thisWaveform))));
    else
        PKS_before = '';
    end
    if length(PKS_before) > 1
        maxPeak = find(prominence == max(prominence));
        maxPeak = maxPeak(1);
        peakLocs_before = peakLocs_before(maxPeak);
        PKS_before = PKS_before(maxPeak);
        width_before = width_before(maxPeak);
    end

    if isempty(PKS_before)
        width_before = NaN;
        [PKS_before, peakLocs_before] = max(squeeze(thisWaveform(1:troughLoc)));
        if numel(PKS_before) > 0 % more than 1, just take first
            PKS_before = PKS_before(1);
            peakLocs_before = peakLocs_before(1);
        end

    end
    usedMaxBefore = 1;
end

usedMaxAfter = 0;
if isempty(PKS_after)
    if size(thisWaveform, 2) - troughLoc > 3
        [PKS_after, peakLocs_after, width_after, prominence] = findpeaks(squeeze(thisWaveform(troughLoc:end)), 'MinPeakProminence', 0.01*max(abs(squeeze(thisWaveform))));
        peakLocs_after = peakLocs_after + troughLoc - 1;
    else
        PKS_after = '';
    end
    if length(PKS_after) > 1
        maxPeak = find(prominence == max(prominence));
        maxPeak = maxPeak(1);
        peakLocs_after = peakLocs_after(maxPeak);
        PKS_after = PKS_after(maxPeak);
        width_after = width_after(maxPeak);
    end
    if isempty(PKS_after)
        width_after = NaN;
        [PKS_after, temp_loc] = max(squeeze(thisWaveform(troughLoc:end)));
        peakLocs_after = temp_loc + troughLoc - 1;
        if numel(PKS_after) > 0 % more than 1, just take first
            PKS_after = PKS_after(1);
            peakLocs_after = peakLocs_after(1);
        end

    end
    usedMaxAfter = 1;
end

% If neither a peak before or after is detected with findpeaks
if usedMaxAfter > 0 && usedMaxBefore > 0
    if PKS_before > PKS_after
        usedMaxBefore = 0;
    else
        usedMaxAfter = 0;
    end
end


% Get the main peaks before and after the trough
[mainPeak_before_size, mainPeakIdx_before] = max(PKS_before);
[mainPeak_after_size, mainPeakIdx_after] = max(PKS_after);
peakLoc_before = peakLocs_before(mainPeakIdx_before);
peakLoc_after = peakLocs_after(mainPeakIdx_after);

% Combine peak information
if usedMaxBefore == 1 && mainPeak_before_size(1) < minProminence * 0.5
    PKS = PKS_after;
    peakLocs = peakLocs_after;
elseif usedMaxAfter == 1 && mainPeak_after_size(1) < minProminence * 0.5
    PKS = PKS_before;
    peakLocs = peakLocs_before;
else
    PKS = [PKS_before, PKS_after];
    peakLocs = [peakLocs_before, peakLocs_after];
end

%
mainPeak_before_size = mainPeak_before_size(1);
mainPeak_after_size = mainPeak_after_size(1);
mainTrough_size = max(TRS);

% % Determine if the unit is somatic or non-somatic
% if (mainPeak_before_size(1) * param.firstPeakRatio > mainPeak_after_size(1) && width_before < param.minWidthFirstPeak && usedMaxBefore == 0 &&...
%         mainPeak_before_size(1) * param.minMainPeakToTroughRatio > max(TRS) && widthTrough < param.minWidthMainTrough) || ...
%         max(TRS) < max(PKS) && usedMaxBefore == 0)
%     isSomatic = 0; % non-somatic
% else
%     isSomatic = 1; % somatic
% end

% get number of peaks and troughs
nPeaks = numel(PKS); % drop any of the peaks not detected with findpeaks()
nTroughs = numel(TRS);
if length(width_before) > 1
    width_before = width_before(1);
end
if length(width_after) > 1
    width_after = width_after(1);
end
if length(widthTrough) > 1
    widthTrough = widthTrough(1);
end

end