function [nPeaks, nTroughs, isSomatic, peakLocs, troughLocs, waveformDuration_peakTrough, ...
    spatialDecayPoints, spatialDecaySlope, waveformBaseline, thisWaveform] = waveformShape(templateWaveforms, ...
    thisUnit, maxChannel, ephys_sample_rate, channelPositions, baselineThresh, ...
    waveformBaselineWindow, minThreshDetectPeaksTroughs, firstPeakRatio, normalizeSpDecay, plotThis)
% JF
% Get the number of troughs and peaks for each waveform,
% determine whether waveform is likely axonal/dendritic (biggest peak before
% biggest trough - cf: Deligkaris, K., Bullmann, T. & Frey, U.
%   Extracellularly recorded somatic and neuritic signal shapes and classification
%   algorithms for high-density microelectrode array electrophysiology. Front. Neurosci. 10, 421 (2016),
% get waveform duration,
% get waveform baseline maximum absolute value (high
% values are usually indicative of noise units),
% evaluate waveform spatial decay.
% ------
% Inputs
% ------
% templateWaveforms: nTemplates × nTimePoints × nChannels single matrix of
%   template waveforms for each template and channel
% thisUnit: 1 x 1 double vector, current unit number
% maxChannel:  1 x 1 double vector, channel with maximum template waveform current unit number
% ephys_sample_rate: recording sampling rate, in samples per second (eg 30 000)
% channelPositions: [nChannels, 2] double matrix with each row giving the x
%   and y coordinates of that channel, only needed if plotThis is set to true
% baselineThresh: 1 x 1 double vector, minimum baseline value over which
%   units are classified as noise, only needed if plotThis is set to true
% waveformBaselineWindow: QQ describe
% minThreshDetectPeaksTroughs:  QQ describe
% firstPeakRatio: 1 x 1 double. if units have an initial peak before the trough,
%   it must be at least firstPeakRatio times larger than the peak after the trough to qualify as a non-somatic unit.
% plotThis: boolean, whether to plot waveform and detected peaks or not
% ------
% Outputs
% ------
% nPeaks: number of detected peaks
% nTroughs: number of detected troughs
% isSomatic: boolean, is largest detected peak after the largest detected
%   trough (indicative of a somatic spike).)
% peakLocs: location of detected peaks, used in the GUI
% troughLocs: location of detected troughs, used in the GUI
% waveformDuration_minMax: estimated waveform duration, from detected peak
%   to trough, in us
% waveformDuration_peakTrough: estimated waveform duration, from detected peak
%   to trough, in us
% spatialDecayPoints QQ describe
% spatialDecaySlope QQ describe
% waveformBaselineFlatness
% thisWaveform: this unit's template waveforms at and around the peak
%   channel, used in the GUI
%
% Centroid-based 'real' unit location calculation: Enny van Beest


% (find peaks and troughs using MATLAB's built-in function)
thisWaveform = templateWaveforms(thisUnit, :, maxChannel);

if any(isnan(thisWaveform)) % kilosort can sometimes return all NaNs in a waveform, we classify these units as noise
    nPeaks = NaN;
    nTroughs = NaN;
    isSomatic = NaN;
    peakLocs = NaN;
    troughLocs = NaN;
    waveformDuration_peakTrough = NaN;
    spatialDecayPoints = nan(1, 6);
    spatialDecaySlope = NaN;
    waveformBaseline = NaN;
else
    % Set minimum threshold for peak/trough detection
    minProminence = minThreshDetectPeaksTroughs * max(abs(squeeze(thisWaveform)));

    % Detect trough
    [TRS, troughLocs] = findpeaks(squeeze(thisWaveform)*-1, 'MinPeakProminence', minProminence);

    % If no trough detected, find the minimum
    if isempty(TRS)
        [TRS, troughLocs] = min(squeeze(thisWaveform));
        nTroughs = 1;
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
    [PKS_before, peakLocs_before] = findpeaks(squeeze(thisWaveform(1:troughLoc)), 'MinPeakProminence', minProminence);
    [PKS_after, peakLocs_after] = findpeaks(squeeze(thisWaveform(troughLoc:end)), 'MinPeakProminence', minProminence);
    peakLocs_after = peakLocs_after + troughLoc - 1;

    % If no peaks detected, find the maximum values
    if isempty(PKS_before)
        [PKS_before, peakLocs_before] = max(squeeze(thisWaveform(1:troughLoc)));
    end
    if isempty(PKS_after)
        [PKS_after, temp_loc] = max(squeeze(thisWaveform(troughLoc:end)));
        peakLocs_after = temp_loc + troughLoc - 1;
    end

    % Get the main peaks before and after the trough
    [mainPeak_before, mainPeakIdx_before] = max(PKS_before);
    [mainPeak_after, mainPeakIdx_after] = max(PKS_after);
    peakLoc_before = peakLocs_before(mainPeakIdx_before);
    peakLoc_after = peakLocs_after(mainPeakIdx_after);

    % Combine peak information
    PKS = [mainPeak_before, mainPeak_after];
    peakLocs = [peakLoc_before, peakLoc_after];

    % get number of peaks and troughs
    nPeaks = numel(PKS);
    nTroughs = numel(TRS);

    % Determine if the unit is somatic or non-somatic
    if (mainPeak_before * firstPeakRatio > mainPeak_after) || max(TRS) < max(PKS)
        isSomatic = 0; % non-somatic
    else
        isSomatic = 1; % somatic
    end

    % (get waveform peak to trough duration)
    % first assess which peak loc to use
    max_waveform_abs_value = max(abs(thisWaveform));
    max_waveform_location = abs(thisWaveform) == max_waveform_abs_value;
    max_waveform_value = thisWaveform(max_waveform_location);
    if max_waveform_value(end) > 0
        peakLoc_forDuration = peakLocs(PKS == max(PKS));
        [~, troughLoc_forDuration] = min(thisWaveform(peakLoc_forDuration:end)); % to calculate waveform duration
        troughLoc_forDuration = troughLoc_forDuration + peakLoc_forDuration - 1;
    else
        troughLoc_forDuration = troughLocs(TRS == max(TRS));
        [~, peakLoc_forDuration] = max(thisWaveform(troughLoc_forDuration:end)); % to calculate waveform duration
        peakLoc_forDuration = peakLoc_forDuration + troughLoc_forDuration - 1;
    end

    % waveform duration in microseconds
    if ~isempty(troughLoc) && ~isempty(peakLoc_forDuration)
        waveformDuration_peakTrough = 1e6 * abs(troughLoc_forDuration-peakLoc_forDuration) / ephys_sample_rate; %in us
    else
        waveformDuration_peakTrough = NaN;
    end

    % (get waveform spatial decay accross channels)
    linearFit = 1;
    [spatialDecaySlope, spatialDecayFit, spatialDecayPoints, spatialDecayPoints_loc, estimatedUnitXY] = ...
        bc.qm.helpers.getSpatialDecay(templateWaveforms, thisUnit, maxChannel, channelPositions, linearFit, normalizeSpDecay);


    % (get waveform baseline fraction)
    if ~isnan(waveformBaselineWindow(1))
        waveformBaseline = max(abs(thisWaveform(waveformBaselineWindow(1): ...
            waveformBaselineWindow(2)))) / max(abs(thisWaveform));
    else
        waveformBaseline = NaN;
    end

    % (plot waveform)
    if plotThis

        colorMtx = colors(8);


        figure();

        subplot(4, 2, 7:8)
        pt1 = scatter(spatialDecayPoints_loc, spatialDecayPoints, [], colorMtx(1, :, :), 'filled');
        hold on;
        lf = plot(spatialDecayPoints_loc, spatialDecayPoints_loc*spatialDecayFit(1)+spatialDecayFit(2), '-', 'Color', colorMtx(2, :, :));

        ylabel('trough size (a.u.)')
        xlabel('distance from peak channel (um)')
        legend(lf, {['linear fit, slope = ', num2str(spatialDecaySlope)]}, 'TextColor', [0.7, 0.7, 0.7], 'Color', 'none')

        subplot(4, 2, 1:6)
        set(gca, 'YDir', 'reverse');
        hold on;
        set(gca, 'XColor', 'w', 'YColor', 'w')
        maxChan = maxChannel;
        maxXC = channelPositions(maxChan, 1);
        maxYC = channelPositions(maxChan, 2);
        chanDistances = ((channelPositions(:, 1) - maxXC).^2 ...
            +(channelPositions(:, 2) - maxYC).^2).^0.5;
        chansToPlot = find(chanDistances < 70);
        wvTime = 1e3 * ((0:size(thisWaveform, 2) - 1) / ephys_sample_rate);
        max_value = max(max(abs(squeeze(templateWaveforms(thisUnit, :, chansToPlot))))) * 5;
        for iChanToPlot = 1:min(20, size(chansToPlot, 1))

            if maxChan == chansToPlot(iChanToPlot) % max channel
                % plot waveform line
                p1 = plot((wvTime + (channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
                    -squeeze(templateWaveforms(thisUnit, :, chansToPlot(iChanToPlot)))'+ ...
                    (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100 * max_value), 'Color', colorMtx(1, :, :));
                hold on;
                % plot peak(s)
                peak = scatter((wvTime(peakLocs) ...
                    +(channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
                    -squeeze(templateWaveforms(thisUnit, peakLocs, chansToPlot(iChanToPlot)))'+ ...
                    (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100 * max_value), [], colorMtx(2, :, :), 'v', 'filled');
                % plot trough(s)
                trough = scatter((wvTime(troughLocs) ...
                    +(channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
                    -squeeze(templateWaveforms(thisUnit, troughLocs, chansToPlot(iChanToPlot)))'+ ...
                    (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100 * max_value), [], colorMtx(3, :, :), 'v', 'filled');
                % plot baseline lines
                l1 = line([(wvTime(waveformBaselineWindow(1)) + (channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
                    (wvTime(waveformBaselineWindow(2)) + (channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10)], ...
                    [baselineThresh * -max(abs(thisWaveform))' + ...
                    (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100 * max_value), baselineThresh * -max(abs(thisWaveform))' + ...
                    (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100 * max_value)], 'Color', colorMtx(4, :, :));

                line([(wvTime(waveformBaselineWindow(1)) + (channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
                    (wvTime(waveformBaselineWindow(2)) + (channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10)], ...
                    [-baselineThresh * -max(abs(thisWaveform))' + ...
                    (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100 * max_value), -baselineThresh * -max(abs(thisWaveform))' + ...
                    (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100 * max_value)], 'Color', colorMtx(4, :, :));

                % plot waveform duration
                pT_locs = [troughLoc_forDuration, peakLoc_forDuration];
                dur = plot([(wvTime(min(pT_locs)) ...
                    +(channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
                    (wvTime(max(pT_locs)) ...
                    +(channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10)], ...
                    [(squeeze(templateWaveforms(thisUnit, 1, chansToPlot(iChanToPlot))))' + ...
                    (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100 * max_value), (squeeze(templateWaveforms(thisUnit, 1, chansToPlot(iChanToPlot))))' + ...
                    (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100 * max_value)], '->', 'Color', colorMtx(6, :, :));


            else
                % plot waveform
                plot((wvTime + (channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
                    -squeeze(templateWaveforms(thisUnit, :, chansToPlot(iChanToPlot)))'+ ...
                    (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100 * max_value), 'Color', [0.7, 0.7, 0.7]);
                hold on;
            end
        end

        celLoc = scatter((estimatedUnitXY(1) + -11)/10+wvTime(42), estimatedUnitXY(2)/100*max_value, 50, 'x', 'MarkerEdgeColor', colorMtx(5, :, :), ...
            'MarkerFaceColor', colorMtx(5, :, :));

        legend([p1, peak, trough, dur, l1], {['is somatic =', num2str(isSomatic), newline], ...
            [num2str(nPeaks), ' peak(s)'], [num2str(nTroughs), ...
            ' trough(s)'], 'duration', 'baseline line'}, ...
            'TextColor', [0.7, 0.7, 0.7], 'Color', 'none')
        box off;
        set(gca, 'YTick', []);
        set(gca, 'XTick', []);
        set(gca, 'Visible', 'off')
        if exist('prettify_plot', 'file')
            prettify_plot('FigureColor', 'w')
        else
            warning('https://github.com/Julie-Fabre/prettify-matlab repo missing - download it and add it to your matlab path to make plots pretty')
            makepretty('none')
        end


    end
end
end
