function [nPeaks, nTroughs, mainPeak_before_size, mainPeak_after_size, mainTrough_size,...
    mainPeak_before_width, mainPeak_after_width, mainTrough_width, peakLocs, troughLocs, waveformDuration_peakTrough, ...
    spatialDecayPoints, spatialDecaySlope, waveformBaseline, thisWaveform, spatialDecayPoints_loc, spatialDecayFit_1] = waveformShape(templateWaveforms, ...
    thisUnit, maxChannel, param, channelPositions, waveformBaselineWindow)
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
% param: structure with fields:
% - ephys_sample_rate: recording sampling rate, in samples per second (eg 30 000)
% - baselineThresh: 1 x 1 double vector, minimum baseline value over which
%   units are classified as noise, only needed if plotThis is set to true
% - minThreshDetectPeaksTroughs:  QQ describe
% - firstPeakRatio: 1 x 1 double. if units have an initial peak before the trough,
%   it must be at least firstPeakRatio times larger than the peak after the trough to qualify as a non-somatic unit.
% - normalizeSpDecay
% - computeSpatialDecay
% - minWidthFirstPeak
% - param.minMainPeakToTroughRatio
% - minWidthMainTrough
% - plotThis: boolean, whether to plot waveform and detected peaks or not
% channelPositions: [nChannels, 2] double matrix with each row giving the x
%   and y coordinates of that channel, only needed if plotThis is set to true
% waveformBaselineWindow: QQ describe
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
nChannels_to_eval = 1;
if any(isnan(thisWaveform)) % kilosort can sometimes return all NaNs in a waveform, we classify these units as noise
    nPeaks = NaN;
    nTroughs = NaN;
    mainPeak_before_size = nan(1,nChannels_to_eval);
    mainPeak_after_size = nan(1,nChannels_to_eval);
    mainTrough_size = nan(1,nChannels_to_eval);
    mainPeak_before_width = nan(1,nChannels_to_eval);
    mainPeak_after_width = nan(1,nChannels_to_eval);
    mainTrough_width = nan(1,nChannels_to_eval);
    peakLocs = NaN;
    troughLocs = NaN;
    waveformDuration_peakTrough = NaN;
    spatialDecayPoints = nan(1, 6);
    spatialDecaySlope = NaN;
    waveformBaseline = NaN;
    spatialDecayPoints_loc = nan(1, 6); 
    spatialDecayFit_1 = NaN;
else
    % get waveform peaks, troughs locations, sizes and widths for top 17
    % channels 
    theseChannels = maxChannel; % - 8 : maxChannel + 8;
    for iChannel = 1%:17 % evaluate peak and trough sizes and widths for top 17 channels
        if theseChannels(iChannel) > 0 && theseChannels(iChannel) <= size(templateWaveforms,3)
            if theseChannels(iChannel) == maxChannel
                thisWaveform = templateWaveforms(thisUnit, :, theseChannels(iChannel));
                [nPeaks(iChannel), nTroughs(iChannel), mainPeak_before_size(iChannel), mainPeak_after_size(iChannel), mainTrough_size(iChannel),...
                    mainPeak_before_width(iChannel), mainPeak_after_width(iChannel), mainTrough_width(iChannel), peakLocs, troughLocs, PKS, TRS, troughLoc] = ...
                    bc.qm.helpers.getWaveformPeakProperties(thisWaveform, param);
            else
                thisWaveform = templateWaveforms(thisUnit, :, theseChannels(iChannel));
                [nPeaks(iChannel), nTroughs(iChannel), mainPeak_before_size(iChannel), mainPeak_after_size(iChannel), mainTrough_size(iChannel),...
                    mainPeak_before_width(iChannel), mainPeak_after_width(iChannel), mainTrough_width(iChannel), ~, ~, ~, ~, ~] = ...
                    bc.qm.helpers.getWaveformPeakProperties(thisWaveform, param);
            end
        else
            nPeaks(iChannel) = NaN;
            nTroughs(iChannel) = NaN;
            mainPeak_before_size(iChannel) = NaN;
            mainPeak_after_size(iChannel) = NaN;
            mainTrough_size(iChannel) = NaN;
            mainPeak_before_width(iChannel) = NaN;
            mainPeak_after_width(iChannel) = NaN;
            mainTrough_width(iChannel) = NaN;

        end
    end


    % (get waveform peak to trough duration)
    % first assess which peak loc to use
    max_waveform_abs_value = max(abs(thisWaveform));
    if length(max_waveform_abs_value) > 1
        max_waveform_abs_value = max_waveform_abs_value(1);
    end
    max_waveform_location = find(abs(thisWaveform) == max_waveform_abs_value);
     if length(max_waveform_location) > 1
       max_waveform_location = max_waveform_location(1);
    end
    max_waveform_value = thisWaveform(max_waveform_location);
    if max_waveform_value(end) > 0
        peakLoc_forDuration = peakLocs(PKS == max(PKS));
        if length(peakLoc_forDuration) > 1
            peakLoc_forDuration = peakLoc_forDuration(1);
        end
        [~, troughLoc_forDuration] = min(thisWaveform(peakLoc_forDuration:end)); % to calculate waveform duration
        troughLoc_forDuration = troughLoc_forDuration + peakLoc_forDuration - 1;
    else
        troughLoc_forDuration = troughLocs(TRS == max(TRS));
        if length(troughLoc_forDuration) > 1
            troughLoc_forDuration = troughLoc_forDuration(1);
        end
        [~, peakLoc_forDuration] = max(thisWaveform(troughLoc_forDuration:end)); % to calculate waveform duration
        peakLoc_forDuration = peakLoc_forDuration + troughLoc_forDuration - 1;
    end

    % waveform duration in microseconds
    if ~isempty(troughLoc) && ~isempty(peakLoc_forDuration)
        waveformDuration_peakTrough = 1e6 * abs(troughLoc_forDuration-peakLoc_forDuration) / param.ephys_sample_rate; %in us
    else
        waveformDuration_peakTrough = NaN;
    end

    % (get waveform spatial decay accross channels)
    linearFit = param.spDecayLinFit;
    [spatialDecaySlope, spatialDecayFit, spatialDecayPoints, spatialDecayPoints_loc, estimatedUnitXY] = ...
        bc.qm.helpers.getSpatialDecay(templateWaveforms, thisUnit, maxChannel, channelPositions, linearFit, param.normalizeSpDecay, param.computeSpatialDecay);
    if linearFit
        spatialDecayFit_1 = spatialDecayFit(2);
    else
        spatialDecayFit_1 = spatialDecayFit(1);
    end
    % (get waveform baseline fraction)
    if ~isnan(waveformBaselineWindow(1))
        waveformBaseline = max(abs(thisWaveform(waveformBaselineWindow(1): ...
            waveformBaselineWindow(2)))) / max(abs(thisWaveform));
    else
        waveformBaseline = NaN;
    end

    % (plot waveform)
    if param.plotDetails

        colorMtx = bc.viz.colors(8);


        figure();

        subplot(4, 2, 7:8)
        pt1 = scatter(spatialDecayPoints_loc, spatialDecayPoints, [], colorMtx(1, :, :), 'filled');
        hold on;
        if linearFit
            lf = plot(spatialDecayPoints_loc, spatialDecayPoints_loc*spatialDecayFit(1)+spatialDecayFit(2), '-', 'Color', colorMtx(2, :, :));
            legend(lf, {['linear fit, slope = ', num2str(spatialDecaySlope)]}, 'TextColor', [0.7, 0.7, 0.7], 'Color', 'none')
        else
            % Generate points for the exponential fit curve
            fitX = linspace(min(spatialDecayPoints_loc), max(spatialDecayPoints_loc), 100);
            spatialDecayFitFun = @(x) spatialDecayFit(1) * exp(-spatialDecaySlope * x);
            fitY = spatialDecayFitFun(fitX);
            
            % Plot the exponential fit curve
            ef = plot(fitX, fitY, '-', 'Color', colorMtx(2, :, :));
            
            % Add legend with decay rate
            legend(ef, {sprintf('Exp. fit, decay rate = %.4f', spatialDecaySlope)}, ...
                'TextColor', [0.7, 0.7, 0.7], 'Color', 'none')
        end
        ylabel('trough size (a.u.)')
        xlabel('distance from peak channel (um)')
        

        subplot(4, 2, 1:6)
        set(gca, 'YDir', 'reverse');
        hold on;
        set(gca, 'XColor', 'k', 'YColor', 'k')
        maxChan = maxChannel;
        maxXC = channelPositions(maxChan, 1);
        maxYC = channelPositions(maxChan, 2);
        chanDistances = ((channelPositions(:, 1) - maxXC).^2 ...
            +(channelPositions(:, 2) - maxYC).^2).^0.5;
        chansToPlot = find(chanDistances < 70);
        wvTime = 1e3 * ((0:size(thisWaveform, 2) - 1) / param.ephys_sample_rate);
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
                    [param.maxWvBaselineFraction * -max(abs(thisWaveform))' + ...
                    (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100 * max_value), param.maxWvBaselineFraction * -max(abs(thisWaveform))' + ...
                    (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100 * max_value)], 'Color', colorMtx(4, :, :));

                line([(wvTime(waveformBaselineWindow(1)) + (channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
                    (wvTime(waveformBaselineWindow(2)) + (channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10)], ...
                    [-param.maxWvBaselineFraction * -max(abs(thisWaveform))' + ...
                    (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100 * max_value), -param.maxWvBaselineFraction * -max(abs(thisWaveform))' + ...
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

        legend([p1, peak, trough, dur, l1], {[''], ...
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
