function [nPeaks, nTroughs, somatic, peakLocs, troughLocs, waveformDuration, spatialDecayPoints, spatialDecaySlope, ...
    waveformBaseline, thisWaveform] = bc_waveformShape(templateWaveforms, thisUnit, maxChannel, ephys_sample_rate, channelPositions, baselineThresh, plotThis)
% JF, Get the number of troughs and peaks for each waveform, and determine
% whether waveform is likely axonal (biggest peak before biggest trough)
% ------
% Inputs
% ------
% thisWaveform: nTemplates × nTimePoints × nChannels single matrix of
%   template waveforms for each template and channel
% ephys_sample_rate: recording sampling rate (eg 30 000)
% plotThis: boolean, whether to plot waveform and detected peaks or not 
% ------
% Outputs
% ------
% nPeaks: number of detected peaks
% nTroughs: number of detected troughs
% somatic: boolean, is largest detected peak after the largest detected
%   trough (indicative of a somatic spike, cf: Deligkaris, K., Bullmann, T. & Frey, U. 
%   Extracellularly recorded somatic and neuritic signal shapes and classification 
%   algorithms for high-density microelectrode array electrophysiology. Front. Neurosci. 10, 421 (2016).)
% 
thisWaveform = templateWaveforms(thisUnit, :, maxChannel);
minProminence = 0.2 * max(abs(squeeze(thisWaveform))); % minimum threshold to detcet peaks/troughs

[PKS, peakLocs, ~] = findpeaks(squeeze(thisWaveform), 'MinPeakProminence', minProminence); % get peaks

[TRS, troughLocs] = findpeaks(squeeze(thisWaveform)*-1, 'MinPeakProminence', minProminence); % get troughs

if isempty(TRS) % if there is no detected trough, just take minimum value as trough
    TRS = min(squeeze(thisWaveform));
    nTroughs = numel(TRS);
    LOCST_all = find(squeeze(thisWaveform) == TRS);
        if numel(TRS) > 1 % if more than one trough, take the first (usually the correct one) %QQ should change to better:
            % by looking for location where the data is most tightly distributed
            TRS = TRS(1);
        end
    
    troughLocs = find(squeeze(thisWaveform) == TRS);
else
    nTroughs = numel(TRS);
end
if isempty(PKS) % if there is no detected peak, just take maximum value as peak
    PKS = max(squeeze(thisWaveform));
    nPeaks = numel(PKS);
    LOCS_all = find(squeeze(thisWaveform) == PKS);
        if numel(PKS) > 1 % if more than one peak, take the first (usually the correct one) %QQ should change to better:
            % by looking for location where the data is most tightly distributed
            PKS = PKS(1);
            %peakWidths = peakWidths(1);
        end
    peakLocs = find(squeeze(thisWaveform) == PKS);
    
else
    nPeaks = numel(PKS);
end


peakLoc = peakLocs(PKS == max(PKS)); %QQ should change to better:
            % by looking for location where the data is most tightly distributed
if numel(peakLoc) > 1
    peakLoc = peakLoc(1);

end
troughLoc = troughLocs(TRS == max(TRS)); %QQ should change to better:
            % by looking for location where the data is most tightly distributed
if numel(troughLoc) > 1
    troughLoc = troughLoc(1);
end

if peakLoc > troughLoc
    somatic = 1;
else
    somatic = 0;
end
if ~isempty(troughLoc) && ~isempty(peakLoc)
waveformDuration = abs(troughLoc-peakLoc)*0.0333*1000; %in ms 
else
    waveformDuration = abs(min(thisWaveform)-max(thisWaveform))*0.0333*1000; %in ms
    waveformDuration = waveformDuration(1);
end
if maxChannel > 10
    spatialDecayPoints = max(abs(squeeze(templateWaveforms(thisUnit, :, maxChannel:-2:maxChannel-10))));
else
    spatialDecayPoints = max(abs(squeeze(templateWaveforms(thisUnit, :, maxChannel:2:maxChannel+10))));
end
spatialDecaySlope = polyfit(spatialDecayPoints, 1:6,1); % fit first order polynomial to data. first output is slope of polynomial, second is a constant
spatialDecaySlope = spatialDecaySlope(1);

waveformBaseline = max(abs(thisWaveform(20:30)))/max(abs(thisWaveform));

if plotThis
    figure();
    set(gca, 'YDir', 'reverse');
    hold on;
    set(gca, 'XColor', 'w', 'YColor', 'w')
    maxChan = maxChannel;
    maxXC = channelPositions(maxChan, 1);
    maxYC = channelPositions(maxChan, 2);
    chanDistances = ((channelPositions(:, 1) - maxXC).^2 ...
        +(channelPositions(:, 2) - maxYC).^2).^0.5;
    chansToPlot = find(chanDistances < 70);
    vals =[];
    wvTime = 1e3*((0:size(thisWaveform, 2) - 1) / ephys_sample_rate);
    for iChanToPlot = 1:min(20, size(chansToPlot, 1))
        vals(iChanToPlot) = max(abs(squeeze(templateWaveforms(thisUnit, :, chansToPlot(iChanToPlot)))));
        if maxChan == chansToPlot(iChanToPlot)
            p1 = plot( (wvTime + (channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
                 -squeeze(templateWaveforms(thisUnit, :, chansToPlot(iChanToPlot)))'+ ...
                (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100), 'Color', 'b');
            hold on;
            peak = scatter(  (wvTime(peakLocs) ...
                +(channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
                -squeeze(templateWaveforms(thisUnit, peakLocs, chansToPlot(iChanToPlot)))'+ ...
                (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100), [],rgb('Orange'), 'v', 'filled');

            trough = scatter((wvTime(troughLocs) ...
                +(channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
                 -squeeze(templateWaveforms(thisUnit, troughLocs, chansToPlot(iChanToPlot)))'+ ...
                (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100), [],rgb('Gold'), 'v', 'filled');
            l1 = line([(wvTime(1) +(channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10) ...
                (wvTime(30) +(channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10)],...
                [baselineThresh*-max(squeeze(templateWaveforms(thisUnit, :, chansToPlot(iChanToPlot))))'+ ...
                (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100), baselineThresh*-max(squeeze(templateWaveforms(thisUnit, :, chansToPlot(iChanToPlot))))'+ ...
                (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100)], 'Color', 'red');
            line([(wvTime(1) +(channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10) ...
                (wvTime(30) +(channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10)],...
                [-baselineThresh*-max(squeeze(templateWaveforms(thisUnit, :, chansToPlot(iChanToPlot))))'+ ...
                (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100), -baselineThresh*-max(squeeze(templateWaveforms(thisUnit, :, chansToPlot(iChanToPlot))))'+ ...
                (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100)], 'Color', 'red');
        else
            p2 = plot( (wvTime + (channelPositions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
                -squeeze(templateWaveforms(thisUnit, :, chansToPlot(iChanToPlot)))'+ ...
                (channelPositions(chansToPlot(iChanToPlot), 2) ./ 100), 'Color', 'k');
            hold on;
        end
    end

    tempWvTitleText = ['\\fontsize{9}Template waveform: {\\color[rgb]{%s}# detected peaks/troughs, ', newline,...
                '\\color[rgb]{%s}is somatic \\color[rgb]{%s} cell-like duration \\color[rgb]{%s}spatial decay}'];


    legend([p1, peak, trough, p2, l1 ], {['is somatic =', num2str(somatic), newline], ...
        [num2str(nPeaks), ' peak(s)'], [num2str(nTroughs), ...
        ' trough(s)'], ['spatial decay slope =' , num2str(spatialDecaySlope)], ['cell-like baseline line']}, 'Color', 'none')
    makepretty;
   
end
end

