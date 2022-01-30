function unitQualityGUI(memMapData, ephysData, qMetrics, param, probeLocation)
%set up dynamic figure
h = figure;
set(h, 'KeyPressFcn', @KeyPressCb);

%initial conditions
iCluster = 1;
iCount = 1;
timeSecs = 1;
timeChunkStart = 5000;
timeChunk = timeSecs * ephysData.ephys_sample_rate;
timeChunkStop = timeSecs * ephysData.ephys_sample_rate;
uniqueTemps = unique(ephysData.spike_templates);
%plot initial conditions
plotUnitQuality(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, probeLocation, uniqueTemps);

%change on keypress
    function KeyPressCb(~, evnt)
        fprintf('key pressed: %s\n', evnt.Key);
        if strcmpi(evnt.Key, 'rightarrow')
            iCluster = iCluster + 1;
            clf;
            plotUnitQuality(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, probeLocation,uniqueTemps);
        elseif strcmpi(evnt.Key, 'leftarrow')
            iCluster = iCluster - 1;
            clf;
            plotUnitQuality(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, probeLocation,uniqueTemps);
        elseif strcmpi(evnt.Key, 'uparrow')
            timeChunkStart = timeChunkStop;
            timeChunkStop = timeChunkStop + timeChunk;
            clf;
            plotUnitQuality(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, probeLocation,uniqueTemps);
        elseif strcmpi(evnt.Key, 'downarrow')
            timeChunkStop = timeChunkStart;
            timeChunkStart = timeChunkStart - timeChunk;
            clf;
            plotUnitQuality(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, probeLocation,uniqueTemps);

        end
    end
end


function plotUnitQuality(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, probeLocation,uniqueTemps)
thisUnit = uniqueTemps(iCluster);
%% red box = unit quality assessed by metric is below param threshold

%% qq: 0. plot probe location in allenCCF
%     subplot(6,13,[1,14,27,40,53, 66],'YDir','reverse');
%         image(probe_ccf(curr_probe).trajectory_areas);
%     colormap(curr_axes,cmap);
%     caxis([1,size(cmap,1)])
%     set(curr_axes,'YTick',trajectory_area_centers,'YTickLabels',trajectory_area_labels);
%     set(curr_axes,'XTick',[]);
%     title([num2str(probe2ephys(curr_probe).day) num2str(probe2ephys(curr_probe).site)]);

%% 1. plot unit location on probe qq: replace by unit * spike rate
%code below from the lovely AP_cellraster
subplot(6, 13, [1, 14, 27, 40, 53, 66], 'YDir', 'reverse');
hold on;

norm_spike_n = mat2gray(log10(accumarray(ephysData.spike_templates, 1)+1));
unit_dots = plot(norm_spike_n, ephysData.channel_positions(qMetrics.maxChannels, :), '.k', 'MarkerSize', 20, 'ButtonDownFcn', @unit_click);
curr_unit_dots = plot(0, 0, '.r', 'MarkerSize', 20);
xlim([-0.1, 1]);
ylim([min(ephysData.channel_positions(:, 2)) - 50, max(ephysData.channel_positions(:, 2)) + 50]);
ylabel('Depth (\mum)')
xlabel('Normalized log rate')
% plot allenCCF map alongside
title('Location on probe')

%% 2. plot unit template waveform and detected peaks

subplot(6, 13, [2:7, 15:20])
cla;
maxChan = qMetrics.maxChannels(thisUnit);
maxXC = ephysData.channel_positions(maxChan, 1);
maxYC = ephysData.channel_positions(maxChan, 2);
chanDistances = ((ephysData.channel_positions(:, 1) - maxXC).^2 ...
    +(ephysData.channel_positions(:, 2) - maxYC).^2).^0.5;
chansToPlot = find(chanDistances < 100);
for iChanToPlot = 1:size(chansToPlot, 1)

    if maxChan == chansToPlot(iChanToPlot)
        plot((ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            -squeeze(ephysData.templates(thisUnit, :, chansToPlot(iChanToPlot)))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) ./ 15), 'Color', 'r');
        hold on;
        scatter((ephysData.waveform_t(qMetrics.peakLocs{iCluster}) ...
            +(ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            -squeeze(ephysData.templates(thisUnit, qMetrics.peakLocs{iCluster}, chansToPlot(iChanToPlot)))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) ./ 15), 'v', 'filled');

        scatter((ephysData.waveform_t(qMetrics.troughLocs{iCluster}) ...
            +(ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            -squeeze(ephysData.templates(thisUnit, qMetrics.troughLocs{iCluster}, chansToPlot(iChanToPlot)))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) ./ 15), 'v', 'filled');

    else
        plot((ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            -squeeze(ephysData.templates(thisUnit, :, chansToPlot(iChanToPlot)))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) ./ 15), 'Color', 'k');
    end
    hold on;
end
makeprettyNoText;
xlabel('Position+Time');
ylabel('Position');
set(gca, 'YDir', 'reverse')
if qMetrics.nPeaks(iCluster) > param.maxNPeaks || qMetrics.nTroughs(iCluster) > param.maxNTroughs
    title('\color{purple}Template waveform and detected peaks/troughs');
else
    title('\color{green}Template waveform and detected peaks/troughs');
end

%% 3. plot unit mean raw waveform (and individual traces)
subplot(6, 13, [8:13, 21:26], 'YDir', 'reverse')
cla;

for iChanToPlot = 1:size(chansToPlot, 1)
    if maxChan == chansToPlot(iChanToPlot)
        plot((ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            squeeze(qMetrics.rawWaveforms(iCluster).spkMapMean(chansToPlot(iChanToPlot), :))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) * 15), 'Color', 'r');
    else
        plot((ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            squeeze(qMetrics.rawWaveforms(iCluster).spkMapMean(chansToPlot(iChanToPlot), :))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) * 15), 'Color', 'k');
    end

    hold on;
end
makeprettyNoText;
if qMetrics.rawAmplitude(iCluster) < param.minAmplitude
    title('\color{purple}Mean raw waveform');
else
    title('\color{green}Mean raw waveform');
end
set(gca, 'YDir', 'reverse')
xlabel('Position+Time');
ylabel('Position');

%% 4. plot unit ACG

theseSpikeTimes = ephysData.spike_times_timeline(ephysData.spike_templates == thisUnit);
[ccg, ccg_t] = CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
    ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', 0.001, 'duration', 0.5, 'norm', 'rate'); %function


subplot(6, 13, 28:33)
area(ccg_t, ccg(:, 1, 1));
xlim([0, 0.250]); %Check FR
xlabel('time (s)');
hold on;
line([0.02, 0.02], [0, max(ccg(:, 1, 1))], 'Color', 'r')
line([-0.02, -0.02], [0, max(ccg(:, 1, 1))], 'Color', 'r')
[ccg, ccg_t] = CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
    ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', 0.01, 'duration', 10, 'norm', 'rate'); %function
asymptoteLine = nanmean(ccg(end-100:end));
line([-0.25, 0.25], [asymptoteLine, asymptoteLine], 'Color', 'r')

makeprettyNoText;
xlabel('time (s)');
ylabel('sp/s');
if qMetrics.Fp(iCluster) > param.maxRPVviolations
    title('\color{purple}ACG');
else
    title('\color{green}ACG');
end

%% 5. plot unit ISI (with refractory period and asymptote lines)

theseISIs = diff(theseSpikeTimes);
[c, b] = hist(theseISIs, 1000);

subplot(6, 13, [34:39])
area(b, c)


xlabel('time (s)');
hold on;
line([0.02, 0.02], [0, max(c)], 'Color', 'r')

makeprettyNoText;
xlabel('time (s)');
ylabel('sp/s');
if qMetrics.Fp(iCluster) > param.maxRPVviolations
    title('\color{purple}ISI');
else
    title('\color{green}ISI');
end
xlim([0, 1])

%% 6. plot isolation distance

%% 7. (optional) plot raster

%% 8. plot raw data
subplot(6, 13, [54:59, 60:65])
title('Raw unwhitened data')
hold on;
plotSubRaw(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop);

%% 9. plot template amplitudes and mean f.r. over recording (QQ: add experiment time epochs?)
subplot(6, 13, [67:71, 73:77])
ephysData.recordingDuration = (max(ephysData.spike_times_timeline) - min(ephysData.spike_times_timeline));

theseAmplis = ephysData.template_amplitudes(ephysData.spike_templates == thisUnit);
theseSpikeTimes = theseSpikeTimes - theseSpikeTimes(1); %so doesn't start negative. QQ do alignement before
yyaxis left
scatter(theseSpikeTimes, theseAmplis, 'blue', 'filled')
hold on;
currTimes = theseSpikeTimes(theseSpikeTimes > timeChunkStart/ephysData.recordingDuration & theseSpikeTimes < timeChunkStop/ephysData.recordingDuration);
currAmplis = theseAmplis(theseSpikeTimes > timeChunkStart/ephysData.recordingDuration & theseSpikeTimes < timeChunkStop/ephysData.recordingDuration);
scatter(currTimes, currAmplis, 'black', 'filled');
xlabel('Experiment time (s)');
ylabel('Template amplitude scaling');
axis tight
hold on;
ylim([0, round(max(theseAmplis))])
set(gca, 'YColor', 'b')
%           yyaxis right
%           binSize = 20;
%         timeBins = 0:binSize:ceil(ephysData.spike_times(end));
%         [n,x] = hist(theseSpikeTimes, timeBins);
%         n = n./binSize;
%
%         stairs(x,n, 'LineWidth', 2.0, 'Color', [1 0.5 0]);
%         ylim([0,2*round(max(n))])
%         set(gca,'YColor',[1 0.5 0])
%         ylabel('Firing rate (sp/sec)');

%% 10. plot ampli fit

subplot(6, 13, [78])
h = barh(qMetrics.ampliBinCenters{iCluster}, qMetrics.ampliBinCounts{iCluster}, 'red');
hold on;
h.FaceAlpha = 0.5;
plot(qMetrics.ampliFit{iCluster}, qMetrics.ampliBinCenters{iCluster}, 'red')

%% to add: MH, isodistance, CCG with n most similar templates
end


function plotSubRaw(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop)
%get the used channels
chanAmps = squeeze(max(ephysData.templates(iCluster, :, :))-min(ephysData.templates(iCluster, :, :)));
maxChan = find(chanAmps == max(chanAmps), 1);
maxXC = ephysData.channel_positions(maxChan, 1);
maxYC = ephysData.channel_positions(maxChan, 2);
chanDistances = ((ephysData.channel_positions(:, 1) - maxXC).^2 ...
    +(ephysData.channel_positions(:, 2) - maxYC).^2).^0.5;
chansToPlot = find(chanDistances < 170);

%get spike locations
pull_spikeT = -40:41;
thisC = ephysData.spike_templates(iCluster);
theseTimesCenter = ephysData.spike_times(ephysData.spike_templates == thisC);
theseTimesCenter = theseTimesCenter(theseTimesCenter > timeChunkStart/ephysData.ephys_sample_rate);
theseTimesCenter = theseTimesCenter(theseTimesCenter < timeChunkStop/ephysData.ephys_sample_rate);
if ~isempty(theseTimesCenter)
    %theseTimesCenter=theseTimesCenter(1);
    theseTimesFull = theseTimesCenter * ephysData.ephys_sample_rate + pull_spikeT;
    %theseTimesFull=unique(sort(theseTimesFull));
end
%plot
%   cCount=cumsum(repmat(abs(max(max(memMapData(chansToPlot, timeChunkStart:timeChunkStop)))),size(chansToPlot,1),1),1);
cCount = cumsum(repmat(1000, size(chansToPlot, 1), 1), 1);


t = timeChunkStart:timeChunkStop;
LinePlotReducer(@plot, t, double(memMapData(chansToPlot, timeChunkStart:timeChunkStop))+double(cCount), 'k');
if ~isempty(theseTimesCenter)
    hold on;
    for iTimes = 1:size(theseTimesCenter, 1)
        if ~any(mod(theseTimesFull(iTimes, :), 1))
            LinePlotReducer(@plot, theseTimesFull(iTimes, :), double(memMapData(chansToPlot, theseTimesFull(iTimes, :)))+double(cCount), 'r');

        end
    end
end
LinePlotExplorer(gcf);
%overlay the spikes

end