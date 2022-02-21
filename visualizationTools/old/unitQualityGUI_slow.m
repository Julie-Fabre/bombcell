function unitQualityGUI(memMapData, ephysData, qMetrics, param, probeLocation, goodUnits)
%set up dynamic figure
h = figure;
set(h, 'KeyPressFcn', @KeyPressCb);

%initial conditions
iCluster = 1;
iCount = 1;
timeSecs = 0.1;
timeChunkStart = 1000;
timeChunk = timeSecs * ephysData.ephys_sample_rate;
timeChunkStop = timeSecs * ephysData.ephys_sample_rate;
uniqueTemps = unique(ephysData.spike_templates);
%plot initial conditions
iChunk = 1;
plotUnitQuality(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, probeLocation, ...
    uniqueTemps, goodUnits, iChunk);

%change on keypress
    function KeyPressCb(~, evnt)
        fprintf('key pressed: %s\n', evnt.Key);
        if strcmpi(evnt.Key, 'rightarrow')
            iCluster = iCluster + 1;
            clf;
            tic
            plotUnitQuality(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, ...
                probeLocation,uniqueTemps, goodUnits, iChunk);
            toc
        elseif strcmpi(evnt.Key, 'leftarrow')
            iCluster = iCluster - 1;
            clf;
            plotUnitQuality(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, ...
                probeLocation,uniqueTemps, goodUnits, iChunk);
        elseif strcmpi(evnt.Key, 'uparrow')
            iChunk = iChunk + 1;
            clf;
            plotUnitQuality(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, ...
                probeLocation,uniqueTemps, goodUnits, iChunk);
        elseif strcmpi(evnt.Key, 'downarrow')
            iChunk = iChunk - 1;
            clf;
            plotUnitQuality(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, ...
                probeLocation,uniqueTemps, goodUnits, iChunk);

        end
    end
end


function plotUnitQuality(memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, ...
    probeLocation,uniqueTemps, goodUnits, iChunk)
thisUnit = uniqueTemps(iCluster);

clf;

set(gcf, 'color', 'white')
if goodUnits(iCluster) ==1
    sgtitle(['\color[rgb]{0 .5 0}Unit ', num2str(iCluster), ', good']);
else
    sgtitle(['\color{red}Unit ', num2str(iCluster), ', bad']);
end
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
unitCmap = zeros(length(goodUnits),3);
unitCmap(goodUnits ==1 ,:,:) = repmat([0, 0.5, 0], length(find(goodUnits==1)),1);
unitCmap(goodUnits ==0 ,:,:) = repmat([1, 0 , 0], length(find(goodUnits==0)),1);
norm_spike_n = mat2gray(log10(accumarray(ephysData.spike_templates, 1)+1));
unit_dots = scatter(norm_spike_n(uniqueTemps), ephysData.channel_positions(qMetrics.maxChannels(uniqueTemps), 2),5, unitCmap, ...
     'filled', 'ButtonDownFcn', @unit_click);
curr_unit_dots = scatter(norm_spike_n(thisUnit), ephysData.channel_positions(qMetrics.maxChannels(thisUnit), 2),100, unitCmap(iCluster,:,:), ...
     'filled','MarkerEdgeColor',[0 0 0],'LineWidth',4);
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
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) ./ 100), 'Color', 'b');
        hold on;
        scatter((ephysData.waveform_t(qMetrics.peakLocs{iCluster}) ...
            +(ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            -squeeze(ephysData.templates(thisUnit, qMetrics.peakLocs{iCluster}, chansToPlot(iChanToPlot)))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) ./ 100), 'v', 'filled');

        scatter((ephysData.waveform_t(qMetrics.troughLocs{iCluster}) ...
            +(ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            -squeeze(ephysData.templates(thisUnit, qMetrics.troughLocs{iCluster}, chansToPlot(iChanToPlot)))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) ./ 100), 'v', 'filled');

    else
        plot((ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            -squeeze(ephysData.templates(thisUnit, :, chansToPlot(iChanToPlot)))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) ./ 100), 'Color', 'k');
    end
    hold on;
end
makeprettyNoText;
xlabel('Position+Time');
ylabel('Position');
set(gca, 'YDir', 'reverse')
if qMetrics.nPeaks(iCluster) > param.maxNPeaks || qMetrics.nTroughs(iCluster) > param.maxNTroughs
    if qMetrics.axonal(iCluster) == 1
        title(['\fontsize{9}Template waveform: {\color[rgb]{1 0 0}# detected peaks/troughs, '...
            '\color[rgb]{1 0 0}is soma \color{red}}'])
    else
       title(['\fontsize{9}Template waveform: {\color[rgb]{1 0 0}# detected peaks/troughs, '...
            '\color[rgb]{0 .5 0}is soma \color{red}}'])
    end
else
    if qMetrics.axonal(iCluster) == 1
        title(['\fontsize{9}Template waveform: {\color[rgb]{0 .5 0}# detected peaks/troughs, '...
            '\color[rgb]{1 0 0}is soma \color{red}}'])
    else
        title(['\fontsize{9}Template waveform: {\color[rgb]{0 .5 0}# detected peaks/troughs, '...
            '\color[rgb]{0 .5 0}is soma \color{red}}'])
    end
    
end
TextLocation([num2str(qMetrics.nPeaks(iCluster)), ' peak(s), ', num2str(qMetrics.nTroughs(iCluster)), ' trough(s)', ...
    ' is soma =', num2str(1 - qMetrics.axonal(iCluster))],'Location', 'North')

%% 3. plot unit mean raw waveform (and individual traces)

subplot(6, 13, [8:13, 21:26])
cla;
qMetrics.rawWaveforms(1).peakChan
for iChanToPlot = 1:size(chansToPlot, 1)
    if maxChan == chansToPlot(iChanToPlot)
        plot((ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            -squeeze(qMetrics.rawWaveforms(iCluster).spkMapMean(chansToPlot(iChanToPlot), :))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) * 10), 'Color', 'b');
    else
        plot((ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            -squeeze(qMetrics.rawWaveforms(iCluster).spkMapMean(chansToPlot(iChanToPlot), :))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) * 10), 'Color', 'k');
    end

    hold on;
end

makeprettyNoText;
TextLocation(['Amplitude =', num2str(qMetrics.rawAmplitude(iCluster)), 'uV'],'Location', 'North')
if qMetrics.rawAmplitude(iCluster) < param.minAmplitude
    title('\color[rgb]{1 0 0}Mean raw waveform');
else
    title('\color[rgb]{0 .5 0}Mean raw waveform');
end
set(gca, 'YDir', 'reverse')
xlabel('Position+Time');
ylabel('Position');


%% 4. plot unit ACG

theseSpikeTimes = ephysData.spike_times_timeline(ephysData.spike_templates == thisUnit);

%  [Fp, r, overestimate] = bc_fractionRPviolations(numel(theseSpikeTimes),theseSpikeTimes, param.tauR, ...
%      param.tauC, max(theseSpikeTimes)-min(theseSpikeTimes), 1);


[ccg, ccg_t] = CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
    ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', 0.001, 'duration', 0.5, 'norm', 'rate'); %function


subplot(6, 13, 28:31)
area(ccg_t, ccg(:, 1, 1));
hold on;
line([0.002, 0.002], [0, max(ccg(:, 1, 1))], 'Color', 'r')
line([-0.002, -0.002], [0, max(ccg(:, 1, 1))], 'Color', 'r')
[ccg, ~] = CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
    ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', 0.01, 'duration', 10, 'norm', 'rate'); %function
asymptoteLine = nanmean(ccg(end-100:end));
line([-0.1, 0.1], [asymptoteLine, asymptoteLine], 'Color', 'r')

makeprettyNoText

xlim([0, 0.05]); %Check FR
xlabel('time (s)');
ylabel('sp/s');
if qMetrics.Fp(iCluster) > param.maxRPVviolations
    title('\color[rgb]{1 0 0}ACG');
else
    title('\color[rgb]{0 .5 0}ACG');
end


%% 5. plot unit ISI (with refractory period and asymptote lines)

theseTimes = theseSpikeTimes- theseSpikeTimes(1);
theseISI = diff(theseSpikeTimes);
theseisiclean = theseISI(theseISI >= param.tauC); % removed duplicate spikes 
[isiProba, edgesISI] = histcounts(theseisiclean*1000, [0:0.5:50]);
subplot(6, 13, [32:35])
bar(edgesISI(1:end-1)+mean(diff(edgesISI)), isiProba); %Check FR
xlabel('Interspike interval (ms)')
ylabel('# of spikes')
    
hold on;
yLim = ylim;
line([2, 2], [0, yLim(2)], 'Color', 'r')

makeprettyNoText;
if qMetrics.Fp(iCluster) > param.maxRPVviolations
    title('\color[rgb]{1 0 0}ISI');
else
    title('\color[rgb]{0 .5 0}ISI');
end
TextLocation([num2str(qMetrics.Fp(iCluster)), ' % r.p.v.'],'Location', 'North')

%% 6. plot isolation distance

subplot(6, 13, 36:39)

scatter(qMetrics.Xplot{iCluster}(:, 1), qMetrics.Xplot{iCluster}(:, 2), 10, '.b') % Scatter plot with points of size 10
hold on
scatter(qMetrics.Yplot{iCluster}(:, 1), qMetrics.Yplot{iCluster}(:, 2), 10, qMetrics.d2_mahal{iCluster}, 'o', 'filled')
colormap(brewermap([], '*YlOrRd'))
hb = colorbar;
ylabel(hb, 'Mahalanobis Distance')
legend('this cluster', 'other clusters', 'Location', 'best');

%% 7. (optional) plot raster

%% 10. plot ampli fit

subplot(6, 13, [78])
h = barh(qMetrics.ampliBinCenters{iCluster}, qMetrics.ampliBinCounts{iCluster}, 'red');
hold on;
h.FaceAlpha = 0.5;
plot(qMetrics.ampliFit{iCluster}, qMetrics.ampliBinCenters{iCluster}, 'blue', 'LineWidth', 4)
if qMetrics.percSpikesMissing(iCluster) > param.maxPercSpikesMissing
    title('\color[rgb]{1 0 0}% spikes missing');
else
    title('\color[rgb]{0 .5 0}% spikes missing');
end
TextLocation([num2str(qMetrics.percSpikesMissing(iCluster)), ' % spikes missing'],'Location', 'SouthWest')

%% 8. plot raw data

subplot(6, 13, [41:52, 55:59, 60:65])
title('Raw unwhitened data')
hold on;
plotSubRaw(memMapData, ephysData, iCluster, iCount, uniqueTemps, timeChunkStart, timeChunkStop, iChunk);

%% 9. plot template amplitudes and mean f.r. over recording (QQ: add experiment time epochs?)


        
subplot(6, 13, [67:70, 73:76])
ephysData.recordingDuration = (max(ephysData.spike_times_timeline) - min(ephysData.spike_times_timeline));

theseAmplis = ephysData.template_amplitudes(ephysData.spike_templates == thisUnit);
theseSpikeTimes = theseSpikeTimes - theseSpikeTimes(1); %so doesn't start negative. QQ do alignement before
timeChunks = [min(theseSpikeTimes), max(theseSpikeTimes)];

    
yyaxis left
scatter(theseSpikeTimes, theseAmplis, 'black', 'filled')
hold on;
currTimes = theseSpikeTimes(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1  & theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
currAmplis = theseAmplis(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1  & theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
scatter(currTimes, currAmplis, 'blue', 'filled');
xlabel('Experiment time (s)');
ylabel('Template amplitude scaling');
axis tight
hold on;
ylim([0, round(max(theseAmplis))])
set(gca, 'YColor', 'k')
yyaxis right
binSize = 20;
timeBins = 0:binSize:ceil(ephysData.spike_times(end)/ephysData.ephys_sample_rate);
[n,x] = hist(theseSpikeTimes, timeBins);
n = n./binSize;

stairs(x,n, 'LineWidth', 2.0, 'Color', [1 0.5 0]);
ylim([0,2*round(max(n))])
set(gca,'YColor',[1 0.5 0])
ylabel('Firing rate (sp/sec)');

if qMetrics.nSpikes(iCluster) > param.minNumSpikes
    title('\color[rgb]{0 .5 0}Spikes');
else
    title('\color[rgb]{1 0 0}Spikes');
end
TextLocation(['# spikes = ' num2str(qMetrics.nSpikes(iCluster))],'Location', 'SouthWest')




%% to add: CCG with n most similar templates
end


function plotSubRaw(memMapData, ephysData, iCluster, iCount, uniqueTemps, timeChunkStart, timeChunkStop, iChunk)
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
thisC = uniqueTemps(iCluster);
theseTimesCenter = ephysData.spike_times(ephysData.spike_templates == thisC)./ephysData.ephys_sample_rate;
if iChunk < 0
    disp('Don''t do that')
    iChunk = 1;
end
firstSpike=theseTimesCenter(iChunk)-0.1;%first spike occurance 
theseTimesCenter = theseTimesCenter(theseTimesCenter >= firstSpike);
theseTimesCenter = theseTimesCenter(theseTimesCenter <= firstSpike + 0.2);
if ~isempty(theseTimesCenter)
    %theseTimesCenter=theseTimesCenter(1);
    theseTimesFull = theseTimesCenter * ephysData.ephys_sample_rate + pull_spikeT;
    %theseTimesFull=unique(sort(theseTimesFull));
end
%plot
%   cCount=cumsum(repmat(abs(max(max(memMapData(chansToPlot, timeChunkStart:timeChunkStop)))),size(chansToPlot,1),1),1);
cCount = cumsum(repmat(1000, size(chansToPlot, 1), 1), 1);


t = (firstSpike*ephysData.ephys_sample_rate):((firstSpike+0.2)*ephysData.ephys_sample_rate);
LinePlotReducer(@plot, t, double(memMapData(chansToPlot, firstSpike*ephysData.ephys_sample_rate...
    :(firstSpike +0.2)*ephysData.ephys_sample_rate))+double(cCount), 'k');
if ~isempty(theseTimesCenter)
    hold on;
    for iTimes = 1:size(theseTimesCenter, 1)
        if ~any(mod(theseTimesFull(iTimes, :), 1))
            LinePlotReducer(@plot, theseTimesFull(iTimes, :) , double(memMapData(chansToPlot, theseTimesFull(iTimes, :)))+double(cCount), 'b');

        end
    end
end
LinePlotExplorer(gcf);
%overlay the spikes

end