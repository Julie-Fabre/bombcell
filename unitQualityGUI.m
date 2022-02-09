function unitQualityGUI(memMapData, ephysData, qMetrics, param, probeLocation, goodUnits)
%% set up dynamic figure
unitQualityGuiHandle = figure('color','w');
set(unitQualityGuiHandle, 'KeyPressFcn', @KeyPressCb);

%% initial conditions
iCluster = 1;
iCount = 1;
timeSecs = 0.1;
timeChunkStart = 1000;
timeChunkStop = timeSecs * ephysData.ephys_sample_rate;
uniqueTemps = unique(ephysData.spike_templates);

%% plot initial conditions
iChunk = 1;
initializePlot(ephysData, qMetrics, goodUnits)
updateUnit(unitQualityGuiHandle,memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, ...
                probeLocation,uniqueTemps, goodUnits, iChunk);
%% change on keypress
    function KeyPressCb(~, evnt)
        fprintf('key pressed: %s\n', evnt.Key);
        if strcmpi(evnt.Key, 'rightarrow')
            iCluster = iCluster + 1;
            clf;
            updateUnit(unitQualityGuiHandle,memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, ...
                probeLocation,uniqueTemps, goodUnits, iChunk);
        elseif strcmpi(evnt.Key, 'leftarrow')
            iCluster = iCluster - 1;
            clf;
            updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, ...
                probeLocation,uniqueTemps, goodUnits, iChunk);
        elseif strcmpi(evnt.Key, 'uparrow')
            iChunk = iChunk + 1;
            clf;
            updateRawSnippet(unitQualityGuiHandle, memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, ...
                probeLocation,uniqueTemps, goodUnits, iChunk);
        elseif strcmpi(evnt.Key, 'downarrow')
            iChunk = iChunk - 1;
            clf;
            updateRawSnippet(unitQualityGuiHandle, memMapData, ephysData, iCluster, iCount, timeChunkStart, timeChunkStop, qMetrics, param, ...
                probeLocation,uniqueTemps, goodUnits, iChunk);
        end
    end
end

function initializePlot(ephysData, qMetrics, goodUnits)
    %% initialize and plot units over depth
    subplot(6, 13, [1, 14, 27, 40, 53, 66], 'YDir', 'reverse');
    hold on;
    unitCmap = zeros(length(goodUnits),3);
    unitCmap(goodUnits ==1 ,:,:) = repmat([0, 0.5, 0], length(find(goodUnits==1)),1);
    unitCmap(goodUnits ==0 ,:,:) = repmat([1, 0 , 0], length(find(goodUnits==0)),1);
    norm_spike_n = mat2gray(log10(accumarray(ephysData.spike_templates, 1)+1));
    unitDots = plot(norm_spike_n(uniqueTemps), ephysData.channel_positions(qMetrics.maxChannels(uniqueTemps), 2),5, unitCmap, ...
         'filled', 'ButtonDownFcn', @unit_click);
    currUnitDots = scatter(0, 0, 100, unitCmap(iCluster,:,:), ...
         'filled','MarkerEdgeColor',[0 0 0],'LineWidth',4);
    xlim([-0.1, 1.1]);
    ylim([min(ephysData.channel_positions(:, 2)) - 50, max(ephysData.channel_positions(:, 2)) + 50]);
    ylabel('Depth (\mum)')
    xlabel('Normalized log rate')
    title('Location on probe')

    %% initialize template waveforms
    subplot(6, 13, [2:7, 15:20])
    hold on;
    max_n_channels_plot = 12;
    templateWaveformLines = arrayfun(@(x) plot(nan(82,1),nan(82,1),'linewidth',2,'color','k'),1:max_n_channels_plot-1);
    maxTemplateWaveformLines = arrayfun(@(x) plot(nan(82,1),nan(82,1),'linewidth',2,'color','b'),1);
    xlabel('Position+Time');
    ylabel('Position');
    set(gca, 'YDir', 'reverse')

    %% initialize raw waveforms 
    subplot(6, 13, [8:13, 21:26])
    rawWaveformLines = arrayfun(@(x) plot(nan(82,1),nan(82,1),'linewidth',2,'color','k'),1:max_n_channels_plot);
    maxRawWaveformLines = arrayfun(@(x) plot(nan(82,1),nan(82,1),'linewidth',2,'color','b'),1);
    set(gca, 'YDir', 'reverse')
    xlabel('Position+Time');
    ylabel('Position');
    
    %% initialize ACG 
    subplot(6, 13, 28:31)
    hold on;
    acgBar = arrayfun(@(x) bar(nan(51:101,1),nan(51:101,1)),1);
    acgRefLine = line([NaN, NaN], [NaN, NaN], 'Color', 'r','linewidth',1.2);
    acgAsyLine = line([NaN, NaN], [NaN, NaN], 'Color', 'r','linewidth',1.2);
    xlabel('time (s)');
    ylabel('sp/s');
    
    %% initialize ISI
    subplot(6, 13, [32:35])
    hold on;
    isiBar = arrayfun(@(x) bar(nan(100,1),nan(100,1)),1);
    isiRefLine = line([NaN, NaN], [NaN, NaN], 'Color', 'r','linewidth',1.2);
    xlabel('Interspike interval (ms)')
    ylabel('# of spikes')
    
    %% initialize isoDistance
    subplot(6, 13, 36:39)
    hold on;
    currIsoD = scatter(NaN, NaN, 10, '.b'); % Scatter plot with points of size 10
    otherIsoD = scatter(NaN, NaN, 10, NaN, 'o', 'filled');
    colormap(brewermap([], '*YlOrRd'))
    hb = colorbar;
    ylabel(hb, 'Mahalanobis Distance')
    legend('this cluster', 'other clusters', 'Location', 'best');
    
    %% initialize raw data
    rawPlotH = subplot(6, 13, [41:52, 55:59, 60:65]);
    title('Raw unwhitened data')
    hold on;

    %% initialize amplitude * spikes 
    subplot(6, 13, [67:70, 73:76])
    hold on;
    yyaxis left;
    tempAmpli = scatter(NaN, NaN, 'black', 'filled');
    currTempAmpli = scatter(currTimes, currAmplis, 'blue', 'filled');
    xlabel('Experiment time (s)');
    ylabel('Template amplitude scaling');
    axis tight
    hold on;
    set(gca, 'YColor', 'k')
    yyaxis right
    spikeFR = stairs(NaN,NaN, 'LineWidth', 2.0, 'Color', [1 0.5 0]);
    set(gca,'YColor',[1 0.5 0])
    ylabel('Firing rate (sp/sec)');


    %% initialize amplitude fit 
    subplot(6, 13, [78])
    hold on;
    ampliBins = barh(NaN, NaN, 'blue');
    ampliBins.FaceAlpha = 0.5;
    ampliFit = plot(NaN, NaN, [1.0000, 0.8398, 0], 'LineWidth', 4);

    %% save all handles 
    guiData = struct;
    % location plot
    guiData.unitDots = unitDots;
    guiData.currUnitDots = currUnitDots;
    % template waveforms
    guiData.templateWaveformLines = templateWaveformLines; 
    guiData.maxTemplateWaveformLines = maxTemplateWaveformLines;
    % raw waveforms 
    guiData.rawWaveformLines = rawWaveformLines;
    guiData.maxRawWaveformLines = maxRawWaveformLines;
    % ACG
    guiData.acgBar= acgBar;
    guiData.acgRefLine = acgRefLine;
    guiData.acgAsyLine = acgAsyLine;
    % ISI
    guiData.isiBar= isiBar;
    guiData.isiRefLine = isiRefLine;
    % isoD 
    guiData.currIsoD = currIsoD;
    guiData.otherIsoD = otherIsoD;
    % raw data
    guiData.rawPlotH = rawPlotH;
    % amplitudes * spikes
    guiData.tempAmpli = tempAmpli;
    guiData.currTempAmpli = currTempAmpli;
    guiData.spikeFR = spikeFR;
    % amplitude fit 
    guiData.ampliBins = ampliBins;
    guiData.ampliFit = ampliFit;
    
    % upload guiData 
    guidata(unitQualityGuiHandle, guiData);
end

function updateUnit(unitQualityGuiHandle)
% Get guidata
guiData = guidata(unitQualityGuiHandle);

thisUnit = uniqueTemps(iCluster);

if goodUnits(iCluster) ==1
    sgtitle(['\color[rgb]{0 .5 0}Unit ', num2str(iCluster), ', good']);
else
    sgtitle(['\color{red}Unit ', num2str(iCluster), ', bad']);
end

curr_unit_dots = scatter(norm_spike_n(thisUnit), ephysData.channel_positions(qMetrics.maxChannels(thisUnit), 2),100, unitCmap(iCluster,:,:), ...
     'filled','MarkerEdgeColor',[0 0 0],'LineWidth',4);
 
 %% 1. plot unit location on probe qq: replace by unit * spike rate
tic
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
toc
%% 2. plot unit template waveform and detected peaks
tic
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
            '\color[rgb]{1 0 0}is somatic \color{red}}'])
    else
       title(['\fontsize{9}Template waveform: {\color[rgb]{1 0 0}# detected peaks/troughs, '...
            '\color[rgb]{0 .5 0}is somatic \color{red}}'])
    end
else
    if qMetrics.axonal(iCluster) == 1
        title(['\fontsize{9}Template waveform: {\color[rgb]{0 .5 0}# detected peaks/troughs, '...
            '\color[rgb]{1 0 0}is somatic \color{red}}'])
    else
        title(['\fontsize{9}Template waveform: {\color[rgb]{0 .5 0}# detected peaks/troughs, '...
            '\color[rgb]{0 .5 0}is somatic \color{red}}'])
    end
    
end
TextLocation([num2str(qMetrics.nPeaks(iCluster)), ' peak(s), ', num2str(qMetrics.nTroughs(iCluster)), ' trough(s)', ...
    ' is somatic =', num2str(1 - qMetrics.axonal(iCluster))],'Location', 'North')
toc
%% 3. plot unit mean raw waveform (and individual traces)
tic
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
toc

%% 4. plot unit ACG
tic
theseSpikeTimes = ephysData.spike_times_timeline(ephysData.spike_templates == thisUnit);

%  [Fp, r, overestimate] = bc_fractionRPviolations(numel(theseSpikeTimes),theseSpikeTimes, param.tauR, ...
%      param.tauC, max(theseSpikeTimes)-min(theseSpikeTimes), 1);


[ccg, ccg_t] = CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
    ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', 0.001, 'duration', 0.05, 'norm', 'rate'); %function


subplot(6, 13, 28:31)
area(ccg_t, ccg(:, 1, 1));
hold on;
line([0.002, 0.002], [0, max(ccg(:, 1, 1))], 'Color', 'r')
line([-0.002, -0.002], [0, max(ccg(:, 1, 1))], 'Color', 'r')
[ccg, ~] = CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
    ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', 0.1, 'duration', 10, 'norm', 'rate'); %function
asymptoteLine = nanmean(ccg(end-100:end));
line([-0.1, 0.1], [asymptoteLine, asymptoteLine], 'Color', 'r')
tic
makeprettyNoText
toc
xlim([0, 0.05]); %Check FR
xlabel('time (s)');
ylabel('sp/s');
if qMetrics.Fp(iCluster) > param.maxRPVviolations
    title('\color[rgb]{1 0 0}ACG');
else
    title('\color[rgb]{0 .5 0}ACG');
end

toc
%% 5. plot unit ISI (with refractory period and asymptote lines)
tic
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
toc
%% 6. plot isolation distance
tic
subplot(6, 13, 36:39)

scatter(qMetrics.Xplot{iCluster}(:, 1), qMetrics.Xplot{iCluster}(:, 2), 10, '.b') % Scatter plot with points of size 10
hold on
scatter(qMetrics.Yplot{iCluster}(:, 1), qMetrics.Yplot{iCluster}(:, 2), 10, qMetrics.d2_mahal{iCluster}, 'o', 'filled')
colormap(brewermap([], '*YlOrRd'))
hb = colorbar;
ylabel(hb, 'Mahalanobis Distance')
legend('this cluster', 'other clusters', 'Location', 'best');
toc
%% 7. (optional) plot raster

%% 10. plot ampli fit
tic
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
toc
%% 8. plot raw data
tic
subplot(6, 13, [41:52, 55:59, 60:65])
title('Raw unwhitened data')
hold on;
plotSubRaw(memMapData, ephysData, iCluster, iCount, uniqueTemps, timeChunkStart, timeChunkStop, iChunk);
toc
%% 9. plot template amplitudes and mean f.r. over recording (QQ: add experiment time epochs?)


        tic
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


toc


end

function updateRawSnippet(unitQualityGuiHandle)
% Get guidata
guiData = guidata(unitQualityGuiHandle);
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