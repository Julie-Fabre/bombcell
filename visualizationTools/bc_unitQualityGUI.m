
%% unit quality gui: plot various quality metric plots for single units
% toggle between units with the right and left arrows
% the slowest part by far of this is plotting the raw data, need to figure out how
% to make this faster 
% to add: 
% - toggle next most similar units (ie space bar)
% - individual raw waveforms 
% - add raster plot
% - click on units 
% - probe locations 

function unitQualityGuiHandle = bc_unitQualityGUI(memMapData, ephysData, qMetric, param, probeLocation, unitType, plotRaw)

%% set up dynamic figure
unitQualityGuiHandle = figure('color', 'w');
set(unitQualityGuiHandle, 'KeyPressFcn', @KeyPressCb);

%% initial conditions
iCluster = 1;
iCount = 1;
uniqueTemps = unique(ephysData.spike_templates);
goodUnit_idx = find(unitType==1); 
multiUnit_idx = find(unitType==2); 
noiseUnit_idx = find(unitType==0); 
%% plot initial conditions
iChunk = 1;
initializePlot(unitQualityGuiHandle, ephysData, qMetric, unitType, uniqueTemps, plotRaw, param)
updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
    probeLocation, unitType, uniqueTemps, iChunk, plotRaw);

%% change on keypress
    function KeyPressCb(~, evnt)
        %fprintf('key pressed: %s\n', evnt.Key);
        if strcmpi(evnt.Key, 'rightarrow')
            iCluster = iCluster + 1;
            updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
                probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
        elseif strcmpi(evnt.Key, 'g') %toggle to next single-unit 
            iCluster = goodUnit_idx(find(goodUnit_idx>iCluster,1,'first'));
            updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
                probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
        elseif strcmpi(evnt.Key, 'm') %toggle to next multi-unit 
            iCluster = multiUnit_idx(find(multiUnit_idx>iCluster,1,'first'));
            updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
                probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
        elseif strcmpi(evnt.Key, 'n') %toggle to next  noise/non-somatic unit
            iCluster = noiseUnit_idx(find(noiseUnit_idx>iCluster,1,'first'));
            updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
                probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
        elseif strcmpi(evnt.Key, 'leftarrow')
            iCluster = iCluster - 1;
            updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
                probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
        elseif strcmpi(evnt.Key, 'uparrow')
            iChunk = iChunk + 1;
            if iChunk> length(ephysData.spike_times_timeline(ephysData.spike_templates == iCluster));
                iChunk=1;
            end
            updateRawSnippet(unitQualityGuiHandle, memMapData, ephysData, iCluster, iCount, qMetric, param, ...
    probeLocation, uniqueTemps, iChunk, plotRaw);
        elseif strcmpi(evnt.Key, 'downarrow')
            iChunk = iChunk - 1;
            if iChunk==0
                iChunk = length(ephysData.spike_times_timeline(ephysData.spike_templates == iCluster));
            end
            updateRawSnippet(unitQualityGuiHandle, memMapData, ephysData, iCluster, iCount, qMetric, param, ...
    probeLocation, uniqueTemps, iChunk, plotRaw);
        elseif strcmpi(evnt.Key, 'u') %select particular unit 
            iCluster = str2num(cell2mat(inputdlg('Go to unit:')));
            updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
                probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
        end
    end
end

function initializePlot(unitQualityGuiHandle, ephysData, qMetric, unitType, uniqueTemps, plotRaw, param)

%% main title

mainTitle = sgtitle('');

%% initialize and plot units over depth

subplot(6, 13, [1, 14, 27, 40, 53, 66], 'YDir', 'reverse');
hold on;
unitCmap = zeros(length(unitType), 3);
unitCmap(unitType == 1, :, :) = repmat([0, 0.5, 0], length(find(unitType == 1)), 1);
unitCmap(unitType == 0, :, :) = repmat([1, 0, 0], length(find(unitType == 0)), 1);
unitCmap(unitType == 2, :, :) = repmat([0.29, 0, 0.51], length(find(unitType == 2)), 1);
norm_spike_n = mat2gray(log10(accumarray(ephysData.spike_templates, 1)+1));
unitDots = scatter(norm_spike_n(uniqueTemps), ephysData.channel_positions(qMetric.maxChannels(uniqueTemps), 2), 5, unitCmap, ...
    'filled', 'ButtonDownFcn', @unit_click);
currUnitDots = scatter(0, 0, 100, unitCmap(1, :, :), ...
    'filled', 'MarkerEdgeColor', [0, 0, 0], 'LineWidth', 4);
xlim([-0.1, 1.1]);
ylim([min(ephysData.channel_positions(:, 2)) - 50, max(ephysData.channel_positions(:, 2)) + 50]);
ylabel('Depth (\mum)')
xlabel('Normalized log rate')
title('Location on probe')

%% initialize template waveforms
subplot(6, 13, [2:7, 15:20])
hold on;
max_n_channels_plot = 20;
templateWaveformLines = arrayfun(@(x) plot(nan(82, 1), nan(82, 1), 'linewidth', 2, 'color', 'k'), 1:max_n_channels_plot);
maxTemplateWaveformLines = arrayfun(@(x) plot(nan(82, 1), nan(82, 1), 'linewidth', 2, 'color', 'b'), 1);
peaks = scatter(nan(10, 1), nan(10, 1), [], rgb('Orange'), 'v', 'filled');
troughs = scatter(nan(10, 1), nan(10, 1), [], rgb('Gold'), 'v', 'filled');
xlabel('Position+Time');
ylabel('Position');
set(gca, 'YDir', 'reverse')
tempTitle = title('');
tempLegend = legend([maxTemplateWaveformLines, peaks, troughs, templateWaveformLines(1)], {'', '', '',''},'color','none');

%% initialize raw waveforms
subplot(6, 13, [8:13, 21:26])
hold on;
rawWaveformLines = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', 'k'), 1:max_n_channels_plot);
maxRawWaveformLines = arrayfun(@(x) plot(nan(82, 1), nan(82, 1), 'linewidth', 2, 'color', 'b'), 1);
set(gca, 'YDir', 'reverse')
xlabel('Position+Time');
ylabel('Position');
rawTitle = title('');
rawLegend = legend([maxRawWaveformLines], {''},'color','none');

%% initialize ACG
if plotRaw && param.computeDistanceMetrics
    subplot(6, 13, 28:31)
else
    subplot(6, 13, 28:33)
end
hold on;
acgBar = arrayfun(@(x) bar(0:0.1:25, nan(251, 1)), 1);
acgRefLine = line([NaN, NaN], [NaN, NaN], 'Color', 'r', 'linewidth', 1.2);
acgAsyLine = line([NaN, NaN], [NaN, NaN], 'Color', 'r', 'linewidth', 1.2);
xlabel('time (ms)');
ylabel('sp/s');
acgTitle = title('');

%% initialize ISI

if plotRaw && param.computeDistanceMetrics
    subplot(6, 13, 32:35)
else
    subplot(6, 13, 34:39)

end
hold on;
isiBar = arrayfun(@(x) bar((0 + 0.25):0.5:(50 - 0.25), nan(100, 1)), 1);
isiRefLine = line([NaN, NaN], [NaN, NaN], 'Color', 'r', 'linewidth', 1.2);
xlabel('Interspike interval (ms)')
ylabel('# of spikes')
isiTitle = title('');
isiLegend = legend([isiBar], {''});

%% initialize isoDistance
if param.computeDistanceMetrics
if plotRaw
    subplot(6, 13, 36:39)
else
    subplot(6, 13, 41:46)
end
hold on;
currIsoD = scatter(NaN, NaN, 10, '.b'); % Scatter plot with points of size 10
rpvIsoD = scatter(NaN, NaN, 10, '.m'); % Scatter plot with points of size 10
otherIsoD = scatter(NaN, NaN, 10, NaN, 'o', 'filled');

colormap(brewermap([], '*YlOrRd'))
hb = colorbar;
ylabel(hb, 'Mahalanobis Distance')
legend('this cluster',  'rpv spikes', 'other clusters');
isoDTitle = title('');
end
%% initialize raw data
if plotRaw
    rawPlotH = subplot(6, 13, [41:52, 55:59, 60:65]);
    hold on;
    title('Raw unwhitened data')
    set(rawPlotH, 'XColor','w', 'YColor','w')
    rawPlotLines = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', 'k'), 1:max_n_channels_plot);
    rawSpikeLines = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', 'b'), 1:max_n_channels_plot);
end

%% initialize amplitude * spikes
ampliAx = subplot(6, 13, [67:70, 73:76]);
hold on;
yyaxis left;
tempAmpli = scatter(NaN, NaN, 'black', 'filled');
currTempAmpli = scatter(NaN, NaN, 'blue', 'filled');
rpvAmpli= scatter(NaN, NaN, 10, 'magenta', 'filled');
xlabel('Experiment time (s)');
ylabel('Template amplitude scaling');
axis tight
hold on;
set(gca, 'YColor', 'k')
yyaxis right
spikeFR = stairs(NaN, NaN, 'LineWidth', 2.0, 'Color', [1, 0.5, 0]);
set(gca, 'YColor', [1, 0.5, 0])
ylabel('Firing rate (sp/sec)');
ampliTitle = title('');
ampliLegend = legend([tempAmpli,rpvAmpli], {'', ''});

%% initialize amplitude fit
ampliFitAx = subplot(6, 13, [78]);
hold on;
ampliBins = barh(NaN, NaN, 'blue');
ampliBins.FaceAlpha = 0.5;
ampliFit = plot(NaN, NaN, 'Color', rgb('Orange'), 'LineWidth', 4);
ampliFitTitle = title('');
ampliFitLegend = legend([ampliFit], {''}, 'Location', 'South');

%% save all handles
guiData = struct;
% main title
guiData.mainTitle = mainTitle;
% location plot
guiData.unitDots = unitDots;
guiData.currUnitDots = currUnitDots;
guiData.unitCmap = unitCmap;
guiData.norm_spike_n = norm_spike_n;
% template waveforms
guiData.templateWaveformLines = templateWaveformLines;
guiData.maxTemplateWaveformLines = maxTemplateWaveformLines;
guiData.tempTitle = tempTitle;
guiData.tempLegend = tempLegend;
guiData.peaks = peaks;
guiData.troughs = troughs;
% raw waveforms
guiData.rawWaveformLines = rawWaveformLines;
guiData.maxRawWaveformLines = maxRawWaveformLines;
guiData.rawTitle = rawTitle;
guiData.rawLegend = rawLegend;
% ACG
guiData.acgBar = acgBar;
guiData.acgRefLine = acgRefLine;
guiData.acgAsyLine = acgAsyLine;
guiData.acgTitle = acgTitle;
% ISI
guiData.isiBar = isiBar;
guiData.isiRefLine = isiRefLine;
guiData.isiTitle = isiTitle;
guiData.isiLegend = isiLegend;
% isoD
if param.computeDistanceMetrics
guiData.currIsoD = currIsoD;
guiData.otherIsoD = otherIsoD;
guiData.isoDTitle = isoDTitle;
guiData.rpvIsoD = rpvIsoD;
end
% raw data
if plotRaw
    guiData.rawPlotH = rawPlotH;
    guiData.rawPlotLines = rawPlotLines;
    guiData.rawSpikeLines = rawSpikeLines;
end
% amplitudes * spikes
guiData.ampliAx = ampliAx;
guiData.tempAmpli = tempAmpli;
guiData.currTempAmpli = currTempAmpli;
guiData.spikeFR = spikeFR;
guiData.ampliTitle = ampliTitle;
guiData.ampliLegend = ampliLegend;
% amplitude fit
guiData.ampliFitAx = ampliFitAx;
guiData.ampliBins = ampliBins;
guiData.ampliFit = ampliFit;
guiData.ampliFitTitle = ampliFitTitle;
guiData.ampliFitLegend = ampliFitLegend;
guiData.rpvAmpli = rpvAmpli;
% upload guiData
guidata(unitQualityGuiHandle, guiData);
end

function updateUnit(unitQualityGuiHandle, memMapData, ephysData, iCluster, qMetric, param, ...
    probeLocation, unitType, uniqueTemps, iChunk, plotRaw)

%% Get guidata
guiData = guidata(unitQualityGuiHandle);
thisUnit = uniqueTemps(iCluster);
colorsGdBad = [1, 0, 0; 0, 0.5, 0];
%% main title
if unitType(iCluster) == 1
    set(guiData.mainTitle, 'String', ['Unit ', num2str(iCluster), ', single unit'], 'Color', [0, .5, 0]);
elseif unitType(iCluster) == 0
    set(guiData.mainTitle, 'String', ['Unit ', num2str(iCluster), ', noise/non-somatic'], 'Color', [1, 0, 0]);
elseif unitType(iCluster) == 2
    set(guiData.mainTitle, 'String', ['Unit ', num2str(iCluster), ', multi-unit'], 'Color', [0.29, 0, 0.51]);
end

%% plot 1: update curr unit location
set(guiData.currUnitDots, 'XData', guiData.norm_spike_n(thisUnit), 'YData', ephysData.channel_positions(qMetric.maxChannels(thisUnit), 2), 'CData', guiData.unitCmap(iCluster, :))

for iCh = 1:20
    set(guiData.templateWaveformLines(iCh), 'XData', nan(82, 1), 'YData', nan(82, 1))
    set(guiData.rawWaveformLines(iCh), 'XData', nan(82, 1), 'YData', nan(82, 1))
end
%% plot 2: update unit template waveform and detected peaks
% guiData.templateWaveformLines = templateWaveformLines;
%     guiData.maxTemplateWaveformLines = maxTemplateWaveformLines;
%     guiData.tempTitle = tempTitle;
%     guiData.tempLegend = tempLegend;

maxChan = qMetric.maxChannels(thisUnit);
maxXC = ephysData.channel_positions(maxChan, 1);
maxYC = ephysData.channel_positions(maxChan, 2);
chanDistances = ((ephysData.channel_positions(:, 1) - maxXC).^2 ...
    +(ephysData.channel_positions(:, 2) - maxYC).^2).^0.5;
chansToPlot = find(chanDistances < 100);
vals =[];
for iChanToPlot = 1:min(20, size(chansToPlot, 1))
    vals(iChanToPlot) = max(abs(squeeze(ephysData.templates(thisUnit, :, chansToPlot(iChanToPlot)))));
    if maxChan == chansToPlot(iChanToPlot)
        set(guiData.maxTemplateWaveformLines, 'XData', (ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            'YData', -squeeze(ephysData.templates(thisUnit, :, chansToPlot(iChanToPlot)))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) ./ 100));
        hold on;
        set(guiData.peaks, 'XData', (ephysData.waveform_t(qMetric.peakLocs{iCluster}) ...
            +(ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            'YData', -squeeze(ephysData.templates(thisUnit, qMetric.peakLocs{iCluster}, chansToPlot(iChanToPlot)))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) ./ 100));

        set(guiData.troughs, 'XData', (ephysData.waveform_t(qMetric.troughLocs{iCluster}) ...
            +(ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            'YData', -squeeze(ephysData.templates(thisUnit, qMetric.troughLocs{iCluster}, chansToPlot(iChanToPlot)))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) ./ 100));
        set(guiData.templateWaveformLines(iChanToPlot), 'XData', nan(82, 1), ...
            'YData', nan(82, 1));

    else
        set(guiData.templateWaveformLines(iChanToPlot), 'XData', (ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            'YData', -squeeze(ephysData.templates(thisUnit, :, chansToPlot(iChanToPlot)))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) ./ 100));
    end
end

tempWvTitleText = ['\\fontsize{9}Template waveform: {\\color[rgb]{%s}# detected peaks/troughs, ', newline,...
            '\\color[rgb]{%s}is somatic  \\color[rgb]{%s}spatial decay}'];
       
set(guiData.tempTitle, 'String',  sprintf(tempWvTitleText,num2str(colorsGdBad(double(qMetric.nPeaks(iCluster) <= param.maxNPeaks || qMetric.nTroughs(iCluster) <= param.maxNTroughs)+1,:)),...
    num2str(colorsGdBad(double(qMetric.somatic(iCluster) == 1)+1,:)),  ...
    num2str(colorsGdBad(double(qMetric.spatialDecaySlope(iCluster) > param.minSpatialDecaySlope)+1,:))));

set(guiData.tempLegend, 'String', {['is somatic =', num2str(qMetric.somatic(iCluster)), newline], ...
    [num2str(qMetric.nPeaks(iCluster)), ' peak(s)'], [num2str(qMetric.nTroughs(iCluster)), ...
    ' trough(s)'], ['spatial decay slope =' , num2str(qMetric.spatialDecaySlope(iCluster))]})

%% plot 3: plot unit mean raw waveform (and individual traces)

for iChanToPlot = 1:min(20, size(chansToPlot, 1))
    if maxChan == chansToPlot(iChanToPlot)
        set(guiData.maxRawWaveformLines, 'XData', (ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            'YData', -squeeze(qMetric.rawWaveforms(iCluster).spkMapMean(chansToPlot(iChanToPlot), :))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) * 10));
        set(guiData.rawWaveformLines(iChanToPlot), 'XData', nan(82, 1), ...
            'YData', nan(82, 1));

    else
        set(guiData.rawWaveformLines(iChanToPlot), 'XData', (ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            'YData', -squeeze(qMetric.rawWaveforms(iCluster).spkMapMean(chansToPlot(iChanToPlot), :))'+ ...
            (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) * 10));
    end
end
set(guiData.rawLegend, 'String', ['Amplitude =', num2str(qMetric.rawAmplitude(iCluster)), 'uV'])
if qMetric.rawAmplitude(iCluster) < param.minAmplitude
    set(guiData.rawTitle, 'String', '\color[rgb]{1 0 1}Mean raw waveform');
else
    set(guiData.rawTitle, 'String', '\color[rgb]{0 .5 0}Mean raw waveform');
end

%% 4. plot unit ACG

theseSpikeTimes = ephysData.spike_times_timeline(ephysData.spike_templates == thisUnit);

[ccg, ccg_t] = CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
    ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', 0.001, 'duration', 0.5, 'norm', 'rate'); %function

set(guiData.acgBar, 'XData', ccg_t(250:501)*1000, 'YData', squeeze(ccg(250:501, 1, 1)));
set(guiData.acgRefLine, 'XData', [2, 2], 'YData', [0, max(ccg(:, 1, 1))])
[ccg2, ~] = CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
    ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', 0.1, 'duration', 10, 'norm', 'rate'); %function
asymptoteLine = nanmean(ccg2(end-100:end));
set(guiData.acgAsyLine, 'XData', [0, 250], 'YData', [asymptoteLine, asymptoteLine])

if qMetric.Fp(iCluster) > param.maxRPVviolations
    set(guiData.acgTitle, 'String', '\color[rgb]{1 0 1}ACG');
else
    set(guiData.acgTitle, 'String', '\color[rgb]{0 .5 0}ACG');
end

%% 5. plot unit ISI (with refractory period and asymptote lines)

theseISI = diff(theseSpikeTimes);
theseISIclean = theseISI(theseISI >= param.tauC); % removed duplicate spikes
theseOffendingSpikes = find(theseISIclean < (2/1000)); 

%theseOffendingSpikes = [theseOffendingSpikes; theseOffendingSpikes-1];
[isiProba, edgesISI] = histcounts(theseISIclean*1000, [0:0.5:50]);

set(guiData.isiBar, 'XData', edgesISI(1:end-1)+mean(diff(edgesISI)), 'YData', isiProba); %Check FR
set(guiData.isiRefLine, 'XData', [2, 2], 'YData', [0, max(isiProba)])

if qMetric.Fp(iCluster) > param.maxRPVviolations
    set(guiData.isiTitle, 'String', '\color[rgb]{1 0 1}ISI');
else
    set(guiData.isiTitle, 'String', '\color[rgb]{0 .5 0}ISI');
end
set(guiData.isiLegend, 'String', [num2str(qMetric.Fp(iCluster)), ' % r.p.v.'])

%% 6. plot isolation distance
if param.computeDistanceMetrics
set(guiData.currIsoD, 'XData', qMetric.Xplot{iCluster}(:, 1), 'YData', qMetric.Xplot{iCluster}(:, 2))
set(guiData.rpvIsoD, 'XData', qMetric.Xplot{iCluster}(theseOffendingSpikes, 1), 'YData', qMetric.Xplot{iCluster}(theseOffendingSpikes, 2))
set(guiData.otherIsoD, 'XData', qMetric.Yplot{iCluster}(:, 1), 'YData', qMetric.Yplot{iCluster}(:, 2), 'CData', qMetric.d2_mahal{iCluster})
end
%% 7. (optional) plot raster

%% 10. plot ampli fit

    
set(guiData.ampliBins, 'XData', qMetric.ampliBinCenters{iCluster}, 'YData', qMetric.ampliBinCounts{iCluster});

set(guiData.ampliFit, 'XData', qMetric.ampliFit{iCluster}, 'YData', qMetric.ampliBinCenters{iCluster})
if qMetric.percSpikesMissing(iCluster) > param.maxPercSpikesMissing
    set(guiData.ampliFitTitle, 'String', '\color[rgb]{1 0 1}% spikes missing');
else
    set(guiData.ampliFitTitle, 'String', '\color[rgb]{0 .5 0}% spikes missing');
end
set(guiData.ampliFitLegend, 'String', {[num2str(qMetric.percSpikesMissing(iCluster)), ' % spikes missing'], 'rpv spikes'})
set(guiData.ampliFitAx, 'YLim', [min(qMetric.ampliBinCenters{iCluster}), max(qMetric.ampliBinCenters{iCluster})])

%% 9. plot template amplitudes and mean f.r. over recording (QQ: add experiment time epochs?)

ephysData.recordingDuration = (max(ephysData.spike_times_timeline) - min(ephysData.spike_times_timeline));
theseAmplis = ephysData.template_amplitudes(ephysData.spike_templates == thisUnit);

% for debugging if wierd amplitude fit results: bc_percSpikesMissing(theseAmplis, theseSpikeTimes, [min(theseSpikeTimes), max(theseSpikeTimes)], 1);

set(guiData.tempAmpli, 'XData', theseSpikeTimes, 'YData', theseAmplis)
set(guiData.rpvAmpli, 'XData', theseSpikeTimes(theseOffendingSpikes), 'YData', theseAmplis(theseOffendingSpikes))
currTimes = theseSpikeTimes(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1 & theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
currAmplis = theseAmplis(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1 & theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
set(guiData.currTempAmpli, 'XData', currTimes, 'YData', currAmplis);
set(guiData.ampliAx.YAxis(1), 'Limits', [0, round(max(theseAmplis))])

binSize = 20;
timeBins = 0:binSize:ceil(ephysData.spike_times(end)/ephysData.ephys_sample_rate);
while length(timeBins)==1    
    binSize = binSize/2;
    timeBins = 0:binSize:ceil(ephysData.spike_times(end)/ephysData.ephys_sample_rate);
end
[n, x] = hist(theseSpikeTimes, timeBins);
n = n ./ binSize;

set(guiData.spikeFR, 'XData', x, 'YData', n);
set(guiData.ampliAx.YAxis(2), 'Limits', [0, 2 * ceil(max(n))])


if qMetric.nSpikes(iCluster) > param.minNumSpikes
    set(guiData.ampliTitle, 'String', '\color[rgb]{0 .5 0}Spikes');
else
    set(guiData.ampliTitle, 'String', '\color[rgb]{1 0 1}Spikes');
end
set(guiData.ampliLegend, 'String', {['# spikes = ', num2str(qMetric.nSpikes(iCluster))], 'rpv spikes'})

%% 8. plot raw data
if plotRaw
    plotSubRaw(guiData.rawPlotH, guiData.rawPlotLines, guiData.rawSpikeLines, memMapData, ephysData, iCluster, uniqueTemps, iChunk);
end
end

function updateRawSnippet(unitQualityGuiHandle, memMapData, ephysData, iCluster, iCount, qMetric, param, ...
    probeLocation, uniqueTemps, iChunk, plotRaw)

if plotRaw % Get guidata
    guiData = guidata(unitQualityGuiHandle);
    thisUnit = uniqueTemps(iCluster);

    %% 8. plot raw data

    plotSubRaw(guiData.rawPlotH, guiData.rawPlotLines, guiData.rawSpikeLines, memMapData, ephysData, iCluster, uniqueTemps, iChunk);

    %% 9. plot template amplitudes and mean f.r. over recording (QQ: add experiment time epochs?)
    % guiData.tempAmpli = tempAmpli;
    %     guiData.currTempAmpli = currTempAmpli;
    %     guiData.spikeFR = spikeFR;
    %     guiData.ampliTitle = ampliTitle;
    %     guiData.ampliLegend = ampliLegend;
    theseSpikeTimes = ephysData.spike_times_timeline(ephysData.spike_templates == thisUnit);
    ephysData.recordingDuration = (max(ephysData.spike_times_timeline) - min(ephysData.spike_times_timeline));
    theseAmplis = ephysData.template_amplitudes(ephysData.spike_templates == thisUnit);
    set(guiData.tempAmpli, 'XData', theseSpikeTimes, 'YData', theseAmplis)
    currTimes = theseSpikeTimes(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1 & theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
    currAmplis = theseAmplis(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1 & theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
    set(guiData.currTempAmpli, 'XData', currTimes, 'YData', currAmplis);
    set(guiData.ampliAx.YAxis(1), 'Limits', [0, round(max(theseAmplis))])

    binSize = 20;
    timeBins = 0:binSize:ceil(ephysData.spike_times(end)/ephysData.ephys_sample_rate);
    [n, x] = hist(theseSpikeTimes, timeBins);
    n = n ./ binSize;

    set(guiData.spikeFR, 'XData', x, 'YData', n);
    set(guiData.ampliAx.YAxis(2), 'Limits', [0, 2 * round(max(n))])


    if qMetric.nSpikes(iCluster) > param.minNumSpikes
        set(guiData.ampliTitle, 'String', '\color[rgb]{0 .5 0}Spikes');
    else
        set(guiData.ampliTitle, 'String', '\color[rgb]{1 0 0}Spikes');
    end
    set(guiData.ampliLegend, 'String', ['# spikes = ', num2str(qMetric.nSpikes(iCluster))])

end
end


function plotSubRaw(rawPlotH, rawPlotLines, rawSpikeLines, memMapData, ephysData, iCluster, uniqueTemps, iChunk)
%get the used channels
chanAmps = squeeze(max(ephysData.templates(iCluster, :, :))-min(ephysData.templates(iCluster, :, :)));
maxChan = find(chanAmps == max(chanAmps), 1);
maxXC = ephysData.channel_positions(maxChan, 1);
maxYC = ephysData.channel_positions(maxChan, 2);
chanDistances = ((ephysData.channel_positions(:, 1) - maxXC).^2 ...
    +(ephysData.channel_positions(:, 2) - maxYC).^2).^0.5;
chansToPlot = find(chanDistances < 100);

%get spike locations
timeToPlot = 0.1;
pull_spikeT = -40:41;
thisC = uniqueTemps(iCluster);
theseTimesCenter = ephysData.spike_times(ephysData.spike_templates == thisC) ./ ephysData.ephys_sample_rate;
if iChunk < 0
    disp('Don''t do that')
    iChunk = 1;
end
if length(theseTimesCenter)>10+iChunk
    firstSpike = theseTimesCenter(iChunk+10) - 0.05; %tenth spike occurance %
else
    firstSpike = theseTimesCenter(iChunk)-0.05; %first spike occurance
end
% Not sure why this was +10?
theseTimesCenter = theseTimesCenter(theseTimesCenter >= firstSpike);
theseTimesCenter = theseTimesCenter(theseTimesCenter <= firstSpike+timeToPlot);
if ~isempty(theseTimesCenter)
    %theseTimesCenter=theseTimesCenter(1);
    theseTimesFull = theseTimesCenter * ephysData.ephys_sample_rate + pull_spikeT;
    %theseTimesFull=unique(sort(theseTimesFull));
end

cCount = cumsum(repmat(1000, size(chansToPlot, 1), 1), 1);


t = int32(firstSpike * ephysData.ephys_sample_rate):int32((firstSpike + timeToPlot) * ephysData.ephys_sample_rate);
subplot(rawPlotH)
plotidx =  int32(firstSpike*ephysData.ephys_sample_rate) ...
    :int32((firstSpike + timeToPlot)*ephysData.ephys_sample_rate);
t(plotidx<1|plotidx>size(memMapData,2))=[];
plotidx(plotidx<1|plotidx>size(memMapData,2))=[];
thisMemMap = double(memMapData(chansToPlot,plotidx))+double(cCount);
for iClear = 1:length(rawSpikeLines)
     set(rawSpikeLines(iClear), 'XData', NaN,'YData',NaN)
end
if length(rawSpikeLines)<length(chansToPlot)
    rawSpikeLines(end+1:length(chansToPlot))=rawSpikeLines(end);
    rawPlotLines(end+1:length(chansToPlot))=rawPlotLines(end);
end
 for iChanToPlot = 1:length(chansToPlot)
     set(rawPlotLines(iChanToPlot), 'XData', t,'YData',thisMemMap(iChanToPlot,:));
     if ~isempty(theseTimesCenter)
        for iTimes = 1:size(theseTimesCenter, 1)
            if ~any(mod(theseTimesFull(iTimes, :), 1))
                set(rawSpikeLines(iChanToPlot), 'XData', theseTimesFull(iTimes, :),'YData',thisMemMap(iChanToPlot, ...
                    int32(theseTimesFull(iTimes, :))-t(1)));
            end
        end
     end

 end


end

% function unit_click(unitQualityGuiHandle,eventdata)
% 
% % Get guidata
% guiData = guidata(unitQualityGuiHandle);
% 
% % Get the clicked unit, update current unit
% unit_x = get(guiData.unitDots,'XData');
% unit_y = get(guiData.unitDots,'YData');
% 
% [~,clicked_unit] = min(sqrt(sum(([unit_x;unit_y] - ...
%     evnt.Key.IntersectionPoint(1:2)').^2,1)));
% 
% gui_data.curr_unit = clicked_unit;
% 
% % Upload gui data and draw
% guidata(cellraster_gui,gui_data);
% update_plot(cellraster_gui);
% 
% end