
%% unit quality gui: plot various quality metric plots for single units
% This function is very messy - don't attempt to modify. Send me an email or raise an issue :)
%
% toggle between units with the right and left arrows
% the slowest part by far of this is plotting the raw data, need to figure out how
% to make this faster
% to add:
% - toggle next most similar units (ie space bar)
% - individual raw waveforms
% - add raster plot
% - click on units
% - probe locations

function unitQualityGuiHandle = unitQualityGUI_synced(memMapData, ephysData, qMetric, forGUI, rawWaveforms, ...
    param, probeLocation, unitType, plotRaw)

%% set up dynamic figures
unitQualityGuiHandle.mainFigure = figure('Name', 'Unit Quality GUI - Main', 'Position', [100, 100, 900, 900], 'Color', 'w');
unitQualityGuiHandle.histogramFigure = figure('Name', 'Unit Quality GUI - Histograms', 'Position', [920, 100, 900, 900], 'Color', 'w');

% Set up KeyPressFcn for both figures
set(unitQualityGuiHandle.mainFigure, 'KeyPressFcn', @KeyPressCb);
set(unitQualityGuiHandle.histogramFigure, 'KeyPressFcn', @KeyPressCb);

%% initial conditions
iCluster = 1;
iCount = 1;
uniqueTemps = unique(ephysData.spike_templates);
goodUnit_idx = find(unitType == 1);
multiUnit_idx = find(unitType == 2);
noiseUnit_idx = find(unitType == 0); % noise
if param.splitGoodAndMua_NonSomatic
    nonSomaUnit_idx = find(unitType == 3 | unitType == 4);
else
    nonSomaUnit_idx = find(unitType == 3);
end

%% plot initial conditions
iChunk = 1;
initializePlot(unitQualityGuiHandle, ephysData, qMetric, unitType, uniqueTemps, plotRaw, param)
updateUnit(unitQualityGuiHandle, memMapData, ephysData, rawWaveforms, iCluster, qMetric, forGUI, param, ...
    probeLocation, unitType, uniqueTemps, iChunk, plotRaw);

%% change on keypress
    function KeyPressCb(~, evnt)
        %fprintf('key pressed: %s\n', evnt.Key);
        if isempty(iCluster)
            iCluster = 1;
        end
        if strcmpi(evnt.Key, 'rightarrow')
            iCluster = iCluster + 1;
            if size(uniqueTemps, 1) < iCluster
                disp('Done cycling through units.')
                iCluster = iCluster - 1;
            else
                updateUnit(unitQualityGuiHandle, memMapData, ephysData, rawWaveforms, iCluster, qMetric, forGUI, param, ...
                    probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
            end


        elseif strcmpi(evnt.Key, 'g') % toggle to next single-unit
            iCluster = goodUnit_idx(find(goodUnit_idx > iCluster, 1, 'first'));
            if ~isempty(iCluster)
                updateUnit(unitQualityGuiHandle, memMapData, ephysData, rawWaveforms, iCluster, qMetric, forGUI, param, ...
                    probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
            else
                disp('Done cycling through good units.')
            end
        elseif strcmpi(evnt.Key, 'm') % toggle to next multi-unit
            iCluster = multiUnit_idx(find(multiUnit_idx > iCluster, 1, 'first'));
            if ~isempty(iCluster)
                updateUnit(unitQualityGuiHandle, memMapData, ephysData, rawWaveforms, iCluster, qMetric, forGUI, param, ...
                    probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
            else
                disp('Done cycling through MUA units.')
            end
        elseif strcmpi(evnt.Key, 'n') % toggle to next noise unit
            iCluster = noiseUnit_idx(find(noiseUnit_idx > iCluster, 1, 'first'));
            if ~isempty(iCluster)
                updateUnit(unitQualityGuiHandle, memMapData, ephysData, rawWaveforms, iCluster, qMetric, forGUI, param, ...
                    probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
            else
                disp('Done cycling through noise units.')
            end
        elseif strcmpi(evnt.Key, 'a') % toggle to next non-somatic unit
            iCluster = nonSomaUnit_idx(find(nonSomaUnit_idx > iCluster, 1, 'first'));
            if ~isempty(iCluster)
                updateUnit(unitQualityGuiHandle, memMapData, ephysData, rawWaveforms, iCluster, qMetric, forGUI, param, ...
                    probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
            else
                disp('Done cycling through non-somatic units.')
            end
        elseif strcmpi(evnt.Key, 'leftarrow')
            iCluster = iCluster - 1;
            if iCluster < 1
                disp('Done cycling through units.')
                iCluster = iCluster + 1;
            else
                updateUnit(unitQualityGuiHandle, memMapData, ephysData, rawWaveforms, iCluster, qMetric, forGUI, param, ...
                    probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
            end

            updateUnit(unitQualityGuiHandle, memMapData, ephysData, rawWaveforms, iCluster, qMetric, forGUI, param, ...
                probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
        elseif strcmpi(evnt.Key, 'uparrow')
            iChunk = iChunk + 1;
            if iChunk > length(ephysData.spike_times(ephysData.spike_templates == iCluster))
                iChunk = 1;
            end
            updateRawSnippet(unitQualityGuiHandle, memMapData, ephysData, iCluster, iCount, qMetric, param, ...
                probeLocation, uniqueTemps, iChunk, plotRaw);
        elseif strcmpi(evnt.Key, 'downarrow')
            iChunk = iChunk - 1;
            if iChunk == 0
                iChunk = length(ephysData.spike_times(ephysData.spike_templates == iCluster));
            end
            updateRawSnippet(unitQualityGuiHandle, memMapData, ephysData, iCluster, iCount, qMetric, param, ...
                probeLocation, uniqueTemps, iChunk, plotRaw);
        elseif strcmpi(evnt.Key, 'u') %select particular unit
            iCluster = str2num(cell2mat(inputdlg('Go to unit:')));
            updateUnit(unitQualityGuiHandle, memMapData, ephysData, rawWaveforms, iCluster, qMetric, forGUI, param, ...
                probeLocation, unitType, uniqueTemps, iChunk, plotRaw);
        end
    end
end

function initializePlot(unitQualityGuiHandle, ephysData, qMetric, unitType, uniqueTemps, plotRaw, param)

%% Histogram figure
guiData = struct;
figure(unitQualityGuiHandle.histogramFigure);

% Define metrics, thresholds, and plot conditions
[metricNames, metricThresh1, metricThresh2, plotConditions, metricNames_SHORT, metricLineCols] = defineMetrics(param);

numSubplots = sum(plotConditions);

% Calculate the best grid layout
numRows = floor(sqrt(numSubplots));
numCols = ceil(numSubplots/numRows);

% Color matrix setup
red_colors = [; ...
    0.8627, 0.0784, 0.2353; ... % Crimson
    1.0000, 0.1412, 0.0000; ... % Scarlet
    0.7255, 0.0000, 0.0000; ... % Cherry
    0.5020, 0.0000, 0.1255; ... % Burgundy
    0.5020, 0.0000, 0.0000; ... % Maroon
    0.8039, 0.3608, 0.3608; ... % Indian Red
    ];

blue_colors = [; ...
    0.2549, 0.4118, 0.8824; ... % Royal Blue
    0.0000, 0.0000, 0.5020; ... % Navy Blue
    ];
% Color matrix setup
darker_yellow_orange_colors = [; ...
    0.7843, 0.7843, 0.0000; ... % Dark Yellow
    0.8235, 0.6863, 0.0000; ... % Dark Golden Yellow
    0.8235, 0.5294, 0.0000; ... % Dark Orange
    0.8039, 0.4118, 0.3647; ... % Dark Coral
    0.8235, 0.3176, 0.2275; ... % Dark Tangerine
    0.8235, 0.6157, 0.6510; ... % Dark Salmon
    0.7882, 0.7137, 0.5765; ... % Dark Goldenrod
    0.8235, 0.5137, 0.3922; ... % Dark Light Coral
    0.7569, 0.6196, 0.0000; ... % Darker Goldenrod
    0.8235, 0.4510, 0.0000; ... % Darker Orange
    ];


colorMtx = [red_colors; blue_colors; darker_yellow_orange_colors]; % shuffle colors

%sgtitle([num2str(sum(unitType == 1)), ' single units, ', num2str(sum(unitType == 2)), ' multi-units, ', ...
%    num2str(sum(unitType == 0)), ' noise units, ', num2str(sum(unitType == 3)), ' non-somatic units.']);

currentSubplot = 1;
for i = 1:length(plotConditions)
    if plotConditions(i)
        ax = subplot(numRows, numCols, currentSubplot);
        hold(ax, 'on');

        metricName = metricNames{i};
        metricData = qMetric.(metricName);

        % Plot histogram
        if i > 2
            h = histogram(ax, metricData, 40, 'FaceColor', colorMtx(i, 1:3), 'Normalization', 'probability');
        else
            h = histogram(ax, metricData, 'FaceColor', colorMtx(i, 1:3), 'Normalization', 'probability');
        end
        binsize_offset = h.BinWidth / 2;
        % Add horizontal lines instead of rectangles
        yLim = ylim(ax);
        xLim = xlim(ax);
        lineY = yLim(1) - 0.04 * (yLim(2) - yLim(1)); % Position lines slightly below the plot

        if ~isnan(metricThresh1(i)) || ~isnan(metricThresh2(i))
            if ~isnan(metricThresh1(i)) && ~isnan(metricThresh2(i))
                line(ax, [xLim(1) + binsize_offset, metricThresh1(i) + binsize_offset], [lineY, lineY], 'Color', metricLineCols(i, 1:3), 'LineWidth', 6);
                line(ax, [metricThresh1(i) + binsize_offset, metricThresh2(i) + binsize_offset], [lineY, lineY], 'Color', metricLineCols(i, 4:6), 'LineWidth', 6);
                line(ax, [metricThresh2(i) + binsize_offset, xLim(2) + binsize_offset], [lineY, lineY], 'Color', metricLineCols(i, 7:9), 'LineWidth', 6);
            elseif ~isnan(metricThresh1(i))
                line(ax, [xLim(1) + binsize_offset, metricThresh1(i) + binsize_offset], [lineY, lineY], 'Color', metricLineCols(i, 1:3), 'LineWidth', 6);
                line(ax, [metricThresh1(i) + binsize_offset, xLim(2) + binsize_offset], [lineY, lineY], 'Color', metricLineCols(i, 4:6), 'LineWidth', 6);
            elseif ~isnan(metricThresh2(i))
                line(ax, [xLim(1) + binsize_offset, metricThresh2(i) + binsize_offset], [lineY, lineY], 'Color', metricLineCols(i, 1:3), 'LineWidth', 6);
                line(ax, [metricThresh2(i) + binsize_offset, xLim(2) + binsize_offset], [lineY, lineY], 'Color', metricLineCols(i, 4:6), 'LineWidth', 6);
            end
        end

        if i == 1
            ylabel(ax, 'frac. units', 'FontSize', 12)
        end
        xlabel(ax, metricNames_SHORT{i}, 'FontSize', 11)

        % Add quiver for current unit's value at the bottom
        arrowX = mean(metricData); % Use mean as a placeholder; this will be updated later
        arrowY = lineY - 0.1 * (yLim(2) - yLim(1));
        arrowHandle = quiver(ax, arrowX, arrowY, 0, 0.1*(yLim(2) - yLim(1)), 0, 'MaxHeadSize', 0.5, 'Color', 'k', 'LineWidth', 2);

        % Add red text label on x-axis for the unit's value
        %textY = yLim(1); % Position text directly on the x-axis
        %textHandle = text(ax, arrowX, textY, sprintf('%.4f', arrowX), ...
        %   'HorizontalAlignment', 'center', 'VerticalAlignment', 'top', ...
        %   'FontWeight', 'bold', 'Color', 'r', 'Rotation', 45);

        % Adjust axis limits to accommodate the lines and text
        ylim(ax, [yLim(1) - 0.1 * (yLim(2) - yLim(1)), yLim(2)]);

        % Store the axes handle, arrow handle, and text handle for later updates
        guiData.histogramAxes.(metricName) = ax;
        guiData.histogramArrows.(metricName) = arrowHandle;
        %guiData.histogramTexts.(metricName) = textHandle;
        ax.FontSize = 12;

        axis tight;

        currentSubplot = currentSubplot + 1;
    end
end
guidata(unitQualityGuiHandle.histogramFigure, guiData);

%% Main figure

%% main title
figure(unitQualityGuiHandle.mainFigure);
mainTitle = sgtitle('');

%% initialize and plot units over depth

subplot(6, 13, [1, 14, 27, 40, 53, 66], 'YDir', 'reverse');

hold on;
unitCmap = zeros(length(unitType), 3);
unitCmap(unitType == 1, :) = repmat([0, 0.5, 0], length(find(unitType == 1)), 1);
unitCmap(unitType == 0, :) = repmat([1, 0, 0], length(find(unitType == 0)), 1);
unitCmap(unitType == 2, :) = repmat([1.0000, 0.5469, 0], length(find(unitType == 2)), 1);
unitCmap(unitType == 3, :) = repmat([0.2500, 0.4100, 0.8800], length(find(unitType == 3)), 1);
unitCmap(unitType == 4, :) = repmat([0.2500, 0.4100, 0.8800], length(find(unitType == 4)), 1);
norm_spike_n = mat2gray(log10(accumarray(ephysData.spike_templates, 1)+1));
unitDots = scatter(norm_spike_n(uniqueTemps), ephysData.channel_positions(qMetric.maxChannels, 2), 5, unitCmap, ...
    'filled', 'ButtonDownFcn', @unit_click);
currUnitDots = scatter(0, 0, 100, unitCmap(1, :, :), ...
    'filled', 'MarkerEdgeColor', [0, 0, 0], 'LineWidth', 4);
xlim([-0.1, 1.1]);
ylim([min(ephysData.channel_positions(:, 2)) - 50, max(ephysData.channel_positions(:, 2)) + 50]);
ylabel('Depth from tip of probe (\mum)')
xlabel('Norm. log rate')
%title('Location on probe')

%% initialize template waveforms
if plotRaw
    subplot(6, 13, [2:7, 15:20])
elseif param.extractRaw
    subplot(6, 13, [2:7, 15:20, 28:33])
else
    subplot(6, 13, [2:13, 15:26, 28:39])
end
hold on;
max_n_channels_plot = 20;
templateWaveformLines = arrayfun(@(x) plot(nan(82, 1), nan(82, 1), 'linewidth', 2, 'color', 'k'), 1:max_n_channels_plot);
maxTemplateWaveformLines = arrayfun(@(x) plot(nan(82, 1), nan(82, 1), 'linewidth', 2, 'color', 'b'), 1);
peaks = scatter(nan(10, 1), nan(10, 1), [], prettify_rgb('Orange'), 'v', 'filled');
troughs = scatter(nan(10, 1), nan(10, 1), [], prettify_rgb('Gold'), 'v', 'filled');
%xlabel('Position+Time');
%ylabel('Position');
set(gca, 'YDir', 'reverse')
tempTitle = title('');
tempLegend = legend([maxTemplateWaveformLines, peaks, troughs, ...
    ], {'', '', ''}, 'color', 'none');
tempYLim = gca;
set(gca, 'XTick', [], 'YTick', []);

%% initialize raw waveforms
if param.extractRaw
    if plotRaw
        subplot(6, 13, [8:13, 21:26])
    else
        subplot(6, 13, [8:13, 21:26, 34:39])
    end
    hold on;
    rawWaveformLines = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', 'k'), 1:max_n_channels_plot);
    maxRawWaveformLines = arrayfun(@(x) plot(nan(82, 1), nan(82, 1), 'linewidth', 2, 'color', 'b'), 1);
    set(gca, 'YDir', 'reverse')
    %xlabel('Position+Time');
    %ylabel('Position');
    rawTitle = title('');
    rawLegend = legend(maxRawWaveformLines, {''}, 'color', 'none');
    rawWaveformYLim = gca;
    set(gca, 'XTick', [], 'YTick', []);
end

%% spDeK plot
if plotRaw && param.computeDistanceMetrics
    subplot(6, 13, 29:31)
elseif param.computeDistanceMetrics
    subplot(6, 13, 42:44)
elseif plotRaw
    subplot(6, 13, 29:33)
else
    subplot(6, 13, 42:46)
end
hold on;
spDecayPoints = scatter(NaN, NaN, 'black', 'filled');
spDecayFit = plot(NaN, NaN, 'Color', prettify_rgb('FireBrick'), 'LineWidth', 2);
spDecayLegend = legend(spDecayFit, {''}, 'Location', 'best');
spDecayTitle = title('');
ylabel('ampli. (a.u.)');
%xlabel('distance');

%% initialize ACG
if plotRaw && param.computeDistanceMetrics
    subplot(6, 13, 33:35)
elseif param.computeDistanceMetrics
    subplot(6, 13, 46:49)
elseif plotRaw
    subplot(6, 13, 35:39)
else
    subplot(6, 13, 48:52)
end
hold on;
acgBar = arrayfun(@(x) bar(0:0.1:5, nan(51, 1)), 1);
acgRefLine = line([NaN, NaN], [NaN, NaN], 'Color', 'r', 'linewidth', 1.2);
acgAsyLine = line([NaN, NaN], [NaN, NaN], 'Color', 'r', 'linewidth', 1.2);
xlabel('time (ms)');
ylabel('sp/s');
acgTitle = title('');
acgLegend = legend(acgBar, {''}, 'Location', 'best');

%% initialize ISI

% if plotRaw && param.computeDistanceMetrics
%     subplot(6, 13, 33:35)
% elseif param.computeDistanceMetrics
%     subplot(6, 13, 46:49)
% else
%     subplot(6, 13, 48:52)
% end
%
%
% hold on;
% isiBar = arrayfun(@(x) bar((0 + 0.25):0.5:(50 - 0.25), nan(100, 1)), 1);
% isiRefLine = line([NaN, NaN], [NaN, NaN], 'Color', 'r', 'linewidth', 1.2);
% xlabel('Interspike interval (ms)')
% ylabel('# of spikes')
% isiTitle = title('');
% acgLegend = legend(isiBar, {''});

%% initialize isoDistance
% if param.computeDistanceMetrics %temporarily disabled this plot - better plot coming soon!
%     if plotRaw
%         subplot(6, 13, 36:39)
%     else
%         subplot(6, 13, 41:46)
%     end
%     hold on;
%     currIsoD = scatter(NaN, NaN, 10, '.b'); % Scatter plot with points of size 10
%     rpvIsoD = scatter(NaN, NaN, 10, '.m'); % Scatter plot with points of size 10
%     otherIsoD = scatter(NaN, NaN, 10, NaN, 'o', 'filled');
%
%     colormap(brewermap([], '*YlOrRd'))
%     hb = colorbar;
%     ylabel(hb, 'Mahalanobis Distance')
%     legend('this cluster', 'rpv spikes', 'other clusters');
%     isoDTitle = title('');
% end

%% initialize raw data
if plotRaw
    rawPlotH = subplot(6, 13, [41:52, 55:59, 60:65]);
    hold on;
    title('Raw unwhitened data')
    set(rawPlotH, 'XColor', 'w', 'YColor', 'w')
    rawPlotLines = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', 'k'), 1:max_n_channels_plot);
    rawSpikeLines = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', 'b'), 1:max_n_channels_plot);

end

%% initialize amplitude * spikes
if plotRaw
    ampliAx = subplot(6, 13, [68:70, 74:76]);
else
    ampliAx = subplot(6, 13, [55:57, 68:70, 74:76]);
end
hold on;
yyaxis left;
tempAmpli = scatter(NaN, NaN, 'black', 'filled');
currTempAmpli = scatter(NaN, NaN, 'blue', 'filled');
rpvAmpli = scatter(NaN, NaN, 10, [1.0000, 0.5469, 0], 'filled');
ampliLine = line([NaN, NaN], [NaN, NaN], 'LineWidth', 2.0, 'Color', [0, 0.5, 0]);
%timeChunkLines = arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', 'k'), 1:param.deltaTimeChunk);
xlabel('Experiment time (s)');
ylabel('Template scaling');
axis tight
hold on;
set(gca, 'YColor', 'k')
yyaxis right
spikeFR = stairs(NaN, NaN, 'LineWidth', 2.0, 'Color', [1, 0.5, 0]);
set(gca, 'YColor', [1, 0.5, 0])
ylabel('Firing rate (sp/sec)');

ampliTitle = title('');
ampliLegend = legend([tempAmpli, rpvAmpli, ampliLine], {'', '', ''}, 'Location', 'best');

%% initialize amplitude fit
ampliFitAx = subplot(6, 13, [78]);
hold on;
ampliBins = barh(NaN, NaN, 'blue');
ampliBins.FaceAlpha = 0.5;
ampliFit = plot(NaN, NaN, 'Color', prettify_rgb('Orange'), 'LineWidth', 2);
ampliFitTitle = title('');
ampliFitLegend = legend(ampliFit, {''}, 'Location', 'best');

%% save all handles

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
guiData.tempYLim = tempYLim;

% raw waveforms
if param.extractRaw
    guiData.rawWaveformLines = rawWaveformLines;
    guiData.maxRawWaveformLines = maxRawWaveformLines;
    guiData.rawTitle = rawTitle;
    guiData.rawLegend = rawLegend;
    guiData.rawWaveformYLim = rawWaveformYLim;
end

% ACG
guiData.acgBar = acgBar;
guiData.acgRefLine = acgRefLine;
guiData.acgAsyLine = acgAsyLine;
guiData.acgTitle = acgTitle;
guiData.acgLegend = acgLegend;
%Spatial decay
guiData.spDecayPoints = spDecayPoints;
guiData.spDecayFit = spDecayFit;
guiData.spDecayLegend = spDecayLegend;
guiData.spDecayTitle = spDecayTitle;


% ISI
% guiData.isiBar = isiBar;
% guiData.isiRefLine = isiRefLine;
% guiData.isiTitle = isiTitle;
% guiData.isiLegend = acgLegend;
% isoD
% if param.computeDistanceMetrics
%     guiData.currIsoD = currIsoD;
%     guiData.otherIsoD = otherIsoD;
%     guiData.isoDTitle = isoDTitle;
%     guiData.rpvIsoD = rpvIsoD;
% end
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
guiData.ampliLine = ampliLine;
% amplitude fit
guiData.ampliFitAx = ampliFitAx;
guiData.ampliBins = ampliBins;
guiData.ampliFit = ampliFit;
guiData.ampliFitTitle = ampliFitTitle;
guiData.ampliFitLegend = ampliFitLegend;
guiData.rpvAmpli = rpvAmpli;
% upload guiData
guidata(unitQualityGuiHandle.mainFigure, guiData);
end

function updateUnit(unitQualityGuiHandle, memMapData, ephysData, rawWaveforms, iCluster, qMetric, forGUI, param, ...
    probeLocation, unitType, uniqueTemps, iChunk, plotRaw)

%% Get guidata
guiData = guidata(unitQualityGuiHandle.mainFigure);
thisUnit = uniqueTemps(iCluster);
colorsGdBad = [1, 0, 0; 0, 0.5, 0];
colorsSomatic = [0.25, 0.41, 0.88; 0, 0.5, 0; 0.25, 0.41, 0.88];

%% main title
if unitType(iCluster) == 1
    set(guiData.mainTitle, 'String', ['Unit ', num2str(num2str(iCluster)), ...
        ' (phy ID #: ', num2str(qMetric.phy_clusterID(iCluster)), '; qMetric row #: ', num2str(iCluster), '), single unit'], 'Color', [0, .5, 0]);
elseif unitType(iCluster) == 0
    set(guiData.mainTitle, 'String', ['Unit ', num2str(num2str(iCluster)), ...
        ' (phy ID #: ', num2str(qMetric.phy_clusterID(iCluster)), '; qMetric row #: ', num2str(iCluster), '), noise'], 'Color', [1.0000, 0, 0]);
elseif unitType(iCluster) == 2
    set(guiData.mainTitle, 'String', ['Unit ', num2str(num2str(iCluster)), ...
        ' (phy ID #: ', num2str(qMetric.phy_clusterID(iCluster)), '; qMetric row #: ', num2str(iCluster), '), multi-unit'], 'Color', [1.0000, 0.5469, 0]);
elseif unitType(iCluster) == 3 && param.splitGoodAndMua_NonSomatic
    set(guiData.mainTitle, 'String', ['Unit ', num2str(num2str(iCluster)), ...
        ' (phy ID #: ', num2str(qMetric.phy_clusterID(iCluster)), '; qMetric row #: ', num2str(iCluster), '), non-somatic single unit'], 'Color', [0.25, 0.41, 0.88]);
elseif unitType(iCluster) == 3
    set(guiData.mainTitle, 'String', ['Unit ', num2str(num2str(iCluster)), ...
        ' (phy ID #: ', num2str(qMetric.phy_clusterID(iCluster)), '; qMetric row #: ', num2str(iCluster), '), non-somatic unit'], 'Color', [0.25, 0.41, 0.88]);
elseif unitType(iCluster) == 4
    set(guiData.mainTitle, 'String', ['Unit ', num2str(num2str(iCluster)), ...
        ' (phy ID #: ', num2str(qMetric.phy_clusterID(iCluster)), '; qMetric row #: ', num2str(iCluster), '), non-somatic multi unit'], 'Color', [0.54, 0, 0.54]);
end

%% plot 1: update curr unit location
set(guiData.currUnitDots, 'XData', guiData.norm_spike_n(thisUnit), 'YData', ephysData.channel_positions(qMetric.maxChannels(iCluster), 2), 'CData', guiData.unitCmap(iCluster, :))

for iCh = 1:20
    set(guiData.templateWaveformLines(iCh), 'XData', nan(82, 1), 'YData', nan(82, 1))
    if param.extractRaw
        set(guiData.rawWaveformLines(iCh), 'XData', nan(82, 1), 'YData', nan(82, 1))
    end
end

%% plot 2: update unit template waveform and detected peaks
% guiData.templateWaveformLines = templateWaveformLines;
%     guiData.maxTemplateWaveformLines = maxTemplateWaveformLines;
%     guiData.tempTitle = tempTitle;
%     guiData.tempLegend = tempLegend;

maxChan = qMetric.maxChannels(iCluster);
maxXC = ephysData.channel_positions(maxChan, 1);
maxYC = ephysData.channel_positions(maxChan, 2);
chanDistances = ((ephysData.channel_positions(:, 1) - maxXC).^2 ...
    +(ephysData.channel_positions(:, 2) - maxYC).^2).^0.5;
chansToPlot = find(chanDistances < 100); % should take into account "bad" channels here

% Calculate scaling factor based on the range of the maximum channel
maxChannelWaveform = squeeze(ephysData.templates(thisUnit, :, maxChan));
scalingFactor = range(-maxChannelWaveform) * 3; % Increased from 2.5 to 3 for better visibility

vals = zeros(min(20, size(chansToPlot, 1)), size(ephysData.templates, 2));

for iChanToPlot = 1:min(20, size(chansToPlot, 1))
    if chansToPlot(iChanToPlot) > size(ephysData.templates, 3)
        continue;
    end
    thisChannelWaveform = squeeze(ephysData.templates(thisUnit, :, chansToPlot(iChanToPlot)))';

    vals(iChanToPlot, :) = -thisChannelWaveform + (ephysData.channel_positions(chansToPlot(iChanToPlot), 2) / 100 * scalingFactor);

    if maxChan == chansToPlot(iChanToPlot)
        set(guiData.maxTemplateWaveformLines, 'XData', (ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            'YData', vals(iChanToPlot, :));

        % Update peaks and troughs
        set(guiData.peaks, 'XData', (ephysData.waveform_t(forGUI.peakLocs{iCluster}) + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            'YData', vals(iChanToPlot, forGUI.peakLocs{iCluster}));
        set(guiData.troughs, 'XData', (ephysData.waveform_t(forGUI.troughLocs{iCluster}) + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            'YData', vals(iChanToPlot, forGUI.troughLocs{iCluster}));

        set(guiData.templateWaveformLines(iChanToPlot), 'XData', nan(82, 1), 'YData', nan(82, 1));
    else
        set(guiData.templateWaveformLines(iChanToPlot), 'XData', (ephysData.waveform_t + (ephysData.channel_positions(chansToPlot(iChanToPlot), 1) - 11) / 10), ...
            'YData', vals(iChanToPlot, :));
    end
end

% Adjust y-axis limits to prevent overlap with legend
yMin = min(min(vals, [], 2));
yMax = max(max(vals, [], 2)) + 0.5 * range(vals(:)); % Extra space for legend
try
    ylim(guiData.tempYLim, [yMin, yMax]);
catch
end
if any(~isnan(qMetric.spatialDecaySlope))
    param.computeSpatialDecay = 1;
else
    param.computeSpatialDecay = 0;
end


% Calculate all conditions first for better readability
peaksTroughsCondition = qMetric.nPeaks(iCluster) <= param.maxNPeaks & ...
    qMetric.nTroughs(iCluster) <= param.maxNTroughs;
if param.computeSpatialDecay && param.spDecayLinFit
    spatialDecayCondition = qMetric.spatialDecaySlope(iCluster) >= param.minSpatialDecaySlope;
elseif param.computeSpatialDecay
    spatialDecayCondition = qMetric.spatialDecaySlope(iCluster) >= param.minSpatialDecaySlopeExp & qMetric.spatialDecaySlope(iCluster) <= param.maxSpatialDecaySlopeExp;
end
baselineFlatnessCondition = qMetric.waveformBaselineFlatness(iCluster) <= param.maxWvBaselineFraction;

durationCondition = qMetric.waveformDuration_peakTrough(iCluster) >= param.minWvDuration & ...
    qMetric.waveformDuration_peakTrough(iCluster) <= param.maxWvDuration;

scndPeakRatioCondition = abs(qMetric.scndPeakToTroughRatio(iCluster)) <= param.maxScndPeakToTroughRatio_noise;

somaticFeaturesCondition = abs(qMetric.troughToPeak2Ratio(iCluster)) < param.minTroughToPeak2Ratio_nonSomatic & ...
    qMetric.mainPeak_before_width(iCluster) < param.minWidthFirstPeak_nonSomatic & ...
    qMetric.mainTrough_width(iCluster) < param.minWidthMainTrough_nonSomatic & ...
    abs(qMetric.peak1ToPeak2Ratio(iCluster)) > param.maxPeak1ToPeak2Ratio_nonSomatic;

mainPeakRatioCondition = abs(qMetric.mainPeakToTroughRatio(iCluster)) <= param.maxMainPeakToTroughRatio_nonSomatic;

% Get color strings
peaksTroughsColor = num2str(colorsGdBad(double(peaksTroughsCondition)+1, :));
baselineFlatnessColor = num2str(colorsGdBad(double(baselineFlatnessCondition)+1, :));
durationColor = num2str(colorsGdBad(double(durationCondition)+1, :));
scndPeakRatioColor = num2str(colorsGdBad(double(scndPeakRatioCondition)+1, :));
somaticFeaturesColor = num2str(colorsSomatic(double(somaticFeaturesCondition)+2, :));
mainPeakRatioColor = num2str(colorsSomatic(double(mainPeakRatioCondition)+1, :));
if param.computeSpatialDecay
    spatialDecayColor = num2str(colorsGdBad(double(spatialDecayCondition)+1, :));
end
% Set the title with all components
if param.computeSpatialDecay
    tempWvTitleText = ['\\fontsize{9}Template waveform: {\\color[rgb]{%s}# peaks/troughs, ', newline, ...
        '\\color[rgb]{%s}spatial decay, ', '\\color[rgb]{%s}baseline flatness, ', ...
        '\\color[rgb]{%s}waveform duration', newline, '\\color[rgb]{%s}peak_2/trough,  ', ...
        '\\color[rgb]{%s}peak_1/peak_2, \\color[rgb]{%s}peak_{main}/trough}'];

    set(guiData.tempTitle, 'String', sprintf(tempWvTitleText, ...
        peaksTroughsColor, ...
        spatialDecayColor, ...
        baselineFlatnessColor, ...
        durationColor, ...
        scndPeakRatioColor, ...
        somaticFeaturesColor, ...
        mainPeakRatioColor));
else
    tempWvTitleText = ['\\fontsize{9}Template waveform: {\\color[rgb]{%s}# peaks/troughs, ', newline, ...
        '\\color[rgb]{%s}baseline flatness, ', ...
        '\\color[rgb]{%s}waveform duration', newline, '\\color[rgb]{%s}peak_2/trough,  ', ...
        '\\color[rgb]{%s}peak_1/peak_2, \\color[rgb]{%s}peak_{main}/trough}'];

    set(guiData.tempTitle, 'String', sprintf(tempWvTitleText, ...
        peaksTroughsColor, ...
        baselineFlatnessColor, ...
        durationColor, ...
        scndPeakRatioColor, ...
        somaticFeaturesColor, ...
        mainPeakRatioColor));
end


if param.computeSpatialDecay
    set(guiData.tempLegend, 'String', {['peak_2/trough =', num2str(abs(qMetric.scndPeakToTroughRatio(iCluster))), newline, ...
        'spatial decay =', num2str(qMetric.spatialDecaySlope(iCluster)), newline, ...
        'baseline flatness =', num2str(qMetric.waveformBaselineFlatness(iCluster)), newline, ...
        'waveform duration =', num2str(qMetric.waveformDuration_peakTrough(iCluster))], ...
        ['peak_1/ peak_2 =', num2str(abs(qMetric.peak1ToPeak2Ratio(iCluster)))], ...
        ['peak_{main}/trough =', num2str(abs(qMetric.mainPeakToTroughRatio(iCluster)))], ...
        })
else
    set(guiData.tempLegend, 'String', {['peak_2/trough =', num2str(abs(qMetric.scndPeakToTroughRatio(iCluster))), newline, ...
        'baseline flatness =', num2str(qMetric.waveformBaselineFlatness(iCluster)), newline, ...
        'waveform duration =', num2str(qMetric.waveformDuration_peakTrough(iCluster))], ...
        ['peak_1/ peak_2 =', num2str(abs(qMetric.peak1ToPeak2Ratio(iCluster)))], ...
        ['peak_{main}/trough =', num2str(abs(qMetric.mainPeakToTroughRatio(iCluster)))], ...
        })
end
set(guiData.tempLegend, 'Location', 'southwest');

%% plot 3: plot unit mean raw waveform (and individual traces)
if param.extractRaw
    maxChanRaw = rawWaveforms.peakChan(iCluster);
    chansToPlotRaw = chansToPlot + diff([maxChan, maxChanRaw]);
    chansToPlotRaw(chansToPlotRaw < 1) = [];
    chansToPlotRaw(chansToPlotRaw > size(rawWaveforms.average, 2)) = [];
    chansToPlotRaw(chansToPlotRaw > size(ephysData.channel_positions, 1)) = [];


    % Calculate scaling factor for raw waveforms
    maxRawWaveform = squeeze(rawWaveforms.average(iCluster, maxChanRaw, :));
    rawScalingFactor = range(-maxRawWaveform) * 3; %// Increased from 2.5 to 3 for better visibility

    valsRaw = zeros(min(20, size(chansToPlotRaw, 1)), size(rawWaveforms.average, 3));

    for iChanToPlot = 1:min(20, size(chansToPlotRaw, 1))
        thisRawWaveform = squeeze(rawWaveforms.average(iCluster, chansToPlotRaw(iChanToPlot), :))';
        valsRaw(iChanToPlot, :) = -thisRawWaveform + (ephysData.channel_positions(chansToPlotRaw(iChanToPlot), 2) / 100 * rawScalingFactor);

        if maxChan + diff([maxChan, maxChanRaw]) == chansToPlotRaw(iChanToPlot)
            set(guiData.maxRawWaveformLines, 'XData', (ephysData.waveform_t + (ephysData.channel_positions(chansToPlotRaw(iChanToPlot), 1) - 11) / 10), ...
                'YData', valsRaw(iChanToPlot, :));
            set(guiData.rawWaveformLines(iChanToPlot), 'XData', nan(82, 1), 'YData', nan(82, 1));
        else
            set(guiData.rawWaveformLines(iChanToPlot), 'XData', (ephysData.waveform_t + (ephysData.channel_positions(chansToPlotRaw(iChanToPlot), 1) - 11) / 10), ...
                'YData', valsRaw(iChanToPlot, :));
        end
    end

    % Adjust y-axis limits for raw waveforms to prevent overlap with legend
    yMinRaw = min(min(valsRaw, [], 2));
    yMaxRaw = max(max(valsRaw, [], 2)) + 0.5 * range(valsRaw(:)); %// Extra space for legend
    try
        ylim(guiData.rawWaveformYLim, [yMinRaw, yMaxRaw]);
    catch
    end

    set(guiData.rawLegend, 'String', ['Amplitude =', num2str(round(qMetric.rawAmplitude(iCluster))), 'uV', newline, ...
        'SNR =', num2str(qMetric.signalToNoiseRatio(iCluster))], 'Location', 'southwest')
    if ~isnan(qMetric.rawAmplitude)
        if qMetric.rawAmplitude(iCluster) >= param.minAmplitude && qMetric.signalToNoiseRatio(iCluster) >= param.minSNR
            set(guiData.rawTitle, 'String', ['\color[rgb]{0 0 0}Mean raw waveform: \color[rgb]{0 0.5 0} amplitude, \color[rgb]{0 0.5 0} SNR ']);
        elseif qMetric.rawAmplitude(iCluster) > param.minAmplitude
            set(guiData.rawTitle, 'String', ['\color[rgb]{0 0 0}Mean raw waveform: \color[rgb]{0 0.5 0} amplitude, \color[rgb]{1 0 0} SNR ']);
        elseif qMetric.signalToNoiseRatio(iCluster) >= param.minSNR
            set(guiData.rawTitle, 'String', ['\color[rgb]{0 0 0}Mean raw waveform: \color[rgb]{1 0 0} amplitude, \color[rgb]{0 0.5 0} SNR ']);
        else
            set(guiData.rawTitle, 'String', ['\color[rgb]{0 0 0}Mean raw waveform: \color[rgb]{1 0 0} amplitude, \color[rgb]{1 0 0} SNR ']);
        end
    else
        if qMetric.signalToNoiseRatio(iCluster) >= param.minSNR
            set(guiData.rawTitle, 'String', ['\color[rgb]{0 0 0}Mean raw waveform: \color[rgb]{0 0 0} amplitude, \color[rgb]{0 0.5 0} SNR ']);

        else
            set(guiData.rawTitle, 'String', ['\color[rgb]{0 0 0}Mean raw waveform: \color[rgb]{0 0 0} amplitude, \color[rgb]{1 0 0} SNR ']);
        end
    end
    % if ~isempty(vals)
    %     try
    %     if any(vals > 0 )
    %         ylim(guiData.rawWaveformYLim, [min(min(vals,[],2)), max(max(vals,[],2))+1]) % space for legend
    %     else
    %         ylim(guiData.rawWaveformYLim, [max(max(vals,[],2)), min(min(vals,[],2))+1]) % space for legend
    %     end
    %     catch
    %     end
    % end


end
%set(guiData.rawLegend, ');

%% 4. plot unit ACG
tauR_values = (param.tauR_valuesMin:param.tauR_valuesStep:param.tauR_valuesMax) .* 1000;
theseSpikeTimes = ephysData.spike_times(ephysData.spike_templates == thisUnit);

[ccg, ccg_t] = bc.ep.helpers.CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
    ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', 0.001, 'duration', 0.5, 'norm', 'rate'); %function

set(guiData.acgBar, 'XData', ccg_t(251:301)*1000, 'YData', squeeze(ccg(251:301, 1, 1)));
set(guiData.acgRefLine, 'XData', [tauR_values(qMetric.RPV_window_index(iCluster)), ...
    tauR_values(qMetric.RPV_window_index(iCluster))], 'YData', [0, max(ccg(:, 1, 1))])
[ccg2, ~] = bc.ep.helpers.CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
    ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', 0.1, 'duration', 10, 'norm', 'rate'); %function
asymptoteLine = nanmean(ccg2(end-100:end));
set(guiData.acgAsyLine, 'XData', [0, 50], 'YData', [asymptoteLine, asymptoteLine])

if qMetric.fractionRPVs_estimatedTauR(iCluster) > param.maxRPVviolations
    set(guiData.acgTitle, 'String', '\color[rgb]{1.0000 0.5469 0}ACG');
else
    set(guiData.acgTitle, 'String', '\color[rgb]{0 .5 0}ACG');
end
set(guiData.acgLegend, 'String', [num2str(qMetric.fractionRPVs_estimatedTauR(iCluster)*100), ' % r.p.v.'])
% %% 5. plot unit ISI (with refractory period and asymptote lines)
%
% theseISI = diff(theseSpikeTimes);
% theseISIclean = theseISI(theseISI >= param.tauC); % removed duplicate spikes
% theseOffendingSpikes = find(theseISIclean < (2 / 1000));
%
% %theseOffendingSpikes = [theseOffendingSpikes; theseOffendingSpikes-1];
% [isiProba, edgesISI] = histcounts(theseISIclean*1000, [0:0.5:50]);
%
% set(guiData.isiBar, 'XData', edgesISI(1:end-1)+mean(diff(edgesISI)), 'YData', isiProba); %Check FR
% set(guiData.isiRefLine, 'XData', [tauR_values(qMetric.RPV_tauR_estimate(iCluster)), ...
%     tauR_values(qMetric.RPV_tauR_estimate(iCluster))], 'YData', [0, max(isiProba)])
%
% if qMetric.fractionRPVs_estimatedTauR(iCluster) > param.maxRPVviolations
%     set(guiData.isiTitle, 'String', '\color[rgb]{1 0 1}ISI');
% else
%     set(guiData.isiTitle, 'String', '\color[rgb]{0 .5 0}ISI');
% end
% set(guiData.isiLegend, 'String', [num2str(qMetric.fractionRPVs_estimatedTauR(iCluster)*100), ' % r.p.v.'])

%% 5. plot spatial decay
if param.computeSpatialDecay
    set(guiData.spDecayLegend, 'String', {['spatial decay slope =', num2str(qMetric.spatialDecaySlope(iCluster))]})


    %forGUI.
    set(guiData.spDecayPoints, 'XData', forGUI.spatialDecayPoints_loc(iCluster, :), 'YData', forGUI.spatialDecayPoints(iCluster, :))

    tempspDecayTitleText = ['\\fontsize{9}\\color[rgb]{%s}spatial decay'];

    if param.spDecayLinFit
        % plot(forGUI.spatialDecayPoints_loc(iCluster,:), forGUI.spatialDecayPoints_loc(iCluster,:)*qMetric.spatialDecaySlope(iCluster)+forGUI.spatialDecayFit(iCluster),... '-', 'Color', colorMtx(2, :, :));
    else
        % Generate points for the exponential fit curve
        fitX = linspace(min(forGUI.spatialDecayPoints_loc(iCluster, :)), max(forGUI.spatialDecayPoints_loc(iCluster, :)), 100);
        spatialDecayFitFun = @(x) forGUI.spatialDecayFit(iCluster) * exp(-qMetric.spatialDecaySlope(iCluster)*x);
        fitY = spatialDecayFitFun(fitX);
        set(guiData.spDecayFit, 'XData', fitX, 'YData', fitY)
    end

    if param.spDecayLinFit
        set(guiData.spDecayTitle, 'String', sprintf(tempspDecayTitleText, ...
            num2str(colorsGdBad(double(qMetric.spatialDecaySlope(iCluster) <= param.minSpatialDecaySlope)+1, :))));
    else
        set(guiData.spDecayTitle, 'String', sprintf(tempspDecayTitleText, ...
            num2str(colorsGdBad(double(qMetric.spatialDecaySlope(iCluster) >= param.minSpatialDecaySlopeExp && qMetric.spatialDecaySlope(iCluster) <= param.maxSpatialDecaySlopeExp)+1, :))));
    end


end

%% 6. plot isolation distance
% if param.computeDistanceMetrics
%     set(guiData.currIsoD, 'XData', forGUI.Xplot{iCluster}(:, 1), 'YData', forGUI.Xplot{iCluster}(:, 2))
%     set(guiData.rpvIsoD, 'XData', forGUI.Xplot{iCluster}(theseOffendingSpikes, 1), 'YData', forGUI.Xplot{iCluster}(theseOffendingSpikes, 2))
%     set(guiData.otherIsoD, 'XData', forGUI.Yplot{iCluster}(:, 1), 'YData', forGUI.Yplot{iCluster}(:, 2), 'CData', forGUI.d2_mahal{iCluster})
% end

%% 7. (optional) plot raster

%% 10. plot ampli fit

if ~isnan(forGUI.ampliBinCenters{iCluster})
    set(guiData.ampliBins, 'XData', forGUI.ampliBinCenters{iCluster}, 'YData', forGUI.ampliBinCounts{iCluster});

    set(guiData.ampliFit, 'XData', forGUI.ampliGaussianFit{iCluster}, 'YData', forGUI.ampliBinCenters{iCluster})
    if qMetric.percentageSpikesMissing_gaussian(iCluster) > param.maxPercSpikesMissing
        set(guiData.ampliFitTitle, 'String', '\color[rgb]{1.0000 0.5469 0}% missing');
    else
        set(guiData.ampliFitTitle, 'String', '\color[rgb]{0 .5 0}% missing');
    end
    set(guiData.ampliFitLegend, 'String', {[num2str(qMetric.percentageSpikesMissing_gaussian(iCluster)), ' % missing'], 'rpv spikes'})
    set(guiData.ampliFitAx, 'YLim', [min(forGUI.ampliBinCenters{iCluster}), max(forGUI.ampliBinCenters{iCluster})])
else
    set(guiData.ampliBins, 'XData', forGUI.ampliBinCenters{iCluster}, 'YData', forGUI.ampliBinCounts{iCluster});
    set(guiData.ampliFit, 'XData', forGUI.ampliGaussianFit{iCluster}, 'YData', forGUI.ampliBinCenters{iCluster})
    set(guiData.ampliFitTitle, 'String', '\color[rgb]{1.0000 0.5469 0}% spikes missing');
    set(guiData.ampliFitLegend, 'String', {[num2str(qMetric.percentageSpikesMissing_gaussian(iCluster)), ' % missing'], 'rpv spikes'})

end

%% 9. plot template amplitudes and mean f.r. over recording (QQ: add experiment time epochs?)

ephysData.recordingDuration = (max(ephysData.spike_times) - min(ephysData.spike_times));
theseAmplis = ephysData.template_amplitudes(ephysData.spike_templates == thisUnit);
theseISI = diff(theseSpikeTimes);
theseISIclean = theseISI(theseISI >= param.tauC); % removed duplicate spikes
theseOffendingSpikes = find(theseISIclean < (2 / 1000));
% for debugging if wierd amplitude fit results: percSpikesMissing(theseAmplis, theseSpikeTimes, [min(theseSpikeTimes), max(theseSpikeTimes)], 1);

set(guiData.tempAmpli, 'XData', theseSpikeTimes, 'YData', theseAmplis)
set(guiData.rpvAmpli, 'XData', theseSpikeTimes(theseOffendingSpikes), 'YData', theseAmplis(theseOffendingSpikes))
if length(theseSpikeTimes) < iChunk
    iChunk = 1;
end
currTimes = theseSpikeTimes(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1 & theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
currAmplis = theseAmplis(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1 & theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
set(guiData.currTempAmpli, 'XData', currTimes, 'YData', currAmplis);
try
    set(guiData.ampliAx.YAxis(1), 'Limits', [0, round(nanmax(theseAmplis))])
catch
end

binSize = 20;
timeBins = 0:binSize:ceil(ephysData.spike_times(end));
while length(timeBins) == 1
    binSize = binSize / 2;
    timeBins = 0:binSize:ceil(ephysData.spike_times(end));
end
[n, x] = hist(theseSpikeTimes, timeBins);
n = n ./ binSize;

set(guiData.spikeFR, 'XData', x, 'YData', n);
set(guiData.ampliAx.YAxis(2), 'Limits', [0, 2 * ceil(max(n))])


set(guiData.ampliLine, 'XData', [qMetric.useTheseTimesStart(iCluster), qMetric.useTheseTimesStop(iCluster)], ...
    'YData', [max(theseAmplis) * 0.9, max(theseAmplis) * 0.9]);

if qMetric.nSpikes(iCluster) >= param.minNumSpikes && qMetric.presenceRatio(iCluster) >= param.minPresenceRatio
    set(guiData.ampliTitle, 'String', '\color[rgb]{0 0 0}Spikes: \color[rgb]{0 .5 0}number, \color[rgb]{0 .5 0}presence ratio');
elseif qMetric.nSpikes(iCluster) >= param.minNumSpikes
    set(guiData.ampliTitle, 'String', '\color[rgb]{0 0 0}Spikes: \color[rgb]{0 .5 0}number, \color[rgb]{1.0000 0.5469 0}presence ratio');
else
    set(guiData.ampliTitle, 'String', '\color[rgb]{0 0 0}Spikes: \color[rgb]{1.0000 0.5469 0}number, \color[rgb]{1.0000 0.5469 0}presence ratio');

end
set(guiData.ampliLegend, 'String', {['# spikes = ', num2str(qMetric.nSpikes(iCluster)), newline, ...
    'presence ratio = ', num2str(qMetric.presenceRatio(iCluster))], 'rpv spikes', ...
    ' ''good'' time chunks'})

%% 8. plot raw data
if plotRaw
    plotSubRaw(guiData.rawPlotH, guiData.rawPlotLines, guiData.rawSpikeLines, memMapData, ephysData, iCluster, uniqueTemps, iChunk);
end

%% 9. update hg plot
try
guiData2 = guidata(unitQualityGuiHandle.histogramFigure);
% Update histogram figure
figure(unitQualityGuiHandle.histogramFigure);
metricNames = fieldnames(guiData2.histogramAxes);
for i = 1:length(metricNames)
    metricName = metricNames{i};
    ax = guiData2.histogramAxes.(metricName);
    arrowHandle = guiData2.histogramArrows.(metricName);
    %    textHandle = guiData2.histogramTexts.(metricName);

    % Get current unit's value for this metric
    unitValue = qMetric.(metricName)(iCluster);

    % Get axis limits
    yLim = get(ax, 'YLim');
    xLim = get(ax, 'XLim');

    % Update arrow position and size
    %lineY =  ax.Children(3).YData(1) ; % Position of horizontal lines
    %arrowY = lineY - lineY * 0.01; % Start position of arrow slightly below the horizontal lines
    %arrowHeight = lineY; % Increase the height of the arrow
    set(arrowHandle, 'XData', unitValue, 'UData', 0);

    % Update text position and value
    %textY = arrowY + arrowHeight + 0.02 * (yLim(2) - yLim(1)); % Position text above the arrow
    %set(textHandle, 'Position', [unitValue, textY, 0]);

    % Format value string without trailing zeros
    %valueStr = sprintf('%.4f', unitValue);
    %valueStr = regexprep(valueStr, '0+$', ''); % Remove trailing zeros
    %valueStr = regexprep(valueStr, '\.$', ''); % Remove trailing decimal point if all zeros were removed
    %set(textHandle, 'String', valueStr);

    % Ensure the unit value is within the visible range
    %if unitValue < xLim(1) || unitValue > xLim(2)
    %    newXLim = [min(xLim(1), unitValue), max(xLim(2), unitValue)];
    %    xlim(ax, newXLim);
    %end

    % Adjust y-axis limits to accommodate taller quiver and text
    %newYLim = [yLim(1) - 0.15 * (yLim(2) - yLim(1)), yLim(2)];
    %ylim(ax, newYLim);
end

catch % figure was closed by user
    
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
    theseSpikeTimes = ephysData.spike_times(ephysData.spike_templates == thisUnit);
    ephysData.recordingDuration = (max(ephysData.spike_times) - min(ephysData.spike_times));
    theseAmplis = ephysData.template_amplitudes(ephysData.spike_templates == thisUnit);
    set(guiData.tempAmpli, 'XData', theseSpikeTimes, 'YData', theseAmplis)
    currTimes = theseSpikeTimes(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1 & theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
    currAmplis = theseAmplis(theseSpikeTimes >= theseSpikeTimes(iChunk)-0.1 & theseSpikeTimes <= theseSpikeTimes(iChunk)+0.1);
    set(guiData.currTempAmpli, 'XData', currTimes, 'YData', currAmplis);
    set(guiData.ampliAx.YAxis(1), 'Limits', [0, round(max(theseAmplis))])

    binSize = 20;
    timeBins = 0:binSize:ceil(ephysData.spike_times_samples(end)/ephysData.ephys_sample_rate);
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
if length(theseTimesCenter) > 10 + iChunk
    firstSpike = theseTimesCenter(iChunk+10) - 0.05; %tenth spike occurance %
else
    firstSpike = theseTimesCenter(iChunk) - 0.05; %first spike occurance
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


t = int32(firstSpike*ephysData.ephys_sample_rate):int32((firstSpike + timeToPlot)*ephysData.ephys_sample_rate);
subplot(rawPlotH)
plotidx = int32(firstSpike*ephysData.ephys_sample_rate) ...
    :int32((firstSpike + timeToPlot)*ephysData.ephys_sample_rate);
t(plotidx < 1 | plotidx > size(memMapData, 2)) = [];
plotidx(plotidx < 1 | plotidx > size(memMapData, 2)) = [];
thisMemMap = double(memMapData(chansToPlot, plotidx)) + double(cCount);
for iClear = 1:length(rawSpikeLines)
    set(rawSpikeLines(iClear), 'XData', NaN, 'YData', NaN)
end
if length(rawSpikeLines) < length(chansToPlot)
    rawSpikeLines(end+1:length(chansToPlot)) = rawSpikeLines(end);
    rawPlotLines(end+1:length(chansToPlot)) = rawPlotLines(end);
end
for iChanToPlot = 1:length(chansToPlot)
    set(rawPlotLines(iChanToPlot), 'XData', t, 'YData', thisMemMap(iChanToPlot, :));
    if ~isempty(theseTimesCenter)
        for iTimes = 1:size(theseTimesCenter, 1)
            if ~any(mod(theseTimesFull(iTimes, :), 1))
                set(rawSpikeLines(iChanToPlot), 'XData', theseTimesFull(iTimes, :), 'YData', thisMemMap(iChanToPlot, ...
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

function [metricNames, metricThresh1, metricThresh2, plotConditions, metricNames_SHORT, metricLineCols] = defineMetrics(param)
metricNames = {'nPeaks', 'nTroughs', 'scndPeakToTroughRatio', 'peak1ToPeak2Ratio', 'mainPeakToTroughRatio', ...
    'fractionRPVs_estimatedTauR', 'RPV_window_index', 'percentageSpikesMissing_gaussian', ...
    'percentageSpikesMissing_symmetric', 'nSpikes', 'rawAmplitude', 'spatialDecaySlope', ...
    'waveformDuration_peakTrough', 'waveformBaselineFlatness', 'presenceRatio', 'signalToNoiseRatio', ...
    'maxDriftEstimate', 'cumDriftEstimate', 'isolationDistance', 'Lratio'};


metricNames_SHORT = {'# peaks', '# troughs', 'peak_2/trough', 'peak_1/peak_2', 'peak_{main}/trough', ...
    'frac. RPVs', 'RPV_window_index', '% spikes missing', ...
    '%SpikesMissing-symmetric', '# spikes', 'amplitude', 'spatial decay', ...
    'waveform duration', 'baseline flatness', 'presence ratio', 'SNR', ...
    'maximum drift', 'cum. drift', 'isolation dist.', 'L-ratio'};


if param.spDecayLinFit
    metricThresh1 = [param.maxNPeaks, param.maxNTroughs, param.maxScndPeakToTroughRatio_noise, param.maxPeak1ToPeak2Ratio_nonSomatic, param.maxMainPeakToTroughRatio_nonSomatic, ...
        param.maxRPVviolations, NaN, param.maxPercSpikesMissing, NaN, NaN, NaN, param.minSpatialDecaySlope, ...
        param.minWvDuration, param.maxWvBaselineFraction, NaN, NaN, ...
        param.maxDrift, NaN, param.isoDmin, NaN];


    metricThresh2 = [NaN, NaN, param.maxScndPeakToTroughRatio_noise, NaN, NaN, ...
        NaN, NaN, NaN, NaN, param.minNumSpikes, param.minAmplitude, NaN, ...
        param.maxWvDuration, NaN, param.minPresenceRatio, param.minSNR, ...
        NaN, NaN, NaN, param.lratioMax];


else
    metricThresh1 = [param.maxNPeaks, param.maxNTroughs, param.maxScndPeakToTroughRatio_noise, param.maxPeak1ToPeak2Ratio_nonSomatic, param.maxMainPeakToTroughRatio_nonSomatic, ...
        param.maxRPVviolations, NaN, param.maxPercSpikesMissing, NaN, NaN, NaN, param.minSpatialDecaySlopeExp, ...
        param.minWvDuration, param.maxWvBaselineFraction, NaN, NaN, ...
        param.maxDrift, NaN, param.isoDmin, NaN];


    metricThresh2 = [NaN, NaN, param.maxScndPeakToTroughRatio_noise, NaN, NaN, ...
        NaN, NaN, NaN, NaN, param.minNumSpikes, param.minAmplitude, param.maxSpatialDecaySlopeExp, ...
        param.maxWvDuration, NaN, param.minPresenceRatio, param.minSNR, ...
        NaN, NaN, NaN, param.lratioMax];
end
plotConditions = [true, true, true, true, true, true, ...
    param.tauR_valuesMin ~= param.tauR_valuesMax, ...
    true, false, true, param.extractRaw, ...
    param.computeSpatialDecay == 1, ...
    true, true, true, param.extractRaw, ...
    param.computeDrift, param.computeDrift, ...
    param.computeDistanceMetrics && ~isnan(param.isoDmin), ...
    param.computeDistanceMetrics && ~isnan(param.isoDmin)];


metricLineCols = [0.2, 0.2, 0.2, 1, 0, 0, 0, 0, 0; ... % 1 'nPeaks'
    0.2, 0.2, 0.2, 1, 0, 0, 0, 0, 0; ... % 2 'nTroughs'
    0.2, 0.2, 0.2, 1, 0, 0, 1, 0, 0; ... % 3 'scndPeakToTroughRatio'
    0.2, 0.2, 0.2, 0.25, 0.41, 0.88, 0, 0, 0; ... % 4 'peak1ToPeak2Ratio'
    0.2, 0.2, 0.2, 0.25, 0.41, 0.88, 0, 0, 0; ... % 5 'mainPeakToTroughRatio'
    0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ... % 6 'fractionRPVs_estimatedTauR'
    0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ... % 7 'RPV_tauR_estimate',
    0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ... % 8 'percentageSpikesMissing_gaussian'
    0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ... % 9 'percentageSpikesMissing_symmetric'
    1.0000, 0.5469, 0, 0, 0.5, 0, 0, 0, 0; ... % 10  '# spikes'
    1.0000, 0.5469, 0, 0, 0.5, 0, 0, 0, 0; ... % 11 'amplitude'
    1, 0, 0, 0.2, 0.2, 0.2, 1, 0, 0; ... % 12 'spatial decay'
    1, 0, 0, 0.2, 0.2, 0.2, 1, 0, 0; ... % 13 'waveform duration'
    0.2, 0.2, 0.2, 1, 0, 0, 1, 0, 0; ... % 14 'baseline flatness'
    1.0000, 0.5469, 0, 0, 0.5, 0, 0, 0, 0; ... % 15 'presence ratio'
    1.0000, 0.5469, 0, 0, 0.5, 0, 0, 0, 0; ... % 16 'SNR'
    0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ... % 17 'maximum drift'
    0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ... % 18 'cum. drift'
    1.0000, 0.5469, 0, 0, 0.5, 0, 0, 0, 0; ... % 19 'isolation dist.'
    0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ... % 20 'L-ratio'
    ];

indices_ordered = [1, 2, 14, 13, 3, 12, 4, 5, 11, 16, 6, 10, 15, 8, 17, 18, 19, 20];
metricNames = metricNames(indices_ordered);
metricNames_SHORT = metricNames_SHORT(indices_ordered);
metricThresh1 = metricThresh1(indices_ordered);
metricThresh2 = metricThresh2(indices_ordered);
plotConditions = plotConditions(indices_ordered);
metricLineCols = metricLineCols(indices_ordered, :);
end

