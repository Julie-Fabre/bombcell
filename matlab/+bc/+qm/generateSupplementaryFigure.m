function [fig, methodsText] = generateSupplementaryFigure(exampleDatasetPath, allDatasetPaths, varargin)
% bc.qm.generateSupplementaryFigure - Generate publication-ready supplementary figure
%
% Creates a multi-panel figure with:
%   (a) Histograms of quality metric distributions (using standard BombCell style)
%   (b) Overlaid waveforms by unit type for an example dataset (left, 2/3 width)
%   (c) Bar chart showing mean +/- SE unit type proportions across all datasets (right, 1/3 width)
%
% Also generates a methods section text with actual statistics.
%
% ------
% Inputs
% ------
% exampleDatasetPath : char
%     Path to the BombCell results directory for the example dataset.
%     This dataset is used for panels (a) and (b).
%
% allDatasetPaths : cell array of char
%     Cell array of paths to BombCell results directories for all datasets.
%     These are used to compute statistics in panel (c) and methods text.
%
% Optional name-value pairs:
%   'param' : struct, default []
%       BombCell parameter structure. If empty, loads from exampleDatasetPath.
%   'figWidth' : double, default 14
%       Figure width in inches.
%   'dpi' : double, default 300
%       Resolution for saved figure.
%   'savePath' : char, default ''
%       If provided, saves figure to this path (PNG format).
%   'colorScheme' : struct, default []
%       Custom colors for unit types. Fields should be 'GOOD', 'MUA', 'NOISE', 'NON_SOMA'.
%
% ------
% Outputs
% ------
% fig : figure handle
%     The generated multi-panel figure.
% methodsText : char
%     Ready-to-use methods section text with actual statistics.
%
% ------
% Example
% ------
%   % Single example dataset for detailed panels
%   examplePath = '/path/to/recording1/bombcell';
%
%   % All datasets for statistics
%   allPaths = {
%       '/path/to/recording1/bombcell',
%       '/path/to/recording2/bombcell',
%       '/path/to/recording3/bombcell'
%   };
%
%   % Generate figure and methods text
%   [fig, methodsText] = bc.qm.generateSupplementaryFigure(examplePath, allPaths, ...
%       'savePath', 'supplementary_figure_qc.png');
%
%   % Print methods section
%   disp(methodsText);

    % Parse inputs
    p = inputParser;
    addRequired(p, 'exampleDatasetPath', @ischar);
    addRequired(p, 'allDatasetPaths', @iscell);
    addParameter(p, 'param', [], @(x) isstruct(x) || isempty(x));
    addParameter(p, 'figWidth', 14, @isnumeric);
    addParameter(p, 'dpi', 300, @isnumeric);
    addParameter(p, 'savePath', '', @ischar);
    addParameter(p, 'colorScheme', [], @(x) isstruct(x) || isempty(x));
    parse(p, exampleDatasetPath, allDatasetPaths, varargin{:});

    param = p.Results.param;
    figWidth = p.Results.figWidth;
    dpi = p.Results.dpi;
    savePath = p.Results.savePath;
    colorScheme = p.Results.colorScheme;

    % Default color scheme
    if isempty(colorScheme)
        colorScheme = struct();
        colorScheme.NOISE = [0.545, 0, 0];           % Dark red
        colorScheme.GOOD = [0.133, 0.545, 0.133];    % Forest green
        colorScheme.MUA = [0.855, 0.647, 0.125];     % Goldenrod
        colorScheme.NON_SOMA = [0.255, 0.412, 0.882]; % Royal blue
        colorScheme.NON_SOMA_GOOD = [0.255, 0.412, 0.882];
        colorScheme.NON_SOMA_MUA = [0.529, 0.808, 0.922]; % Light sky blue
    end

    % 1. Load example dataset
    [param, qMetric, templateWaveforms, unitType, uniqueTemplates] = loadDataset(exampleDatasetPath, param);

    % 2. Aggregate statistics across all datasets
    stats = aggregateDatasetStatistics(allDatasetPaths, param);

    % 3. Get metric definitions for histograms
    [qMetric, param] = bc.qm.prettify_names(qMetric, param);
    [metricNames, metricThresh1, metricThresh2, plotConditions, metricNames_SHORT, metricLineCols] = defineMetrics(param);

    % Filter to valid metrics
    validIdx = find(plotConditions);
    nHistMetrics = length(validIdx);
    nHistRows = ceil(nHistMetrics / 4);
    nHistCols = min(4, nHistMetrics);

    % 4. Determine layout
    splitNonSomatic = param.splitGoodAndMua_NonSomatic;
    if splitNonSomatic
        nUnitTypes = 5;
        unitTypeLabels = {'Noise', 'Good', 'MUA', 'Non-soma Good', 'Non-soma MUA'};
    else
        nUnitTypes = 4;
        unitTypeLabels = {'Noise', 'Good', 'MUA', 'Non-somatic'};
    end

    % Calculate figure height based on content
    % Height for histograms (~2.2 inches per row) + waveforms/bar chart (~2.5 inches)
    figHeight = (nHistRows * 2.2) + 2.5;

    % 5. Create figure
    fig = figure('Color', 'w', 'Units', 'inches', 'Position', [1, 1, figWidth, figHeight]);

    % 6. Plot panels: (a) histograms on top, (b) waveforms + (c) bar chart on bottom
    if nHistRows <= 3
        plotTwoRowLayout(fig, qMetric, param, unitType, uniqueTemplates, templateWaveforms, ...
            metricNames, metricThresh1, metricThresh2, plotConditions, metricNames_SHORT, metricLineCols, ...
            stats, colorScheme, unitTypeLabels, nUnitTypes, nHistRows, nHistCols, nHistMetrics);
    else
        plotThreeRowLayout(fig, qMetric, param, unitType, uniqueTemplates, templateWaveforms, ...
            metricNames, metricThresh1, metricThresh2, plotConditions, metricNames_SHORT, metricLineCols, ...
            stats, colorScheme, unitTypeLabels, nUnitTypes, nHistRows, nHistCols, nHistMetrics);
    end

    % 7. Generate methods text
    methodsText = generateMethodsTextWithStats(param, stats, qMetric);

    % 8. Save if requested
    if ~isempty(savePath)
        % Save figure
        print(fig, savePath, '-dpng', sprintf('-r%d', dpi));
        fprintf('Saved supplementary figure to %s\n', savePath);

        % Save methods text
        [pathStr, name, ~] = fileparts(savePath);
        methodsPath = fullfile(pathStr, [name, '_methods.txt']);
        fid = fopen(methodsPath, 'w');
        fprintf(fid, '%s', methodsText);
        fclose(fid);
        fprintf('Saved methods text to %s\n', methodsPath);
    end
end

%% Helper Functions

function [param, qMetric, templateWaveforms, unitType, uniqueTemplates] = loadDataset(dataPath, paramIn)
    % Load BombCell results from a dataset path

    % Load qMetrics
    qMetricPath = fullfile(dataPath, 'templates._bc_qMetrics.parquet');
    if exist(qMetricPath, 'file')
        qMetric = parquetread(qMetricPath);
    else
        error('Could not find qMetrics at %s', qMetricPath);
    end

    % Load parameters
    if isempty(paramIn)
        paramPath = fullfile(dataPath, '_bc_parameters._bc_qMetrics.parquet');
        if exist(paramPath, 'file')
            paramTable = parquetread(paramPath);
            param = table2struct(paramTable);
        else
            error('Could not find parameters at %s', paramPath);
        end
    else
        param = paramIn;
    end
    param.plotGlobal = 1;

    % Get ephys path from param or use dataPath
    if isfield(param, 'ephysKilosortPath')
        ephysPath = param.ephysKilosortPath;
    else
        ephysPath = dataPath;
    end

    % Load template waveforms
    try
        templates = readNPY(fullfile(ephysPath, 'templates.npy'));

        % Get unique templates
        spike_templates_0idx = readNPY(fullfile(ephysPath, 'spike_templates.npy'));
        spikeTemplates = spike_templates_0idx + 1;

        if exist(fullfile(ephysPath, 'spike_clusters.npy'), 'file')
            spike_clusters_0idx = readNPY(fullfile(ephysPath, 'spike_clusters.npy'));
            spikeClusters = int32(spike_clusters_0idx) + 1;
            uniqueTemplates = unique(spikeClusters);
        else
            uniqueTemplates = unique(spikeTemplates);
        end

        % Extract template waveforms on max channel
        [~, maxChannel] = max(max(abs(templates), [], 2), [], 3);
        templateWaveforms = zeros(size(templates, 1), size(templates, 2));
        for i = 1:size(templates, 1)
            templateWaveforms(i, :) = templates(i, :, maxChannel(i));
        end
    catch
        warning('Could not load template waveforms from %s', ephysPath);
        uniqueTemplates = (1:height(qMetric))';
        templateWaveforms = zeros(length(uniqueTemplates), 82);
    end

    % Get unit type
    unitType = bc.qm.getQualityUnitType(param, qMetric, dataPath);
end

function stats = aggregateDatasetStatistics(allDatasetPaths, param)
    % Compute unit type proportions across all datasets

    splitNonSomatic = param.splitGoodAndMua_NonSomatic;
    if splitNonSomatic
        categories = {'NOISE', 'GOOD', 'MUA', 'NON_SOMA_GOOD', 'NON_SOMA_MUA'};
        catCodes = [0, 1, 2, 3, 4];
    else
        categories = {'NOISE', 'GOOD', 'MUA', 'NON_SOMA'};
        catCodes = [0, 1, 2, 3];
    end

    nDatasets = length(allDatasetPaths);
    proportions = struct();
    counts = struct();
    for i = 1:length(categories)
        proportions.(categories{i}) = zeros(1, nDatasets);
        counts.(categories{i}) = zeros(1, nDatasets);
    end
    totalUnits = zeros(1, nDatasets);
    validDatasets = true(1, nDatasets);

    for d = 1:nDatasets
        try
            [~, ~, ~, unitType, ~] = loadDataset(allDatasetPaths{d}, param);
            nUnits = length(unitType);
            totalUnits(d) = nUnits;

            for i = 1:length(categories)
                count = sum(unitType == catCodes(i));
                counts.(categories{i})(d) = count;
                proportions.(categories{i})(d) = count / nUnits;
            end
        catch ME
            warning('Could not load dataset %s: %s', allDatasetPaths{d}, ME.message);
            validDatasets(d) = false;
        end
    end

    % Filter to valid datasets
    totalUnits = totalUnits(validDatasets);
    for i = 1:length(categories)
        proportions.(categories{i}) = proportions.(categories{i})(validDatasets);
        counts.(categories{i}) = counts.(categories{i})(validDatasets);
    end

    % Compute statistics
    stats = struct();
    stats.nDatasets = sum(validDatasets);
    stats.totalUnits = sum(totalUnits);
    stats.categories = categories;

    for i = 1:length(categories)
        cat = categories{i};
        props = proportions.(cat);
        stats.(cat).proportions = props;
        stats.(cat).mean = mean(props);
        stats.(cat).se = std(props) / sqrt(length(props));
        stats.(cat).counts = counts.(cat);
        stats.(cat).totalCount = sum(counts.(cat));
    end
end

function plotTwoRowLayout(fig, qMetric, param, unitType, uniqueTemplates, templateWaveforms, ...
    metricNames, metricThresh1, metricThresh2, plotConditions, metricNames_SHORT, metricLineCols, ...
    stats, colorScheme, unitTypeLabels, nUnitTypes, nHistRows, nHistCols, nHistMetrics)

    % Layout: (a) histograms on top, (b) waveforms + (c) bar chart on bottom
    % Bottom row: waveforms (left, 2/3) + bar chart (right, 1/3)

    nCols = 6;  % Use 6 columns for 2/3 + 1/3 split
    totalRows = nHistRows + 1;

    % Panel (a): Histograms - top rows
    validIdx = find(plotConditions);
    for idx = 1:nHistMetrics
        i = validIdx(idx);
        row = ceil(idx / nHistCols);
        col = mod(idx - 1, nHistCols) + 1;

        % Map to 6-column grid (histograms span all columns, 4 histograms per row)
        % Each histogram takes 1.5 columns worth of space
        histColSpan = floor(nCols / nHistCols);
        startCol = (col - 1) * histColSpan + 1;
        endCol = min(col * histColSpan, nCols);

        subplotIdx = (row - 1) * nCols + startCol;
        ax = subplot(totalRows, nCols, subplotIdx:subplotIdx + histColSpan - 1);
        plotHistogramPanel(ax, qMetric, metricNames{i}, metricThresh1(i), metricThresh2(i), ...
            metricNames_SHORT{i}, metricLineCols(i, :), idx);
        if idx == 1
            text(ax, -0.25, 1.15, 'a', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
        end
    end

    % Bottom row: waveforms (left 2/3, columns 1-4) + bar chart (right 1/3, columns 5-6)
    bottomRowStart = nHistRows * nCols + 1;

    % Panel (b): Waveforms - bottom left (2/3 width = 4 columns)
    waveformCols = 4;
    for i = 1:nUnitTypes
        col = i;
        if col <= waveformCols
            ax = subplot(totalRows, nCols, bottomRowStart + col - 1);
            plotWaveformPanel(ax, qMetric, unitType, uniqueTemplates, templateWaveforms, ...
                i-1, unitTypeLabels{i}, colorScheme, param);
            if i == 1
                text(ax, -0.15, 1.15, 'b', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
            end
        end
    end

    % Panel (c): Bar chart - bottom right (1/3 width = 2 columns)
    ax = subplot(totalRows, nCols, [bottomRowStart + 4, bottomRowStart + 5]);
    plotBarChart(ax, stats, colorScheme, param.splitGoodAndMua_NonSomatic);
end

function plotThreeRowLayout(fig, qMetric, param, unitType, uniqueTemplates, templateWaveforms, ...
    metricNames, metricThresh1, metricThresh2, plotConditions, metricNames_SHORT, metricLineCols, ...
    stats, colorScheme, unitTypeLabels, nUnitTypes, nHistRows, nHistCols, nHistMetrics)

    % Layout: (a) histograms on top, (b) waveforms + (c) bar chart on bottom
    % Same as two-row layout but with more histogram rows

    nCols = 6;  % Use 6 columns for 2/3 + 1/3 split
    totalRows = nHistRows + 1;

    % Panel (a): Histograms - top rows
    validIdx = find(plotConditions);
    for idx = 1:nHistMetrics
        i = validIdx(idx);
        row = ceil(idx / nHistCols);
        col = mod(idx - 1, nHistCols) + 1;

        % Map to 6-column grid
        histColSpan = floor(nCols / nHistCols);
        startCol = (col - 1) * histColSpan + 1;

        subplotIdx = (row - 1) * nCols + startCol;
        ax = subplot(totalRows, nCols, subplotIdx:subplotIdx + histColSpan - 1);
        plotHistogramPanel(ax, qMetric, metricNames{i}, metricThresh1(i), metricThresh2(i), ...
            metricNames_SHORT{i}, metricLineCols(i, :), idx);
        if idx == 1
            text(ax, -0.25, 1.15, 'a', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
        end
    end

    % Bottom row: waveforms (left 2/3) + bar chart (right 1/3)
    bottomRowStart = nHistRows * nCols + 1;

    % Panel (b): Waveforms - bottom left (2/3 width = 4 columns)
    waveformCols = 4;
    for i = 1:nUnitTypes
        col = i;
        if col <= waveformCols
            ax = subplot(totalRows, nCols, bottomRowStart + col - 1);
            plotWaveformPanel(ax, qMetric, unitType, uniqueTemplates, templateWaveforms, ...
                i-1, unitTypeLabels{i}, colorScheme, param);
            if i == 1
                text(ax, -0.15, 1.15, 'b', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
            end
        end
    end

    % Panel (c): Bar chart - bottom right (1/3 width = 2 columns)
    ax = subplot(totalRows, nCols, [bottomRowStart + 4, bottomRowStart + 5]);
    plotBarChart(ax, stats, colorScheme, param.splitGoodAndMua_NonSomatic);
end

function plotWaveformPanel(ax, qMetric, unitType, uniqueTemplates, templateWaveforms, ...
    unitCode, label, colorScheme, param)

    hold(ax, 'on');

    % Get units of this type
    unitMask = (unitType == unitCode);
    unitIds = find(unitMask);
    nUnits = length(unitIds);

    % Get color
    colorFields = {'NOISE', 'GOOD', 'MUA', 'NON_SOMA', 'NON_SOMA_MUA'};
    if unitCode < length(colorFields)
        colorField = colorFields{unitCode + 1};
        if isfield(colorScheme, colorField)
            color = colorScheme.(colorField);
        else
            color = [0, 0, 0];
        end
    else
        color = [0, 0, 0];
    end

    if nUnits > 0
        alpha = max(0.03, min(0.3, 15 / nUnits));
        for j = 1:nUnits
            uid = unitIds(j);
            if uid <= size(templateWaveforms, 1)
                wf = templateWaveforms(uid, :);
                plot(ax, wf, 'Color', [color, alpha], 'LineWidth', 0.8);
            end
        end
    end

    % Clean styling
    set(ax, 'XColor', 'none', 'YColor', 'none');
    box(ax, 'off');
    title(ax, sprintf('%s\n(n = %d)', label, nUnits), 'FontSize', 10, 'FontWeight', 'bold');

    % Set x limits
    if isfield(param, 'spikeWidth') && param.spikeWidth <= 70
        xlim(ax, [1, param.spikeWidth]);
    elseif isfield(param, 'spikeWidth')
        startIdx = max(1, round(21 * param.spikeWidth / 82));
        xlim(ax, [startIdx, param.spikeWidth]);
    end
end

function plotHistogramPanel(ax, qMetric, metricName, thresh1, thresh2, shortName, lineColors, idx)

    hold(ax, 'on');

    % Color matrix for histogram bars
    colorMtx = getHistogramColors();
    colorIdx = mod(idx - 1, size(colorMtx, 1)) + 1;
    barColor = colorMtx(colorIdx, :);

    % Get metric data
    if ismember(metricName, qMetric.Properties.VariableNames)
        metricData = qMetric.(metricName);
        metricData = metricData(~isnan(metricData) & ~isinf(metricData));
    else
        metricData = [];
    end

    if ~isempty(metricData)
        % Plot histogram
        if contains(metricName, 'nPeaks') || contains(metricName, 'nTroughs')
            h = histogram(ax, metricData, 'FaceColor', barColor, 'Normalization', 'probability');
        else
            h = histogram(ax, metricData, 40, 'FaceColor', barColor, 'Normalization', 'probability');
        end
        binsize_offset = h.BinWidth / 2;

        % Add threshold lines below histogram
        yLim = ylim(ax);
        xLim = xlim(ax);
        lineY = yLim(1) - 0.02 * (yLim(2) - yLim(1));

        if ~isnan(thresh1) || ~isnan(thresh2)
            if ~isnan(thresh1) && ~isnan(thresh2)
                line(ax, [xLim(1) + binsize_offset, thresh1 + binsize_offset], [lineY, lineY], ...
                    'Color', lineColors(1:3), 'LineWidth', 6);
                line(ax, [thresh1 + binsize_offset, thresh2 + binsize_offset], [lineY, lineY], ...
                    'Color', lineColors(4:6), 'LineWidth', 6);
                line(ax, [thresh2 + binsize_offset, xLim(2) + binsize_offset], [lineY, lineY], ...
                    'Color', lineColors(7:9), 'LineWidth', 6);
            elseif ~isnan(thresh1)
                line(ax, [xLim(1) + binsize_offset, thresh1 + binsize_offset], [lineY, lineY], ...
                    'Color', lineColors(1:3), 'LineWidth', 6);
                line(ax, [thresh1 + binsize_offset, xLim(2) + binsize_offset], [lineY, lineY], ...
                    'Color', lineColors(4:6), 'LineWidth', 6);
            elseif ~isnan(thresh2)
                line(ax, [xLim(1) + binsize_offset, thresh2 + binsize_offset], [lineY, lineY], ...
                    'Color', lineColors(1:3), 'LineWidth', 6);
                line(ax, [thresh2 + binsize_offset, xLim(2) + binsize_offset], [lineY, lineY], ...
                    'Color', lineColors(4:6), 'LineWidth', 6);
            end
        end

        % Adjust y limits for threshold lines
        ylim(ax, [yLim(1) - 0.1 * (yLim(2) - yLim(1)), yLim(2)]);
    end

    % Labels
    xlabel(ax, shortName, 'FontSize', 9);
    if idx == 1
        ylabel(ax, 'frac. units', 'FontSize', 9);
    end

    % Clean styling
    ax.FontSize = 8;
    box(ax, 'off');
    ax.XAxis.TickLength = [0.02, 0.02];
end

function plotBarChart(ax, stats, colorScheme, splitNonSomatic)

    hold(ax, 'on');

    if splitNonSomatic
        categories = {'GOOD', 'MUA', 'NOISE', 'NON_SOMA_GOOD', 'NON_SOMA_MUA'};
        displayLabels = {'Good', 'MUA', 'Noise', 'Non-soma Good', 'Non-soma MUA'};
    else
        categories = {'GOOD', 'MUA', 'NOISE', 'NON_SOMA'};
        displayLabels = {'Good', 'MUA', 'Noise', 'Non-somatic'};
    end

    nCats = length(categories);
    xPos = 1:nCats;
    means = zeros(1, nCats);
    ses = zeros(1, nCats);
    colors = zeros(nCats, 3);

    for i = 1:nCats
        cat = categories{i};
        if isfield(stats, cat)
            means(i) = stats.(cat).mean * 100;
            ses(i) = stats.(cat).se * 100;
        end
        if isfield(colorScheme, cat)
            colors(i, :) = colorScheme.(cat);
        else
            colors(i, :) = [0.5, 0.5, 0.5];
        end
    end

    % Plot bars
    barWidth = 0.5;
    for i = 1:nCats
        bar(ax, xPos(i), means(i), barWidth, 'FaceColor', colors(i, :), ...
            'EdgeColor', 'w', 'LineWidth', 0.5);
    end

    % Add error bars
    errorbar(ax, xPos, means, ses, 'k', 'LineStyle', 'none', 'LineWidth', 1, 'CapSize', 3);

    % Styling
    set(ax, 'XTick', xPos, 'XTickLabel', displayLabels, 'FontSize', 9);
    ylabel(ax, '% units', 'FontSize', 9);
    maxVal = max(means + ses);
    ylim(ax, [0, maxVal * 1.15]);
    box(ax, 'off');
    ax.XAxis.TickLength = [0, 0];

    % Add n annotation
    text(ax, 0.98, 0.92, sprintf('n=%d', stats.nDatasets), ...
        'Units', 'normalized', 'HorizontalAlignment', 'right', ...
        'FontSize', 8, 'Color', [0.5, 0.5, 0.5]);

    % Panel label
    text(ax, -0.08, 1.08, 'c', 'Units', 'normalized', 'FontSize', 14, 'FontWeight', 'bold');
end

function methodsText = generateMethodsTextWithStats(param, stats, qMetric)
    % Generate methods text with statistics

    % Get base methods text
    [baseText, references, ~] = bc.qm.generateMethodsText(param, 'qualityMetrics', qMetric);

    % Add summary statistics
    summaryLines = {
        '', ...
        '--- Summary Statistics ---', ...
        sprintf('Across %d recording session(s) (%d total units), the following proportions were observed (mean +/- SE):', ...
            stats.nDatasets, stats.totalUnits)
    };

    categories = stats.categories;
    categoryLabels = {'noise', 'good single units', 'multi-unit activity', 'non-somatic units', 'non-somatic MUA'};

    for i = 1:length(categories)
        cat = categories{i};
        if isfield(stats, cat)
            meanPct = stats.(cat).mean * 100;
            sePct = stats.(cat).se * 100;
            totalCount = stats.(cat).totalCount;
            if i <= length(categoryLabels)
                label = categoryLabels{i};
            else
                label = lower(cat);
            end
            summaryLines{end+1} = sprintf('  - %s: %.1f +/- %.1f%% (%d units total)', ...
                label, meanPct, sePct, totalCount);
        end
    end

    summaryText = strjoin(summaryLines, newline);

    % Add references
    refText = [newline, newline, 'References:', newline];
    for i = 1:length(references)
        refText = [refText, sprintf('  - %s', references{i}), newline];
    end

    methodsText = [baseText, summaryText, refText];
end

function colorMtx = getHistogramColors()
    % Color matrix for histogram bars (same as plotGlobalQualityMetric)
    red_colors = [
        0.8627, 0.0784, 0.2353;
        1.0000, 0.1412, 0.0000;
        0.7255, 0.0000, 0.0000;
        0.5020, 0.0000, 0.1255;
        0.5020, 0.0000, 0.0000;
        0.8039, 0.3608, 0.3608;
    ];

    blue_colors = [
        0.2549, 0.4118, 0.8824;
        0.0000, 0.0000, 0.5020;
    ];

    darker_yellow_orange_colors = [
        0.7843, 0.7843, 0.0000;
        0.8235, 0.6863, 0.0000;
        0.8235, 0.5294, 0.0000;
        0.8039, 0.4118, 0.3647;
        0.8235, 0.3176, 0.2275;
        0.8235, 0.6157, 0.6510;
        0.7882, 0.7137, 0.5765;
        0.8235, 0.5137, 0.3922;
        0.7569, 0.6196, 0.0000;
        0.8235, 0.4510, 0.0000;
    ];

    colorMtx = [red_colors; blue_colors; darker_yellow_orange_colors];
end

function [metricNames, metricThresh1, metricThresh2, plotConditions, metricNames_SHORT, metricLineCols] = defineMetrics(param)
    % Define metrics for histograms (same as plotGlobalQualityMetric)

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

    if isfield(param, 'spDecayLinFit') && param.spDecayLinFit
        metricThresh1 = [param.maxNPeaks, param.maxNTroughs, param.maxScndPeakToTroughRatio_noise, param.maxPeak1ToPeak2Ratio_nonSomatic, param.maxMainPeakToTroughRatio_nonSomatic, ...
            param.maxRPVviolations, NaN, param.maxPercSpikesMissing, NaN, NaN, NaN, param.minSpatialDecaySlope, ...
            param.minWvDuration, param.maxWvBaselineFraction, NaN, NaN, ...
            param.maxDrift, NaN, param.isoDmin, NaN];

        metricThresh2 = [NaN, NaN, NaN, NaN, NaN, ...
            NaN, NaN, NaN, NaN, param.minNumSpikes, param.minAmplitude, NaN, ...
            param.maxWvDuration, NaN, param.minPresenceRatio, param.minSNR, ...
            NaN, NaN, NaN, param.lratioMax];
    else
        if isfield(param, 'minSpatialDecaySlopeExp')
            spatialMin = param.minSpatialDecaySlopeExp;
            spatialMax = param.maxSpatialDecaySlopeExp;
        else
            spatialMin = NaN;
            spatialMax = NaN;
        end

        metricThresh1 = [param.maxNPeaks, param.maxNTroughs, param.maxScndPeakToTroughRatio_noise, param.maxPeak1ToPeak2Ratio_nonSomatic, param.maxMainPeakToTroughRatio_nonSomatic, ...
            param.maxRPVviolations, NaN, param.maxPercSpikesMissing, NaN, NaN, NaN, spatialMin, ...
            param.minWvDuration, param.maxWvBaselineFraction, NaN, NaN, ...
            param.maxDrift, NaN, param.isoDmin, NaN];

        metricThresh2 = [NaN, NaN, NaN, NaN, NaN, ...
            NaN, NaN, NaN, NaN, param.minNumSpikes, param.minAmplitude, spatialMax, ...
            param.maxWvDuration, NaN, param.minPresenceRatio, param.minSNR, ...
            NaN, NaN, NaN, param.lratioMax];
    end

    plotConditions = [true, true, true, true, true, true, ...
        isfield(param, 'tauR_valuesMin') && isfield(param, 'tauR_valuesMax') && param.tauR_valuesMin ~= param.tauR_valuesMax, ...
        true, false, true, isfield(param, 'extractRaw') && param.extractRaw, ...
        isfield(param, 'computeSpatialDecay') && param.computeSpatialDecay == 1, ...
        true, true, true, isfield(param, 'extractRaw') && param.extractRaw, ...
        isfield(param, 'computeDrift') && param.computeDrift, ...
        isfield(param, 'computeDrift') && param.computeDrift, ...
        isfield(param, 'computeDistanceMetrics') && param.computeDistanceMetrics && isfield(param, 'isoDmin') && ~isnan(param.isoDmin), ...
        isfield(param, 'computeDistanceMetrics') && param.computeDistanceMetrics && isfield(param, 'isoDmin') && ~isnan(param.isoDmin)];

    metricLineCols = [0.2, 0.2, 0.2, 1, 0, 0, 0, 0, 0; ...
        0.2, 0.2, 0.2, 1, 0, 0, 0, 0, 0; ...
        0.2, 0.2, 0.2, 1, 0, 0, 1, 0, 0; ...
        0.2, 0.2, 0.2, 0.25, 0.41, 0.88, 0, 0, 0; ...
        0.2, 0.2, 0.2, 0.25, 0.41, 0.88, 0, 0, 0; ...
        0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ...
        0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ...
        0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ...
        0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ...
        1.0000, 0.5469, 0, 0, 0.5, 0, 0, 0, 0; ...
        1.0000, 0.5469, 0, 0, 0.5, 0, 0, 0, 0; ...
        1, 0, 0, 0.2, 0.2, 0.2, 1, 0, 0; ...
        1, 0, 0, 0.2, 0.2, 0.2, 1, 0, 0; ...
        0.2, 0.2, 0.2, 1, 0, 0, 1, 0, 0; ...
        1.0000, 0.5469, 0, 0, 0.5, 0, 0, 0, 0; ...
        1.0000, 0.5469, 0, 0, 0.5, 0, 0, 0, 0; ...
        0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ...
        0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ...
        1.0000, 0.5469, 0, 0, 0.5, 0, 0, 0, 0; ...
        0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0];

    % Reorder metrics
    indices_ordered = [1, 2, 14, 13, 3, 12, 4, 5, 11, 16, 6, 10, 15, 8, 17, 18, 19, 20];
    metricNames = metricNames(indices_ordered);
    metricNames_SHORT = metricNames_SHORT(indices_ordered);
    metricThresh1 = metricThresh1(indices_ordered);
    metricThresh2 = metricThresh2(indices_ordered);
    plotConditions = plotConditions(indices_ordered);
    metricLineCols = metricLineCols(indices_ordered, :);
end
