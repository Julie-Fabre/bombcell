function plotGlobalQualityMetric(varargin)
% Modified function that supports multiple input options
% ------
% Inputs
% ------
%
% Option 1: plotGlobalQualityMetric(ephys_path, save_path)
%   ephys_path: Path to the ephys data directory containing Kilosort outputs
%   save_path: Path where to save the output plot figures as PNG files
%
% Option 2: plotGlobalQualityMetric(param, ephys_path, save_path)
%   param
%   ephys_path: Path to the ephys data directory containing Kilosort outputs
%   save_path: Path where to save the output plot figures as PNG files
%
% Option 3: plotGlobalQualityMetric(qMetric, param, unitType, uniqueTemplates, templateWaveforms, save_path)
% ------
% Outputs
% ------
% No direct outputs, but creates plots of quality metrics and optionally saves them as PNG files

% Parse inputs
if nargin == 2
    % Option 1: Ephys path and figure save path
    ephys_path = varargin{1};
    save_path = varargin{2};
    load_data = true;
    param_provided = false;
elseif nargin == 3
    % Option 2: Parameter structure, ephys path, and figure save path
    param = varargin{1};
    ephys_path = varargin{2};
    save_path = varargin{3};
    load_data = true;
    param_provided = true;
elseif nargin >= 5
    % Option 3: All original inputs plus optional figure save path
    qMetric = varargin{1};
    param = varargin{2};
    unitType = varargin{3};
    uniqueTemplates = varargin{4};
    templateWaveforms = varargin{5};
    if nargin >= 6
        save_path = varargin{6};
    else
        save_path = '';
    end
    load_data = false;
    param_provided = true;
else
    error('Invalid number of input arguments. Use either 2, 3, or 5-6 inputs.');
end

% If we need to load data
if load_data

    % Load parameters and qMetrics
    try
        qMetricTable = parquetread(fullfile(save_path, 'templates._bc_qMetrics.parquet'));
        if ~param_provided
            paramTable = parquetread(fullfile(save_path, '_bc_parameters._bc_qMetrics.parquet'));
            param = table2struct(paramTable);
        end
        param.plotGlobal = 1;
    catch
        error('Could not load qMetrics or parameters. Make sure they exist at the specified path.');
    end

    qMetric = qMetricTable;

    % Load template waveforms
    try
        templates = readNPY(fullfile(ephys_path, 'templates.npy'));

        % Get unique templates
        spike_templates_0idx = readNPY(fullfile(ephys_path, 'spike_templates.npy'));
        spikeTemplates = spike_templates_0idx + 1;

        if exist(fullfile(ephys_path, 'spike_clusters.npy'), 'file')
            spike_clusters_0idx = readNPY(fullfile(ephys_path, 'spike_clusters.npy')); % already manually-curated
            spikeClusters = int32(spike_clusters_0idx) + 1;
            uniqueTemplates = unique(spikeClusters);
        else
            uniqueTemplates = unique(spikeTemplates);
        end

        % Extract template waveforms (simplified)
        [~, maxChannel] = max(max(abs(templates), [], 2), [], 3);
        templateWaveforms = zeros(size(templates, 1), size(templates, 2));
        for i = 1:size(templates, 1)
            templateWaveforms(i, :) = templates(i, :, maxChannel(i));
        end

    catch
        warning('Could not load template waveforms. Some plots may not be displayed.');
        uniqueTemplates = 1:height(qMetric);
        templateWaveforms = zeros(length(uniqueTemplates), param.spikeWidth);
    end

    % Get unit type
    unitType = bc.qm.getQualityUnitType(param, qMetric, save_path);

   
end
    % Call the original plotting function with the figure save path
    plotGlobalQualityMetricInternal(qMetric, param, unitType, uniqueTemplates, templateWaveforms, save_path);
end
function plotGlobalQualityMetricInternal(qMetric, param, unitType, uniqueTemplates, templateWaveforms, save_path)
% Internal function containing the original plotting code
% ------
% Inputs
% ------
% qMetric: Quality metrics table
% param: Parameters structure
% unitType: Unit type classification
% uniqueTemplates: Unique templates
% templateWaveforms: Template waveforms
% save_path: Path to save figure PNG files (empty string to not save)
% ------
% Outputs
% ------

% 1. multi-venn diagram of units classified as noise/mua by each quality metric
if param.plotGlobal
    % check quality metric and apram names
    [qMetric, param] = bc.qm.prettify_names(qMetric, param);

    %% plot summary of unit categorization

    % upSet plots (3 or 4) : Noise, Non-axonal, MUA, (Non-axonal MUA)
    bc.viz.upSetPlot_wrapper(qMetric, param, unitType, save_path);

    % Save figure if a save path is provided
    if ~isempty(save_path)
        if ~exist(save_path, 'dir')
            mkdir(save_path);
        end
    end

        %% plot summary of waveforms classified as noise/mua/good
        % 1. single/multi/noise/axonal waveforms
        waveform_fig = figure('Color', 'w');

        if param.splitGoodAndMua_NonSomatic == 0
            unitTypeString = {'Noise', 'Good', 'MUA', 'Non-somatic'};
        else
            unitTypeString = {'Noise', 'Somatic Good', 'Somatic MUA', 'Non-somatic Good', 'Non-somatic MUA'};
        end
        uniqueTemplates_idx = 1:size(uniqueTemplates, 1);
        for iUnitType = 0:length(unitTypeString) - 1
            subplot(2, ceil(length(unitTypeString)/2), iUnitType+1)
            title([unitTypeString{iUnitType+1}, ' unit waveforms']);
            hold on;
            singleU = uniqueTemplates_idx(find(unitType == iUnitType));
            set(gca, 'XColor', 'w', 'YColor', 'w')
            singleUnitLines = arrayfun(@(x) plot(squeeze(templateWaveforms(singleU(x), :)), 'linewidth', 1, 'Color', [0, 0, 0, 0.2]), 1:size(singleU, 2));
            % Flexible xlim based on spike width
            if param.spikeWidth <= 70
                % For shorter waveforms, show full range
                xlim([1, param.spikeWidth])
            else
                % For longer waveforms, skip initial baseline portion
                startIdx = max(1, round(21 * param.spikeWidth / 82));
                xlim([startIdx, param.spikeWidth])
            end
        end

        % Save waveform figure if a save path is provided
        if ~isempty(save_path)
            saveas(waveform_fig, fullfile(save_path, 'waveform_classification.png'));
        end

        %% plot distributions of unit quality metric values for each quality metric

        metrics_fig = figure('Position', [100, 100, 1500, 900], 'Color', 'w');


        % check if quality ,metric non somatic + noise ratios are present
        if ~ismember('mainPeakToTroughRatio', qMetric.Properties.VariableNames)
            qMetric.scndPeakToTroughRatio = abs(qMetric.mainPeak_after_size./qMetric.mainTrough_size); ...
                qMetric.peak1ToPeak2Ratio = abs(qMetric.mainPeak_before_size./qMetric.mainPeak_after_size);
            qMetric.mainPeakToTroughRatio = max([qMetric.mainPeak_before_size, qMetric.mainPeak_after_size], [], 2) ./ qMetric.mainTrough_size;
            qMetric.troughToPeak2Ratio = abs(qMetric.mainTrough_size./qMetric.mainPeak_before_size);
        end

        invalid_peaks = (qMetric.troughToPeak2Ratio > param.minTroughToPeak2Ratio_nonSomatic & ...
            qMetric.mainPeak_before_width < param.minWidthFirstPeak_nonSomatic & ...
            qMetric.mainTrough_width < param.minWidthMainTrough_nonSomatic);
        qMetric.peak1ToPeak2Ratio(invalid_peaks) = 0;
        % Define metrics, thresholds, and plot conditions
        [metricNames, metricThresh1, metricThresh2, plotConditions, metricNames_SHORT, metricLineCols] = defineMetrics(param);

        numSubplots = sum(plotConditions);

        % Calculate the best grid layout
        numRows = floor(sqrt(numSubplots));
        numCols = ceil(numSubplots/numRows);

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

                % Add horizontal lines
                yLim = ylim(ax);
                xLim = xlim(ax);
                lineY = yLim(1) - 0.02 * (yLim(2) - yLim(1)); % Position lines slightly below the plot

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
                    ylabel(ax, 'frac. units', 'FontSize', 13)
                end
                xlabel(ax, metricNames_SHORT{i}, 'FontSize', 13)

                % Adjust axis limits to accommodate the lines
                ylim(ax, [yLim(1) - 0.1 * (yLim(2) - yLim(1)), yLim(2)]);
                ax.FontSize = 12;

                axis tight;


                currentSubplot = currentSubplot + 1;
            end
        end

        % Save metrics figure if a save path is provided
        if ~isempty(save_path)
            saveas(metrics_fig, fullfile(save_path, 'quality_metrics_distribution.png'));
        end
end
end

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

    metricThresh2 = [NaN, NaN, NaN, NaN, NaN, ...
        NaN, NaN, NaN, NaN, param.minNumSpikes, param.minAmplitude, NaN, ...
        param.maxWvDuration, NaN, param.minPresenceRatio, param.minSNR, ...
        NaN, NaN, NaN, param.lratioMax];
else
    metricThresh1 = [param.maxNPeaks, param.maxNTroughs, param.maxScndPeakToTroughRatio_noise, param.maxPeak1ToPeak2Ratio_nonSomatic, param.maxMainPeakToTroughRatio_nonSomatic, ...
        param.maxRPVviolations, NaN, param.maxPercSpikesMissing, NaN, NaN, NaN, param.minSpatialDecaySlopeExp, ...
        param.minWvDuration, param.maxWvBaselineFraction, NaN, NaN, ...
        param.maxDrift, NaN, param.isoDmin, NaN];

    metricThresh2 = [NaN, NaN, NaN, NaN, NaN, ...
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
    0, 0.5, 0, 1.0000, 0.5469, 0, 0, 0, 0; ... % 7 'RPV_window_index',
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