function plotGlobalQualityMetric(qMetric, param, unitType, uniqueTemplates, templateWaveforms)
% JF,
% ------
% Inputs
% ------
%
% ------
% Outputs
% ------

% 1. multi-venn diagram of units classified as noise/mua by each quality metric
if param.plotGlobal

    %% plot summary of unit categorization

    % upSet plots (3 or 4) : Noise, Non-axonal, MUA, (Non-axonal MUA)
    bc.viz.upSetPlot_wrapper(qMetric, param, unitType)

    %% plot summary of waveforms classified as noise/mua/good
    % 1. single/multi/noise/axonal waveforms
    figure('Color', 'w');

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
        if param.spikeWidth == 61 %Kilosort 4
            xlim([1, 61])
        else
            xlim([21, 82])
        end
    end

    %% plot distributions of unit quality metric values for each quality metric

    figure('Position', [100, 100, 1500, 900], 'Color', 'w');
    try

        qMetric.scndPeakToTroughRatio = abs(qMetric.mainPeak_after_size./qMetric.mainTrough_size);
invalid_peaks = (abs(qMetric.mainTrough_size./qMetric.mainPeak_before_size) > param.minMainPeakToTroughRatio | ...
                            qMetric.mainPeak_before_width > param.minWidthFirstPeak | ...
                            qMetric.mainTrough_width > param.minWidthMainTrough);
peak1_2_ratio = (abs(qMetric.mainPeak_before_size./qMetric.mainPeak_after_size));

qMetric.peak1ToPeak2Ratio = peak1_2_ratio;
qMetric.peak1ToPeak2Ratio(invalid_peaks) = 0;
qMetric.mainPeakToTroughRatio = abs(max([qMetric.mainPeak_before_size, qMetric.mainPeak_after_size], [], 2)./qMetric.mainTrough_size);
         % Define metrics, thresholds, and plot conditions
    [metricNames, metricThresh1, metricThresh2, plotConditions, metricNames_SHORT, metricLineCols] = defineMetrics(param);

    numSubplots = sum(plotConditions);

    % Calculate the best grid layout
    numRows = floor(sqrt(numSubplots));
    numCols = ceil(numSubplots/numRows);

    red_colors = [
    0.8627 0.0784 0.2353;  % Crimson
    1.0000 0.1412 0.0000;  % Scarlet
    0.7255 0.0000 0.0000;  % Cherry
    0.5020 0.0000 0.1255;  % Burgundy
    0.5020 0.0000 0.0000;  % Maroon
    0.8039 0.3608 0.3608   % Indian Red
];

    blue_colors = [
    0.2549 0.4118 0.8824;  % Royal Blue
    0.0000 0.0000 0.5020   % Navy Blue
];
    % Color matrix setup
     darker_yellow_orange_colors = [
    0.7843 0.7843 0.0000;  % Dark Yellow
    0.8235 0.6863 0.0000;  % Dark Golden Yellow
    0.8235 0.5294 0.0000;  % Dark Orange
    0.8039 0.4118 0.3647;  % Dark Coral
    0.8235 0.3176 0.2275;  % Dark Tangerine
    0.8235 0.6157 0.6510;  % Dark Salmon
    0.7882 0.7137 0.5765;  % Dark Goldenrod
    0.8235 0.5137 0.3922;  % Dark Light Coral
    0.7569 0.6196 0.0000;  % Darker Goldenrod
    0.8235 0.4510 0.0000   % Darker Orange
];


    colorMtx = [red_colors; blue_colors;darker_yellow_orange_colors]; % shuffle colors

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
            binsize_offset = h.BinWidth/2;
            
            % Add horizontal lines
            yLim = ylim(ax);
            xLim = xlim(ax);
            lineY = yLim(1) - 0.02 * (yLim(2) - yLim(1)); % Position lines slightly below the plot
            
            if ~isnan(metricThresh1(i)) || ~isnan(metricThresh2(i))
                if ~isnan(metricThresh1(i)) && ~isnan(metricThresh2(i))
                    line(ax, [xLim(1)+binsize_offset, metricThresh1(i)+binsize_offset], [lineY, lineY], 'Color', metricLineCols(i,1:3), 'LineWidth', 6);
                    line(ax, [metricThresh1(i)+binsize_offset, metricThresh2(i)+binsize_offset], [lineY, lineY], 'Color', metricLineCols(i,4:6), 'LineWidth', 6);
                    line(ax, [metricThresh2(i)+binsize_offset, xLim(2)+binsize_offset], [lineY, lineY], 'Color', metricLineCols(i,7:9), 'LineWidth', 6);
                elseif ~isnan(metricThresh1(i))
                    line(ax, [xLim(1)+binsize_offset, metricThresh1(i)+binsize_offset], [lineY, lineY], 'Color', metricLineCols(i,1:3), 'LineWidth', 6);
                    line(ax, [metricThresh1(i)+binsize_offset, xLim(2)+binsize_offset], [lineY, lineY], 'Color', metricLineCols(i,4:6), 'LineWidth', 6);
                elseif ~isnan(metricThresh2(i))
                    line(ax, [xLim(1)+binsize_offset, metricThresh2(i)+binsize_offset], [lineY, lineY], 'Color', metricLineCols(i,1:3), 'LineWidth', 6);
                    line(ax, [metricThresh2(i)+binsize_offset, xLim(2)+binsize_offset], [lineY, lineY], 'Color', metricLineCols(i,4:6), 'LineWidth', 6);
                end
            end
            
            if i == 1
                ylabel(ax, 'frac. units','FontSize', 13)
            end
            xlabel(ax, metricNames_SHORT{i},'FontSize', 13)
            
            % Adjust axis limits to accommodate the lines
            ylim(ax, [yLim(1) - 0.1 * (yLim(2) - yLim(1)), yLim(2)]);
            ax.FontSize = 12; 
            
            axis tight;



            
            currentSubplot = currentSubplot + 1;
        end
    end
    catch
        warning('could not plot global plots')
    end



end
function [metricNames, metricThresh1, metricThresh2, plotConditions, metricNames_SHORT, metricLineCols] = defineMetrics(param)
    metricNames = {'nPeaks', 'nTroughs', 'scndPeakToTroughRatio', 'peak1ToPeak2Ratio', 'mainPeakToTroughRatio', ...
                   'fractionRPVs_estimatedTauR', 'RPV_tauR_estimate', 'percentageSpikesMissing_gaussian', ...
                   'percentageSpikesMissing_symmetric', 'nSpikes', 'rawAmplitude', 'spatialDecaySlope', ...
                   'waveformDuration_peakTrough', 'waveformBaselineFlatness', 'presenceRatio', 'signalToNoiseRatio', ...
                   'maxDriftEstimate', 'cumDriftEstimate', 'isoD', 'Lratio'};

    metricNames_SHORT = {'# peaks', '# troughs', 'peak_2/trough', 'peak_1/peak_2', 'peak_{main}/trough', ...
                   'frac. RPVs', 'RPV_tauR_estimate', '% spikes missing', ...
                   '%SpikesMissing-symmetric', '# spikes', 'amplitude', 'spatial decay', ...
                   'waveform duration', 'baseline flatness', 'presence ratio', 'SNR', ...
                   'maximum drift', 'cum. drift', 'isolation dist.', 'L-ratio'};

    if param.spDecayLinFit
        metricThresh1 = [param.maxNPeaks, param.maxNTroughs, param.minTroughToPeakRatio, param.firstPeakRatio, param.minTroughToPeakRatio, ...
                         param.maxRPVviolations, NaN, param.maxPercSpikesMissing, NaN, NaN, NaN, param.minSpatialDecaySlope, ...
                         param.minWvDuration, param.maxWvBaselineFraction,  NaN, NaN, ...
                         param.maxDrift, NaN, param.isoDmin, NaN];
        
        metricThresh2 = [NaN, NaN, param.minTroughToPeakRatio, NaN, NaN, ...
                         NaN, NaN, NaN, NaN, param.minNumSpikes, param.minAmplitude, NaN, ...
                         param.maxWvDuration, NaN, param.minPresenceRatio, param.minSNR, ...
                         NaN, NaN, NaN, param.lratioMax];
    else
        metricThresh1 = [param.maxNPeaks, param.maxNTroughs, param.minTroughToPeakRatio, param.firstPeakRatio, param.minTroughToPeakRatio, ...
                         param.maxRPVviolations, NaN, param.maxPercSpikesMissing, NaN, NaN, NaN, param.minSpatialDecaySlopeExp, ...
                         param.minWvDuration, param.maxWvBaselineFraction, NaN, NaN, ...
                         param.maxDrift, NaN, param.isoDmin, NaN];
        
        metricThresh2 = [NaN, NaN, param.minTroughToPeakRatio, NaN, NaN, ...
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

    metricLineCols = [0.2, 0.2, 0.2, 1, 0, 0, 0, 0, 0;...%1 'nPeaks'
    0.2, 0.2, 0.2, 1, 0, 0, 0, 0, 0;...%, 2 'nTroughs'
    0.2, 0.2, 0.2, 1, 0, 0, 1, 0, 0;...%, 3 'scndPeakToTroughRatio'
    0.2, 0.2, 0.2, 0.25,0.41,0.88, 0, 0, 0;...%, 4 'peak1ToPeak2Ratio'
    0.2, 0.2, 0.2, 0.25,0.41,0.88, 0, 0, 0;...%, 5 'mainPeakToTroughRatio', ...
    0, 0.5, 0, 1.0000,0.5469, 0, 0, 0, 0;...% 6 'fractionRPVs_estimatedTauR'
    0, 0.5, 0, 1.0000,0.5469, 0, 0, 0, 0;...%, 7 'RPV_tauR_estimate', 
    0, 0.5, 0, 1.0000,0.5469, 0, 0, 0, 0;...%'8 percentageSpikesMissing_gaussian'
    0, 0.5, 0, 1.0000,0.5469, 0, 0, 0, 0;...%'9 percentageSpikesMissing_symmetric'
    1.0000,0.5469, 0, 0, 0.5, 0, 0, 0, 0;...;%,10  # spikes'
    1.0000,0.5469, 0, 0, 0.5, 0, 0, 0, 0;...% 11, 'amplitude', 
    1,0,0, 0.2, 0.2, 0.2, 1, 0, 0;...% 12 'spatial decay', 
    1,0,0, 0.2, 0.2, 0.2, 1, 0, 0;...% 13 'waveform duration',
    0.2, 0.2, 0.2, 1,0,0, 1, 0, 0;...% 14 'baseline flatness', 
    1.0000,0.5469, 0, 0, 0.5, 0, 0, 0, 0;...% 15'presence ratio', 
    1.0000,0.5469, 0, 0, 0.5, 0, 0, 0, 0;...% 16'SNR',
    1.0000,0.5469, 0, 0, 0.5, 0, 0, 0, 0;...% 17 'maximum drift',
    1.0000,0.5469, 0, 0, 0.5, 0, 0, 0, 0;...% 18 'cum. drift', 
    0, 0.5, 0, 1.0000,0.5469, 0, 0, 0, 0;% 19'isolation dist.', 
    1.0000,0.5469, 0, 0, 0.5, 0, 0, 0, 0;...% 20'L-ratio'
               ];
    indices_ordered = [1,2,14,13,3,12,4,5,11,16,6,10,15,8,17,18,19,20];
    metricNames = metricNames(indices_ordered);
    metricNames_SHORT = metricNames_SHORT(indices_ordered);
    metricThresh1 = metricThresh1(indices_ordered);
    metricThresh2 = metricThresh2(indices_ordered);
    plotConditions = plotConditions(indices_ordered);
    metricLineCols = metricLineCols(indices_ordered,:);
end
end