function bc_plotGlobalQualityMetric(qMetric, param, unitType, uniqueTemplates, templateWaveforms)
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
    figHandle = 1;

    qMetricNames{1} = {'# peaks', '#troughs', 'waveform baseline', 'spatial decay slope', 'waveform duration', 'non-somatic'};
    if param.computeDistanceMetrics && ~isnan(param.isoDmin)
        qMetricNames{2} = {'refractory period violations', 'undetected spikes', '# spikes', 'waveform amplitude', ...
            'presence ratio', 'max drift', 'isolation distance', 'l-ratio'};
    else
        qMetricNames{2} = {'refractory period violations', 'undetected spikes', '# spikes', 'waveform amplitude', ...
            'presence ratio', 'max drift'};
    end
    %     % euler diagram - previous method, now replaced by upset plot below

    %     bc_plotEulerDiagram(qMetric, param, qMetricNames, figHandle)

    % upSet plot
    % bc_upSetPlot(qMetric, param, qMetricNames, figHandle)

    %% plot summary of waveforms classified as noise/mua/good
    % 1. single/multi/noise/axonal waveforms
    figure('Color', 'w');

    unitTypeString = {'Noise', 'Single', 'Multi', 'Non-somatic'};
    uniqueTemplates_idx = 1:size(uniqueTemplates, 1);
    for iUnitType = 0:3
        subplot(2, 2, iUnitType+1)
        title([unitTypeString{iUnitType+1}, ' unit template waveforms']);
        hold on;
        singleU = uniqueTemplates_idx(find(unitType == iUnitType));
        set(gca, 'XColor', 'w', 'YColor', 'w')
        singleUnitLines = arrayfun(@(x) plot(squeeze(templateWaveforms(singleU(x), :)), 'linewidth', 1, 'Color', 'k'), 1:size(singleU, 2));
    end

    %% plot distributions of unit quality metric values for each quality metric
    figure();
    colorMtx = bc_colors(15);
    colorMtx = [colorMtx(1:15, :, :), repmat(0.7, 15, 1)]; % make 70% transparent
    colorMtx = colorMtx([1:4:end, 2:4:end, 3:4:end, 4:4:end], :); % shuffle colors to get more distinct pairs

    title([num2str(sum(unitType == 1)), ' single units, ', num2str(sum(unitType == 2)), ' multi-units, ', ...
        num2str(sum(unitType == 0)), ' noise units, ', num2str(sum(unitType == 3)), ' non-somatic units.']);
    hold on;
    set(gcf, 'color', 'w')

    subplot(4, 5, 1)
    hold on;
    histogram(qMetric.nPeaks, 'FaceColor', colorMtx(1, 1:3), 'FaceAlpha', colorMtx(1, 4));
    yLim = ylim;
    line([param.maxNPeaks + 0.5, param.maxNPeaks + 0.5], [0, yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('# peaks')
    makepretty;

    subplot(4, 5, 2)

    hold on;
    histogram(qMetric.nTroughs, 'FaceColor', colorMtx(2, 1:3), 'FaceAlpha', colorMtx(2, 4));
    yLim = ylim;
    line([param.maxNTroughs + 0.5, param.maxNTroughs + 0.5], [0, yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('# troughs')
    makepretty;

    subplot(4, 5, 3)

    hold on;
    histogram(1-qMetric.isSomatic, 'FaceColor', colorMtx(3, 1:3), 'FaceAlpha', colorMtx(3, 4));
    yLim = ylim;
    line([param.somatic - 0.5, param.somatic - 0.5], [0, yLim(2)], 'Color', 'r', 'LineWidth', 2)
    xticks([0, 1])
    xticklabels({'Y', 'N'})
    ylabel('unit count')
    xlabel('somatic?')
    makepretty;

    subplot(4, 5, 4)

    hold on;

    histogram(qMetric.fractionRPVs_estimatedTauR*100, 'FaceColor', colorMtx(4, 1:3), 'FaceAlpha', colorMtx(4, 4), 'BinEdges', [0:5:100]);

    set(gca, 'yscale', 'log')
    yLim = ylim;
    line([param.maxRPVviolations * 100, param.maxRPVviolations * 100], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel(['refractory period', newline, 'violations (%)'])
    makepretty;

    subplot(4, 5, 5)

    hold on;
    tauR_values = param.tauR_valuesMin:param.tauR_valuesStep:param.tauR_valuesMax;
    
    histogram(tauR_values(qMetric.RPV_tauR_estimate), 'FaceColor', colorMtx(5, 1:3), 'FaceAlpha', colorMtx(5, 4), 'BinEdges', [param.tauR_valuesMin-1/1000:param.tauR_valuesStep:param.tauR_valuesMax+1/1000]);

    set(gca, 'yscale', 'log')
    yLim = ylim;
    ylabel('unit count')
    if length(tauR_values) == 1
        xlabel('refractory period used (s)')
    else
        xlabel(['estimated', newline, 'refractory period (s)'])
    end
    makepretty;


    subplot(4, 5, 6)

    hold on;
    histogram(qMetric.percentageSpikesMissing_gaussian, 'FaceColor', colorMtx(6, 1:3), 'FaceAlpha', colorMtx(6, 4), 'BinEdges', ...
        [0:5:max(qMetric.percentageSpikesMissing_gaussian)]);

    yLim = ylim;
    line([param.maxPercSpikesMissing, param.maxPercSpikesMissing], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel(['% spikes below', newline, 'detection threshold,', newline, 'gaussian assumption'])
    makepretty;

    subplot(4, 5, 7)

    hold on;
    histogram(qMetric.percentageSpikesMissing_symmetric, 'FaceColor', colorMtx(7, 1:3), 'FaceAlpha', colorMtx(7, 4), 'BinEdges', ...
        [0:5:max(qMetric.percentageSpikesMissing_symmetric)]);

    yLim = ylim;
    line([param.maxPercSpikesMissing, param.maxPercSpikesMissing], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel(['% spikes below', newline, 'detection threshold,', newline, 'symmetric assumption'])
    makepretty;


    subplot(4, 5, 8)

    hold on;
    set(gca, 'xscale', 'log')
    histogram(qMetric.nSpikes, 'FaceColor', colorMtx(8, 1:3), 'FaceAlpha', colorMtx(8, 4), 'BinEdges', [0:100:max(qMetric.nSpikes)]);

    yLim = ylim;
    line([param.minNumSpikes, param.minNumSpikes], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('# spikes')
    makepretty;
    
    if param.extractRaw
    subplot(4, 5, 9)

    hold on;
    set(gca, 'xscale', 'log')
    histogram(qMetric.rawAmplitude, 'FaceColor', colorMtx(9, 1:3), 'FaceAlpha', colorMtx(9, 4), 'BinEdges', [0:10:max(qMetric.rawAmplitude)]);
    yLim = ylim;
    line([param.minAmplitude, param.minAmplitude], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel(['mean raw waveform', newline, ' peak amplitude (uV)'])
    makepretty;
    end

    subplot(4, 5, 10)
    hold on;
    histogram(qMetric.spatialDecaySlope, 'FaceColor', colorMtx(10, 1:3), 'FaceAlpha', colorMtx(10, 4), 'BinEdges', [min(qMetric.spatialDecaySlope):0.0001:max(qMetric.spatialDecaySlope)]);
    yLim = ylim;
    line([param.minSpatialDecaySlope, param.minSpatialDecaySlope], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel(['spatial decay', newline, 'slope'])
    makepretty;
    

    subplot(4, 5, 11)
    hold on;
    histogram(qMetric.waveformDuration_peakTrough, 'FaceColor', colorMtx(11, 1:3), 'FaceAlpha', colorMtx(11, 4), 'BinEdges', ...
        [0:40:max(qMetric.waveformDuration_peakTrough)]);
    yLim = ylim;
    line([param.minWvDuration + 0.5, param.minWvDuration + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    line([param.maxWvDuration + 0.5, param.maxWvDuration + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel(['waveform', newline, 'duration'])
    makepretty;


    subplot(4, 5, 12)
    hold on;
    histogram(qMetric.waveformBaselineFlatness, 'FaceColor', colorMtx(12, 1:3), 'FaceAlpha', colorMtx(12, 4), 'BinEdges', ...
        [0:0.05:max(qMetric.waveformBaselineFlatness)]);
    yLim = ylim;
    line([param.maxWvBaselineFraction, param.maxWvBaselineFraction], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel(['waveform baseline', newline, '''flatness'''])
    makepretty;

    subplot(4, 5, 13) % presence ratio
    hold on;
    histogram(qMetric.presenceRatio, 'FaceColor', colorMtx(13, 1:3), 'FaceAlpha', colorMtx(13, 4), 'BinEdges', ...
        [0:0.05:max(qMetric.presenceRatio)]);
    yLim = ylim;
    line([param.minPresenceRatio, param.minPresenceRatio], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('presence ratio')
    makepretty;
    
    if param.extractRaw
    subplot(4, 5, 14) % signal to noise ratio
    hold on;
    set(gca, 'xscale', 'log')
    histogram(qMetric.signalToNoiseRatio, 'FaceColor', colorMtx(14, 1:3), 'FaceAlpha', colorMtx(14, 4)); %, 'BinEdges')%, ...
    %         [0:20:max(qMetric.signalToNoiseRatio(~isinf(qMetric.signalToNoiseRatio)))]);
    yLim = ylim;
    line([param.minSNR, param.minSNR], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel(['signal-to-noise', newline, 'ratio'])
    makepretty;
    end

    if param.computeDrift
        subplot(4, 5, 15) % max drift estimate
        hold on;
        histogram(qMetric.maxDriftEstimate, 'FaceColor', colorMtx(15, 1:3), 'FaceAlpha', colorMtx(15, 4), 'BinEdges', ...
            [0:5:max(qMetric.maxDriftEstimate)]);
        yLim = ylim;
        %line([param.plotDetails, param.maxWvBaselineFraction], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('unit count')
        xlabel('max drift estimate')
        makepretty;

        subplot(4, 5, 16) % max drift estimate
        hold on;
        histogram(qMetric.cumDriftEstimate, 'FaceColor', colorMtx(15, 1:3), 'FaceAlpha', colorMtx(15, 4), 'BinEdges', ...
            [0:50:max(qMetric.cumDriftEstimate)]);
        yLim = ylim;
        %line([param.plotDetails, param.maxWvBaselineFraction], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('unit count')
        xlabel('cum drift estimate')
        makepretty;
    end


    if param.computeDistanceMetrics && ~isnan(param.isoDmin)
        subplot(4, 5, 18)

        hold on;
        histogram(qMetric.isoD, 'FaceColor', colorMtx(1, 1:3), 'FaceAlpha', colorMtx(1, 4), 'BinEdges', [0:10:max(qMetric.isoD)]);
        yLim = ylim;
        line([param.isoDmin + 0.5, param.isoDmin + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('unit count')
        xlabel('isolation distance')
        makepretty;

        subplot(4, 5, 19)

        hold on;
        histogram(qMetric.Lratio, 'FaceColor', colorMtx(1, 1:3), 'FaceAlpha', colorMtx(1, 4), 'BinEdges', [0:10:max(qMetric.Lratio)]);
        yLim = ylim;
        line([param.lratioMax + 0.5, param.lratioMax + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('unit count')
        xlabel('l-ratio')
        makepretty;

    end


end
