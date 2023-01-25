function bc_plotGlobalQualityMetric(qMetric, param, unitType, uniqueTemplates,rawWaveformsFull)
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
    try
        uniqueTemplates = unique(spikeTemplates);
        subplot(231)
        title('Single unit template waveforms');
        hold on;
        singleU = uniqueTemplates(find(unitType == 1));
        set(gca, 'XColor', 'w', 'YColor', 'w')
        singleUnitLines = arrayfun(@(x) plot(squeeze(templateWaveforms(singleU(x), :, qMetric.maxChannels(singleU(x)))), 'linewidth', 1, 'Color', 'k'), 1:size(singleU, 1));

        subplot(232)
        set(gca, 'XColor', 'w', 'YColor', 'w')
        multiU = uniqueTemplates(find(unitType == 2));
        title('Multi unit template waveforms');
        hold on;
        multiUnitLines = arrayfun(@(x) plot(squeeze(templateWaveforms(multiU(x), :, qMetric.maxChannels(multiU(x)))), 'linewidth', 1, 'Color', 'k'), 1:size(multiU, 1));

        subplot(233)
        set(gca, 'XColor', 'w', 'YColor', 'w')
        noiseU = uniqueTemplates(find(unitType == 0));
        title('Noise unit template waveforms');
        hold on;
        noiseUnitLines = arrayfun(@(x) plot(squeeze(templateWaveforms(noiseU(x), :, qMetric.maxChannels(noiseU(x)))), 'linewidth', 1, 'Color', 'k'), 1:size(noiseU, 1));
    catch
    end
    subplot(234)
    set(gca, 'XColor', 'w', 'YColor', 'w')
    singleUidx = find(unitType == 1);
    hold on;
    rawSingleUnitLines = arrayfun(@(x) plot(squeeze(rawWaveformsFull(singleUidx(x), rawWaveformsPeakChan(singleUidx(x)), :)), 'linewidth', 1, 'Color', 'k'), ...
        1:size(singleUidx, 1));
    %     rawSingleUnitLines = arrayfun(@(x) plot(squeeze(qMetric.rawWaveforms(singleUidx(x)).spkMapMean(qMetric.rawWaveforms(singleUidx(x)).peakChan,:)), 'linewidth', 1, 'Color', 'k'), ...
    %          1:size(singleUidx,1));
    %
    subplot(235)
    set(gca, 'XColor', 'w', 'YColor', 'w')
    multiUidx = find(unitType == 2);
    hold on;
    rawMultiUnitLines = arrayfun(@(x) plot(squeeze(rawWaveformsFull(multiUidx(x), rawWaveformsPeakChan(multiUidx(x)), :)), 'linewidth', 1, 'Color', 'k'), ...
        1:size(multiUidx, 1));
    %
    %     rawMultiUnitLines = arrayfun(@(x) plot(squeeze(qMetric.rawWaveforms(multiUidx(x)).spkMapMean(qMetric.rawWaveforms(multiUidx(x)).peakChan,:)), 'linewidth', 1, 'Color', 'k'), ...
    %          1:size(multiUidx,1));

    subplot(236)
    set(gca, 'XColor', 'w', 'YColor', 'w')
    noiseUidx = find(unitType == 0);
    hold on;
    rawNoiseUnitLines = arrayfun(@(x) plot(squeeze(rawWaveformsFull(noiseUidx(x), rawWaveformsPeakChan(noiseUidx(x)), :)), 'linewidth', 1, 'Color', 'k'), ...
        1:size(noiseUidx, 1));

    %     rawNoiseUnitLines = arrayfun(@(x) plot(squeeze(qMetric.rawWaveforms(noiseUidx(x)).spkMapMean(qMetric.maxChannels(noiseUidx(x)),:)), 'linewidth', 1, 'Color', 'k'), ...
    %          1:size(noiseUidx,1));
    %

    % 2. histogram for each quality metric, red line indicates
    % classification threshold
    %% plot distributions of unit quality metric values for each quality metric 
    figure();
    fn = fieldnames(mystruct);
for k=1:numel(fn)
    if( isnumeric(mystruct.(fn{k})) )
        % do stuff
    end
end

    title([num2str(sum(unitType == 1)), ' single units, ', num2str(sum(unitType == 2)), ' multi-units, ', num2str(sum(unitType == 0)), ' noise units'])
    set(gcf, 'color', 'w')

    subplot(3, 5, 1)

    hold on;
    histogram(qMetric.nPeaks, 'FaceColor', colorMtx(1, 1:3), 'FaceAlpha', colorMtx(1, 4));
    yLim = ylim;
    line([param.maxNPeaks + 0.5, param.maxNPeaks + 0.5], [0, yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('# peaks')
    makepretty;

    subplot(3, 5, 2)

    hold on;
    histogram(qMetric.nTroughs, 'FaceColor', colorMtx(2, 1:3), 'FaceAlpha', colorMtx(2, 4));
    yLim = ylim;
    line([param.maxNTroughs + 0.5, param.maxNTroughs + 0.5], [0, yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('# troughs')
    makepretty;

    subplot(3, 5, 3)

    hold on;
    histogram(1-qMetric.somatic, 'FaceColor', colorMtx(3, 1:3), 'FaceAlpha', colorMtx(3, 4));
    yLim = ylim;
    line([param.somatic - 0.5, param.somatic - 0.5], [0, yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('non somatic')
    makepretty;

    subplot(3, 5, 4)

    hold on;
    try
        histogram(qMetric.Fp, 'FaceColor', colorMtx(4, 1:3), 'FaceAlpha', colorMtx(4, 4), 'BinEdges', [0:5:max(qMetric.Fp(:))]);
    catch
        histogram(qMetric.Fp, 'FaceColor', colorMtx(4, 1:3), 'FaceAlpha', colorMtx(4, 4), 'BinEdges', [0:5:10]);
    end
    set(gca, 'yscale', 'log')
    yLim = ylim;
    line([param.maxRPVviolations + 0.5, param.maxRPVviolations + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel(['refractory period', newline, 'violations (%)'])
    makepretty;

    subplot(3, 5, 5)

    hold on;
    histogram(qMetric.percSpikesMissing, 'FaceColor', colorMtx(5, 1:3), 'FaceAlpha', colorMtx(5, 4), 'BinEdges', [0:5:max(qMetric.percSpikesMissing)]);

    yLim = ylim;
    line([param.maxPercSpikesMissing + 0.5, param.maxPercSpikesMissing + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel(['spikes below', newline, 'detection threshold (%)'])
    makepretty;

    subplot(3, 5, 6)

    hold on;
    set(gca, 'xscale', 'log')
    histogram(qMetric.nSpikes, 'FaceColor', colorMtx(6, 1:3), 'FaceAlpha', colorMtx(6, 4), 'BinEdges', [0:100:max(qMetric.nSpikes)]);

    yLim = ylim;
    line([param.minNumSpikes + 0.5, param.minNumSpikes + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('# spikes')
    makepretty;

    subplot(3, 5, 7)

    hold on;
    set(gca, 'xscale', 'log')
    histogram(qMetric.rawAmplitude, 'FaceColor', colorMtx(7, 1:3), 'FaceAlpha', colorMtx(7, 4), 'BinEdges', [0:10:max(qMetric.rawAmplitude)]);
    yLim = ylim;
    line([param.minAmplitude + 0.5, param.minAmplitude + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel(['mean raw waveform', newline, ' peak amplitude (uV)'])
    makepretty;

    subplot(3, 5, 8)
    hold on;
    histogram(qMetric.spatialDecaySlope, 'FaceColor', colorMtx(8, 1:3), 'FaceAlpha', colorMtx(8, 4), 'BinEdges', [min(qMetric.spatialDecaySlope):10:max(qMetric.spatialDecaySlope)]);
    yLim = ylim;
    line([param.minSpatialDecaySlope + 0.5, param.minSpatialDecaySlope + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('spatial decay slope')
    makepretty;

    subplot(3, 5, 9)
    hold on;
    histogram(qMetric.waveformDuration, 'FaceColor', colorMtx(9, 1:3), 'FaceAlpha', colorMtx(9, 4), 'BinEdges', [0:40:max(qMetric.waveformDuration)]);
    yLim = ylim;
    line([param.minWvDuration + 0.5, param.minWvDuration + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    line([param.maxWvDuration + 0.5, param.maxWvDuration + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('waveform duration')
    makepretty;

    subplot(3, 5, 10)
    hold on;
    histogram(qMetric.waveformBaseline, 'FaceColor', colorMtx(10, 1:3), 'FaceAlpha', colorMtx(10, 4), 'BinEdges', [0:0.05:max(qMetric.waveformBaseline)]);
    yLim = ylim;
    line([param.maxWvBaselineFraction, param.maxWvBaselineFraction], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('waveform baseline ''flatness''')
    makepretty;


    if param.computeDistanceMetrics && ~isnan(param.isoDmin)
        subplot(3, 5, 11)

        hold on;
        histogram(qMetric.isoD, 'FaceColor', colorMtx(11, 1:3), 'FaceAlpha', colorMtx(11, 4), 'BinEdges', [0:10:max(qMetric.isoD)]);
        yLim = ylim;
        line([param.isoDmin + 0.5, param.isoDmin + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('unit count')
        xlabel('isolation distance')
        makepretty;
    end


end
