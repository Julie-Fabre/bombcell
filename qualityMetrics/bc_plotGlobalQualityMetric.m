% 1. multi-venn diagram of units classified as noise/mua by each quality metric
if param.plotGlobal
    figure();
    if param.computeDistanceMetrics && ~isnan(param.isoDmin)
        colorMtx = [lines(7), repmat(0.6, 7, 1)];
        colorMtx = [colorMtx; [rgb('Orange'), 0.6]];

        setsw = {find(qMetric.nPeaks > param.maxNPeaks), find(qMetric.nTroughs > param.maxNTroughs), ...
            find(qMetric.somatic == 0), find(qMetric.Fp > param.maxRPVviolations), ...
            find(qMetric.percSpikesMissing > param.maxPercSpikesMissing), ...
            find(qMetric.nSpikes <= param.minNumSpikes), find(qMetric.rawAmplitude <= param.minAmplitude),...
            find(qMetric.isoD <= param.isoDmin)};
        emptyCell = find(cellfun(@isempty,setsw));
        if ~isempty(emptyCell)
            for iEC = 1:length(emptyCell)
            setsw{emptyCell(iEC)} = 0;
            end
        end
        title('# of units classified as noise/mua/non-somatic with quality metrics')
        subplot(1, 5, 1:4)
        vennEulerDiagram(setsw, {'# peaks', '#troughs', 'non-somatic', 'refractory period violations', ...
            'undetected spikes', '# spikes', 'waveform amplitude', 'isolation distance'}, ...
            'ColorOrder', colorMtx(:, 1:3), ...
            'ShowIntersectionCounts', 1);
        subplot(1, 5, 5) % hacky way to get a legend
        set(gca, 'XColor', 'w', 'YColor', 'w')
        hold on;
        arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', colorMtx(x, :)), 1:8);
        legend({'# peaks', '#troughs', 'non-somatic', 'refractory period violations', ...
            'undetected spikes', '# spikes', 'waveform amplitude', 'isolation distance'})
        set(gcf, 'color', 'w')

    else
        colorMtx = [lines(7), repmat(0.6, 7, 1)];

        setsw = {find(qMetric.nPeaks > param.maxNPeaks), find(qMetric.nTroughs > param.maxNTroughs), ...
            find(qMetric.somatic == 0), find(qMetric.Fp > param.maxRPVviolations), ...
            find(qMetric.percSpikesMissing > param.maxPercSpikesMissing), ...
            find(qMetric.nSpikes <= param.minNumSpikes), find(qMetric.rawAmplitude <= param.minAmplitude)};
        title('# of units classified as noise/mua/non-somatic with quality metrics')
        subplot(1, 5, 1:4)
        emptyCell = find(cellfun(@isempty,setsw));
        if ~isempty(emptyCell)
            for iEC = 1:length(emptyCell)
            setsw{emptyCell(iEC)} = 0;
            end
        end
        
        vennEulerDiagram(setsw, {'# peaks', '#troughs', 'non-somatic', 'refractory period violations', 'undetected spikes', '# spikes', 'waveform amplitude'}, ...
            'ColorOrder', colorMtx(:, 1:3), ...
            'ShowIntersectionCounts', 1);
        subplot(1, 5, 5) % hacky way to get a legend
        set(gca, 'XColor', 'w', 'YColor', 'w')
        hold on;
        arrayfun(@(x) plot(NaN, NaN, 'linewidth', 2, 'color', colorMtx(x, :)), 1:7);
        legend({'# peaks', '#troughs', 'non-somatic', 'refractory period violations', 'undetected spikes', '# spikes', 'waveform amplitude'})
        set(gcf, 'color', 'w')
    end
    
    
    % 1. single/multi/noise/axonal waveforms 
    figure('Color', 'w');
    subplot(121)
    title('Single unit template waveforms');hold on;
    singleU = uniqueTemplates(find(unitType==1));
    set(gca, 'XColor', 'w', 'YColor', 'w')
    singleUnitLines = arrayfun(@(x) plot(squeeze(templateWaveforms(singleU(x),:,qMetric.maxChannels(singleU(x)))), 'linewidth', 1, 'Color', 'k'), 1:size(singleU,1));
    
%     subplot(132)
%     multiU = uniqueTemplates(find(unitType==2));
%     title('Multi unit template waveforms');hold on;
%     multiUnitLines = arrayfun(@(x) plot(squeeze(templateWaveforms(multiU(x),:,qMetric.maxChannels(multiU(x)))), 'linewidth', 1, 'Color', 'k'), 1:size(multiU,1));
    
    subplot(122)
    set(gca, 'XColor', 'w', 'YColor', 'w')
    noiseU = uniqueTemplates(find(unitType==0));
    title('Noise unit template waveforms');hold on;
    noiseUnitLines = arrayfun(@(x) plot(squeeze(templateWaveforms(noiseU(x),:,qMetric.maxChannels(noiseU(x)))), 'linewidth', 1, 'Color', 'k'), 1:size(noiseU,1));

    
    % 2. histogram for each quality metric, red line indicates
    % classification threshold
    figure();
    suptitle([num2str(sum(unitType==1)) ' single units, ', num2str(sum(unitType==2)), ' multi-units, ', num2str(sum(unitType==0)), ' noise units'])
    set(gcf, 'color', 'w')

    subplot(4, 2, 1)
    
    hold on;
    histogram(qMetric.nPeaks, 'FaceColor', colorMtx(1, 1:3), 'FaceAlpha', colorMtx(1, 4));
    yLim = ylim;
    line([param.maxNPeaks + 0.5, param.maxNPeaks + 0.5], [0, yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('# peaks')
    makepretty;

    subplot(4, 2, 2)
    
    hold on;
    histogram(qMetric.nTroughs, 'FaceColor', colorMtx(2, 1:3), 'FaceAlpha', colorMtx(2, 4));
    yLim = ylim;
    line([param.maxNTroughs + 0.5, param.maxNTroughs + 0.5], [0, yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('# troughs')
    makepretty;

    subplot(4, 2, 3)
    
    hold on;
    histogram(1-qMetric.somatic, 'FaceColor', colorMtx(3, 1:3), 'FaceAlpha', colorMtx(3, 4));
    yLim = ylim;
    line([param.somatic - 0.5, param.somatic - 0.5], [0, yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('non somatic')
    makepretty;

    subplot(4, 2, 4)
    
    hold on;
    histogram(qMetric.Fp, 'FaceColor', colorMtx(4, 1:3), 'FaceAlpha', colorMtx(4, 4), 'BinEdges', [0:5:max(qMetric.Fp)]);
    set(gca, 'yscale', 'log')
    yLim = ylim;
    line([param.maxRPVviolations + 0.5, param.maxRPVviolations + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('refractory period violations (%)')
    makepretty;

    subplot(4, 2, 5)
    
    hold on;
    histogram(qMetric.percSpikesMissing, 'FaceColor', colorMtx(5, 1:3), 'FaceAlpha', colorMtx(5, 4), 'BinEdges', [0:5:max(qMetric.percSpikesMissing)]);
    
    yLim = ylim;
    line([param.maxPercSpikesMissing + 0.5, param.maxPercSpikesMissing + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('spikes below detection threshold (%)')
    makepretty;

    subplot(4, 2, 6)
    
    hold on;
    set(gca, 'xscale', 'log')
    histogram(qMetric.nSpikes, 'FaceColor', colorMtx(6, 1:3), 'FaceAlpha', colorMtx(6, 4), 'BinEdges', [0:100:max(qMetric.nSpikes)]);
    
    yLim = ylim;
    line([param.minNumSpikes + 0.5, param.minNumSpikes + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('# spikes')
    makepretty;

    subplot(4, 2, 7)
    
    hold on;
     set(gca, 'xscale', 'log')
    histogram(qMetric.rawAmplitude, 'FaceColor', colorMtx(7, 1:3), 'FaceAlpha', colorMtx(7, 4), 'BinEdges', [0:10:max(qMetric.rawAmplitude)]);
    yLim = ylim;
    line([param.minAmplitude + 0.5, param.minAmplitude + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
    ylabel('unit count')
    xlabel('mean raw waveform peak amplitude (uV)')
    makepretty;

    if param.computeDistanceMetrics && ~isnan(param.isoDmin)
        subplot(4, 2, 8)
        
        hold on;
        histogram(qMetric.isoD, 'FaceColor', colorMtx(8, 1:3), 'FaceAlpha', colorMtx(8, 4), 'BinEdges', [0:10:max(qMetric.isoD)]);
        yLim = ylim;
        line([param.isoDmin + 0.5, param.isoDmin + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('unit count')
        xlabel('isolation distance')
        makepretty;
    end
    
    



end
