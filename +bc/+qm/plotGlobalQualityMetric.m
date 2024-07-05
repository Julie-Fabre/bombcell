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
    for iUnitType = 0:length(unitTypeString)-1
        subplot(2, ceil(length(unitTypeString)/2), iUnitType+1)
        title([unitTypeString{iUnitType+1}, ' unit template waveforms']);
        hold on;
        singleU = uniqueTemplates_idx(find(unitType == iUnitType));
        set(gca, 'XColor', 'w', 'YColor', 'w')
        singleUnitLines = arrayfun(@(x) plot(squeeze(templateWaveforms(singleU(x), :)), 'linewidth', 1, 'Color', [0, 0, 0, 0.2]), 1:size(singleU, 2));
        if param.spikeWidth==61 %Kilosort 4
            xlim([1 61])
        else
            xlim([21, 82])
        end
    end

    %% plot distributions of unit quality metric values for each quality metric
    figure();
    try
        colorMtx = bc.viz.colors(15);
        colorMtx = [colorMtx(1:15, :, :), repmat(0.7, 15, 1)]; % make 70% transparent
        colorMtx = colorMtx([1:4:end, 2:4:end, 3:4:end, 4:4:end], :); % shuffle colors to get more distinct pairs

        title([num2str(sum(unitType == 1)), ' single units, ', num2str(sum(unitType == 2)), ' multi-units, ', ...
            num2str(sum(unitType == 0)), ' noise units, ', num2str(sum(unitType == 3)), ' non-somatic units.']);
        hold on;

        subplot(4, 5, 1)
        hold on;
        rectangle('Position', [min(qMetric.nPeaks)-0.5, 0, param.maxNPeaks - min(qMetric.nPeaks)+1, 1 ], 'FaceColor', [0, .5, 0, 0.2])        
        histogram(qMetric.nPeaks, 'FaceColor', colorMtx(1, 1:3), 'FaceAlpha', colorMtx(1, 4), 'Normalization', 'probability');
        yLim = ylim;
        line([param.maxNPeaks + 0.5, param.maxNPeaks + 0.5], [0, yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('norm. unit count')
        xlabel('# peaks')


        subplot(4, 5, 2)
        hold on;
        rectangle('Position', [min(qMetric.nTroughs)-0.5, 0, param.maxNTroughs - min(qMetric.nTroughs)+1,1], 'FaceColor', [0, .5, 0, 0.2])
        histogram(qMetric.nTroughs, 'FaceColor', colorMtx(2, 1:3), 'FaceAlpha', colorMtx(2, 4), 'Normalization', 'probability');
        yLim = ylim;
        line([param.maxNTroughs + 0.5, param.maxNTroughs + 0.5], [0, yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('norm. unit count')
        xlabel('# troughs')


        subplot(4, 5, 3)
        hold on;
        rectangle('Position', [-0.5, 0, param.somatic , 1], 'FaceColor', [0, .5, 0, 0.2])
        histogram(1-qMetric.isSomatic, 'FaceColor', colorMtx(3, 1:3), 'FaceAlpha', colorMtx(3, 4), 'Normalization', 'probability');
        yLim = ylim;
        line([param.somatic - 0.5, param.somatic - 0.5], [0, yLim(2)], 'Color', 'r', 'LineWidth', 2)
        xticks([0, 1])
        xticklabels({'Y', 'N'})
        ylabel('norm. unit count')
        xlabel('somatic?')


        subplot(4, 5, 4)
        hold on;
        rectangle('Position', [-0.5, 1e-3, param.maxRPVviolations*100, 1-1e-3], 'FaceColor', [0, .5, 0, 0.2])
        
        histogram(qMetric.fractionRPVs_estimatedTauR*100, 'FaceColor', colorMtx(4, 1:3), 'FaceAlpha', colorMtx(4, 4), 'BinEdges', [0:5:100], 'Normalization', 'probability');
        set(gca, 'yscale', 'log')
        yLim = ylim;
        line([param.maxRPVviolations * 100, param.maxRPVviolations * 100], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('norm. unit count')
        xlabel(['refractory period', newline, 'violations (%)'])

        if param.tauR_valuesMin ~= param.tauR_valuesMax
            subplot(4, 5, 5)
            hold on;
            tauR_values = param.tauR_valuesMin:param.tauR_valuesStep:param.tauR_valuesMax;
            histogram(tauR_values(qMetric.RPV_tauR_estimate), 'FaceColor', colorMtx(5, 1:3), 'FaceAlpha', colorMtx(5, 4), 'BinEdges', ...
                [param.tauR_valuesMin - 1 / 1000:param.tauR_valuesStep:param.tauR_valuesMax + 1 / 1000], 'Normalization', 'probability');
            set(gca, 'yscale', 'log')
            yLim = ylim;
            ylabel('norm. unit count')
            if length(tauR_values) == 1
                xlabel('estimated refractory period (s)')
            else
                xlabel(['estimated', newline, 'refractory period (s)'])
            end
        end

        subplot(4, 5, 6)
        hold on;
        rectangle('Position', [-0.5,0, param.maxPercSpikesMissing+0.5, 1], 'FaceColor', [0, .5, 0, 0.2])
        histogram(qMetric.percentageSpikesMissing_gaussian, 'FaceColor', colorMtx(6, 1:3), 'FaceAlpha', colorMtx(6, 4), 'BinEdges', ...
            [0:5:max(qMetric.percentageSpikesMissing_gaussian)], 'Normalization', 'probability');
        yLim = ylim;
        line([param.maxPercSpikesMissing, param.maxPercSpikesMissing], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('norm. unit count')
        xlabel(['% spikes below', newline, 'detection threshold,', newline, 'gaussian assumption'])

        subplot(4, 5, 7)
        hold on;
        histogram(qMetric.percentageSpikesMissing_symmetric, 'FaceColor', colorMtx(7, 1:3), 'FaceAlpha', colorMtx(7, 4), 'BinEdges', ...
            [0:5:max(qMetric.percentageSpikesMissing_symmetric)], 'Normalization', 'probability');
        yLim = ylim;
        % line([param.maxPercSpikesMissing, param.maxPercSpikesMissing], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        %xLim = xlim;
        %rectangle('Position',[xLim(1) yLim(1)  param.maxPercSpikesMissing
        %- 0.5-xLim(1) yLim(2)-yLim(1)], 'FaceColor',[0, .5, 0, 0.2]) % not
        %currently used
        ylabel('norm. unit count')
        xlabel(['% spikes below', newline, 'detection threshold,', newline, 'symmetric assumption'])

        subplot(4, 5, 8)
        hold on;
        set(gca, 'xscale', 'log')
        minVal = min(qMetric.nSpikes); % Minimum 
        maxVal = max(qMetric.nSpikes); % Maximum value
        binEdges = logspace(log10(minVal), log10(maxVal), 20);
        rectangle('Position', [param.minNumSpikes,0, binEdges(end)-param.minNumSpikes, 1], 'FaceColor', [0, .5, 0, 0.2])
        histogram(qMetric.nSpikes, 'FaceColor', colorMtx(8, 1:3), 'FaceAlpha', colorMtx(8, 4), 'BinEdges', binEdges, 'Normalization', 'probability');
        yLim = ylim;
        line([param.minNumSpikes, param.minNumSpikes], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        xLim = xlim;
        rectangle('Position', [param.minNumSpikes, yLim(1), xLim(2) - param.minNumSpikes, yLim(2) - yLim(1)], 'FaceColor', [0, .5, 0, 0.2])
        ylabel('norm. unit count')
        xlabel('# spikes')


        if param.extractRaw
            subplot(4, 5, 9)
            hold on;
            set(gca, 'xscale', 'log')
            minVal = min(qMetric.rawAmplitude(qMetric.rawAmplitude>0)); % Minimum 
            maxVal = max(qMetric.rawAmplitude); % Maximum value
            binEdges = logspace(log10(minVal), log10(maxVal), 20);
            rectangle('Position', [param.minAmplitude, 0, binEdges(end) - param.minAmplitude, 1], 'FaceColor', [0, .5, 0, 0.2])
            histogram(qMetric.rawAmplitude, 'FaceColor', colorMtx(9, 1:3), 'FaceAlpha', colorMtx(9, 4), 'BinEdges', binEdges, 'Normalization', 'probability');
            yLim = ylim;
            line([param.minAmplitude, param.minAmplitude], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
            ylabel('norm. unit count')
            xlabel(['mean raw waveform', newline, ' peak amplitude (uV)'])
        end

        subplot(4, 5, 10)
        hold on;
        rectangle('Position', [min(qMetric.spatialDecaySlope), 0, abs(param.minSpatialDecaySlope -min(qMetric.spatialDecaySlope)), 1], 'FaceColor', [0, .5, 0, 0.2])
        histogram(qMetric.spatialDecaySlope, 'FaceColor', colorMtx(10, 1:3), 'FaceAlpha', colorMtx(10, 4), 'BinEdges',...
            [min(qMetric.spatialDecaySlope):(max(qMetric.spatialDecaySlope) - min(qMetric.spatialDecaySlope))./10:max(qMetric.spatialDecaySlope)], 'Normalization', 'probability');
        yLim = ylim;
        line([param.minSpatialDecaySlope, param.minSpatialDecaySlope], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('norm. unit count')
        xlabel(['spatial decay', newline, 'slope'])

        subplot(4, 5, 11)
        hold on;
        rectangle('Position', [param.minWvDuration, 0, param.maxWvDuration - param.minWvDuration, 1], 'FaceColor', [0, .5, 0, 0.2])  
        histogram(qMetric.waveformDuration_peakTrough, 'FaceColor', colorMtx(11, 1:3), 'FaceAlpha', colorMtx(11, 4), 'BinEdges', ...
            [0:40:max(qMetric.waveformDuration_peakTrough)], 'Normalization', 'probability');
        yLim = ylim;
        line([param.minWvDuration + 0.5, param.minWvDuration + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        line([param.maxWvDuration + 0.5, param.maxWvDuration + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('norm. unit count')
        xlabel(['waveform', newline, 'duration'])

        subplot(4, 5, 12)
        hold on;
        rectangle('Position', [0, 0, param.maxWvBaselineFraction, 1], 'FaceColor', [0, .5, 0, 0.2])
        histogram(qMetric.waveformBaselineFlatness, 'FaceColor', colorMtx(12, 1:3), 'FaceAlpha', colorMtx(12, 4), 'BinEdges', ...
            [0:0.05:max(qMetric.waveformBaselineFlatness)], 'Normalization', 'probability');
        yLim = ylim;
        line([param.maxWvBaselineFraction, param.maxWvBaselineFraction], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('norm. unit count')
        xlabel(['waveform baseline', newline, '''flatness'''])

        subplot(4, 5, 13) % presence ratio
        hold on;
        rectangle('Position', [param.minPresenceRatio, 0, max(qMetric.presenceRatio)-param.minPresenceRatio,1], 'FaceColor', [0, .5, 0, 0.2])
        histogram(qMetric.presenceRatio, 'FaceColor', colorMtx(13, 1:3), 'FaceAlpha', colorMtx(13, 4), 'BinEdges', ...
            [0:0.05:max(qMetric.presenceRatio)], 'Normalization', 'probability');
        yLim = ylim;
        line([param.minPresenceRatio, param.minPresenceRatio], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
        ylabel('norm. unit count')
        xlabel('presence ratio')

        if param.extractRaw
            subplot(4, 5, 14) % signal to noise ratio
            hold on;
            set(gca, 'xscale', 'log')
            minVal = min(qMetric.signalToNoiseRatio(qMetric.signalToNoiseRatio>0)); % Minimum 
            maxVal = max(qMetric.signalToNoiseRatio); % Maximum value
            binEdges = logspace(log10(minVal), log10(maxVal), 20);
            rectangle('Position', [param.minSNR, 0, max(qMetric.signalToNoiseRatio)-param.minSNR, 1], 'FaceColor', [0, .5, 0, 0.2])
            histogram(qMetric.signalToNoiseRatio, 'FaceColor', colorMtx(14, 1:3), 'FaceAlpha', colorMtx(14, 4), 'BinEdges', binEdges, 'Normalization', 'probability');
            yLim = ylim;
            line([param.minSNR, param.minSNR], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
            ylabel('norm. unit count')
            xlabel(['signal-to-noise', newline, 'ratio'])
        end

        if param.computeDrift
            subplot(4, 5, 15) % max drift estimate
            hold on;
            rectangle('Position',[-0.5, 0, param.maxDrift+0.5, 1], 'FaceColor',[0, .5, 0, 0.2])
            histogram(qMetric.maxDriftEstimate, 'FaceColor', colorMtx(15, 1:3), 'FaceAlpha', colorMtx(15, 4), 'BinEdges', ...
                [0:5:max(qMetric.maxDriftEstimate)], 'Normalization', 'probability');
            yLim = ylim;
            line([param.maxDrift , param.maxDrift ], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
            ylabel('norm. unit count')
            xlabel('max drift estimate')

            subplot(4, 5, 16) % max drift estimate
            hold on;
            histogram(qMetric.cumDriftEstimate, 'FaceColor', colorMtx(15, 1:3), 'FaceAlpha', colorMtx(15, 4), 'BinEdges', ...
                [0:50:max(qMetric.cumDriftEstimate)], 'Normalization', 'probability');
            yLim = ylim;
            %line([param.plotDetails, param.maxWvBaselineFraction], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
            ylabel('norm. unit count')
            xlabel('cum drift estimate')

        end


        if param.computeDistanceMetrics && ~isnan(param.isoDmin)
            subplot(4, 5, 18)
            hold on;
            rectangle('Position', [param.isoDmin, 0, max(qMetric.isoD) - param.isoDmin, 1], 'FaceColor', [0, .5, 0, 0.2])
            histogram(qMetric.isoD, 'FaceColor', colorMtx(1, 1:3), 'FaceAlpha', colorMtx(1, 4), 'BinEdges', [0:10:max(qMetric.isoD)], 'Normalization', 'probability');
            yLim = ylim;
            line([param.isoDmin + 0.5, param.isoDmin + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
            ylabel('norm. unit count')
            xlabel('isolation distance')


            subplot(4, 5, 19)
            hold on;
            set(gca, 'xscale', 'log')
            minVal = min(qMetric.Lratio(qMetric.Lratio>0)); % Minimum 
            maxVal = max(qMetric.Lratio); % Maximum value
            binEdges = logspace(log10(minVal), log10(maxVal), 20);
            rectangle('Position',[binEdges(1), 0, param.lratioMax-binEdges(1)+1, 1], 'FaceColor',[0, .5, 0, 0.2])
            histogram(qMetric.Lratio, 'FaceColor', colorMtx(1, 1:3), 'FaceAlpha', colorMtx(1, 4), 'BinEdges', binEdges, 'Normalization', 'probability');
            yLim = ylim;
            line([param.lratioMax + 0.5, param.lratioMax + 0.5], [yLim(1), yLim(2)], 'Color', 'r', 'LineWidth', 2)
            ylabel('norm. unit count')
            xlabel('l-ratio')


        end
    catch
        warning('could not plot global plots')
    end
    if exist('prettify_plot', 'file')
        prettify_plot('FigureColor', 'w')
    else
        warning('https://github.com/Julie-Fabre/prettify-matlab repo missing - download it and add it to your matlab path to make plots pretty')
    end
end
