function upSetPlot_wrapper(qMetric, param, unitType)
% More info on UpSet plots in the original publication:
% Alexander Lex, Nils Gehlenborg, Hendrik Strobelt, Romain Vuillemot,
% Hanspeter Pfister. UpSet: Visualization of Intersecting Sets
% IEEE Transactions on Visualization and Computer Graphics (InfoVis),
% 20(12): 1983--1992, doi:10.1109/TVCG.2014.2346248, 2014.
%
% this MATLAB code inspired from this FEX code:
% https://uk.mathworks.com/matlabcentral/fileexchange/123695-upset-plot,
% written by Zhaoxu Liu / slandarer

%% Noise UpSet plot
if sum(ismember(unitType, 0)) > 0

    figHandle_noise = figure('Name', 'Noise vs neuronal units', 'Color', 'w');
    if param.spDecayLinFit
        UpSet_data_noise = [ ...
            isnan(qMetric.nPeaks) | qMetric.nPeaks > param.maxNPeaks, ...
            isnan(qMetric.nTroughs) | qMetric.nTroughs > param.maxNTroughs, ...
            qMetric.spatialDecaySlope > param.minSpatialDecaySlope, ...
            qMetric.waveformDuration_peakTrough < param.minWvDuration | qMetric.waveformDuration_peakTrough > param.maxWvDuration, ...
            qMetric.waveformBaselineFlatness > param.maxWvBaselineFraction, ...
            abs(qMetric.mainPeak_after_size./qMetric.mainTrough_size) > param.minTroughToPeakRatio, ...
            ];
    else
        UpSet_data_noise = [ ...
            isnan(qMetric.nPeaks) | qMetric.nPeaks > param.maxNPeaks, ...
            isnan(qMetric.nTroughs) | qMetric.nTroughs > param.maxNTroughs, ...
            qMetric.spatialDecaySlope < param.minSpatialDecaySlopeExp | qMetric.spatialDecaySlope > param.maxSpatialDecaySlopeExp, ...
            qMetric.waveformDuration_peakTrough < param.minWvDuration | qMetric.waveformDuration_peakTrough > param.maxWvDuration, ...
            qMetric.waveformBaselineFlatness > param.maxWvBaselineFraction, ...
            abs(qMetric.mainPeak_after_size./qMetric.mainTrough_size) > param.minTroughToPeakRatio, ...
            ];
    end

    UpSet_labels_noise = { ...
        'waveform peak #', ...
        'waveform trough #', ...
        'waveform spatial decay', ...
        'waveform duration', ...
        'waveform baseline flatness', ...
        'waveform 2nd peak to trough ratio'; ...
        };

    bc.viz.upSetPlot(UpSet_data_noise, UpSet_labels_noise, figHandle_noise);
else
    disp('No noise units with current param settings - consider changing your param values')
end

%% Non-somatic UpSet plot
if sum(ismember(unitType, 3)) > 0

    figHandle_nonsoma = figure('Name', 'Non-soma vs soma units', 'Color', 'w');

    UpSet_data_nonsoma = [ ...
        (abs(qMetric.mainPeak_before_size./qMetric.mainPeak_after_size) > param.firstPeakRatio & ...
        qMetric.mainPeak_before_width < param.minWidthFirstPeak & ...
        qMetric.mainTrough_width < param.minWidthMainTrough), ...
        abs(max([qMetric.mainPeak_before_size, qMetric.mainPeak_after_size], [], 2)./qMetric.mainTrough_size) > param.minMainPeakToTroughRatio 
        ];
    

    UpSet_labels_nonsoma = { 'waveform 1rst to 2nd peak ratio',...
        'waveform main peak to trough ratio'};

    bc.viz.upSetPlot(UpSet_data_nonsoma, UpSet_labels_nonsoma, figHandle_nonsoma);
else
    disp('No non-somatic units with current param settings - consider changing your param values')
end
% %% Non-somatic UpSet plot - coming soon
% figHandle_nonSoma = figure('Name','Non-somatic vs somatic units', 'Color', 'w');
%
% UpSet_data_nonSoma = [];
% UpSet_labels_noise = {''};
%
% upSetPlot(UpSet_data_nonSoma, UpSet_labels_nonSoma, figHandle_nonSoma);

%% MUA UpSet plot
if sum(ismember(unitType, [1, 2])) > 0
    figHandle_mua = figure('Name', 'Multi vs single units', 'Color', 'w');

    UpSet_data_mua = [qMetric.percentageSpikesMissing_gaussian > param.maxPercSpikesMissing, ...
        qMetric.nSpikes < param.minNumSpikes, ...
        qMetric.fractionRPVs_estimatedTauR > param.maxRPVviolations, ...
        qMetric.presenceRatio < param.minPresenceRatio];
    UpSet_labels_mua = {'% missing spikes', '# spikes', 'fraction RPVs', 'presence ratio'};

    if param.computeDistanceMetrics && ~isnan(param.isoDmin)
        UpSet_data_mua = [UpSet_data_mua, ...
            qMetric.isoD < param.isoDmin, ...
            qMetric.Lratio > param.lratioMax];
        UpSet_labels_mua{end+1} = 'isolation dist.';
        UpSet_labels_mua{end+1} = 'l-ratio';

    end
    if param.extractRaw
        UpSet_data_mua = [UpSet_data_mua, ...
            qMetric.rawAmplitude < param.minAmplitude, ...
            qMetric.signalToNoiseRatio < param.minSNR];
        UpSet_labels_mua{end+1} = 'amplitude';
        UpSet_labels_mua{end+1} = 'SNR';
    end
    if param.computeDrift
        UpSet_data_mua = [UpSet_data_mua, ...
            qMetric.maxDriftEstimate > param.maxDrift];
        UpSet_labels_mua{end+1} = 'max drift';
    end

    UpSet_data_mua = UpSet_data_mua(ismember(unitType, [1, 2]), :); %Keep only MUA and single units - remove noise and non-somatic
    bc.viz.upSetPlot(UpSet_data_mua, UpSet_labels_mua, figHandle_mua);
else
    disp('No MUA or good units with current param settings - consider changing your param values')
end

%% Non-somatic MUA UpSet plot
if param.splitGoodAndMua_NonSomatic
    figHandle_muaNonSoma = figure('Name', 'Non-somatic multi vs single units', 'Color', 'w');

    UpSet_data_muaNonSoma = [qMetric.percentageSpikesMissing_gaussian > param.maxPercSpikesMissing, ...
        qMetric.nSpikes < param.minNumSpikes, ...
        qMetric.fractionRPVs_estimatedTauR > param.maxRPVviolations, ...
        qMetric.presenceRatio < param.minPresenceRatio];
    UpSet_labels_muaNonSoma = {'% missing spikes', '# spikes', 'fraction RPVs', 'presence ratio'};

    if param.computeDistanceMetrics && ~isnan(param.isoDmin)
        UpSet_data_muaNonSoma = [UpSet_data_muaNonSoma, ...
            qMetric.isoD < param.isoDmin, ...
            qMetric.Lratio > param.lratioMax];
        UpSet_labels_mua{end+1} = 'isolation dist.';
        UpSet_labels_mua{end+1} = 'l-ratio';

    end
    if param.extractRaw
        UpSet_data_muaNonSoma = [UpSet_data_muaNonSoma, ...
            qMetric.rawAmplitude < param.minAmplitude, ...
            qMetric.signalToNoiseRatio < param.minSNR];
        UpSet_labels_muaNonSoma{end+1} = 'amplitude';
        UpSet_labels_muaNonSoma{end+1} = 'SNR';
    end
    if param.computeDrift
        UpSet_data_muaNonSoma = [UpSet_data_muaNonSoma, ...
            qMetric.maxDriftEstimate > param.maxDrift];
        UpSet_labels_muaNonSoma{end+1} = 'max drift';
    end

    UpSet_data_muaNonSoma = UpSet_data_muaNonSoma(ismember(unitType, [3, 4]), :); %Keep only non-somatic MUA and single units - remove noise and somatic
    bc.viz.upSetPlot(UpSet_data_muaNonSoma, UpSet_labels_muaNonSoma, figHandle_muaNonSoma);
end
end