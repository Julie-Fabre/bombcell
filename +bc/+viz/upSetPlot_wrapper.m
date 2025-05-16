function figHandles = upSetPlot_wrapper(qMetric, param, unitType, save_path)
% More info on UpSet plots in the original publication:
% Alexander Lex, Nils Gehlenborg, Hendrik Strobelt, Romain Vuillemot,
% Hanspeter Pfister. UpSet: Visualization of Intersecting Sets
% IEEE Transactions on Visualization and Computer Graphics (InfoVis),
% 20(12): 1983--1992, doi:10.1109/TVCG.2014.2346248, 2014.
%
% this MATLAB code inspired from this FEX code:
% https://uk.mathworks.com/matlabcentral/fileexchange/123695-upset-plot,
% written by Zhaoxu Liu / slandarer

% Check if save_path is provided
if nargin < 4
    save_path = '';
end

% Array to store figure handles
figHandles = [];

%% Noise UpSet plot
figHandle_noise = figure('Name', 'Noise vs neuronal units', 'Color', 'w', 'Position', [100, 100, 900, 900]);
figHandles = [figHandles, figHandle_noise];

if param.spDecayLinFit
    UpSet_data_noise = [ ...
        isnan(qMetric.nPeaks) | qMetric.nPeaks > param.maxNPeaks, ...
        isnan(qMetric.nTroughs) | qMetric.nTroughs > param.maxNTroughs, ...
        qMetric.spatialDecaySlope > param.minSpatialDecaySlope, ...
        qMetric.waveformDuration_peakTrough < param.minWvDuration | qMetric.waveformDuration_peakTrough > param.maxWvDuration, ...
        qMetric.waveformBaselineFlatness > param.maxWvBaselineFraction, ...
        qMetric.scndPeakToTroughRatio > param.maxScndPeakToTroughRatio_noise, ...
        ];
else
    UpSet_data_noise = [ ...
        isnan(qMetric.nPeaks) | qMetric.nPeaks > param.maxNPeaks, ...
        isnan(qMetric.nTroughs) | qMetric.nTroughs > param.maxNTroughs, ...
        qMetric.spatialDecaySlope < param.minSpatialDecaySlopeExp | qMetric.spatialDecaySlope > param.maxSpatialDecaySlopeExp, ...
        qMetric.waveformDuration_peakTrough < param.minWvDuration | qMetric.waveformDuration_peakTrough > param.maxWvDuration, ...
        qMetric.waveformBaselineFlatness > param.maxWvBaselineFraction, ...
        qMetric.scndPeakToTroughRatio > param.maxScndPeakToTroughRatio_noise, ...
        ];
end

UpSet_labels_noise = { ...
    '# peaks', ...
    '# troughs', ...
    'spatial decay', ...
    'duration', ...
    'baseline flatness', ...
    'peak_2/trough'; ...
    };

red_colors = [; ...
    0.8627, 0.0784, 0.2353; ... % Crimson
    1.0000, 0.1412, 0.0000; ... % Scarlet
    0.7255, 0.0000, 0.0000; ... % Cherry
    0.5020, 0.0000, 0.1255; ... % Burgundy
    0.5020, 0.0000, 0.0000; ... % Maroon
    0.8039, 0.3608, 0.3608; ... % Indian Red
    ];
if sum(UpSet_data_noise, 'all') > 0
    bc.viz.upSetPlot(UpSet_data_noise, UpSet_labels_noise, figHandle_noise, red_colors);
    hold on;
    sgtitle('Units classified as noise')
    
    % Save the figure if save_path is provided
    if ~isempty(save_path)
        if ~exist(save_path, 'dir')
            mkdir(save_path);
        end
        saveas(figHandle_noise, fullfile(save_path, 'noise_units_upset.png'));
    end
else
    disp('No noise units with current param settings - consider changing your param values')
end

%% Non-somatic UpSet plot
figHandle_nonSoma = figure('Name', 'Non-soma vs soma units', 'Color', 'w', 'Position', [100, 100, 900, 900]);
figHandles = [figHandles, figHandle_nonSoma];

UpSet_data_nonSoma = [ ...
        (qMetric.troughToPeak2Ratio < param.minTroughToPeak2Ratio_nonSomatic &...
        qMetric.mainPeak_before_width < param.minWidthFirstPeak_nonSomatic &...
        qMetric.mainTrough_width < param.minWidthMainTrough_nonSomatic &...
        qMetric.peak1ToPeak2Ratio > param.maxPeak1ToPeak2Ratio_nonSomatic) , ...
        qMetric.mainPeakToTroughRatio > param.maxMainPeakToTroughRatio_nonSomatic ...
        ];


UpSet_labels_nonSoma = {'peak_1/peak_2', ...
    'peak_{main}/trough'};
blue_colors = [; ...
    0.2549, 0.4118, 0.8824; ... % Royal Blue
    0.0000, 0.0000, 0.5020; ... % Navy Blue
    ];
if sum(UpSet_data_nonSoma, 'all') > 0
    bc.viz.upSetPlot(UpSet_data_nonSoma, UpSet_labels_nonSoma, figHandle_nonSoma, blue_colors);
    hold on;
    sgtitle('Units classified as non-somatic');
    
    % Save the figure if save_path is provided
    if ~isempty(save_path)
        saveas(figHandle_nonSoma, fullfile(save_path, 'nonsomatic_units_upset.png'));
    end
else
    disp('No non-somatic units with current param settings - consider changing your param values')
end

%% MUA UpSet plot
figHandle_mua = figure('Name', 'Multi vs single units', 'Color', 'w', 'Position', [100, 100, 900, 900]);
figHandles = [figHandles, figHandle_mua];

UpSet_data_mua = [qMetric.percentageSpikesMissing_gaussian > param.maxPercSpikesMissing, ...
    qMetric.nSpikes < param.minNumSpikes, ...
    qMetric.fractionRPVs_estimatedTauR > param.maxRPVviolations, ...
    qMetric.presenceRatio < param.minPresenceRatio];
UpSet_labels_mua = {'% missing spikes', '# spikes', 'fraction RPVs', 'presence ratio'};

if param.computeDistanceMetrics && ~isnan(param.isoDmin)
    UpSet_data_mua = [UpSet_data_mua, ...
        qMetric.isolationDistance < param.isoDmin, ...
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
if sum(UpSet_data_mua, 'all') > 0
    bc.viz.upSetPlot(UpSet_data_mua, UpSet_labels_mua, figHandle_mua, darker_yellow_orange_colors);
    hold on;
    sgtitle('Units classified as MUA');
    
    % Save the figure if save_path is provided
    if ~isempty(save_path)
        saveas(figHandle_mua, fullfile(save_path, 'mua_units_upset.png'));
    end
else
    disp('No MUA or good units with current param settings - consider changing your param values')
end

%% Non-somatic MUA UpSet plot
if param.splitGoodAndMua_NonSomatic
    figHandle_muaNonSoma = figure('Name', 'Non-somatic multi vs single units', 'Color', 'w', 'Position', [100, 100, 900, 900]);
    figHandles = [figHandles, figHandle_muaNonSoma];

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
    
    % Fixed typo in variable name from UpSet_data_muaNonsoma to UpSet_data_muaNonSoma
    if sum(UpSet_data_muaNonSoma, 'all') > 0
        bc.viz.upSetPlot(UpSet_data_muaNonSoma, UpSet_labels_muaNonSoma, figHandle_muaNonSoma, darker_yellow_orange_colors);
        hold on;
        sgtitle('Units classified as non-somatic & MUA');
        
        % Save the figure if save_path is provided
        if ~isempty(save_path)
            saveas(figHandle_muaNonSoma, fullfile(save_path, 'nonsomatic_mua_units_upset.png'));
        end
    end
end
end