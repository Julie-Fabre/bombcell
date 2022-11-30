
mkdir(fullfile(savePath))

if ~exist(savePath, 'dir')
    mkdir(fullfile(savePath))
end
if param.saveAsMat
    save(fullfile(savePath, 'qMetric.mat'), 'qMetric', '-v7.3')
end
if param.saveAsParquet
    param.rawFolder = [];
    parquetwrite([fullfile(savePath, '_bc_parameters._bc_qMetrics.parquet')], struct2table(param, 'AsArray', 1))

    qMetricSummary = table('Size', [length(qMetric.clusterID), 10], 'VariableTypes', ...
        {'double', 'double', 'double', 'double', 'double', 'double', 'double', ...
        'double', 'double', 'double'}, 'VariableNames', ...
        {'percentageSpikesMissing', 'clusterID', 'fractionRefractoryPeriodViolations', 'nSpikes', 'nPeaks', 'nTroughs', 'isSomatic', ...
        'waveformDuration', 'spatialDecaySlope', 'waveformBaselineFlatness'});
    qMetricSummary.clusterID = qMetric.clusterID';
    %qMetricSummary.percSpikesMissing = arrayfun(@(x) nanmean(qMetric.percSpikesMissing(qMetric.useTheseTimes{x})), 1:size(qMetric.percSpikesMissing,1));

    qMetricSummary.percentageSpikesMissing = arrayfun(@(x) nanmean(qMetric.percSpikesMissing(x, ...
        qMetric.percSpikesMissing(x, :) <= param.maxPercSpikesMissing)), ...
        1:size(qMetric.percSpikesMissing, 1))';
    qMetricSummary.fractionRefractoryPeriodViolations = arrayfun(@(x) nanmean(qMetric.Fp(x, qMetric.Fp(x, :) ...
        <= param.maxRPVviolations)), 1:size(qMetric.percSpikesMissing, 1))';

    qMetricSummary.nSpikes = qMetric.nSpikes';
    qMetricSummary.nPeaks = qMetric.nPeaks';
    qMetricSummary.nTroughs = qMetric.nTroughs';
    qMetricSummary.isSomatic = qMetric.somatic';
    qMetricSummary.waveformDuration = qMetric.waveformDuration';
    qMetricSummary.spatialDecaySlope = qMetric.spatialDecaySlope';
    qMetricSummary.waveformBaselineFlatness = qMetric.waveformBaseline';


    parquetwrite([savePath, filesep, 'templates._bc_qMetrics.parquet'], qMetricSummary)

    parquetwrite([savePath, filesep, 'templates._bc_qMetrics.parquet'], qMetricSummary)
end
