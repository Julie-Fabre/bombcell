
mkdir(fullfile(savePath))

if ~exist(savePath, 'dir')
    mkdir(fullfile(savePath))
end
if param.saveAsMat
    save(fullfile(savePath, 'qMetric.mat'), 'qMetric', '-v7.3')
    parquetwrite([fullfile(savePath, '_bc_parameters._bc_qMetrics.parquet')], struct2table(param, 'AsArray', 1))

end

parquetwrite([fullfile(savePath, '_bc_parameters._bc_qMetrics.parquet')], struct2table(param, 'AsArray', 1))

if param.saveAsParquet

    
    qMetricSummary = table('Size', [length(qMetric.clusterID), 12], 'VariableTypes', ...
        {'double', 'double', 'double', 'double', 'double', 'double', 'double', ...
        'double', 'double', 'double','double', 'double'}, 'VariableNames', ...
        {'percentageSpikesMissing', 'clusterID', 'fractionRPVs',...
        'nSpikes', 'nPeaks', 'nTroughs', 'isSomatic', ...
        'waveformDuration', 'spatialDecaySlope', 'waveformBaselineFlatness', 'rawAmplitude', 'maxChannels'});
    qMetricSummary.clusterID = qMetric.clusterID';
    %qMetricSummary.percSpikesMissing = arrayfun(@(x) nanmean(qMetric.percSpikesMissing(qMetric.useTheseTimes{x})), 1:size(qMetric.percSpikesMissing,1));

    qMetricSummary.percentageSpikesMissing = arrayfun(@(x) nanmean(qMetric.percentageSpikesMissing(x, ...
        qMetric.percentageSpikesMissing(x, :) <= param.maxPercSpikesMissing)), ...
        1:size(qMetric.percentageSpikesMissing, 1))';
    qMetricSummary.fractionRPVs = arrayfun(@(x) nanmean(qMetric.fractionRPVs(x, qMetric.fractionRPVs(x, :) ...
        <= param.maxRPVviolations)), 1:size(qMetric.percentageSpikesMissing, 1))';

    qMetricSummary.nSpikes = qMetric.nSpikes';
    qMetricSummary.nPeaks = qMetric.nPeaks';
    qMetricSummary.nTroughs = qMetric.nTroughs';
    qMetricSummary.isSomatic = qMetric.isSomatic';
    qMetricSummary.waveformDuration = qMetric.waveformDuration';
    qMetricSummary.spatialDecaySlope = qMetric.spatialDecaySlope';
    qMetricSummary.waveformBaselineFlatness = qMetric.waveformBaselineFlatness';
    
    qMetricSummary.rawAmplitude = qMetric.rawAmplitude';
    qMetricSummary.maxChannels = qMetric.maxChannels;

    parquetwrite([savePath, filesep, 'templates._bc_qMetrics.parquet'], qMetricSummary)
end
