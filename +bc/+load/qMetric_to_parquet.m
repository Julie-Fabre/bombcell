%% qMetric to parquet 
% For now, create only one parquet files and keep matlab struct for the GUI

yourQMetricFile = dir('/home/netshare/zinu/JF070/2022-06-13/ephys/site1/qMetrics/qMetric.mat');

load([yourQMetricFile.folder filesep yourQMetricFile.name])

qMetricSummary = table('Size',[length(qMetric.clusterID), 10],'VariableTypes',...
    {'double', 'double', 'double', 'double', 'double', 'double', 'double',...
    'double','double','double'},'VariableNames',...
    {'percSpikesMissing', 'clusterID', 'Fp', 'nSpikes', 'nPeaks', 'nTroughs', 'somatic', ...
    'waveformDuration', 'spatialDecaySlope', 'waveformBaseline'});
qMetricSummary.clusterID = qMetric.clusterID';
%qMetricSummary.percSpikesMissing = arrayfun(@(x) nanmean(qMetric.percSpikesMissing(qMetric.useTheseTimes{x})), 1:size(qMetric.percSpikesMissing,1));
qMetricSummary.percSpikesMissing = arrayfun( @(x) nanmean(qMetric.percSpikesMissing(x, qMetric.percSpikesMissing(x,:) <= param.maxPercSpikesMissing)), ...
    1:size(qMetric.percSpikesMissing,1))';
qMetricSummary.Fp = arrayfun( @(x) nanmean(qMetric.Fp(x, qMetric.Fp(x,:) <= param.maxRPVviolations)), ...
    1:size(qMetric.percSpikesMissing,1))';
qMetricSummary.nSpikes = qMetric.nSpikes';
qMetricSummary.nPeaks = qMetric.nPeaks';
qMetricSummary.nTroughs = qMetric.nTroughs';
qMetricSummary.somatic = qMetric.somatic';
qMetricSummary.waveformDuration = qMetric.waveformDuration';
qMetricSummary.spatialDecaySlope = qMetric.spatialDecaySlope';
qMetricSummary.waveformBaseline = qMetric.waveformBaseline';

parquetwrite([yourQMetricFile.folder filesep 'templates._jf_qMetrics.pqt'],qMetricSummary)

%parquetread([yourQMetricFile.folder filesep 'templates._jf_qMetrics.pqt'])

