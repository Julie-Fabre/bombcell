function qMetric = bc_saveQMetrics(param, qMetric, forGUI, savePath)
% JF, Reformat and save quality metrics
% ------
% Inputs
% ------
% param: matlab structure defining extraction and classification parameters 
%   (see bc_qualityParamValues for required fields
%   and suggested starting values)
% qMetric: matlab structure computed in the main loop of
%   bc_runAllQualityMetrics, each field is a nUnits x 1 vector containing 
%   the quality metric value for that unit 
% forGUI: matlab structure computed in the main loop of bc_runAllQualityMetrics,
%   for use in bc_unitQualityGUI
% savePath: character array defining the path where you want to save your
%   quality metrics and parameters 
% ------
% Outputs
% ------
% qMetric: reformated qMetric structure into a table array

if ~exist(savePath, 'dir')
    mkdir(fullfile(savePath))
end

% save parameters
if ~istable(param)
    parquetwrite([fullfile(savePath, '_bc_parameters._bc_qMetrics.parquet')], struct2table(param))
end
% save quality metrics
if param.saveMatFileForGUI
    save(fullfile(savePath, 'templates.qualityMetricDetailsforGUI.mat'), 'forGUI', '-v7.3')
end

% save fraction refractory period violations for all different tauR times
parquetwrite([fullfile(savePath, 'templates._bc_fractionRefractoryPeriodViolationsPerTauR.parquet')], array2table(qMetric.fractionRPVs))
qMetric.fractionRPVs_estimatedTauR = arrayfun(@(x) qMetric.fractionRPVs(x, qMetric.RPV_tauR_estimate(x)), 1:size(qMetric.fractionRPVs,1));
qMetric = rmfield(qMetric, 'fractionRPVs');

% save the rest of quality metrics and fraction refractory period
% violations for each unit's estimated tauR
% make sure everything is a double first
FNames = fieldnames(qMetric);
for fid = 1:length(FNames)
    eval(['qMetric.', FNames{fid}, '=double(qMetric.', FNames{fid}, ');'])
end
qMetricArray = double(squeeze(reshape(table2array(struct2table(qMetric, 'AsArray', true)), size(qMetric.maxChannels, 2), ...
    length(fieldnames(qMetric)))));
qMetricTable = array2table(qMetricArray);
qMetricTable.Properties.VariableNames = fieldnames(qMetric);

parquetwrite([fullfile(savePath, 'templates._bc_qMetrics.parquet')], qMetricTable)

% overwrite qMetric with the table, to be consistent with it for next steps
% of the pipeline
qMetric = qMetricTable;

end