function [param, qMetric, fractionRPVs_allTauR] = bc_loadSavedMetrics(savePath)
% JF, Load saved quality metrics
% ------
% Inputs
% ------
% 
% ------
% Outputs
% ------
qMetric = parquetread(fullfile(savePath, 'templates._bc_qMetrics.parquet'));
fractionRPVs_allTauR = parquetread([fullfile(savePath, 'templates._bc_fractionRefractoryPeriodViolationsPerTauR.parquet')]);
param = parquetread([fullfile(savePath, '_bc_parameters._bc_qMetrics.parquet')]);
end