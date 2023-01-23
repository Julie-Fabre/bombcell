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
fractionRPVs_allTauR = readNPY(fullfile(savePath, 'templates._bc_fractionRefractoryPeriodViolationsPerTauR.npy'));
param = parquetread([fullfile(savePath, '_bc_parameters._bc_qMetrics.parquet')]);
end