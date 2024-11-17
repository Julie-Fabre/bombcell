function [paramEP, ephysProperties, acg] = loadSavedProperties(savePath)
% JF, Load saved ephys properties
% ------
% Inputs
% ------
% 
% ------
% Outputs
% ------
ephysProperties = parquetread(fullfile(savePath, 'templates._bc_ephysProperties.parquet'));
acg = parquetread([fullfile(savePath, 'templates._bc_acg.parquet')]);
paramEP = parquetread([fullfile(savePath, '_bc_parameters._bc_ephysProperties.parquet')]);
end