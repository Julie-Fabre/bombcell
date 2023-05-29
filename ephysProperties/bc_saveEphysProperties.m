function ephysPropertiesTable = bc_saveEphysProperties(paramEP, ephysProperties, savePath)
% JF, Reformat and save ephysProperties
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
% ephysProperties: reformated qMetric structure into a table array

if ~exist(savePath, 'dir')
    mkdir(fullfile(savePath))
end

% save parameters
parquetwrite([fullfile(savePath, '_bc_parameters._bc_ephysProperties.parquet')], struct2table(paramEP))

% save acg
parquetwrite([fullfile(savePath, 'templates._bc_acg.parquet')], array2table(ephysProperties.acg))
ephysProperties = rmfield(ephysProperties, 'acg');

% save the rest of quality metrics and fraction refractory period
% violations for each unit's estimated tauR
% make sure everything is a double first
FNames = fieldnames(ephysProperties );
for fid = 1:length(FNames)
    eval(['ephysProperties.', FNames{fid}, '=double(ephysProperties.', FNames{fid}, ');'])
end
ephysPropertiesArray = double(squeeze(reshape(table2array(struct2table(ephysProperties, 'AsArray', true)), size(ephysProperties.clusterID, 2), ...
    length(fieldnames(ephysProperties)))));
ephysPropertiesTable = array2table(ephysPropertiesArray);
ephysPropertiesTable.Properties.VariableNames = fieldnames(ephysProperties);

parquetwrite([fullfile(savePath, 'templates._bc_ephysProperties.parquet')], ephysPropertiesTable)

end