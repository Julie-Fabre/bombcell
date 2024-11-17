function ephysPropertiesTable = saveEphysProperties(paramEP, ephysProperties, savePath)
% JF, Reformat and save ephysProperties
% ------
% Inputs
% ------
% paramEP: matlab structure defining extraction and classification parameters 
%   (see bc_ephysPropValues for required fields
%   and suggested starting values)
% ephysProperties: matlab structure computed in the main loop of
%    bc_computeAllEphysProperties, each field is a nUnits x 1 vector containing 
%   the quality metric value for that unit 
% savePath: character array defining the path where you want to save your
%   quality metrics and parameters 
% ------
% Outputs
% ------
% ephysProperties: reformated ephysProperties structure into a table array

if ~exist(savePath, 'dir')
    mkdir(fullfile(savePath))
end

% save parameters
if isempty(paramEP.gain_to_uV)
        paramEP.gain_to_uV = 'NaN';
end
if ~istable(paramEP)
    paramEP = struct2table(paramEP);
end
parquetwrite([fullfile(savePath, '_bc_parameters._bc_ephysProperties.parquet')], paramEP)

% save acg
parquetwrite([fullfile(savePath, 'templates._bc_acg.parquet')], array2table(ephysProperties.acg))
ephysProperties = rmfield(ephysProperties, 'acg');

% save rest of ephys properties 
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