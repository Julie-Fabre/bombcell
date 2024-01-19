
function [ephysProperties, unitClassif] = bc_ephysPropertiesPipeline(ephysPath, savePath, rerunEP, region)


%% compute ephys properties 
ephysPropertiesExist = dir(fullfile(savePath, 'templates._bc_ephysProperties.parquet'));

if isempty(ephysPropertiesExist) || rerunEP
    paramEP = bc_ephysPropValues;
    [spikeTimes_samples, spikeTemplates, templateWaveforms, templateAmplitudes, ...
    pcFeatures, ~, channelPositions, ~] = bc_loadEphysData(ephysPath);
    ephysProperties = bc_computeAllEphysProperties(spikeTimes_samples, spikeTemplates, templateWaveforms,...
        templateAmplitudes, pcFeatures, channelPositions, paramEP, savePath);

elseif ~isempty(ephysPropertiesExist)
    [paramEP, ephysProperties, ~] = bc_loadSavedProperties(savePath); 
end

%% classify cells 
if ~isempty(region) &&...
        ismember(region, {'CP', 'STR', 'Striatum', 'DMS', 'DLS', 'PS',...
        'Ctx', 'Cortical', 'Cortex'}) % cortex and striaum spelled every possible way 
    unitClassif = bc_classifyCells(ephysProperties, paramEP);
    
else
    unitClassif = nan(size(ephysProperties,1),1);
end

end