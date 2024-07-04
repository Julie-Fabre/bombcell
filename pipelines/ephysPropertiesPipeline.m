
function [ephysProperties, unitClassif] = ephysPropertiesPipeline(ephysPath, savePath, rerunEP, region)


%% compute ephys properties 
ephysPropertiesExist = dir(fullfile(savePath, 'templates._bc_ephysProperties.parquet'));

if isempty(ephysPropertiesExist) || rerunEP
    paramEP = ep.ephysPropValues;
    [spikeTimes_samples, spikeTemplates, templateWaveforms, templateAmplitudes, ...
    pcFeatures, ~, channelPositions, ~] = load.loadEphysData(ephysPath);
    ephysProperties = ep.computeAllEphysProperties(spikeTimes_samples, spikeTemplates, templateWaveforms,...
        templateAmplitudes, pcFeatures, channelPositions, paramEP, savePath);

elseif ~isempty(ephysPropertiesExist)
    [paramEP, ephysProperties, ~] = ep.loadSavedProperties(savePath); 
end

%% classify cells 
if ~isempty(region) &&...
        ismember(region, {'CP', 'STR', 'Striatum', 'DMS', 'DLS', 'PS',...
        'Ctx', 'Cortical', 'Cortex'}) % cortex and striaum spelled every possible way 
    unitClassif = clsfy.classifyCells(ephysProperties, paramEP, region);
    
else
    unitClassif = nan(size(ephysProperties,1),1);
end

end