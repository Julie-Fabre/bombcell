
function [ephysProperties, unitClassif] = runAllEphysPipeline(ephysPath, savePath, rerunEP, region)


%% compute ephys properties 
ephysPropertiesExist = dir(fullfile(savePath, 'templates._bc_ephysProperties.parquet'));

if isempty(ephysPropertiesExist) || rerunEP
    paramEP = bc.ep.ephysPropValues;
    [spikeTimes_samples, spikeTemplates, templateWaveforms, templateAmplitudes, ...
    pcFeatures, ~, channelPositions] = bc.load.loadEphysData(ephysPath);
    ephysProperties = bc.ep.computeAllEphysProperties(spikeTimes_samples, spikeTemplates, templateWaveforms,...
        templateAmplitudes, pcFeatures, channelPositions, paramEP, savePath);

elseif ~isempty(ephysPropertiesExist)
    [paramEP, ephysProperties, ~] = bc.ep.loadSavedProperties(savePath); 
end

%% classify cells 
if ~isempty(region) &&...
        ismember(region, {'CP', 'STR', 'Striatum', 'DMS', 'DLS', 'PS',...
        'Ctx', 'Cortical', 'Cortex'}) % cortex and striaum spelled every possible way 
    unitClassif = bc.clsfy.classifyCells(ephysProperties, paramEP, region);
    
else
    unitClassif = nan(size(ephysProperties,1),1);
end

end