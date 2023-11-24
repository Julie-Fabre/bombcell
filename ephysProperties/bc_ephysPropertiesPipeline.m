function ephysProperties = bc_ephysPropertiesPipeline(ephysPath, savePath, rerunEP, region)

%% load ephys data 
paramEP = bc_ephysPropValues;

%% compute ephys properties 
ephysPropertiesExist = dir(fullfile(savePath, 'templates._bc_ephysProperties.parquet'));

if isempty(ephysPropertiesExist) || rerunEP

    [spikeTimes_samples, spikeTemplates, templateWaveforms,~, ~, ~, ~] = bc_loadEphysData(ephysPath);
    ephysProperties = bc_computeAllEphysProperties(spikeTimes_samples, spikeTemplates, templateWaveforms, paramEP, savePath);

elseif ~isempty(ephysPropertiesExist)
    [paramEP, ephysProperties, ~] = bc_loadSavedProperties(savePath); 
end

if ~isempty(region)
    % classify striatal, GPe and cortical cells
end

end