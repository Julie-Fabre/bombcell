function ephysProperties = bc_ephysPropertiesPipeline_JF(animal, day, site, recording, experiment, rerun, runEP, region)

%% load ephys data 
paramEP = bc_ephysPropValues;

%% compute ephys properties 
ephysDirPath = AP_cortexlab_filenameJF(animal, day, experiment, 'ephys_dir',site, recording);
savePath = fullfile(ephysDirPath, 'ephysProperties');
ephysPropertiesExist = dir(fullfile(savePath, 'templates._bc_ephysProperties.parquet'));

if (runEP && isempty(ephysPropertiesExist)) || rerun

    ephysPath = AP_cortexlab_filenameJF(animal,day,experiment,'ephys',site, recording);
    [spikeTimes_samples, spikeTemplates, templateWaveforms, templateAmplitudes, pcFeatures, ~, channelPositions] = bc_loadEphysData(ephysPath);
    winv = readNPY([ephysPath filesep 'whitening_mat_inv.npy']);
    ephysProperties = bc_computeAllEphysProperties(spikeTimes_samples, spikeTemplates, templateWaveforms,...
     templateAmplitudes, pcFeatures, channelPositions, paramEP, savePath);
elseif ~isempty(ephysPropertiesExist)
    [paramEP, ephysProperties, ~] = bc_loadSavedProperties(savePath); 
end

if ~isempty(region)
    % classify cells 
end

end