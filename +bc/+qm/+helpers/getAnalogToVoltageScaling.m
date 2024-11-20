function scalingFactors = getAnalogToVoltageScaling(param) 


% get scaling factor 
if strcmp(param.ephysMetaFile, 'NaN') == 0
    if contains(param.ephysMetaFile, 'oebin')
        % open ephys format
        scalingFactor = bc.load.readOEMetaFile(param.ephysMetaFile); % single sclaing factor per channel for now 
        scalingFactors = repmat(scalingFactor, [param.nChannels - param.nSyncChannels, 1]);
    else
        % spikeGLX format
        [scalingFactors, ~, ~] = bc.load.readSpikeGLXMetaFile(param);
    end
else
     scalingFactors = repmat(param.gain_to_uV, [param.nChannels - param.nSyncChannels, 1]);
end
