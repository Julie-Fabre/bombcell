function rawAmplitude = bc_getRawAmplitude(rawWaveforms, rawFolder)


spikeFile = dir(fullfile(rawFolder, '*.ap.bin')); % spikeGLX format 
if isempty(spikeFile)
    spikeFile = dir(fullfile(rawFolder, '*.dat')); %openEphys format 
    metaFileDir = dir([rawFolder, '/../../*.oebin']);
    scalingFactor = bc_readOEMetaFile(metaFileDir);
    
    
else
    % 1.0 or 2.0? 
    metaFileDir = dir([rawFolder, '/*.ap.meta']);
    scalingFactor = bc_readSpikeGLXMetaFile(metaFileDir);
end
rawWaveforms = rawWaveforms .* scalingFactor;
rawAmplitude = abs(max(rawWaveforms)) + abs(min(rawWaveforms));
end

