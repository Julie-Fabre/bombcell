function rawAmplitude = bc_getRawAmplitude(rawWaveforms, metaFileDir)
% JF, Get the amplitude of the mean raw waveform for a unit 
% ------
% Inputs
% ------
% rawWaveforms: nTimePoints × 1 double vector of the mean raw waveform
%   for one unit
% metaFileDir: dir structure containing the location of the raw .meta or .oebin file.
% ------
% Outputs
% ------
% rawAmplitude: raw amplitude in microVolts of the mean raw wwaveform for
% this unit 
% 

if strcmp(metaFileDir.name, 'oebin')
    % open ephys format
   scalingFactor = bc_readOEMetaFile(metaFileDir);
else
    % spikeGLX format
    scalingFactor = bc_readSpikeGLXMetaFile(metaFileDir);
end

rawWaveforms = rawWaveforms .* scalingFactor;
rawAmplitude = abs(max(rawWaveforms)) + abs(min(rawWaveforms));
end
















