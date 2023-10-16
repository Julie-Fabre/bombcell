function rawAmplitude = bc_getRawAmplitude(rawWaveforms, metaFile, probeType)
% JF, Get the amplitude of the mean raw waveform for a unit
% ------
% Inputs
% ------
% rawWaveforms: nTimePoints × 1 double vector of the mean raw waveform
%   for one unit
% metaFileDir: dir structure containing the location of the raw .meta or .oebin file.
% probeType: optional. only used if you are using spikeGLX *and* the meta
% file does not contain any probetype field (imDatPrb_type or imProbeOpt)
% ------
% Outputs
% ------
% rawAmplitude: raw amplitude in microVolts of the mean raw waveform for
%   this unit

% sanitize inputs
if nargin < 3 || isempty(probeType)
    probeType = NaN;
end

% get scaling factor 
if contains(metaFile, 'oebin')
    % open ephys format
    scalingFactor = bc_readOEMetaFile(metaFile);
else
    % spikeGLX format
    [scalingFactor, ~, ~] = bc_readSpikeGLXMetaFile(metaFile, probeType);
end

% scale waveforms to get amplitude in microVolts 
rawWaveforms = rawWaveforms .* scalingFactor;
rawAmplitude = abs(max(rawWaveforms)) + abs(min(rawWaveforms));
end
