function rawAmplitude_uV = getRawAmplitude(rawWaveforms, metaFile, probeType, gain_to_uV)
% JF, Get the amplitude of the mean raw waveform for a unit
% ------
% Inputs
% ------
% rawWaveforms: nTimePoints × 1 double vector of the mean raw waveform
%   for one unit in mV
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
    probeType = 'NaN';
else
    probeType = num2str(probeType);
end

% get scaling factor 
if strcmp(metaFile, 'NaN') == 0
    if contains(metaFile, 'oebin')
        % open ephys format
        scalingFactor = bc.load.readOEMetaFile(metaFile);
    else
        % spikeGLX format
        [scalingFactor, ~, ~] = bc.load.readSpikeGLXMetaFile(metaFile, probeType);
    end
else
    scalingFactor = gain_to_uV;
end
    
    % scale waveforms to get amplitude in microVolts 
    rawWaveforms = rawWaveforms .* scalingFactor;
    rawAmplitude_uV = abs(max(rawWaveforms)) + abs(min(rawWaveforms));
    %rawAmplitude_mV = rawAmplitude_uV ./1000;
end
