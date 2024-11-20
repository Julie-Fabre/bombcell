function [scalingFactor_uV, channelMapImro, probeType] = readSpikeGLXMetaFile(metaFile, probeType, peakChan)
% JF
% read spikeGLX meta file and calculate scaling factor value to convert raw data to
% microvolts
% ------
% Inputs
% ------
% metaFile: string, full path to meta file (should be a structure.oebin file)
% ------
% Outputs
% ------
% scaling factor: double, scaling factor value to convert raw data to
% microvolts
%

meta = bc.dependencies.SGLX_readMeta.ReadMeta(metaFile);

% channelMapImro 
channelMapImro = meta.imRoFile;
if isempty(channelMapImro) % default was used
    if strcmp(probeType, '0')
        channelMapImro = 'NPtype21_bank0_ref0';
    end
end

% probeType 
probeType = meta.imDatPrb_type;

%% scaling factor 
% gain 
gain_allChannels = bc.dependencies.SGLX_readMeta.ChanGainsIM(meta);
allChannels_index = bc.dependencies.SGLX_readMeta.OriginalChans(meta);
thisChannel = find(allChannels_index == peakChan);
thisGain = gain_allChannels(thisChannel);

% bits_encoding 
bits_encoding = meta.imMaxInt;

% voltage range
Vrange = meta.imAiRangeMax;

% calculate scaling factor
scalingFactor = Vrange / bits_encoding / thisGain;
scalingFactor_uV = scalingFactor * 1000;
end