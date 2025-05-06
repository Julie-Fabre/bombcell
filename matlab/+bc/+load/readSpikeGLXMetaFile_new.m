function [scalingFactor_uV, channelMapImro, probeType] = readSpikeGLXMetaFile_new(param)
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
metaFile = param.ephysMetaFile;
probeType = param.probeType;
recordingChannels_n = param.nChannels - param.nSyncChannels;

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

allChannels_index(allChannels_index > recordingChannels_n) = []; % remove sync

[~, sort_idx]= sort(allChannels_index);
gain_allChannels_ordered = gain_allChannels(sort_idx);

% bits_encoding 
bits_encoding = str2num(meta.imMaxInt);

% voltage range
Vrange = str2num(meta.imAiRangeMax);

% calculate scaling factor
scalingFactor = Vrange / bits_encoding ./ gain_allChannels_ordered;
scalingFactor_uV = scalingFactor * 1e6;
end