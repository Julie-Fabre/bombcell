function [scalingFactor_uV, channelMapImro, probeType] = readSpikeGLXMetaFile(param)
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
recordingChannels_n = param.nChannels - param.nSyncChannels;

meta = bc.dependencies.SGLX_readMeta.ReadMeta(metaFile);

% probeType 
probeType = meta.imDatPrb_type;

% channelMapImro 
if isfield(meta, 'imRoFile')
    channelMapImro = meta.imRoFile;
elseif isfield(meta, 'imroFile')
    channelMapImro = meta.imroFile;
end
if isempty(channelMapImro) % default was used
    if strcmp(probeType, '0')
        channelMapImro = 'NPtype21_bank0_ref0';
    end
end



%% scaling factor 
% gain 
gain_allChannels = bc.dependencies.SGLX_readMeta.ChanGainsIM(meta);
allChannels_index = bc.dependencies.SGLX_readMeta.OriginalChans(meta);

allChannels_index(allChannels_index > recordingChannels_n) = []; % remove sync

[~, sort_idx]= sort(allChannels_index);
gain_allChannels_ordered = gain_allChannels(sort_idx);

% bits_encoding 
if isfield(meta, 'imMaxInt')
    bits_encoding = str2num(meta.imMaxInt);
else
    if ismember(probeType, {'1', '3', '0', '1020', '1030', '1100', '1120', '1121', '1122', '1123', '1200', '1300', '1110'}) %NP1, NP1-like
        bits_encoding = 2^10 / 2; % 10-bit analog to digital
    elseif ismember(probeType, {'21', '2003', '2004', '24', '2013', '2014', '2020'}) % NP2, NP2-like
        bits_encoding = 2^14 / 2; % 14-bit analog to digital
    else
        error('unrecognized probe type. Check the imDatPrb_type value in your meta file and create a github issue / email us to add support for this probe type')
    end
end
% voltage range
if isfield(meta, 'imAiRangeMax')
    Vrange = str2num(meta.imAiRangeMax);
else
     if ismember(probeType, {'1', '3', '0', '1020', '1030', '1100', '1120', '1121', '1122', '1123', '1200', '1300', '1110'}) %NP1, NP2-like
        Vrange = 0.6; % from -0.6 to 0.6 V 
    elseif ismember(probeType, {'21', '2003', '2004', '24', '2013', '2014', '2020'}) % NP2, NP2-like
        Vrange = 0.5; % from -0.5 to 0.5 V
    else
        error('unrecognized probe type. Check the imDatPrb_type value in your meta file and create a github issue / email us to add support for this probe type')
    end
end

% calculate scaling factor
scalingFactor = Vrange / bits_encoding ./ gain_allChannels_ordered;
scalingFactor_uV = scalingFactor * 1e6;

end
