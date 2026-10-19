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
% probe type groups. ADC bit depth and input range differ between the
% pre-commercial (phase 1) and commercial NP2.0 probes, so keep them apart.
% Reference: https://github.com/billkarsh/ProbeTable (Tables/probe_features.ini)
np1_types = {'0', '1', '3', '1020', '1030', '1100', '1110', '1120', '1121', '1122', '1123', '1200', '1300'}; % NP1, NP1-like
np2_preCommercial_types = {'21', '24'}; % NP2.0 phase 1 (NP2000, NP2010)
np2_commercial_types = {'2003', '2004', '2005', '2006', '2013', '2014', '2020', '2021', '2022', '2300'}; % NP2.0 commercial

% gain
% ChanGainsIM is indexed by original (acquired) channel, while the binary
% file - and so the raw waveform peak channels - only holds the saved
% channels (e.g. after removing bad channels with CatGT or saving a subset
% in SpikeGLX, see snsSaveChanSubset). Map each saved AP channel back to
% its original channel to get its gain, in file order.
gain_allChannels = bc.dependencies.SGLX_readMeta.ChanGainsIM(meta);
allChannels_index = bc.dependencies.SGLX_readMeta.OriginalChans(meta);
[nSavedAP, ~, ~] = bc.dependencies.SGLX_readMeta.ChannelCountsIM(meta);

savedAPChannels_index = allChannels_index(1:nSavedAP); % saved AP channels come first; drops LF and sync
gain_allChannels_ordered = reshape(gain_allChannels(savedAPChannels_index), [], 1);

if nSavedAP ~= recordingChannels_n
    warning(['The meta file lists %d saved AP channels (snsApLfSy) but param.nChannels - param.nSyncChannels = %d. ', ...
        'Set param.nChannels to nSavedChans (%s) from your meta file.'], ...
        nSavedAP, recordingChannels_n, meta.nSavedChans);
end

% bits_encoding 
if isfield(meta, 'imMaxInt')
    bits_encoding = str2num(meta.imMaxInt);
else
    if ismember(probeType, np1_types)
        bits_encoding = 2^10 / 2; % 10-bit analog to digital
    elseif ismember(probeType, np2_preCommercial_types)
        bits_encoding = 2^14 / 2; % 14-bit analog to digital
    elseif ismember(probeType, np2_commercial_types)
        bits_encoding = 2^12 / 2; % 12-bit analog to digital
    else
        error('unrecognized probe type. Check the imDatPrb_type value in your meta file and create a github issue / email us to add support for this probe type')
    end
end
% voltage range
if isfield(meta, 'imAiRangeMax')
    Vrange = str2num(meta.imAiRangeMax);
else
     if ismember(probeType, np1_types)
        Vrange = 0.6; % 1.2 Vpp: from -0.6 to 0.6 V 
    elseif ismember(probeType, np2_preCommercial_types)
        Vrange = 0.5; % 1.0 Vpp: from -0.5 to 0.5 V
    elseif ismember(probeType, np2_commercial_types)
        Vrange = 0.62; % 1.24 Vpp: from -0.62 to 0.62 V
    else
        error('unrecognized probe type. Check the imDatPrb_type value in your meta file and create a github issue / email us to add support for this probe type')
    end
end

% calculate scaling factor
scalingFactor = Vrange / bits_encoding ./ gain_allChannels_ordered;
scalingFactor_uV = scalingFactor * 1e6;

end
