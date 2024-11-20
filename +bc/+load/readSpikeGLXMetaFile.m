function [scalingFactor, channelMapImro, probeType] = readSpikeGLXMetaFile(metaFile, probeType)
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

% Vrange / bits_encoding 
fI2V = bc.dependencies.SGLX_readMeta.Int2Volts(meta);

% calculate scaling factor
scalingFactor = fI2V / thisGain;

end