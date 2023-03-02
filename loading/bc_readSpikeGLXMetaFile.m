function [scalingFactor, channelMapImro, probeType] = bc_readSpikeGLXMetaFile(metaFile)
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

filetext = fileread(metaFile);

expr_scaling = 'imDatPrb_type=*';
[~,startIndex] =  regexp(filetext,expr_scaling);
if isempty(startIndex)
    expr_scaling = 'imProbeOpt=*';
    [~,startIndex] =  regexp(filetext,expr_scaling);
end
probeType = filetext(startIndex+1);

expr_chanMap = 'imRoFile=';
[~,startIndexChanMap] =  regexp(filetext,expr_chanMap);
expr_afterChanMap = 'imSampRate';
[~,endIndexChanMap] =  regexp(filetext,expr_afterChanMap);
if isempty(startIndexChanMap) % new convention in new spike glx argh
    expr_chanMap = 'imroFile=';
    [~,startIndexChanMap] =  regexp(filetext,expr_chanMap);
    expr_afterChanMap = 'nDataDirs';
    [~,endIndexChanMap] =  regexp(filetext,expr_afterChanMap);
end

channelMapImro = filetext(startIndexChanMap+1:endIndexChanMap-2-length(expr_afterChanMap));
if isempty(channelMapImro) % default was used 
    if strcmp(probeType ,'0')
        channelMapImro = 'NPtype21_bank0_ref0';
    end
end


if strcmp(probeType ,'1') || strcmp(probeType ,'3') || strcmp(probeType ,'0') %1.0, 3B
    Vrange = 1.2e6; % from -0.6 to 0.6
    bits_encoding = 10; % 10-bit analog to digital 
    gain = 500; % fixed gain
elseif strcmp(probeType ,'2')%2.0
    Vrange = 1e6; % from -0.5 to 0.5
    bits_encoding = 14; % 14-bit analog to digital 
    gain = 80; % fixed gain
end
scalingFactor = Vrange / (2 ^ bits_encoding) / gain; 
end