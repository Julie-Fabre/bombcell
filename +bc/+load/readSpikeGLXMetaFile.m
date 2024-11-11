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

% read meta file
filetext = fileread(metaFile);

% try and get probe type from meta file (imDatPrb_type or imProbeOpt fields)
expr_scaling = 'imDatPrb_type=*';
[~, startIndex] = regexp(filetext, expr_scaling);
if isempty(startIndex) % try second option: there are two different saving conventions
    expr_scaling = 'imProbeOpt=*';
    [~, startIndex] = regexp(filetext, expr_scaling);
end
if isempty(startIndex) % if still no probe type information is found, use the param value
    if strcmp(probeType, 'NaN')
        error(['no probe type found in spikeGLX meta file and no param.probeType specified. ', ...
            'Edit the param.probeType value in bc_qualityParamValues.'])
    end
else
    endIndex = find(filetext(startIndex:end) == newline, 1, 'first') + startIndex - 2;
    if isempty(endIndex)
        endIndex = length(filetext);
    end
    probeType = filetext(startIndex+1:endIndex-1);
end

% get channel map information
expr_chanMap = 'imRoFile=';
[~, startIndexChanMap] = regexp(filetext, expr_chanMap);
expr_afterChanMap = 'imSampRate';
[~, endIndexChanMap] = regexp(filetext, expr_afterChanMap);
if isempty(startIndexChanMap) % new convention in new spike glx argh
    expr_chanMap = 'imroFile=';
    [~, startIndexChanMap] = regexp(filetext, expr_chanMap);
    expr_afterChanMap = 'nDataDirs';
    [~, endIndexChanMap] = regexp(filetext, expr_afterChanMap);
end

channelMapImro = filetext(startIndexChanMap+1:endIndexChanMap-2-length(expr_afterChanMap));
if isempty(channelMapImro) % default was used
    if strcmp(probeType, '0')
        channelMapImro = 'NPtype21_bank0_ref0';
    end
end

% get bits_encoding
expr_imMax = 'imMaxInt=';
[~, startIndeximMax] = regexp(filetext, expr_imMax);
if isempty(startIndeximMax) % new convention in new spike glx argh
    if ismember(probeType, {'1', '3', '0', '1020', '1030', '1100', '1120', '1121', '1122', '1123', '1200', '1300', '1110'}) %NP1, NP1-like
        bits_encoding = 2^10; % 10-bit analog to digital
    elseif ismember(probeType, {'21', '2003', '2004', '24', '2013', '2014', '2020'}) % NP2, NP2-like
        bits_encoding = 2^14; % 14-bit analog to digital
    else
        error('unrecognized probe type. Check the imDatPrb_type value in your meta file and create a github issue / email us to add support for this probe type')
    end
else
    endIndex = find(filetext(startIndeximMax:end) == newline, 1, 'first') + startIndeximMax - 2;
    if isempty(endIndex)
        endIndex = length(filetext);
    end
    bits_encoding = str2num(filetext(startIndeximMax+1:endIndex-1));
end

% get
expr_vMax = 'imAiRangeMax =';
[~, startIndexvMax] = regexp(filetext, expr_vMax);
if isempty(startIndexvMax) % new convention in new spike glx argh
    if ismember(probeType, {'1', '3', '0', '1020', '1030', '1100', '1120', '1121', '1122', '1123', '1200', '1300', '1110'}) %NP1, NP2-like
        Vrange = 1.2e6; % from -0.6 to 0.6 V = 1.2 V = 1.2 e6 microvolts
    elseif ismember(probeType, {'21', '2003', '2004', '24', '2013', '2014', '2020'}) % NP2, NP2-like
        Vrange = 1e6; % from -0.5 to 0.5 V = 1 V = 1 e6 microvolts
    else
        error('unrecognized probe type. Check the imDatPrb_type value in your meta file and create a github issue / email us to add support for this probe type')
    end
else
    endIndex = find(filetext(startIndexvMax:end) == newline, 1, 'first') + startIndexvMax - 2;
    if isempty(endIndex)
        endIndex = length(filetext);
    end
    Vrange = 2 * str2num(filetext(startIndexvMax+1:endIndex-1)) * 1e6;
end

% gain - QQ read out from table, modify. this could be wrong in some cases.
%
if ismember(probeType, {'1', '3', '0', '1020', '1030', '1100', '1120', '1121', '1122', '1123', '1200', '1300', '1110'}) %NP1, NP2-like
    gain = 500; % 10-bit analog to digital
elseif ismember(probeType, {'21', '2003', '2004', '24', '2013', '2014', '2020'}) % NP2, NP2-like
    gain = 80; % 14-bit analog to digital
else
    error('unrecognized probe type. Check the imDatPrb_type value in your meta file and create a github issue / email us to add support for this probe type')
end

% calculate scaling factor
scalingFactor = Vrange / bits_encoding / gain;
end
