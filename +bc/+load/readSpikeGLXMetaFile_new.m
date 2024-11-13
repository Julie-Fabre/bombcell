function [scalingFactors_mV, channelMapImro, probeType] = readSpikeGLXMetaFile_new(metaFile, probeType)
% Read SpikeGLX meta file and calculate scaling factor to convert raw data to microvolts
% Based on / largely copied from Jennifer Colonell's SGLX_readMeta implementation: https://github.com/jenniferColonell/SpikeGLX_Datafile_Tools
%
% Inputs:
%   metaFile: string, full path to meta file
%   probeType: string, (optional) probe type if not found in meta file
%
% Outputs:
%   scalingFactor: double, scaling factor to convert raw data to microvolts
%   channelMapImro: string, channel map information
%   probeType: string, detected or provided probe type

% 
% Parse meta file
fid = fopen(metaFile, 'r');
C = textscan(fid, '%[^=] = %[^\r\n]');
fclose(fid);

% Convert to structure (same as SGLX_readMeta implementation)
meta = struct();
for i = 1:length(C{1})
    tag = C{1}{i};
    if tag(1) == '~'
        tag = sprintf('%s', tag(2:end));
    end
    meta.(tag) = C{2}{i};
end

% Get probe type
if isfield(meta, 'imDatPrb_type')
    probeType = meta.imDatPrb_type;
elseif isfield(meta, 'imProbeOpt')
    probeType = meta.imProbeOpt;
elseif strcmp(probeType, 'NaN')
    error(['No probe type found in meta file and no probeType specified. ', ...
        'Please specify a probe type parameter.']);
end

% Get channel map information
if isfield(meta, 'imroTbl')
    channelMapImro = meta.imroTbl;
elseif isfield(meta, 'imRoTbl') % Alternative spelling
    channelMapImro = meta.imRoTbl;
else
    channelMapImro = '';
    if strcmp(probeType, '0')
        channelMapImro = 'NPtype21_bank0_ref0';
    end
end

% Get gain using the official method
[APgains, ~] = getGains(meta); % We only need AP gain for scaling

% Get maximum integer value (bits encoding)
if isfield(meta, 'imMaxInt')
    bits_encoding = str2double(meta.imMaxInt);
else
    % Determine based on probe type
    probeNum = str2double(probeType);
    if ismember(probeNum, [0, 1020, 1030, 1200, 1100, 1120, 1121, 1122, 1123, 1300])
        bits_encoding = 512; % NP1: 10-bit ADC = 2^9
    elseif ismember(probeNum, [21, 24, 2013])
        bits_encoding = 16384; % NP2: 14-bit ADC = 2^14
    else
        error('Unrecognized probe type: %s', probeType);
    end
end

% Get voltage range
if isfield(meta, 'imAiRangeMax')
    Vrange_mV = str2double(meta.imAiRangeMax) * 2 * 1e6; % Convert to microvolts
else
    probeNum = str2double(probeType);
    if ismember(probeNum, [0, 1020, 1030, 1200, 1100, 1120, 1121, 1122, 1123, 1300])
        Vrange_mV = 1.2e6; % NP1: ±0.6V
    elseif ismember(probeNum, [21, 24, 2013])
        Vrange_mV = 1.0e6; % NP2: ±0.5V
    else
        error('Unrecognized probe type for voltage range: %s', probeType);
    end
end

% Calculate scaling factor
scalingFactors_mV = Vrange_mV / bits_encoding / APgains;
end

function [APgain, LFgain] = getGains(meta)
% Helper function to get gains using the official method
% list of probe types with NP 1.0 imro format
np1_imro = [0, 1020, 1030, 1200, 1100, 1120, 1121, 1122, 1123, 1300];

% number of channels acquired
acqCountList = str2num(meta.acqApLfSy);
APgain = zeros(acqCountList(1)); % default type = float64
LFgain = zeros(acqCountList(2)); % empty array for 2.0

if isfield(meta, 'imDatPrb_type')
    probeType = str2double(meta.imDatPrb_type);
else
    probeType = 0;
end

if ismember(probeType, np1_imro)
    % imro + probe allows setting gain independently for each channel
    if isfield(meta, 'typeEnabled')
        % 3A data
        C = textscan(meta.imroTbl, '(%*s %*s %*s %d %d', ...
            'EndOfLine', ')', 'HeaderLines', 1);
    else
        % 3B data
        C = textscan(meta.imroTbl, '(%*s %*s %*s %d %d %*s', ...
            'EndOfLine', ')', 'HeaderLines', 1);
    end
    APgain = double(cell2mat(C(1)));
    LFgain = double(cell2mat(C(2)));
else
    % get gain from  imChan0apGain, if present
    if isfield(meta, 'imChan0apGain')
        APgain = APgain + str2num(meta.imChan0apGain);
        if acqCountList(2) > 0
            LFgain = LFgain + str2num(meta.imChan0lfGain);
        end
    elseif (probeType == 1110)
        % active UHD, for metadata lacking imChan0apGain, get gain from
        % imro table header
        currList = sscanf(meta.imroTbl, '(%d,%d,%d,%d,%d');
        APgain = APgain + currList(4);
        LFgain = LFgain + currList(5);
    elseif (probeType == 21) || (probeType == 24)
        % development NP 2.0; APGain = 80 for all AP
        % return 0 for LFgain (no LF channels)
        APgain = APgain + 80;
    elseif (probeType == 2013)
        % commercial NP 2.0; APGain = 100 for all AP
        % return 0 for LFgain (no LF channels)
        APgain = APgain + 100;
    else
        fprintf('unknown gain, setting APgain to 1\n');
        APgain = APgain + 1;
    end
end
end