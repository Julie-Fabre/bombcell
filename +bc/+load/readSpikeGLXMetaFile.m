function [scalingFactor, channelMapImro, probeType] = readSpikeGLXMetaFile(metaFile, probeType)
% Read SpikeGLX meta file and calculate scaling factor to convert raw data to microvolts
% Based on Jennifer Colonell's SGLX_readMeta implementation: https://github.com/jenniferColonell/SpikeGLX_Datafile_Tools
%
% Inputs:
%   metaFile: string, full path to meta file
%   probeType: string, (optional) probe type if not found in meta file
%
% Outputs:
%   scalingFactor: double, scaling factor to convert raw data to microvolts
%   channelMapImro: string, channel map information
%   probeType: string, detected or provided probe type

    % Parse meta file using the official method
    fid = fopen(metaFile, 'r');
    C = textscan(fid, '%[^=] = %[^\r\n]');
    fclose(fid);
    
    % Convert to structure
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
    elseif isfield(meta, 'imRoTbl')  % Alternative spelling
        channelMapImro = meta.imRoTbl;
    else
        channelMapImro = '';
        if strcmp(probeType, '0')
            channelMapImro = 'NPtype21_bank0_ref0';
        end
    end
    
    % Get maximum integer value (bits encoding)
    if isfield(meta, 'imMaxInt')
        bits_encoding = str2double(meta.imMaxInt);
    else
        % Determine based on probe type
        probeNum = str2double(probeType);
        if ismember(probeNum, [1, 3, 0, 1020, 1030, 1100, 1120, 1121, 1122, 1123, 1200, 1300, 1110])
            bits_encoding = 512;  % NP1: 10-bit ADC = 2^9
        elseif ismember(probeNum, [21, 2003, 2004, 24, 2013, 2014, 2020])
            bits_encoding = 16384;  % NP2: 14-bit ADC = 2^14
        else
            error('Unrecognized probe type: %s', probeType);
        end
    end
    
    % Get voltage range
    if isfield(meta, 'imAiRangeMax')
        Vrange = str2double(meta.imAiRangeMax) * 2 * 1e6;  % Convert to microvolts
    else
        probeNum = str2double(probeType);
        if ismember(probeNum, [1, 3, 0, 1020, 1030, 1100, 1120, 1121, 1122, 1123, 1200, 1300, 1110])
            Vrange = 1.2e6;  % NP1: ±0.6V
        elseif ismember(probeNum, [21, 2003, 2004, 24, 2013, 2014, 2020])
            Vrange = 1.0e6;  % NP2: ±0.5V
        else
            error('Unrecognized probe type for voltage range: %s', probeType);
        end
    end
    
    % Get gain - Use ChanGainsIM from the reference implementation
    if isfield(meta, 'imroTbl') || isfield(meta, 'imRoTbl')
        % Parse the gain from the imro table
        if isfield(meta, 'typeEnabled')
            % 3A data
            C = textscan(meta.imroTbl, '(%*s %*s %*s %d %d', ...
                'EndOfLine', ')', 'HeaderLines', 1);
        else
            % 3B data
            C = textscan(meta.imroTbl, '(%*s %*s %*s %d %d %*s', ...
                'EndOfLine', ')', 'HeaderLines', 1);
        end
        if ~isempty(C{1})
            gain = double(C{1}(1));  % Use first channel's gain
        else
            gain = getDefaultGain(probeType);
        end
    else
        gain = getDefaultGain(probeType);
    end
    
    % Calculate scaling factor
    scalingFactor = Vrange / bits_encoding / gain;
end

function gain = getDefaultGain(probeType)
    % Helper function to get default gain based on probe type
    probeNum = str2double(probeType);
    if ismember(probeNum, [1, 3, 0, 1020, 1030, 1100, 1120, 1121, 1122, 1123, 1200, 1300, 1110])
        gain = 500;  % NP1
    elseif ismember(probeNum, [21, 2003, 2004, 24, 2013, 2014, 2020])
        gain = 80;   % NP2
    else
        error('Unrecognized probe type for gain: %s', probeType);
    end
end