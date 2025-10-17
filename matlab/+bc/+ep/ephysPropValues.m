function paramEP = ephysPropValues(ephysMetaDir, rawFile, ephysKilosortPath, gain_to_uV)
% 
% JF, Load a parameter structure defining extraction and
% classification parameters
% ------
% Inputs
% ------
%
% ------
% Outputs
% ------
% paramEP: matlab structure defining ephys properties extraction and
%   classification parameters
% 

if nargin < 1
    ephysMetaDir = '';
end
if nargin < 2
    rawFile = 'NaN';
end
if nargin < 3
    ephysKilosortPath = '';
end

if nargin < 4
    gain_to_uV = NaN;
end

paramEP = struct; 
paramEP.plotDetails = 0;
paramEP.verbose = 1;

% recording parameters
paramEP.ephys_sample_rate = 30000; % samples per second
paramEP.nChannels = 385; %number of recorded channels (including any sync channels)
    % recorded in the raw data. This is usually 384 or 385 for neuropixels
    % recordings
paramEP.nSyncChannels = 1;
if  ~isempty(ephysMetaDir)
    paramEP.ephysMetaFile = [ephysMetaDir.folder, filesep, ephysMetaDir.name];
    paramEP.gain_to_uV = NaN;
else
    paramEP.ephysMetaFile = 'NaN';
    paramEP.gain_to_uV = gain_to_uV;
end
paramEP.rawFile = rawFile;

% duplicate spikes parameters 
paramEP.removeDuplicateSpikes = 0;
paramEP.duplicateSpikeWindow_s = 0.00001; % in seconds 
paramEP.saveSpikes_withoutDuplicates = 1;
paramEP.recomputeDuplicateSpikes = 0;

% amplitude / raw waveform parameters
paramEP.detrendWaveform = 1; % If this is set to 1, each raw extracted spike is
    % detrended (we remove the best straight-fit line from the spike)
    % using MATLAB's builtin function detrend.
paramEP.nRawSpikesToExtract = 100; % how many raw spikes to extract for each unit 
paramEP.saveMultipleRaw = 0; % If you wish to save the nRawSpikesToExtract as well, 
    % currently needed if you want to run unit match https://github.com/EnnyvanBeest/UnitMatch
    % to track chronic cells over days after this
paramEP.decompressData = 0; % whether to decompress .cbin ephys data 

% Flexible spike width calculation based on sampling rate
% First need to get the actual sampling rate from metadata if available
actualSamplingRate = paramEP.ephys_sample_rate; % default
if ~isempty(ephysMetaDir) && exist(paramEP.ephysMetaFile, 'file')
    try
        % Try to read actual sampling rate from metadata
        [~, ~, actualSamplingRate, ~] = bc.load.loadMetaData(paramEP.ephysMetaFile);
        if ~isempty(actualSamplingRate) && actualSamplingRate > 0
            paramEP.ephys_sample_rate = actualSamplingRate;
        end
    catch
        % If reading fails, use default
    end
end

% For ephys properties, default to older KS behavior (82 samples) unless detected
% Try to detect KS version from template waveform size if possible
kilosortVersion = 2.5; % Default to older version
if ~isempty(ephysKilosortPath)
    templateFile = fullfile(ephysKilosortPath, 'templates.npy');
    if exist(templateFile, 'file')
        try
            templates = readNPY(templateFile);
            if size(templates, 2) == 61
                kilosortVersion = 4;
            end
        catch
            % If reading fails, use default
        end
    end
end

% Calculate spike width based on actual sampling rate
paramEP.spikeWidth = bc.qm.helpers.calculateSpikeWidth(paramEP.ephys_sample_rate, kilosortVersion);

% For backwards compatibility, store the standard spike widths
if kilosortVersion == 4
    paramEP.standardSpikeWidth = 61; % width in samples at 30kHz
else
    paramEP.standardSpikeWidth = 82; % width in samples at 30kHz
end

paramEP.extractRaw = 0; %whether to extract raw waveforms or not 
paramEP.probeType = 1; % if you are using spikeGLX and your meta file does 
    % not contain information about your probe type for some reason
    % specify it here: '1' for 1.0 (3Bs) and '2' for 2.0 (single or 4-shanks)
    % For additional probe types, make a pull request with more
    % information.  If your spikeGLX meta file contains information about your probe
    % type, or if you are using open ephys, this paramater wil be ignored.
paramEP.detrendWaveforms = 0;
paramEP.reextractRaw = 0; % re extract raw waveforms or not 
paramEP.nChannels = 385;

% ephys properties
paramEP.ephys_sample_rate = 30000;

% ACG 
paramEP.ACGbinSize = 0.001;
paramEP.ACGduration = 1;

%Proportion Long ISI
paramEP.longISI = 2;

% waveform parameters
paramEP.minThreshDetectPeaksTroughs = 0.2; % this is multiplied by the max value 
    % in a units waveform to give the minimum prominence to detect peaks using
    % matlab's findpeaks function.
paramEP.maxWvBaselineFraction = 0.3; % maximum absolute value in waveform baseline
    % should not exceed this fraction of the waveform's abolute peak value
paramEP.normalizeSpDecay = 1;
paramEP.spDecayLinFit = 1;
% QQ set this the same as qmetrics (load useChunks Start and stop)
%paramEP.computeTimeChunks = 1; % compute ephysProperties for different time chunks 
%paramEP.deltaTimeChunk = 360; %time in seconds 

% Scale width parameters based on spike width
standardWidth = paramEP.standardSpikeWidth; % 61 or 82 depending on KS version
paramEP.minWidthFirstPeak = max(2, round(4 * paramEP.spikeWidth / standardWidth)); % in samples
paramEP.minMainPeakToTroughRatio = 10;
paramEP.minWidthMainTrough = max(3, round(5 * paramEP.spikeWidth / standardWidth)); % in samples
paramEP.firstPeakRatio = 3;

% cell classification parameters
% - striatum 
paramEP.propISI_CP_threshold = 0.1;
paramEP.templateDuration_CP_threshold = 400;
paramEP.postSpikeSup_CP_threshold = 40;

% - cortex 
paramEP.templateDuration_Ctx_threshold = 400;

end
