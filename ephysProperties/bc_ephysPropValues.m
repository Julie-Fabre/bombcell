function paramEP = bc_ephysPropValues
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

paramEP = struct; 
paramEP.plotThis = 0;
paramEP.verbose = 1;

% duplicate spikes parameters 
paramEP.removeDuplicateSpikes = 1;
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
paramEP.spikeWidth = 82; % width in samples 
paramEP.extractRaw = 1; %whether to extract raw waveforms or not 
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

% QQ set this the same as qmetrics (load useChunks Start and stop)
%paramEP.computeTimeChunks = 1; % compute ephysProperties for different time chunks 
%paramEP.deltaTimeChunk = 360; %time in seconds 

% cell classification parameters
% - striatum 
paramEP.propISI_CP_threshold = 0.1;
paramEP.templateDuration_CP_threshold = 400;
paramEP.postSpikeSup_CP_threshold = 40;

% - cortex 
paramEP.templateDuration_Ctx_threshold = 400;

end
