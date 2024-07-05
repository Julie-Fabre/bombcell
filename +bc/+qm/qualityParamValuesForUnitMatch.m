function paramBC = qualityParamValuesForUnitMatch(ephysMetaDir, rawFile, ephysKilosortPath, gain_to_uV, kilosortVersion)
paramBC = struct;


if nargin < 5 
    kilosortVersion = 4; 
end

%% calculating quality metrics parameters 
% plotting parameters 
paramBC.plotDetails = 0; % generates a lot of plots, 
% mainly good if you running through the code line by line to check things,
% to debug, or to get nice plots for a presentation
paramBC.plotGlobal = 1; % plot summary of quality metrics 
paramBC.verbose = 1; % update user on progress
paramBC.reextractRaw = 0; % re extract raw waveforms or not - safer this way..
paramBC.extractRaw = 1; %whether to extract raw waveforms or not 

% saving parameters 
paramBC.saveAsTSV = 1; % additionally save outputs in .tsv file - this is 
    % useful if you want to use phy after bombcell: each quality metric value
    % will appear as a column in the Cluster view
paramBC.unitType_for_phy = 1; % whether to save the output of unitType in .tsv file for phy
if nargin < 3
    warning('no ephys kilosort path defined in bc_qualityParamValues, will save output tsv file in the savePath location')
else
    paramBC.ephysKilosortPath = ephysKilosortPath;
end
paramBC.saveMatFileForGUI = 1; % save certain outputs at .mat file - useful for GUI

% duplicate spikes parameters 
paramBC.removeDuplicateSpikes = 1;
paramBC.duplicateSpikeWindow_s = 0.00001; % in seconds 
paramBC.saveSpikes_withoutDuplicates = 1;
paramBC.recomputeDuplicateSpikes = 0;

% amplitude parameters
paramBC.detrendWaveform = 0; % If this is set to 1, each raw extracted spike is
    % detrended (we remove the best straight-fit line from the spike)
    % using MATLAB's builtin function detrend. 
paramBC.nRawSpikesToExtract = 1000;%inf; %inf if you don't encounter memory issues and want to load all spikes; % how many raw spikes to extract for each unit 
paramBC.saveMultipleRaw = 1; % If you wish to save the nRawSpikesToExtract as well, 
% currently needed if you want to run unit match https://github.com/EnnyvanBeest/UnitMatch
% to track chronic cells over days after this
paramBC.decompressData = 1; % whether to decompress .cbin ephys data 
if kilosortVersion == 4
    paramBC.spikeWidth = 61; % width in samples 
else
    paramBC.spikeWidth = 82; % width in samples 
end
paramBC.probeType = 1; % if you are using spikeGLX and your meta file does 
    % not contain information about your probe type for some reason
    % specify it here: '1' for 1.0 (3Bs) and '2' for 2.0 (single or 4-shanks)
    % For additional probe types, make a pull request with more
    % information.  If your spikeGLX meta file contains information about your probe
    % type, or if you are using open ephys, this paramater wil be ignored.

% signal to noise ratio
if kilosortVersion == 4
    paramBC.waveformBaselineNoiseWindow = 10; %time in samples at beginning of times
        % extracted to computer the mean raw waveform - this needs to be before the
        % waveform starts 
else
    paramBC.waveformBaselineNoiseWindow = 20; %time in samples at beginning of times
        % extracted to computer the mean raw waveform - this needs to be before the
        % waveform starts 
end

% refractory period parameters - change closer together
paramBC.tauR_valuesMin = 2/1000; % refractory period time (s), usually 0.0020 change
paramBC.tauR_valuesStep = 0.5./1000; % refractory period time (s), usually 0.0020
paramBC.tauR_valuesMax = 2./1000; % refractory period time (s), usually 0.0020
paramBC.tauC = 0.1/1000; % censored period time (s)

% percentage spikes missing parameters 
paramBC.computeTimeChunks = 0; % compute fraction refractory period violations 
% and percent sp[ikes missing for different time chunks 
paramBC.deltaTimeChunk = 360; %time in seconds 

% presence ratio 
paramBC.presenceRatioBinSize = 60; % in seconds 

% drift estimate
paramBC.driftBinSize = 60; % in seconds
paramBC.computeDrift = 0; % whether to compute each units drift. this is a 
% critically slow step that takes around 2seconds per unit 

% waveform parameters
if kilosortVersion == 4
    paramBC.waveformBaselineWindowStart = 1;
    paramBC.waveformBaselineWindowStop = 11; % in samples 
else
    paramBC.waveformBaselineWindowStart = 20;
    paramBC.waveformBaselineWindowStop = 30; % in samples 
end
paramBC.minThreshDetectPeaksTroughs = 0.2; % this is multiplied by the max value 
paramBC.firstPeakRatio = 1.1; % if units have an initial peak before the trough,
% in a units waveform to give the minimum prominence to detect peaks using
% matlab's findpeaks function.
paramBC.normalizeSpDecay = 1; % whether to normalize spatial decay points relative to 
% maximum - this makes the spatrial decay slop calculation more invariant to the 
% spike-sorting algorithm used

% recording parametrs
paramBC.ephys_sample_rate = 30000; % samples per second
paramBC.nChannels = 385; %number of recorded channels (including any sync channels)
% recorded in the raw data. This is usually 384 or 385 for neuropixels
% recordings
paramBC.nSyncChannels = 1;
if exist('ephysMetaDir','var') && ~isempty(ephysMetaDir)
    paramBC.ephysMetaFile = [ephysMetaDir.folder, filesep, ephysMetaDir.name];
    paramBC.gain_to_uV = NaN;
else
    paramBC.ephysMetaFile = 'NaN';
    if exist('gain_to_uV','var')
        paramBC.gain_to_uV = gain_to_uV;
    else
        paramBC.gain_to_uV = NaN;
    end
end
if exist('rawFile','var')
    paramBC.rawFile = rawFile;
else
    paramBC.rawFile = 'NaN';
end

% distance metric parameters
paramBC.computeDistanceMetrics = 0; % whether to compute distance metrics - this can be time consuming 
paramBC.nChannelsIsoDist = 4; % number of nearby channels to use in distance metric computation 


%% classifying units into good/mua/noise parameters 
paramBC.minAmplitude = 20; 
paramBC.maxRPVviolations = 0.1; % fraction
paramBC.maxPercSpikesMissing = 20; % Percentage
paramBC.minNumSpikes = 300;

paramBC.minSignalToNoiseRatio = 0.9; % JF: this should be removed unless you guys use it - bombcell does not. 
paramBC.maxDrift = 100;
paramBC.minPresenceRatio = 0.7;
paramBC.minSNR = 0.1;

%waveform 
paramBC.maxNPeaks = 2;
paramBC.maxNTroughs = 1;
paramBC.somatic = 1; 
paramBC.minWvDuration = 100; % in us
paramBC.maxWvDuration = 800; % in us
paramBC.minSpatialDecaySlope = -0.005;
paramBC.maxWvBaselineFraction = 0.3;

%distance metrics
paramBC.isoDmin = 20; 
paramBC.lratioMax = 0.1;
paramBC.ssMin = NaN; 

% split good and mua non-somatic
paramBC.splitGoodAndMua_NonSomatic = 1;

end
% %bc_qualityParamValues
% BCparam = struct;
% 
% % Plotting parameters
% BCparam.plotThis = 0;
% BCparam.plotGlobal = 1;
% BCparam.verbose=1; % update user on progress
% 
% % saving parameters
% BCparam.saveAsMat=0;
% BCparam.saveAsParquet = 1;
% 
% % Raw extraction parameters
% BCparam.reextractRaw=1; %Re extract raw waveforms -- should be done e.g. if number of templates change (Phy/Pyks)
% BCparam.nRawSpikesToExtract = 200; % You need enough waveforms for unitmatch
% BCparam.saveMultipleRaw = 1; % If you wish to save the nRawSpikesToExtract as well
% 
% % Waveform parameters
% BCparam.minAmplitude = 20; 
% BCparam.maxNPeaks = 2;
% BCparam.maxNTroughs = 1;
% BCparam.somatic = 1; 
% BCparam.minWvDuration = 100; %ms
% BCparam.maxWvDuration = 800; %ms
% BCparam.minSpatialDecaySlope = -20;
% BCparam.maxWvBaselineFraction = 0.3;
% BCparam.SpikeWidth=83; 
% 
% % refractory period parameters
% BCparam.tauR = 0.0020; %refractory period time (s)
% BCparam.tauC = 0.0001; %censored period time (s)
% BCparam.maxRPVviolations = 10;
% 
% % percentage spikes missing parameters 
% BCparam.maxPercSpikesMissing = 20;
% BCparam.computeTimeChunks = 1;
% BCparam.deltaTimeChunk = 1200; %time in seconds 
% 
% % presence ratio
% paramBC.presenceRatioBinSize = 60; % in seconds
% 
% % minimum number of spikes for a unit to be 'good'
% BCparam.minNumSpikes = 300;
% 
% % recording parametrs
% BCparam.ephys_sample_rate = 30000;
% BCparam.nChannels = 385;
% 
% % distance metric parameters
% BCparam.computeDistanceMetrics = 0;
% BCparam.nChannelsIsoDist = 4;
% BCparam.isoDmin = NaN;
% BCparam.lratioMin = NaN;
% BCparam.ssMin = NaN; 
% 
% % ACG parameters
% BCparam.ACGbinSize = 0.001;
% BCparam.ACGduration = 1;
% 
% % ISI parameters
% BCparam.longISI = 2;
% 
% % cell classification parameters
% BCparam.propISI = 0.1;
% BCparam.templateDuration = 400;
% BCparam.pss = 40;
% 
% 
% 