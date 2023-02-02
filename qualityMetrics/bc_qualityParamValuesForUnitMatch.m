function paramBC = bc_qualityParamValuesForUnitMatch(ephysMetaDir, rawFile)
paramBC = struct;

%% calculating quality metrics parameters 
% plotting parameters 
paramBC.plotDetails = 0; % generates a lot of plots, 
% mainly good if you running through the code line by line to check things,
% to debug, or to get nice plots for a presentation
paramBC.plotGlobal = 1; % plot summary of quality metrics 
paramBC.verbose = 1; % update user on progress
paramBC.reextractRaw = 0; % re extract raw waveforms or not 

% saving parameters 
paramBC.saveAsParquet = 1; % save outputs at .parquet file 
paramBC.saveMatFileForGUI = 1; % save certain outputs at .mat file - useful for GUI

% amplitude parameters
paramBC.nRawSpikesToExtract = 200; % how many raw spikes to extract for each unit 
paramBC.saveMultipleRaw = 1; % If you wish to save the nRawSpikesToExtract as well, 
% currently needed if you want to run unit match https://github.com/EnnyvanBeest/UnitMatch
% to track chronic cells over days after this
paramBC.decompressData = 1; % whether to decompress .cbin ephys data 
paramBC.spikeWidth = 82; % width in samples 

% signal to noise ratio
paramBC.waveformBaselineNoiseWindow = 20; %time in samples at beginning of times
% extracted to computer the mean raw waveform - this needs to be before the
% waveform starts 

% refractory period parameters
paramBC.tauR_valuesMin = 0.5/1000; % refractory period time (s), usually 0.0020
paramBC.tauR_valuesStep = 0.5./1000; % refractory period time (s), usually 0.0020
paramBC.tauR_valuesMax = 10./1000; % refractory period time (s), usually 0.0020
paramBC.tauC = 0.1/1000; % censored period time (s)

% percentage spikes missing parameters 
paramBC.computeTimeChunks = 1; % compute fraction refractory period violations 
% and percent sp[ikes missing for different time chunks 
paramBC.deltaTimeChunk = 360; %time in seconds 

% presence ratio 
paramBC.presenceRatioBinSize = 60; % in seconds 

% drift estimate
paramBC.driftBinSize = 60; % in seconds
paramBC.computeDrift = 0; % whether to compute each units drift. this is a 
% critically slow step that takes around 2seconds per unit 

% waveform parameters
paramBC.waveformBaselineWindowStart = 20;
paramBC.waveformBaselineWindowStop = 30; % in samples 
paramBC.minThreshDetectPeaksTroughs = 0.2; % this is multiplied by the max value 
% in a units waveform to give the minimum prominence to detect peaks using
% matlab's findpeaks function.

% recording parametrs
paramBC.ephys_sample_rate = 30000; % samples per second
paramBC.nChannels = 385; %number of recorded channels (including any sync channels)
% recorded in the raw data. This is usually 384 or 385 for neuropixels
% recordings
paramBC.nSyncChannels = 1;
paramBC.ephysMetaFile = [ephysMetaDir.folder, filesep, ephysMetaDir.name];
paramBC.rawFile = rawFile;

% distance metric parameters
paramBC.computeDistanceMetrics = 0; % whether to compute distance metrics - this can be time consuming 
paramBC.nChannelsIsoDist = 4; % number of nearby channels to use in distance metric computation 


%% classifying units into good/mua/noise parameters 
paramBC.minAmplitude = 20; 
paramBC.maxRPVviolations = 0.1; % fraction
paramBC.maxPercSpikesMissing = 20;
paramBC.minNumSpikes = 300;

paramBC.minSignalToNoiseRatio = 0.9;
paramBC.maxDrift = 100;
paramBC.minPresenceRatio = 0.7;
paramBC.minSNR = 0.1;

%waveform 
paramBC.maxNPeaks = 2;
paramBC.maxNTroughs = 1;
paramBC.somatic = 1; 
paramBC.minWvDuration = 100; % in us
paramBC.maxWvDuration = 800; % in us
paramBC.minSpatialDecaySlope = -0.003;
paramBC.maxWvBaselineFraction = 0.3;

%distance metrics
paramBC.isoDmin = 20; 
paramBC.lratioMax = 0.1;
paramBC.ssMin = NaN; 
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
% param.presenceRatioBinSize = 60; % in seconds
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
