function param = bc_qualityParamValues(ephysMetaDir, rawFile)
% JF, Load a parameter structure defining extraction and
% classification parameters
% ------
% Inputs
% ------
% ephysMetaDir: dir() structure of the path to your .meta or .oebin meta
%   file
% rawFile: character array defining the path where your uncompressed raw
%   ephys data is
% ------
% Outputs
% ------
% param: matlab structure defining extraction and
% classification parameters
% 

param = struct; %initialize structure 

%% calculating quality metrics parameters 
% plotting parameters 
param.plotDetails = 0; % generates a lot of plots, 
% mainly good if you running through the code line by line to check things,
% to debug, or to get nice plots for a presentation
param.plotGlobal = 1; % plot summary of quality metrics 
param.verbose = 1; % update user on progress
param.reextractRaw = 0; % re extract raw waveforms or not 

% saving parameters 
param.saveAsParquet = 1; % save outputs at .parquet file 
param.saveMatFileForGUI = 0; % save certain outputs at .mat file - useful for GUI

% amplitude parameters
param.nRawSpikesToExtract = 100; % how many raw spikes to extract for each unit 
param.saveMultipleRaw = 0; % If you wish to save the nRawSpikesToExtract as well, 
% currently needed if you want to run unit match https://github.com/EnnyvanBeest/UnitMatch
% to track chronic cells over days after this
param.decompressData = 1; % whether to decompress .cbin ephys data 
param.spikeWidth = 82; % width in samples 

% signal to noise ratio
param.waveformBaselineNoiseWindow = 20; %time in samples at beginning of times
% extracted to computer the mean raw waveform - this needs to be before the
% waveform starts 

% refractory period parameters
param.tauR_valuesMin = 0.5/1000; % refractory period time (s), usually 0.0020
param.tauR_valuesStep = 0.5./1000; % refractory period time (s), usually 0.0020
param.tauR_valuesMax = 10./1000; % refractory period time (s), usually 0.0020
param.tauC = 0.1/1000; % censored period time (s)

% percentage spikes missing parameters 
param.computeTimeChunks = 1; % compute fraction refractory period violations 
% and percent sp[ikes missing for different time chunks 
param.deltaTimeChunk = 360; %time in seconds 

% presence ratio 
param.presenceRatioBinSize = 60; % in seconds 

% drift estimate
param.driftBinSize = 60; % in seconds

% waveform parameters
param.waveformBaselineWindowStart = 20;
param.waveformBaselineWindowStop = 30; % in samples 
param.minThreshDetectPeaksTroughs = 0.2; % this is multiplied by the max value 
% in a units waveform to give the minimum prominence to detect peaks using
% matlab's findpeaks function.

% recording parametrs
param.ephys_sample_rate = 30000; % samples per second
param.nChannels = 385; %number of recorded channels (including any sync channels)
% recorded in the raw data. This is usually 384 or 385 for neuropixels
% recordings
param.nSyncChannels = 1;
param.ephysMetaFile = [ephysMetaDir.folder, filesep, ephysMetaDir.name];
param.rawFile = rawFile;

% distance metric parameters
param.computeDistanceMetrics = 0; % whether to compute distance metrics - this can be time consuming 
param.nChannelsIsoDist = 4; % number of nearby channels to use in distance metric computation 


%% classifying units into good/mua/noise parameters 
param.minAmplitude = 20; 
param.maxRPVviolations = 0.1; % fraction
param.maxPercSpikesMissing = 20;
param.minNumSpikes = 300;

param.minSignalToNoiseRatio = 0.9;
param.maxDrift = 10;
param.minPresenceRatio = 0.7;
param.minSNR = 2;

%waveform 
param.maxNPeaks = 2;
param.maxNTroughs = 1;
param.somatic = 1; 
param.minWvDuration = 100; % in us
param.maxWvDuration = 800; % in us
param.minSpatialDecaySlope = -0.003;
param.maxWvBaselineFraction = 0.3;

%distance metrics
param.isoDmin = 20; 
param.lratioMax = 0.1;
param.ssMin = NaN; 
end