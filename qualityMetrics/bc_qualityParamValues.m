%bc_qualityParamValues
param = struct;
param.plotThis = 0;
param.plotGlobal =1;
% refractory period parameters
param.tauR = 0.0020; %refractory period time (s)
param.tauC = 0.0001; %censored period time (s)
param.maxRPVviolations = 10;
% percentage spikes missing parameters 
param.maxPercSpikesMissing = 20;
param.computeTimeChunks = 0;
param.deltaTimeChunk = NaN; 
% number of spikes
param.minNumSpikes = 300;
% waveform parameters
param.maxNPeaks = 2;
param.maxNTroughs = 1;
param.somatic = 1; 
param.minWvDuration = 100; %ms
param.maxWvDuration = 800; %ms
param.minSpatialDecaySlope = -20;
param.maxWvBaselineFraction = 0.3;
% amplitude parameters
param.rawFolder = [ephysap_path, '/..'];
param.nRawSpikesToExtract = 100; 
param.minAmplitude = 40; 
% recording parametrs
param.ephys_sample_rate = 30000;
param.nChannels = 385;
% distance metric parameters
param.computeDistanceMetrics = 0;
param.nChannelsIsoDist = 4;
param.isoDmin = NaN;
param.lratioMin = NaN;
param.ssMin = NaN; 
% ACG parameters
param.ACGbinSize = 0.001;
param.ACGduration = 1;
% ISI parameters
param.longISI = 2;
% cell classification parameters
param.propISI = 0.1;
param.templateDuration = 400;
param.pss = 40;
