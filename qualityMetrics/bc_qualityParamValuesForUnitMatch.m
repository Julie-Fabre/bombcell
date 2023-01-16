%bc_qualityParamValues
BCparam = struct;
BCparam.plotThis = 0;
BCparam.plotGlobal = 1;
BCparam.verbose=1; % update user on progress
BCparam.reextractRaw=1; %Re extract raw waveforms -- should be done e.g. if number of templates change (Phy/Pyks)
% refractory period parameters
BCparam.tauR = 0.0020; %refractory period time (s)
BCparam.tauC = 0.0001; %censored period time (s)
BCparam.maxRPVviolations = 10;
% percentage spikes missing parameters 
BCparam.maxPercSpikesMissing = 20;
BCparam.computeTimeChunks = 1;
BCparam.deltaTimeChunk = 1200; %time in seconds 
% number of spikes
BCparam.minNumSpikes = 300;
% waveform parameters
BCparam.maxNPeaks = 2;
BCparam.maxNTroughs = 1;
BCparam.somatic = 1; 
BCparam.minWvDuration = 100; %ms
BCparam.maxWvDuration = 800; %ms
BCparam.minSpatialDecaySlope = -20;
BCparam.maxWvBaselineFraction = 0.3;
BCparam.SpikeWidth=83; 
% amplitude parameters

BCparam.nRawSpikesToExtract = 200; % You need enough waveforms for unitmatch
BCparam.saveMultipleRaw = 1; % If you wish to save the nRawSpikesToExtract as well
BCparam.minAmplitude = 20; 
% recording parametrs
BCparam.ephys_sample_rate = 30000;
BCparam.nChannels = 385;
% distance metric parameters
BCparam.computeDistanceMetrics = 0;
BCparam.nChannelsIsoDist = 4;
BCparam.isoDmin = NaN;
BCparam.lratioMin = NaN;
BCparam.ssMin = NaN; 
% ACG parameters
BCparam.ACGbinSize = 0.001;
BCparam.ACGduration = 1;
% ISI parameters
BCparam.longISI = 2;
% cell classification parameters
BCparam.propISI = 0.1;
BCparam.templateDuration = 400;
BCparam.pss = 40;
% Rule to extract max chan
BCparam.rawWaveformMaxDef = 'firstSTD';

