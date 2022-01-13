%% classify striatal cell types as in Peters et al., 2021


%% load dataset 
animal = 'JF020';
experiments = AP_find_experimentsJF(animal, 'rating', 1);
experiments =experiments(1); 
[ephys_path, ~] = AP_cortexlab_filenameJF(animal, experiments.day, experiments.experiment(2), 'ephys', 1);
[ephysap_path, ~] = AP_cortexlab_filenameJF(animal, experiments.day, experiments.experiment(2), 'ephys_ap', 1);

    
templates = readNPY([ephys_path, filesep, 'templates.npy']);
channel_positions = readNPY([ephys_path, filesep, 'channel_positions.npy']);
spike_times = double(readNPY([ephys_path, filesep, 'spike_times.npy'])); % sample rate hard-coded as 30000 - should load this in from params 
spike_templates = readNPY([ephys_path, filesep, 'spike_templates.npy']) + 1; % 0-idx -> 1-idx
template_amplitude = readNPY([ephys_path, filesep, 'amplitudes.npy']);
spike_clusters = readNPY([ephys_path, filesep, 'spike_clusters.npy']) + 1;
pc_features = readNPY([ephys_path, filesep, 'pc_features.npy']) ;
pc_feature_ind = readNPY([ephys_path, filesep, 'pc_feature_ind.npy']) + 1;


%% parameters
param = struct;
param.plotThis = 0;
% refractory period parameters
param.tauR = 0.0010; %refractory period time (s)
param.tauC = 0.0002; %censored period time (s)
param.maxRPVviolations = 0.2;
% percentage spikes missing parameters 
param.maxPercSpikesMissing = 30;
param.computeTimeChunks = 0;
param.deltaTimeChunk = NaN; 
% number of spikes
param.minNumSpikes = 300;
% waveform parameters
param.maxNPeaks = 2;
param.maxNTroughs = 1;
param.axonal = 0; 
% amplitude parameters
param.rawFolder = [ephysap_path, '/..'];
param.nRawSpikesToExtract = 100; 
param.minAmplitude = 20; 
% recording parametrs
param.ephys_sample_rate = 30000;
param.nChannels = 385;
% distance metric parameters
param.computeDistanceMetrics = 0;
param.nChannelsIsoDist = NaN;
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
%% compute quality metrics 
[qMetric, goodUnits] = bc_runAllQualityMetrics(param, spike_times, spike_templates, ...
    templates, template_amplitude,pc_features,pc_feature_ind);

%% compute ephys properties 
ephysProp = bc_computeAllEphysProperties(spike_times, spike_templates, templates, param);

%% classify striatal cells 
cellTypes = bc_classifyStriatalCells(ephysProp, qMetric, param);