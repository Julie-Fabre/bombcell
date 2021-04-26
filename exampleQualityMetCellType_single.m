
%% example classify cell types
%  thisAnimal = 'AP025';
%  thisDate = '2017-09-30';
%  thisExperiment = 1;
%  curr_day=6;
cd('C:\Users\Julie\Dropbox\MATLAB\ephys_JF\qualityMetrics\')
% corona = 1; %local file or not


% protocol = 'vanillaChoiceworld'; % (this is the name of the Signals protocol)

ephys_path = strcat(experiments(curr_day).location, '\ephys\kilosort2\');
save_path = fullfile(experiments(curr_day).location, '/analysis/');

if ~exist(save_path, 'dir')
    mkdir(save_path)
end

%% parameters
param = struct;
param.dontdo = 0; %re-calulate metrics and ephysparams if they already exist
% for calulating qMetrics
param.plotThis = 0; %plot metrics/params for each unit
param.dist = 0; %calculate distance metrics or not (this takes >3 timeslonger with)
param.driftdo = 1; %calculate slow drift, and metrics for chunks of time with most spikes present
param.chunkBychunk = 0; %calulate metrics for each chunk
param.tauR = 0.0010; %refractory period time (s)
param.tauC = 0.0002; %censored period time (s)
param.nChannelsIsoDist = 4; %like tetrodes
param.chanDisMax = 300; %maximum distance
param.raw = 1; %calculate metrics also for raw data
param.strOnly = 0; %only use str_templates
% for calulating eParams
param.ACGbinSize = 0.001; %bin size to calc. ACG
param.ACGduration = 1; %ACG full duration
param.maxFRbin = 10; %
param.histBins = 1000;
% to choose good units
param.minNumSpikes = 300;
param.minIsoDist = 0;
param.minLratio = 0;
param.minSScore = 0.01;
param.minSpatDeKlowbound = 1.5;

param.maxNumPeak = 3;
param.minAmpli = 77;
param.maxRPV = 2;
param.somaCluster = 1;
param.plotMetricsCtypes = 0;
% for burst merging - WIP, not implemented yet
param.maxPercMissing = 30;
param.maxChanDistance = 40;
param.waveformMinSim = 0.8;
param.spikeMaxLab = 0.15;
param.minPeakRatio = 0.7;
param.maxdt = 10;
% for cell-type classification
param.cellTypeDuration = 400;
param.cellTypePostS = 40;

% parameters to load raw data
raw.n_channels = 384; % number of channels
raw.ephys_datatype = 'int16';
raw.kp = strcat(ephys_path(1:end), '..');
raw.dat = dir([raw.kp, filesep, '*.dat']);
if isempty(raw.dat)
    raw.kp = strcat(ephys_path(1:end), '\..\experiment1\recording1\continuous\Neuropix-3a-100.0');
    raw.dat = dir([raw.kp, filesep, '*.dat']);
    %raw.dat = dir([raw.kp, filesep, '*.bin']);
end

raw.ephys_ap_filename = [raw.dat(1).folder, filesep, raw.dat(1).name];
raw.ap_dat_dir = dir(raw.ephys_ap_filename);
raw.pull_spikeT = -40:41; % number of points to pull for each waveform
raw.microVoltscaling = 0.19499999284744263; %in structure.oebin for openephys, this never changed so hard-coded here-not loading it in.

raw.dataTypeNBytes = numel(typecast(cast(0, raw.ephys_datatype), 'uint8'));
raw.n_samples = raw.ap_dat_dir.bytes / (raw.n_channels * raw.dataTypeNBytes);
raw.ap_data = memmapfile(raw.ephys_ap_filename, 'Format', {raw.ephys_datatype, [raw.n_channels, raw.n_samples], 'data'});
raw.max_pull_spikes = 200; % number of spikes to pull

%%run function
[qMetric, ephysParams, ephysData, behavData, goodUnits, msn, fsi, tan] = classifyUnitQualityCellTypeJF(save_path, ephys_path, thisAnimal, thisDate, thisExperiment, corona, param, raw, trained);
