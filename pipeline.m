%% Pipeline 
%% 1. define parameters for quality metrics, ephys properties and
%% classification 
%define parameters
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


%% load experiment 
% you can use any other script to load your data, you need to end up with:
% xxxxx
animals={'AP024'};
curr_animal = 1; % (set which animal to use)
animal = animals{curr_animal};
protocol = 'vanillaChoiceworld'; % (this is the name of the Signals protocol)
experiments = AP_find_experimentsJF(animal, protocol, true);
experiments = experiments([experiments.imaging] & [experiments.ephys]);
curr_day = 1; % (set which day to use)
day = experiments(curr_day).day; % date
thisDay = experiments(curr_day).day; % date
thisDate = thisDay;
experiment = experiments(curr_day).experiment; % experiment number
load_parts.cam=false;
load_parts.imaging=true;
load_parts.ephys=true;

%loading
[ephys_path, ephys_exists] = AP_cortexlab_filenameJF(animal, day, experiment, 'ephys',[],[]);
%ephys_path = strcat(experiments(curr_day).location, '\ephys\kilosort2\');
corona = 0;
ephysData = loadEphysDataJF(ephys_path, animal, day, experiment); %load and format ephysData to use later 

%% run quality metrics 
getQualityMetrics;

%% run ephys properties
getEphysProperties;

keep qMetric ephysParams ephysData param 

%% classify cells 
classifyStriatum; 

%very quick plotting-just to check 
celltype_col = ...
    [0.9,0.4,0.6; ...
    0.4,0.5,0.6; ...
    0.5,0.3,0.1; ...
    1,0.5,0];
waveform_t = 1e3*((0:size(ephysData.template_waveforms,2)-1)/30000);
figure();
hold on;
for iCellType=1:4
    plot(waveform_t, nanmean(ephysData.template_waveforms(cellTypesClassif(iCellType).cells,:)),'Color',celltype_col(iCellType,:))
    plotshaded(waveform_t, [ nanmean(ephysData.template_waveforms(cellTypesClassif(iCellType).cells,:)) - ...
        nanstd(ephysData.template_waveforms(cellTypesClassif(iCellType).cells,:)); ...
        nanmean(ephysData.template_waveforms(cellTypesClassif(iCellType).cells,:)) + ...
        nanstd(ephysData.template_waveforms(cellTypesClassif(iCellType).cells,:))],celltype_col(iCellType,:))
end
makepretty;

figure();
hold on;
for iCellType=1:4
    subplot(1,4,iCellType)
    area(0:0.001:1, nanmean(ephysParams.ACG(cellTypesClassif(iCellType).cells,:)),'FaceColor',celltype_col(iCellType,:))
    plotshaded(0:0.001:1,[ nanmean(ephysParams.ACG(cellTypesClassif(iCellType).cells,:)) - ...
        nanstd(ephysParams.ACG(cellTypesClassif(iCellType).cells,:)); ...
        nanmean(ephysParams.ACG(cellTypesClassif(iCellType).cells,:)) + ...
        nanstd(ephysParams.ACG(cellTypesClassif(iCellType).cells,:))],celltype_col(iCellType,:))
end

% check waveform ampli 
unit = 1;
figure();
plot(waveform_t, ephysData.template_waveforms(unit,:),'Color','k')
ylabel('amplitude (a.u.)')
hold on; 
makepretty;
yyaxis right
%qMetric.waveformRaw(iUnit,:), qMetric.thisChannelRaw(iUnit)
plot(waveform_t, qMetric.waveformRaw(unit,:))%plot example cell 
legend({'template', ['raw, amplitude = ', num2str(qMetric.waveformRawAmpli(unit))]})
ylabel('amplitude (\muVolts)')
xlim([waveform_t(1), waveform_t(end)])
xlabel('time (ms)')
makepretty;

%save qualityMetrics, ephysProperties and classification