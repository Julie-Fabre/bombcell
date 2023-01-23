%% load data 
animals = {'JF089'};
bhvProtocol = '';

animal = animals{1};
experiments = AP_find_experimentsJF(animal, bhvProtocol, true);
experiments = experiments([experiments.ephys]);

day = experiments(1).day;
experiment = experiments(1).experiment;
site = 1;

ephysPath = AP_cortexlab_filenameJF(animal,day,experiment,'ephys',site);
[spikeTimes_samples, spikeTemplates, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysPath);
ephysap_path = AP_cortexlab_filenameJF(animal,day,experiment,'ephys_ap',site);
%ephysMetaDir = dir([ephysap_path, '/../../../structure.oebin']);
ephysMetaDir = dir([ephysap_path, '/../*ap.meta']);
ephysDirPath = AP_cortexlab_filenameJF(animal,day,experiment,'ephys_dir',site);
savePath = fullfile(ephysDirPath, 'qMetrics'); 

%% run qmetrics 
param = bc_qualityParamValues(ephysMetaDir, ephysap_path); 


%% compute quality metrics 
ephysDirPath = AP_cortexlab_filenameJF(animal, day, experiment, 'ephys_dir',site);
savePath = fullfile(ephysDirPath, 'qMetrics');
qMetricsExist = dir(fullfile(savePath, 'qMetric*.mat'));

if isempty(qMetricsExist) || rerun
    [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
        templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);
else
    % load, classify 
end

%% unit quality GUI 

% load gui stuffs 

unitQualityGUI(ap_data.data.data,ephysData,qMetric, param, probeLocation, unitType, plotRaw);
