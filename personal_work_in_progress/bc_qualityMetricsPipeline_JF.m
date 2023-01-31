%% load data 
animals = {'JF070'};
bhvProtocol = '';

animal = animals{1};
experiments = AP_find_experimentsJF(animal, bhvProtocol, true);
experiments = experiments([experiments.ephys]);

day = experiments(end).day;
experiment = experiments(1).experiment;
site = 1;

ephysPath = AP_cortexlab_filenameJF(animal,day,experiment,'ephys',site);
[spikeTimes_samples, spikeTemplates, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysPath);
ephysap_path = dir(AP_cortexlab_filenameJF(animal,day,experiment,'ephys_includingCompressed',site));
%ephysMetaDir = dir([ephysap_path, '/../../../structure.oebin']);
ephysMetaDir = dir([ephysap_path.folder, '/../*ap.meta']);
ephysDirPath = AP_cortexlab_filenameJF(animal,day,experiment,'ephys_dir',site);
savePath = fullfile(ephysDirPath, 'qMetrics'); 
decompressDataLocal = '/media/julie/ExtraHD/decompressedData'; % where to save raw decompressed ephys data 

%% decompress data 
rawFile = bc_manageDataCompression(ephysap_path, decompressDataLocal);

%% run qmetrics 
param = bc_qualityParamValues(ephysMetaDir, rawFile); 
%param.computeDistanceMetrics = 1;

%% compute quality metrics 
ephysDirPath = AP_cortexlab_filenameJF(animal, day, experiment, 'ephys_dir',site);
savePath = fullfile(ephysDirPath, 'qMetrics');
qMetricsExist = dir(fullfile(savePath, 'qMetric*.mat'));

if isempty(qMetricsExist) || rerun
    [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
        templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);
else
    [param, qMetric, fractionRPVs_allTauR] = bc_loadSavedMetrics(savePath); 
    unitType = bc_getQualityUnitType(param, qMetric);
end

%% view units + quality metrics in GUI 
% load data for GUI
bc_loadMetricsForGUI;


bc_unitQualityGUI(memMapData, ephysData, qMetric, forGUI, rawWaveforms, param,...
   probeLocation, unitType, plotRaw);
