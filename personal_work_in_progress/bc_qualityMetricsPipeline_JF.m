function [unitType, qMetric] = bc_qualityMetricsPipeline_JF(animal, day, site, recording, experiment_num, protocol, rerunQM, plotGUI, runQM)

%% load data 
% bc_qualityMetricsPipeline_JF('JF093','2023-03-06',1,[],1,[],1,1,1)

cl_myPaths;
experiments = AP_find_experimentsJF(animal, protocol, true);
experiments = experiments([experiments.ephys]);


experiment = experiments(experiment_num).experiment;

ephysPath = AP_cortexlab_filenameJF(animal,day,experiment,'ephys',site,recording);
[spikeTimes_samples, spikeTemplates, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysPath);
ephysap_path = dir(AP_cortexlab_filenameJF(animal,day,experiment,'ephys_includingCompressed',site, recording));

%ephysMetaDir = dir([ephysap_path, '/../../../structure.oebin']);
ephysMetaDir = dir([ephysap_path.folder, filesep, '*ap.meta']);
ephysDirPath = AP_cortexlab_filenameJF(animal,day,experiment,'ephys_dir',site, recording);
savePath = fullfile(ephysDirPath, 'qMetrics'); 
saveFileFolder = fullfile(extraHDPath, animal, day, ['site', num2str(site)]);

%% decompress data 
decompress = 0;%QQ
if decompress
    rawFile = bc_manageDataCompression(ephysap_path, saveFileFolder);
else
    rawFile = 'NaN';

end
%% run qmetrics 
param = bc_qualityParamValues_JF(ephysMetaDir, rawFile, '', ''); 
if ~decompress
    param.extractRaw = 0;
end
%param.computeDistanceMetrics = 1;

%% compute quality metrics 
ephysDirPath = AP_cortexlab_filenameJF(animal, day, experiment, 'ephys_dir',site, recording);
savePath = fullfile(ephysDirPath, 'qMetrics');
qMetricsExist = dir(fullfile(savePath, 'templates._bc_qMetrics.parquet'));

if isempty(qMetricsExist)% try diff location
    ksDirPath = AP_cortexlab_filenameJF(animal, day, experiment, 'ephys',site, recording);

    savePath = ksDirPath;
    qMetricsExist = dir(fullfile(ksDirPath, 'templates._bc_qMetrics.parquet'));
end

if(runQM && isempty(qMetricsExist)) || rerunQM
    try
        [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
            templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);
    catch
        param = bc_qualityParamValues_noRaw(ephysMetaDir, rawFile, '', ''); 
        [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
            templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);
    end
elseif ~isempty(qMetricsExist)
    [param, qMetric, fractionRPVs_allTauR] = bc_loadSavedMetrics(savePath); 
    unitType = bc_getQualityUnitType(param, qMetric);
else
    uniqueTemplates = unique(spikeTemplates);
    unitType = nan(length(uniqueTemplates),1);
end

%% view units + quality metrics in GUI 
% load data for GUI
if plotGUI
    bc_loadMetricsForGUI;
    param.extractRaw= 0;
    bc_unitQualityGUI(memMapData, ephysData, qMetric, forGUI, rawWaveforms, param,...
       probeLocation, unitType, 0);
end
end