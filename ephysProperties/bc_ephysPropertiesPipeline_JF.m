function ephysProperties = bc_ephysPropertiesPipeline_JF(animal, day, site, recording, experiment, rerun, runEP, region)

% bc_qualityMetricsPipeline_JF('JF093','2023-03-06',1,[],1,[],1,1,1)

cl_myPaths;
experiments = AP_find_experimentsJF(animal,'', true);
experiments = experiments([experiments.ephys]);


experiment = experiments(experiment).experiment;

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
%% load ephys data 
paramEP = bc_ephysPropValues(ephysMetaDir, rawFile, '', '');
if ~decompress
    paramEP.extractRaw = 0;
end
%% compute ephys properties 
ephysDirPath = AP_cortexlab_filenameJF(animal, day, experiment, 'ephys_dir',site, recording);
savePath = fullfile(ephysDirPath, 'ephysProperties');
ephysPropertiesExist = dir(fullfile(savePath, 'templates._bc_ephysProperties.parquet'));

if (runEP && isempty(ephysPropertiesExist)) || rerun

    ephysPath = AP_cortexlab_filenameJF(animal,day,experiment,'ephys',site, recording);
    [spikeTimes_samples, spikeTemplates, templateWaveforms, templateAmplitudes, pcFeatures, ~, channelPositions] = bc_loadEphysData(ephysPath);
    winv = readNPY([ephysPath filesep 'whitening_mat_inv.npy']);
    ephysProperties = bc_computeAllEphysProperties(spikeTimes_samples, spikeTemplates, templateWaveforms,...
     templateAmplitudes, pcFeatures, channelPositions, paramEP, savePath);
elseif ~isempty(ephysPropertiesExist)
    [paramEP, ephysProperties, ~] = bc_loadSavedProperties(savePath); 
end

if ~isempty(region)
    % classify cells 
end

end