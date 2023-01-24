%% set paths 
ephysKilosortPath = '/home/netshare/zinu/JF078/2022-05-25/ephys/kilosort2/site2';% path to your kilosort output files 
ephysRawDir = dir('/home/netshare/zinu/JF078/2022-05-25/ephys/site2/2022_05_25-JF078-1_g0_t0.imec1.ap.*bin'); % path to yourraw .bin or .dat data
ephysMetaDir = dir('/home/netshare/zinu/JF078/2022-05-25/ephys/site2/2022_05_25-JF078-1_g0_t0.imec1.ap.*meta'); % path to your meta file
saveLocation = '/home/netshare/zinu/JF078/2022-05-25/ephys/site2'; % where you want to save the quality metrics 
savePath = fullfile(saveLocation, 'qMetrics'); 
decompressDataLocal = '/media/julie/ExtraHD/decompressedData'; % where to save raw decompressed ephys data 

%% load data 
[spikeTimes_samples, spikeTemplates, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysKilosortPath);

%% detect whether data is compressed, decompress locally if necessary
if strcmp(ephysRawDir.name(end-4:end), '.cbin') &&...
        isempty(dir([decompressDataLocal, filesep, ephysRawDir.name(1:end-5), '.bin']))
    fprintf('Decompressing ephys data file %s locally to %s... \n', ephysRawDir.name, decompressDataLocal)
    
    decompDataFile = bc_extractCbinData([ephysRawDir.folder, filesep, ephysRawDir.name],...
        [], [], [], decompressDataLocal);
    rawFile = decompDataFile;
elseif strcmp(ephysRawDir.name(end-4:end), '.cbin') &&...
        ~isempty(dir([decompressDataLocal, filesep, ephysRawDir.name(1:end-5), '.bin']))
    fprintf('Using previously decompressed ephys data file in %s ... \n', decompressDataLocal)
    
    rawFile = [decompressDataLocal, filesep, ephysRawDir.name(1:end-5), '.bin'];
else
    rawFile = [ephysRawDir.folder, filesep, ephysRawDir.name];
end

%% which quality metric parameters to extract and thresholds 
param = bc_qualityParamValues(ephysMetaDir, rawFile); 

%% compute quality metrics 
rerun = 0;
qMetricsExist = ~isempty(dir(fullfile(savePath, 'qMetric*.mat'))) || ~isempty(dir(fullfile(savePath, 'templates._bc_qMetrics.parquet')));

if qMetricsExist == 0 || rerun
    [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
        templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);
else
    [param, qMetric] = bc_loadSavedMetrics(savePath); 
    unitType = bc_getQualityUnitType(param, qMetric);
end

%% view units + quality metrics in GUI 
% load data for GUI
%bc_loadMetricsForGUI;

% GUI guide: 
% left/right arrow: toggle between units 
% g : go to next good unit 
% m : go to next multi-unit 
% n : go to next noise unit 
% up/down arrow: toggle between time chunks in the raw data
% u: brings up a input dialog to enter the unit you want to go to 
 
%bc_unitQualityGUI(memMapData, ephysData, qMetric, rawWaveforms, param,...
%    probeLocation, unitType, plotRaw);


