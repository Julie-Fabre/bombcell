%% set paths 
ephysKilosortPath = '/home/netshare/zinu/JF078/2022-05-25/ephys/kilosort2/site2';% eg /home/netshare/zinu/JF067/2022-02-17/ephys/kilosort2/site1, where this path contains 
                                           % kilosort output
ephysDirPath = '/home/netshare/zinu/JF078/2022-05-25/ephys/site2'; % where your 
ephysRawDir = dir('/home/netshare/zinu/JF078/2022-05-25/ephys/site2/2022_05_25-JF078-1_g0_t0.imec1.ap.*bin');
ephysMetaDir = dir('/home/netshare/zinu/JF078/2022-05-25/ephys/site2/2022_05_25-JF078-1_g0_t0.imec1.ap.*meta');
decompressDataLocal = '/media/julie/ExtraHD/decompressedData';%'/media/julie/Elements/JF078/2022-05-25/ephys';%/media/julie/ExtraHD/decompressedData';
savePath = fullfile(ephysDirPath, 'qMetrics'); 

%% load data 
[spikeTimes_samples, spikeTemplates, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysKilosortPath);

%% which quality metric parameters to extract and thresholds 
bc_qualityParamValues; 

%% detect whether data is compressed, decompress locally if necessary
if strcmp(ephysRawDir.name(end-4:end), '.cbin') &&...
        isempty(dir([decompressDataLocal, filesep, ephysRawDir.name(1:end-5), '.bin']))
    fprintf('Decompressing ephys data file %s locally to %s... \n', ephysRawDir.name, decompressDataLocal)
    
    decompDataFile = bc_extractCbinData([ephysRawDir.folder, filesep, ephysRawDir.name],...
        [], [], [], decompressDataLocal);
    param.rawFile = decompDataFile;
elseif strcmp(ephysRawDir.name(end-4:end), '.cbin') &&...
        ~isempty(dir([decompressDataLocal, filesep, ephysRawDir.name(1:end-5), '.bin']))
    fprintf('Using previously decompressed ephys data file in %s ... \n', decompressDataLocal)
    
    param.rawFile = [decompressDataLocal, filesep, ephysRawDir.name(1:end-5), '.bin'];
else
    param.rawFile = [ephysRawDir.folder, filesep, ephysRawDir.name];
end

%% compute quality metrics 
rerun = 1;
qMetricsExist = ~isempty(dir(fullfile(savePath, 'qMetric*.mat'))) || ~isempty(dir(fullfile(savePath, 'templates._bc_qMetrics.parquet')));

if qMetricsExist == 0 || rerun
    qMetric = bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
        templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);
else
    bc_loadSavedMetrics; 
    bc_getQualityUnitType;
end
%% view units + quality metrics in GUI 
% load data for GUI
bc_loadMetricsForGUI;

% GUI guide: 
% left/right arrow: toggle between units 
% g : go to next good unit 
% m : go to next multi-unit 
% n : go to next noise unit 
% up/down arrow: toggle between time chunks in the raw data
% u: brings up a input dialog to enter the unit you want to go to 
 
bc_unitQualityGUI(memMapData, ephysData, qMetric, rawWaveforms, param,...
    probeLocation, unitType, plotRaw);


