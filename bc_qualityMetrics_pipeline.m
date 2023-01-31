%% set paths 
ephysKilosortPath = '/home/julie/Downloads/AL032/';% path to your kilosort output files 
ephysRawDir = dir('/home/netshare/znas/AL032/2019-11-21/ephys_K1/*.*bin'); % path to yourraw .bin or .dat data
ephysMetaDir = dir('/home/netshare/znas/AL032/2019-11-21/ephys_K1/*.*meta'); % path to your meta file
saveLocation = '/media/julie/ExtraHD/AL032/2019-11-21/'; % where you want to save the quality metrics 
savePath = fullfile(saveLocation, 'qMetrics'); 
decompressDataLocal = '/media/julie/ExtraHD/decompressedData'; % where to save raw decompressed ephys data 

%% load data 
[spikeTimes_samples, spikeTemplates, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysKilosortPath);

%% detect whether data is compressed, decompress locally if necessary
rawFile = bc_manageDataCompression(ephysRawDir, decompressDataLocal);

%% which quality metric parameters to extract and thresholds 
param = bc_qualityParamValues(ephysMetaDir, rawFile); 
% param = bc_qualityParamValuesForUnitMatch(ephysMetaDir, rawFile) % Run
% this if you want to use UnitMatch after
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
bc_loadMetricsForGUI;

% GUI guide: 
% left/right arrow: toggle between units 
% g : go to next good unit 
% m : go to next multi-unit 
% n : go to next noise unit 
% up/down arrow: toggle between time chunks in the raw data
% u: brings up a input dialog to enter the unit you want to go to 
 
%bc_unitQualityGUI(memMapData, ephysData, qMetric, rawWaveforms, param,...
%    probeLocation, unitType, plotRaw);


