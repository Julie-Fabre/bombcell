%% ~~ Example bombcell pipeline ~~
% Adjust the paths in the 'set paths' section and the parameters in bc_qualityParamValues
% This pipeline will (1) load your ephys data, (2) decompress your raw data if it
% is in .cbin format (3) run bombcell on your data and save the output and
% finally (4) bring up summary plots and a GUI to flip through classifyed
% cells.
% The first time, this pipeline will be significantly slower (10-20' more)
% than after because it extracts raw waveforms. Subsequent times these
% pre-extracted waveforms are simply loaded in.
% We recomment running this pipeline on a few datasets and deciding on
% quality metric thresholds depending on the histogram and GUI. 


%% set paths 
%\\zinu.cortexlab.net\Subjects\CB016\2021-10-07\ephys\CB016_2021-10-07_NatImages_g0\CB016_2021-10-07_NatImages_g0
ephysKilosortPath = '/home/netshare/zinu/CB016/2021-10-07/ephys/CB016_2021-10-07_NatImages_g0/CB016_2021-10-07_NatImages_g0/pyKS/output/';% path to your kilosort output files 
ephysRawDir = dir('/home/netshare/zinu/CB016/2021-10-07/ephys/CB016_2021-10-07_NatImages_g0/CB016_2021-10-07_NatImages_g0/*.*bin'); % path to yourraw .bin or .dat data
ephysMetaDir = dir('/home/netshare/zinu/CB016/2021-10-07/ephys/CB016_2021-10-07_NatImages_g0/CB016_2021-10-07_NatImages_g0/*.*meta'); % path to your meta file
saveLocation = '/media/julie/ExtraHD/CB016'; % where you want to save the quality metrics 
savePath = fullfile(saveLocation, 'qMetrics'); 
decompressDataLocal = '/media/julie/ExtraHD/decompressedData'; % where to save raw decompressed ephys data 

%% load data 
[spikeTimes_samples, spikeTemplates, templateWaveforms, templateAmplitudes, pcFeatures, ...
    pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysKilosortPath);

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
loadRawTraces = 0;
bc_loadMetricsForGUI;

% GUI guide: 
% left/right arrow: toggle between units 
% g : go to next good unit 
% m : go to next multi-unit 
% n : go to next noise unit
% up/down arrow: toggle between time chunks in the raw data
% u: brings up a input dialog to enter the unit you want to go to
unitQualityGuiHandle = bc_unitQualityGUI(memMapData, ephysData, qMetric, forGUI, rawWaveforms, ...
    param, probeLocation, unitType, loadRawTraces);


