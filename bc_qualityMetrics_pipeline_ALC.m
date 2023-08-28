%% ~~ bombcell pipeline ~~
% Adjust the paths in the 'set paths' section and the parameters in bc_qualityParamValues
% This pipeline will:
%   (1) load your ephys data, 
%   (2) decompress your raw data if it is in .cbin format 
%   (3) run bombcell on your data and save the output and
%   (4) bring up summary plots and a GUI to flip through classified cells.
% The first time, this pipeline will be significantly slower (10-20' more)
% than after because it extracts raw waveforms. Subsequent times these
% pre-extracted waveforms are simply loaded in.
% We recommend running this pipeline on a few datasets and deciding on
% quality metric thresholds depending on the summary plots (histograms 
% of the distributions of quality metrics for each unit) and GUI. 

folder_bombcell   = [ userpath '\Alessandro\bombcell']; % for reading ephys metadata
folder_npymatlab  = [ userpath '\Alessandro\npy-matlab']; % for reading ephys metadata
folder_oephystool = [ userpath '\Alessandro\open-ephys-analysis-tools']; % for reading ephys metadata
folder_nyumatlab  = [ userpath '\Alessandro\Code-NYU_MATLAB']; % for reading ephys metadata
addpath(genpath(folder_bombcell));
addpath(genpath(folder_npymatlab));
addpath(genpath(folder_oephystool));
addpath(genpath(folder_nyumatlab));

%%

animalID        = 'AL211110a';
experimentDate  = []; % no need to specify if only one ephys day for that mouse
load_data = false;
[~, experimentDate, datafile_fullpath] = load_datafile('syncedData', [], animalID, experimentDate, [], load_data);

ephysDataRootDir  = ['Z:\Users\Alessandro La Chioma\Data\' animalID filesep 'DataEphys' filesep experimentDate];
ephysKilosortPath = fullfile(ephysDataRootDir, 'KilosortOutput_experiment1_recording1');
ephysRawDir       = dir([ephysDataRootDir filesep 'Raw\' animalID '_' experimentDate '*\**\Record Node *\**\continuous\Neuropix-PXI-100.0\continuous.dat']);
if isempty(ephysRawDir)
    ephysRawDir  = dir([ephysDataRootDir filesep 'Raw\' animalID '_' experimentDate '*\**\Record Node *\**\continuous\Neuropix-PXI-100.ProbeA-AP\continuous.dat']);
end
assert(length(ephysRawDir)==1,'No or multiple continuous.dat files found (it must be just 1)')
ephysMetaDir      = dir([ephysDataRootDir filesep 'Raw\' animalID '_' experimentDate '*\**\*.oebin']);
saveLocation      = fullfile(ephysDataRootDir, 'QualityMetrics_experiment1_recording1');
savePath = fullfile(saveLocation, 'qMetrics'); 
% decompressDataLocal = '/media/julie/ExtraHD/decompressedData'; % where to save raw decompressed ephys data 
decompressDataLocal = '';

% %% set paths - EDIT THESE 
% ephysKilosortPath = 'Z:\Users\Alessandro La Chioma\Data\AL221025b\DataEphys\2022-11-08\KilosortOutput_experiment1_recording1/';% path to your kilosort output files 
% ephysRawDir = dir('Z:\Users\Alessandro La Chioma\Data\AL221025b\DataEphys\2022-11-08\Raw\AL221025b_2022-11-08_10-39-01\Record Node 102\experiment1\recording1\continuous\Neuropix-PXI-100.0/continuous.dat'); % path to yourraw .bin or .dat data
% ephysMetaDir = dir('Z:\Users\Alessandro La Chioma\Data\AL221025b\DataEphys\2022-11-08\Raw\AL221025b_2022-11-08_10-39-01\Record Node 102\experiment1\recording1\structure.oebin'); % path to your .meta or .oebin meta file
% saveLocation = 'Z:\Users\Alessandro La Chioma\Data\AL221025b\DataEphys\2022-11-08\QualityMetrics_experiment1_recording1'; % where you want to save the quality metrics 
% savePath = fullfile(saveLocation, 'qMetrics'); 
% % decompressDataLocal = '/media/julie/ExtraHD/decompressedData'; % where to save raw decompressed ephys data 
% decompressDataLocal = '';

%% load phys

[syncedData] = load_datafile('syncedData', fileparts(datafile_fullpath));
v2struct(syncedData); clear syncedData % unpack data_struct

% Convert bombcell numbering into Kilosort/phy numbering
% Bombcell numbering:  0, 3 = noise, 1 = good, 2 = mua
% Kilosort/phy numbering: 0 = noise, 1 = mua, 2 = good, 3 = unsorted.
ClusterGroup    = phys.ephys1.ClusterGroup;

fprintf('Kilosort Nr. units good: %d/%d (%3.1f%%), mua: %d/%d (%3.1f%%), noise: %d/%d (%3.1f%%)\n', sum(ClusterGroup==2),length(ClusterGroup),sum(ClusterGroup==2)/length(ClusterGroup)*100,  sum(ClusterGroup==1),length(ClusterGroup),sum(ClusterGroup==1)/length(ClusterGroup)*100, sum(ClusterGroup==0 | ClusterGroup==3),length(ClusterGroup),sum(ClusterGroup==0 | ClusterGroup==3)/length(ClusterGroup)*100)

%% load data 
[spikeTimes_samples, spikeTemplates, templateWaveforms, templateAmplitudes, pcFeatures, ...
    pcFeatureIdx, channelPositions] = bc_loadEphysData(ephysKilosortPath);

%% detect whether data is compressed, decompress locally if necessary
rawFile = bc_manageDataCompression(ephysRawDir, decompressDataLocal);

%% which quality metric parameters to extract and thresholds 
param = bc_qualityParamValues(ephysMetaDir, rawFile); 
% param = bc_qualityParamValuesForUnitMatch(ephysMetaDir, rawFile) % Run this if you want to use UnitMatch after

%% compute quality metrics 
rerun = 1;
qMetricsExist = ~isempty(dir(fullfile(savePath, 'qMetric*.mat'))) || ~isempty(dir(fullfile(savePath, 'templates._bc_qMetrics.parquet')));

if qMetricsExist == 0 || rerun
    [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
        templateWaveforms, templateAmplitudes,pcFeatures,pcFeatureIdx,channelPositions, savePath);
else
    [param, qMetric] = bc_loadSavedMetrics(savePath); 
    unitType = bc_getQualityUnitType(param, qMetric);
end

% Bombcell numbering:  0, 3 = noise, 1 = good, 2 = mua
fprintf('BombCell Nr. units good: %d/%d (%3.1f%%), mua: %d/%d (%3.1f%%), noise: %d/%d (%3.1f%%)\n', sum(unitType==1),length(unitType),sum(unitType==1)/length(unitType)*100,  sum(unitType==2),length(unitType),sum(unitType==2)/length(unitType)*100, sum(unitType==0 | unitType==3),length(unitType),sum(unitType==0 | unitType==3)/length(unitType)*100)

%% view units + quality metrics in GUI 
% load data for GUI
loadRawTraces = 1; % default: don't load in raw data (this makes the GUI significantly faster)
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

%% Convert bombcell numbering into Kilosort/phy numbering, save
% Bombcell numbering:  0, 3 = noise, 1 = good, 2 = mua
% Kilosort/phy numbering: 0 = noise, 1 = mua, 2 = good, 3 = unsorted.

if ~exist('phys','var') || ~exist('ClusterGroup','var')
   [syncedData] = load_datafile('syncedData', fileparts(datafile_fullpath));
    v2struct(syncedData); clear syncedData % unpack data_struct
    ClusterGroup    = phys.ephys1.ClusterGroup;
end
fprintf('\n')
fprintf('Kilosort Nr. units good: %d/%d (%3.1f%%), mua: %d/%d (%3.1f%%), noise: %d/%d (%3.1f%%)\n', sum(ClusterGroup==2),length(ClusterGroup),sum(ClusterGroup==2)/length(ClusterGroup)*100,  sum(ClusterGroup==1),length(ClusterGroup),sum(ClusterGroup==1)/length(ClusterGroup)*100, sum(ClusterGroup==0 | ClusterGroup==3),length(ClusterGroup),sum(ClusterGroup==0 | ClusterGroup==3)/length(ClusterGroup)*100)

ClusterGroup_bc = nan(size(ClusterGroup));
ClusterGroup_bc(ClusterGroup == 0) = 0;
ClusterGroup_bc(ClusterGroup == 3) = 0;
ClusterGroup_bc(ClusterGroup == 1) = 2;
ClusterGroup_bc(ClusterGroup == 2) = 1;

phys.ephys1.ClusterGroup_bc   = ClusterGroup_bc;
phys.ephys1.BombCell.qMetric  = qMetric;
phys.ephys1.BombCell.param    = param;
phys.ephys1.BombCell.unitType = unitType;

fprintf('BombCell Nr. units good: %d/%d (%3.1f%%), mua: %d/%d (%3.1f%%), noise: %d/%d (%3.1f%%)\n', sum(unitType==1),length(unitType),sum(unitType==1)/length(unitType)*100,  sum(unitType==2),length(unitType),sum(unitType==2)/length(unitType)*100, sum(unitType==0 | unitType==3),length(unitType),sum(unitType==0 | unitType==3)/length(unitType)*100)

save(datafile_fullpath, 'phys', '-append');
fprintf('BombCell quality metrics added to %s\n\n', datafile_fullpath)
