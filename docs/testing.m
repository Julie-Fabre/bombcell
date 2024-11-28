% Set paths
test_dataset_location = '/home/julie/Downloads/kilosort4-20241127T195144Z-001/kilosort4/';

% These paths below are the paths you will need to input to load data and save the computed quality metrics / ephys properties. Here we are leaving ephysRawFile as "NaN" to not load raw data (it is too cumbersome to store these large files on github). All metrics relating to raw data (amplitude, signal to noise ratio) will not be computed. 
ephysKilosortPath = test_dataset_location;
ephysRawFile = "NaN"; % path to your raw .bin or .dat data
ephysMetaDir = dir([test_dataset_location '*ap*.*meta']); % path to your .meta or .oebin meta file
savePath = [test_dataset_location 'qMetrics']; % where you want to save the quality metrics 
% Two paramaters to pay attention to: the kilosort version (change to kilosortVersion = 4 if you are using kilosort4) and the gain_to_uV scaling factor (this is the scaling factor to apply to your data to get it in microVolts).
kilosortVersion = 4; % if using kilosort4, you need to change this value. Otherwise it does not matter. 
gain_to_uV = NaN; % use this if you are not using spikeGLX or openEphys to record your data. this value, 
% when mulitplied by your raw data should convert it to  microvolts. 
% Load data
% This function loads are your ephys data. Use this function rather than any custom one as it handles zero-indexed values in a particular way. 
[spikeTimes_samples, spikeClusters, templateWaveforms, templateAmplitudes, pcFeatures, ...
    pcFeatureIdx, channelPositions] = bc.load.loadEphysData(ephysKilosortPath, savePath);
% Run quality metrics
% Set your paramaters. 
% These define both how you will run quality metrics and how thresholds will be applied to quality metrics to classify units into good/MUA/noise/non-axonal. This function loads default, permissive values. It's highly recommended for you to iteratively tweak these values to find values that suit your particular use case! 
param = bc.qm.qualityParamValues(ephysMetaDir, ephysRawFile, ephysKilosortPath, gain_to_uV, kilosortVersion);
% Pay particular attention to param.nChannels
% param.nChannels must correspond to the total number of channels in your raw data, including any sync channels. For Neuropixels probes, this value should typically be either 384 or 385 channels. param.nSyncChannels must correspond to the number of sync channels you recorded. This value is typically 1 or 0.
param.nChannels = 385;
param.nSyncChannels = 1;

% if using SpikeGLX, you can use this function: 
if ~isempty(ephysMetaDir)
    if endsWith(ephysMetaDir.name, '.ap.meta') %spikeGLX file-naming convention
        meta = bc.dependencies.SGLX_readMeta.ReadMeta(ephysMetaDir.name, ephysMetaDir.folder);
        [AP, ~, SY] = bc.dependencies.SGLX_readMeta.ChannelCountsIM(meta);
        param.nChannels = AP + SY;
        param.nSyncChannels = SY;
    end
end
% Run all your quality metrics! 
% This function runs all quality metrics, saves the metrics in your savePath folder and outputs some global summary plots that can give you a good idea of how things went. 
[qMetric, unitType] = bc.qm.runAllQualityMetrics(param, spikeTimes_samples, spikeClusters, ...
        templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions, savePath);
% Inspect
% After running quality metrics, espacially the first few times, it's a good idea to inspect your data and the quality metrics using the built-in GUI. Use your keyboard to navigate the GUI: 
% left/right arrow: toggle between units 
% g : go to next good unit 
% m : go to next multi-unit 
% n : go to next noise unit
% up/down arrow: toggle between time chunks in the raw data
% u: brings up a input dialog to enter the unit you want to go to

loadRawTraces = 0; % default: don't load in raw data (this makes the GUI significantly faster)
bc.load.loadMetricsForGUI;

unitQualityGuiHandle = bc.viz.unitQualityGUI_synced(memMapData, ephysData, qMetric, forGUI, rawWaveforms, ...
    param, probeLocation, unitType, loadRawTraces);

      
% Run ephys properties
% Optionally get ephys properties for your cell. Bombcell will also attempt to classify your data if it is (a) from the cortex or striatum and (b) you specify this in the "region" variable.
rerunEP = 0;
region = ''; % options include 'Striatum' and 'Cortex'
[ephysProperties, unitClassif] = bc.ep.runAllEphysProperties(ephysKilosortPath, savePath, rerunEP, region);
