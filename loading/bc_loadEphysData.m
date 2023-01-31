function [spikeTimes_samples, spikeTemplates, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions, channelMap] = bc_loadEphysData(ephys_path, datasetidx)

if nargin<2
    datasetidx=1;
end

spike_templates_0idx = readNPY([ephys_path filesep 'spike_templates.npy']);
spikeTemplates = spike_templates_0idx + 1;
if exist(fullfile(ephys_path,'spike_times_corrected.npy')) % When running pyKS stitched you need the 'aligned / corrected' spike times
    spikeTimes_samples = double(readNPY([ephys_path filesep  'spike_times_corrected.npy']));
    spikeTimes_datasets = double(readNPY([ephys_path filesep  'spike_datasets.npy']))+1; %  which dataset? (zero-indexed so +1)
else
    spikeTimes_samples = double(readNPY([ephys_path filesep  'spike_times.npy']));
    spikeTimes_datasets = ones(size(spikeTimes_samples));
end
templateAmplitudes = readNPY([ephys_path filesep 'amplitudes.npy']);

templateWaveforms = readNPY([ephys_path filesep 'templates.npy']);
try %not computed in early kilosort3 version
    pcFeatures = readNPY([ephys_path filesep  'pc_features.npy']);
    pcFeatureIdx = readNPY([ephys_path filesep  'pc_feature_ind.npy']) + 1;
catch
    pcFeatures = NaN;
    pcFeatureIdx = NaN;
end 
channelPositions = readNPY([ephys_path filesep  'channel_positions.npy']) ; 
channelMap = readNPY([ephys_path filesep  'channel_map.npy']) + 1;


%% Only use data set of interest
spikeTimes_samples=spikeTimes_samples(spikeTimes_datasets==datasetidx);
spikeTemplates=spikeTemplates(spikeTimes_datasets==datasetidx);
templateAmplitudes=templateAmplitudes(spikeTimes_datasets==datasetidx);
pcFeatures=pcFeatures(spikeTimes_datasets==datasetidx,:,:);

end