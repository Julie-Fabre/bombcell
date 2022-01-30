function [spikeTimes, spikeTemplates, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, usedChannels] = bc_loadEphysData(ephys_path)

spike_templates_0idx = readNPY([ephys_path filesep 'spike_templates.npy']);
spikeTemplates = spike_templates_0idx + 1;
spikeTimes = double(readNPY([ephys_path filesep  'spike_times.npy']));

templateAmplitudes = readNPY([ephys_path filesep 'amplitudes.npy']);

templateWaveforms = readNPY([ephys_path filesep 'templates.npy']);

pcFeatures = readNPY([ephys_path filesep  'pc_features.npy']);
pcFeatureIdx = readNPY([ephys_path filesep  'pc_feature_ind.npy']) + 1;

usedChannels = readNPY([ephys_path filesep  'channel_map.npy']) + 1;

end