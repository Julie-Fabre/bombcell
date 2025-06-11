%% create smaller dataset

dataPath = '/media/julie/Expansion/Toy_dataset/site1/';
savePath = '//home/julie/Dropbox/MATLAB/onPaths/bombcell/demos/toy_data/';

unitsToKeep = [1:15]; % a good sample of good, bad and noisy units. 
unitsToKeep_zeroIdx = unitsToKeep - 1;

% load data
amplitudes = readNPY([dataPath, 'amplitudes.npy']);
%channel_map = readNPY([dataPath, 'channel_map.npy']); -> this stays the
%same
%channel_positions = readNPY([dataPath, 'channel_positions.npy']); -> this stays the
%same
pc_features = readNPY([dataPath, 'pc_features.npy']);
pc_feature_ind = readNPY([dataPath, 'pc_feature_ind.npy']);
spike_clusters = readNPY([dataPath, 'spike_clusters.npy']);
spike_templates = readNPY([dataPath, 'spike_templates.npy']);
spike_times = readNPY([dataPath, 'spike_times.npy']);
templates = readNPY([dataPath, 'templates.npy']);
%whitening_mat_inv = readNPY([dataPath, 'whitening_mat_inv.npy']);-> this stays the
%same

% subset data
unitsToKeep_spikes = ismember(spike_templates, unitsToKeep_zeroIdx);

amplitudes_subset = amplitudes(unitsToKeep_spikes);
spike_clusters_subset = spike_clusters(unitsToKeep_spikes);
spike_templates_subset = spike_templates(unitsToKeep_spikes);
spike_times_subset = spike_times(unitsToKeep_spikes);
pc_features_subset = pc_features(unitsToKeep_spikes,:,:);

pc_feature_ind_subset = pc_feature_ind(unitsToKeep,:);
templates_subset = templates(unitsToKeep,:,:);

% save 
writeNPY(amplitudes_subset, [savePath 'amplitudes.npy']);
writeNPY(spike_clusters_subset , [savePath 'spike_clusters.npy']);
writeNPY(spike_templates_subset, [savePath 'spike_templates.npy']);
writeNPY(spike_times_subset, [savePath 'spike_times.npy']);
writeNPY(pc_features_subset, [savePath 'pc_features.npy']);
writeNPY(pc_feature_ind_subset, [savePath 'pc_feature_ind.npy']);
writeNPY(templates_subset, [savePath 'templates.npy']);