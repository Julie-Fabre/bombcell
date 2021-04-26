function ephysData = loadEphysDataJF(ephys_path, animal, day, experiment )
load_parts.ephys=true; 
AP_load_experimentJF;

spike_templates_0idx = readNPY([ephys_path filesep 'spike_templates.npy']);
spike_times = readNPY([ephys_path filesep  'spike_times.npy']);

pc_features = readNPY([ephys_path filesep  'pc_features.npy']);
pc_features_ind = readNPY([ephys_path filesep  'pc_feature_ind.npy']) + 1;

spike_times_full = spike_times;
spike_templates_full = spike_templates_0idx + 1;


%% save in a structure
ephysData = struct;

ephysData.channel_map = channel_map;
ephysData.channel_positions = channel_positions;
ephysData.cluster_groups = cluster_groups;
ephysData.ephys_sample_rate = ephys_sample_rate;
ephysData.spike_depths = spike_depths; %only good ones
ephysData.spike_templates = spike_templates; %only good ones
ephysData.spike_times_full = spike_times_full;
ephysData.spike_templates_full = spike_templates_full;
ephysData.spike_times_timeline = spike_times_timeline; %only good ones
ephysData.template_amplitudes = template_amplitudes; %only good ones
ephysData.template_depths = template_depths;
ephysData.templates = templates;
ephysData.template_waveforms = waveforms;
ephysData.winv = winv;
ephysData.good_templates = good_templates;
ephysData.pc_features = pc_features(good_spike_idx,:,:);
ephysData.pc_features_full = pc_features;
ephysData.pc_features_ind_full = pc_features_ind;
ephysData.pc_features_ind = pc_features_ind(good_templates_idx+1,:);
%ephysData.str_templates = str_templates;
%ephysData.spike_rate = spike_rate;
ephysData.new_spike_idx = new_spike_idx;


end