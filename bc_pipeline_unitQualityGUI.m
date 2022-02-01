kilosort_path=ephysPath;
n_channels = 384;
ephys_datatype = 'int16';
dat = dir(ephysap_path);
ephys_ap_filename = [dat(1).folder filesep dat(1).name];
ap_dat_dir = dir(ephys_ap_filename);
pull_spikeT = -40:41;%number of points to pull for each spike
microVoltscaling = 0.19499999284744263;%in structure.oebin for openephys

% Load channels used by kilosort (it drops channels with no activity), 1-idx
used_channels_idx = readNPY([kilosort_path filesep 'channel_map.npy']) + 1;

% Load AP-band data (by memory map)
% get bytes per sample
dataTypeNBytes = numel(typecast(cast(0, ephys_datatype), 'uint8')); % determine number of bytes per sample

% samples per channel
n_samples = ap_dat_dir.bytes/(n_channels*dataTypeNBytes);  % Number of samples per channel

% memory map file
ap_data = memmapfile(ephys_ap_filename,'Format',{ephys_datatype,[n_channels,n_samples],'data'});

% put ephys data into structure 
ephysData = struct;
ephysData.spike_times = spikeTimes;
ephysData.spike_times_timeline = spikeTimes ./ 30000;
ephysData.spike_templates = spikeTemplates;
ephysData.templates = templateWaveforms;
ephysData.template_amplitudes = templateAmplitudes;
ephysData.channel_positions = readNPY([ephysPath filesep 'channel_positions.npy']);
ephysData.ephys_sample_rate = 30000;
ephysData.waveform_t = 1e3*((0:size(templateWaveforms, 2) - 1) / 30000);
ephysParams = struct;

probeLocation=[];
unitQualityGUI(ap_data.data.data,ephysData,qMetric, param, probeLocation);
% 
% allen_atlas_path = '/home/julie/Dropbox/Atlas/allenCCF';
% tv = readNPY([allen_atlas_path, filesep, 'template_volume_10um.npy']);
% av = readNPY([allen_atlas_path, filesep, 'annotation_volume_10um_by_index.npy']);
% st = loadStructureTreeJF([allen_atlas_path, filesep, 'structure_tree_safe_2017.csv']);  
% 
%  trajectory_area_boundaries = ...
%         [1;find(diff(probe_ccf(curr_probe).trajectory_areas) ~= 0);length(probe_ccf(curr_probe).trajectory_areas)];    
%     trajectory_area_centers = trajectory_area_boundaries(1:end-1) + diff(trajectory_area_boundaries)/2;
%     trajectory_area_labels = st.safe_name(probe_ccf(curr_probe).trajectory_areas(round(trajectory_area_centers)));
%      
