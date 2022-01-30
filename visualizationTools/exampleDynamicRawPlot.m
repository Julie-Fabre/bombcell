%% load stuffs 
animals = {'AP024','AP025','AP026','AP027','AP028','AP029', 'AL019'};
save_file = 'C:\Users\Julie\Documents\Analysis\';
load([save_file 'ephysData.mat']);
iCount=1;
kilosort_path='C:\Users\Julie\Documents\AP025\2017-10-01\kilosort2';
n_channels = 384;
ephys_datatype = 'int16';
dat = dir([kilosort_path filesep '*.dat']);
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

%% dynamic plot

dynamicRawPlot(ap_data.data.data, ephysData);