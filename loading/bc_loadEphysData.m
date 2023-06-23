function [spikeTimes_samples, spikeTemplates, templateWaveforms, templateAmplitudes, ...
    pcFeatures, pcFeatureIdx, channelPositions, goodChannels] = bc_loadEphysData(ephys_path, datasetidx)
% JF, Load ephys data (1-indexed)
% ------
% Inputs
% ------
% ephys_path: character array defining the path to your kilosorted output files 
% datasetidx: 1 x 1 double vector, only use if you have chronically
% recorded stitched datasets. 
% ------
% Outputs
% ------
% spikeTimes_samples: nSpikes × 1 uint64 vector giving each spike time in samples (*not* seconds)
% spikeTemplates: nSpikes × 1 uint32 vector giving the identity of each
%   spike's matched template
% templateWaveforms: nTemplates × nTimePoints × nChannels single matrix of
%   template waveforms for each template and channel
% templateAmplitudes: nSpikes × 1 double vector of the amplitude scaling factor
%   that was applied to the template when extracting that spike
% pcFeatures: nSpikes × nFeaturesPerChannel × nPCFeatures  single
%   matrix giving the PC values for each spike
% pcFeatureIdx: nTemplates × nPCFeatures uint32  matrix specifying which
%   channels contribute to each entry in dim 3 of the pc_features matrix
% channelPositions: nChannels x 2 double matrix, each row gives the x and y 
%   coordinates of each channel
% goodChannels: nChannels x 1 uint32 vector defining the channels used by
%   kilosort (some are dropped during the spike sorting process)
%



spike_templates_0idx = readNPY([ephys_path filesep 'spike_templates.npy']);
spikeTemplates = spike_templates_0idx + 1;
if exist(fullfile(ephys_path,'spike_times_corrected.npy')) % When running pyKS stitched you need the 'aligned / corrected' spike times
    spikeTimes_samples = double(readNPY([ephys_path filesep  'spike_times_corrected.npy']));
    spikeTimes_datasets = double(readNPY([ephys_path filesep  'spike_datasets.npy'])) + 1; %  which dataset? (zero-indexed so +1)
else
    spikeTimes_samples = double(readNPY([ephys_path filesep  'spike_times.npy']));
    spikeTimes_datasets = ones(size(spikeTimes_samples));
end
templateAmplitudes = readNPY([ephys_path filesep 'amplitudes.npy']);

% Load and unwhiten templates
templateWaveforms_whitened = readNPY([ephys_path filesep 'templates.npy']);
winv = readNPY([ephys_path filesep 'whitening_mat_inv.npy']);
templateWaveforms = zeros(size(templateWaveforms_whitened));
for t = 1:size(templateWaveforms,1)
    templateWaveforms(t,:,:) = squeeze(templateWaveforms_whitened(t,:,:))*winv;
end

try %not computed in early kilosort3 version
    pcFeatures = readNPY([ephys_path filesep  'pc_features.npy']);
    pcFeatureIdx = readNPY([ephys_path filesep  'pc_feature_ind.npy']) + 1;
catch
    pcFeatures = NaN;
    pcFeatureIdx = NaN;
end 
channelPositions = readNPY([ephys_path filesep  'channel_positions.npy']) ; 
goodChannels = readNPY([ephys_path filesep  'channel_map.npy']) + 1;


%% Only use data set of interest - for unit match
if nargin>2  %- for unit match
   
    spikeTimes_samples = spikeTimes_samples(spikeTimes_datasets == datasetidx);
    spikeTemplates = spikeTemplates(spikeTimes_datasets == datasetidx);
    templateAmplitudes = templateAmplitudes(spikeTimes_datasets == datasetidx);
    pcFeatures=pcFeatures(spikeTimes_datasets == datasetidx,:,:);
end

end