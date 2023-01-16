%get memmap
bc_getRawMemMap;

% put ephys data into structure
ephysData = struct;
ephysData.spike_times_samples = spikeTimes_samples;
ephysData.ephys_sample_rate = 30000;
ephysData.spike_times = spikeTimes_samples ./ ephysData.ephys_sample_rate;
ephysData.spike_templates = spikeTemplates;
ephysData.templates = templateWaveforms;
ephysData.template_amplitudes = templateAmplitudes;
ephysData.channel_positions = channelPositions;

ephysData.waveform_t = 1e3 * ((0:size(templateWaveforms, 2) - 1) / 30000);
ephysParams = struct;
plotRaw = 1;
probeLocation = [];


load(fullfile(savePath, 'qMetric.mat'))
param = parquetread([fullfile(savePath, '_bc_parameters._bc_qMetrics.parquet')]);
rawWaveforms.average = readNPY([fullfile(savePath, 'templates._bc_rawWaveforms.npy')]);
rawWaveforms.peakChan = readNPY([fullfile(savePath, 'templates._bc_rawWaveformPeakChannels.npy')]);
