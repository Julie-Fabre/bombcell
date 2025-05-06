function [rawWaveformsFull, rawWaveformsPeakChan, baselineNoiseAmplitude, baselineNoiseAmplitudeIndex, emptyWaveforms] = ...
    loadRawWaveforms(savePath, spikeClusters, spikeWidth, nSpikeChannels, waveformBaselineNoiseWindow)

rawWaveformFolder = dir(fullfile(savePath, 'templates._bc_rawWaveforms_kilosort_format.npy'));
unique_clusters = unique(spikeClusters);

if ~isempty(rawWaveformFolder)
    rawWaveformsFull = readNPY(fullfile(savePath, 'templates._bc_rawWaveforms_kilosort_format.npy'));
    rawWaveformsPeakChan = readNPY(fullfile(savePath, 'templates._bc_rawWaveformPeakChannels_kilosort_format.npy'));
    baselineNoiseAmplitude = readNPY(fullfile(savePath, 'templates._bc_baselineNoiseAmplitude_kilosort_format.npy'));
    baselineNoiseAmplitudeIndex = readNPY(fullfile(savePath, 'templates._bc_baselineNoiseAmplitudeIndex_kilosort_format.npy'));

    % Find empty rows (where all elements are zero)
    row_sums = squeeze(sum(sum(abs(rawWaveformsFull), 2), 3));
    empty_row_indices = find(isnan(row_sums));
    
    % Check which empty rows should actually contain data
    emptyWaveforms = empty_row_indices(ismember(empty_row_indices, unique_clusters));

else
    rawWaveformsFull = nan(max(unique_clusters), nSpikeChannels, spikeWidth);
    rawWaveformsPeakChan = nan(max(unique_clusters), 1);
    emptyWaveforms = unique_clusters;
    baselineNoiseAmplitude = nan(numel(unique_clusters) * waveformBaselineNoiseWindow, 1);
    baselineNoiseAmplitudeIndex = nan(numel(unique_clusters) * waveformBaselineNoiseWindow, 1);
      
end
end
