function checkAndConvertRawWaveforms(savePath, spikeTemplates, spikeClusters)

rawWaveformFolderNew = dir(fullfile(savePath, 'templates._bc_rawWaveforms_kilosort_format.npy'));
rawWaveformFolderOld = dir(fullfile(savePath, 'templates._bc_rawWaveforms.npy'));

% Check if need to convert to new format. old format was always created
% with spike_templates (pre- any splitting or merging, so use this to
% get indices)
if ~isempty(rawWaveformFolderOld) && isempty(rawWaveformFolderNew)
    try
    % Load old format data
    rawWaveformsFullOld = readNPY(fullfile(savePath, 'templates._bc_rawWaveforms.npy'));
    rawWaveformsPeakChanOld = readNPY(fullfile(savePath, 'templates._bc_rawWaveformPeakChannels.npy'));
    
    baselineNoiseAmplitudeOld = readNPY(fullfile(savePath, 'templates._bc_baselineNoiseAmplitude.npy'));
    baselineNoiseAmplitudeIndexOld = readNPY(fullfile(savePath, 'templates._bc_baselineNoiseAmplitudeIndex.npy'));

    % Get dimensions
    [~, timepoints, channels] = size(rawWaveformsFullOld);
    max_cluster_id = max(spikeClusters);
    
    % Get unique clusters and their indices in the old format
    unique_clusters_old = unique(spikeTemplates);
    original_indices = 1:max(unique_clusters_old);
    unique_clusters_new = unique(spikeClusters);

    % Initialize arrays with new size based on maximum cluster ID
    rawWaveformsFull = nan(max_cluster_id, timepoints, channels);
    rawWaveformsPeakChan = nan(max_cluster_id, 1);
    baselineNoiseAmplitudeIndex = unique_clusters_old(baselineNoiseAmplitudeIndexOld);


    % Fill arrays using vectorized operations
    valid_idx = find(unique_clusters_old <= max(original_indices));
    for i = 1:length(valid_idx)
        cluster_id = unique_clusters_old(valid_idx(i));
        old_idx = valid_idx(i);
        rawWaveformsFull(cluster_id, :, :) = rawWaveformsFullOld(old_idx, :, :);
        rawWaveformsPeakChan(cluster_id) = rawWaveformsPeakChanOld(old_idx);
    end


    % Find empty rows (where all elements are zero)
    row_sums = squeeze(sum(sum(abs(rawWaveformsFull), 2), 3));
    empty_row_indices = find(row_sums == 0);
    % Check which empty rows should actually contain data
    emptyWaveforms = empty_row_indices(ismember(empty_row_indices, unique_clusters_new));

    % Save in new format
    writeNPY(rawWaveformsFull, fullfile(savePath, 'templates._bc_rawWaveforms_kilosort_format.npy'));
    writeNPY(rawWaveformsPeakChan, fullfile(savePath, 'templates._bc_rawWaveformPeakChannels_kilosort_format.npy'));
    writeNPY(baselineNoiseAmplitudeOld, fullfile(savePath, 'templates._bc_baselineNoiseAmplitude_kilosort_format.npy'));
    writeNPY(baselineNoiseAmplitudeIndex, fullfile(savePath, 'templates._bc_baselineNoiseAmplitudeIndex_kilosort_format.npy'));
    catch % error: we should delee the files so wavefroms and snr can be re-extracted properly
        delete(fullfile(savePath, 'templates._bc_rawWaveforms.npy'))
        delete(fullfile(savePath, 'templates._bc_rawWaveformPeakChannels.npy'))
        delete(fullfile(savePath, 'templates._bc_baselineNoiseAmplitude.npy'))
        delete(fullfile(savePath, 'templates._bc_baselineNoiseAmplitudeIndex.npy'))

    end
end