
function [rawWaveformsFull, rawWaveformsPeakChan, signalToNoiseRatio] = extractRawWaveformsFast(param, spikeTimes_samples, ...
    spikeClusters, reExtract, savePath, verbose)
% JF, Get raw waveforms for all templates
% ------
% Inputs
% ------
% param with:
% rawFile: string containing the location of the raw .bin or .dat file location
% nChannels: number of recorded channels (including sync), (eg 385)
% nSpikesToExtract: number of spikes to extract per template
% detrendWaveforms: boolean, whether to detrend spikes or not
% spikeTimes_samples: nSpikes × 1 uint64 vector giving each spike time in samples (*not* seconds)
% spikeTemplates: nSpikes × 1 uint32 vector giving the identity of each
%   spike's matched template
% reextract: boolean, whether to reextract raw waveforms or not
% verbose: boolean, display progress bar or not
% savePath: where to save output data
% ------
% Outputs
% ------
% rawWaveformsFull: nUnits × nTimePoints × nChannels single matrix of
%   mean raw waveforms for each unit and channel
% rawWaveformsPeakChan: nUnits x 1 vector of each unit's channel with the maximum
%   amplitude
% signalToNoiseRatio: nUnits x 1 vector defining the absolute maximum
%       value of the mean raw waveform for that value divided by the variance
%       of the data before detected waveforms. implementation : Enny van Beest

%% Check if data needs to be extracted
if param.extractRaw
    nChannels = param.nChannels; % (385)
    spikeWidth = param.spikeWidth;
    % load raw waveforms and check if any empty
    nSpikeChannels = nChannels - param.nSyncChannels;
    [rawWaveformsFull, rawWaveformsPeakChan, baselineNoiseAmplitude, baselineNoiseAmplitudeIndex, emptyWaveforms] = ...
        bc.qm.helpers.loadRawWaveforms(savePath, spikeClusters, spikeWidth, nSpikeChannels, param.waveformBaselineNoiseWindow);
    if reExtract
        emptyWaveforms = unique(spikeClusters);
    end

    if any(emptyWaveforms)

        %% Extract raw waveforms

        %% Initialize parameters
        try
            nSpikesToExtract = param.nRawSpikesToExtract;
        catch
            nSpikesToExtract = param.nSpikesToExtract;
        end

        switch spikeWidth
            case 82
                % spikeWidth = 82: kilosort <4, baseline = 1:41
                halfWidth = spikeWidth / 2;
            case 61
                % spikeWidth = 61: kilosort 4, baseline = 1:20
                halfWidth = 20;
        end
        dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));
        nClust = numel(emptyWaveforms);
        rawFileInfo = dir(param.rawFile);
        BatchSize = 5000;
        if param.saveMultipleRaw && ~isfolder(fullfile(savePath, 'RawWaveforms'))
            mkdir(fullfile(savePath, 'RawWaveforms'))
        end

        fprintf('\n Extracting raw waveforms from %s ...', param.rawFile)

        % Get binary file name
        fid = fopen(param.rawFile, 'r');

        % loop over spike clusters
        for iCluster = 1:size(emptyWaveforms, 1)
            % Get cluster information
            rawWaveforms(iCluster).clInd = emptyWaveforms(iCluster);
            rawWaveforms(iCluster).spkInd = spikeTimes_samples(spikeClusters == emptyWaveforms(iCluster));

            % Determine # of spikes to extract
            if numel(rawWaveforms(iCluster).spkInd) >= nSpikesToExtract
                spksubi = round(linspace(1, numel(rawWaveforms(iCluster).spkInd), nSpikesToExtract))';
                rawWaveforms(iCluster).spkIndsub = rawWaveforms(iCluster).spkInd(spksubi);
            else
                rawWaveforms(iCluster).spkIndsub = rawWaveforms(iCluster).spkInd;
            end
            nSpkLocal = numel(rawWaveforms(iCluster).spkIndsub);

            % Initialize spike map for this cluster
            rawWaveforms(iCluster).spkMap = nan(nChannels-param.nSyncChannels, spikeWidth, nSpkLocal);

            % loop over spikes for this cluster
            for iSpike = 1:nSpkLocal
                thisSpikeIdx = rawWaveforms(iCluster).spkIndsub(iSpike);

                if ((thisSpikeIdx - spikeWidth) * nChannels) * dataTypeNBytes > spikeWidth && ...
                        (thisSpikeIdx + spikeWidth) * nChannels * dataTypeNBytes < rawFileInfo.bytes
                    bytei = ((thisSpikeIdx - halfWidth) * nChannels) * dataTypeNBytes;
                    fseek(fid, bytei, 'bof');
                    data0 = fread(fid, nChannels*spikeWidth, 'int16=>int16');
                    frewind(fid);
                    data = reshape(data0, nChannels, []);

                    if param.detrendWaveform
                        rawWaveforms(iCluster).spkMap(:, :, iSpike) = permute(detrend(double(permute(data(1:nChannels-param.nSyncChannels, :), [2, 1]))), [2, 1]);
                    else
                        rawWaveforms(iCluster).spkMap(:, :, iSpike) = data(1:nChannels-param.nSyncChannels, :);
                    end
                end
            end

            % Save multiple raw if needed
            if param.saveMultipleRaw
                tmpspkmap = permute(rawWaveforms(iCluster).spkMap, [2, 1, 3]);
                nBatch = ceil(nSpkLocal./BatchSize);
                for bid = 1:nBatch
                    spkId = (bid - 1) * BatchSize + (1:BatchSize);
                    spkId(spkId > nSpkLocal) = [];
                    tmpspkmap(:, :, spkId) = smoothdata(double(tmpspkmap(:, :, spkId)), 1, 'gaussian', 5);
                    tmpspkmap(:, :, spkId) = tmpspkmap(:, :, spkId) - mean(tmpspkmap(1:param.waveformBaselineNoiseWindow, :, spkId), 1);
                end
                tmpspkmap = arrayfun(@(X) nanmedian(tmpspkmap(:, :, (X - 1)*floor(size(tmpspkmap, 3)/2)+1:X*floor(size(tmpspkmap, 3)/2)), 3), 1:2, 'Uni', 0);
                tmpspkmap = cat(3, tmpspkmap{:});
                writeNPY(tmpspkmap, fullfile(savePath, 'RawWaveforms', ['Unit', num2str(emptyWaveforms(iCluster)-1), '_RawSpikes.npy']))
            end
            
            % Calculate mean and peak channel
            rawWaveforms(iCluster).spkMapMean = nanmean(rawWaveforms(iCluster).spkMap, 3);
            rawWaveformsFull(rawWaveforms(iCluster).clInd, :, :) = rawWaveforms(iCluster).spkMapMean - ...
                mean(rawWaveforms(iCluster).spkMapMean(:, 1:param.waveformBaselineNoiseWindow), 2);
            spkMapMean_sm = smoothdata(rawWaveforms(iCluster).spkMapMean, 2, 'gaussian', 5);

            [~, peakChan] = max(max(spkMapMean_sm, [], 2)-min(spkMapMean_sm, [], 2));
            rawWaveformsPeakChan(rawWaveforms(iCluster).clInd) = peakChan;

            % Get current cluster's baseline values
            current_baseline = nanmean(rawWaveforms(iCluster).spkMap(peakChan, 1:param.waveformBaselineNoiseWindow, :), 3);

            % Calculate required size for baseline arrays
            required_length = (rawWaveforms(iCluster).clInd) * param.waveformBaselineNoiseWindow;

            % Extend arrays if needed
            if required_length > length(baselineNoiseAmplitude)
                baselineNoiseAmplitude(end+1:required_length) = 0;
                baselineNoiseAmplitudeIndex(end+1:required_length) = 0;
            end

            % Store baseline for this unit
            idx_start = rawWaveforms(iCluster).clInd * param.waveformBaselineNoiseWindow + 1;
            idx_end = (rawWaveforms(iCluster).clInd + 1) * param.waveformBaselineNoiseWindow;

            baselineNoiseAmplitude(idx_start:idx_end) = current_baseline;
            baselineNoiseAmplitudeIndex(idx_start:idx_end) = rawWaveforms(iCluster).clInd;


            % Clear spike map to save memory
            rawWaveforms(iCluster).spkMap = [];

            if (mod(iCluster, 100) == 0 || iCluster == nClust) && verbose
                fprintf(['\n   Finished ', num2str(iCluster), ' / ', num2str(nClust), ' units.']);
            end
        end
        fclose(fid);

        if ~isfolder(savePath)
            mkdir(savePath)
        end
        writeNPY(rawWaveformsFull, fullfile(savePath, 'templates._bc_rawWaveforms_kilosort_format.npy'))
        writeNPY(rawWaveformsPeakChan, fullfile(savePath, 'templates._bc_rawWaveformPeakChannels_kilosort_format.npy'))
        writeNPY(baselineNoiseAmplitude, fullfile(savePath, 'templates._bc_baselineNoiseAmplitude_kilosort_format.npy'))
        writeNPY(baselineNoiseAmplitudeIndex, fullfile(savePath, 'templates._bc_baselineNoiseAmplitudeIndex_kilosort_format.npy'))


        % Convert and save compact format
        unique_clusters = unique(spikeClusters);
        n_unique_clusters = length(unique_clusters);

        % Initialize compact arrays
        rawWaveformsCompact = zeros(n_unique_clusters, size(rawWaveformsFull, 2), size(rawWaveformsFull, 3), 'single');
        rawWaveformsPeakChanCompact = zeros(n_unique_clusters, 1);

        % Fill compact arrays
        for i = 1:n_unique_clusters
            cluster_id = unique_clusters(i);
            rawWaveformsCompact(i, :, :) = rawWaveformsFull(cluster_id, :, :);
            rawWaveformsPeakChanCompact(i) = rawWaveformsPeakChan(cluster_id);
        end

        % Save compact format
        writeNPY(rawWaveformsCompact, fullfile(savePath, 'templates._bc_rawWaveforms.npy'));
        writeNPY(rawWaveformsPeakChanCompact, fullfile(savePath, 'templates._bc_rawWaveformPeakChannels.npy'));


    end

    %% estimate signal-to-noise ratio
    unique_clus = unique(spikeClusters);

    average_baseline_cat = readNPY(fullfile(savePath, 'templates._bc_baselineNoiseAmplitude_kilosort_format.npy'));
    average_baseline_idx_cat = readNPY(fullfile(savePath, 'templates._bc_baselineNoiseAmplitudeIndex_kilosort_format.npy'));

    % signal to noise ratio (Enny van Beest)
    signalToNoiseRatio = cell2mat(arrayfun(@(X) ...
        max(abs(squeeze(rawWaveformsFull(X, rawWaveformsPeakChan(X), :))))./ ...
        mad(average_baseline_cat(average_baseline_idx_cat == X)), ...
        unique_clus, 'Uni', false));

else
    unique_clus = unique(spikeClusters);
    signalToNoiseRatio = nan(numel(unique_clus), 1);
    rawWaveformsFull = [];
    rawWaveformsPeakChan = [];
end
