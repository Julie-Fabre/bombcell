function [nonEmptyUnits, duplicateSpikes_idx, spikeTimes_samples, spikeTemplates, templateAmplitudes, ...
    pcFeatures, rawWaveformsFull, rawWaveformsPeakChan, signalToNoiseRatio] = ...
    bc_removeDuplicateSpikes(spikeTimes_samples, spikeTemplates, templateAmplitudes, ...
    pcFeatures,  rawWaveformsFull, rawWaveformsPeakChan, signalToNoiseRatio,...
    removeDuplicateSpikes_flag, ...
    duplicateSpikeWindow_s, ephys_sample_rate, saveSpikes_withoutDuplicates_flag, savePath, recompute)
% JF, Remove any duplicate spikes
% Some spike sorters (including kilosort) can sometimes count spikes twice
% if for instance the residuals are re-fitted. see https://github.com/MouseLand/Kilosort/issues/29
% ------
% Inputs
% ------
%
% ------
% Outputs
% ------
%

if removeDuplicateSpikes_flag
    % Check if we need to extract duplicate spikes
    if recompute || isempty(dir([savePath, filesep, 'spikes._bc_duplicateSpikes.npy']))
        % Parameters
        duplicateSpikeWindow_samples = duplicateSpikeWindow_s * ephys_sample_rate;
        batch_size = 10000;
        overlap_size = 100;
        numSpikes_full = length(spikeTimes_samples);

        % initialize and re-allocate
        duplicateSpikes_idx = false(1, numSpikes_full);

        % check for duplicate spikes in batches
        for start_idx = 1:batch_size - overlap_size:numSpikes_full
            end_idx = min(start_idx+batch_size-1, numSpikes_full);
            batch_spikeTimes_samples = spikeTimes_samples(start_idx:end_idx);
            batch_spikeTemplates = spikeTemplates(start_idx:end_idx);
            batch_templateAmplitudes = templateAmplitudes(start_idx:end_idx);

            [~, ~, batch_removeIdx] = removeDuplicates(batch_spikeTimes_samples, batch_spikeTemplates, batch_templateAmplitudes, duplicateSpikeWindow_samples);

            duplicateSpikes_idx(start_idx:end_idx) = batch_removeIdx;

            if end_idx == numSpikes_full
                break;
            end
        end
        % save data if required
        if saveSpikes_withoutDuplicates_flag
            writeNPY(duplicateSpikes_idx, [savePath, filesep, 'spikes._bc_duplicateSpikes.npy'])
        end

    else
        duplicateSpikes_idx = readNPY([savePath, filesep, 'spikes._bc_duplicateSpikes.npy']);
    end

    % check if there are any empty units
    unique_templates = unique(spikeTemplates);
    nonEmptyUnits = unique(spikeTemplates(~duplicateSpikes_idx));
    emptyUnits_idx = ~ismember(unique_templates, nonEmptyUnits);

    % remove any empty units from ephys data 
    spikeTimes_samples = spikeTimes_samples(~duplicateSpikes_idx);
    spikeTemplates = spikeTemplates(~duplicateSpikes_idx);
    templateAmplitudes = templateAmplitudes(~duplicateSpikes_idx);
    if ~isempty(pcFeatures)
        pcFeatures = pcFeatures(~duplicateSpikes_idx, :, :);
    end
    rawWaveformsFull = rawWaveformsFull(~emptyUnits_idx, :, :);
    rawWaveformsPeakChan = rawWaveformsPeakChan(~emptyUnits_idx);
    if ~isempty(signalToNoiseRatio)
        signalToNoiseRatio = signalToNoiseRatio(~emptyUnits_idx);
    end


end


    function [spikeTimes_samples, spikeTemplates, removeIdx] = removeDuplicates(spikeTimes_samples, spikeTemplates, templateAmplitudes, duplicateSpikeWindow_samples)
        numSpikes = length(spikeTimes_samples);
        removeIdx = false(1, numSpikes);

        % Intra-unit duplicate removal
        for i = 1:numSpikes
            if removeIdx(i)
                continue;
            end

            for j = i + 1:numSpikes
                if removeIdx(j)
                    continue;
                end

                if spikeTemplates(i) == spikeTemplates(j)
                    if abs(spikeTimes_samples(i)-spikeTimes_samples(j)) <= duplicateSpikeWindow_samples
                        if templateAmplitudes(i) < templateAmplitudes(j)
                            spikeTimes_samples(i) = NaN;
                            removeIdx(i) = true;
                            break;
                        else
                            spikeTimes_samples(j) = NaN;
                            removeIdx(j) = true;
                        end
                    end
                end
            end
        end


        % Inter-unit duplicate removal
        unitSpikeCounts = accumarray(spikeTemplates, 1);
        for i = 1:length(spikeTimes_samples)
            if removeIdx(i)
                continue;
            end

            for j = i + 1:length(spikeTimes_samples)
                if removeIdx(j)
                    continue;
                end

                if spikeTemplates(i) ~= spikeTemplates(j)
                    if abs(spikeTimes_samples(i)-spikeTimes_samples(j)) <= duplicateSpikeWindow_samples
                        if unitSpikeCounts(spikeTemplates(i)) < unitSpikeCounts(spikeTemplates(j))
                            spikeTimes_samples(i) = NaN;
                            removeIdx(i) = true;
                            break;
                        else
                            spikeTimes_samples(j) = NaN;
                            removeIdx(j) = true;
                        end
                    end
                end
            end
        end
    end


end
