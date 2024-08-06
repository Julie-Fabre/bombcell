function [nonEmptyUnits, duplicateSpikes_idx, spikeTimes_samples, spikeTemplates, templateAmplitudes, ...
    pcFeatures, rawWaveformsFull, rawWaveformsPeakChan, signalToNoiseRatio, maxChannels] = ...
    removeDuplicateSpikes(spikeTimes_samples, spikeTemplates, templateAmplitudes, ...
    pcFeatures, rawWaveformsFull, rawWaveformsPeakChan, signalToNoiseRatio, ...
    maxChannels, removeDuplicateSpikes_flag, ...
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

            [~, ~, batch_removeIdx] = removeDuplicates(batch_spikeTimes_samples, ...
                batch_spikeTemplates, batch_templateAmplitudes, duplicateSpikeWindow_samples, ...
                maxChannels);

            duplicateSpikes_idx(start_idx:end_idx) = batch_removeIdx;

            if end_idx == numSpikes_full
                break;
            end
        end
        % save data if required
        if saveSpikes_withoutDuplicates_flag
            try
                writeNPY(duplicateSpikes_idx, [savePath, filesep, 'spikes._bc_duplicateSpikes.npy'])
            catch
                warning('unable to save duplicate spikes')
            end
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

    if ~isempty(pcFeatures) && any(~isnan(pcFeatures), 'all')

        pcFeatures = pcFeatures(~duplicateSpikes_idx, :, :);
    end

    if ~isempty(rawWaveformsFull)
        rawWaveformsFull = rawWaveformsFull(~emptyUnits_idx, :, :);
        rawWaveformsPeakChan = rawWaveformsPeakChan(~emptyUnits_idx);
    end

    if ~isempty(signalToNoiseRatio)
        signalToNoiseRatio = signalToNoiseRatio(~emptyUnits_idx);
    end

    fprintf('\n Removed %.0f spike duplicates out of %.0f total spikes. \n', sum(duplicateSpikes_idx), length(duplicateSpikes_idx))

else
    nonEmptyUnits = unique(spikeTemplates);
    duplicateSpikes_idx = zeros(size(spikeTimes_samples, 1), 1);

end


    function [spikeTimes_samples, spikeTemplates, removeIdx] = removeDuplicates(spikeTimes_samples, ...
            spikeTemplates, templateAmplitudes, duplicateSpikeWindow_samples, maxChannels)
        numSpikes = length(spikeTimes_samples);
        removeIdx = false(1, numSpikes);

        % Intra-unit duplicate removal
        for iSpike1 = 1:numSpikes
            if removeIdx(iSpike1)
                continue;
            end

            for iSpike2 = iSpike1 + 1:numSpikes
                if removeIdx(iSpike2)
                    continue;
                end
                if maxChannels(spikeTemplates(iSpike2)) ~= maxChannels(spikeTemplates(iSpike1)) % spikes are not on same channel
                    continue;
                end

                if spikeTemplates(iSpike1) == spikeTemplates(iSpike2)
                    if abs(spikeTimes_samples(iSpike1)-spikeTimes_samples(iSpike2)) <= duplicateSpikeWindow_samples
                        if templateAmplitudes(iSpike1) < templateAmplitudes(iSpike2)
                            spikeTimes_samples(iSpike1) = NaN;
                            removeIdx(iSpike1) = true;
                            break;
                        else
                            spikeTimes_samples(iSpike2) = NaN;
                            removeIdx(iSpike2) = true;
                        end
                    end
                end
            end
        end


        % Inter-unit duplicate removal
        unitSpikeCounts = accumarray(spikeTemplates, 1);
        for iSpike1 = 1:length(spikeTimes_samples)
            if removeIdx(iSpike1)
                continue;
            end

            for iSpike2 = iSpike1 + 1:length(spikeTimes_samples)
                if removeIdx(iSpike2)
                    continue;
                end
                if maxChannels(spikeTemplates(iSpike2)) ~= maxChannels(spikeTemplates(iSpike1)) % spikes are not on same channel
                    continue;
                end

                if spikeTemplates(iSpike1) ~= spikeTemplates(iSpike2)
                    if abs(spikeTimes_samples(iSpike1)-spikeTimes_samples(iSpike2)) <= duplicateSpikeWindow_samples
                        if unitSpikeCounts(spikeTemplates(iSpike1)) < unitSpikeCounts(spikeTemplates(iSpike2))
                            spikeTimes_samples(iSpike1) = NaN;
                            removeIdx(iSpike1) = true;
                            break;
                        else
                            spikeTimes_samples(iSpike2) = NaN;
                            removeIdx(iSpike2) = true;
                        end
                    end
                end
            end
        end
    end


end
