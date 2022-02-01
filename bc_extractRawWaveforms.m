function [rawWaveforms, ap_data] = bc_extractRawWaveforms(rawFolder, nChannels, nSpikesToExtract, spikeTimes, spikeTemplates, used_channels_idx, verbose)
% JF, Get raw waveforms for all templates
% ------
% Inputs
% ------
% ap_data.data.data: number of recorded channels (including sync), (eg 385)
% nSpikesToExtract: number of spikes to extract per template
% spikeTimes: nSpikes × 1 uint64 vector giving each spike time in samples (*not* seconds)
% spikeTemplates: nSpikes × 1 uint32 vector giving the identity of each
%   spike's matched template
% rawFolder: string containing the location of the raw .dat or .bin file
% verbose: boolean, display progress bar or not
% ------
% Outputs
% ------1
% rawWaveforms: struct with fields:
%   spkMapMean: nUnits × nTimePoints × ap_data.data.data single matrix of
%   mean raw waveforms for each unit and channel
%   peakChan: nUnits x 1 vector of each unit's channel with the maximum
%   amplitude

%% check if waveforms already extracted
% Get binary file name
spikeFile = dir(fullfile(rawFolder, '*.ap.bin'));
if isempty(spikeFile)
    spikeFile = dir(fullfile(rawFolder, '/*.dat')); %openEphys format
end

rawWaveformFolder = dir(fullfile(spikeFile.folder, 'rawWaveforms.mat'));

    %% Intitialize
    % Get spike times and indices

    pull_spikeT = -41:40;

    clustInds = unique(spikeTemplates);
    nClust = numel(clustInds);


    fname = spikeFile.name;

    dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));
    try
        n_samples = spikeFile.bytes/ (nChannels * dataTypeNBytes);
        ap_data = memmapfile(fullfile(spikeFile.folder, fname),'Format',{'int16',[nChannels,n_samples],'data'});
    catch
        nChannels = nChannels -1;
        n_samples = spikeFile.bytes/ (nChannels * dataTypeNBytes);
        ap_data = memmapfile(fullfile(spikeFile.folder, fname),'Format',{'int16',[nChannels,n_samples],'data'});n_samples = spikeFile.bytes/ (nChannels * dataTypeNBytes);
    end
    
if ~isempty(rawWaveformFolder)
   load(fullfile(spikeFile.folder, 'rawWaveforms.mat'));
else


    

    %% Interate over spike clusters and find all the data associated with them
    rawWaveforms = struct;
    allSpikeTimes = spikeTimes;
    % array
    for iCluster = 1:nClust

        curr_template = clustInds(iCluster); %gg(iUnit);
        %ns = find(new_spike_idx == curr_template);

        curr_spikes_idx = find(spikeTemplates == curr_template);
        if ~isempty(curr_spikes_idx)
            curr_pull_spikes = unique(round(linspace(1, length(curr_spikes_idx), nSpikesToExtract)));
            if curr_pull_spikes(1) == 0
                curr_pull_spikes(1) = [];
            end
            curr_spikeT = spikeTimes(curr_spikes_idx(curr_pull_spikes));
            curr_spikeT_pull = double(curr_spikeT) + pull_spikeT;

            out_of_bounds_spikes = any(curr_spikeT_pull < 1, 2) | ...
                any(curr_spikeT_pull > size(ap_data.data.data, 2), 2);
            curr_spikeT_pull(out_of_bounds_spikes, :) = [];

            curr_spike_waveforms = reshape(ap_data.data.data(:, reshape(curr_spikeT_pull', [], 1)), nChannels, length(pull_spikeT), []);
            if ~isempty(curr_spike_waveforms)
                curr_spike_waveforms_car = curr_spike_waveforms - nanmedian(curr_spike_waveforms, 1);
                curr_spike_waveforms_car_sub = curr_spike_waveforms_car - curr_spike_waveforms_car(:, 1, :);

                waveforms_mean(curr_template, :, :) = ...
                    permute(nanmean(curr_spike_waveforms_car_sub(used_channels_idx, :, :), 3), [3, 2, 1]); %* raw.microVoltscaling;
                rawWaveforms(iCluster).spkMapMean = waveforms_mean(curr_template, :, :);

                thisChannelRaw = find(squeeze(max(waveforms_mean(curr_template, :, :))) == ...
                    max(squeeze(max(waveforms_mean(curr_template, :, :)))));
                rawWaveforms(iCluster).peakChan = thisChannelRaw;

                if numel(thisChannelRaw) > 1
                    thisOne = find(max(abs(waveforms_mean(curr_template, :, thisChannelRaw))) == max(max(abs(waveforms_mean(curr_template, :, thisChannelRaw)))));
                    if numel(thisOne) > 1
                        thisOne = thisOne(1);
                    end
                    rawWaveforms(iCluster).peakChan = thisChannelRaw(thisOne);


                end
%                 figure()
%                 plot(rawWaveforms(iCluster).spkMapMean(rawWaveforms(iCluster).peakChan, :))
            end
        end

    end
    

    if (mod(iCluster, 20) == 0 || iCluster == nClust) && verbose
        fprintf(['\n   Finished ', num2str(iCluster), ' of ', num2str(nClust), ' units.']);
    end


%    fclose(fid);
    rawWaveformFolder = dir(fullfile(spikeFile.folder, 'rawWaveforms.mat'));
    %if isempty(rawWaveformFolder)
        save(fullfile(spikeFile.folder, 'rawWaveforms.mat'), 'rawWaveforms', '-v7.3');
    %end
end
end