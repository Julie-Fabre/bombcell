function rawWaveforms = bc_extractRawWaveformsFast(rawFolder, nChannels, nSpikesToExtract, spikeTimes, spikeTemplates, rawFolder, verbose)
% JF, Get raw waveforms for all templates
% ------
% Inputs
% ------
% nChannels: number of recorded channels (including sync), (eg 385)
% nSpikesToExtract: number of spikes to extract per template
% spikeTimes: nSpikes × 1 uint64 vector giving each spike time in samples (*not* seconds)
% spikeTemplates: nSpikes × 1 uint32 vector giving the identity of each
%   spike's matched template
% rawFolder: string containing the location of the raw .dat or .bin file
% verbose: boolean, display progress bar or not
% ------
% Outputs
% ------
% rawWaveforms: struct with fields:
%   spkMapMean: nUnits × nTimePoints × nChannels single matrix of
%   mean raw waveforms for each unit and channel
%   peakChan: nUnits x 1 vector of each unit's channel with the maximum
%   amplitude

%% check if waveforms already extracted
rawWaveformFolder = dir(fullfile(param.rawFolder, 'rawWaveforms.mat'));

if ~isempty(rawWaveformFolder)
    load(fullfile(param.rawFolder, 'rawWaveforms.mat'));
else

    %% Intitialize
    % Get spike times and indices

    spikeWidth = 82;
    halfWidth = spikeWidth / 2;

    clustInds = unique(spikeTemplates);
    nClust = numel(clustInds);

    % Get binary file name
    spikeFile = dir(fullfile(rawFolder, '*.ap.bin'));
    if isempty(spikeFile)
        spikeFile = dir(fullfile(rawFolder, '*.dat')); %openEphys format
    end
    fname = spikeFile.name;
    fid = fopen(fullfile(rawFolder, fname), 'r');
    d = dir(fullfile(rawFolder, fname));

    dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));
    n_samples = d.bytes / (nChannels * dataTypeNBytes);

    %% Interate over spike clusters and find all the data associated with them
    rawWaveforms = struct;
    allSpikeTimes = spikeTimes;
    % array
    for iCluster = 1:nClust

        spikeIndices = allSpikeTimes(spikeTemplates == clustInds(iCluster));
        if numel(spikeIndices) >= nSpikesToExtract % QQ implement better spike selection selection method
            spksubi = round(linspace(1, numel(spikeIndices), nSpikesToExtract))';
            spikeIndices = spikeIndices(spksubi);
        end
        nSpikesEctractHere = numel(spikeIndices);

        spikeMap = nan(nChannels-1, spikeWidth, nSpikesEctractHere);
        for iSpike = 1:nSpikesEctractHere
            thisSpikeIdx = spikeIndices(iSpike);
            if thisSpikeIdx > halfWidth && (thisSpikeIdx + halfWidth) * dataTypeNBytes < d.bytes % check that it's not out of bounds
                byteIdx = ((thisSpikeIdx - halfWidth) * nChannels) * dataTypeNBytes;
                fseek(fid, byteIdx, 'bof'); % from beginning of file
                data = fread(fid, [nChannels, spikeWidth], 'int16=>int16'); % read individual waveform from binary file
                frewind(fid);
                %data = reshape(data0, nChannels, spikeWidth);
                if size(data, 2) == spikeWidth
                    spikeMap(:, :, iSpike) = data(1:nChannels-1, :, :); %remove sync channel
                end
            end
        end
        spikeMapMean = nanmean(spikeMap, 3);

        rawWaveforms(iCluster).spkMapMean = spikeMapMean - mean(spikeMapMean(:, 1:10), 2);

        spkMapMean_sm = smoothdata(spikeMapMean, 1, 'gaussian', 5);

        [~, rawWaveforms(iCluster).peakChan] = max(max(abs(spkMapMean_sm), [], 2), [], 1);

        clf;
        for iSpike = 1:10
            plot(spikeMap(1, :, iSpike));
            hold on;
        end
        clf;
        plot(rawWaveforms(iCluster).spkMapMean(1, :));
        hold on;
        if (mod(iCluster, 20) == 0 || iCluster == nClust) && verbose
            fprintf(['\n   Finished ', num2str(iCluster), ' of ', num2str(nClust), ' units.']);
            %figure; imagesc(spkMapMean_sm)
            %title(['Unit ID: ', num2str(i)]);
            %colorbar;
        end

    end

    fclose(fid);
    rawWaveformFolder = dir(fullfile(rawFolder, 'rawWaveforms.mat'));
    if isempty(rawWaveformFolder)
        save(fullfile(rawFolder, 'rawWaveforms.mat'), 'rawWaveforms', '-v7.3');
    end
end
end