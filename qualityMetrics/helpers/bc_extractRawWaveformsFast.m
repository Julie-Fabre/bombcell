
function [rawWaveformsFull, rawWaveformsPeakChan, signalToNoiseRatio] = bc_extractRawWaveformsFast(param, spikeTimes_samples, ...
    spikeTemplates, reExtract, savePath, verbose)
% JF, Get raw waveforms for all templates
% ------
% Inputs
% ------
% param with:
% rawFile: raw .bin or .dat file location
% nChannels: number of recorded channels (including sync), (eg 385)
% nSpikesToExtract: number of spikes to extract per template
% rawFile: string containing the location of the raw .dat or .bin file
%
% spikeTimes_samples: nSpikes × 1 uint64 vector giving each spike time in samples (*not* seconds)
% spikeTemplates: nSpikes × 1 uint32 vector giving the identity of each
%   spike's matched template
% verbose: boolean, display progress bar or not
% savePath: where to save output data
% ------
% Outputs
% ------
% rawWaveformsFull: nUnits × nTimePoints × nChannels single matrix of
%   mean raw waveforms for each unit and channel
% rawWaveformsPeakChan: nUnits x 1 vector of each unit's channel with the maximum
%   amplitude

%% Check if data already extracted
rawWaveformFolder = dir(fullfile(savePath, 'templates._bc_rawWaveforms.npy'));

if ~isempty(rawWaveformFolder) && reExtract == 0

    rawWaveformsFull = readNPY(fullfile(savePath, 'templates._bc_rawWaveforms.npy'));
    rawWaveformsPeakChan = readNPY(fullfile(savePath, 'templates._bc_rawWaveformPeakChannels.npy'));

else

    %% Initialize stuff
    % Get spike times and indices
    nChannels = param.nChannels; % (385)
    nSpikesToExtract = param.nRawSpikesToExtract;
    spikeWidth = param.spikeWidth;
    halfWidth = spikeWidth / 2;
    dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));
    clustInds = unique(spikeTemplates);
    nClust = numel(clustInds);
    rawFileInfo = dir(param.rawFile);

    if param.saveMultipleRaw && ~isfolder(fullfile(rawFileInfo.folder,['RawWaveforms_' rawFileInfo.name]))
        mkdir(fullfile(rawFileInfo.folder,['RawWaveforms_' rawFileInfo.name]))
    end

    fprintf('Extracting raw waveforms from %s ... \n', param.rawFile)
    % Get binary file name
    fid = fopen(param.rawFile, 'r');

    %% Interate over spike clusters and find all the data associated with them
    rawWaveforms = struct;
    rawWaveformsFull = nan(nClust, nChannels-param.nSyncChannels, spikeWidth);
    rawWaveformsPeakChan = nan(nClust, 1);

    for iCluster = 1:nClust
        rawWaveforms(iCluster).clInd = clustInds(iCluster);
        rawWaveforms(iCluster).spkInd = spikeTimes_samples(spikeTemplates == clustInds(iCluster));
        if numel(rawWaveforms(iCluster).spkInd) >= nSpikesToExtract
            spksubi = round(linspace(1, numel(rawWaveforms(iCluster).spkInd), nSpikesToExtract))';
            rawWaveforms(iCluster).spkIndsub = rawWaveforms(iCluster).spkInd(spksubi);
        else
            rawWaveforms(iCluster).spkIndsub = rawWaveforms(iCluster).spkInd;
        end
        nSpkLocal = numel(rawWaveforms(iCluster).spkIndsub);

        rawWaveforms(iCluster).spkMap = nan(384, spikeWidth, nSpkLocal);
        for iSpike = 1:nSpkLocal
            thisSpikeIdx = rawWaveforms(iCluster).spkIndsub(iSpike);
                
            if thisSpikeIdx > halfWidth && (thisSpikeIdx + halfWidth) * dataTypeNBytes < rawFileInfo.bytes % check that it's not out of bounds

                bytei = ((thisSpikeIdx - halfWidth) * nChannels) * dataTypeNBytes;
                fseek(fid, bytei, 'bof');
                data0 = fread(fid, nChannels*spikeWidth, 'int16=>int16'); % read individual waveform from binary file
                frewind(fid);
                data = reshape(data0, nChannels, []);
                %         if whitenBool
                %             [data, mu, invMat, whMat]=whiten(double(data));
                %         end
                %         if size(data, 2) == spikeWidth
                %             rawWaveforms(iCluster).spkMap(:, :, iSpike) = data;
                %         end
                rawWaveforms(iCluster).spkMap(:, :, iSpike) = data(1:nChannels-param.nSyncChannels, :, :); %remove sync channel
                
            end

        end

        if param.saveMultipleRaw
            tmpspkmap = permute(rawWaveforms(iCluster).spkMap,[2,1,3]); % Compatible with UnitMatch QQ
            writeNPY(tmpspkmap, fullfile(rawFileInfo.folder,['RawWaveforms_' rawFileInfo.name],['Unit' num2str(iCluster) '_RawSpikes.npy']))
        end

        rawWaveforms(iCluster).spkMapMean = nanmean(rawWaveforms(iCluster).spkMap, 3);
        rawWaveformsFull(iCluster, :, :) = rawWaveforms(iCluster).spkMapMean - mean(rawWaveforms(iCluster).spkMapMean(:, 1:10), 2);

        spkMapMean_sm = smoothdata(rawWaveforms(iCluster).spkMapMean, 1, 'gaussian', 5);

        [~, rawWaveformsPeakChan(iCluster)] = max(max(spkMapMean_sm, [], 2)-min(spkMapMean_sm, [], 2));
%          figure();
%          plot(spkMapMean_sm(rawWaveformsPeakChan(iCluster),:))

        if (mod(iCluster, 100) == 0 || iCluster == nClust) && verbose
            fprintf(['\n   Finished ', num2str(iCluster), ' / ', num2str(nClust), ' units.']);
        end

    end

    fclose(fid);


    %save(fullfile(spikeFile.folder, 'rawWaveforms.mat'), 'rawWaveforms', '-v7.3');
    if ~isfolder(savePath)
        mkdir(savePath)
    end
    writeNPY(rawWaveformsFull, fullfile(savePath, 'templates._bc_rawWaveforms.npy'))
    writeNPY(rawWaveformsPeakChan, fullfile(savePath, 'templates._bc_rawWaveformPeakChannels.npy'))

      % save average 
    average_baseline = arrayfun(@(x) squeeze(nanmean(rawWaveforms(x).spkMap(rawWaveformsPeakChan(x),...
        1:param.waveformBaselineNoiseWindow,:),2)), 1:nClust, 'UniformOutput',false);
    average_baseline_cat = cat(1, average_baseline{:});
    average_baseline_spikeCount = arrayfun(@(x) size(rawWaveforms(x).spkMap,3), 1:nClust);
    average_baseline_idx = arrayfun(@(x) ones(average_baseline_spikeCount(x),1)*x, 1:nClust, 'UniformOutput',false);
    average_baseline_idx_cat = cat(1, average_baseline_idx{:});
    
    writeNPY(average_baseline_cat, fullfile(savePath, 'templates._bc_baselineNoiseAmplitude.npy'))
    writeNPY(average_baseline_idx_cat, fullfile(savePath, 'templates._bc_baselineNoiseAmplitudeIndex.npy'))
end
%% estimate signal-to-noise ratio 
clustInds = unique(spikeTemplates);
    nClust = numel(clustInds);
if ~isempty(fullfile(savePath, 'templates._bc_baselineNoiseAmplitude.npy'))
    
    average_baseline_cat = readNPY(fullfile(savePath, 'templates._bc_baselineNoiseAmplitude.npy'));
    average_baseline_idx_cat = readNPY(fullfile(savePath, 'templates._bc_baselineNoiseAmplitudeIndex.npy'));
    baseline_mad = arrayfun(@(x) median(average_baseline_cat(average_baseline_idx_cat==x) - ...
         nanmean(average_baseline_cat(average_baseline_idx_cat==x))), 1:nClust); % median absolute deviation of
    % time just before waveform. we use this as a proxy to evaluate the
    % overall amount of noise for each unit's channel 
    signalToNoiseRatio = max(max(rawWaveformsFull(:,rawWaveformsPeakChan,:))) ./ baseline_mad;
else
    fprintf('No saved waveform baseline file found, skipping signal to noise calculation')
    signalToNoiseRatio = nan(nClust,1);
end

