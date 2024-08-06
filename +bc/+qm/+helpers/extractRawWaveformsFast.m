
function [rawWaveformsFull, rawWaveformsPeakChan, signalToNoiseRatio] = extractRawWaveformsFast(param, spikeTimes_samples, ...
    spikeTemplates, reExtract, savePath, verbose)
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
rawWaveformFolder = dir(fullfile(savePath, 'templates._bc_rawWaveforms.npy'));

if ~isempty(rawWaveformFolder) && reExtract == 0 % no need to extract data, 
    % simply load it in
    rawWaveformsFull = readNPY(fullfile(savePath, 'templates._bc_rawWaveforms.npy'));
    rawWaveformsPeakChan = readNPY(fullfile(savePath, 'templates._bc_rawWaveformPeakChannels.npy'));

else
%% Extract raw waveforms
    %% Initialize parameters
    nChannels = param.nChannels; % (385)
    nSpikesToExtract = param.nRawSpikesToExtract;
    spikeWidth = param.spikeWidth;
    switch spikeWidth
        case 82
            % spikeWidth = 82: kilosort <4, baseline = 1:41
            halfWidth = spikeWidth/2;
        case 61
            % spikeWidth = 61: kilosort 4, baseline = 1:20
            halfWidth = 20; 
    end
    dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));
    clustInds = unique(spikeTemplates);
    nClust = numel(clustInds);
    rawFileInfo = dir(param.rawFile);
    BatchSize = 5000;
    if param.saveMultipleRaw && ~isfolder(fullfile(savePath,'RawWaveforms'))
        mkdir(fullfile(savePath,'RawWaveforms'))
    end

    % Kilosort 4: number of "baseline samples" needs to be defined

    fprintf('\n Extracting raw waveforms from %s ...', param.rawFile)
    % Get binary file name
    fid = fopen(param.rawFile, 'r');

    %% Interate over spike clusters and find spikes associated with them
    % Initialize and pre-allocate variables
    rawWaveforms = struct;
    rawWaveformsFull = nan(nClust, nChannels-param.nSyncChannels, spikeWidth);
    rawWaveformsPeakChan = nan(nClust, 1);
    average_baseline = cell(1,nClust);

    % loop over spike clusters 
    for iCluster = 1:nClust
        % Get cluster information
        rawWaveforms(iCluster).clInd = clustInds(iCluster);
        rawWaveforms(iCluster).spkInd = spikeTimes_samples(spikeTemplates == clustInds(iCluster));

        % Determine # of spikes to extract
        if numel(rawWaveforms(iCluster).spkInd) >= nSpikesToExtract
            spksubi = round(linspace(1, numel(rawWaveforms(iCluster).spkInd), nSpikesToExtract))';
            rawWaveforms(iCluster).spkIndsub = rawWaveforms(iCluster).spkInd(spksubi);
        else
            rawWaveforms(iCluster).spkIndsub = rawWaveforms(iCluster).spkInd;
        end
        nSpkLocal = numel(rawWaveforms(iCluster).spkIndsub);

        % loop over spikes for this cluster
        rawWaveforms(iCluster).spkMap = nan(nChannels-param.nSyncChannels, spikeWidth, nSpkLocal);
        for iSpike = 1:nSpkLocal
            thisSpikeIdx = rawWaveforms(iCluster).spkIndsub(iSpike);
                
            if ((thisSpikeIdx - spikeWidth) * nChannels) * dataTypeNBytes > spikeWidth &&...
                    (thisSpikeIdx + spikeWidth) * nChannels * dataTypeNBytes < rawFileInfo.bytes % check that it's not out of bounds

                bytei = ((thisSpikeIdx - halfWidth) * nChannels) * dataTypeNBytes;
                fseek(fid, bytei, 'bof');
                data0 = fread(fid, nChannels*spikeWidth, 'int16=>int16'); % read individual waveform from binary file
                frewind(fid);
                data = reshape(data0, nChannels, []);
                
                % detrend spike if required 
                if param.detrendWaveform
                    rawWaveforms(iCluster).spkMap(:, :, iSpike) = permute(detrend(double(permute(data(1:nChannels-param.nSyncChannels, :),[2,1]))), [2,1]);
                else
                    rawWaveforms(iCluster).spkMap(:, :, iSpike) = data(1:nChannels-param.nSyncChannels, :); %remove sync channel
                end
             
            end

        end

        % % TODO align raw spikes to each other (using the trough) 
        % clearvars meanWaveform_temp peakChan_temp
        % % baseline subtract, smooth
        % meanWaveform_temp = nanmean(rawWaveforms(iCluster).spkMap,3);
        % [~, peakChan_temp] = max(max(meanWaveform_temp, [], 2) - min(meanWaveform_temp, [], 2)); % maximum channel per cluster 
        % [~, troughLocation] = min(squeeze(rawWaveforms(iCluster).spkMap(peakChan_temp, :, :)));
        
        
        % save waveforms for unitmatch 
        if param.saveMultipleRaw
            tmpspkmap = permute(rawWaveforms(iCluster).spkMap,[2,1,3]); % Compatible with UnitMatch QQ
            %Do smoothing in batches
            nBatch = ceil(nSpkLocal./BatchSize);
            for bid = 1:nBatch
                spkId = (bid-1)*BatchSize+(1:BatchSize);
                spkId(spkId>nSpkLocal) = []; 
                tmpspkmap(:,:,spkId) = smoothdata(double(tmpspkmap(:,:,spkId)),1,'gaussian',5); % smooth first
                tmpspkmap(:,:,spkId) = tmpspkmap(:,:,spkId) - mean(tmpspkmap(1:param.waveformBaselineNoiseWindow,:,spkId),1); % Subtract baseline 
            end
            % Save two averages for UnitMatch
            tmpspkmap = arrayfun(@(X) nanmedian(tmpspkmap(:,:,(X-1)*floor(size(tmpspkmap,3)/2)+1:X*floor(size(tmpspkmap,3)/2)),3),1:2,'Uni',0);
            tmpspkmap = cat(3,tmpspkmap{:});
            writeNPY(tmpspkmap, fullfile(savePath,'RawWaveforms',['Unit' num2str(clustInds(iCluster)-1) '_RawSpikes.npy'])) % Back to 0-indexed (same as Kilosort)
        end

        % get average, baseline-subtracted and smoothed raw waveform
        rawWaveforms(iCluster).spkMapMean = nanmean(rawWaveforms(iCluster).spkMap, 3); % initialize and pre-allocate
        rawWaveformsFull(iCluster, :, :) = rawWaveforms(iCluster).spkMapMean - ...
            mean(rawWaveforms(iCluster).spkMapMean(:, 1:param.waveformBaselineNoiseWindow), 2); % remove baseline
        spkMapMean_sm = smoothdata(rawWaveforms(iCluster).spkMapMean, 2, 'gaussian', 5); % smooth along dimension 2 (time)

        [~, rawWaveformsPeakChan(iCluster)] = max(max(spkMapMean_sm, [], 2) - min(spkMapMean_sm, [], 2)); % maximum channel per cluster 
        average_baseline{iCluster} = squeeze(nanmean(rawWaveforms(iCluster).spkMap(rawWaveformsPeakChan(iCluster),...
            1:param.waveformBaselineNoiseWindow,:),3)); % waveform baseline (for signal-to-noise calculation)

        % delete current cluster raw spikes (for memory)
        rawWaveforms(iCluster).spkMap = [];

        % display progress
        if (mod(iCluster, 100) == 0 || iCluster == nClust) && verbose
            fprintf(['\n   Finished ', num2str(iCluster), ' / ', num2str(nClust), ' units.']);
        end

    end
    % close file 
    fclose(fid);

    % save extracted mean raw waveforms 
    if ~isfolder(savePath)
        mkdir(savePath)
    end
    writeNPY(rawWaveformsFull, fullfile(savePath, 'templates._bc_rawWaveforms.npy'))
    writeNPY(rawWaveformsPeakChan, fullfile(savePath, 'templates._bc_rawWaveformPeakChannels.npy'))

    % save mean raw waveform baseline average (for signal-to-noise
    % calculation)
    average_baseline_cat = cat(2, average_baseline{:})';
    average_baseline_idx = arrayfun(@(x) ones(param.waveformBaselineNoiseWindow,1)*x, 1:nClust, 'UniformOutput',false);
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

    signalToNoiseRatio = cell2mat(arrayfun(@(X) max(abs(squeeze(rawWaveformsFull(X,rawWaveformsPeakChan(X),:)))) ./...
        mad(average_baseline_cat(average_baseline_idx_cat==X)),1:nClust,'Uni',0))';

    %signalToNoiseRatio = cell2mat(arrayfun(@(X) max(abs(squeeze(rawWaveformsFull(X,rawWaveformsPeakChan(X),:)))) ./...
    %    var(average_baseline_cat(average_baseline_idx_cat==X)),1:nClust,'Uni',0))';

    %signalToNoiseRatio = cell2mat(arrayfun(@(X) max(abs(squeeze(rawWaveformsFull(X,rawWaveformsPeakChan(X),:))),[],'omitnan') ./...
    %    var(average_baseline_cat(average_baseline_idx_cat==X)),1:nClust,'Uni',0), 'omitnan')';

else
    fprintf('No saved waveform baseline file found, skipping signal to noise calculation')
    signalToNoiseRatio = nan(nClust,1);
end

else
    rawWaveformsFull = [];
    rawWaveformsPeakChan = [];
    signalToNoiseRatio = [];
end
