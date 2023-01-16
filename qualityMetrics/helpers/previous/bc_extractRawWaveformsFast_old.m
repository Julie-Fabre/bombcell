
function [rawWaveformsFull, rawWaveformsPeakChan]= bc_extractRawWaveformsFast_old(param, ...
    spikeTimes_samples, spikeTemplates, reExtract , savePath, templateWaveforms, uniqueTemplates,verbose)

% JF, Get raw waveforms for all templates
% ------
% Inputs
% ------
% nChannels: number of recorded channels (including sync), (eg 385)
% nSpikesToExtract: number of spikes to extract per template
% spikeTimes_samples: nSpikes × 1 uint64 vector giving each spike time in samples (*not* seconds)
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
% Get binary file name
spikeFile = dir(param.rawFile);
nChannels = param.nChannels;
nSpikesToExtract = param.nRawSpikesToExtract;

rawWaveformFolder = dir(fullfile(savePath, 'templates._bc_rawWaveforms.npy'));
old_rawWaveformFolder = dir(fullfile(savePath, 'templates._jf_rawWaveforms.npy'));
dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));

fname = spikeFile.name;


if ~isempty(rawWaveformFolder) && reExtract == 0

    rawWaveformsFull = readNPY(fullfile(savePath, 'templates._bc_rawWaveforms.npy'));
    rawWaveformsPeakChan = readNPY(fullfile(savePath, 'templates._bc_rawWaveformPeakChannels.npy'));
    if param.saveMultipleRaw
        spikeMap = readNPY(fullfile(savePath, 'templates._bc_multi_rawWaveforms.npy'));
    end
elseif ~isempty(old_rawWaveformFolder) && reExtract == 0
    rawWaveformsFull = readNPY(fullfile(savePath, 'templates._jf_rawWaveforms.npy'));
    rawWaveformsPeakChan = readNPY(fullfile(savePath, 'templates._jf_rawWaveformPeakChannels.npy'));
    if param.saveMultipleRaw
        try
            spikeMap = readNPY(fullfile(savePath, 'templates._jf_Multi_rawWaveforms.npy'));
        catch
            fprintf('Could not load saved individual raw waveforms from %s, keeping only average raw waveforms \n', savePath)
        end
    end
else

    %% Intitialize
    

    fid = fopen(fullfile(spikeFile.folder, fname), 'r');
    spikeWidth = 82;
    halfWidth = spikeWidth / 2;
    clustInds = unique(spikeTemplates);
    nClust = numel(clustInds);


    fname = spikeFile.name;

    dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));
%     try % memMap to check you have correct number of channels, if not remove one channel 
%         n_samples = spikeFile.bytes/ (nChannels * dataTypeNBytes);
%         memmapfile(fullfile(spikeFile.folder, fname),'Format',{'int16',[nChannels,n_samples],'data'});
%     catch
%         disp(['Guessing correct number of channels is ', num2str(nChannels-1)])
%         nChannels = nChannels - 1;
%     end
    
    %% Interate over spike clusters and find all the data associated with them
    rawWaveforms = struct;
    rawWaveforms.spkMapMean = nan(nClust, 384, 82);
    rawWaveforms.peakChan = nan(nClust, 1);
    allspikeTimes_samples = spikeTimes_samples;
    disp('Extracting raw waveforms ...')
    % array
    for iCluster = 1:nClust
iCluster=iCluster+1
        spikeIndices = allspikeTimes_samples(spikeTemplates == clustInds(iCluster));
        if numel(spikeIndices) >= nSpikesToExtract % extract a random subset of regularly spaced raw waveforms
            spksubi = round(linspace(1, numel(spikeIndices), nSpikesToExtract))';
            spikeIndices = spikeIndices(spksubi);
        end
        nSpikesEctractHere = numel(spikeIndices);
        if nChannels == 385
            spikeMap = nan(nChannels-1, spikeWidth, nSpikesEctractHere);
        else
            spikeMap = nan(nChannels, spikeWidth, nSpikesEctractHere);
       
        end
        for iSpike = 1:nSpikesEctractHere
            thisSpikeIdx = spikeIndices(iSpike);
            if thisSpikeIdx > halfWidth && (thisSpikeIdx + halfWidth) * dataTypeNBytes < spikeFile.bytes % check that it's not out of bounds
                byteIdx = int64(((thisSpikeIdx - halfWidth) * nChannels) * dataTypeNBytes); % int64 to prevent overflow on crappy windows machines that are incredibly inferior to linux
                fseek(fid, byteIdx, 'bof'); % from beginning of file
                data = fread(fid, [nChannels, spikeWidth], 'int16=>int16'); % read individual waveform from binary file
                frewind(fid);

                if size(data, 2) == spikeWidth && nChannels == 385
                    spikeMap(:, :, iSpike) = data(1:nChannels-1, :, :); %remove sync channel
                elseif size(data, 2) == spikeWidth
                    spikeMap(:, :, iSpike) = data(1:nChannels, :, :); 
                end


            end
        end
        spikeMapMean = nanmean(spikeMap, 3);

        rawWaveforms.spkMapMean(iCluster,:,:) = spikeMapMean - mean(spikeMapMean(:, 1:10), 2);

        spkMapMean_sm = smoothdata(spikeMapMean, 1, 'gaussian', 5);

        [~, rawWaveforms.peakChan(iCluster)] = max(max(abs(spkMapMean_sm), [], 2), [], 1);%QQ buggy sometimes

        [~, maxChannels] = max(max(abs(templateWaveforms), [], 2), [], 3);
        
        clf;
               subplot(311)
                for iSpike = 1:10
                    plot(spikeMap(rawWaveforms.peakChan(iCluster), :, iSpike));
                    hold on;
                end
                
                subplot(312)
                plot(squeeze(rawWaveforms.spkMapMean(iCluster, rawWaveforms.peakChan(iCluster), :)));
                hold on;
                
                
                subplot(313)
                plot(squeeze(templateWaveforms(uniqueTemplates(iCluster),:,maxChannels(uniqueTemplates(iCluster)))));
                hold on;
                plot(squeeze(templateWaveforms(uniqueTemplates(iCluster),:,goodChannels(rawWaveforms(iCluster).peakChan))));
                
                
        if (mod(iCluster, 100) == 0 || iCluster == nClust) && verbose
            fprintf(['\n   Extracted ', num2str(iCluster), '/', num2str(nClust), ' raw waveforms']);
            %figure; imagesc(spkMapMean_sm)
            %title(['Unit ID: ', num2str(i)]);
            %colorbar;
        end

    end

    fclose(fid);
    rawWaveformsFull = rawWaveforms.spkMapMean;
    rawWaveformsPeakChan  = rawWaveforms.peakChan;
    rawWaveformFolder = dir(fullfile(savePath,'templates._bc_rawWaveforms.npy'));
        if isempty(rawWaveformFolder) || reExtract
            %save(fullfile(spikeFile.folder, 'rawWaveforms.mat'), 'rawWaveforms', '-v7.3');
            writeNPY(rawWaveformsFull, fullfile(savePath, 'templates._bc_rawWaveforms.npy'))
            writeNPY(rawWaveformsPeakChan, fullfile(savePath, 'templates._bc_rawWaveformPeakChannels.npy'))

            if param.saveMultipleRaw
                %writeNPY(spikeMap, fullfile(savePath, 'templates._bc_multi_rawWaveforms.npy'))
            end
        end
end
end
