function [rawWaveformsFull, rawWaveformsPeakChan] = bc_extractRawWaveformsFast(param, spikeTimes_samples,...
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

%% Initialize stuff
% Get spike times and indices
nChannels = param.nChannels; % (385)
nSpikesToExtract = param.nRawSpikesToExtract;
spkWid = 82; 
halfWid = spkWid/2;
dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));
clustInds = unique(spikeTemplates);
nClust = numel(clustInds);

% Get binary file name
fid = fopen(param.rawFile, 'r');



%% Interate over spike clusters and find all the data associated with them
rawWaveforms = struct;

for iClust = 1:nClust
    rawWaveforms(iClust).clInd = clustInds(iClust);
    rawWaveforms(iClust).spkInd = spikeTimes_samples(spikeTemplates == clustInds(iClust));
    if numel(rawWaveforms(iClust).spkInd) >= nSpikesToExtract
        spksubi = round(linspace(1,numel(rawWaveforms(iClust).spkInd),nSpikesToExtract))';                        
        rawWaveforms(iClust).spkIndsub = rawWaveforms(iClust).spkInd(spksubi);        
    else
        rawWaveforms(iClust).spkIndsub = rawWaveforms(iClust).spkInd;
    end
    nSpkLocal = numel(rawWaveforms(iClust).spkIndsub);

    rawWaveforms(iClust).spkMap = nan(nChannels,spkWid,nSpkLocal);
    for iSpike = 1:nSpkLocal
        spki = rawWaveforms(iClust).spkIndsub(iSpike);
        bytei = ((spki-halfWid)*nChannels)*dataTypeNBytes;
        fseek(fid,bytei,'bof');
        data0 = fread(fid, nChannels*spkWid, 'int16=>int16'); % read individual waveform from binary file
        frewind(fid);
        data = reshape(data0,nChannels,[]);
%         if whitenBool
%             [data, mu, invMat, whMat]=whiten(double(data));
%         end
        if size(data,2) == spkWid
            rawWaveforms(iClust).spkMap(:,:,iSpike) = data;    
        end
    end
    rawWaveforms(iClust).spkMapMean = nanmean(rawWaveforms(iClust).spkMap,3);
    rawWaveformsFull(iCluster,:,:) = rawWaveforms(iClust).spkMapMean-mean(rawWaveforms(iClust).spkMapMean(:,1:10),2);
    
    spkMapMean_sm = smoothdata(rawWaveforms(iClust).spkMapMean,1,'gaussian',5);
        
    [~,rawWaveformsPeakChan(iCluster)] = max(max(spkMapMean_sm,[],2)-min(spkMapMean_sm,[],2));
   %  figure();
    % plot(spkMapMean_sm(rawWaveforms(iClust).peakChan,:))
    if (mod(iClust,100) == 0 || iClust == nClust) && verbose
        fprintf(['\n   Finished ',num2str(iClust),' of ',num2str(nClust),' units.']);
        figure; imagesc(spkMapMean_sm)
        title(['Unit ID: ',num2str(iClust)]);
        colorbar;
    end
    
end

fclose(fid);

        rawWaveformFolder = dir(fullfile(savePath,'templates._bc_rawWaveforms.npy'));
        if isempty(rawWaveformFolder) || reExtract
            %save(fullfile(spikeFile.folder, 'rawWaveforms.mat'), 'rawWaveforms', '-v7.3');
            writeNPY(rawWaveformsFull, fullfile(savePath, 'templates._bc_rawWaveforms.npy'))
            writeNPY(rawWaveformsPeakChan, fullfile(savePath, 'templates._bc_rawWaveformPeakChannels.npy'))

            if param.saveMultipleRaw
                writeNPY(spikeMap, fullfile(savePath, 'templates._bc_multi_rawWaveforms.npy'))
            end
        end

end