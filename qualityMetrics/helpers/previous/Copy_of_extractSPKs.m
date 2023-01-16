function rawWaveforms = extractSPKs(rawFile, nChannels, nSpikesToExtract, spikeTimes, spikeTemplates, verbose)
%% Intitialize stuff
% Get spike times and indices
nChans = nChannels; % (385)
nSpks = nSpikesToExtract;
spkWid = 82; 
halfWid = spkWid/2;

clustInds = unique(spikeTemplates);
nClust = numel(clustInds);

% Get binary file name
fid = fopen(rawFile, 'r');



%% Interate over spike clusters and find all the data associated with them
rawWaveforms = struct;
% spkIndsAll0 = zeros(nClust*nSpks,2)*nan; % This will be matrix that has clustID in column 1 and spkInds in column 2 for everyone, so that we can read through binary file only once, sequentially

for iClust = 1:nClust
    rawWaveforms(iClust).clInd = clustInds(iClust);
    rawWaveforms(iClust).spkInd = spikeTimes(spikeTemplates == clustInds(iClust));
    if numel(rawWaveforms(iClust).spkInd) >= nSpks
        spksubi = round(linspace(1,numel(rawWaveforms(iClust).spkInd),nSpks))';                        
        rawWaveforms(iClust).spkIndsub = rawWaveforms(iClust).spkInd(spksubi);        
    else
        rawWaveforms(iClust).spkIndsub = rawWaveforms(iClust).spkInd;
    end
    nSpkLocal = numel(rawWaveforms(iClust).spkIndsub);
%     spkIndsAll0((i-1)*nSpks+1:(i-1)*nSpks+nSpkLocal,1) = clustInds(i);
%     spkIndsAll0((i-1)*nSpks+1:(i-1)*nSpks+nSpkLocal,2) = rawWaveforms(i).spkIndsub;
    
    rawWaveforms(iClust).spkMap = nan(nChans,spkWid,nSpkLocal);
    for iSpike = 1:nSpkLocal
        spki = rawWaveforms(iClust).spkIndsub(iSpike);
        bytei = ((spki-halfWid)*nChans)*2;
        fseek(fid,bytei,'bof');
        data0 = fread(fid, nChans*spkWid, 'int16=>int16'); % read individual waveform from binary file
        frewind(fid);
        data = reshape(data0,nChans,[]);
%         if whitenBool
%             [data, mu, invMat, whMat]=whiten(double(data));
%         end
        if size(data,2) == spkWid
            rawWaveforms(iClust).spkMap(:,:,iSpike) = data;    
        end
    end
    rawWaveforms(iClust).spkMapMean = nanmean(rawWaveforms(iClust).spkMap,3);
    rawWaveforms(iClust).spkMapMean = rawWaveforms(iClust).spkMapMean-mean(rawWaveforms(iClust).spkMapMean(:,1:10),2);
    
    spkMapMean_sm = smoothdata(rawWaveforms(iClust).spkMapMean,1,'gaussian',5);
        
    [~,rawWaveforms(iClust).peakChan] = max(max(spkMapMean_sm,[],2)-min(spkMapMean_sm,[],2));
   %  figure();
    % plot(spkMapMean_sm(rawWaveforms(iClust).peakChan,:))
    if (mod(iClust,100) == 0 || iClust == nClust) && plotbool
        fprintf(['\n   Finished ',num2str(iClust),' of ',num2str(nClust),' units.']);
        figure; imagesc(spkMapMean_sm)
        title(['Unit ID: ',num2str(iClust)]);
        colorbar;
    end
    
end

fclose(fid);

save(fullfile(inits.savefolder,'rawWaveforms.mat'),'rawWaveforms','-v7.3');

end