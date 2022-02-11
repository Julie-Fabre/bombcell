function spkShapes = extractSPKs(rawFolder, nChannels, nSpikesToExtract, spikeTimes, spikeTemplates, verbose)
%% Intitialize stuff
% Get spike times and indices
nChans = nChannels; % (385)
nSpks = nSpikesToExtract;
spkWid = 90; halfWid = spkWid/2;
spkt = spikeTimes;

clustInds = unique(spikeTemplates);
nClust = numel(clustInds);

% Get binary file name
spikeFile = dir(fullfile(rawFolder, '*.ap.bin'));
    if isempty(spikeFile)
        spikeFile = dir(fullfile(rawFolder, '/*.dat')); %openEphys format
    end
    
spkFile = dir(fullfile(spikeFile.folder,'*.dat'));
fname = spkFile.name;
fid = fopen(fullfile(spikeFile.folder, fname), 'r');



%% Interate over spike clusters and find all the data associated with them
spkShapes = struct;
% spkIndsAll0 = zeros(nClust*nSpks,2)*nan; % This will be matrix that has clustID in column 1 and spkInds in column 2 for everyone, so that we can read through binary file only once, sequentially

for i = 1:nClust
    spkShapes(i).clInd = clustInds(i);
    spkShapes(i).spkInd = spkt(spikeTemplates == clustInds(i));
    if numel(spkShapes(i).spkInd) >= nSpks
        spksubi = round(linspace(1,numel(spkShapes(i).spkInd),nSpks))';                        
        spkShapes(i).spkIndsub = spkShapes(i).spkInd(spksubi);        
    else
        spkShapes(i).spkIndsub = spkShapes(i).spkInd;
    end
    nSpkLocal = numel(spkShapes(i).spkIndsub);
%     spkIndsAll0((i-1)*nSpks+1:(i-1)*nSpks+nSpkLocal,1) = clustInds(i);
%     spkIndsAll0((i-1)*nSpks+1:(i-1)*nSpks+nSpkLocal,2) = spkShapes(i).spkIndsub;
    
    spkShapes(i).spkMap = nan(nChans,spkWid,nSpkLocal);
    for j = 1:nSpkLocal
        spki = spkShapes(i).spkIndsub(j);
        bytei = ((spki-halfWid)*nChans)*2;
        fseek(fid,bytei,'bof');
        data0 = fread(fid, nChans*spkWid, 'int16=>int16'); % read individual waveform from binary file
        frewind(fid);
        data = reshape(data0,nChans,[]);
        if whitenBool
            [data, mu, invMat, whMat]=whiten(double(data));
        end
        if size(data,2) == spkWid
            spkShapes(i).spkMap(:,:,j) = data;    
        end
    end
    spkShapes(i).spkMapMean = nanmean(spkShapes(i).spkMap,3);
    spkShapes(i).spkMapMean = spkShapes(i).spkMapMean-mean(spkShapes(i).spkMapMean(:,1:10),2);
    
    spkMapMean_sm = smoothdata(spkShapes(i).spkMapMean,1,'gaussian',5);
        
    [~,spkShapes(i).peakChan] = max(max(spkMapMean_sm,[],2)-min(spkMapMean_sm,[],2));
    figure();
    plot(spkMapMean_sm(spkShapes(i).peakChan,:))
    if (mod(i,20) == 0 || i == nClust) && plotbool
        fprintf(['\n   Finished ',num2str(i),' of ',num2str(nClust),' units.']);
        figure; imagesc(spkMapMean_sm)
        title(['Unit ID: ',num2str(i)]);
        colorbar;
    end
    
end

fclose(fid);

save(fullfile(inits.savefolder,'spkShapes.mat'),'spkShapes','-v7.3');

end