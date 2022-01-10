function rawWaveforms = dc_extractRawWaveformsFast(nChannels, nSpikesToExtract, spikeTimes, spikeClusters, rawFolder, plotThis, whitenThis)

%% Intitialize stuff
% Get spike times and indices
spkWid = 82;
halfWid = spkWid / 2;

clustInds = unique(spikeClusters);
nClust = numel(clustInds);

% Get binary file name
spkFile = dir(fullfile(rawFolder, '*.ap.bin'));
fname = spkFile.name;
fid = fopen(fullfile(rawFolder, fname), 'r');

%% Interate over spike clusters and find all the data associated with them
rawWaveforms = struct;
% spkIndsAll0 = zeros(nClust*nSpks,2)*nan; % This will be matrix that has clustID in column 1 and spkInds in column 2 for everyone, so that we can read through binary file only once, sequentially

for i = 1:nClust
    rawWaveforms(i).clInd = clustInds(i);
    rawWaveforms(i).spkInd = spikeTimes(spikeClusters == clustInds(i));
    if numel(rawWaveforms(i).spkInd) >= nSpikesToExtract
        spksubi = round(linspace(1, numel(rawWaveforms(i).spkInd), nSpikesToExtract))';
        rawWaveforms(i).spkIndsub = rawWaveforms(i).spkInd(spksubi);
    else
        rawWaveforms(i).spkIndsub = rawWaveforms(i).spkInd;
    end
    nSpkLocal = numel(rawWaveforms(i).spkIndsub);
    %     spkIndsAll0((i-1)*nSpks+1:(i-1)*nSpks+nSpkLocal,1) = clustInds(i);
    %     spkIndsAll0((i-1)*nSpks+1:(i-1)*nSpks+nSpkLocal,2) = spkShapes(i).spkIndsub;

    rawWaveforms(i).spkMap = nan(nChannels, spkWid, nSpkLocal);
    for j = 1:nSpkLocal
        spki = rawWaveforms(i).spkIndsub(j);
        bytei = ((spki - halfWid) * nChannels) * 2;
        fseek(fid, bytei, 'bof');
        data0 = fread(fid, nChannels*spkWid, 'int16=>int16'); % read individual waveform from binary file
        frewind(fid);
        data = reshape(data0, nChannels, []);
        if whitenThis
            [data, mu, invMat, whMat] = whiten(double(data));
        end
        if size(data, 2) == spkWid
            rawWaveforms(i).spkMap(:, :, j) = data;
        end
    end
    rawWaveforms(i).spkMapMean = nanmean(rawWaveforms(i).spkMap, 3);
    rawWaveforms(i).spkMapMean = rawWaveforms(i).spkMapMean - mean(rawWaveforms(i).spkMapMean(:, 1:10), 2);

    spkMapMean_sm = smoothdata(rawWaveforms(i).spkMapMean, 1, 'gaussian', 5);

    [~, rawWaveforms(i).peakChan] = max(max(spkMapMean_sm, [], 2)-min(spkMapMean_sm, [], 2));

    if (mod(i, 20) == 0 || i == nClust) && plotThis
        fprintf(['\n   Finished ', num2str(i), ' of ', num2str(nClust), ' units.']);
        figure; imagesc(spkMapMean_sm)
        title(['Unit ID: ', num2str(i)]);
        colorbar;
    end

end

fclose(fid);

save(fullfile(rawFolder, 'rawWaveforms.mat'), 'rawWaveforms', '-v7.3');

end