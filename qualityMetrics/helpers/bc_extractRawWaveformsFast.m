
function rawWaveforms = bc_extractRawWaveformsFast(param, spikeTimes_samples, spikeTemplates, reExtract, verbose, maxChannels)
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
if nargin<6%only apply when decompressing ' on the fly' 
    UseMaxChannels = 0;
else
    UseMaxChannels = 1;
end
rawFolder = param.rawFolder;
if isfield(param,'tmpFolder')
    tmpFolder = param.tmpFolder;
else
    tmpFolder = rawFolder;
end
nChannels = param.nChannels;
nSpikesToExtract =  param.nRawSpikesToExtract;


%% check if waveforms already extracted
% Get binary file name
if iscell(tmpFolder)
    tmpFolder = fileparts(tmpFolder{1});
elseif sum(tmpFolder(end-2:end) == '/..') == 3
    [tmpFolder, filename] = fileparts(tmpFolder(1:end-3));
end
spikeFile = dir(fullfile(tmpFolder, [filename '.*bin']));
if isempty(spikeFile)
    spikeFile = dir(fullfile(tmpFolder, '/*.dat')); %openEphys format
end
if size(spikeFile,1) > 1
    spikeFile = dir(fullfile(tmpFolder, '*tcat*.ap.*bin'));
end

if iscell(rawFolder)
    rawFolder = fileparts(rawFolder{1});
elseif sum(rawFolder(end-2:end) == '/..') == 3
    [rawFolder, filename] = fileparts(rawFolder(1:end-3));
end

rawWaveformFolder = dir(fullfile(rawFolder, 'rawWaveforms.mat'));

fname = spikeFile.name;
dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));

if any(strfind(fname,'cbin'))
    disp('This is compressed data. Use Python integration... If you don''t have that option please uncompress data first')
    UsePython = 1; %Choose if you want to compress or usepython integration 
    % Read original bytes
    meta = ReadMeta2(spikeFile.folder);
    n_samples = round(str2num(meta.fileSizeBytes)/dataTypeNBytes/nChannels);
    SR = meta.imSampRate;
else
    UsePython = 0;
end

d = dir(fullfile(tmpFolder, fname));


if ~isempty(rawWaveformFolder) && reExtract == 0
    load(fullfile(rawFolder, 'rawWaveforms.mat'));
else

    %% Intitialize

    if ~UsePython
        fid = fopen(fullfile(spikeFile.folder, fname), 'r');
    end
    spikeWidth = 83;
    halfWidth = floor(spikeWidth / 2);
    clustInds = unique(spikeTemplates);
    nClust = numel(clustInds);

    %     try % memMap to check you have correct number of channels, if not remove one channel
    %         n_samples = spikeFile.bytes/ (nChannels * dataTypeNBytes);
    %         memmapfile(fullfile(spikeFile.folder, fname),'Format',{'int16',[nChannels,n_samples],'data'});
    %     catch
    %         disp(['Guessing correct number of channels is ', num2str(nChannels-1)])
    %         nChannels = nChannels - 1;
    %     end

    %% Interate over spike clusters and find all the data associated with them
    rawWaveforms = struct;
    allSpikeTimes = spikeTimes_samples;
    disp('Extracting raw waveforms ...')
    % array
    for iCluster = 1:nClust
        spikeIndices = allSpikeTimes(spikeTemplates == clustInds(iCluster)); %
        if numel(spikeIndices) >= nSpikesToExtract % extract a random subset of regularly spaced raw waveforms
            spikeIndices = sort(datasample(spikeIndices, nSpikesToExtract));

            %             spksubi = round(linspace(1, numel(spikeIndices), nSpikesToExtract))';
            %             spikeIndices = spikeIndices(spksubi);
        end
        nSpikesEctractHere = numel(spikeIndices);
        if nChannels == 385
            spikeMap = nan(nChannels-1, spikeWidth, nSpikesEctractHere,'single');
        else
            spikeMap = nan(nChannels, spikeWidth, nSpikesEctractHere,'single');
        end
        if UsePython
            allsamples = arrayfun(@(X) spikeIndices(X)-halfWidth:spikeIndices(X)+halfWidth,1:length(spikeIndices),'UniformOutput',0);
            allsamples = cat(1,allsamples{:})';
            batchsize = 65000;
            batchn = ceil((max(allsamples(:))-min(allsamples(:)))./batchsize);
            allsamplestmp = min(allsamples(:)):max(allsamples(:))+batchsize;
            batchidx = arrayfun(@(X) allsamplestmp(batchsize*(X-1)+1):allsamplestmp(batchsize*X),1:batchn,'UniformOutput',0);
            batchidx = find(cell2mat(cellfun(@(X) any(ismember(X,allsamples)),batchidx,'UniformOutput',0)));
           
            if UseMaxChannels
                channels2take = maxChannels(iCluster)-2:maxChannels(iCluster)+2;
                channels2take(channels2take<1|channels2take>nChannels-1) = [];
            else
                channels2take = 1:nChannels;
            end
            for bid = 1:length(batchidx)
                try
                    endidx = batchsize*batchidx(bid)+spikeWidth;
                    if endidx>length(allsamplestmp) || allsamplestmp(endidx)>n_samples
                        endidx=min([length(allsamplestmp) find(allsamplestmp==n_samples)]);
                    end
                    tmpdataidx = arrayfun(@(X) find(ismember(allsamplestmp(batchsize*(batchidx(bid)-1)+1):allsamplestmp(endidx),allsamples(:,X))),1:size(allsamples,2),'UniformOutput',0);
                    tmpdata = zeros(1,length(allsamplestmp(batchsize*(batchidx(bid)-1)+1):allsamplestmp(endidx)),'uint16');
                    tmpdata = cellfun(@(X) tmpdata(X),tmpdataidx,'UniformOutput',0);
                    tmpdataidx2 = find(~cell2mat(cellfun(@isempty,tmpdataidx,'UniformOutput',0)));
                    tmpdataidx2(cell2mat(arrayfun(@(X) length(tmpdata{X})<spikeWidth,tmpdataidx2,'UniformOutput',0))) = [];

                    if isempty(tmpdataidx2)
                        continue
                    end
                catch ME
                    keyboard
                    disp(ME)
                    disp('Make sure to use MATLAB>2022a and compatible python version, in an environment that has the modules phylib, pathlib, and matlab installed')
                    disp('e.g. pyversion("C:\Users\EnnyB\anaconda3\envs\phy\pythonw.exe")')
                    disp('Also make sure you input the path in a python-compatible way!')
                end

                for chid=channels2take
                    tmpdata = pyrunfile("Ephys_Reader_FromMatlab.py","chunk",...
                        datapath = strrep(fullfile(spikeFile.folder,fname),'\','/'),start_time=allsamplestmp(batchsize*(batchidx(bid)-1)+1)-1,end_time=allsamplestmp(endidx),channel=chid-1); %0-indexed!!
                    tmpdata=uint16(tmpdata);
                    tmpdata = cellfun(@(X) tmpdata(X),tmpdataidx,'UniformOutput',0);

                    % Put it back in the correct order
                    spikeMap(chid,:,tmpdataidx2)=cat(1,tmpdata{tmpdataidx2})';
                end

            end
        else
            for iSpike = 1:nSpikesEctractHere
                thisSpikeIdx = spikeIndices(iSpike);
                if thisSpikeIdx > halfWidth && (thisSpikeIdx + halfWidth) * dataTypeNBytes < d.bytes % check that it's not out of bounds

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
        end
        spikeMapMean = nanmean(spikeMap, 3);
        spikeMap = permute(spikeMap,[1,3,2]);
        rawWaveforms(iCluster).spkMap = permute(spikeMap - mean(spikeMap(:,:,1:10),3),[1,3,2]);
        rawWaveforms(iCluster).spkMapMean = spikeMapMean - mean(spikeMapMean(:, 1:10), 2);

        spkMapMean_sm = smoothdata(rawWaveforms(iCluster).spkMapMean, 2, 'gaussian', 5); %Switched the dimension here. I guess you want the waveform to be smooth??

        [~, rawWaveforms(iCluster).peakChan] = max(max(abs(spkMapMean_sm), [], 2), [], 1);%QQ buggy sometimes

        %         [~, maxChannels] = max(max(abs(templateWaveforms), [], 2), [], 3);
        %         close all;
        %
        %                 clf;
        %                 for iSpike = 1:10
        %                     plot(spikeMap(rawWaveforms(iCluster).peakChan, :, iSpike));
        %                     hold on;
        %                 end
        %                 figure()
        %                 clf;
        %                 plot(rawWaveforms(iCluster).spkMapMean(rawWaveforms(iCluster).peakChan, :));
        %                 hold on;
        %
        %
        %                 figure()
        %                 clf;
        %                 plot(squeeze(templateWaveforms(uniqueTemplates(iCluster),:,maxChannels(uniqueTemplates(iCluster)))));
        %                 hold on;
        %                 plot(squeeze(templateWaveforms(uniqueTemplates(iCluster),:,goodChannels(rawWaveforms(iCluster).peakChan))));


        if (mod(iCluster, 20) == 0 || iCluster == nClust) && verbose
            fprintf(['\n   Extracted ', num2str(iCluster), '/', num2str(nClust), ' raw waveforms.']);
            %figure; imagesc(spkMapMean_sm)
            %title(['Unit ID: ', num2str(i)]);
            %colorbar;
        end
    end

    fclose(fid);
    rawWaveformFolder = dir(fullfile(rawFolder, 'rawWaveforms.mat'));
    if isempty(rawWaveformFolder) || reExtract
        save(fullfile(rawFolder, 'rawWaveforms.mat'), 'rawWaveforms', '-v7.3');
    end
end
end