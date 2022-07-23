function [rawWaveformsFull, rawWaveformsPeakChan, spikeMap] = bc_extractRawWaveformsFast(param, spikeTimes_samples, spikeTemplates,...
    reExtract, verbose)
% JF, Get raw waveforms for all templates
% ------
% Inputs
% ------
% param with:
% rawFolder
% nChannels: number of recorded channels (including sync), (eg 385)
% nSpikesToExtract: number of spikes to extract per template
%
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

rawFolder = param.rawFolder;
if isfield(param,'tmpFolder') && ~isempty(param.tmpFolder) && ~strcmp(param.tmpFolder,'/..')
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


rawWaveformFolder = dir(fullfile(spikeFile.folder, 'templates._jf_rawWaveforms.npy'));

fname = spikeFile.name;
dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));

if any(strfind(fname,'cbin'))
    [~, matlabDate] = version; % check MATLAB version supports decompressing this way 
    if datetime(matlabDate) >= datetime(2022,1,1)
        disp('Raw data is in compressed format, decompressing via python... If you don''t have that option please uncompress data first')
        UsePython = 1; %Choose if you want to compress or usepython integration
        % Read original bytes
        meta = ReadMeta2(spikeFile.folder);
        n_samples = round(str2num(meta.fileSizeBytes)/dataTypeNBytes/nChannels);
        SR = meta.imSampRate;
    else
        error('Raw data is in compressed format, to decompress via python you need at least MATLAB 2022a. Either download a more recent version of MATLAB or uncompress data yourself')
    end
else
    UsePython = 0;
end

d = dir(fullfile(tmpFolder, fname));


if ~isempty(rawWaveformFolder) && reExtract == 0

    rawWaveformsFull = readNPY(fullfile(rawFolder, 'templates._jf_rawWaveforms.npy'));
    rawWaveformsPeakChan = readNPY(fullfile(rawFolder, 'templates._jf_rawWaveformPeakChannels.npy'));
    if param.saveMultipleRaw
        spikeMap = readNPY(fullfile(rawFolder, 'templates._jf_Multi_rawWaveforms.npy'));

    end
else

    %% Intitialize

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

    %% Iterate over spike clusters and find all the data associated with them
    rawWaveforms = struct;
    allSpikeTimes = spikeTimes_samples;
    disp('Extracting raw waveforms ...')

    if nChannels == 385
        spikeMap = nan(nClust,nChannels-1, spikeWidth, nSpikesToExtract,'single');
        ch2take = 1:nChannels-1;
    else
        spikeMap = nan(nClust,nChannels, spikeWidth, nSpikesToExtract,'single');
        ch2take = 1:nChannels;
    end
    % find n spike times for all units
    spikeIndices = arrayfun(@(X)  allSpikeTimes(spikeTemplates == clustInds(X)),1:nClust,'UniformOutput',0); % %Find spike samples per unit
    spikeIndices = cellfun(@(X) sort(datasample(X,nSpikesToExtract)),spikeIndices,'UniformOutput',0); % find nSpikesToExtract of these % with replacement on purpose for nspikes<100. Will be fixed later
    spikeIndices = cat(2,spikeIndices{:});

    % array

    if UsePython
        allsamples = arrayfun(@(X) spikeIndices(X)-halfWidth:spikeIndices(X)+halfWidth,1:length(spikeIndices(:)),'UniformOutput',0);
        allsamples = cat(1,allsamples{:})';
        batchsize = 2*round(str2num(SR))-spikeWidth+1; %batches of 1second used for compression, probably optimal to extract 1 second at a time
        batchn = ceil((max(allsamples(:))-min(allsamples(:)))./batchsize);
        allsamplestmp = min(allsamples(:)):max(allsamples(:)+batchsize);
        batchidx = arrayfun(@(X) allsamplestmp(batchsize*(X-1)+1):allsamplestmp(batchsize*X),1:batchn,'UniformOutput',0);
        %             batchidx = find(cell2mat(cellfun(@(X) any(ismember(X,allsamples)),batchidx,'UniformOutput',0)));
        timethis = tic;
        for bid = 1:length(batchidx)
            % Find which units have a spike in this batch
            [spkidx,unitidx] = find(ismember(spikeIndices,batchidx{bid}));
            if ~any(spkidx)
                continue
            end
            try
                endidx = batchidx{bid}(end)+spikeWidth;
                if endidx>length(allsamplestmp) || allsamplestmp(endidx)>n_samples
                    endidx=min([length(allsamplestmp) find(allsamplestmp==n_samples)]);
                end

                % Extract piece of data using ephys_reader python
                % integration
                tmpdata = pyrunfile("Ephys_Reader_FromMatlab.py","chunk",...
                    datapath = strrep(fullfile(spikeFile.folder,fname),'\','/'),start_time=batchidx{bid}(1),end_time=endidx); %0-indexed!!
                tmpdata=uint16(tmpdata);

                % Loop over clusters and spikes and put them in the
                % correct position in the matrix

                for spkid = 1:length(spkidx)
                    % Put it back in the correct order
                    tmpid = find(ismember(batchidx{bid},spikeIndices(spkidx(spkid),unitidx(spkid))));
                    if tmpid<=halfWidth || tmpid+halfWidth>size(tmpdata,1)
                        continue
                    end
                    spikeMap(unitidx(spkid),:,:,spkidx(spkid))= tmpdata(tmpid-halfWidth:tmpid+halfWidth,ch2take)';
                end

            catch ME
                disp(ME)
                disp('Make sure to use MATLAB>2022a and compatible python version, in an environment that has the modules phylib, pathlib, and matlab installed')
                disp('e.g. pyversion("C:\Users\EnnyB\anaconda3\envs\phy\pythonw.exe")')
                disp('Also make sure you input the path in a python-compatible way!')
            end
            if (mod(bid, 100) == 0 || bid == length(batchidx)) && verbose
                disp(['Extracted ', num2str(round(bid./length(batchidx).*1000)./10), '% Elapsed time ' num2str(round(toc(timethis)./60.*100)./100) ' minutes']);
                %figure; imagesc(spkMapMean_sm)
                %title(['Unit ID: ', num2str(i)]);
                %colorbar;
            end
        end
    else
       
        fid = fopen(fullfile(spikeFile.folder, fname), 'r');
    
        for iCluster = 1:nClust
            spikeIndicestmp = unique(spikeIndices(:,iCluster)); %  Get rid of duplicate spikes
            for iSpike = 1:length(spikeIndicestmp)
                thisSpikeIdx = spikeIndicestmp(iSpike);
                if thisSpikeIdx > halfWidth && (thisSpikeIdx + halfWidth) * dataTypeNBytes < d.bytes % check that it's not out of bounds

                    byteIdx = int64(((thisSpikeIdx - halfWidth) * nChannels) * dataTypeNBytes); % int64 to prevent overflow on crappy windows machines that are incredibly inferior to linux
                    fseek(fid, byteIdx, 'bof'); % from beginning of file
                    data = fread(fid, [nChannels, spikeWidth], 'int16=>int16'); % read individual waveform from binary file
                    frewind(fid);
                    if size(data, 2) == spikeWidth && nChannels == 385
                        spikeMap(iCluster,:, :, iSpike) = data(1:nChannels-1, :, :); %remove sync channel
                    elseif size(data, 2) == spikeWidth
                        spikeMap(iCluster,:, :, iSpike) = data(1:nChannels, :, :);
                    end
                end
            end
            if (mod(iCluster, 20) == 0 || iCluster == nClust) && verbose
                fprintf(['\n   Extracted ', num2str(iCluster), '/', num2str(nClust), ' raw waveforms.']);
                %figure; imagesc(spkMapMean_sm)
                %title(['Unit ID: ', num2str(i)]);
                %colorbar;
            end
           
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

        end
        fclose(fid);

        % Individual spikes are noisy; already gaussian smooth on this -
        % otherwise we're going to end up with max. noise averages
        % level across time
        spikeMap = smoothdata(spikeMap,3,'gaussian',5);
        % Subtract first 10 samples to level spikes
        spikeMap = permute(spikeMap,[1,2,4,3]);
        spikeMap = permute(spikeMap - mean(spikeMap(:,:,:,1:10),4),[1,2,4,3]);
        % Smooth across channels
        spikeMap = smoothdata(spikeMap,2,'gaussian',5);


        % Now average across 100 spikes
        rawWaveformsFull = nanmean(spikeMap, 4);

        for iCluster = 1:nClust
            %                spkMapMean_sm = smoothdata(squeeze(rawWaveformsFull(iCluster, :,:)), 1, 'gaussian', 5);
            % Find direction of spike 
           
            spkMapMean_sm = nanmean(squeeze(rawWaveformsFull(iCluster,:,:)),1);
            if isfield(param,'rawWaveformMaxDef') && strcmp (param.rawWaveformMaxDef, 'firstSTD' )
                % Find 'peak' (which is the first max deflection from 0,
                % because there can be undershoot which is larger but this is
                % not the true max channel!
                slopeidx = find(abs(spkMapMean_sm)>nanmean(spkMapMean_sm(1:10))+nanstd(spkMapMean_sm),1,'first');
                slopesign = sign(spkMapMean_sm(slopeidx));
                [~, rawWaveformsPeakChan(iCluster,:)] = max(max(squeeze(rawWaveformsFull(iCluster,:,:)).*slopesign, [], 2), [], 1);%QQ buggy sometimes % 
            else

              [~, rawWaveformsPeakChan(iCluster,:)] = max(max(abs(squeeze(rawWaveformsFull(iCluster,:,:))), [], 2), [], 1);%QQ buggy sometimes % 
            end
            
        end


     

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



        %     end

        rawWaveformFolder = dir(fullfile(rawFolder,'templates._jf_rawWaveforms.npy'));
        if isempty(rawWaveformFolder) || reExtract
            %save(fullfile(spikeFile.folder, 'rawWaveforms.mat'), 'rawWaveforms', '-v7.3');
            writeNPY(rawWaveformsFull, fullfile(rawFolder, 'templates._jf_rawWaveforms.npy'))
            writeNPY(rawWaveformsPeakChan, fullfile(rawFolder, 'templates._jf_rawWaveformPeakChannels.npy'))

            if param.saveMultipleRaw
                writeNPY(spikeMap, fullfile(rawFolder, 'templates._jf_Multi_rawWaveforms.npy'))
            end
        end
    end
end