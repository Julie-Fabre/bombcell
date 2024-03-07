function decompDataFile = bc_extractCbinData(fileName, sStartEnd, allChannelIndices, doParfor, saveFileFolder, onlySaveSyncChannel, verbose)
%
% requires the zmat package:
% https://github.com/fangq/zmat
% ------
% Inputs
% ------
% fileName - full path to the .cbin filename you want to read from
% sStartEnd - a 1x2 array of [sampleStart, sampleEnd] you want to read
% allChannelIndices - all channel indices present in recoding. This should
%   be a vector from 1 to the max number of channels (e.g. 1:385)
% doParfor - a flag whether to use a parfor or a for loop inside the function
%           depends on specific usage scenario. In same cases it is better
%           to use parfor inside, sometimes outside of the function. Note:
%           using parfor regularly crashed on my computer, so I have
%           disabled by default. 
% saveFileFolder - where to save your data 
% onlySaveSyncChannel - if true, only save the sync channel. This speeds up
%   the function
% verbose - if true, dispply information about progress
% ------
% Outputs
% ------
% decompDataFile - nSamples x nChannels array of decompressed data
%
% adapted from a script by Micheal Krumin
% JF: added sanity checks, more options including parfor,
% save output as matrix 
% added size, byte, method info (zmat version used previously
% that handled this no longer exists)
% save data chunk by chunk (otherwise matlab can crash if the
% files are too big (> 70GB), default is non parfor because not compatible
% with this 
% 20230322 added option to only save sync channel (hardcoded as channel 385
% for now) 

%% sanitize/check inputs 
if nargin < 1
    %     for testing
    fileName = '/home/netshare/zinu/JF070/2022-06-18/ephys/site1_shank0/2022-06_18_JF070_shank1-1_g0_t0.imec0.ap.cbin';
end
% Assuming the ch file has the same basename and is in the same folder as cbin
chName = [fileName(1:end-4), 'ch'];

% reading .ch json file - contains info about compression
fid = fopen(chName, 'r');
data = fread(fid, 'uint8=>char');
fclose(fid);
cbinMeta = jsondecode(data');


if nargin < 2 || isempty(sStartEnd)
    sStartEnd = [cbinMeta.chunk_bounds(1), cbinMeta.chunk_bounds(end)];
end

if nargin < 3 || isempty(allChannelIndices)
    allChannelIndices = 1:cbinMeta.n_channels;
elseif allChannelIndices(end) >  cbinMeta.n_channels
    warning(sprintf('max channel index invalid, changing from %s to %s',num2str(allChannelIndices(end)), num2str(cbinMeta.n_channels)))
    startEnd(1) = cbinMeta.chunk_bounds(1);
end

if nargin < 4 || isempty(doParfor)
    doParfor = false;
end

if nargin < 6
    onlySaveSyncChannel = false;
end

if nargin < 7
    verbose = true;
end

if sStartEnd(1) < cbinMeta.chunk_bounds(1) 
    warning(sprintf('samples to read outside of file range, changing start sample from %s to %s',num2str(sStartEnd(1)), num2str(cbinMeta.chunk_bounds(1))))
    startEnd(1) = cbinMeta.chunk_bounds(1);
end

if sStartEnd(2) > cbinMeta.chunk_bounds(end) 
   warning(sprintf('samples to read outside of file range, changing end sample from %s to %s',num2str(sStartEnd(2)), num2str(cbinMeta.chunk_bounds(end))))
   startEnd(2) = cbinMeta.chunk_bounds(end);
end

sampleStart = sStartEnd(1);
sampleEnd = sStartEnd(2);


%% decompress and save data 

% build zmat info struct
zmatInfo = struct;
zmatInfo.type = cbinMeta.dtype;
tmp = cast(1, cbinMeta.dtype);
zmatInfo.byte = whos('tmp').bytes; % figuring out bytesPerSample programmatically
zmatInfo.method = cbinMeta.algorithm;
zmatInfo.status = 1;
zmatInfo.level = cbinMeta.comp_level;

% figuring out which chunks to read
iChunkStart = find(sampleStart >= cbinMeta.chunk_bounds, 1, 'last');
iChunkEnd = find(sampleEnd <= cbinMeta.chunk_bounds, 1, 'first') - 1;

% nSamples in the compressed data chunks
nSamplesPerChunk = diff(cbinMeta.chunk_bounds(iChunkStart:iChunkEnd+1));
iSampleStart = max(sampleStart - cbinMeta.chunk_bounds(iChunkStart:iChunkEnd), 1);
iSampleEnd = min(sampleEnd - cbinMeta.chunk_bounds(iChunkStart:iChunkEnd), nSamplesPerChunk);

nChunks = iChunkEnd - iChunkStart + 1;

nChannels = cbinMeta.n_channels;
nSamples = cbinMeta.chunk_bounds([1:nChunks] + iChunkStart) - cbinMeta.chunk_bounds([1:nChunks] + iChunkStart - 1);
chunkSizeBytes = cbinMeta.chunk_offsets([1:nChunks] + iChunkStart) - cbinMeta.chunk_offsets([1:nChunks] + iChunkStart - 1);
offset = cbinMeta.chunk_offsets([1:nChunks] + iChunkStart - 1);

% get file names and prepare for saving 
fN = dir(fileName);
if isfolder(saveFileFolder)
    if onlySaveSyncChannel
        decompDataFile = [saveFileFolder, filesep, fN.name(1:end-14), '_bc_decompressed_sync_channel', fN.name(end-13:end-8),'.mat'];
        decompDataFile2 = [saveFileFolder, filesep, 'sync.mat'];
    
    else
        decompDataFile = [saveFileFolder, filesep, fN.name(1:end-14), '_bc_decompressed', fN.name(end-13:end-8),'.ap.bin'];
    end
else
    decompDataFile = saveFileFolder;
end

if ~onlySaveSyncChannel
    fidOut = fopen(decompDataFile,'w');
end

if verbose
    fprintf('\n decompressing data from %s to %s', fileName, decompDataFile)
end

% main loop
if doParfor
    data = cell(nChunks, 1);
    parfor iChunk = 1:nChunks
        % size of expected decompressed data for that chunk
        zmatLocalInfo = zmatInfo;
        zmatLocalInfo.size = [nSamples(iChunk)*nChannels, 1];
        
        % read a chunk from the compressed data
        fid = fopen(fileName, 'r');
        fseek(fid, offset(iChunk), 'bof');
        compData = fread(fid, chunkSizeBytes(iChunk), '*uint8');
        fclose(fid);
        
        % store decompressed data 
        decompData = zmat(compData, zmatLocalInfo);
        decompData = reshape(decompData, nSamples(iChunk), nChannels);
        chunkData = cumsum(decompData(:, allChannelIndices), 1);
        data{iChunk} = chunkData(iSampleStart(iChunk):iSampleEnd(iChunk), :);
    end
    dataOut = cell2mat(data);
    
    % save data 
    if onlySaveSyncChannel
        error('saving only sync channel not yet implemented in parfor')
    else
        fwrite(fidOut, dataOut, 'int16');
    end

else
    syncdata = [];
    for iChunk = 1:nChunks
        % size of expected decompressed data for that chunk
        zmatLocalInfo = zmatInfo;
        zmatLocalInfo.size = [nSamples(iChunk)*nChannels, 1];

        % read a chunk from the compressed data
        fid = fopen(fileName, 'r');
        fseek(fid, offset(iChunk), 'bof');
        
        % reformat and store decompressed data 
        compData = fread(fid, chunkSizeBytes(iChunk), '*uint8');
        fclose(fid);
        decompData = zmat(compData, zmatLocalInfo);
        decompData = reshape(decompData, nSamples(iChunk), nChannels);
        chunkData = cumsum(decompData(:, allChannelIndices), 1);
        data = chunkData(iSampleStart(iChunk):iSampleEnd(iChunk), :);
        reshaped_data = reshape(permute(data, [2,1]), [nSamples(iChunk)*nChannels, 1]);
        
        % save data 
        if ~onlySaveSyncChannel
            fwrite(fidOut, reshaped_data, 'int16');
        else
            syncdata = [syncdata; reshaped_data(385:385:end)];
        end
        
        % verbose: progress fraction
        if ((mod(iChunk, 500) == 0) || iChunk == nChunks || iChunk == 1) && verbose
            disp(['     ', num2str(iChunk), '/' num2str(nChunks)])
        end
    end
    if onlySaveSyncChannel
        save(decompDataFile, 'syncdata')
        save(decompDataFile2, 'syncdata')
    end

end
if ~onlySaveSyncChannel
    fclose(fidOut);
end

