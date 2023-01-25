function decompDataFile = bc_extractCbinData(fileName, sStartEnd, chIdx, doParfor, saveFileFolder)
% adapted from a script by Micheal Krumin
%
% requires the zmat package:
% https://github.com/fangq/zmat

% fileName - full path to the .cbin filename you want to read from
% sStartEnd - a 1x2 array of [sampleStart, sampleEnd] you want to read
% chIdx - channel indices
% doParfor - a flag whether to use a parfor or a for loop inside the function
%           depends on specific usage scenario. In same cases it is better
%           to use parfor inside, sometimes outside of the function.
% saveFileFolder - where to save your data 
% saveAsMtx - if true, save .npy matrix. if false, save in binary format
% dataOut - nSamples x nChannels array of decompressed data
% example usage:
% bc_extractCbinData('/home/netshare/zinu/XG006/2022-06-30/ephys/site1/2022-06_30_xg006_g0_t0.imec0.ap.cbin', [], [], [], 'home/ExtraHD/', 0)

% JF: added sanity checks, more options including parfor,
% save output as matrix 
% added size, byte, method info (zmat version used previously
% that handled this no longer exists)
% save data chunk by chunk (otherwise matlab can crash if the
% files are too big (> 70GB), default is non parfor because not compatible
% with this 



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

% sanitize/check inputs 
if nargin < 2 || isempty(sStartEnd)
    sStartEnd = [cbinMeta.chunk_bounds(1), cbinMeta.chunk_bounds(end)];
end

if nargin < 3 || isempty(chIdx)
    chIdx = 1:cbinMeta.n_channels;
elseif chIdx(end) >  cbinMeta.n_channels
    warning(sprintf('max channel index invalid, changing from %s to %s',num2str(chIdx(end)), num2str(cbinMeta.n_channels)))
    startEnd(1) = cbinMeta.chunk_bounds(1);
end

if nargin < 4 || isempty(doParfor)
    doParfor = false;
end

if sStartEnd(1) < cbinMeta.chunk_bounds(1) 
    warning(sprintf('samples to read outside of file range, changing start sample from %s to %s',num2str(sStartEnd(1)), num2str(cbinMeta.chunk_bounds(1))))
    startEnd(1) = cbinMeta.chunk_bounds(1);
end

if sStartEnd(1) < cbinMeta.chunk_bounds(1) 
   warning(sprintf('samples to read outside of file range, changing end sample from %s to %s',num2str(sStartEnd(2)), num2str(cbinMeta.chunk_bounds(end))))
   startEnd(2) = cbinMeta.chunk_bounds(end);
end

sampleStart = sStartEnd(1);
sampleEnd = sStartEnd(2);


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
% nSamples we will actually need from each chunk
% nSamplesToExtract = iSampleEnd - iSampleStart + 1;
% these are start and end indices for extracted samples in the output data array
% startIdx = cumsum([1; nSamplesToExtract(1:end-1)]);
% endIdx = cumsum(nSamplesToExtract);

% nSamplesOut = sampleEnd - sampleStart + 1;
% nChannelsOut = numel(chIdx);
% data = zeros(nSamplesOut, nChannelsOut, cbinMeta.dtype);

nChunks = iChunkEnd - iChunkStart + 1;
data = cell(nChunks, 1);
nChannels = cbinMeta.n_channels;
nSamples = cbinMeta.chunk_bounds([1:nChunks] + iChunkStart) - cbinMeta.chunk_bounds([1:nChunks] + iChunkStart - 1);
chunkSizeBytes = cbinMeta.chunk_offsets([1:nChunks] + iChunkStart) - cbinMeta.chunk_offsets([1:nChunks] + iChunkStart - 1);
offset = cbinMeta.chunk_offsets([1:nChunks] + iChunkStart - 1);

fN = dir(fileName);
decompDataFile = [saveFileFolder, filesep, fN.name(1:end-14), '_bc_decompressed', fN.name(end-13:end-8),'.ap.bin'];
fidOut = fopen(decompDataFile,'w');

if doParfor
    parfor iChunk = 1:nChunks
        %     chunkInd = iChunk + iChunkStart - 1;
        % size of expected decompressed data for that chunk
        zmatLocalInfo = zmatInfo;
        zmatLocalInfo.size = [nSamples(iChunk)*nChannels, 1];
        
        % read a chunk from the compressed data
        fid = fopen(fileName, 'r');
        fseek(fid, offset(iChunk), 'bof');
        compData = fread(fid, chunkSizeBytes(iChunk), '*uint8');
        fclose(fid);
        
        decompData = zmat(compData, zmatLocalInfo);
        decompData = reshape(decompData, nSamples(iChunk), nChannels);
        chunkData = cumsum(decompData(:, chIdx), 1);
        %     data(startIdx(iChunk):endIdx(iChunk), :) = chunkData(iSampleStart(iChunk):iSampleEnd(iChunk), :);
        data{iChunk} = chunkData(iSampleStart(iChunk):iSampleEnd(iChunk), :);
    end
    dataOut = cell2mat(data);
    fwrite(fidOut, dataOut, 'int16');

else
    
    for iChunk = 1:nChunks

        zmatLocalInfo = zmatInfo;
        zmatLocalInfo.size = [nSamples(iChunk)*nChannels, 1];

        
        % read a chunk from the compressed data
        fid = fopen(fileName, 'r');
        fseek(fid, offset(iChunk), 'bof');
        compData = fread(fid, chunkSizeBytes(iChunk), '*uint8');
        fclose(fid);
        decompData = zmat(compData, zmatLocalInfo);
        decompData = reshape(decompData, nSamples(iChunk), nChannels);
        chunkData = cumsum(decompData(:, chIdx), 1);
        data = chunkData(iSampleStart(iChunk):iSampleEnd(iChunk), :);
        reshaped_data = reshape(permute(data, [2,1]), [nSamples(iChunk)*nChannels, 1]);
        %plot(reshaped_data(end-30000:end))
        %plot(reshaped_data(385:385:end))
        fwrite(fidOut, reshaped_data, 'int16');


    end
%     dataOut = cell2mat(data);
%     fwrite(fidOut, dataOut, 'int16');
    % save in particular locations. open file size 
%     for iChannel = 1:nChannels % can't decompress channel-wise 
%         
%         zmatLocalInfo = zmatInfo;
%         zmatLocalInfo.size = [nSamples(iChunk)*nChannels, 1];
% 
%         
%         % read a chunk from the compressed data
%         fid = fopen(fileName, 'r');
%         fseek(fid, offset(iChunk), 'bof');
%         compData = fread(fid, chunkSizeBytes(iChunk), '*uint8');
%         fclose(fid);
%         decompData = zmat(compData, zmatLocalInfo);
%         decompData = reshape(decompData, nSamples(iChunk), nChannels);
%         chunkData = cumsum(decompData(:, chIdx), 1);
%         data{iChunk} = chunkData(iSampleStart(iChunk):iSampleEnd(iChunk), :);
%         fwrite(fidOut, dataOut, 'int16');
% 
% 
%     end
    
end

fclose(fidOut);
%fidOut_read = fopen(decompDataFile, 'r');
