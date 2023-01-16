function bc_subsetRawData(rawBinFile, nChannels, endCut)
% JF,
% ------
% Inputs
% ------

% ------
% Outputs
% ------
% rawBinFile = '/home/netshare/zinu/JF070/2022-06-12/ephys/site1/experiment1/recording1/continuous/Neuropix-3a-100.0/continuous.dat';
% nChannels = 384;
dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));
dirRaw = dir(rawBinFile);
fileSize = dirRaw.bytes/384 - 100;
fid = fopen(rawBinFile, 'r');


byteIdx = int64((1 * nChannels)*dataTypeNBytes); % int64 to prevent overflow on crappy windows machines that are incredibly inferior to linux



chunkSize = 1000000;

fid = []; fidOut = [];

d = dir(rawBinFile);
%endCut = 100;
%begCut = 100;
nSampsTotal = d.bytes/nChannels/2 - endCut;

nChunksTotal = ceil(nSampsTotal/chunkSize);
endChunk = nSampsTotal - (chunkSize)*(nChunksTotal-1);
try
  
  [pathstr, name, ext] = fileparts(rawBinFile);
  fid = fopen(rawBinFile, 'r');
  if nargin < 3
    outputFilename  = [pathstr filesep name '_tRange' ext];
  else
    outputFilename  = [outputDir filesep name '_tRange' ext];
  end
  fidOut = fopen(outputFilename, 'w');
  
  % theseInds = 0;
  chunkInd = 1;
  tTrace = zeros(1, nSampsTotal);
  while 1
    
    fprintf(1, 'chunk %d/%d\n', chunkInd, nChunksTotal);
    if chunkInd == nChunksTotal 
        dat = fread(fid, [nChannels endChunk], '*int16');
    else
        dat = fread(fid, [nChannels chunkSize], '*int16');
    end
    
    
    
    if ~isempty(dat)
      
      %         theseInds = theseInds(end):theseInds(end)+chunkSize-1;
      
      fwrite(fidOut, dat, 'int16');
          
    else
      break
    end
    
    chunkInd = chunkInd+1;
  end
  
  fclose(fid);
  fclose(fidOut);
  
catch me
  
  if ~isempty(fid)
    fclose(fid);
  end
  
  if ~isempty(fidOut)
    fclose(fidOut);
  end
  
  
  rethrow(me)
  
end
