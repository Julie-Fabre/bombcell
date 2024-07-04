
if iscell(param.rawFolder)
    param.rawFolder = fileparts(param.rawFolder{1});
elseif sum(param.rawFolder(end-2:end) == '/..') == 3
    param.rawFolder = fileparts(param.rawFolder(1:end-3));
end
spikeFile = dir(fullfile(param.rawFolder, '*.ap.*bin'));
if isempty(spikeFile)
    spikeFile = dir(fullfile(param.rawFolder, '/*.dat')); %openEphys format
end
spikeFile=spikeFile(1);
dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));
% Read original bytes
meta = ReadMeta2(spikeFile.folder);
n_samples = round(str2num(meta.fileSizeBytes)/dataTypeNBytes/param.nChannels);

%Create temporary file
Data = table;
Data = tall(Data);
fname = spikeFile.name;
batchsize = 50000;
batchn = ceil(n_samples./batchsize);
tic
for chid = param.nChannels
    memMapDatatmp = zeros(batchsize,batchn,'int16'); % create tall array to keep data in memory
    parfor bid = 1:batchn
        disp(['running channel ' num2str(chid) ' batch ' num2str(bid) '/' num2str(batchn)])

        stidx = (batchsize*(bid-1));
        endidx = (batchsize*bid);
        if endidx>n_samples
            endidx=n_samples;
        end
        try
            tmpdata = pyrunfile("Ephys_Reader_FromMatlab.py","chunk",...
                datapath = strrep(fullfile(spikeFile.folder,fname),'\','/'),start_time=stidx,end_time=endidx,channel=chid-1); %0-indexed!!
        catch ME
            disp(ME)
            disp('Make sure to use MATLAB>2022a and compatible python version, in an environment that has the modules phylib, pathlib, and matlab installed')
            disp('e.g. pyversion("C:\Users\EnnyB\anaconda3\envs\phy\pythonw.exe")')
            disp('Also make suer you input the path in a python-compatible way!')
        end
        %to get rid of the python stuff, put the data in a matlab vector:
        tmpdata2=zeros(1,(endidx-stidx),'uint16');
        tmpdata2(1,1:end)=tmpdata;
       
    end
    memMapDatatmp = memMapDatatmp(:);
    eval(['Data.Ch' num2str(chid) '= memMapDatatmp'])

end
toc

% figure
% subplot(2,1,1)
% plot(tmpdata)
% title('Using python ephys reader after compression')
% subplot(2,1,2)
% plot(syncDat(stidx+1:endidx))
% title('Before compression')
 %% old version
if 0
    try %hacky way of figuring out if sync channel present or not
        n_samples = spikeFile.bytes / (param.nChannels * dataTypeNBytes);
        ap_data = memmapfile(fullfile(spikeFile.folder, fname), 'Format', {'uint16', fix([param.nChannels, n_samples]), 'data'});
    catch
        nChannels = param.nChannels - 1;
        n_samples = spikeFile.bytes / (nChannels * dataTypeNBytes);
        ap_data = memmapfile(fullfile(spikeFile.folder, fname), 'Format', {'uint16', fix([nChannels, n_samples]), 'data'});
    end
    memMapData = ap_data.Data.data;
end