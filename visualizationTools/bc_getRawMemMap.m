
if iscell(param.rawFolder)
    param.rawFolder = fileparts(param.rawFolder{1});
elseif sum(param.rawFolder(end-2:end) == '/..') == 3
    param.rawFolder = fileparts(param.rawFolder(1:end-3));
end
spikeFile = dir(fullfile(param.rawFolder, '*.ap.bin'));
if isempty(spikeFile)
    spikeFile = dir(fullfile(param.rawFolder, '/*.dat')); %openEphys format
end
spikeFile=spikeFile(1);

fname = spikeFile.name;

dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));
try %hacky way of figuring out if sync channel present or not
    n_samples = spikeFile.bytes / (param.nChannels * dataTypeNBytes);
    ap_data = memmapfile(fullfile(spikeFile.folder, fname), 'Format', {'int16', [param.nChannels, n_samples], 'data'});
catch
    nChannels = param.nChannels - 1;
    n_samples = spikeFile.bytes / (nChannels * dataTypeNBytes);
    ap_data = memmapfile(fullfile(spikeFile.folder, fname), 'Format', {'int16', [nChannels, n_samples], 'data'});
end
memMapData = ap_data.Data.data;
