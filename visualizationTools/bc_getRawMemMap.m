function memMapData = bc_getRawMemMap(rawFolder, nChannels)
if iscell(rawFolder)
    rawFolder = fileparts(rawFolder{1});
elseif sum(rawFolder(end-2:end) == '/..') == 3
    rawFolder = fileparts(rawFolder(1:end-3));
end
spikeFile = dir(fullfile(rawFolder, '*.ap.bin'));
if isempty(spikeFile)
    spikeFile = dir(fullfile(rawFolder, '/*.dat')); %openEphys format
end


fname = spikeFile.name;

dataTypeNBytes = numel(typecast(cast(0, 'uint16'), 'uint8'));
try %hacky way of figuring out if sync channel present or not
    n_samples = spikeFile.bytes / (nChannels * dataTypeNBytes);
    ap_data = memmapfile(fullfile(spikeFile.folder, fname), 'Format', {'int16', [nChannels, n_samples], 'data'});
catch
    nChannels = nChannels - 1;
    n_samples = spikeFile.bytes / (nChannels * dataTypeNBytes);
    ap_data = memmapfile(fullfile(spikeFile.folder, fname), 'Format', {'int16', [nChannels, n_samples], 'data'});
end
memMapData = ap_data.Data.data;
end