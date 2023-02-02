if exist('paramBC','var') %for unit match 
    param = paramBC;
end

spikeFile=dir(param.rawFile);

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
