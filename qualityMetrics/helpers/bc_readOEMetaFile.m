function scalingFactor = bc_readOEMetaFile(metaFileDir)

filetext = fileread(fullfile(metaFileDir.folder, metaFileDir.name));
expr = '"bit_volts": '; % this is actually the scaling factor
[~,endIndex] =  regexp(filetext,expr);

scalingFactor = str2num(filetext(endIndex(1)+1 : endIndex(1)+8)) .* 2.34; %2.34 uv/bit for a gain setting of 500 - gain not saved in open ephys metafile anymore ??

end