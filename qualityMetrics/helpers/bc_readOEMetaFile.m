function scalingFactor = bc_readOEMetaFile(metaFileDir)

filetext = fileread(fullfile(metaFileDir.folder, metaFileDir.name));
expr = '"bit_volts": ';
[~,endIndex] =  regexp(filetext,expr);

scalingFactor = filetext(endIndex(1)+1 : endIndex(1)+8);

end