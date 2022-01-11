function scalingFactor = bc_readSpikeGLXMetaFile(metaFileDir)

filetext = fileread(fullfile(metaFileDir.folder, metaFileDir.name));
expr = 'imDatPrb_type=*';
[~,endIndex] =  regexp(filetext,expr);
if isempty(endIndex)
    expr = 'imProbeOpt=*';
    [~,endIndex] =  regexp(filetext,expr);
end
probeType = filetext(endIndex+1);

if strcmp(probeType ,'1') || strcmp(probeType ,'3')
    Vrange = 1.2e6; % from -0.6 to 0.6
    bits_encoding = 10; % 10-bit analog to digital 
    gain = 500; % fixed gain
elseif strcmp(probeType ,'2')
    Vrange = 1e6; % from -0.5 to 0.5
    bits_encoding = 14; % 14-bit analog to digital 
    gain = 80; % fixed gain
end
scalingFactor = Vrange / (2 ^ bits_encoding) / gain; 
end