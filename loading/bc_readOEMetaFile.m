function scalingFactor = bc_readOEMetaFile(metaFile)
% JF
% read open ephys meta file and extract scaling factor (bit_volts) value 
% ------
% Inputs
% ------
% metaFile: string, full path to meta file (should be a structure.oebin file)
% ------
% Outputs
% ------
% scaling factor: double, scaling factor value to convert raw data to
% microvolts
%
filetext = fileread(metaFile);
expr = '"bit_volts": '; % this is actually the scaling factor
[~,endIndex] =  regexp(filetext,expr);

scalingFactor = str2num(filetext(endIndex(1)+1 : endIndex(1)+8)) .* 2.34; %2.34 uv/bit for a gain setting of 500 - gain not saved in open ephys metafile anymore

end