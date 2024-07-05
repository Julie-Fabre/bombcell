function scalingFactor = readOEMetaFile(metaFile)
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
% the uV/bit scaling factor (2.34 uV/bit for the standard AP gain setting
% of 500; 4.68 uV/bit for the standard LFP gain setting of 250) is already
% applied by open ephys, and you just need to multiply by 0.195:
% - OE docs ref: https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/166789121/Flat+binary+format
% - sanity check: AP raw numbers are quantized by 12, which *0.195 = 2.34. (Same for the LFP, quantized by 24).
%

%filetext = fileread(metaFile);
%expr = '"bit_volts": '; % this is actually the scaling factor
%[~, endIndex] = regexp(filetext, expr);

%scalingFactor = str2num(filetext(endIndex(1)+1:endIndex(1)+8));

scalingFactor = 0.1949999928474426; %hard-coded for now because this a property of how the data is saved, and not of probe type ect. (e.g. see https://groups.google.com/g/open-ephys/c/9CKgVPoEF7M?pli=1)
end
