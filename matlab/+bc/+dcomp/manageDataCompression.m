function [rawFile] = manageDataCompression(ephysRawDir, decompressDataLocal)
% JF, Decompress raw ephys data if it is in .cbin format
% ------
% Inputs
% ------
% ephysRawDir: dir() structure of the path to your raw .bin, .cbin or .dat file 
% decompressDataLocal: character array defining the path where decompressed
%   data will be saved 
% ------
% Outputs
% ------
% rawFile: character array defining the path where your uncompressed raw
%   ephys data is
% 
if isstruct(ephysRawDir)
    if strcmp(ephysRawDir.name(end-4:end), '.cbin') &&...
            isempty(dir([decompressDataLocal, filesep, ephysRawDir.name(1:end-14), '_bc_decompressed*.bin']))
        fprintf('Decompressing ephys data file %s locally to %s... \n', ephysRawDir.name, decompressDataLocal)
        
        decompDataFile = bc.dcomp.extractCbinData([ephysRawDir.folder, filesep, ephysRawDir.name],...
            [], [], [], decompressDataLocal, 0);
        rawFile = decompDataFile;
    elseif strcmp(ephysRawDir.name(end-4:end), '.cbin') &&...
            ~isempty(dir([decompressDataLocal, filesep, ephysRawDir.name(1:end-14), '_bc_decompressed*.bin']))
        fprintf('Using previously decompressed ephys data file in %s ... \n', decompressDataLocal)
        decompressDataLocalFile = dir([decompressDataLocal, filesep, ephysRawDir.name(1:end-14), '_bc_decompressed*.bin']);
        
        rawFile = [decompressDataLocalFile.folder, filesep, decompressDataLocalFile.name];
    else
        rawFile = [ephysRawDir.folder, filesep, ephysRawDir.name];
    end

else
    rawFile = 'NaN';
end
end