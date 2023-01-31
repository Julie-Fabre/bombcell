function [rawFile] = bc_manageDataCompression(ephysRawDir, decompressDataLocal)

if strcmp(ephysRawDir.name(end-4:end), '.cbin') &&...
        isempty(dir([decompressDataLocal, filesep, ephysRawDir.name(1:end-14), '_bc_decompressed*.bin']))
    fprintf('Decompressing ephys data file %s locally to %s... \n', ephysRawDir.name, decompressDataLocal)
    
    decompDataFile = bc_extractCbinData([ephysRawDir.folder, filesep, ephysRawDir.name],...
        [], [], [], decompressDataLocal);
    rawFile = decompDataFile;
elseif strcmp(ephysRawDir.name(end-4:end), '.cbin') &&...
        ~isempty(dir([decompressDataLocal, filesep, ephysRawDir.name(1:end-14), '_bc_decompressed*.bin']))
    fprintf('Using previously decompressed ephys data file in %s ... \n', decompressDataLocal)
    decompressDataLocalFile = dir([decompressDataLocal, filesep, ephysRawDir.name(1:end-14), '_bc_decompressed*.bin']);
    
    rawFile = [decompressDataLocalFile.folder, filesep, decompressDataLocalFile.name];
else
    rawFile = [ephysRawDir.folder, filesep, ephysRawDir.name];
end
end