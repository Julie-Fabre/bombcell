
%% bc_checkIfCompressedBin
function bc_checkAndUncompressBinData(apDataFile, localSaveFolder)
apDataInfo = dir(apDataFile);
if any(strfind(apDataFile, 'cbin'))
    if ~exist(fullfile(localSaveFolder, strrep(apDataFile, 'cbin', 'bin')), 'file')
        disp('Raw ephys data is onlt present in compressed format. Uncompressing locally...')
        % Decompression
        success = pyrunfile("bc_uncompressBin.py", "succes", datapath = apDataFile, ...
            JsonPath = strrep(apDataFile, 'cbin', 'ch'), savepath = strrep(fullfile(localSaveFolder, apDataInfo.name), 'cbin', 'bin'));
        ephysap_path = fullfile(localSaveFolder, strrep(apDataInfo.name, 'cbin', 'bin'));
    else
        ephysap_path = fullfile(localSaveFolder, strrep(apDataInfo.name, 'cbin', 'bin'));
    end
end
end