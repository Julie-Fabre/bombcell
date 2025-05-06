function paramBC = qualityParamValuesForUnitMatch(ephysMetaDir, rawFile, ephysKilosortPath, gain_to_uV, kilosortVersion)
% defaults 
paramBC = bc.qm.qualityParamValues(ephysMetaDir, rawFile, ephysKilosortPath, gain_to_uV, kilosortVersion);

% unit match specific
paramBC.detrendWaveform = 0; % If this is set to 1, each raw extracted spike is
    % detrended (we remove the best straight-fit line from the spike)
    % using MATLAB's builtin function detrend. 
paramBC.nRawSpikesToExtract = 1000;%inf; %inf if you don't encounter memory issues and want to load all spikes; % how many raw spikes to extract for each unit 
paramBC.saveMultipleRaw = 1; % If you wish to save the nRawSpikesToExtract as well, 
% currently needed if you want to run unit match https://github.com/EnnyvanBeest/UnitMatch
% to track chronic cells over days after this
paramBC.decompressData = 1; % whether to decompress .cbin ephys data 

