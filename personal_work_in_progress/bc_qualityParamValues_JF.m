function param = bc_qualityParamValues_JF(ephysMetaDir, rawFile, ephysKilosortPath, gain_to_uV)

param = bc_qualityParamValues(ephysMetaDir, rawFile, ephysKilosortPath, gain_to_uV);
param.removeDuplicateSpikes = 0; 

end