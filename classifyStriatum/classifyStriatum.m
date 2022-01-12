% get good units 
goodUnits = qMetric.numSpikes >= param.minNumSpikes & qMetric.waveformRawAmpli >= param.minAmpli & ...
    qMetric.fractionRPVchunk <= param.maxRPV ...
    & qMetric.somatic == param.somaCluster & qMetric.pMissing < param.maxPercMissing;

%classify into 4 groups
msn = ephysParams.postSpikeSuppression < 40 & ephysParams.templateDuration > param.cellTypeDuration;
fsi = ephysParams.prop_long_isi < 0.1 & ...
    ephysParams.postSpikeSuppression < param.cellTypePostS & ...
    ephysParams.templateDuration < param.cellTypeDuration;
tan = ephysParams.postSpikeSuppression >= 40 & ephysParams.templateDuration > param.cellTypeDuration;
uin = ephysParams.prop_long_isi > 0.1 & ...
    ephysParams.postSpikeSuppression < param.cellTypePostS & ...
    ephysParams.templateDuration < param.cellTypeDuration;

cellTypesClassif = struct;
cellTypesClassif(1).cells = msn;
cellTypesClassif(2).cells = fsi;
cellTypesClassif(3).cells = tan;
cellTypesClassif(4).cells = uin;
