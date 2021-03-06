function cellType = bc_classifyStriatalCells(ephysProp, qMetric, param)
    cellType = struct; 
    % get good units 
goodUnits = qMetric.percSpikesMissing <= param.maxPercSpikesMissing & qMetric.nSpikes > param.minNumSpikes & ...
        qMetric.nPeaks <= param.maxNPeaks & qMetric.nTroughs <= param.maxNTroughs & qMetric.Fp <= param.maxRPVviolations & ...
        qMetric.axonal == param.axonal & qMetric.rawAmplitude > param.minAmplitude; 


%classify into 4 groups
cellType.msn = ephysProp.pss < param.pss & ephysProp.templateDuration > param.templateDuration & goodUnits;
cellType.fsi = ephysProp.propLongISI < param.propISI & ...
    ephysProp.pss < param.pss & ...
    ephysProp.templateDuration < param.templateDuration & goodUnits;
cellType.tan = ephysProp.pss >= param.pss & ephysProp.templateDuration > param.templateDuration & goodUnits;
cellType.uin = ephysProp.propLongISI >  param.propISI & ...
    ephysProp.pss < param.pss & ...
    ephysProp.templateDuration < param.templateDuration & goodUnits;
end