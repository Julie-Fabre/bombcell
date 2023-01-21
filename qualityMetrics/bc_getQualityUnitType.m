function unitType = bc_getQualityUnitType(param, qMetric)
% JF, Classify units into good/mua/noise/non-somatic 
% ------
% Inputs
% ------
% 
% ------
% Outputs
% ------
if param.computeDistanceMetrics && ~isnan(param.isoDmin)
    unitType = nan(length(qMetric.percentageSpikesMissing), 1);
    
    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | qMetric.isSomatic ~= param.somatic ...
        | qMetric.spatialDecaySlope <=  param.minSpatialDecaySlope | qMetric.waveformDuration < param.minWvDuration |...
        qMetric.waveformDuration > param.maxWvDuration | qMetric.waveformBaselineFlatness >= param.maxWvBaselineFraction) = 0; % NOISE or NON-SOMATIC
     unitType(any(qMetric.percentageSpikesMissing <= param.maxPercSpikesMissing, 2) & qMetric.nSpikes > param.minNumSpikes & ...
        any(qMetric.fractionRPVs <= param.maxRPVviolations, 2) & ...
        qMetric.rawAmplitude > param.minAmplitude & qMetric.isoDmin >= param.isoDmin & isnan(unitType)) = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)) = 2; % MULTI UNIT

else
    unitType = nan(length(qMetric.percentageSpikesMissing), 1);
    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | qMetric.isSomatic ~= param.somatic ...
        | qMetric.spatialDecaySlope <=  param.minSpatialDecaySlope | qMetric.waveformDuration < param.minWvDuration |...
        qMetric.waveformDuration > param.maxWvDuration | qMetric.waveformBaselineFlatness >= param.maxWvBaselineFraction) = 0; % NOISE or NON-SOMATIC
    unitType(any(qMetric.percentageSpikesMissing <= param.maxPercSpikesMissing, 2) & qMetric.nSpikes > param.minNumSpikes & ...
        any(qMetric.fractionRPVs <= param.maxRPVviolations, 2) & ...
        qMetric.rawAmplitude > param.minAmplitude & isnan(unitType)) = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)') = 2; % MULTI UNIT

end
