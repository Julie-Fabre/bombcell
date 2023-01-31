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
    unitType = nan(length(qMetric.percentageSpikesMissing_gaussian), 1);
    
    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs |  ...
        qMetric.spatialDecaySlope >=  param.minSpatialDecaySlope | qMetric.waveformDuration_peakTrough < param.minWvDuration |...
        qMetric.waveformDuration_peakTrough > param.maxWvDuration | qMetric.waveformBaselineFlatness >= param.maxWvBaselineFraction) = 0; % NOISE 
    unitType(qMetric.isSomatic' ~= param.somatic & isnan(unitType)) = 3; % NON-SOMATIC 
    unitType(qMetric.percentageSpikesMissing_gaussian' <= param.maxPercSpikesMissing & qMetric.nSpikes' > param.minNumSpikes & ...
        qMetric.fractionRPVs_estimatedTauR' <= param.maxRPVviolations & ...
        qMetric.rawAmplitude' > param.minAmplitude & qMetric.isoD' >= param.isoDmin &...
        qMetric.Lratio' <= param.lratioMax & isnan(unitType)) = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)) = 2; % MULTI UNIT

else
    unitType = nan(length(qMetric.percentageSpikesMissing_gaussian), 1);
    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs |  ...
        qMetric.spatialDecaySlope >=  param.minSpatialDecaySlope | qMetric.waveformDuration_peakTrough < param.minWvDuration |...
        qMetric.waveformDuration_peakTrough > param.maxWvDuration | qMetric.waveformBaselineFlatness >= param.maxWvBaselineFraction) = 0; % NOISE
    unitType(qMetric.isSomatic ~= param.somatic & isnan(unitType)) = 3; % NON-SOMATIC 
    unitType(qMetric.percentageSpikesMissing_gaussian <= param.maxPercSpikesMissing & qMetric.nSpikes > param.minNumSpikes & ...
        qMetric.fractionRPVs_estimatedTauR <= param.maxRPVviolations & ...
        qMetric.rawAmplitude > param.minAmplitude & isnan(unitType)) = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)) = 2; % MULTI UNIT

end
