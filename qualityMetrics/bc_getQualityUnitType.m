function unitType = bc_getQualityUnitType(param, qMetric, savePath)
% JF, Classify units into good/mua/noise/non-somatic
% ------
% Inputs
% ------
%
% ------
% Outputs
% ------

% check paramaters 
param = bc_checkParameterFields(param);

if nargin < 3 && param.unitType_for_phy == 1
    savePath = pwd;
    warning('no save path specified. using current working directory')
end

if isstruct(qMetric) % if saving failed, qMetric is a structure and the fractionRPVs_estimatedTauR field we need below is not computed yet
    if ~isfield('fractionRPVs_estimatedTauR', qMetric)
        qMetric.fractionRPVs_estimatedTauR = arrayfun(@(x) qMetric.fractionRPVs(x, qMetric.RPV_tauR_estimate(x)), 1:size(qMetric.fractionRPVs, 1));
        qMetric = rmfield(qMetric, 'fractionRPVs');
    end
end


if param.computeDistanceMetrics && ~isnan(param.isoDmin) && param.computeDrift && param.extractRaw
    unitType = nan(length(qMetric.percentageSpikesMissing_gaussian), 1);

    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | ...
        qMetric.spatialDecaySlope >= param.minSpatialDecaySlope | qMetric.waveformDuration_peakTrough < param.minWvDuration | ...
        qMetric.waveformDuration_peakTrough > param.maxWvDuration | qMetric.waveformBaselineFlatness >= param.maxWvBaselineFraction) = 0; % NOISE
    unitType(qMetric.isSomatic ~= param.somatic & isnan(unitType)) = 3; % NON-SOMATIC
    unitType(qMetric.percentageSpikesMissing_gaussian <= param.maxPercSpikesMissing & qMetric.nSpikes > param.minNumSpikes & ...
        qMetric.fractionRPVs_estimatedTauR <= param.maxRPVviolations & ...
        qMetric.rawAmplitude > param.minAmplitude & qMetric.signalToNoiseRatio >= param.minSNR & ...
        qMetric.presenceRatio >= param.minPresenceRatio & qMetric.maxDriftEstimate <= param.maxDrift & ...
        qMetric.rawAmplitude > param.minAmplitude & qMetric.isoD >= param.isoDmin & ...
        qMetric.Lratio <= param.lratioMax & isnan(unitType)) = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)) = 2; % MULTI UNIT

elseif param.computeDistanceMetrics && ~isnan(param.isoDmin) && param.computeDrift
    unitType = nan(length(qMetric.percentageSpikesMissing_gaussian), 1);

    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | ...
        qMetric.spatialDecaySlope >= param.minSpatialDecaySlope | qMetric.waveformDuration_peakTrough < param.minWvDuration | ...
        qMetric.waveformDuration_peakTrough > param.maxWvDuration | qMetric.waveformBaselineFlatness >= param.maxWvBaselineFraction) = 0; % NOISE
    unitType(qMetric.isSomatic ~= param.somatic & isnan(unitType)) = 3; % NON-SOMATIC
    unitType(qMetric.percentageSpikesMissing_gaussian' <= param.maxPercSpikesMissing & qMetric.nSpikes > param.minNumSpikes & ...
        qMetric.fractionRPVs_estimatedTauR <= param.maxRPVviolations & ...
        qMetric.rawAmplitude > param.minAmplitude & ...
        qMetric.presenceRatio >= param.minPresenceRatio & qMetric.maxDriftEstimate <= param.maxDrift & ...
        qMetric.isoD >= param.isoDmin & ...
        qMetric.Lratio <= param.lratioMax & isnan(unitType)) = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)) = 2; % MULTI UNIT

elseif param.computeDrift && param.extractRaw

    unitType = nan(length(qMetric.percentageSpikesMissing_gaussian), 1);

    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | ...
        qMetric.spatialDecaySlope >= param.minSpatialDecaySlope | qMetric.waveformDuration_peakTrough < param.minWvDuration | ...
        qMetric.waveformDuration_peakTrough > param.maxWvDuration | qMetric.waveformBaselineFlatness >= param.maxWvBaselineFraction) = 0; % NOISE
    unitType(qMetric.isSomatic ~= param.somatic & isnan(unitType)) = 3; % NON-SOMATIC
    unitType(qMetric.percentageSpikesMissing_gaussian <= param.maxPercSpikesMissing & qMetric.nSpikes > param.minNumSpikes & ...
        qMetric.fractionRPVs_estimatedTauR <= param.maxRPVviolations & ...
        qMetric.rawAmplitude > param.minAmplitude & qMetric.signalToNoiseRatio >= param.minSNR & ...
        qMetric.presenceRatio >= param.minPresenceRatio & qMetric.maxDriftEstimate <= param.maxDrift & ...
        qMetric.rawAmplitude > param.minAmplitude & isnan(unitType)) = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)) = 2; % MULTI UNIT

elseif param.computeDistanceMetrics && ~isnan(param.isoDmin) && param.extractRaw
    unitType = nan(length(qMetric.percentageSpikesMissing_gaussian), 1);

    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | ...
        qMetric.spatialDecaySlope >= param.minSpatialDecaySlope | qMetric.waveformDuration_peakTrough < param.minWvDuration | ...
        qMetric.waveformDuration_peakTrough > param.maxWvDuration | qMetric.waveformBaselineFlatness >= param.maxWvBaselineFraction) = 0; % NOISE
    unitType(qMetric.isSomatic ~= param.somatic & isnan(unitType)) = 3; % NON-SOMATIC
    unitType(qMetric.percentageSpikesMissing_gaussian <= param.maxPercSpikesMissing & qMetric.nSpikes > param.minNumSpikes & ...
        qMetric.fractionRPVs_estimatedTauR <= param.maxRPVviolations & ...
        qMetric.rawAmplitude > param.minAmplitude & qMetric.signalToNoiseRatio >= param.minSNR & ...
        qMetric.presenceRatio >= param.minPresenceRatio & isnan(unitType) & ...
        qMetric.rawAmplitude > param.minAmplitude & qMetric.isoD >= param.isoDmin & ...
        qMetric.Lratio <= param.lratioMax & isnan(unitType)) = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)) = 2; % MULTI UNIT

elseif param.computeDrift
    unitType = nan(length(qMetric.percentageSpikesMissing_gaussian), 1);

    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | ...
        qMetric.spatialDecaySlope >= param.minSpatialDecaySlope | qMetric.waveformDuration_peakTrough < param.minWvDuration | ...
        qMetric.waveformDuration_peakTrough > param.maxWvDuration | qMetric.waveformBaselineFlatness >= param.maxWvBaselineFraction) = 0; % NOISE
    unitType(qMetric.isSomatic ~= param.somatic & isnan(unitType)) = 3; % NON-SOMATIC
    unitType(qMetric.percentageSpikesMissing_gaussian' <= param.maxPercSpikesMissing & qMetric.nSpikes > param.minNumSpikes & ...
        qMetric.fractionRPVs_estimatedTauR <= param.maxRPVviolations & ...
        qMetric.rawAmplitude > param.minAmplitude & ...
        qMetric.presenceRatio >= param.minPresenceRatio & qMetric.maxDriftEstimate <= param.maxDrift & ...
        isnan(unitType)) = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)) = 2; % MULTI UNIT

elseif param.computeDistanceMetrics && ~isnan(param.isoDmin)
    unitType = nan(length(qMetric.percentageSpikesMissing_gaussian), 1);

    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | ...
        qMetric.spatialDecaySlope >= param.minSpatialDecaySlope | qMetric.waveformDuration_peakTrough < param.minWvDuration | ...
        qMetric.waveformDuration_peakTrough > param.maxWvDuration | qMetric.waveformBaselineFlatness >= param.maxWvBaselineFraction) = 0; % NOISE
    unitType(qMetric.isSomatic ~= param.somatic & isnan(unitType)) = 3; % NON-SOMATIC
    unitType(qMetric.percentageSpikesMissing_gaussian <= param.maxPercSpikesMissing & qMetric.nSpikes > param.minNumSpikes & ...
        qMetric.fractionRPVs_estimatedTauR <= param.maxRPVviolations & ...
        qMetric.rawAmplitude > param.minAmplitude & ...
        qMetric.presenceRatio >= param.minPresenceRatio & ...
        qMetric.isoD >= param.isoDmin & ...
        qMetric.Lratio <= param.lratioMax & isnan(unitType)) = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)) = 2; % MULTI UNIT

elseif param.extractRaw
    unitType = nan(length(qMetric.percentageSpikesMissing_gaussian), 1);
    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | ...
        qMetric.spatialDecaySlope >= param.minSpatialDecaySlope | qMetric.waveformDuration_peakTrough < param.minWvDuration | ...
        qMetric.waveformDuration_peakTrough > param.maxWvDuration | qMetric.waveformBaselineFlatness >= param.maxWvBaselineFraction) = 0; % NOISE
    unitType(qMetric.isSomatic ~= param.somatic & isnan(unitType)) = 3; % NON-SOMATIC
    unitType(qMetric.percentageSpikesMissing_gaussian <= param.maxPercSpikesMissing & qMetric.nSpikes > param.minNumSpikes & ...
        qMetric.fractionRPVs_estimatedTauR <= param.maxRPVviolations & ...
        qMetric.rawAmplitude > param.minAmplitude & qMetric.signalToNoiseRatio >= param.minSNR & ...
        qMetric.presenceRatio >= param.minPresenceRatio & isnan(unitType)) = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)) = 2; % MULTI UNIT

else
    unitType = nan(length(qMetric.percentageSpikesMissing_gaussian), 1);
    unitType(qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | ...
        qMetric.spatialDecaySlope >= param.minSpatialDecaySlope | qMetric.waveformDuration_peakTrough < param.minWvDuration | ...
        qMetric.waveformDuration_peakTrough > param.maxWvDuration | qMetric.waveformBaselineFlatness >= param.maxWvBaselineFraction) = 0; % NOISE
    unitType(qMetric.isSomatic ~= param.somatic & isnan(unitType)) = 3; % NON-SOMATIC
    unitType(qMetric.percentageSpikesMissing_gaussian <= param.maxPercSpikesMissing & qMetric.nSpikes > param.minNumSpikes & ...
        qMetric.fractionRPVs_estimatedTauR <= param.maxRPVviolations & ...
        qMetric.presenceRatio >= param.minPresenceRatio & isnan(unitType)) = 1; % SINGLE SEXY UNIT
    unitType(isnan(unitType)) = 2; % MULTI UNIT

end

unitType_string= cell(size(unitType,1),1);
unitType_string(unitType == 0) = {'NOISE'};
unitType_string(unitType == 3) = {'NON-SOMA'};
unitType_string(unitType == 1) = {'GOOD'};
unitType_string(unitType == 2) = {'MUA'};


% save unitTypefor phy if param.unitType_for_phy is equal to 1
if isfield(param, 'unitType_for_phy') %  ensure back-compatibility if users have a previous version of param
    if param.unitType_for_phy == 1
        if isfield(param, 'saveAsTSV') % ensure back-compatibility if users have a previous version of param
            if param.saveAsTSV == 1
                cluster_id_vector = qMetric.clusterID - 1; % from bombcell to phy nomenclature
                if isfield(param, 'ephysKilosortPath')
                    saveTSV_path = param.ephysKilosortPath;
                else
                    saveTSV_path = savePath;
                end
                
                cluster_table = table(cluster_id_vector, unitType_string, 'VariableNames', {'cluster_id', 'bc_unitType'});
                writetable(cluster_table, [saveTSV_path, filesep, 'cluster_bc_unitType.tsv'], 'FileType', 'text', 'Delimiter', '\t');
            end
        end
    end

end