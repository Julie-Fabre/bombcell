function [unitType, unitType_string] = bc_getQualityUnitType(param, qMetric, savePath)
% JF, Classify units into good/mua/noise/non-somatic
% ------
% Inputs
% ------
% param: struct containing paramaters for classification
% qMetric: struct containing fields of size nUnits x 1
% ------
% Outputs
% ------
% unitType - nUnits x 1 double array indicating the type of each unit:
%   unitType==0 defines all noise units
%   unitType==1 defines all good units
%   unitType==2 defines all MUA units
%   unitType==3 defines all non-somatic units. If
%       param.splitGoodAndMua_NonSomatic is true, it specifically defines good
%       non-somatic units
%   unitType==4 defines all MUA non-somatic units if param.splitGoodAndMua_NonSomatic is true
% unitType_string - nUnits x 1 string array indicating the type of each unit (good, mua, noise, non-somatic).


%% Sanitize and check inputs 
% Sanitize parameter input
param = bc_checkParameterFields(param);

% Check whether to save this classification for automated loading by phy
if (nargin < 3 || isempty(savePath)) && param.unitType_for_phy == 1
    savePath = pwd;
    warning('no save path specified. using current working directory')
end

% Process qMetric if it's a structure and required field is not computed
if isstruct(qMetric) % if saving failed, qMetric is a structure and the fractionRPVs_estimatedTauR field we need below is not computed yet
    if ~isfield(qMetric, 'fractionRPVs_estimatedTauR')
        qMetric.fractionRPVs_estimatedTauR = arrayfun(@(x) qMetric.fractionRPVs(x, qMetric.RPV_tauR_estimate(x)), 1:size(qMetric.fractionRPVs, 1));
        qMetric = rmfield(qMetric, 'fractionRPVs');
    end
end

% Initialize unitType array
unitType = nan(length(qMetric.percentageSpikesMissing_gaussian), 1);

%% Classify units
% Classify noise units
unitType(isnan(qMetric.nPeaks) | qMetric.nPeaks > param.maxNPeaks | qMetric.nTroughs > param.maxNTroughs | ...
    qMetric.spatialDecaySlope > param.minSpatialDecaySlope | qMetric.waveformDuration_peakTrough < param.minWvDuration | ...
    qMetric.waveformDuration_peakTrough > param.maxWvDuration | qMetric.waveformBaselineFlatness > param.maxWvBaselineFraction) = 0; % NOISE

% Classify mua units
unitType((qMetric.percentageSpikesMissing_gaussian > param.maxPercSpikesMissing | qMetric.nSpikes < param.minNumSpikes | ...
    qMetric.fractionRPVs_estimatedTauR > param.maxRPVviolations | qMetric.presenceRatio < param.minPresenceRatio) & ...
    isnan(unitType)) = 2; % MUA

if param.computeDistanceMetrics && ~isnan(param.isoDmin)
    unitType((qMetric.isoD < param.isoDmin | qMetric.Lratio > param.lratioMax) & ...
        isnan(unitType)) = 2; % MUA
end

if param.extractRaw
  unitType((qMetric.rawAmplitude < param.minAmplitude | qMetric.signalToNoiseRatio < param.minSNR) & ...
      isnan(unitType)) = 2; % MUA
end

if param.computeDrift
     unitType(qMetric.maxDriftEstimate > param.maxDrift & isnan(unitType)) = 2; % MUA
end

% Classify good units
unitType(isnan(unitType)) = 1; % SINGLE SEXY UNIT

% Classify non-somatic units
if param.splitGoodAndMua_NonSomatic
    unitType(qMetric.isSomatic ~= param.somatic & unitType == 1) = 3; % GOOD NON-SOMATIC
    unitType(qMetric.isSomatic ~= param.somatic & unitType == 2) = 4; % MUA NON-SOMATIC
else
    unitType(qMetric.isSomatic ~= param.somatic & ismember(unitType,[1 2])) = 3; % NON-SOMATIC
end

% Get unit type string
unitType_string = cell(size(unitType, 1), 1);
unitType_string(unitType == 0) = {'NOISE'};
unitType_string(unitType == 1) = {'GOOD'};
unitType_string(unitType == 2) = {'MUA'};
if param.splitGoodAndMua_NonSomatic
    unitType_string(unitType == 3) = {'NON-SOMA GOOD'};
    unitType_string(unitType == 4) = {'NON-SOMA MUA'};
else
    unitType_string(unitType == 3) = {'NON-SOMA'};
end

%% Save classification for phy
% save unitType for phy if param.unitType_for_phy is equal to 1
try
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
catch
    warning('unable to save tsv file of unitTypes')
end
end