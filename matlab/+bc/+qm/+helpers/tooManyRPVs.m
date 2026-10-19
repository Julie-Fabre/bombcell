function tooMany = tooManyRPVs(qMetric, param)
% TOOMANYRPVS Units failing the refractory period criterion. Units for which
% it cannot be said whether they are clean or contaminated (RPV_status == 0,
% see bc.qm.slidingRPstatus and bc.qm.pointEstimateRPstatus) are counted as
% passing when param.keepUncertainRPVUnits is true, and as failing otherwise.
% same shape as the metrics, so it can be combined with the other criteria
tooMany = qMetric.fractionRPVs_estimatedTauR > param.maxRPVviolations;
if hasField(qMetric, 'RPV_status')
    uncertain = reshape(qMetric.RPV_status, size(tooMany)) == 0;
    if isfield(param, 'keepUncertainRPVUnits') && param.keepUncertainRPVUnits
        tooMany = tooMany & ~uncertain;
    else
        tooMany = tooMany | uncertain;
    end
end
end

function tf = hasField(s, name)
if istable(s)
    tf = ismember(name, s.Properties.VariableNames);
else
    tf = isfield(s, name);
end
end
