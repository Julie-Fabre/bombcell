function tooMany = tooManyRPVs(qMetric, param)
% TOOMANYRPVS Units failing the refractory period criterion. With the
% sliding refractory period method, units for which it cannot be said
% whether they are clean or contaminated (RPV_status == 0, see
% bc.qm.slidingRPstatus) are counted as passing when
% param.keepUncertainRPVUnits is true.
% same shape as the metrics, so it can be combined with the other criteria
tooMany = qMetric.fractionRPVs_estimatedTauR > param.maxRPVviolations;
if isfield(param, 'keepUncertainRPVUnits') && param.keepUncertainRPVUnits && ...
        hasField(qMetric, 'RPV_status')
    tooMany = tooMany & ~(reshape(qMetric.RPV_status, size(tooMany)) == 0);
end
end

function tf = hasField(s, name)
if istable(s)
    tf = ismember(name, s.Properties.VariableNames);
else
    tf = isfield(s, name);
end
end
