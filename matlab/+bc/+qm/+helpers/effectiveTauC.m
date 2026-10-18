function tauC = effectiveTauC(param)
% EFFECTIVETAUC Censored period used for refractory period violations:
% param.tauC, raised to bombcell's duplicate spike window if duplicate
% spikes are removed, since pairs closer than that window can no longer occur.
tauC = param.tauC;
if isfield(param, 'removeDuplicateSpikes') && param.removeDuplicateSpikes && ...
        isfield(param, 'duplicateSpikeWindow_s')
    tauC = max(tauC, param.duplicateSpikeWindow_s);
end
end
