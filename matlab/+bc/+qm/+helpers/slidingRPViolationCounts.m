function [tauR_test, nViolations, effectiveDuration] = slidingRPViolationCounts(spikeTimes, duration, tauR_values, tauC, tauR_reject)
% SLIDINGRPVIOLATIONCOUNTS Observed violations at each tested refractory
% period, as used by the sliding refractory period method.
% ------
% Outputs
% ------
% tauR_test: the refractory periods tested (those > max(tauC, tauR_reject))
% nViolations: number of spike pairs with tauC <= dt < tauR, for each tauR in tauR_test
% effectiveDuration: duration minus the censored period around each spike
% All outputs are empty if the metric cannot be computed (too few spikes, no
% tauR to test, or a censored period longer than the recording).
tauR_test = [];
nViolations = [];
nSpikes = numel(spikeTimes);
effectiveDuration = duration - 2 * nSpikes * tauC;
if nSpikes <= 1 || effectiveDuration <= 0
    effectiveDuration = [];
    return;
end
tauR_values = tauR_values(:)';
tauR_test = tauR_values(tauR_values > max(tauC, tauR_reject));
nViolations = zeros(size(tauR_test));
for iTauR = 1:numel(tauR_test)
    nViolations(iTauR) = bc.qm.helpers.countPairViolations(spikeTimes, tauC, tauR_test(iTauR));
end
if isempty(tauR_test)
    effectiveDuration = [];
end
end
