function status = pointEstimateRPstatus(theseSpikeTimes, timeChunks, contamination, tauR, param)
% JF, three-way outcome of a point estimate of contamination ('hill' or
% 'llobet') against the contamination threshold param.maxRPVviolations (as
% point_estimate_rp_status in the python version).
%
% By default (param.maxRPVestimateCV is NaN or missing) the point estimate
% alone decides: 1 (pass) if contamination <= maxRPVviolations, -1 (fail)
% otherwise. Units with few spikes have few expected violations, so their
% estimate is noisy: a unit with no violations has an estimate of 0 whether
% or not it is clean. Following Vincent & Economo (2024, eNeuro), who found
% that single-unit contamination estimates are only reliable once their
% coefficient of variation (CV) is below ~20%, which requires tens of
% minutes of recording at > 5 Hz, setting param.maxRPVestimateCV (e.g. 0.2)
% flags such units:
%
% The CV of the estimate is computed for a unit at the threshold C* =
% maxRPVviolations, from the expected number of violations with N = infinity
% contaminating neurons (Vincent & Economo, Eq. 8, equivalent to Llobet et
% al., 2022):
%   lambda = K * C* * (2 - C*), K = sum over chunks of (tauR - tauC) * n^2 / D
% where n is the number of spikes and D the effective duration of each chunk.
% The violation count is Poisson, so by the delta method
%   CV = sqrt(lambda) / (C* * dlambda/dC) = sqrt(lambda) / (2 * K * C* * (1 - C*))
% If CV <= maxRPVestimateCV, the point estimate decides as above. Otherwise
% the observed violations are tested against lambda as in slidingRPstatus:
%    1 (pass): contamination is below C* with param.confidenceThreshold
%       confidence.
%   -1 (fail): contamination is above C* with that confidence.
%    0 (uncertain): neither, there are too few spikes to tell.
% ------
% Inputs
% ------
% theseSpikeTimes: the unit's spike times in seconds
% timeChunks: time chunk edges to use
% contamination: the unit's point estimate of contamination
% tauR: the refractory period (s) at which contamination was estimated
% param: structure with fields maxRPVviolations, maxRPVestimateCV, tauC and
%   confidenceThreshold
% ------
% Outputs
% ------
% status: 1, 0 or -1 as above, NaN if contamination is NaN

if isnan(contamination)
    status = NaN;
    return;
end
maxContamination = param.maxRPVviolations;
status = 1 - 2 * (contamination > maxContamination);
maxCV = NaN;
if isfield(param, 'maxRPVestimateCV') && ~isempty(param.maxRPVestimateCV)
    maxCV = param.maxRPVestimateCV;
end
if isnan(maxCV) || maxContamination <= 0 || maxContamination >= 1
    return;
end
tauC = param.tauC;
confidenceThreshold = 0.9;
if isfield(param, 'confidenceThreshold')
    confidenceThreshold = param.confidenceThreshold;
end

nViolations = 0;
K = 0;
for iChunk = 1:numel(timeChunks) - 1
    chunkSpikeTimes = theseSpikeTimes(theseSpikeTimes >= timeChunks(iChunk) & ...
        theseSpikeTimes < timeChunks(iChunk+1));
    [tauR_test, chunkViolations, effectiveDuration] = bc.qm.helpers.slidingRPViolationCounts( ...
        chunkSpikeTimes, timeChunks(iChunk+1) - timeChunks(iChunk), tauR, tauC, 0);
    if isempty(tauR_test)
        continue;
    end
    nViolations = nViolations + chunkViolations;
    K = K + (tauR - tauC) * numel(chunkSpikeTimes)^2 / effectiveDuration;
end
if K == 0
    return;
end

expectedViolations = K * maxContamination * (2 - maxContamination);
cv = sqrt(expectedViolations) / (2 * K * maxContamination * (1 - maxContamination));
if cv <= maxCV
    return;
end

% confidence that contamination < max: P(more violations than observed | C = max)
if 1 - poisscdf(nViolations, expectedViolations) >= confidenceThreshold
    status = 1;
% confidence that contamination > max: 1 - P(at least as many violations | C = max)
elseif poisscdf(nViolations - 1, expectedViolations) >= confidenceThreshold
    status = -1;
else
    status = 0;
end
end
