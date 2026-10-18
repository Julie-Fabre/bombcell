function status = slidingRPstatus(theseSpikeTimes, timeChunks, tauR_values, param)
% JF, three-way outcome of the sliding refractory period test against the
% contamination threshold param.maxRPVviolations (as sliding_rp_status in
% the python version).
%
% The sliding refractory period method (Roth et al.) only confirms that a
% unit is clean: a unit with too few spikes cannot reach the confidence
% threshold for any contamination level, and is therefore failed even
% though there is no evidence that it is contaminated. This function tells
% such units apart:
%    1 (pass): contamination is below maxRPVviolations with at least
%       param.confidenceThreshold confidence, at some tested refractory period.
%    0 (uncertain): the unit does not pass, but there is a tested refractory
%       period at which it is not confidently contaminated (contamination
%       above maxRPVviolations with that confidence) and has too few spikes
%       to pass even with no violations at all. The unit may be clean with
%       that refractory period, and there is not enough data to tell.
%   -1 (fail): every other unit that does not pass. At each tested
%       refractory period it is either confidently contaminated, or had
%       enough spikes to pass there and did not. Units with many spikes that
%       sit just above the threshold therefore fail rather than being uncertain.
%
% Violations and expected violations are summed across time chunks (a sum
% of Poisson counts is Poisson), so all the unit's spikes contribute to one
% test.
% ------
% Inputs
% ------
% theseSpikeTimes: the unit's spike times in seconds
% timeChunks: time chunk edges to use
% tauR_values: refractory period values to test (s)
% param: structure with fields maxRPVviolations, tauC, confidenceThreshold
%   and tauR_reject
% ------
% Outputs
% ------
% status: 1, 0 or -1 as above, NaN if the metric could not be computed

maxContamination = param.maxRPVviolations;
tauC = param.tauC;
confidenceThreshold = 0.9;
if isfield(param, 'confidenceThreshold')
    confidenceThreshold = param.confidenceThreshold;
end
tauR_reject = 0.0005;
if isfield(param, 'tauR_reject')
    tauR_reject = param.tauR_reject;
end

nViolations = 0;
expectedViolations = 0;
anyChunk = false;
for iChunk = 1:numel(timeChunks) - 1
    chunkSpikeTimes = theseSpikeTimes(theseSpikeTimes >= timeChunks(iChunk) & ...
        theseSpikeTimes < timeChunks(iChunk+1));
    [tauR_test, chunkViolations, effectiveDuration] = bc.qm.helpers.slidingRPViolationCounts( ...
        chunkSpikeTimes, timeChunks(iChunk+1) - timeChunks(iChunk), tauR_values, tauC, tauR_reject);
    if isempty(tauR_test)
        continue;
    end
    nViolations = nViolations + chunkViolations;
    expectedViolations = expectedViolations + bc.qm.helpers.slidingRPExpectedViolations( ...
        numel(chunkSpikeTimes), effectiveDuration, tauR_test, tauC, maxContamination);
    anyChunk = true;
end
if ~anyChunk
    status = NaN;
    return;
end

% confidence that contamination < max: P(more violations than observed | C = max)
confidenceClean = 1 - poisscdf(nViolations, expectedViolations);
% confidence that contamination > max: 1 - P(at least as many violations | C = max)
confidenceContaminated = poisscdf(nViolations - 1, expectedViolations);

if any(confidenceClean >= confidenceThreshold)
    status = 1;
    return;
end
% with no violations, confidenceClean = 1 - exp(-expectedViolations)
couldPass = expectedViolations >= -log(1 - confidenceThreshold);
contaminated = confidenceContaminated >= confidenceThreshold;
if any(~contaminated & ~couldPass)
    status = 0;
else
    status = -1;
end
end
