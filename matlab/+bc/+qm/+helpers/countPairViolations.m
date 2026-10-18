function nViolations = countPairViolations(spikeTimes, tauC, tauR)
% COUNTPAIRVIOLATIONS Count all spike pairs (not only consecutive spikes)
% whose time difference dt satisfies tauC <= dt < tauR. Pairs with dt == 0
% are never counted. Runs in O(N log N). Matches _count_pair_violations
% (include_tauR=False) in the python version.

st = sort(double(spikeTimes(:)));
N = numel(st);
if N < 2
    nViolations = 0;
    return;
end
idx = (1:N)';
if tauC > 0
    lo = countBelow(st, st + tauC, false); % spikes with time < t + tauC
else
    lo = countBelow(st, st, true); % spikes with time <= t (excludes dt == 0)
end
hi = countBelow(st, st + tauR, false); % spikes with time < t + tauR
lo = max(lo, idx); % never count a spike against itself or earlier spikes
nViolations = sum(max(hi - lo, 0));
end

function counts = countBelow(sortedValues, queries, inclusive)
% number of sortedValues < query (or <= query if inclusive), for each query.
% Relies on sort being stable: whichever array comes first wins ties.
nQ = numel(queries);
if inclusive
    allValues = [sortedValues; queries];
    isQuery = [false(numel(sortedValues), 1); true(nQ, 1)];
else
    allValues = [queries; sortedValues];
    isQuery = [true(nQ, 1); false(numel(sortedValues), 1)];
end
[~, order] = sort(allValues);
isQuerySorted = isQuery(order);
nValuesBefore = cumsum(~isQuerySorted);
queryIdx = order(isQuerySorted);
if ~inclusive
    % queries were placed first, so their indices are 1..nQ
    counts = zeros(nQ, 1);
    counts(queryIdx) = nValuesBefore(isQuerySorted);
else
    counts = zeros(nQ, 1);
    counts(queryIdx - numel(sortedValues)) = nValuesBefore(isQuerySorted);
end
end
