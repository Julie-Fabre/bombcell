function [isolationDist, Lratio, silhouetteScore, histogram_mahalUnit_counts, histogram_mahalUnit_edges, histogram_mahalNoise_counts, histogram_mahalNoise_edges] = getDistanceMetrics_2(pc_features, pc_feature_ind, thisUnit, numberSpikes, spikesIdx, allSpikesIdx, nChansToUse, plotThis)

nPCs = size(pc_features, 2);
theseChannels = pc_feature_ind(thisUnit, 1:nChansToUse);
theseFeatures = reshape(pc_features(spikesIdx, :, 1:nChansToUse), numberSpikes, []);

uniqueIDs = unique(allSpikesIdx(allSpikesIdx>0));
uniqueIDs = uniqueIDs(uniqueIDs ~= thisUnit);  % Exclude thisUnit

% Preallocate arrays
mahalanobisDistances = nan(numel(uniqueIDs), 1);
otherUnits_double = nan(numel(uniqueIDs), 1);

% Vectorized operation to get all other features at once
otherFeatures = pc_features(~spikesIdx, :, 1:nChansToUse);
otherFeatures = reshape(otherFeatures, [], nPCs * nChansToUse);

% Calculate Mahalanobis distances for all other units at once
validFeatures = all(~isnan(otherFeatures), 2);
if sum(validFeatures) > size(theseFeatures, 2) && size(theseFeatures, 1) > size(theseFeatures, 2)
    mahalD = mahal(otherFeatures(validFeatures, :), theseFeatures);
else
    mahalD = nan(size(otherFeatures, 1), 1);
end

% Calculate L-ratio
if numberSpikes > 0
    L = sum(1 - chi2cdf(mahalD, nPCs * nChansToUse));
    Lratio = L / numberSpikes;
else
    Lratio = NaN;
end

% Calculate isolation distance if applicable
if sum(validFeatures) > numberSpikes && numberSpikes > nChansToUse * nPCs
    sortedMahalD = sort(mahalD);
    isolationDist = sortedMahalD(min(numberSpikes, length(sortedMahalD)));
else
    isolationDist = NaN;
end

% Initialize other outputs
silhouetteScore = NaN;
histogram_mahalUnit_counts = NaN;
histogram_mahalUnit_edges = NaN;
histogram_mahalNoise_counts = NaN;
histogram_mahalNoise_edges = NaN;

end