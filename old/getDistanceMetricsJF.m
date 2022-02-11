function [isoD,Lratio, silhouetteScore,mahalD, halfWayPoint,L] = getDistanceMetricsJF(pc_features, ...
    pc_feature_ind, thisUnit, numberSpikes, spikesIdx, allSpikesIdx, nChansToUse)

% based on functions in https://github.com/cortex-lab/sortingQuality
%nChansToUse = 4; %tetrodes
nPCs = size(pc_features, 2); %should be 3 PCs


% features for this cluster
theseFeatures = reshape(pc_features(spikesIdx, :, 1:nChansToUse), numberSpikes, []);
theseChannels = pc_feature_ind(thisUnit, 1:nChansToUse);

nCount = 1;

% features for all other clusters on 4 first channels of this cluster
uC = unique(pc_feature_ind(:, 1));
theseMahalD = nan(numel(uC), 1);
theseOtherFeaturesInd = zeros(0, size(pc_features, 2), nChansToUse);
theseOtherFeatures = zeros(0, size(pc_features, 2), nChansToUse);
for iOtherUnit = 1:numel(uC)
    thisOtherUnit = uC(iOtherUnit);
    if thisOtherUnit ~= thisUnit
        theseOtherChans = pc_feature_ind(iOtherUnit, :);
        for iChannel = 1:nChansToUse
            if ismember(theseChannels(iChannel), theseOtherChans)
                theseOtherSpikes = allSpikesIdx == thisOtherUnit;
                thisCommonChan = find(theseOtherChans == theseChannels(iChannel), 1);
                theseOtherFeatures(nCount:nCount+sum(theseOtherSpikes)-1, :, iChannel) = ...
                    pc_features(theseOtherSpikes, :, thisCommonChan);
                theseOtherFeaturesInd(nCount:nCount+sum(theseOtherSpikes)-1, :, iChannel) = ...
                    ones(size(pc_features(theseOtherSpikes, :, thisCommonChan), 1), ...
                    size(pc_features(theseOtherSpikes, :, thisCommonChan), 2), 1) * double(thisOtherUnit);
                %thisOtherFeaturesInd = ones(numel(nCount:nCount+sum(theseOtherSpikes)-1),1)*double(thisOtherUnit);
                %theseOtherFeaturesInd = [theseOtherFeaturesInd, thisOtherFeaturesInd];

            end
        end
        if any(ismember(theseChannels(:), theseOtherChans))
            nCount = nCount + sum(theseOtherSpikes);
            %if ~isempty(theseOtherFeaturesSingle(iOtherUnit).THIS)
            [r, ~, ~] = ind2sub(size(theseOtherFeaturesInd), find(theseOtherFeaturesInd == double(thisOtherUnit)));
            %find(theseOtherFeaturesInd==double(thisOtherUnit))
            if size(theseFeatures, 1) > size(theseFeatures, 2) && size(r, 1) > size(theseFeatures, 2)
                theseMahalD(iOtherUnit) = nanmean(mahal(reshape(theseOtherFeatures(r, :, :), size(r, 1), nPCs*nChansToUse), theseFeatures));
                thesU(iOtherUnit) = double(thisOtherUnit);
            else
                theseMahalD(iOtherUnit) = NaN;
                thesU(iOtherUnit) = double(thisOtherUnit);
            end
            %end
        end
    end
end

if nCount > numberSpikes && numberSpikes > nChansToUse * nPCs %isolation distance not defined
    %isolation distance
    halfWayPoint = size(theseFeatures, 1);
    
    %l-ratio
    theseOtherFeatures = reshape(theseOtherFeatures, size(theseOtherFeatures, 1), []);
    mahalD = mahal(theseOtherFeatures, theseFeatures);
    mahalD = sort(mahalD);
    isoD = mahalD(halfWayPoint);
    L = sum(1-chi2cdf(mahalD, nPCs*nChansToUse)); % assumes a chi square distribution - QQ add test for multivariate data
    Lratio = L / numberSpikes;

    %silhouette score
    closestCluster = thesU(find(theseMahalD == min(theseMahalD)));
    mahalDself = mahal(theseFeatures, theseFeatures);
    [r, ~, ~] = ind2sub(size(theseOtherFeaturesInd), find(theseOtherFeaturesInd == double(closestCluster)));
    % find(theseOtherFeaturesInd==double(thisOtherUnit))

    mahalDclosest = mahal(reshape(theseOtherFeatures(r, :, :), size(r, 1), nPCs*nChansToUse), theseFeatures);
    silhouetteScore = (nanmean(mahalDclosest) - nanmean(mahalDself)) / max([mahalDclosest; mahalDself]);

    %d-prime

    %allFeat = [theseFeatures; theseOtherFeatures];

    %y = logical(zeros(size(allFeat,1),1));
    %y(1:size(theseFeatures,1)) = 1;

    %MdlLinear = fitcdiscr(allFeat,y);
    %classify(
elseif ~isempty(theseOtherFeatures) && numberSpikes > nChansToUse * nPCs
    %isolation distance
    halfWayPoint = NaN;
    isoD = NaN;
    
    %l-ratio
    theseOtherFeatures = reshape(theseOtherFeatures, size(theseOtherFeatures, 1), []);
    mahalD = mahal(theseOtherFeatures, theseFeatures);
    mahalD = sort(mahalD);
    L = sum(1-chi2cdf(mahalD, nPCs*nChansToUse)); % assumes a chi square distribution - QQ add test for multivariate data
    Lratio = L / numberSpikes;

    %silhouette score
    closestCluster = thesU(find(theseMahalD == min(theseMahalD)));
    mahalDself = mahal(theseFeatures, theseFeatures);
    [r, ~, ~] = ind2sub(size(theseOtherFeaturesInd), find(theseOtherFeaturesInd == double(closestCluster)));
    % find(theseOtherFeaturesInd==double(thisOtherUnit))

    mahalDclosest = mahal(reshape(theseOtherFeatures(r, :, :), size(r, 1), nPCs*nChansToUse), theseFeatures);
    silhouetteScore = (nanmean(mahalDself) - nanmean(mahalDclosest)) / max(mahalDclosest);
else
    %isolation distance
    halfWayPoint = NaN;
    mahalD = NaN;
    isoD = NaN;
    
    %l-ratio
    L = NaN; 
    Lratio = NaN;

    %silhouette score
    silhouetteScore = NaN;

end


end