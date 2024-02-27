function [isoD, Lratio, silhouetteScore, d2_mahal, theseFeatures, otherFeatures_linear] = bc_getDistanceMetrics(pc_features, ...
    pc_feature_ind, thisUnit, numberSpikes, spikesIdx, allSpikesIdx, nChansToUse, plotThis)
% JF, Get distance metrics
% ------
% Inputs
% ------
% pc_features: nSpikes × nFeaturesPerChannel × nPCFeatures  single
%   matrix giving the PC values for each spike.
% pc_feature_ind: nTemplates × nPCFeatures uint32  matrix specifying which
%   channels contribute to each entry in dim 3 of the pc_features matrix
% thisUnit: unit number
% numberSpikes: number of spikes for that unit
% spikesIdx: boolean vector indicating which spikes belong to thisUnit
% allSpikesIdx: nSpikes × 1 uint32 vector giving the identity of each
%   spike's matched template, for all templates
% nChansToUse: number of channels to use to compute distance metrics (eg 4)
% plotThis: boolean, whether to plot the mahalobnis distance between spikes
%   of thisUnit and otherUnits on the nChansToUse closest channels
% ------
% Outputs
% ------
% isoD: isolation distance
% Lratio: l-ratio
% silhouetteScore: silhouette score
% d2_mahal QQ describe
% Xplot QQ describe
% Yplot QQ describe
%

nPCs = size(pc_features, 2); %should be 3 PCs

% get current unit's max `nChansToUse` channels
theseChannels = pc_feature_ind(thisUnit, 1:nChansToUse);

% current unit's features
theseFeatures = reshape(pc_features(spikesIdx, :, 1:nChansToUse), numberSpikes, []);

% Precompute unique identifiers and allocate space for outputs
uniqueIDs = unique(pc_feature_ind(:, 1));
mahalanobisDistances = nan(numel(uniqueIDs), 1);
otherFeaturesInd = zeros(0, size(pc_features, 2), nChansToUse);
otherFeatures = zeros(0, size(pc_features, 2), nChansToUse);
nCount = 1; % initialize counter

% Iterate over each unique ID
for iID = 1:numel(uniqueIDs)
    currentID = uniqueIDs(iID);

    % Skip if current ID matches the unit of interest
    if currentID == thisUnit
        continue;
    end

    % Identify channels associated with the current ID
    currentChannels = pc_feature_ind(iID, :);
    otherSpikes = allSpikesIdx == currentID;

    % Process channels that are common between current channels and the unit of interest
    for iChannel = 1:nChansToUse
        if ismember(theseChannels(iChannel), currentChannels)
            commonChannelIndex = find(currentChannels == theseChannels(iChannel), 1);
            channelSpikes = pc_features(otherSpikes, :, commonChannelIndex);
            otherFeatures(nCount:nCount+size(channelSpikes, 1)-1, :, iChannel) = channelSpikes;
            otherFeaturesInd(nCount:nCount+size(channelSpikes, 1)-1, :, iChannel) = currentID;
            nCount = nCount + size(channelSpikes, 1);
        end
    end

    % Calculate Mahalanobis distance if applicable
    if any(ismember(theseChannels(:), currentChannels))
        [rowIndices, ~, ~] = find(otherFeaturesInd == currentID);
        if size(theseFeatures, 1) > size(theseFeatures, 2) && numel(rowIndices) > size(theseFeatures, 2)
            otherFeatures = reshape(otherFeatures(rowIndices, :, :), numel(rowIndices), nPCs*nChansToUse);
            mahalanobisDistances(iID) = nanmean(mahal(otherFeatures, theseFeatures));
        else
            mahalanobisDistances(iID) = NaN;
        end
    end
end

% Predefine outputs to handle cases where conditions are not met
halfWayPoint = NaN;
isoD = NaN;
L = NaN;
Lratio = NaN;
silhouetteScore = NaN;

% Reshape features for mahalanobis distance calculation if there are other features
if ~isempty(theseOtherFeatures) && numberSpikes > nChansToUse * nPCs
    theseOtherFeatures = reshape(theseOtherFeatures, size(theseOtherFeatures, 1), []);
    mahalD = sort(mahal(theseOtherFeatures, theseFeatures)); % Sorted squared Mahalanobis distances

    % Calculate L-ratio
    L = sum(1-chi2cdf(mahalD, nPCs*nChansToUse)); % Assuming chi-square distribution
    Lratio = L / numberSpikes;

    % Find the closest cluster for silhouette score calculation
    closestCluster = thesU(find(theseMahalD == min(theseMahalD), 1, 'first'));
    mahalDself = mahal(theseFeatures, theseFeatures); % Self Mahalanobis distances

    % Find indices of features closest to the cluster
    [r, ~, ~] = ind2sub(size(theseOtherFeaturesInd), find(theseOtherFeaturesInd == double(closestCluster)));
    mahalDclosest = mahal(reshape(theseOtherFeatures(r, :, :), size(r, 1), nPCs*nChansToUse), theseFeatures);

    if nCount > numberSpikes && numberSpikes > nChansToUse * nPCs
        % Calculate isolation distance if applicable
        isoD = mahalD(ceil(size(theseFeatures, 1)/2));

        % Calculate silhouette score differently based on condition
        silhouetteScore = (nanmean(mahalDclosest) - nanmean(mahalDself)) / max([mahalDclosest; mahalDself]);
    else
        % Alternate silhouette score calculation when isolation distance is not defined
        silhouetteScore = (nanmean(mahalDself) - nanmean(mahalDclosest)) / max(mahalDclosest);
    end
end


if numberSpikes > nChansToUse * nPCs && exist('r', 'var')
    otherFeatures_linear = reshape(theseOtherFeatures(r, :, :), size(r, 1), nPCs*nChansToUse);

    % Calculate Mahalanobis distance for other spikes to the current unit features
    d2_mahal = mahal(otherFeatures_linear, theseFeatures);

    % Calculate Mahalanobis distance for the current unit relative to itself (for comparison)
    d2_mahal_self = mahal(theseFeatures, theseFeatures);


    if plotThis

        figure();
        subplot(3, 1, 1) % histograms
        hold on;
       
        % Histogram for distances of the current unit
        histogram(d2_mahal_self, 'BinWidth', 1, 'Normalization', 'probability', 'DisplayStyle', 'stairs', 'LineWidth', 2);
        % Histogram for distances of other spikes
        histogram(d2_mahal, 'BinWidth', 1, 'Normalization', 'probability', 'DisplayStyle', 'stairs', 'LineWidth', 2, 'EdgeColor', 'r');
        
        title('Normalized Mahalanobis Distances');
        xlabel('Squared mahalanobis distance');
        ylabel('Probability');
        legend({'Current Unit', 'Other Spikes'}, 'Location', 'Best');

        subplot(3, 1, 2) % cumulative distributions
        
        
        
        subplot(3, 1, 3)% 1 - cdf(chi ) .^2 



        % Add legends and other global annotations outside the loop, if they apply globally
        legend({'this cluster''s spikes', 'nearby clusters'' spikes', ['isolation distance = ', num2str(isoD)], ...
            ['silhouette score = ', num2str(silhouetteScore)], ['l-ratio = ', num2str(Lratio)]}, 'Location', 'bestoutside', 'TextColor', [0.7, 0.7, 0.7]);

        % Apply plot beautification if available
        if exist('prettify_plot', 'file')
            prettify_plot('FigureColor', 'w');
        else
            warning('https://github.com/Julie-Fabre/prettify-matlab repo missing - download it and add it to your matlab path to make plots pretty');
        end


        % uncomment below to additionnally plot all PCs against each other
        %figure();
        % % Calculate the number of subplots needed
        % nDims = nChansToUse * nPCs;
        % nSubplots = nDims * (nDims - 1) / 2; % n choose 2 for combinations
        % 
        % % Counter for subplot indexing
        % subplotIdx = 1;
        % 
        % for iDimX = 1:nDims
        %     for iDimY = iDimX + 1:nDims % Start from iDimX+1 to avoid diagonal and ensure unique pairs
        %         subplot(6, 11, subplotIdx); % Arrange in 6x11 grid to fit all combinations
        %         scatter(theseFeatures(:, iDimX), theseFeatures(:, iDimY), 10, [0.7, 0.7, 0.7], 'x'); % Xplot scatter plot
        %         hold on;
        %         scatter(otherFeatures_linear(:, iDimX), otherFeatures_linear(:, iDimY), 10, d2_mahal, 'o', 'filled'); % Yplot scatter plot with Mahalanobis distance coloring
        %         % Configure colorbar
        %         if iDimX == 1 && iDimY == 2
        %             hb = colorbar;
        %             hb.Color = [0.7, 0.7, 0.7];
        %             ylabel(hb, 'Mahalanobis Distance');
        %         end
        % 
        %         % Labels and titles for clarity
        %         xlabel(['PC', num2str(iDimX-(floor(iDimX/4) * 4)+1), ', channel ', num2str(ceil(iDimX/4))]);
        %         ylabel(['PC', num2str(iDimY-(floor(iDimY/4) * 4)+1), ', channel ', num2str(ceil(iDimY/4))]);
        % 
        %         % Increment subplot index
        %         subplotIdx = subplotIdx + 1;
        % 
        %         hold off; % Ready for next plot
        %     end
        % end
        % 

    end
else
    d2_mahal = NaN;
    otherFeatures_linear = NaN;
    theseFeatures = NaN;
end
end