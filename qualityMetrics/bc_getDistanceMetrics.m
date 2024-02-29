function [isolationDist, Lratio, silhouetteScore, histogram_mahalUnit_counts, histogram_mahalUnit_edges, histogram_mahalNoise_counts, histogram_mahalNoise_edges] = bc_getDistanceMetrics(pc_features, ...
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
otherUnits_double = nan(numel(uniqueIDs), 1);
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
            otherFeatures_reshaped = reshape(otherFeatures(rowIndices, :, :), numel(rowIndices), nPCs*nChansToUse);
            mahalanobisDistances(iID) = nanmean(mahal(otherFeatures_reshaped, theseFeatures));
            otherUnits_double(iID) = double(currentID);
        else
            mahalanobisDistances(iID) = NaN;
            otherUnits_double(iID) = double(currentID);
        end
    end
end

% Predefine outputs to handle cases where conditions are not met
isolationDist = NaN;
Lratio = NaN;
silhouetteScore = NaN;
mahalD = NaN; 
otherFeatures_linear = NaN;
histogram_mahalUnit_counts = NaN;
histogram_mahalUnit_edges = NaN;
histogram_mahalNoise_counts = NaN;
histogram_mahalNoise_edges = NaN;

% Reshape features for mahalanobis distance calculation if there are other features
if ~isempty(otherFeatures) && numberSpikes > nChansToUse * nPCs
    otherFeatures = reshape(otherFeatures, size(otherFeatures, 1), []);
    mahalD = sort(mahal(otherFeatures, theseFeatures)); % Sorted squared Mahalanobis distances

    % Calculate L-ratio
    L = sum(1-chi2cdf(mahalD, nPCs*nChansToUse)); % Assuming chi-square distribution
    Lratio = L / numberSpikes;

    % Find the closest cluster for silhouette score calculation
    closestCluster = otherUnits_double(find(mahalanobisDistances == min(mahalanobisDistances), 1, 'first'));
    mahalDself = mahal(theseFeatures, theseFeatures); % Self Mahalanobis distances

    % Find indices of features closest to the cluster
    [r, ~, ~] = ind2sub(size(otherFeaturesInd), find(otherFeaturesInd == double(closestCluster)));
    % mahalDclosest = mahal(reshape(otherFeatures(r, :, :), size(r, 1), nPCs*nChansToUse), theseFeatures);

    if nCount > numberSpikes && numberSpikes > nChansToUse * nPCs
        % Calculate isolation distance if applicable
        isolationDist = mahalD(numberSpikes);

        % Calculate silhouette score differently based on condition
        % silhouetteScore = (nanmean(mahalDclosest) - nanmean(mahalDself)) / max([mahalDclosest; mahalDself]);
    else
        % Alternate silhouette score calculation when isolation distance is not defined
        % silhouetteScore = (nanmean(mahalDself) - nanmean(mahalDclosest)) / max(mahalDclosest);
    end
end


if numberSpikes > nChansToUse * nPCs && exist('r', 'var')
    otherFeatures_linear = reshape(otherFeatures(:, :, :), size(otherFeatures, 1), nPCs*nChansToUse);
    %bestOtherFeatures_linear = reshape(otherFeatures(r, :, :), size(r, 1), nPCs*nChansToUse);
    
    % Calculate Mahalanobis distance for other spikes to the current unit features
    %d2_mahal = mahal(otherFeatures_linear);
   % d2_mahal_best = mahal(bestOtherFeatures_linear, theseFeatures);

    % Calculate Mahalanobis distance for the current unit relative to itself (for comparison)
    d2_mahal_self = mahal(theseFeatures, theseFeatures);
    
    [histogram_mahalUnit_counts, histogram_mahalUnit_edges] = histcounts(mahalDself,1:1:200);
    [histogram_mahalNoise_counts, histogram_mahalNoise_edges] = histcounts(mahalD,1:1:200);

    if plotThis

        figure();
        subplot(1, 2, 1) % histograms
        hold on;
       
        % Histogram for distances of the current unit
        histogram(mahalDself, 'BinWidth', 1, 'Normalization', 'probability', 'DisplayStyle', 'stairs', 'LineWidth', 2);
        % Histogram for distances of other spikes
        histogram(mahalD, 'BinWidth', 1, 'Normalization', 'probability', 'DisplayStyle', 'stairs', 'LineWidth', 2, 'EdgeColor', 'r');
        
        title(['L-ratio = ' num2str(Lratio)]);
        xlabel('Squared mahalanobis distance');
        ylabel('Probability');
        set(gca, 'XScale', 'log')
        legend({'Current Unit', 'Other Spikes'}, 'Location', 'Best');
        
        % subplot(1, 3, 2)% 1 - cdf(chi ) .^2 
        % degrees_of_freedom = nPCs * nChansToUse;
        % 
        % % Calculate the CDF of the chi-square distribution
        % chi_square_cdf = chi2cdf(mahalD, degrees_of_freedom);
        % 
        % % Calculate 1 - CDF for the chi-square distribution
        % one_minus_cdf = 1 - chi_square_cdf;
        % plot(mahalD, one_minus_cdf, 'LineWidth', 2);
        % title(['L-ratio = ' num2str(Lratio)]);
        % xlabel('Squared mahalanobis distance');
        % ylabel('1 - CDF');
        % sub = num2str(degrees_of_freedom);
        % legendText = ['1 - CDF($\chi_{' sub '}^2$)'];
        % legend(legendText, 'Location', 'best', 'Interpreter', 'latex');
        

        subplot(1, 2, 2)%cumulative distributions
        nSpikesInUnit = size(theseFeatures,1);
 
        sOther = sort(mahalD);
        sSelf = sort(mahalDself);
                % Calculate cumulative counts
        cumulativeSelf = (1:nSpikesInUnit)';
        cumulativeOther = cumsum(ones(size(sOther)));
        
        % Plot cumulative distributions
        plot( sSelf,cumulativeSelf, 'LineWidth', 2, 'DisplayName', 'Cluster Spikes');
        hold on;
        plot( sOther,cumulativeOther, '--', 'LineWidth', 2, 'DisplayName', 'Noise Spikes');
        set(gca, 'XScale', 'log')
        set(gca, 'YScale', 'log')
        
        % Calculate and plot the isolation distance
        if length(sOther) >= nSpikesInUnit
            plot(  [isolationDist isolationDist],[1, nSpikesInUnit],'k:', 'LineWidth', 2, 'DisplayName', 'Isolation Distance');
        end
        
        xlim([0, max([sSelf; sOther])*1.1]);
        xlabel('Squared mahalanobis distance');
        ylabel('Cumulative Count');
        title(['Isolation distance = ', num2str(isolationDist)]);
        legend('Location', 'best');
        hold off;
        

        %subplot(2, 2, 4)% plot ordered distances 
        %hold on;
        %scatter(theseFeatures(:, 1), theseFeatures(:, 2), 10, d2_mahal, 'o', 'filled') % Scatter plot with points of size 10
        %scatter(bestOtherFeatures_linear(:, 1), bestOtherFeatures_linear(:, 2), 10, [0.7, 0.7, 0.7],'x')
        % % Calculate covariance and mean for the current unit (self)
        % covSelf = cov(theseFeatures(:,1:2));
        % meanSelf = mean(theseFeatures(:,1:2));
        
        % % Calculate covariance and mean for the closest other units
        % covOther = cov(bestOtherFeatures_linear(:,1:2));
        % meanOther = mean(bestOtherFeatures_linear(:,1:2));
        % 
        % % Calculate ellipses
        % theta = linspace(0, 2*pi, 100);
        % ellipseSelf = (chol(covSelf)' * [cos(theta); sin(theta)])' + meanSelf;
        % ellipseOther = (chol(covOther)' * [cos(theta); sin(theta)])' + meanOther;
        % 
        % % Draw ellipses
        % plot(ellipseSelf(:,1), ellipseSelf(:,2), 'LineWidth', 2, 'Color', 'blue');
        % plot(ellipseOther(:,1), ellipseOther(:,2), 'LineWidth', 2, 'Color', 'red');

        % 
        % hb = colorbar;
        % hb.Color =  [0.7, 0.7, 0.7];
        % ylabel(hb, 'Squared mahalanobis Distance')
        % legend( 'Current unit', 'Other spikes (closest unit)');
        % xlabel('Squared mahalanobnis distance');
        % %title(['Sihouette score = ' num2str(silhouetteScore)])
        % ylabel('Count');



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
end
end