function [isolationDist, Lratio, silhouetteScore, histogram_mahalUnit_counts, histogram_mahalUnit_edges, ...
    histogram_mahalNoise_counts, histogram_mahalNoise_edges] = getDistanceMetrics(pc_features, ...
    pc_feature_ind, thisUnit, numberSpikes, spikesIdx, allSpikesIdx, param)
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
% param: structure with fields: 
% - nChannelsIsoDist: number of channels to use to compute distance metrics (eg 4)
% - plotThis: boolean, whether to plot the mahalobnis distance between spikes
%   of thisUnit and otherUnits on the nChannelsIsoDist closest channels
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

% get current unit's max `nChannelsIsoDist` channels
theseChannels = pc_feature_ind(thisUnit, 1:param.nChannelsIsoDist);

% current unit's features
theseFeatures = reshape(pc_features(spikesIdx, :, 1:param.nChannelsIsoDist), numberSpikes, []);

% Precompute unique identifiers and allocate space for outputs
uniqueIDs = unique(allSpikesIdx(allSpikesIdx>0));
otherFeaturesInd = zeros(0, size(pc_features, 2), param.nChannelsIsoDist);
otherFeatures = zeros(0, size(pc_features, 2), param.nChannelsIsoDist);
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
    for iChannel = 1:param.nChannelsIsoDist
        if ismember(theseChannels(iChannel), currentChannels)
            commonChannelIndex = find(currentChannels == theseChannels(iChannel), 1);
            channelSpikes = pc_features(otherSpikes, :, commonChannelIndex);
            otherFeatures(nCount:nCount+size(channelSpikes, 1)-1, :, iChannel) = channelSpikes;
            otherFeaturesInd(nCount:nCount+size(channelSpikes, 1)-1, :, iChannel) = currentID;
            nCount = nCount + size(channelSpikes, 1);
        end
    end

end

% Predefine outputs to handle cases where conditions are not met
isolationDist = NaN;
Lratio = NaN;
silhouetteScore = NaN;
mahalD = NaN; 
histogram_mahalUnit_counts = NaN;
histogram_mahalUnit_edges = NaN;
histogram_mahalNoise_counts = NaN;
histogram_mahalNoise_edges = NaN;

% Reshape features for mahalanobis distance calculation if there are other features
if ~isempty(otherFeatures) && numberSpikes > param.nChannelsIsoDist * nPCs
    otherFeatures = reshape(otherFeatures, size(otherFeatures, 1), []);
    mahalD = sort(mahal(otherFeatures, theseFeatures)); % Sorted squared Mahalanobis distances

    % Calculate L-ratio
    L = sum(1-chi2cdf(mahalD, nPCs*param.nChannelsIsoDist)); % Assuming chi-square distribution
    Lratio = L / numberSpikes;


    if nCount > numberSpikes && numberSpikes > param.nChannelsIsoDist * nPCs
        % Calculate isolation distance if applicable
        isolationDist = mahalD(numberSpikes);

        % Calculate silhouette score differently based on condition
        % silhouetteScore = (nanmean(mahalDclosest) - nanmean(mahalDself)) / max([mahalDclosest; mahalDself]);
    else
        % Alternate silhouette score calculation when isolation distance is not defined
        % silhouetteScore = (nanmean(mahalDself) - nanmean(mahalDclosest)) / max(mahalDclosest);
    end
end


if numberSpikes > param.nChannelsIsoDist * nPCs 
    mahalDself = mahal(theseFeatures, theseFeatures); % Self Mahalanobis distances
  
    [histogram_mahalUnit_counts, histogram_mahalUnit_edges] = histcounts(mahalDself,1:1:200);
    [histogram_mahalNoise_counts, histogram_mahalNoise_edges] = histcounts(mahalD,1:1:200);

    if param.plotDetails

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
        legend({'this unit''s spikes', 'Other Spikes'}, 'Location', 'Best');
        

        subplot(1, 2, 2)%cumulative distributions
        nSpikesInUnit = size(theseFeatures,1);
 
        sOther = sort(mahalD);
        sSelf = sort(mahalDself);
                % Calculate cumulative counts
        cumulativeSelf = (1:nSpikesInUnit)';
        cumulativeOther = cumsum(ones(size(sOther)));
        
        % Plot cumulative distributions
        plot( sSelf,cumulativeSelf, 'LineWidth', 2, 'DisplayName', 'This unit''s spikes');
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

        % Apply plot beautification if available
        if exist('prettify_plot', 'file')
            prettify_plot('FigureColor', 'w');
        else
            warning('https://github.com/Julie-Fabre/prettify-matlab repo missing - download it and add it to your matlab path to make plots pretty');
        end

        % uncomment below to additionnally plot all PCs against each other
        %figure();
        % % Calculate the number of subplots needed
        % nDims = nChannelsIsoDist * nPCs;
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