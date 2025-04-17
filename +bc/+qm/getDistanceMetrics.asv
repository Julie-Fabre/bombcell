function [isolationDist, Lratio, silhouetteScore, histogram_mahalUnit_counts, histogram_mahalUnit_edges, ...
    histogram_mahalNoise_counts, histogram_mahalNoise_edges] = getDistanceMetrics(pc_features, ...
    pc_feature_ind, thisUnit, numberSpikes, spikesIdx, allSpikesIdx, param)
% ------
% Inputs
% ------
% pc_features: nSpikes × nFeaturesPerChannel × nPCFeatures single
%   matrix giving the PC values for each spike.
% pc_feature_ind: nTemplates × nPCFeatures uint32 matrix specifying which
%   channels contribute to each entry in dim 3 of the pc_features matrix
% thisUnit: unit number
% numberSpikes: number of spikes for that unit
% spikesIdx: boolean vector indicating which spikes belong to thisUnit
% allSpikesIdx: nSpikes × 1 uint32 vector giving the identity of each
%   spike's matched template, for all templates
% param: structure with fields: 
% - nChannelsIsoDist: number of channels to use to compute distance metrics (eg 4)
% - plotThis: boolean, whether to plot the mahalobnis distance between spikes
% ------
% Outputs
% ------
% isolationDist: isolation distance - matches maskedClusterQualityCore method
% Lratio: l-ratio
% silhouetteScore: silhouette score
% histogram_mahalUnit_counts: counts for histogram of within-unit distances
% histogram_mahalUnit_edges: edges for histogram of within-unit distances
% histogram_mahalNoise_counts: counts for histogram of noise distances
% histogram_mahalNoise_edges: edges for histogram of noise distances
    
    % Calculate total number of spikes in the dataset
    N = length(allSpikesIdx);
    nPCs = size(pc_features, 2); % should be 3 PCs per channel
    nFeaturesTotal = nPCs * param.nChannelsIsoDist;
    
    % Initialize output values to NaN
    isolationDist = NaN;
    Lratio = NaN;
    silhouetteScore = NaN;
    histogram_mahalUnit_counts = NaN;
    histogram_mahalUnit_edges = NaN;
    histogram_mahalNoise_counts = NaN;
    histogram_mahalNoise_edges = NaN;
    
    % Get channels for this unit
    theseChannels = pc_feature_ind(thisUnit, 1:param.nChannelsIsoDist);
    
    % Early exit condition 
    if numberSpikes < nFeaturesTotal || numberSpikes >= N/2
        return;
    end
    
    % Extract features for this unit 
    fetThisCluster = reshape(pc_features(spikesIdx, :, 1:param.nChannelsIsoDist), numberSpikes, []);
    
    % Finding other spikes
    otherFeatures = [];
    nInd = 1;
    
    % Get unique cluster IDs
    uniqueIDs = unique(allSpikesIdx(allSpikesIdx > 0));
    
    % Process each other cluster
    for c2 = 1:length(uniqueIDs)
        currentID = uniqueIDs(c2);
        
        % Skip if it's the current cluster
        if currentID == thisUnit
            continue;
        end
        
        % Get channels for this other cluster
        chansC2Has = pc_feature_ind(currentID, :);
        
        % Check for each channel in our target unit
        channelsFound = false;
        for f = 1:param.nChannelsIsoDist
            if ismember(theseChannels(f), chansC2Has)
                % Find spikes for this other cluster
                theseOtherSpikes = (allSpikesIdx == currentID);
                
                % Find the index of our channel in the other cluster's channel list
                thisCfetInd = find(chansC2Has == theseChannels(f), 1);
                
                % Calculate how many new spikes we're adding
                nNewSpikes = sum(theseOtherSpikes);
                
                % Preallocate if this is the first batch
                if ~channelsFound
                    tempFeatures = zeros(nNewSpikes, nPCs * param.nChannelsIsoDist);
                    channelsFound = true;
                end
                
                % Add features for this channel to the right position in the feature matrix
                tempFeatures(:, (f-1)*nPCs+1:f*nPCs) = pc_features(theseOtherSpikes, :, thisCfetInd);
                
            end
        end
        
        % If we found any channels, add these spikes to our collection
        if channelsFound
            otherFeatures = [otherFeatures; tempFeatures];
            nInd = nInd + sum(theseOtherSpikes);
        end
    end
    
    % Calculate isolation distance 
    if ~isempty(otherFeatures) && size(otherFeatures, 1) > 0
        % Get Mahalanobis distances for other clusters' spikes
        md = mahal(otherFeatures, fetThisCluster);
        md = sort(md);
        
        % Compute self-distances for other metrics
        mdSelf = mahal(fetThisCluster, fetThisCluster);
        mdSelf = sort(mdSelf);
        
        % Calculate isolation distance
        if length(md) >= numberSpikes
            isolationDist = md(numberSpikes);
        end
        
        % Calculate L-ratio
        L = sum(1 - chi2cdf(md, nFeaturesTotal));
        Lratio = L / numberSpikes;
        
        % Calculate a silhouette-like score
        silhouetteScore = mean(md) / mean(mdSelf);
        
        % Generate histograms for visualization
        [histogram_mahalUnit_counts, histogram_mahalUnit_edges] = histcounts(mdSelf, 1:1:200);
        [histogram_mahalNoise_counts, histogram_mahalNoise_edges] = histcounts(md, 1:1:200);
        
        % Plot if requested
        if isfield(param, 'plotDetails') && param.plotDetails
            figure();
            subplot(1, 2, 1) % histograms
            hold on;
            
            % Histogram for distances of the current unit
            histogram(mdSelf, 'BinWidth', 1, 'Normalization', 'probability', 'DisplayStyle', 'stairs', 'LineWidth', 2);
            % Histogram for distances of other spikes
            histogram(md, 'BinWidth', 1, 'Normalization', 'probability', 'DisplayStyle', 'stairs', 'LineWidth', 2, 'EdgeColor', 'r');
            
            title(['L-ratio = ' num2str(Lratio)]);
            xlabel('Squared mahalanobis distance');
            ylabel('Probability');
            set(gca, 'XScale', 'log')
            legend({'this unit''s spikes', 'Other Spikes'}, 'Location', 'Best');
            
            subplot(1, 2, 2) % cumulative distributions
            nSpikesInUnit = size(fetThisCluster, 1);
            
            % Calculate cumulative counts
            cumulativeSelf = (1:nSpikesInUnit)';
            cumulativeOther = cumsum(ones(size(md)));
            
            % Plot cumulative distributions
            plot(mdSelf, cumulativeSelf, 'LineWidth', 2, 'DisplayName', 'This unit''s spikes');
            hold on;
            plot(md, cumulativeOther, '--', 'LineWidth', 2, 'DisplayName', 'Noise Spikes');
            set(gca, 'XScale', 'log')
            set(gca, 'YScale', 'log')
            
            % Calculate and plot the isolation distance
            if ~isnan(isolationDist)
                plot([isolationDist isolationDist], [1, nSpikesInUnit], 'k:', 'LineWidth', 2, 'DisplayName', 'Isolation Distance');
            end
            
            xlim([0, max([mdSelf; md]) * 1.1]);
            xlabel('Squared mahalanobis distance');
            ylabel('Cumulative Count');
            title(['Isolation distance = ', num2str(isolationDist)]);
            legend('Location', 'best');
            hold off;
        end
    end
end