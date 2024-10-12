function [maxDrift_estimate, cumulativeDrift_estimate] = maxDriftEstimate(pcFeatures, pcFeatureIdx, spikeTemplates, ...
    spikeTimes, channelPositions_z, thisUnit, param)
% JF, Estimate the maximum drift for a particular unit
% ------
% Inputs
% ------
% pc_features: nSpikes × nFeaturesPerChannel × nPCFeatures  single
%   matrix giving the PC values for each spike.
% pc_feature_ind: nTemplates × nPCFeatures uint32  matrix specifying which
%   channels contribute to each entry in dim 3 of the pc_features matrix
% spike_templates: nSpikes × 1 uint32 vector giving the identity of each
%   spike's matched template
% thisUnit: unit number
% param: structure with fields 
% - driftBinSize
% - computeDrift: boolean, whether tocomputeDrift( this is botle-neck slow step
%   that takes almost 2 seconds per unit)
% - plotThis: boolean, whether to plot results (not implemented yet for this
%   function)
% ------
% Outputs
% ------
% maxDriftEstimate: maximum absolute difference between peak channels, in
%   um
% cumulativeDrift_estimate: cummulative absolute difference between peak channels, in
%   um
%
% ------
% References
% ------
% For the metric: Siegle, J.H., Jia, X., Durand, S. et al. Survey of spiking in the mouse
% visual system reveals functional hierarchy. Nature 592, 86–92 (2021). https://doi.org/10.1038/s41586-020-03171-x
% For the center of mass estimation, this is based on the method in:
% https://github.com/cortex-lab/spikes/analysis/ksDriftMap
if param.computeDrift

    %% calculate center of mass for each spike
    pcFeatures_PC1 = squeeze(pcFeatures(spikeTemplates == thisUnit, 1, :)); % take the first PC
    pcFeatures_PC1(pcFeatures_PC1 < 0) = 0; % remove negative entries - we don't want to push the center of mass away from there.
    
    % for each spike, get which channel the maximum value is located on 
    spikePC_feature = nan(size(spikeTemplates,1),size(pcFeatures,3)); 
    spikePC_feature(spikeTemplates>0,:) = double(pcFeatureIdx(spikeTemplates(spikeTemplates>0), :)); % get channels for each spike. only spikeTemplates>0 because in computeTimeChuynks we set spikeTempltes to 0 is they are outside of our "good" time range. 

    spikeDepths_inChannels = sum(channelPositions_z(spikePC_feature(spikeTemplates == thisUnit, :)).*pcFeatures_PC1.^2, 2) ./ sum(pcFeatures_PC1.^2, 2); % center of mass: sum(coords.*features)/sum(features)

    %% estimate cumulative drift

    timeBins = min(spikeTimes):param.driftBinSize:max(spikeTimes);
    median_spikeDepth = arrayfun(@(x) median(spikeDepths_inChannels(spikeTimes >= x & spikeTimes < x + param.driftBinSize)), timeBins);

    maxDrift_estimate = nanmax(median_spikeDepth) - nanmin(median_spikeDepth);
    cumulativeDrift_estimate = sum(abs(diff(median_spikeDepth(~isnan(median_spikeDepth)))));

    if param.plotDetails
        figure(); 
        plot(timeBins, median_spikeDepth)
        xlabel('time (s)')
        ylabel('estimated spike depth (um)')
        title(['cumulative drift = ', num2str(cumulativeDrift_estimate) 'um', newline, 'maximum drift/bin = ', num2str(maxDrift_estimate) 'um'])

    end
else
    maxDrift_estimate = NaN;
    cumulativeDrift_estimate = NaN;
end