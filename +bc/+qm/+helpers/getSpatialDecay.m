function [spatialDecaySlope, spatialDecayFit, spatialDecayPoints, spatialDecayPoints_loc, estimatedUnitXY] = ...
    getSpatialDecay(templateWaveforms, thisUnit, maxChannel, channelPositions, linearFit, normalizePoints, computeSpatialDecay)

% Set default values and validate inputs
if nargin < 6 || isempty(normalizePoints)
    normalizePoints = false;
end

if ~linearFit
    error('Non-linear fit is not yet implemented.');
end

if computeSpatialDecay
    % Constants
    CHANNEL_TOLERANCE = 33; % need to make more restricive. for most geometries, this includes all the channels.
    MIN_CHANNELS_FOR_FIT = 5;
    NUM_CHANNELS_FOR_FIT = 6;

    % Initialize output variables
    spatialDecaySlope = NaN;
    spatialDecayFit = NaN;
    spatialDecayPoints = nan(1, NUM_CHANNELS_FOR_FIT);
    spatialDecayPoints_loc = [];
    estimatedUnitXY = channelPositions(maxChannel, :);

    % Find channels with similar X position
    channels_withSameX = find(abs(channelPositions(:, 1)-channelPositions(maxChannel, 1)) <= CHANNEL_TOLERANCE);

    if numel(channels_withSameX) < MIN_CHANNELS_FOR_FIT
        warning('Insufficient channels with similar X position for fitting.');
        return;
    end

    % Select channels for spatial decay fit
    maxChannelIndex = find(channels_withSameX == maxChannel);
    if maxChannelIndex > NUM_CHANNELS_FOR_FIT
        channels_forSpatialDecayFit = channels_withSameX(maxChannelIndex:-1:maxChannelIndex-NUM_CHANNELS_FOR_FIT+1);
    else
        channels_forSpatialDecayFit = channels_withSameX(maxChannelIndex:min(maxChannelIndex+NUM_CHANNELS_FOR_FIT-1, numel(channels_withSameX)));
    end

    % Calculate spatial decay points
    spatialDecayPoints = max(abs(squeeze(templateWaveforms(thisUnit, :, channels_forSpatialDecayFit))));

    % Calculate relative positions
    relativePositionsXY = channelPositions(channels_forSpatialDecayFit, :) - estimatedUnitXY;
    channelPositions_relative = sqrt(sum(relativePositionsXY.^2, 2));
    [spatialDecayPoints_loc, sortIdx] = sort(channelPositions_relative);
    spatialDecayPoints = spatialDecayPoints(sortIdx);

    % Normalize spatial decay points if requested
    if normalizePoints
        spatialDecayPoints = spatialDecayPoints / max(spatialDecayPoints);
    end

    % Perform linear fit
    spatialDecayFit = polyfit(spatialDecayPoints_loc, spatialDecayPoints, 1);
    spatialDecaySlope = spatialDecayFit(1);

    % Pad spatialDecayPoints with NaNs if necessary
    if length(spatialDecayPoints) < NUM_CHANNELS_FOR_FIT
        spatialDecayPoints = [spatialDecayPoints, nan(1, NUM_CHANNELS_FOR_FIT-length(spatialDecayPoints))];
    end

else
    spatialDecaySlope = NaN;
    spatialDecayFit = NaN;
    spatialDecayPoints = NaN;
    spatialDecayPoints_loc = NaN;
    estimatedUnitXY = NaN;

end

end