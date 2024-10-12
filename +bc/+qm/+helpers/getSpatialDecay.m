function [spatialDecaySlope, spatialDecayFit, spatialDecayPoints, spatialDecayPoints_loc, estimatedUnitXY] = ...
    getSpatialDecay(templateWaveforms, thisUnit, maxChannel, channelPositions, linearFit, normalizePoints, computeSpatialDecay)
% QQ need to change this to be an exponential fit !
% Set default values and validate inputs
if nargin < 6 || isempty(normalizePoints)
    normalizePoints = false;
end


if computeSpatialDecay
    % Constants
    if linearFit
        CHANNEL_TOLERANCE = 33; % need to make more restricive. for most geometries, this includes all the channels.
        MIN_CHANNELS_FOR_FIT = 5;
        NUM_CHANNELS_FOR_FIT = 6;
    else
        CHANNEL_TOLERANCE = 33; % need to make more restricive. for most geometries, this includes all the channels.
        MIN_CHANNELS_FOR_FIT = 8;
        NUM_CHANNELS_FOR_FIT = 10;
    end

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

    if ~linearFit
       % Define the exponential decay function
        expDecayFun = @(b,x) b(1) * exp(-b(2)*x);
        
        % Set options for lsqcurvefit
        options = optimoptions('lsqcurvefit', 'Display', 'off');
        
        % Initial guess for parameters [A, b]
        initialGuess = [1, 0.1];
        
        % Perform exponential fit using lsqcurvefit
        [fitParams, ~, residual, ~, ~, ~, jacobian] = lsqcurvefit(expDecayFun, initialGuess, spatialDecayPoints_loc, spatialDecayPoints', [], [], options);
        
        spatialDecaySlope = fitParams(2);  % The decay rate is the second parameter
        spatialDecayFit = fitParams;
        
        % Calculate confidence intervals
        %ci = nlparci(fitParams, residual, 'jacobian', jacobian);
        
        % Print fit results
        %fprintf('Amplitude: %.4f (95%% CI: %.4f to %.4f)\n', fitParams(1), ci(1,1), ci(1,2));
        %fprintf('Spatial decay rate: %.4f (95%% CI: %.4f to %.4f)\n', spatialDecaySlope, ci(2,1), ci(2,2));
    else

        % Perform linear fit
        spatialDecayFit = polyfit(spatialDecayPoints_loc, spatialDecayPoints, 1);
        spatialDecaySlope = spatialDecayFit(1);
    end

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