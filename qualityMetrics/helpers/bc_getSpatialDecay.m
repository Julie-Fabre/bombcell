function [spatialDecaySlope, spatialDecayFit, spatialDecayPoints, spatialDecayPoints_loc, estimatedUnitXY] = ...
    bc_getSpatialDecay(templateWaveforms, thisUnit, maxChannel, channelPositions, linearFit)

    if linearFit % linear of fit of first 6 channels (at same X position). 
        % In real, good units, these points decrease linearly sharply (and, in further away channels they then decrease exponentially). 
        % In noise artefacts they are mostly flat. 
        channels_withSameX = find(channelPositions(:, 1) <= channelPositions(maxChannel, 1)+33 & ...
            channelPositions(:, 1) >= channelPositions(maxChannel, 1)-33); % for 4 shank probes
        if numel(channels_withSameX) >= 5
            if find(channels_withSameX == maxChannel) > 5
                channels_forSpatialDecayFit = channels_withSameX( ...
                    find(channels_withSameX == maxChannel):-1:find(channels_withSameX == maxChannel)-5);
            else
                channels_forSpatialDecayFit = channels_withSameX( ...
                    find(channels_withSameX == maxChannel):1:min(find(channels_withSameX == maxChannel)+5, size(channels_withSameX, 1)));
            end
    
            % get maximum value %QQ could we do value at detected trough is peak
            % waveform?
            spatialDecayPoints = max(abs(squeeze(templateWaveforms(thisUnit, :, channels_forSpatialDecayFit))));
           
            estimatedUnitXY = channelPositions(maxChannel, :);
            relativePositionsXY = channelPositions(channels_forSpatialDecayFit, :) - estimatedUnitXY;
            channelPositions_relative = sqrt(nansum(relativePositionsXY.^2, 2));
    
            [~, sortexChanPosIdx] = sort(channelPositions_relative);
            spatialDecayPoints_norm = spatialDecayPoints(sortexChanPosIdx);
            spatialDecayPoints_loc = channelPositions_relative(sortexChanPosIdx);
            spatialDecayFit = polyfit(spatialDecayPoints_loc, spatialDecayPoints_norm', 1); % fit first order polynomial to data. first output is slope of polynomial, second is a constant
            spatialDecaySlope = spatialDecayFit(1);
            if length(spatialDecayPoints) < 6
                    if length(spatialDecayPoints) > 1
                        spatialDecayPoints = [spatialDecayPoints_norm, nan(21-length(spatialDecayPoints_norm),1)];
                    else
                        spatialDecayPoints = [spatialDecayPoints_norm; nan(21-length(spatialDecayPoints_norm),1)];
                    end
            end
        else
            warning('No other good channels with same x location')
            spatialDecayFit = NaN;
            spatialDecaySlope = NaN;
            spatialDecayPoints = nan(1, 6);
    
        end
    else % not yet implemented. exponential fit? 
      

    end
end