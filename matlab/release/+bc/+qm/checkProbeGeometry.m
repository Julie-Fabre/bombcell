function computeSpatialDecay = checkProbeGeometry(channelPositions)
% check if there are enough close-by channels to compute spatial decay
% (i.e. is it a high density probe or not) 

if min(diff(unique(channelPositions(:, 2)))) < 30
    computeSpatialDecay = 1;
else
    computeSpatialDecay = 0;
end

end
