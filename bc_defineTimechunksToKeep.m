function [percSpikesMissing, theseSpikeTimes, theseAmplis, timeChunks, useTheseTimes ] = bc_defineTimechunksToKeep(percSpikesMissing, ...
    maxPercSpikesMissing, theseAmplis, theseSpikeTimes, timeChunks)
    if any(percSpikesMissing < maxPercSpikesMissing) % if there are some good time chunks, keep those
        
        useTheseTimes = find(percSpikesMissing < maxPercSpikesMissing);
        percSpikesMissing = nanmean(percSpikesMissing(useTheseTimes(1):useTheseTimes(end)));
        theseAmplis = theseAmplis(theseSpikeTimes < timeChunks(useTheseTimes(end)+1) & ...
            theseSpikeTimes > timeChunks(useTheseTimes(1)));
        theseSpikeTimes = theseSpikeTimes(theseSpikeTimes < timeChunks(useTheseTimes(end)+1) & ...
            theseSpikeTimes > timeChunks(useTheseTimes(1)));
        %QQ change non continous
        timeChunks = timeChunks(useTheseTimes(1):useTheseTimes(end)+1);
    else %otherwise, keep all chunks to compute quality metrics on
        useTheseTimes = ones(numel(percSpikesMissing), 1);
        percSpikesMissing = nanmean(percSpikesMissing);
    end

end