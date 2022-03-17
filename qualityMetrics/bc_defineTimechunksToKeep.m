function [theseSpikeTimes, theseAmplis, timeChunksKeep, useTheseTimes ] = bc_defineTimechunksToKeep(percSpikesMissing, ...
    Fp, maxPercSpikesMissing, maxFp, theseAmplis, theseSpikeTimes, timeChunks)
    if any(percSpikesMissing < maxPercSpikesMissing) && any(Fp < maxFp)  % if there are some good time chunks, keep those
        
        useTheseTimes_temp = find(percSpikesMissing < maxPercSpikesMissing & Fp < maxFp);
        if numel(useTheseTimes_temp) > 1
            continousTimes=find(diff(useTheseTimes_temp)==1); %get continous time chunks to use 
            [continousTimesUseLength,jmax]=max(continousTimes) ;
            continousTimesUseStart=continousTimes(1,jmax);
            useTheseTimes  = useTheseTimes_temp(continousTimesUseStart:continousTimesUseStart + (continousTimesUseLength -min(continousTimes)));
        else
            useTheseTimes = useTheseTimes_temp;
        end
        theseAmplis = theseAmplis(theseSpikeTimes < timeChunks(useTheseTimes(end)+1) & ...
            theseSpikeTimes > timeChunks(useTheseTimes(1)));
        theseSpikeTimes = theseSpikeTimes(theseSpikeTimes < timeChunks(useTheseTimes(end)+1) & ...
            theseSpikeTimes > timeChunks(useTheseTimes(1)));
        %QQ change non continous
        timeChunksKeep = timeChunks(useTheseTimes(1):useTheseTimes(end)+1);
    else %otherwise, keep all chunks to compute quality metrics on, uni will defined as below percSpikesMissing and Fp criteria thresholds later 
        useTheseTimes = ones(numel(percSpikesMissing), 1);
        timeChunksKeep = NaN;
    end

end