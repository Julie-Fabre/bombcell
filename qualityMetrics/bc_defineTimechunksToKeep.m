function [theseSpikeTimes, theseAmplis, timeChunksKeep, useTheseTimes ] = bc_defineTimechunksToKeep(percSpikesMissing, ...
    Fp, maxPercSpikesMissing, maxFp, theseAmplis, theseSpikeTimes, timeChunks)
    if any(percSpikesMissing < maxPercSpikesMissing) && any(Fp < maxFp)  % if there are some good time chunks, keep those
        
        useTheseTimes_temp = find(percSpikesMissing < maxPercSpikesMissing & Fp < maxFp);
        if numel(useTheseTimes_temp) > 0 
            continousTimes = diff(useTheseTimes_temp);
             if any(continousTimes == 1)
                f = find(diff([false,continousTimes==1,false])~=0);
                [continousTimesUseLength,ix] = max(f(2:2:end)-f(1:2:end-1));
                continousTimesUseStart = useTheseTimes_temp(continousTimes(f(2*ix-1)));
                useTheseTimes  = timeChunks(continousTimesUseStart:continousTimesUseStart + (continousTimesUseLength));
            else
                useTheseTimes = timeChunks(useTheseTimes_temp(1):useTheseTimes_temp(1)+1);
            end
        else
            useTheseTimes = timeChunks;
        end
        theseAmplis = theseAmplis(theseSpikeTimes < useTheseTimes(end) & ...
            theseSpikeTimes > useTheseTimes(1));
        theseSpikeTimes = theseSpikeTimes(theseSpikeTimes < useTheseTimes(end) & ...
            theseSpikeTimes > useTheseTimes(1));
        %QQ change non continous
        timeChunksKeep = useTheseTimes(1):useTheseTimes(end);
    else %otherwise, keep all chunks to compute quality metrics on, uni will defined as below percSpikesMissing and Fp criteria thresholds later 
        useTheseTimes = ones(numel(percSpikesMissing), 1);
        timeChunksKeep = NaN;
    end
%dbcont;
end