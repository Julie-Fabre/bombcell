function [theseSpikeTimes, theseAmplis, useThisTimeStart, useThisTimeStop] = bc_defineTimechunksToKeep(percSpikesMissing, ...
    fractionRPVs, maxPercSpikesMissing, maxfractionRPVs, theseAmplis, theseSpikeTimes, timeChunks)
% JF
% define time chunks where unit has low refractory period violations and
% estimated percent spikes missing 
% ------
% Inputs
% ------
% 
% ------
% Outputs
% ------
%


if any(percSpikesMissing < maxPercSpikesMissing) && any(fractionRPVs < maxfractionRPVs) % if there are some good time chunks, keep those

    useTheseTimes_temp = find(percSpikesMissing < maxPercSpikesMissing & fractionRPVs < maxfractionRPVs);
    if numel(useTheseTimes_temp) > 0
        continousTimes = diff(useTheseTimes_temp);
        if any(continousTimes == 1)
            f = find(diff([false, continousTimes == 1, false]) ~= 0);
            [continousTimesUseLength, ix] = max(f(2:2:end)-f(1:2:end-1));
            continousTimesUseStart = useTheseTimes_temp(continousTimes(f(2*ix-1)));
            useTheseTimes = timeChunks(continousTimesUseStart:continousTimesUseStart+(continousTimesUseLength));
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

    useThisTimeStart = useTheseTimes(1);
    useThisTimeStop = useTheseTimes(end);
else %otherwise, keep all chunks to compute quality metrics on, uni will defined as below percSpikesMissing and Fp criteria thresholds later
    useThisTimeStart = 0;
    useThisTimeStop = timeChunks(end);
end

end