function [theseSpikeTimes, theseAmplis, theseSpikeTemplates, useThisTimeStart, useThisTimeStop, useTauR] = bc_defineTimechunksToKeep(percSpikesMissing, ...
    fractionRPVs, maxPercSpikesMissing, maxfractionRPVs, theseAmplis, theseSpikeTimes, theseSpikeTemplates,timeChunks)
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

% use biggest tauR value that gives smallest contamination 
sumRPV = sum(fractionRPVs,2);
useTauR = find(sumRPV == min(sumRPV),1, 'last');

if any(percSpikesMissing < maxPercSpikesMissing) && any(fractionRPVs(:,:,useTauR) < maxfractionRPVs) % if there are some good time chunks, keep those

    useTheseTimes_temp = find(percSpikesMissing < maxPercSpikesMissing & fractionRPVs(:,:,useTauR) < maxfractionRPVs);
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
    theseSpikeTemplates(theseSpikeTimes > useTheseTimes(end) | ...
        theseSpikeTimes < useTheseTimes(1)) = 0;
    theseAmplis = theseAmplis(theseSpikeTimes <= useTheseTimes(end) & ...
        theseSpikeTimes >= useTheseTimes(1));
    theseSpikeTimes = theseSpikeTimes(theseSpikeTimes <= useTheseTimes(end) & ...
        theseSpikeTimes >= useTheseTimes(1));
    
    %QQ change non continous

    useThisTimeStart = useTheseTimes(1);
    useThisTimeStop = useTheseTimes(end);
else %otherwise, keep all chunks to compute quality metrics on, uni will defined as below percSpikesMissing and Fp criteria thresholds later
    useThisTimeStart = 0;
    useThisTimeStop = timeChunks(end);
end

end