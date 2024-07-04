function [theseSpikeTimes, theseAmplis, theseSpikeTemplates, useThisTimeStart, useThisTimeStop, useTauR] = defineTimechunksToKeep(percSpikesMissing, ...
    fractionRPVs, maxPercSpikesMissing, maxfractionRPVs, theseAmplis, theseSpikeTimes, theseSpikeTemplates, timeChunks)
% JF
% define time chunks where the current unit has low refractory period violations and
% estimated percent spikes missing
% ------
% Inputs
% ------
% percSpikesMissing: estimated percentage of spikes missing for the current
%   unit, for each time chunk
% fractionRPVs: estimated percentage of spikes missing for the current
%   unit, for each time chunk
% maxPercSpikesMissing
% maxfractionRPVs
% theseAmplis: current unit spike-to-template scaling factors
% theseSpikeTimes: current unit spike times
% theseSpikeTemplates:  nSpikes × 1 uint32 vector giving the identity of each
%   spike's matched template
% timeChunks: time chunkbins  of the recording in which the percSpikesMissing
%   and fractionRPVs are computed
% ------
% Outputs
% ------
% theseSpikeTimes: current unit spike times in time bins where estimated
%   refractory period violations and estimated percent spikes missing are low
% theseAmplis: current unit spike-to-template scaling factors in time bins where estimated
%   refractory period violations and estimated percent spikes missing are low
% theseSpikeTemplates
% useThisTimeStart: start bin value where current unit has low refractory period violations and
%   estimated percent spikes missing
% useThisTimeStop: start bin value where current unit has low refractory period violations and
%   estimated percent spikes missing
% useTauR: estimated refractory period for the current unit

% use biggest tauR value that gives smallest contamination
sumRPV = sum(fractionRPVs, 1);
useTauR = find(sumRPV == min(sumRPV), 1, 'last');

if any(percSpikesMissing < maxPercSpikesMissing) && any(fractionRPVs(:, useTauR) < maxfractionRPVs) % if there are some good time chunks, keep those

   useTheseTimes_temp = find(percSpikesMissing < maxPercSpikesMissing & fractionRPVs(:,useTauR) < maxfractionRPVs);
    if numel(useTheseTimes_temp) > 0
        continousTimes = diff(useTheseTimes_temp);
        if any(continousTimes == 1)
            % Identify the start and end of consecutive chunks
            chunkStarts = [1; find(continousTimes > 1) + 1];
            chunkEnds = [find(continousTimes > 1); length(useTheseTimes_temp)];
        
            % Calculate the lengths of each chunk
            chunkLengths = chunkEnds - chunkStarts + 1;
        
            % Find the longest chunk
            [longestChunkLength, idx] = max(chunkLengths);
            longestChunkStart = useTheseTimes_temp(chunkStarts(idx));
            longestChunkEnd = useTheseTimes_temp(chunkEnds(idx));
            useTheseTimes = timeChunks(longestChunkStart:longestChunkEnd);

        else
            useTheseTimes = timeChunks(useTheseTimes_temp(1):useTheseTimes_temp(1)+1);
        end
    else % if there are no "good" time chunks, use all chunks for subsequent computations 
        useTheseTimes = timeChunks;
    end


    theseSpikeTemplates(theseSpikeTimes > useTheseTimes(end) | ...
        theseSpikeTimes < useTheseTimes(1)) = 0;
    theseAmplis = theseAmplis(theseSpikeTimes <= useTheseTimes(end) & ...
        theseSpikeTimes >= useTheseTimes(1));
    theseSpikeTimes = theseSpikeTimes(theseSpikeTimes <= useTheseTimes(end) & ...
        theseSpikeTimes >= useTheseTimes(1));

    useThisTimeStart = useTheseTimes(1);
    useThisTimeStop = useTheseTimes(end);

else  % if there are no "good" time chunks, use all chunks for subsequent computations 
    useThisTimeStart = 0;
    useThisTimeStop = timeChunks(end);
end

end