function [theseSpikeTimes, theseAmplis, theseSpikeTemplates, useThisTimeStart, useThisTimeStop, useTauR] = defineTimechunksToKeep(percSpikesMissing, ...
    fractionRPVs, param, theseAmplis, theseSpikeTimes, theseSpikeTemplates, timeChunks, spikeTimes_seconds)
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
% param: structure withg fields:
% - maxPercSpikesMissing
% - maxfractionRPVs
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

% Calculate the sum of refractory period violations across all time chunks
sumRPV = sum(fractionRPVs, 1);

% Find the largest tauR value that gives the smallest contamination
useTauR = find(sumRPV == min(sumRPV), 1, 'last');

% Identify time chunks that meet the criteria for low missing spikes and RPVs
useTheseTimes_temp = find(percSpikesMissing < param.maxPercSpikesMissing & fractionRPVs(:, useTauR) < param.maxRPVviolations);

if numel(useTheseTimes_temp) > 0 % Case where there are good time chunks
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
        
        % Select the time range for the longest chunk
        useTheseTimes = timeChunks(longestChunkStart:longestChunkEnd);
    else 
        % No continuous time chunks: arbitrarily, just use the first one
        useTheseTimes = timeChunks(useTheseTimes_temp(1):useTheseTimes_temp(1)+1);
    end
    
    % Index spikes, templates and amplitudes for chosen time chunk
    validIndices_allUnits = spikeTimes_seconds <= useTheseTimes(end) & spikeTimes_seconds >= useTheseTimes(1);
    theseSpikeTemplates(~validIndices_allUnits) = 0; % set to 0 so these are not used in subsequent quality metrics 

    validIndices_thisUnit = theseSpikeTimes <= useTheseTimes(end) & theseSpikeTimes >= useTheseTimes(1);
    theseAmplis = theseAmplis(validIndices_thisUnit );
    theseSpikeTimes = theseSpikeTimes(validIndices_thisUnit );

    % Set the start and stop times for the selected chunk
    useThisTimeStart = useTheseTimes(1);
    useThisTimeStop = useTheseTimes(end);
else 
    % If there are no "good" time chunks, use all chunks for subsequent computations
    useThisTimeStart = 0;
    useThisTimeStop = timeChunks(end);
end
end