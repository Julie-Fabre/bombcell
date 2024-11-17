function nSpikes = numberSpikes(theseSpikeTimes)
% JF, Count the number of spikes for the current unit
% ------
% Inputs
% ------
% theseSpikeTimes: nSpikesforThisUnit × 1 double vector of time in seconds
%   of each of the unit's spikes.
% ------
% Outputs
% ------
% nSpikes: number of spikes for current unit
% 
nSpikes = numel(theseSpikeTimes); 

end