function nSpikes = bc_numberSpikes(theseSpikeTimes)
% JF, Get the max channel for all templates (channel with largest amplitude)
% ------
% Inputs
% ------
% theseSpikeTimes: nSpikesforThisUnit × 1 double vector of time in seconds
%   of each of the unit's spikes.
% ------
% Outputs
% ------
% nSpikes: number of spikes for that unit 
% 
nSpikes = numel(theseSpikeTimes); 

end