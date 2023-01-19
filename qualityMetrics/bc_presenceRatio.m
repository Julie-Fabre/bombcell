function presenceRatio = bc_presenceRatio(theseSpikeTimes, presenceRatioBin, plotThis)
% JF, Calculate fraction of bins that include one or more spikes from a particular unit
% ------
% Inputs
% ------
% theseSpikeTimes: nSpikesforThisUnit × 1 double vector of time in seconds
%   of each of the unit's spikes.
% presenceRatioBin: 
% plotThis: boolean, whether to plot amplitude distribution and fit or not
% ------
% Outputs
% ------
% presenceRatio
%
% Note that cells can have low scores in this metric if they have selective
%   firing patterns. 
% ------
% Reference 
% ------
% Siegle, J.H., Jia, X., Durand, S. et al. Survey of spiking in the mouse 
% visual system reveals functional hierarchy. Nature 592, 86–92 (2021). https://doi.org/10.1038/s41586-020-03171-x
bins 
arrayfun
% divide recordings times in chunks
% count number of spikes in each chunk 



