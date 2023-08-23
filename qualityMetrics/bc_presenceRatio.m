function presenceRatio = bc_presenceRatio(theseSpikeTimes, theseAmplis, presenceRatioBinSize, startTime, stopTime, plotThis)
% JF, Calculate fraction of bins that include one or more spikes from a particular unit
% ------
% Inputs
% ------
% theseSpikeTimes: nSpikesforThisUnit × 1 double vector of time in seconds
%   of each of the unit's spikes.
% startTime : defining time chunk start value (dividing the recording in time of chunks of param.deltaTimeChunk size)
%       where the percentage of spike missing and percentage of false positives
%       is below param.maxPercSpikesMissing and param.maxRPVviolations
% stopTime: defining time chunk stop value (dividing the recording in time of chunks of param.deltaTimeChunk size)
%       where the percentage of spike missing and percentage of false positives
%       is below param.maxPercSpikesMissing and param.maxRPVviolations
% presenceRatioBin: size of time bins in which to calculate the presence
%   ratio
% plotThis: boolean, whether to plot amplitude distribution and fit or not
% ------
% Outputs
% ------
% presenceRatio : fraction of bins (of bin size presenceRatioBinSize) that
%   contain at least one spike 
%
% Note that cells can have low scores in this metric if they have selective
%   firing patterns. 
% ------
% Reference 
% ------
% Siegle, J.H., Jia, X., Durand, S. et al. Survey of spiking in the mouse 
% visual system reveals functional hierarchy. Nature 592, 86–92 (2021). https://doi.org/10.1038/s41586-020-03171-x

% divide recordings times in chunks
presenceRatio_bins = startTime:presenceRatioBinSize:stopTime;
% count number of spikes in each chunk 
spikesPerBin = arrayfun(@(x) sum(theseSpikeTimes>=presenceRatio_bins(x) & theseSpikeTimes<presenceRatio_bins(x+1)),1:length(presenceRatio_bins)-1);
fullBins = spikesPerBin>0;
presenceRatio = sum(fullBins)/length(spikesPerBin);

if plotThis 
    figure('Color','none');
    colors = [146,0,0; 34,207,34; 103, 103, 103]./255;
    scatter(theseSpikeTimes, theseAmplis, 4,[0, 0.35, 0.71],'filled'); hold on;
    % chunk lines 
    ylims = ylim;
    arrayfun(@(x) line([presenceRatio_bins(x), presenceRatio_bins(x+1)], [ylims(2)*0.9,ylims(2)*0.9], 'Color', colors(fullBins(x)+1,:)),1:length(presenceRatio_bins)-1);
    arrayfun(@(x) line([presenceRatio_bins(x), presenceRatio_bins(x)], [ylims(1),ylims(2)], 'Color', colors(3,:)),1:length(presenceRatio_bins));
    
    xlabel('time (s)')
    ylabel(['amplitude scaling' newline 'factor'])
    prettify_plot('none', 'all', 'none');
end



