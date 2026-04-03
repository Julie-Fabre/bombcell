function ACG = computeACG(theseSpikeTimes, ACGbinSize, ACGduration, plotThis)
% Handle empty or insufficient spike times
if isempty(theseSpikeTimes) || numel(theseSpikeTimes) < 2
    halfBins = round(ACGduration/ACGbinSize/2);
    nBins = 2*halfBins+1;
    ACG = zeros(nBins, 1);
    return
end

[acg, ~] = bc.ep.helpers.CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
        ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', ACGbinSize, 'duration', ACGduration, 'norm', 'rate'); %function
ACG= acg(:, 1, 1);

if plotThis
    figure(); 
    plot(0:ACGbinSize:ACGduration/2, ACG(round(size(ACG,1)/2):end))
    %set(gca, 'XScale', 'log')
    xlabel('time (s)')
    ylabel('spike rate')
    prettify_plot();

end
end