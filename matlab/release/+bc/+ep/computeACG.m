function ACG = computeACG(theseSpikeTimes, ACGbinSize, ACGduration, plotThis)
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