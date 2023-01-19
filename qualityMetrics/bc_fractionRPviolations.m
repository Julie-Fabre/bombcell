function [fractionRPVs, nRPVs, overestimate] = bc_fractionRPviolations(theseSpikeTimes, theseAmplitudes, tauR, tauC, timeChunks, plotThis)
% JF, get the estimated fraction of refractory period violation for a unit
% for each timeChunk
% ------
% Inputs
% ------
% theseSpikeTimes: nSpikesforThisUnit × 1 double vector of time in seconds
%   of each of the unit's spikes.
% theseAmplitudes: nSpikesforThisUnit × 1 double vector of the amplitude scaling factor 
%   that was applied to the template when extracting that spike
%   , only needed if plotThis is set to true
% tauR: refractory period
% tauC: censored period
% timeChunk: experiment duration time chunks
% plotThis: boolean, whether to plot ISIs or not
% ------
% Outputs
% ------
% fractionRPVs: fraction of contamination
% nRPVs: number refractory period violations
% overestimate: boolean, true if the number of refractory period violations 
%    is too high. we then overestimate the fraction of
%    contamination. 
% ------
% Reference 
% ------
% Hill, Daniel N., Samar B. Mehta, and David Kleinfeld. 
% "Quality metrics to accompany spike sorting of extracellular signals."
% Journal of Neuroscience 31.24 (2011): 8699-8705:
% r = 2*(tauR - tauC) * N^2 * (1-Fp) * Fp / T , solve for Fp , fraction
% refractory period violatons. 2 factor because rogue spikes can occur before or
% after true spike
% QQ to add sliding: see IBL spike sorting white paper
% can vary across brain areas (Bar-Gad et al., 2001; Sukiban et al., 2019). We thus developed a metric
% which estimates whether a neuron is contaminated by refractory period violations (indicating po-
% tential overmerge problems in the clustering step) without assuming the length of the refractory
% period. For each of many possible refractory period lengths (ranging from 0.5 ms to 10 ms, in 0.25
% ms bins), we compute the number of spikes (refractory period violations) that would correspond
% to some maximum acceptable amount of contamination (chosen as 10%). We then compute the
% likelihood of observing fewer than this number of spikes in that refractory period under the as-
% sumption of Poisson spiking. For a neuron to pass this metric, this likelihood, or the confidence
% that our neuron is less than 10% contaminated, must be larger than 90% for any one of the possi-
% ble refractory period lengths. This metric rejects neurons with short true refractory periods when
% firing rates are low, as we cannot be statistically confident that the lack of contamination did not
% arise by chance. As the true refractory period increases, neurons with low contamination begin to
% pass the metric. 

fractionRPVs = nan(length(timeChunks)-1, 1); % initialize variable 

if plotThis
    figure('Color','none');
    subplot(2,numel(timeChunks)-1, 1:numel(timeChunks)-1)
    scatter(theseSpikeTimes, theseAmplitudes, 4,[0, 0.35, 0.71],'filled'); hold on;
    % chunk lines 
    ylims = ylim;
    for iTimeChunk = 1:length(timeChunks)
        line([timeChunks(iTimeChunk),timeChunks(iTimeChunk)],...
            [ylims(1),ylims(2)], 'Color', [0.7, 0.7, 0.7])
    end
    xlabel('time (s)')
    ylabel(['amplitude scaling' newline 'factor'])
    makepretty('none')
end
for iTimeChunk = 1:length(timeChunks) - 1 %loop through each time chunk
    % number of spikes in chunk 
    N_chunk = length(theseSpikeTimes(theseSpikeTimes >= timeChunks(iTimeChunk) & theseSpikeTimes < timeChunks(iTimeChunk+1)));
    % total times at which refractory period violations can occur
    a = 2 * (tauR - tauC) * N_chunk^2 / abs(diff(timeChunks(iTimeChunk:iTimeChunk+1)));
    % observed number of refractory period violations
    nRPVs = sum(diff(theseSpikeTimes(theseSpikeTimes >= timeChunks(iTimeChunk) & theseSpikeTimes < timeChunks(iTimeChunk+1))) <= tauR);

    if nRPVs == 0 % if no observed refractory period violations 
        fractionRPVs(iTimeChunk) = 0;
        overestimate = 0;
    else % otherwise solve the equation above 
        rts = roots([-1, 1, -nRPVs / a]);
        fractionRPVs(iTimeChunk) = min(rts);
        overestimate = 0;
        if ~isreal(fractionRPVs(iTimeChunk)) % function returns imaginary number if r is too high: overestimate number.
            overestimate = 1;
            if nRPVs < N_chunk %to not get a negative wierd number or a 0 denominator
                fractionRPVs(iTimeChunk) = nRPVs / (2 * (tauR - tauC) * (N_chunk - nRPVs));
            else
                fractionRPVs(iTimeChunk) = 1;
            end
        end
        if fractionRPVs(iTimeChunk) > 1 %it is nonsense to have a rate >1, the assumptions are failing here
            fractionRPVs(iTimeChunk) = 1;
        end
    end
    
    
    if plotThis

        subplot(2, length(timeChunks) - 1, (length(timeChunks) - 1)+iTimeChunk)
        theseISI = diff(theseSpikeTimes(theseSpikeTimes >= timeChunks(iTimeChunk) & theseSpikeTimes < timeChunks(iTimeChunk+1)));
        theseisiclean = theseISI(theseISI >= tauC); % removed duplicate spikes
        [isiProba, edgesISI] = histcounts(theseisiclean*1000, [0:0.5:10]);
        bar(edgesISI(1:end-1)+mean(diff(edgesISI)), isiProba, 'FaceColor', [0, 0.35, 0.71], ...
             'EdgeColor', [0, 0.35, 0.71]); %Check FR
        if iTimeChunk ==1
        xlabel('Interspike interval (ms)')
        ylabel('# of spikes')
        end
        ylims = ylim;
        line([2, 2], [ylims(1), ylims(2)], 'Color', [0.86, 0.2, 0.13]);
        [fr, ~] = histcounts(theseisiclean*1000, [0:0.5:1000]);
        line([0, 10], [nanmean(fr(800:1000)), nanmean(fr(800:1000))], 'Color',[0.86, 0.2, 0.13], 'LineStyle', '--');
        dummyh = line(nan, nan, 'Linestyle', 'none', 'Marker', 'none', 'Color', 'none');
        legend(dummyh, [num2str(round(fractionRPVs(iTimeChunk)*100,1)), '% rpv'], 'Location', 'NorthEast','TextColor', [0.7, 0.7, 0.7], 'Color', 'none');
       

        makepretty('none')
    end
end

end