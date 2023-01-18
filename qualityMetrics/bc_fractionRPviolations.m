function [Fp, r, overestimate] = bc_fractionRPviolations(theseSpikeTimes, theseAmplitudes, tauR, tauC, timeChunks, plotThis)
% JF, get the estimated fraction of refractory period violation for a unit
% for each timeChunk
% ------
% Inputs
% ------
% N: number of spikes
% tauR: refractory period
% tauC: censored period
% timeChunk: experiment duration time chunks
% plotThis: boolean, whether to plot ISIs or not
% ------
% Outputs
% ------
% Fp: fraction of contamination
% r: number refractory period violations
%
% based on Hill et al., J Neuro, 2011
% r = 2*(tauR - tauC) * N^2 * (1-Fp) * Fp / T , solve for Fp , fraction
% refractory period violatons. 2 factor because rogue spikes can occur before or
% after true spike
Fp = nan(length(timeChunks)-1, 1);
if plotThis
        figure('Color','none');
    subplot(2,numel(timeChunks)-1, [1:numel(timeChunks)-1])
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
for iTimeChunk = 1:length(timeChunks) - 1
    N_chunk = length(theseSpikeTimes(theseSpikeTimes >= timeChunks(iTimeChunk) & theseSpikeTimes < timeChunks(iTimeChunk+1)));
    a = 2 * (tauR - tauC) * N_chunk^2 / abs(diff(timeChunks(iTimeChunk:iTimeChunk+1)));
    r = sum(diff(theseSpikeTimes(theseSpikeTimes >= timeChunks(iTimeChunk) & theseSpikeTimes < timeChunks(iTimeChunk+1))) <= tauR);

    if r == 0
        Fp(iTimeChunk) = 0;
        overestimate = 0;
    else
        rts = roots([-1, 1, -r / a]);
        Fp(iTimeChunk) = min(rts);
        overestimate = 0;
        if ~isreal(Fp(iTimeChunk)) %function returns imaginary number if r is too high: overestimate number.
            overestimate = 1;
            if r < N_chunk %to not get a negative wierd number or a 0 denominator
                Fp(iTimeChunk) = r / (2 * (tauR - tauC) * (N_chunk - r));
            else
                Fp(iTimeChunk) = 1;
            end
        end
        if Fp(iTimeChunk) > 1 %it is nonsense to have a rate >1, the assumptions are failing here
            Fp(iTimeChunk) = 1;
        end
    end
    
    ylims = [0, 0];
    if plotThis
 
%         subplot(2, length(timeChunk) - 1, iTimeChunk)
%         scatter(spikeTrain(spikeTrain >= timeChunk(iTimeChunk) & spikeTrain < timeChunk(iTimeChunk+1)), ...
%             theseAmplis(spikeTrain >= timeChunk(iTimeChunk) & spikeTrain < timeChunk(iTimeChunk+1)), 4, [0, 0.35, 0.71],'filled')
%         xlim([timeChunk(iTimeChunk), timeChunk(iTimeChunk+1)])
%         if iTimeChunk ==1
%         
%             xlabel('Time (s)')
%         ylabel(['Template scaling' newline 'factor amplitude'])
%        % title(['time chunk ' num2str(iTimeChunk)])
%         end
%         makepretty('none')

        
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
        legend(dummyh, [num2str(round(Fp(iTimeChunk)*100,1)), '% rpv'], 'Location', 'NorthEast','TextColor', [0.7, 0.7, 0.7], 'Color', 'none');
       

        makepretty('none')
    end
end

end