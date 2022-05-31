function [Fp, r, overestimate] = bc_fractionRPviolations(N, spikeTrain, theseAmplis, tauR, tauC, timeChunk, plotThis)
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
Fp = nan(length(timeChunk)-1, 1);
if plotThis
    figure();
end
for iTimeChunk = 1:length(timeChunk) - 1
    a = 2 * (tauR - tauC) * N^2 / abs(diff(timeChunk(iTimeChunk:iTimeChunk+1)));
    r = sum(diff(spikeTrain(spikeTrain >= timeChunk(iTimeChunk) & spikeTrain < timeChunk(iTimeChunk+1))) <= tauR);

    if r == 0
        Fp(iTimeChunk) = 0;
        overestimate = 0;
    else
        rts = roots([-1, 1, -r / a]);
        Fp(iTimeChunk) = min(rts);
        overestimate = 0;
        if ~isreal(Fp(iTimeChunk)) %function returns imaginary number if r is too high: overestimate number.
            overestimate = 1;
            if r < N %to not get a negative wierd number or a 0 denominator
                Fp(iTimeChunk) = r / (2 * (tauR - tauC) * (N - r));
            else
                Fp(iTimeChunk) = 1;
            end
        end
    end
    if plotThis
 
        subplot(2, length(timeChunk) - 1, iTimeChunk)
        scatter(spikeTrain(spikeTrain >= timeChunk(iTimeChunk) & spikeTrain < timeChunk(iTimeChunk+1)), ...
            theseAmplis(spikeTrain >= timeChunk(iTimeChunk) & spikeTrain < timeChunk(iTimeChunk+1)), 4, 'filled')
        xlim([timeChunk(iTimeChunk), timeChunk(iTimeChunk+1)])
        xlabel('Time (s)')
        ylabel(['Template scaling' newline 'factor amplitude'])
        title(['time chunk ' num2str(iTimeChunk)])
        subplot(2, length(timeChunk) - 1, (length(timeChunk) - 1)+iTimeChunk)
        theseISI = diff(spikeTrain(spikeTrain >= timeChunk(iTimeChunk) & spikeTrain < timeChunk(iTimeChunk+1)));
        theseisiclean = theseISI(theseISI >= tauC); % removed duplicate spikes
        [isiProba, edgesISI] = histcounts(theseisiclean*1000, [0:0.5:50]);
        bar(edgesISI(1:end-1)+mean(diff(edgesISI)), isiProba); %Check FR
        xlabel('Interspike interval (ms)')
        ylabel('# of spikes')
    end
end

end