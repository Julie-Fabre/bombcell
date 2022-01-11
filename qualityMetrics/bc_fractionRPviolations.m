function [Fp, r, overestimate] = bc_fractionRPviolations(N, spikeTrain, tauR, tauC, T, plotThis)
% JF, get the estimated fraction of refractory period violation for a unit
% ------
% Inputs
% ------
% N: number of spikes
% tauR: refractory period
% tauC: censored period
% T: total experiment duration
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


a = 2 * (tauR - tauC) * N^2 / T;
r = sum(diff(spikeTrain) <= tauR);

if r == 0
    Fp = 0;
    overestimate = 0;
else
    rts = roots([-1, 1, -r / a]);
    Fp = min(rts);
    overestimate = 0;
    if ~isreal(Fp) %function returns imaginary number if r is too high: overestimate number.
        overestimate = 1;
        if r < N %to not get a negative wierd number or a 0 denominator
            Fp = r / (2 * (tauR - tauC) * (N - r));
        else
            Fp = 1;
        end
    end
end

if plotThis
    figure();
    theseTimes = spikeTrain- spikeTrain(1);
    theseISI = diff(spikeTrain);
    theseisiclean = theseISI(theseISI >= tauC); % removed duplicate spikes 
    [isiProba, edgesISI] = histcounts(theseisiclean*1000, [0:0.5:50]);
    bar(edgesISI(1:end-1)+mean(diff(edgesISI)), isiProba); %Check FR
    xlabel('Interspike interval (ms)')
    ylabel('# of spikes')
end
end