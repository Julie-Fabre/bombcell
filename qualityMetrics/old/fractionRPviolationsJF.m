function [Fp, r, overestimate] = fractionRPviolationsJF(N, spikeTrain, tauR, tauC, T)
% JF
% ------
% Inputs
% ------
% N: number of spikes
% tauR: refractory period
% tauC: censored period
% T: total experiment duration
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
end