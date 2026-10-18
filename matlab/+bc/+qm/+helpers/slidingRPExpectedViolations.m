function expectedViolations = slidingRPExpectedViolations(nSpikes, effectiveDuration, tauR, tauC, contamination)
% SLIDINGRPEXPECTEDVIOLATIONS Expected number of spike pairs with
% tauC <= dt < tauR for a unit of nSpikes spikes of which a fraction
% contamination are contaminating (Poisson) spikes, from Llobet et al. (2022):
%   Ve = 2 * (tauR - tauC) / D * Nc * (Nb + (Nc - 1) / 2)
% where Nc = C * N and Nb = (1 - C) * N. Uses implicit expansion over tauR
% and contamination.
nContaminating = nSpikes .* contamination;
nBase = nSpikes - nContaminating;
expectedViolations = 2 .* (tauR - tauC) ./ effectiveDuration ...
    .* nContaminating .* (nBase + (nContaminating - 1) ./ 2);
end
