function [RPVrate, RPVfraction, overestimateBool, estimatedTauR] = fractionRPviolations(theseSpikeTimes, theseAmplitudes, ...
    tauR, param, timeChunks, RPV_tauR_estimate)
% JF, get the estimated fraction of refractory period violation for a unit
% for each timeChunk
% ------
% Inputs
% ------
% theseSpikeTimes: nSpikesforThisUnit x 1 double vector of time in seconds
%   of each of the unit's spikes.
% theseAmplitudes: nSpikesforThisUnit x 1 double vector of the amplitude scaling factor
%   that was applied to the template when extracting that spike
%   , only needed if plotThis is set to true
% tauR: refractory period value(s) to test (scalar or array)
% timeChunk: experiment duration time chunks
% param: structure with fields:
%   - plotDetails: boolean, whether to plot ISIs or not
%   - tauC: censored period (ISIs below this are excluded)
%   - rpvMethod: 'hill', 'llobet', or 'ibl_sliding' (default: based on hillOrLlobetMethod)
%   - hillOrLlobetMethod: (legacy) boolean, true=hill, false=llobet
%   - contaminationValues: (for ibl_sliding) contamination values to test
%   - confidenceThreshold: (for ibl_sliding) confidence threshold (default 0.9)
% RPV_tauR_estimate: index of tauR to use for plotting (optional)
% ------
% Outputs
% ------
% RPVrate: estimated false positive rate of the spikes in the given
%   spike train, using Hill, Llobet, or IBL sliding method
% RPVfraction: fraction of refractory period violations over the total
%   number of spikes.
% overestimateBool: boolean, true if the number of refractory period violations
%   is too high (assumptions failing)
% estimatedTauR: the estimated refractory period (in seconds) that gives
%   minimum contamination (for sliding methods)
% ------
% Reference
% ------
% - Llobet V et al., biorXiv 2023. (see equation 3, page 11)
% - Hill, Daniel N., Samar B. Mehta, and David Kleinfeld.
%   "Quality metrics to accompany spike sorting of extracellular signals."
%   Journal of Neuroscience 31.24 (2011): 8699-8705
% - IBL sliding refractory period: https://github.com/SteinmetzLab/slidingRefractory
%   Accounts for different refractory periods across brain regions

% Determine which method to use
if isfield(param, 'rpvMethod')
    rpvMethod = param.rpvMethod;
elseif isfield(param, 'hillOrLlobetMethod')
    if param.hillOrLlobetMethod
        rpvMethod = 'hill';
    else
        rpvMethod = 'llobet';
    end
else
    rpvMethod = 'hill'; % default
end

% Get censored period
tauC = param.tauC;

% Get IBL sliding parameters
if strcmp(rpvMethod, 'ibl_sliding')
    if isfield(param, 'contaminationValues') && ~isempty(param.contaminationValues)
        contaminationValues = param.contaminationValues;
    else
        contaminationValues = (0.5:0.5:35) / 100; % default: 0.5% to 35%
    end
    if isfield(param, 'confidenceThreshold')
        confidenceThreshold = param.confidenceThreshold;
    else
        confidenceThreshold = 0.9;
    end
end

% Initialize variables
RPVrate = ones(length(timeChunks)-1, length(tauR));
overestimateBool = nan(length(timeChunks)-1, length(tauR));
RPVfraction = nan(length(timeChunks)-1, length(tauR));
estimatedTauR_perChunk = nan(length(timeChunks)-1, 1);

if param.plotDetails
    figure('Color', 'none');
end

for iTimeChunk = 1:length(timeChunks) - 1 % loop through each time chunk
    % spikes in this chunk
    spikeChunk = theseSpikeTimes(theseSpikeTimes >= timeChunks(iTimeChunk) & theseSpikeTimes < timeChunks(iTimeChunk+1));

    % number of spikes in chunk
    N_chunk = length(spikeChunk);

    % chunk duration
    durationChunk = timeChunks(iTimeChunk+1) - timeChunks(iTimeChunk);

    % chunk ISIs
    isisChunk = diff(spikeChunk);

    if strcmp(rpvMethod, 'ibl_sliding')
        % IBL sliding refractory period method
        [RPVrate(iTimeChunk, :), estimatedTauR_perChunk(iTimeChunk), RPVfraction(iTimeChunk, :), overestimateBool(iTimeChunk, :)] = ...
            computeIBLSliding(spikeChunk, durationChunk, tauR, tauC, contaminationValues, confidenceThreshold);
    else
        % Hill or Llobet method - loop through tauR values
        for iTauR_value = 1:length(tauR)
            thisTauR = tauR(iTauR_value);

            % Count violations: ISIs in range [tauC, tauR]
            % This excludes ISIs < tauC (duplicates/artifacts)
            if N_chunk > 1
                nRPVs = sum(isisChunk >= tauC & isisChunk <= thisTauR);
                RPVfraction(iTimeChunk, iTauR_value) = nRPVs / N_chunk;
            else
                nRPVs = 0;
                RPVfraction(iTimeChunk, iTauR_value) = 0;
            end

            overestimateBool(iTimeChunk, iTauR_value) = 0;

            if strcmp(rpvMethod, 'hill')
                % Hill et al. method
                % Effective window is (tauR - tauC)
                k = 2 * (thisTauR - tauC) * N_chunk^2;
                T = durationChunk;

                if nRPVs == 0
                    RPVrate(iTimeChunk, iTauR_value) = 0;
                else
                    rts = roots([k, -k, nRPVs * T]);
                    RPVrate(iTimeChunk, iTauR_value) = min(rts);
                    if ~isreal(RPVrate(iTimeChunk, iTauR_value))
                        % Approximation when roots are complex
                        if nRPVs < N_chunk
                            RPVrate(iTimeChunk, iTauR_value) = nRPVs / (2 * (thisTauR - tauC) * (N_chunk - nRPVs));
                        else
                            RPVrate(iTimeChunk, iTauR_value) = 1;
                            overestimateBool(iTimeChunk, iTauR_value) = 1;
                        end
                    end
                    if RPVrate(iTimeChunk, iTauR_value) > 1
                        RPVrate(iTimeChunk, iTauR_value) = 1;
                        overestimateBool(iTimeChunk, iTauR_value) = 1;
                    end
                end

            else % llobet method
                % Llobet et al. method - uses pairwise ISI violations
                N = length(spikeChunk);
                isi_violations_sum = 0;

                % Count all pair-wise violations in range [tauC, tauR]
                for i = 1:N-1
                    for j = i+1:N
                        isi = spikeChunk(j) - spikeChunk(i);
                        if isi <= thisTauR && isi >= tauC
                            isi_violations_sum = isi_violations_sum + 1;
                        end
                    end
                end

                % Effective duration adjusted for censored periods
                effectiveDuration = durationChunk - 2 * N_chunk * tauC;

                if N_chunk > 0 && (thisTauR - tauC) > 0 && effectiveDuration > 0
                    underRoot = 1 - (isi_violations_sum * effectiveDuration) / (N_chunk^2 * (thisTauR - tauC));
                    if underRoot >= 0
                        RPVrate(iTimeChunk, iTauR_value) = 1 - sqrt(underRoot);
                    else
                        RPVrate(iTimeChunk, iTauR_value) = 1;
                        overestimateBool(iTimeChunk, iTauR_value) = 1;
                    end
                else
                    RPVrate(iTimeChunk, iTauR_value) = 0;
                end
            end
        end

        % For non-sliding methods, estimate tauR as the one with minimum contamination
        [~, minIdx] = min(RPVrate(iTimeChunk, :));
        estimatedTauR_perChunk(iTimeChunk) = tauR(minIdx);
    end

    % Plotting
    if param.plotDetails
        subplot(2, length(timeChunks)-1, (length(timeChunks) - 1)+iTimeChunk)
        theseisiclean = isisChunk(isisChunk >= tauC); % removed duplicate spikes
        [isiProba, edgesISI] = histcounts(theseisiclean*1000, [0:0.5:100]);
        bar(edgesISI(1:end-1)+mean(diff(edgesISI)), isiProba, 'FaceColor', [0, 0.35, 0.71], ...
            'EdgeColor', [0, 0.35, 0.71]);
        if iTimeChunk == 1
            xlabel('Interspike interval (ms)')
            ylabel('# of spikes')
        else
            xticks([])
            yticks([])
        end
        ylims = ylim;
        [fr, ~] = histcounts(theseisiclean*1000, [0:0.5:5000]);
        line([0, 10], [nanmean(fr(800:1000)), nanmean(fr(800:1000))], 'Color', [0.86, 0.2, 0.13], 'LineStyle', '--');

        if isnan(RPV_tauR_estimate)
            for iTauR_value = [1, length(tauR)]
                line([tauR(iTauR_value) * 1000, tauR(iTauR_value) * 1000], [ylims(1), ylims(2)], 'Color', [0.86, 0.2, 0.13]);
            end
            if length(tauR) == 1
                title({[num2str(RPVrate(iTimeChunk, 1)*100, '%.0f'), '% rpv', newline, ...
                    'frac. rpv=', num2str(RPVfraction(iTimeChunk, 1), '%.3f')]});
            else
                title([num2str(round(RPVrate(iTimeChunk, 1)*100), '%.0f'), '% rpv', newline, ...
                    num2str(round(RPVrate(iTimeChunk, length(tauR))*100), '%.0f'), '% rpv']);
            end
        else
            line([tauR(RPV_tauR_estimate) * 1000, tauR(RPV_tauR_estimate) * 1000], [ylims(1), ylims(2)], 'Color', [0.86, 0.2, 0.13]);
            title({[num2str(RPVrate(iTimeChunk, RPV_tauR_estimate)*100, '%.0f'), '% rpv', newline, ...
                'frac. rpv=', num2str(RPVfraction(iTimeChunk, RPV_tauR_estimate), '%.3f')]});
        end
    end
end

% Return the estimated tauR (use chunk with minimum contamination)
[~, bestChunkIdx] = min(min(RPVrate, [], 2));
estimatedTauR = estimatedTauR_perChunk(bestChunkIdx);

if param.plotDetails
    subplot(2, numel(timeChunks)-1, 1:numel(timeChunks)-1)
    scatter(theseSpikeTimes, theseAmplitudes, 4, [0, 0.35, 0.71], 'filled');
    hold on;
    ylims = ylim;
    for iTimeChunk = 1:length(timeChunks)
        line([timeChunks(iTimeChunk), timeChunks(iTimeChunk)], ...
            [ylims(1), ylims(2)], 'Color', [0.7, 0.7, 0.7])
    end
    xlabel('time (s)')
    ylabel(['amplitude scaling', newline, 'factor'])
    if exist('prettify_plot', 'file')
        prettify_plot('FigureColor', 'w')
    else
        warning('https://github.com/Julie-Fabre/prettify-matlab repo missing - download it and add it to your matlab path to make plots pretty')
    end
end

end


function [contamination, estimatedTauR, RPVfraction, overestimateBool] = computeIBLSliding(spikeTimes, duration, tauR_values, tauC, contaminationValues, confidenceThreshold)
% Compute contamination using IBL sliding refractory period method
%
% This method sweeps across refractory periods and contamination values to find
% the minimum contamination with at least the specified confidence.

nTauR = length(tauR_values);
contamination = ones(1, nTauR);
RPVfraction = nan(1, nTauR);
overestimateBool = zeros(1, nTauR);

n_spikes = length(spikeTimes);
if n_spikes <= 1
    contamination(:) = NaN;
    estimatedTauR = NaN;
    return;
end

% Compute ISIs
isis = diff(spikeTimes);

% Effective duration: total duration minus censored periods around each spike
effectiveDuration = duration - 2 * n_spikes * tauC;
if effectiveDuration <= 0
    contamination(:) = NaN;
    estimatedTauR = NaN;
    return;
end

% Firing rate adjusted for effective duration
firingRate = n_spikes / effectiveDuration;

% Track best contamination across all tauR values
minContamination = NaN;
bestTauR = NaN;

for iTauR = 1:nTauR
    thisTauR = tauR_values(iTauR);

    % Skip if tauR is not greater than censored period
    if thisTauR <= tauC
        contamination(iTauR) = NaN;
        continue;
    end

    % Count violations: ISIs in range [tauC, tauR)
    n_violations = sum(isis >= tauC & isis < thisTauR);
    RPVfraction(iTauR) = n_violations / n_spikes;

    % Test each contamination value (from lowest to highest)
    foundContamination = false;
    for iCont = 1:length(contaminationValues)
        contVal = contaminationValues(iCont);

        % Expected violations given contamination
        % Window around each spike where violations can be detected: (tauR - tauC) on each side
        contaminationRate = firingRate * contVal;
        effectiveWindow = thisTauR - tauC;
        expectedViolations = contaminationRate * effectiveWindow * 2 * n_spikes;

        % Confidence score using Poisson distribution
        if expectedViolations > 0
            % P(X >= n_violations) where X ~ Poisson(expectedViolations)
            confidence = 1 - poisscdf(n_violations, expectedViolations);
        else
            if n_violations > 0
                confidence = 0;
            else
                confidence = 1;
            end
        end

        if confidence > confidenceThreshold
            contamination(iTauR) = contVal;
            foundContamination = true;

            % Track overall minimum
            if isnan(minContamination) || contVal < minContamination
                minContamination = contVal;
                bestTauR = thisTauR;
            end
            break;
        end
    end

    if ~foundContamination
        contamination(iTauR) = 1; % Max contamination if no value found
        overestimateBool(iTauR) = 1;
    end
end

estimatedTauR = bestTauR;

end
