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
%   - tauR_reject: (for ibl_sliding) refractory periods <= this (s) are not tested
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
% - Llobet, V., Wyngaard, A., & Barbour, B. Automatic post-processing and merging
%   of multiple spike-sorting analyses with Lussac. bioRxiv (2022), see equation 3.
%   https://doi.org/10.1101/2022.02.08.479192
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
    if isfield(param, 'tauR_reject')
        tauR_reject = param.tauR_reject;
    else
        tauR_reject = 0.0005;
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
            computeIBLSliding(spikeChunk, durationChunk, tauR, tauC, contaminationValues, confidenceThreshold, tauR_reject);
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
                        % Complex roots: more violations than the model allows for
                        % any contamination <= 0.5, so the unit is maximally contaminated
                        RPVrate(iTimeChunk, iTauR_value) = 1;
                        overestimateBool(iTimeChunk, iTauR_value) = 1;
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


function [contamination, estimatedTauR, RPVfraction, overestimateBool] = computeIBLSliding(spikeTimes, duration, tauR_values, tauC, contaminationValues, confidenceThreshold, tauR_reject)
% Compute contamination using the sliding refractory period method
% (Roth et al., SteinmetzLab/slidingRefractory), as _compute_sliding_rp_ibl
% in the python version.
%
% For each tested refractory period tauR and contamination level C, the
% confidence that the true contamination is below C is
% 1 - PoissonCDF(observed violations; expected violations), with expected
% violations from Llobet et al. (2022) (see
% bc.qm.helpers.slidingRPExpectedViolations). Observed violations are all
% spike pairs (not only consecutive spikes) with tauC <= dt < tauR.
% contamination(iTauR) is the lowest contamination reached with at least
% confidenceThreshold confidence at that tauR (1 if none is, NaN if tauR
% is not tested: tauR <= max(tauC, tauR_reject)). Its minimum across tauR
% is the unit's contamination. estimatedTauR is the tauR at which that
% minimum is confirmed with the highest confidence.

nTauR = length(tauR_values);
contamination = nan(1, nTauR);
RPVfraction = nan(1, nTauR);
overestimateBool = zeros(1, nTauR);
estimatedTauR = NaN;

n_spikes = length(spikeTimes);
[tauR_test, nViolations, effectiveDuration] = bc.qm.helpers.slidingRPViolationCounts( ...
    spikeTimes, duration, tauR_values, tauC, tauR_reject);
if isempty(tauR_test)
    return;
end
testedIdx = find(tauR_values(:)' > max(tauC, tauR_reject));

% confidence(iTauR, iCont)
expectedViolations = bc.qm.helpers.slidingRPExpectedViolations(n_spikes, effectiveDuration, ...
    tauR_test(:), tauC, contaminationValues(:)');
confidence = 1 - poisscdf(repmat(nViolations(:), 1, numel(contaminationValues)), expectedViolations);
passing = confidence >= confidenceThreshold;

for iTest = 1:numel(tauR_test)
    firstPassing = find(passing(iTest, :), 1, 'first');
    if isempty(firstPassing)
        contamination(testedIdx(iTest)) = 1; % no tested contamination level is confirmed
        overestimateBool(testedIdx(iTest)) = 1;
    else
        contamination(testedIdx(iTest)) = contaminationValues(firstPassing);
    end
    RPVfraction(testedIdx(iTest)) = nViolations(iTest) / n_spikes;
end

minContIdx = find(any(passing, 1), 1, 'first');
if isempty(minContIdx)
    estimatedTauR = tauR_test(1); % no tauR confirmed: fall back to the shortest tested tauR
else
    [~, bestTest] = max(confidence(:, minContIdx));
    estimatedTauR = tauR_test(bestTest);
end

end
