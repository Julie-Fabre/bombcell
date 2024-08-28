function [percentMissing_gaussian, percentMissing_symmetric, ksTest_pValue, ...
    ampliBin_gaussian, spikeCountsPerAmpliBin_gaussian, gaussianFit_cutoff] = percSpikesMissing(theseAmplitudes, theseSpikeTimes, timeChunks, param)
% JF, Estimate the amount of spikes missing (below the detection threshold)
% by fitting a gaussian to the amplitude distribution for each timeChunk
% defined in timeChunks
% ------
% Inputs
% ------
% theseAmplitudes: nSpikesforThisUnit × 1 double vector of the amplitude scaling factor
%   that was applied to the template when extracting that spike
% theseSpikeTimes: nSpikesforThisUnit × 1 double vector of time in seconds
%   of each of the unit's spikes.
% timeChunks: timeChunks edges in which to compute the metric
% param: structure with field plotThis: boolean, whether to plot amplitude distribution and fit or not
% ------
% Outputs
% ------
% percentMissing_gaussian: estimated percent spike missing for each time
%   chunk, assuming the data is normally distributed
% percentMissing_symmetric: estimated percent spike missing for each time
%   chunk, assuming the data is normally distributed
% ksTest_pValue: Kolmogorov-Smirnov test p value for each time chunk
% spikeCountsPerAmpliBin_gaussian: spike counts per amplitude bin, for GUI
%   plotting
% gaussianFit_cutoff: gaussian fit curve, for GUI plotting
%
% Note that this will underestimate the amount of spikes missing in the
% case e.g. of bursty cells, where there spike amplitude is decreased during the
% burst.

% initialize variables
percentMissing_gaussian = nan(numel(timeChunks)-1, 1);
percentMissing_symmetric = nan(numel(timeChunks)-1, 1);
ksTest_pValue = nan(numel(timeChunks)-1, 1);

if param.plotDetails
    figure('Color', 'none');
end

for iTimeChunk = 1:numel(timeChunks) - 1
    % amplitude histogram
    nBins = 50;
    % get amplitude histogram
    [spikeCountsPerAmpliBin, bins] = histcounts(theseAmplitudes(theseSpikeTimes >= timeChunks(iTimeChunk) & ...
        theseSpikeTimes < timeChunks(iTimeChunk+1)), nBins);

    % initialize values
    percentMissing_gaussian(iTimeChunk) = NaN;
    percentMissing_symmetric(iTimeChunk) = NaN;
    ksTest_pValue(iTimeChunk) = NaN;
    ampliBin_gaussian = NaN;
    spikeCountsPerAmpliBin_gaussian = NaN;
    gaussianFit_cutoff = NaN;

    if sum(spikeCountsPerAmpliBin) < 20 % too few spikes
        continue;
    end

    % Find the bin with the maximum count
    [maxCount, maxBin] = max(spikeCountsPerAmpliBin);

    maxAmpli_val = find(spikeCountsPerAmpliBin == max(spikeCountsPerAmpliBin));
    mode_seed = bins(maxAmpli_val); %guess mode - this is only accurate if mode is present in histogram
    if numel(mode_seed) > 1 %if two or more modes, take the mean
        mode_seed = mean(mode_seed);
    end
    bin_steps = diff(bins(1:2)); %size of a bin

    %% symmetric
    % mirror the unit's amplitudes
    % median smooth ti get more accurate peak ]
    spikeCountsPerAmpliBin_smooth = smoothdata(spikeCountsPerAmpliBin, 'movmedian', 5);
    maxAmpli_val_smooth = find(spikeCountsPerAmpliBin_smooth == max(spikeCountsPerAmpliBin_smooth), 1);

    surrogate_amplitudes = [spikeCountsPerAmpliBin_smooth(end:-1:maxAmpli_val_smooth), fliplr(spikeCountsPerAmpliBin_smooth(end:-1:maxAmpli_val_smooth+1))];
    surrogate_bins = [fliplr(bins(maxAmpli_val_smooth)-bin_steps:-bin_steps:bins(maxAmpli_val_smooth)-bin_steps*floor(size(surrogate_amplitudes, 2)/2)), ...
        bins(maxAmpli_val:end)];

    % remove any bins where the amplitudes would be < 0, that doesn't make any sense
    surrogate_amplitudes(surrogate_bins < 0) = [];
    surrogate_bins(surrogate_bins < 0) = [];
    surrogate_area = sum(surrogate_amplitudes) * bin_steps;

    % estimate the percentage of missing spikes
    pMissing = (surrogate_area - sum(spikeCountsPerAmpliBin) * bin_steps) / surrogate_area * 100;
    if pMissing < 0 % if inferior, distribution isn't symmetric
        pMissing = 0;
    end
    percentMissing_symmetric(iTimeChunk) = pMissing;

    %% gaussian
    ampliBin_gaussian = bins(1:end-1) + bin_steps / 2;
    next_low_bin = ampliBin_gaussian(1) - (bin_steps / 2);

    % Calculate the starting point, ensuring it's non-negative
    start_point = max(0, next_low_bin-floor(next_low_bin/bin_steps)*bin_steps);

    % Create the lower part of the array
    add_points = start_point:bin_steps:next_low_bin;

    ampliBin_gaussian = [add_points, ampliBin_gaussian];
    spikeCountsPerAmpliBin_gaussian = [zeros(size(add_points, 2), 1)', spikeCountsPerAmpliBin];

    
    % Check if the maximum is at either edge of the distribution
    if maxBin == 1 || maxBin == length(spikeCountsPerAmpliBin)

        percentMissing_gaussian(iTimeChunk) = 50;
        percentMissing_symmetric(iTimeChunk) = 50;
        ksTest_pValue(iTimeChunk) = NaN;

        gaussianFit_cutoff = NaN(size(ampliBin_gaussian));

        % Skip the rest of the processing for this time chunk
        continue;
    end

    % Initial seed parameters
    p0 = [max(spikeCountsPerAmpliBin_gaussian), ... % Height
        median(theseAmplitudes), ... % Center
        std(theseAmplitudes), ... % Width
        min(theseAmplitudes)]; % Cutoff


    f = @(x, xdata)gaussian_cut(x, xdata); % get anonymous function handle

    options = optimoptions('lsqcurvefit', 'OptimalityTolerance', 1e-32, 'FunctionTolerance', 1e-32, 'Display', 'off'); %,'StepTolerance', 1e-20,...
    %'MaxFunctionEvaluations', 5000);%'MaxFunctionEvaluations', 10000, 'MaxIterations', 1000);

    % Set lower and upper bounds
    lb = [0, min(theseAmplitudes), 0, min(theseAmplitudes)];
    ub = [Inf, max(theseAmplitudes), range(theseAmplitudes), median(theseAmplitudes)];

    % Perform curve fitting with bounds
    fitOutput = lsqcurvefit(f, p0, ampliBin_gaussian, spikeCountsPerAmpliBin_gaussian, lb, ub, options);

    % Check output is sane - ish
    if any(fitOutput == lb | fitOutput == ub)
        %warning('percSpikesMissing:FitReachedBounds', ...
        %    'Gaussian fit reached bounds in time chunk %d. Results may be unreliable.', iTimeChunk);
    end


    %norm area calculated by fit parameters

    gaussianFit_cutoff = bc.qm.helpers.gaussian_cut(ampliBin_gaussian, fitOutput(1), fitOutput(2), fitOutput(3), fitOutput(4));
    norm_area_ndtr = normcdf((fitOutput(2) - fitOutput(4))/fitOutput(3));
    percentMissing_gaussian(iTimeChunk) = 100 * (1 - norm_area_ndtr);

    %% KS test
    % Create the empirical CDF
    [empirical_cdf, x] = ecdf(theseAmplitudes(theseSpikeTimes >= timeChunks(iTimeChunk) & ...
        theseSpikeTimes < timeChunks(iTimeChunk+1)));

    % Create the theoretical CDF based on the fitted Gaussian
    theoretical_cdf = normcdf(x, fitOutput(2), fitOutput(3));

    % Perform the KS test
    [~, ksTest_pValue(iTimeChunk), ~] = kstest2(empirical_cdf, theoretical_cdf);


end

roundedP = num2str(round(percentMissing_gaussian(iTimeChunk), 1));
if param.plotDetails
    subplot(2, numel(timeChunks)-1, numel(timeChunks)-1+iTimeChunk)
    if sum(spikeCountsPerAmpliBin_gaussian) > 0
        hold on;
        plot(gaussianFit_cutoff, ampliBin_gaussian, 'r');
        barh(ampliBin_gaussian, spikeCountsPerAmpliBin_gaussian, 'FaceColor', [0, 0.35, 0.71], 'EdgeColor', [0, 0.35, 0.71]);

        if iTimeChunk == 1
            xlabel('count')
            ylabel('amplitude')
        end
        if iTimeChunk > 1
            set(gca, 'Yticklabel', [])
        end
        if iTimeChunk == 1
            title({['% missing spikes: ', newline, roundedP]}, 'Color', [0.7, 0.7, 0.7])
        else
            title(roundedP, 'Color', [0.7, 0.7, 0.7]);
        end

    end

end




warning on;

if param.plotDetails
    if exist('prettify_plot', 'file')
        prettify_plot('FigureColor', 'w', 'XLimits', 'keep', 'YLimits', 'all')
    else
        warning('https://github.com/Julie-Fabre/prettify-matlab repo missing - download it and add it to your matlab path to make plots pretty')
        makepretty('none')
    end

    subplot(2, numel(timeChunks)-1, [1:numel(timeChunks) - 1])
    scatter(theseSpikeTimes, theseAmplitudes, 4, [0, 0.35, 0.71], 'filled');
    hold on;
    % chunk lines
    ylims = ylim;
    for iTimeChunk = 1:length(timeChunks)
        line([timeChunks(iTimeChunk), timeChunks(iTimeChunk)], ...
            [ylims(1), ylims(2)], 'Color', [0.7, 0.7, 0.7])
    end
    xlabel('time (s)')
    ylabel(['amplitude scaling', newline, 'factor'])
    if exist('prettify_plot', 'file')
        prettify_plot('FigureColor', 'w', 'XLimits', 'keep', 'YLimits', 'keep')
    else
        warning('https://github.com/Julie-Fabre/prettify-matlab repo missing - download it and add it to your matlab path to make plots pretty')
    end

end

function F = gaussian_cut(x, bin_centers)
    F = x(1) * exp(-(bin_centers - x(2)).^2/(2 * x(3).^2));
    F(bin_centers < x(4)) = 0;
end

end