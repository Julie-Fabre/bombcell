function percent_missing  = bc_percSpikesMissing(theseAmplitudes, theseSpikeTimes, timeChunks, plotThis)
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
% plotThis: boolean, whether to plot amplitude distribution and fit or not
% ------
% Outputs
% ------
% percent_missing: estimated percent spike missing for each time chunk.
% 
% Note that this will underestimate the amount of spikes missing in the
% case or bursty cells, where there spike amplitude is decreased during the
% burst. 

warning off;
percent_missing = nan(numel(timeChunks)-1,1);
if plotThis 
    figure();
    subplot(2,numel(timeChunks)-1, [1:numel(timeChunks)-1])
    scatter(theseSpikeTimes, theseAmplitudes, 4,'filled')
    xlabel('time (s)')
    ylabel('amplitude')
end
for iTimeChunk = 1:numel(timeChunks)-1
    % amplitude histogram
    nBins = 50;
    [num, bins] = histcounts(theseAmplitudes(theseSpikeTimes >= timeChunks(iTimeChunk) &...
        theseSpikeTimes < timeChunks(iTimeChunk+1)), nBins);

    %fit a gaussian to the histogram
    mode_seed = bins(num == max(num)); %guess mode - this is only accurate if mode is present in histogram
    bin_steps = diff(bins(1:2)); %size of a bin
    bin_centers = bins(1:end-1) + bin_steps / 2;
    next_low_bin = bin_centers(1) - bin_steps;
    add_points = next_low_bin:-bin_steps:0; %add points so amplitude values starts at 0
    bin_centers = [add_points, bin_centers];
    num = [zeros(size(add_points, 2), 1)', num];

    if numel(mode_seed) > 1 %if two or more modes, take the mean
        mode_seed = mean(mode_seed);
    end

    p0 = [max(num), mode_seed, 2 * nanstd(theseAmplitudes), prctile(theseAmplitudes, 1)]; % seed


    f = @(x, xdata)gaussian_cut(x, xdata); % get anonymous function handle
    
    options = optimoptions('lsqcurvefit','MaxFunctionEvaluations', 10000, 'MaxIterations', 1000);
    lb = [];
    ub = [];
    fitOutput = lsqcurvefit(f, p0, bin_centers, num,lb, ub, options);

    %norm area calculated by fit parameters
    norm_area_ndtr = normcdf((fitOutput(2) - fitOutput(4))/fitOutput(3)); %ndtr((popt[1] - min_amplitude) /popt[2])
    percent_missing(iTimeChunk) = 100 * (1 - norm_area_ndtr);

    if plotThis
        subplot(2,numel(timeChunks)-1, numel(timeChunks)-1+iTimeChunk)
        barh(bin_centers, num);
        hold on;
        n_fit_no_cut = JF_gaussian_cut(bin_centers, fitOutput(1), fitOutput(2), fitOutput(3), 0);
        plot(n_fit_no_cut, bin_centers, 'r');
        if iTimeChunk == 1
        xlabel('count')
        ylabel('amplitude')
        end
        TextLocation([num2str(percent_missing(iTimeChunk)), '% missing spikes'], 'Location', 'South');
        if iTimeChunk == numel(timeChunks)-1
        legend({'amplitude histogram', 'gaussian fit'})
        end
        makepretty;
    end

    

end
function F = gaussian_cut(x, bin_centers)
    F = x(1) * exp(-(bin_centers - x(2)).^2/(2 * x(3).^2));
    F(bin_centers < x(4)) = 0;
end

end