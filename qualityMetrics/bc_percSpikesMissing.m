function [percent_missing, bin_centers, num, n_fit_cut]  = bc_percSpikesMissing(theseAmplitudes, theseSpikeTimes, timeChunks, plotThis)
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
% percent_missing: estimated percent spike missing for each time chunk
% bin_centers
% n_fit_cut
% 
% Note that this will underestimate the amount of spikes missing in the
% case or bursty cells, where there spike amplitude is decreased during the
% burst. 

warning off;
percent_missing = nan(numel(timeChunks)-1,1);
ylims = [0,0]; % initialize varibale for plotting 
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
    makepretty('none');
    
end
for iTimeChunk = 1:numel(timeChunks)-1
    % amplitude histogram
    nBins = 50;
    [num, bins] = histcounts(theseAmplitudes(theseSpikeTimes >= timeChunks(iTimeChunk) &...
        theseSpikeTimes < timeChunks(iTimeChunk+1)), nBins);
    if sum(num) > 0 
    %fit a gaussian to the histogram
    mode_seed = bins(num == max(num)); %guess mode - this is only accurate if mode is present in histogram
    bin_steps = diff(bins(1:2)); %size of a bin
    bin_centers = bins(1:end-1) + bin_steps / 2;
    next_low_bin = bin_centers(1) - bin_steps;
    add_points = 0:bin_steps:next_low_bin; %add points so amplitude values starts at 0
    bin_centers = [add_points, bin_centers];
    num = [zeros(size(add_points, 2), 1)', num];

    if numel(mode_seed) > 1 %if two or more modes, take the mean
        mode_seed = mean(mode_seed);
    end

    p0 = [max(num), mode_seed, 2 * nanstd(theseAmplitudes), prctile(theseAmplitudes, 1)]; % seed


    f = @(x, xdata)gaussian_cut(x, xdata); % get anonymous function handle
    
    options = optimoptions('lsqcurvefit','OptimalityTolerance', 1e-32, 'FunctionTolerance', 1e-32,'Display','off');%,'StepTolerance', 1e-20,...
        %'MaxFunctionEvaluations', 5000);%'MaxFunctionEvaluations', 10000, 'MaxIterations', 1000);
    lb = [];
    ub = [];
    fitOutput = lsqcurvefit(f, p0, bin_centers, num,lb, ub, options); %QQ need to fix local minimum error

    %norm area calculated by fit parameters
    
    n_fit_cut = JF_gaussian_cut(bin_centers, fitOutput(1), fitOutput(2), fitOutput(3), fitOutput(4));
%    n_fit_no_cut = JF_gaussian_cut(bin_centers, fitOutput(1), fitOutput(2), fitOutput(3), 0);
    norm_area_ndtr = normcdf((fitOutput(2) - fitOutput(4))/fitOutput(3)); %ndtr((popt[1] - min_amplitude) /popt[2])
    percent_missing(iTimeChunk) = 100 * (1 - norm_area_ndtr);
    else
        percent_missing(iTimeChunk) = 100;
    end
        roundedP = num2str(round(percent_missing(iTimeChunk),1));
    if plotThis
        subplot(2,numel(timeChunks)-1, numel(timeChunks)-1+iTimeChunk)
        if sum(num) > 0 
        hold on;
        plot(n_fit_cut, bin_centers, 'r');
        barh(bin_centers, num, 'FaceColor',[0, 0.35, 0.71], 'EdgeColor',[0, 0.35, 0.71]);
        
        if iTimeChunk == 1
            xlabel('count')
            ylabel('amplitude')
        end
        if iTimeChunk == numel(timeChunks)-1
            legend({['gaussian fit:' roundedP, '% missing spikes'], 'amplitude histogram'},'TextColor', [0.7, 0.7, 0.7], 'Color', 'none')
        else
            legend([roundedP, '% missing spikes'], 'Location', 'NorthEast','TextColor', [0.7, 0.7, 0.7], 'Color', 'none');
        end

        end
        makepretty('none');
    end

    

end
function F = gaussian_cut(x, bin_centers)
    F = x(1) * exp(-(bin_centers - x(2)).^2/(2 * x(3).^2));
    F(bin_centers < x(4)) = 0;
end

end