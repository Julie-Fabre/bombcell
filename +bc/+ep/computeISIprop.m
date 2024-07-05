function [propLongISI, coefficient_variation, coefficient_variation2, isi_skewness] = computeISIprop(ISIs, theseSpikes)

% refs:
% Yamin/Cohen 2013
% Stalnaker/Schoenbaum 2016

% (for whole session right now, can modified in the future) 
spiking_stat_window = max(theseSpikes) - min(theseSpikes);
spiking_stat_bins = [min(theseSpikes), max(theseSpikes)];

% Get firing rate across the session
bin_spikes = ...
    histcounts(theseSpikes, ...
    spiking_stat_bins);

min_spikes = 10;
use_spiking_stat_bins = bsxfun(@ge, bin_spikes, prctile(bin_spikes, 80, 2)) & bin_spikes > min_spikes;

longISI = 2;
long_isi_total = 0;
%isi_ratios = [];
for curr_bin = find(use_spiking_stat_bins)
    curr_spike_times = theseSpikes( ...
        theseSpikes > spiking_stat_bins(curr_bin) & ...
        theseSpikes < spiking_stat_bins(curr_bin+1));
    curr_isi = diff(curr_spike_times);

    long_isi_total = long_isi_total + sum(curr_isi(curr_isi > longISI));
    
    % just compute on the first bin if there are several 
    if curr_bin==1
        % Coefficient of Variation (CV) of ISI
        coefficient_variation = std(curr_isi) / mean(curr_isi);
    
        % Coefficient of Variation 2 (CV2) of ISI
        coefficient_variation2 = 2 * mean(abs(diff(curr_isi))) / mean([curr_isi(1:end-1); curr_isi(2:end)]);
    
        % ISI Skewness
        isi_skewness = skewness(ISIs);
    end


    %isi_ratios = [isi_ratios; (2 * abs(curr_isi(2:end)-curr_isi(1:end-1))) ./ ...
    %    (curr_isi(2:end) + curr_isi(1:end-1))]; %WRONG, see Holt 1996
end
if isempty(curr_bin)
    coefficient_variation = NaN;
    
    % Coefficient of Variation 2 (CV2) of ISI
    coefficient_variation2 = NaN;
    
    % ISI Skewness
    isi_skewness = NaN;
end

propLongISI = long_isi_total / ...
    (sum(use_spiking_stat_bins(:)) * spiking_stat_window);


end