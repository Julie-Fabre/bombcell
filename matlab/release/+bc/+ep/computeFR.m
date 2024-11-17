function fr = computeFR(theseSpikes)
spiking_stat_window = max(theseSpikes) - min(theseSpikes);
    spiking_stat_bins = [min(theseSpikes), max(theseSpikes)];

    % Get firing rate across the session
    bin_spikes = ...
        histcounts(theseSpikes, ...
        spiking_stat_bins);

    min_spikes = 10;
    use_spiking_stat_bins = bsxfun(@ge, bin_spikes, prctile(bin_spikes, 80, 2)) & bin_spikes > min_spikes;
    fr = sum(bin_spikes.*use_spiking_stat_bins, 2) ./ ...
        (sum(use_spiking_stat_bins, 2) * spiking_stat_window);
end