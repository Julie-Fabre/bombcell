function propLongISI = computePropLongISI(theseSpikes, longISI)
    
    spiking_stat_window = max(theseSpikes) - min(theseSpikes);
    spiking_stat_bins = [min(theseSpikes), max(theseSpikes)];

    % Get firing rate across the session
    bin_spikes = ...
        histcounts(theseSpikes, ...
        spiking_stat_bins);

    min_spikes = 10;
    use_spiking_stat_bins = bsxfun(@ge, bin_spikes, prctile(bin_spikes, 80, 2)) & bin_spikes > min_spikes;
    
    long_isi_total = 0;
    %isi_ratios = [];
    for curr_bin = find(use_spiking_stat_bins)
        curr_spike_times = theseSpikes( ...
            theseSpikes > spiking_stat_bins(curr_bin) & ...
            theseSpikes < spiking_stat_bins(curr_bin+1));
        curr_isi = diff(curr_spike_times);

        long_isi_total = long_isi_total + sum(curr_isi(curr_isi > longISI));

        %isi_ratios = [isi_ratios; (2 * abs(curr_isi(2:end)-curr_isi(1:end-1))) ./ ...
        %    (curr_isi(2:end) + curr_isi(1:end-1))]; %WRONG, see Holt 1996
    end

    propLongISI = long_isi_total / ...
        (sum(use_spiking_stat_bins(:)) * spiking_stat_window);
    
end
  