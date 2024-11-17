function [mean_firingRate, fanoFactor, max_firingRate, min_firingRate] = computeSpikeProp(theseSpikes)
    spiking_stat_window = max(theseSpikes) - min(theseSpikes);
    spiking_stat_bins = [min(theseSpikes), max(theseSpikes)];

    % Get firing rate across the session
    bin_spikes = histcounts(theseSpikes, spiking_stat_bins);

    min_spikes = 10;
    use_spiking_stat_bins = bsxfun(@ge, bin_spikes, prctile(bin_spikes, 80, 2)) & bin_spikes > min_spikes;
    
    mean_firingRate = sum(bin_spikes.*use_spiking_stat_bins, 2) ./ ...
        (sum(use_spiking_stat_bins, 2) * spiking_stat_window);

    
    if length(theseSpikes) > 1 && max(diff(theseSpikes)) > 1
        % get spike counts 
        spikeCounts = histcounts(theseSpikes, min(theseSpikes):1:max(theseSpikes));
        
        % fano factor
        fanoFactor = var(spikeCounts) / mean(spikeCounts);
    else
        spikeCounts = 1;
        fanoFactor = NaN;
    end

    
    

    % max and min firing rate 
    max_firingRate = prctile(spikeCounts, 95);
    min_firingRate = prctile(spikeCounts, 5);

end

