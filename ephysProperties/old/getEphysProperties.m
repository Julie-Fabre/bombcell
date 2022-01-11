
%% ~~ Ephys properties used in Peters et al., 2021 ~~
ephysParams = struct;
allT = unique(ephysData.spike_templates);

for iUnit = 1:size(allT, 1)
    thisUnit = allT(iUnit);
    theseSpikesIdx = ephysData.spike_templates == thisUnit;
    theseSpikes = ephysData.spike_times_timeline(theseSpikesIdx);
    theseAmplis = ephysData.template_amplitudes(theseSpikesIdx);

    %% postSpikeSuppression
    [ccg, t] = CCGBz([double(theseSpikes); double(theseSpikes)], [ones(size(theseSpikes, 1), 1); ...
        ones(size(theseSpikes, 1), 1) * 2], 'binSize', param.ACGbinSize, 'duration', param.ACGduration, 'norm', 'rate'); %function
    %from the Zugaro lab mod. by Buzsaki lab-way faster than my own!
    thisACG = ccg(:, 1, 1);
    ephysParams.ACG(iUnit, :) = thisACG;
    acgfr = find(ephysParams.ACG(iUnit, 500:1000) >= ...
        nanmean(ephysParams.ACG(iUnit, 600:900))); % nanmean(ephysParams.ACG(iUnit, 900:1000)) also works. 
    if ~isempty(acgfr)
        acgfr = acgfr(1);
    else
        acgfr = NaN;
    end
    ephysParams.postSpikeSuppression(iUnit) = acgfr;
    
%      figure(); 
%      plot(thisACG); hold on;
%      line([0 1000],[nanmean(ephysParams.ACG(iUnit, 900:1000)), nanmean(ephysParams.ACG(iUnit, 900:1000))])
     
    %% templateDuration
    waveformsTemp_mean = ephysData.template_waveforms(thisUnit, :);
    minProminence = 0.2 * max(abs(squeeze(waveformsTemp_mean)));
    qMetric.waveformUnit(iUnit, :) = squeeze(waveformsTemp_mean);
    %figure();plot(qMetric.waveform(iUnit, :))
    [PKS, LOCS] = findpeaks(squeeze(waveformsTemp_mean), 'MinPeakProminence', minProminence);
    [TRS, LOCST] = findpeaks(squeeze(waveformsTemp_mean)*-1, 'MinPeakProminence', minProminence);
    if isempty(TRS)
        TRS = min(squeeze(waveformsTemp_mean));
        if numel(TRS) > 1
            TRS = TRS(1);
        end
        LOCST = find(squeeze(waveformsTemp_mean) == TRS);
    end
    if isempty(PKS)
        PKS = max(squeeze(waveformsTemp_mean));
        if numel(PKS) > 1
            PKS = PKS(1);
        end
        LOCS = find(squeeze(waveformsTemp_mean) == PKS);
    end

    peakLoc = LOCS(PKS == max(PKS));
    if numel(peakLoc) > 1
        peakLoc = peakLoc(1);

    end
    troughLoc = LOCST(TRS == max(TRS));
    if numel(troughLoc) > 1
        troughLoc = troughLoc(1);
    end


    ephysParams.templateDuration(iUnit) = (peakLoc - troughLoc) / ephysData.ephys_sample_rate * 1e6;

    %% firing rate
    ephysParams.spike_rateSimple(iUnit) = numel(theseSpikes) / (max(theseSpikes) - min(theseSpikes));

    spiking_stat_window = max(theseSpikes) - min(theseSpikes);
    spiking_stat_bins = [min(theseSpikes), max(theseSpikes)];

    % Get firing rate across the session
    bin_spikes = ...
        histcounts(theseSpikes, ...
        spiking_stat_bins);

    min_spikes = 10;
    use_spiking_stat_bins = bsxfun(@ge, bin_spikes, prctile(bin_spikes, 80, 2)) & bin_spikes > min_spikes;
    spike_rate = sum(bin_spikes.*use_spiking_stat_bins, 2) ./ ...
        (sum(use_spiking_stat_bins, 2) * spiking_stat_window);
    ephysParams.spike_rateAP(iUnit) = spike_rate;

    %% proportion long isis
    long_isi_total = 0;
    %isi_ratios = [];
    for curr_bin = find(use_spiking_stat_bins)
        curr_spike_times = theseSpikes( ...
            theseSpikes > spiking_stat_bins(curr_bin) & ...
            theseSpikes < spiking_stat_bins(curr_bin+1));
        curr_isi = diff(curr_spike_times);

        long_isi_total = long_isi_total + sum(curr_isi(curr_isi > 2));

        %isi_ratios = [isi_ratios; (2 * abs(curr_isi(2:end)-curr_isi(1:end-1))) ./ ...
        %    (curr_isi(2:end) + curr_isi(1:end-1))]; %WRONG, see Holt 1996
    end

    ephysParams.prop_long_isi(iUnit) = long_isi_total / ...
        (sum(use_spiking_stat_bins(:)) * spiking_stat_window);

%     figure();
%     [counts, bins] = histcounts(curr_isi);
%     cdf = cumsum(counts);
%     plot(0.5*(bins(1:end-1)+bins(2:end)), counts);
%     xlabel('ISI (s)')
%     ylabel('# ISIs')
%     makepretty;
    %% ~~ Other ephys properties ~~
end