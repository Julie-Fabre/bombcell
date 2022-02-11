
%% ~~Quality metrics used in Peters et al., 2021~~
% Inputs:
% -ephysData

qMetric = struct;
allT = unique(ephysData.spike_templates);

for iUnit = 1:size(allT, 1)
    thisUnit = allT(iUnit);
    theseSpikesIdx = ephysData.spike_templates == thisUnit;
    theseSpikes = ephysData.spike_times_timeline(theseSpikesIdx);
    theseAmplis = ephysData.template_amplitudes(theseSpikesIdx);

    %% Waveform peak-to-trough amplitude
    % parameters to load raw data
    raw = struct;
    raw.n_channels = 384; % number of channels
    raw.ephys_datatype = 'int16';
    raw.kp = strcat(ephys_path(1:end), '..');
    raw.dat = dir([raw.kp, filesep, '*.dat']);
    if isempty(raw.dat)
        raw.kp = strcat(ephys_path(1:end), '\..\experiment1\recording1\continuous\Neuropix-3a-100.0');
        raw.dat = dir([raw.kp, filesep, '*.dat']);
        if isempty(raw.dat)
            raw.kp = strcat(ephys_path(1:end), '\..');
            raw.dat = dir([raw.kp, filesep, '*.dat']);
        end
        %raw.dat = dir([raw.kp, filesep, '*.bin']);
    end

    raw.ephys_ap_filename = [raw.dat(1).folder, filesep, raw.dat(1).name];
    raw.ap_dat_dir = dir(raw.ephys_ap_filename);
    raw.pull_spikeT = -40:41; % number of points to pull for each waveform
    raw.microVoltscaling = 0.19499999284744263; %in structure.oebin for openephys, this never changed so hard-coded here-not loading it in.
    % for spike glx, V = i * Vmax / Imax / gain.
    % https://github.com/billkarsh/SpikeGLX/blob/gh-pages/Support/Metadata_20.md 

    raw.dataTypeNBytes = numel(typecast(cast(0, raw.ephys_datatype), 'uint8'));
    raw.n_samples = raw.ap_dat_dir.bytes / (raw.n_channels * raw.dataTypeNBytes);
    raw.ap_data = memmapfile(raw.ephys_ap_filename, 'Format', {raw.ephys_datatype, [raw.n_channels, raw.n_samples], 'data'});
    raw.max_pull_spikes = 200; % number of spikes to pull

    %get amplitude
    [qMetric.waveformRawAmpli(iUnit), qMetric.waveformRaw(iUnit,:), qMetric.thisChannelRaw(iUnit)] = getRawWaveformAmplitude(iUnit, ephysData, raw);

    %% Number of spikes
    qMetric.numSpikes(iUnit) = numel(theseSpikes);

    %% % spikes missing in xx chunks
    %super hacky - you have to cd to directory where gaussFit.py lives for
    %this to work. - change to where your file lives on your computer
    cd('C:\Users\Julie\Dropbox\MATLAB\bombcell\helpers\qualityMetricHelpers')
    try
        [percent_missing_ndtrAll, ~] = ampli_fit_prc_missJF(theseAmplis, 0);
    catch
        percent_missing_ndtrAll = NaN;
    end
    qMetric.pMissing(iUnit) = percent_missing_ndtrAll;

    %% waveform trough before peak ?
    waveformsTemp_mean = ephysData.template_waveforms(thisUnit, :);
    minProminence = 0.2 * max(abs(squeeze(waveformsTemp_mean)));

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

    if peakLoc > troughLoc
        qMetric.somatic(iUnit) = 1;
    else
        qMetric.somatic(iUnit) = 0;
    end

    %% false positives
    [qMetric.fractionRPVchunk(iUnit), qMetric.numRPVchunk(iUnit)] = fractionRPviolationsJF( ...
        numel(theseSpikes), theseSpikes, param.tauR, param.tauC,  max(ephysData.spike_times_timeline)- min(ephysData.spike_times_timeline)); %method from Hill et al., 2011

    %% ~~Additional quality metrics~~
end