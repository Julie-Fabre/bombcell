function [waveformRawAmpli, waveformRaw, thisChannelRaw] = getRawWaveformAmplitude2(iUnit, ephysData, raw)
allT = unique(spikeTemplates);
thisRawUnit = allT(iUnit);
curr_template = thisRawUnit;
%ns = find(ephysData.new_spike_idx == curr_template);
%raw.used_channels_idx = ephysData.channel_map + 1;
spkFile = dir(fullfile(param.rawFolder, '*.ap.bin'));
fname = spkFile.name
d = dir(fullfile(param.rawFolder, fname));
fid = fopen(fullfile(param.rawFolder, fname), 'r');
curr_spikes_idx = find(spikeTemplates == thisRawUnit);
if ~isempty(curr_spikes_idx)
    curr_pull_spikes = unique(round(linspace(1, length(curr_spikes_idx), 200)));
    if curr_pull_spikes(1) == 0
        curr_pull_spikes(1) = [];
    end
    curr_spikeT = spikeTimes(curr_spikes_idx(curr_pull_spikes));
    pull_spikeT = -40:41;
    curr_spikeT_pull = double(curr_spikeT) + pull_spikeT;

    out_of_bounds_spikes = any(curr_spikeT_pull < 1, 2) | ...
        any(curr_spikeT_pull > d.bytes, 2);
    curr_spikeT_pull(out_of_bounds_spikes, :) = [];

   % spki = spikeIndicessub(iSpike);
  %spki = spikeIndicessub(iSpike);
        bytei = ((curr_spikeT(1) - 40) * 385) ;
        fseek(fid, bytei, 'bof');
        data0 = fread(fid, 385*82, 'int16=>int16'); % read individual waveform from binary file
        frewind(fid);
        data = reshape(data0, 385, []);
        
    curr_spike_waveforms = reshape(raw.ap_data.data.data(:, reshape(curr_spikeT_pull', [], 1)), raw.n_channels, length(raw.pull_spikeT), []);
    if ~isempty(curr_spike_waveforms)
        curr_spike_waveforms_car = curr_spike_waveforms - nanmedian(curr_spike_waveforms, 1);
        curr_spike_waveforms_car_sub = curr_spike_waveforms_car - curr_spike_waveforms_car(:, 1, :);

        waveforms_mean(curr_template, :, :) = ...
            permute(nanmean(curr_spike_waveforms_car_sub(raw.used_channels_idx, :, :), 3), [3, 2, 1]) * raw.microVoltscaling;


        thisChannelRaw = find(squeeze(max(waveforms_mean(curr_template, :, :))) == ...
            max(squeeze(max(waveforms_mean(curr_template, :, :)))));
        if numel(thisChannelRaw) > 1
            thisOne = find(max(abs(waveforms_mean(curr_template, :, thisChannelRaw))) == max(max(abs(waveforms_mean(curr_template, :, thisChannelRaw)))));
            if numel(thisOne) > 1
                thisOne = thisOne(1);
            end
            thisChannelRaw = thisChannelRaw(thisOne);
        end

        wavefPeak = max(waveforms_mean(curr_template, :, thisChannelRaw));
        if numel(wavefPeak) > 1
            wavefPeak = mean(wavefPeak);
        end
        wavefTrough = abs(min(waveforms_mean(curr_template, :, thisChannelRaw)));
        if numel(wavefTrough) > 1
            wavefTrough = mean(wavefTrough);
        end
        waveformRawAmpli = wavefPeak + ...
            wavefTrough;
        waveformRaw = squeeze(waveforms_mean(curr_template, :, thisChannelRaw));

    else
        waveformRawAmpli = NaN;
        waveformRaw = nan(1,82);
        
    end
else
    waveformRawAmpli = NaN;
    waveformRaw = nan(1,82);


end
end