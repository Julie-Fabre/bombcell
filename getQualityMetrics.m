%% ~~Quality metrics used in Peters et al., 2021~~
% Inputs: 
% -waveform
% -spikes
% -amplitude 
% -iUnit
qmetrics = struct; 

%% Waveform peak-to-trough duration
waveformsTemp_mean = ephysData.template_waveforms(thisUnit, :);
    qMetric.waveformsTemp_mean(iUnit, :) = waveformsTemp_mean;

    wavefTPeak = max(waveformsTemp_mean);
    if numel(wavefTPeak) > 1
        wavefTPeak = mean(wavefTPeak);
    end
    wavefTTrough = abs(min(waveformsTemp_mean));
    if numel(wavefTTrough) > 1
        wavefTTrough = mean(wavefTTrough);
    end
    qMetric.waveformTemplAmpli(iUnit) = wavefTPeak + ...
        wavefTTrough;
    
%% Number of spikes
qMetric.numSpikes(iUnit) = numel(theseSpikes);

%% % spikes missing 
try
    [percent_missing_ndtrAll, ~] = ampli_fit_prc_missJF(theseAmplis, 0);
catch
    percent_missing_ndtrAll = NaN;
end
qMetric.pMissing(iUnit) = percent_missing_ndtrAll;

%% waveform trough before peak ?

%% false positives 
[qMetric.fractionRPVchunk(iUnit), qMetric.numRPVchunk(iUnit)] = fractionRPviolationsJF( ...
    numel(theseSpikes), theseSpikes, param.tauR, param.tauC, timeChunks(end)-timeChunks(1)); %method from Hill et al., 2011

%% ~~Additional quality metrics~~