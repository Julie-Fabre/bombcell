% pipeline 
plotThis = 0;

iUnit = 1;
theseSpikeTimes = spike_times{iDay}(spike_templates{iDay} == iUnit);
theseAmplis = template_amplitude{iDay}(spike_templates{iDay} == iUnit);
timeChunks = min(spike_times{iDay}(spike_templates{iDay} == iUnit)):500:max(spike_times{iDay}(spike_templates{iDay} == iUnit));
%% percentage spikes missing 
[perc, ~] = bc_percSpikesMissing(theseAmplis, theseSpikeTimes, timeChunks, plotThis); 
%% define timechunks to keep
if any(perc < 30) % if there are some good time chunks, keep those
    useTheseTimes = perc < 30;
else %otherwise, keep all chunks to compute quality metrics on
    useTheseTimes = ones(numel(perc),1);
end
%% number spikes 
nSpikes = bc_numberSpikes(theseSpikeTimes); 

%% number peaks/troughs

%% peak before trough?

%% waveform spatial decay 

%% fraction contam

%% amplitude

%% distance metrics 
