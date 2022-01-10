% pipeline 
plotThis = 0;



%% params
param = struct;
param.tauR = 0.0010; %refractory period time (s)
param.tauC = 0.0002; %censored period time (s)
param.maxPercSpikesMissing = 30;
param.minNumSpikes = 300;
param.maxNtroughsPeaks = 3;
param.axonal = 0; 
param.maxRPVviolations = 0.2;
param.minAmplitude = 70; 
param.plotThis = 1;
ephys_sample_rate = 30000;
%% get waveform max_channel and raw waveforms 
iDay = 1; 
waveformTemplates =  waveforms{iDay};
maxChannels = bc_getWaveformMaxChannel(waveformTemplates);
rawWaveforms = dc_extractRawWaveformsFast(385, 200, spike_times{iDay}, spike_clusters{iDay}, d1_rawfolder, 1, 0);

%% 
iUnit = 1;
theseSpikeTimes = spike_times{iDay}(spike_templates{iDay} == iUnit);
theseAmplis = template_amplitude{iDay}(spike_templates{iDay} == iUnit);
timeChunks = min(spike_times{iDay}(spike_templates{iDay} == iUnit)):600:max(spike_times{iDay}(spike_templates{iDay} == iUnit));
% 10 min time chunks

%% percentage spikes missing 
[perc, ~] = bc_percSpikesMissing(theseAmplis, theseSpikeTimes, timeChunks, param.plotThis); 

%% define timechunks to keep
if any(perc < param.maxPercSpikesMissing) % if there are some good time chunks, keep those
    useTheseTimes = find(perc < param.maxPercSpikesMissing);
    theseAmplis = theseAmplis(theseSpikeTimes < timeChunks(useTheseTimes(end)+1) &...
        theseSpikeTimes > timeChunks(useTheseTimes(1)));
    theseSpikeTimes = theseSpikeTimes(theseSpikeTimes < timeChunks(useTheseTimes(end)+1) &...
        theseSpikeTimes > timeChunks(useTheseTimes(1)));
    %QQ change non continous 
    timeChunks = timeChunks(useTheseTimes(1):useTheseTimes(end)+1);
else %otherwise, keep all chunks to compute quality metrics on
    useTheseTimes = ones(numel(perc),1);
end

%% number spikes 
nSpikes = bc_numberSpikes(theseSpikeTimes); 

%% waveform: number peaks/troughs and is peak before trough (= axonal)
[nPeaks, nTroughs, axonal] = bc_troughsPeaks(waveformTemplates(iUnit, :,maxChannels(iUnit)), ephys_sample_rate, param.plotThis);

%% waveform spatial decay 

%% fraction contam
[Fp, r, overestimate] = bc_fractionRPviolations(numel(theseSpikes), theseSpikes, param.tauR, param.tauC, ...
    timeChunks(end)-timeChunks(1), param.plotThis);

%% amplitude


%% distance metrics 
