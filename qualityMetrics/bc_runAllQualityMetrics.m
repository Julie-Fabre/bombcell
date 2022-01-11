function qMetric = bc_runAllQualityMetrics(param, spikeTimes, spikeTemplates, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx,plotThis)
% JF
% ------
% Inputs
% ------
% param: parameter structure with fields:
%   tauR = 0.0010; %refractory period time (s)
%   tauC = 0.0002; %censored period time (s)
%   maxPercSpikesMissing = 30;
%   minNumSpikes = 300;
%   maxNtroughsPeaks = 3;
%   axonal = 0; 
%   maxRPVviolations = 0.2;
%   minAmplitude = 70; 
%   plotThis = 1;
%   rawFolder = d1_rawfolder;
%   deltaTimeChunk = 600; % 10 min time chunk
%   ephys_sample_rate = 30000;
%   nChannels = 385;
%   nRawSpikesToExtract = 100; 
%   nChannelsIsoDist = 4;
% spikeTimes: nSpikes × 1 uint64 vector giving each spike time in samples (*not* seconds)
% spikeTemplates: nSpikes × 1 uint32 vector giving the identity of each
%   spike's matched template
% templateWaveforms: nTemplates × nTimePoints × nChannels single matrix of
%   template waveforms for each template and channel 
% templateAmplitudes: nSpikes × 1 double vector of the amplitude scaling factor 
%   that was applied to the template when extracting that spike
%
% ------
% Outputs
% ------
% qMetric: structure with fields:
%   percSpikesMissing
%   useTheseTimes
%   nSpikes
%   nPeaks
%   nTroughs
%   axonal
%   Fp
%   rawAmplitude
%   spatialDecay 
%   isoD
%   Lratio
%   silhouetteScore
%% get waveform max_channel and raw waveforms

maxChannels = bc_getWaveformMaxChannel(templateWaveforms);
rawWaveformFolder = dir(fullfile(param.rawFolder, 'rawWaveforms.mat'));

if isempty(rawWaveformFolder)
    
    rawWaveforms = bc_extractRawWaveformsFast(param.nChannels, param.nRawSpikesToExtract, spikeTimes, spikeTemplates, param.rawFolder, 1); % takes ~10'
    
else
    load(fullfile(param.rawFolder, 'rawWaveforms.mat'));
end

%%
qMetric = struct;
uniqueTemplates = unique(spikeTemplates);
spikeTimes = spikeTimes ./ param.ephys_sample_rate; %convert to seconds after using sample indices to extract raw waveforms
timeChunks = min(spikeTimes):param.deltaTimeChunk:max(spikeTimes);

for iUnit = 1:length(uniqueTemplates)
    clearvars thisUnit theseSpikeTimes theseAmplis 
    thisUnit = uniqueTemplates(iUnit);
    theseSpikeTimes = spikeTimes(spikeTemplates == thisUnit);
    theseAmplis = templateAmplitudes(spikeTemplates == thisUnit);
    
    % 10 min time chunks

    %% percentage spikes missing
    percSpikesMissing = bc_percSpikesMissing(theseAmplis, theseSpikeTimes, timeChunks, param.plotThis);
    
    %% define timechunks to keep
    if any(percSpikesMissing < param.maxPercSpikesMissing) % if there are some good time chunks, keep those
        
        useTheseTimes = find(percSpikesMissing < param.maxPercSpikesMissing);
        qMetric.percSpikesMissing(iUnit) = nanmean(percSpikesMissing(useTheseTimes(1):useTheseTimes(end)));
        theseAmplis = theseAmplis(theseSpikeTimes < timeChunks(useTheseTimes(end)+1) & ...
            theseSpikeTimes > timeChunks(useTheseTimes(1)));
        theseSpikeTimes = theseSpikeTimes(theseSpikeTimes < timeChunks(useTheseTimes(end)+1) & ...
            theseSpikeTimes > timeChunks(useTheseTimes(1)));
        %QQ change non continous
        timeChunks = timeChunks(useTheseTimes(1):useTheseTimes(end)+1);
    else %otherwise, keep all chunks to compute quality metrics on
        useTheseTimes = ones(numel(percSpikesMissing), 1);
        qMetric.percSpikesMissing(iUnit) = nanmean(percSpikesMissing);
    end
    qMetric.useTheseTimes(iUnit,:) = useTheseTimes;

    %% number spikes
    qMetric.nSpikes(iUnit) = bc_numberSpikes(theseSpikeTimes);

    %% waveform: number peaks/troughs and is peak before trough (= axonal)
    [qMetric.nPeaks(iUnit), qMetric.nTroughs(iUnit), qMetric.axonal(iUnit)] = bc_troughsPeaks(templateWaveforms(thisUnit, :, maxChannels(iUnit)), ...
        param.ephys_sample_rate, param.plotThis);

    %% fraction contam (false postives)
    [qMetric.Fp(iUnit), ~, ~] = bc_fractionRPviolations(numel(theseSpikeTimes), theseSpikeTimes, param.tauR, param.tauC, ...
        timeChunks(end)-timeChunks(1), param.plotThis);

    %% amplitude
    qMetric.rawAmplitude(iUnit) = bc_getRawAmplitude(rawWaveforms(thisUnit).spkMapMean(rawWaveforms(thisUnit).peakChan,:), ...
        param.rawFolder);
    
    %% distance metrics
    [qMetric.isoD(iUnit),qMetric.Lratio(iUnit), qMetric.silhouetteScore(iUnit),~, ~,~] = bc_getDistanceMetrics(pcFeatures, ...
    pcFeatureIdx, thisUnit, qMetric.nSpikes(iUnit), spikeTemplates==thisUnit, spikeTemplates, param.nChannelsIsoDist, param.plotThis);

end
end