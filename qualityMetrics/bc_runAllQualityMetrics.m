function [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions, goodChannels, savePath)
% JF
% ------
% Inputs
% ------
% param: parameter structure with fields:
%   tauR = 0.0010; %refractory period time (s)
%   tauC = 0.0002; %censored period time (s)
%   maxPercSpikesMissing: maximum percent (eg 30) of estimated spikes below detection
%       threshold to define timechunks in the recording on which to compute
%       quality metrics for each unit.
%   minNumSpikes: minimum number of spikes (eg 300) for unit to classify it as good
%   maxNtroughsPeaks: maximum number of troughs and peaks (eg 3) to classify unit
%       waveform as good
%   somatic: boolean, whether to keep only somatic spikes
%   maxRPVviolations: maximum estimated % (eg 20) of refractory period violations to classify unit as good
%   minAmplitude: minimum amplitude of raw waveform in microVolts to
%       classify unit as good
%   plotThis: boolean, whether to plot figures for each metric and unit - !
%       this will create * a lot * of plots if run on all units - use just
%       for debugging a particular issue / creating plots for one single
%       unit
%   rawFolder: string containing the location of the raw .dat or .bin file
%   deltaTimeChunk: size of time chunks to cut the recording in, in seconds
%       (eg 600 for 10 min time chunks or duration of recording if you don't
%       want time chunks)
%   ephys_sample_rate: recording sample rate (eg 30000)
%   nChannels: number of recorded channels, including any sync channels (eg
%       385)
%   nRawSpikesToExtract: number of spikes to extract from the raw data for
%       each waveform (eg 100)
%   nChannelsIsoDist: number of channels on which to compute the distance
%       metrics (eg 4)
%   computeDistanceMetrics: boolean, whether to compute distance metrics or not
%   isoDmin: minimum isolation distance to classify unit as single-unit
%   lratioMin: minimum l-ratio to classify unit as single-unit
%   ssMin: silhouette score to classify unit as single-unit
%   computeTimeChunks
%
% spikeTimes_samples: nSpikes × 1 uint64 vector giving each spike time in samples (*not* seconds)
%
% spikeTemplates: nSpikes × 1 uint32 vector giving the identity of each
%   spike's matched template
%
% templateWaveforms: nTemplates × nTimePoints × nChannels single matrix of
%   template waveforms for each template and channel
%
% templateAmplitudes: nSpikes × 1 double vector of the amplitude scaling factor
%   that was applied to the template when extracting that spike
%
% pcFeatures: nSpikes × nFeaturesPerChannel × nPCFeatures  single
%   matrix giving the PC values for each spike
%
% pcFeatureIdx: nTemplates × nPCFeatures uint32  matrix specifying which
%   channels contribute to each entry in dim 3 of the pc_features matrix
%
% channelPositions
% goodChannels 
%------
% Outputs
% ------
% qMetric: structure with fields:
%   percSpikesMissing : a gaussian is fit to the spike amplitudes with a
%       'cutoff' parameter below which there are no spikes to estimate the
%       percentage of spikes below the spike-sorting detection threshold - will
%       slightly underestimate in the case of 'bursty' cells with burst
%       adaptation (eg see Fig 5B of Harris/Buzsaki 2000 DOI: 10.1152/jn.2000.84.1.401) 
%   Fp: percentage of false positives, ie spikes within the refractory period
%       defined by param.tauR of anotehr spike. This also excludes
%       duplicated spikes that occur within param.tauC of another spike. 
%   useTheseTimes : param.computeTimeChunks, this defines the time chunks 
%       (deivding the recording in time of chunks of param.deltaTimeChunk size)
%       where the percentage of spike missing and percentage of false positives
%       is below param.maxPercSpikesMissing and param.maxRPVviolations
%   nSpikes : number of spikes for each unit 
%   nPeaks : number of detected peaks in each units template waveform
%   nTroughs : number of detected troughs in each units template waveform
%   somatic : a unit is defined as somatic of its trough precedes its main
%       peak (see Deligkaris/Frey DOI: 10.3389/fnins.2016.00421)
%   rawAmplitude : amplitude in uV of the units mean raw waveform at its peak
%       channel. The peak channel is defined by the template waveform. 
%   spatialDecay : gets the minumum amplitude for each unit 5 channels from
%       the peak channel and calculates the slope of this decrease in amplitude.
%   isoD : isolation distance, a measure of how well a units spikes are seperate from
%       other nearby units spikes
%   Lratio : l-ratio, a similar measure to isolation distance. see
%       Schmitzer-Torbert/Redish 2005  DOI: 10.1016/j.neuroscience.2004.09.066 
%       for a comparison of l-ratio/isolation distance
%   silhouetteScore : another measure similar ti isolation distance and
%       l-ratio. See Rousseeuw 1987 DOI: 10.1016/0377-0427(87)90125-7)
%
% unitType: nUnits x 1 vector indicating whether each unit met the
%   threshold criterion to be classified as a single unit (1), noise
%   (0) or multi-unit (2)

%% if some manual curation already performed, remove bad units

%% prepare for quality metrics computations: get waveform max_channel and raw waveforms
qMetric = struct;
maxChannels = bc_getWaveformMaxChannel(templateWaveforms);
qMetric.maxChannels = maxChannels;


verbose = 1; % update user on progress
reextract = 0; %Re extract raw waveforms
% QQ extract raw waveforms based on 'good' timechunks defined later ? 

[rawWaveformsFull, rawWaveformsPeakChan]= bc_extractRawWaveformsFast(param, ...
    spikeTimes_samples, spikeTemplates, reextract , verbose); 

% takes ~10' for an average dataset
% previous, slower method: 
% [qMetric.rawWaveforms, qMetric.rawMemMap] = bc_extractRawWaveforms(param.rawFolder, param.nChannels, param.nRawSpikesToExtract, ...
%     spikeTimes, spikeTemplates, usedChannels, verbose);

%% loop through units and get quality metrics

uniqueTemplates = unique(spikeTemplates);
spikeTimes_seconds = spikeTimes_samples ./ param.ephys_sample_rate; %convert to seconds after using sample indices to extract raw waveforms
if param.computeTimeChunks
    timeChunks = [min(spikeTimes_seconds):param.deltaTimeChunk:max(spikeTimes_seconds), max(spikeTimes_seconds)];
else
    timeChunks = [min(spikeTimes_seconds), max(spikeTimes_seconds)];
end

disp([newline, 'extracting quality metrics ...'])

for iUnit = 1:length(uniqueTemplates)
    
    clearvars thisUnit theseSpikeTimes theseAmplis
    thisUnit = uniqueTemplates(iUnit);
    qMetric.clusterID(iUnit) = thisUnit;
    theseSpikeTimes = spikeTimes_seconds(spikeTemplates == thisUnit);
    theseAmplis = templateAmplitudes(spikeTemplates == thisUnit);

    %% percentage spikes missing (false negatives)
    [qMetric.percSpikesMissing(iUnit,:), qMetric.ampliBinCenters{iUnit}, qMetric.ampliBinCounts{iUnit}, qMetric.ampliFit{iUnit}] = ...
        bc_percSpikesMissing(theseAmplis, theseSpikeTimes, timeChunks, param.plotThis);

    %% fraction contamination (false positives)
    [qMetric.Fp(iUnit,:), ~, ~] = bc_fractionRPviolations(numel(theseSpikeTimes), theseSpikeTimes, theseAmplis, param.tauR, param.tauC, ...
        timeChunks, param.plotThis);
    
    %% define timechunks to keep: if param.computeTimeChunks, keep times with low percentage spikes missing and low fraction contam
    if param.computeTimeChunks
        [theseSpikeTimes, ~, ~, qMetric.useTheseTimes{iUnit}] = bc_defineTimechunksToKeep(qMetric.percSpikesMissing(iUnit,:), ...
            qMetric.Fp(iUnit,:), param.maxPercSpikesMissing, param.maxRPVviolations, theseAmplis, theseSpikeTimes, timeChunks);
    end
    
    %% number spikes
    qMetric.nSpikes(iUnit) = bc_numberSpikes(theseSpikeTimes);

    %% waveform: (1) number peaks/troughs, (2) is peak before trough (= axonal/dendritic), (3) is waveform duration cell-like, (4) spatial decay, (5) waveformShape
    [qMetric.nPeaks(iUnit), qMetric.nTroughs(iUnit), qMetric.somatic(iUnit), ...
        qMetric.peakLocs{iUnit}, qMetric.troughLocs{iUnit}, qMetric.waveformDuration(iUnit), ...
        qMetric.spatialDecayPoints(iUnit,:), qMetric.spatialDecaySlope(iUnit), qMetric.waveformBaseline(iUnit), qMetric.tempWv(iUnit,:)] = bc_waveformShape(templateWaveforms, ...
        thisUnit, maxChannels(thisUnit), ...
        param.ephys_sample_rate, channelPositions,  param.maxWvBaselineFraction, param.plotThis);

    %% amplitude
%     if size(rawWaveformsPeakChan(iUnit,:), 1) == 1
%         qMetric.rawWaveforms(iUnit).spkMapMean = permute(squeeze(qMetric.rawWaveforms(iUnit).spkMapMean), [2, 1]);
%     end
    qMetric.rawAmplitude(iUnit) = bc_getRawAmplitude(rawWaveformsFull(iUnit,rawWaveformsPeakChan(iUnit,:),:), ...
        param.rawFolder);

    %% distance metrics
    if param.computeDistanceMetrics
        [qMetric.isoD(iUnit), qMetric.Lratio(iUnit), qMetric.silhouetteScore(iUnit), ...
            qMetric.d2_mahal{iUnit}, qMetric.Xplot{iUnit}, qMetric.Yplot{iUnit}] = bc_getDistanceMetrics(pcFeatures, ...
            pcFeatureIdx, thisUnit, sum(spikeTemplates == thisUnit), spikeTemplates == thisUnit, spikeTemplates, param.nChannelsIsoDist, param.plotThis);
    end

    %% Save some raw waveforms for the GUI
    qMetric.rawWaveforms(iUnit).spkMapMean = squeeze(rawWaveformsFull(iUnit,:,:));
end

bc_getQualityUnitType;

if exist('savePath', 'var') %save qualityMetrics
    bc_saveQMetrics;
end
disp([newline, 'finished extracting quality metrics'])

bc_plotGlobalQualityMetric;
end
