function [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes, spikeTemplates, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, channelPositions, savePath)
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
% spikeTimes: nSpikes × 1 uint64 vector giving each spike time in samples (*not* seconds)
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
% ------
% Outputs
% ------
% qMetric: structure with fields:
%   percSpikesMissing
%   useTheseTimes
%   nSpikes
%   nPeaks
%   nTroughs
%   somatic
%   Fp
%   rawAmplitude
%   spatialDecay
%   isoD
%   Lratio
%   silhouetteScore
%
% unitType: nUnits x 1 vector indicating whether each unit met the
%   threshold criterion to be classified as a single unit (1), noise
%   (0) or multi-unit (2)

%% if some manual curation already performed, remove bad units

%% prepare for quality metrics computations: get waveform max_channel and raw waveforms
qMetric = struct;
maxChannels = bc_getWaveformMaxChannel(templateWaveforms);
qMetric.maxChannels = maxChannels;

verbose = 1;
qMetric.rawWaveforms = bc_extractRawWaveformsFast(param.rawFolder, param.nChannels, param.nRawSpikesToExtract, ...
    spikeTimes, spikeTemplates, 1, verbose); % takes ~10' for an average dataset
% [qMetric.rawWaveforms, qMetric.rawMemMap] = bc_extractRawWaveforms(param.rawFolder, param.nChannels, param.nRawSpikesToExtract, ...
%     spikeTimes, spikeTemplates, usedChannels, verbose);

%% loop through units and get quality metrics

uniqueTemplates = unique(spikeTemplates);
spikeTimes = spikeTimes ./ param.ephys_sample_rate; %convert to seconds after using sample indices to extract raw waveforms
if param.computeTimeChunks
    timeChunks = min(spikeTimes):param.deltaTimeChunk:max(spikeTimes);
else
    timeChunks = [min(spikeTimes), max(spikeTimes)];
end

for iUnit = 1:length(uniqueTemplates)
    
    clearvars thisUnit theseSpikeTimes theseAmplis
    thisUnit = uniqueTemplates(iUnit);
    qMetric.clusterID(iUnit) = thisUnit;
    theseSpikeTimes = spikeTimes(spikeTemplates == thisUnit);
    theseAmplis = templateAmplitudes(spikeTemplates == thisUnit);

    %% percentage spikes missing
    [percSpikesMissing, qMetric.ampliBinCenters{iUnit}, qMetric.ampliBinCounts{iUnit}, qMetric.ampliFit{iUnit}] = ...
        bc_percSpikesMissing(theseAmplis, theseSpikeTimes, timeChunks, param.plotThis);

    %% define timechunks to keep
    if param.computeTimeChunks
        [qMetric.percSpikesMissing(iUnit), theseSpikeTimes, ~, timeChunks, qMetric.useTheseTimes{iUnit}] = bc_defineTimechunksToKeep(percSpikesMissing, ...
            param.maxPercSpikesMissing, theseAmplis, theseSpikeTimes, timeChunks);
    else
        qMetric.percSpikesMissing(iUnit) = percSpikesMissing;
    end

    %% number spikes
    qMetric.nSpikes(iUnit) = bc_numberSpikes(theseSpikeTimes);

    %% waveform: (1) number peaks/troughs, (2) is peak before trough (= axonal/dendritic), (3) is waveform duration cell-like, (4) spatial decay, (5) waveformShape
    [qMetric.nPeaks(iUnit), qMetric.nTroughs(iUnit), qMetric.somatic(iUnit), ...
        qMetric.peakLocs{iUnit}, qMetric.troughLocs{iUnit}, qMetric.waveformDuration(iUnit), ...
        qMetric.spatialDecayPoints(iUnit,:), qMetric.spatialDecaySlope(iUnit), qMetric.waveformBaseline(iUnit), qMetric.tempWv(iUnit,:)] = bc_waveformShape(templateWaveforms, thisUnit, maxChannels(thisUnit), ...
        param.ephys_sample_rate, channelPositions,  param.maxWvBaselineFraction, param.plotThis);

 
    %% fraction contam (false postives)
    [qMetric.Fp(iUnit), ~, ~] = bc_fractionRPviolations(numel(theseSpikeTimes), theseSpikeTimes, param.tauR, param.tauC, ...
        timeChunks(end)-timeChunks(1), param.plotThis);

    %% amplitude
    if size(qMetric.rawWaveforms(iUnit).spkMapMean, 1) == 1
        qMetric.rawWaveforms(iUnit).spkMapMean = permute(squeeze(qMetric.rawWaveforms(iUnit).spkMapMean), [2, 1]);
    end
    qMetric.rawAmplitude(iUnit) = bc_getRawAmplitude(qMetric.rawWaveforms(iUnit).spkMapMean(qMetric.rawWaveforms(iUnit).peakChan, :), ...
        param.rawFolder);

    %% distance metrics
    if param.computeDistanceMetrics
        [qMetric.isoD(iUnit), qMetric.Lratio(iUnit), qMetric.silhouetteScore(iUnit), ...
            qMetric.d2_mahal{iUnit}, qMetric.Xplot{iUnit}, qMetric.Yplot{iUnit}] = bc_getDistanceMetrics(pcFeatures, ...
            pcFeatureIdx, thisUnit, sum(spikeTemplates == thisUnit), spikeTemplates == thisUnit, spikeTemplates, param.nChannelsIsoDist, param.plotThis);
    end
end

bc_getQualityUnitType;

if exist('savePath', 'var') %save qualityMetrics
    mkdir(fullfile(savePath))
    disp(['saving quality metrics to ', savePath])
    save(fullfile(savePath, 'qMetric.mat'), 'qMetric', '-v7.3')
    save(fullfile(savePath, 'param.mat'), 'param')
end
disp('finished extracting quality metrics')

bc_plotGlobalQualityMetric;
end
