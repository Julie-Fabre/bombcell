function [qMetric, unitType] = bc_runAllQualityMetrics(param, spikeTimes_samples, spikeTemplates, ...
    templateWaveforms, templateAmplitudes, pcFeatures, pcFeatureIdx, goodChannels, savePath)
% JF, extract and save quality metrics for each unit 
% ------
% Inputs
% ------
% param: parameter structure (see bc_qualityParamValues for required fields
% spikeTimes_samples: nSpikes × 1 uint64 vector giving each spike time in samples (*not* seconds)
% spikeTemplates: nSpikes × 1 uint32 vector giving the identity of each
%   spike's matched template
% templateWaveforms: nTemplates × nTimePoints × nChannels single matrix of
%   template waveforms for each template and channel
% templateAmplitudes: nSpikes × 1 double vector of the amplitude scaling factor
%   that was applied to the template when extracting that spike
% pcFeatures: nSpikes × nFeaturesPerChannel × nPCFeatures  single
%   matrix giving the PC values for each spike
% pcFeatureIdx: nTemplates × nPCFeatures uint32  matrix specifying which
%   channels contribute to each entry in dim 3 of the pc_features matrix
% channelPositions: nChannels x 2 double matrix, each row gives the x and y 
%   coordinates of each channel
% goodChannels: nChannels x 1 uint32 vector defining the channels used by
%   kilosort (some are dropped during the spike sorting process)
% savePath: character array defining the path where bombcell output files
%   will be saved 
% ------
% Outputs
% ------
% qMetric: structure with fields:
%   percentageSpikesMissing : a gaussian is fit to the spike amplitudes with a
%       'cutoff' parameter below which there are no spikes to estimate the
%       percentage of spikes below the spike-sorting detection threshold - will
%       slightly underestimate in the case of 'bursty' cells with burst
%       adaptation (eg see Fig 5B of Harris/Buzsaki 2000 DOI: 10.1152/jn.2000.84.1.401) 
%   fractionRefractoryPeriodViolations: percentage of false positives, ie spikes within the refractory period
%       defined by param.tauR of another spike. This also excludes
%       duplicated spikes that occur within param.tauC of another spike. 
%   useTheseTimes : param.computeTimeChunks, this defines the time chunks 
%       (deivding the recording in time of chunks of param.deltaTimeChunk size)
%       where the percentage of spike missing and percentage of false positives
%       is below param.maxPercSpikesMissing and param.maxRPVviolations
%   nSpikes : number of spikes for each unit 
%   nPeaks : number of detected peaks in each units template waveform
%   nTroughs : number of detected troughs in each units template waveform
%   isSomatic : a unit is defined as Somatic of its trough precedes its main
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
%   (0), multi-unit (2) or non-somatic (3)

%% if some manual curation already performed, remove bad units

%% prepare for quality metrics computations
% initialize structures 
qMetric = struct;
forGUI = struct;

% get unit max channels
maxChannels = bc_getWaveformMaxChannel(templateWaveforms);
qMetric.maxChannels = maxChannels;

% get unique templates 
uniqueTemplates = unique(spikeTemplates);

% extract and save or load in raw waveforms 
[rawWaveformsFull, rawWaveformsPeakChan, signalToNoiseRatio] = bc_extractRawWaveformsFast(param, ...
    spikeTimes_samples, spikeTemplates, param.reextractRaw, savePath, param.verbose); % takes ~10' for 
% an average dataset, the first time it is run, <1min after that

% previous, slower method: 
% [qMetric.rawWaveforms, qMetric.rawMemMap] = bc_extractRawWaveforms(param.rawFolder, param.nChannels, param.nRawSpikesToExtract, ...
%     spikeTimes, spikeTemplates, usedChannels, verbose);

% divide recording into time chunks 
spikeTimes_seconds = spikeTimes_samples ./ param.ephys_sample_rate; %convert to seconds after using sample indices to extract raw waveforms
if param.computeTimeChunks
    timeChunks = [min(spikeTimes_seconds):param.deltaTimeChunk:max(spikeTimes_seconds), max(spikeTimes_seconds)];
else
    timeChunks = [min(spikeTimes_seconds), max(spikeTimes_seconds)];
end

%% loop through units and get quality metrics
fprintf('Extracting quality metrics from %s ... \n', param.rawFile)

for iUnit = 1:length(uniqueTemplates)
    
    clearvars thisUnit theseSpikeTimes theseAmplis theseSpikeTemplates

    % get this unit's attributes 
    thisUnit = uniqueTemplates(iUnit);
    qMetric.clusterID(iUnit) = thisUnit;
    theseSpikeTimes = spikeTimes_seconds(spikeTemplates == thisUnit);
    theseAmplis = templateAmplitudes(spikeTemplates == thisUnit);

    %% remove duplicate spikes 


    %% percentage spikes missing (false negatives)
    [percentageSpikesMissing_gaussian, percentageSpikesMissing_symmetric, ksTest_pValue, ~, ~, ~] = ...
        bc_percSpikesMissing(theseAmplis, theseSpikeTimes, timeChunks, param.plotDetails);

    %% fraction contamination (false positives)
    tauR_window = param.tauR_valuesMin:param.tauR_valuesStep:param.tauR_valuesMax;
    [fractionRPVs, ~, ~] = bc_fractionRPviolations(theseSpikeTimes, theseAmplis, ...
        tauR_window, param.tauC, ...
        timeChunks, param.plotDetails);
    
    %% define timechunks to keep: keep times with low percentage spikes missing and low fraction contamination
    if param.computeTimeChunks
        [theseSpikeTimes, theseAmplis, theseSpikeTemplates, qMetric.useTheseTimesStart(iUnit), qMetric.useTheseTimesStop(iUnit),...
            qMetric.RPV_tauR_estimate(iUnit)] = bc_defineTimechunksToKeep(...
            percentageSpikesMissing_gaussian, fractionRPVs, param.maxPercSpikesMissing, ...
            param.maxRPVviolations, theseAmplis, theseSpikeTimes, spikeTemplates, timeChunks); %QQ add kstest thing, symmetric ect 
    end

    %% re-compute percentage spikes missing and fraction contamination on timechunks
    thisUnits_timesToUse = [qMetric.useTheseTimesStart(iUnit), qMetric.useTheseTimesStop(iUnit)];
    [qMetric.percentageSpikesMissing_gaussian(iUnit), qMetric.percentageSpikesMissing_symmetric(iUnit), ...
        qMetric.ksTest_pValue(iUnit), forGUI.ampliBinCenters{iUnit}, forGUI.ampliBinCounts{iUnit}, ...
        forGUI.ampliGaussianFit{iUnit}] = bc_percSpikesMissing(theseAmplis, theseSpikeTimes, ...
        thisUnits_timesToUse, param.plotDetails);

    [qMetric.fractionRPVs(iUnit,:), ~, ~] = bc_fractionRPviolations(theseSpikeTimes, theseAmplis, ...
        tauR_window, param.tauC, thisUnits_timesToUse, param.plotDetails);

    %% presence ratio (potential false negatives)
    [qMetric.presenceRatio(iUnit)] = bc_presenceRatio(theseSpikeTimes, theseAmplis, param.presenceRatioBinSize, ...
        qMetric.useTheseTimesStart(iUnit), qMetric.useTheseTimesStop(iUnit), param.plotDetails);

    %% maximum cumulative drift estimate 
    [qMetric.maxDriftEstimate(iUnit),qMetric.cumDriftEstimate(iUnit)] = bc_maxDriftEstimate(pcFeatures, pcFeatureIdx, theseSpikeTemplates, ...
        theseSpikeTimes, goodChannels(:,2), thisUnit, param.driftBinSize, param.plotDetails);
    
    %% number spikes
    qMetric.nSpikes(iUnit) = bc_numberSpikes(theseSpikeTimes);

    %% waveform
    waveformBaselineWindow = [param.waveformBaselineWindowStart, param.waveformBaselineWindowStop];
    [qMetric.nPeaks(iUnit), qMetric.nTroughs(iUnit), qMetric.isSomatic(iUnit), forGUI.peakLocs{iUnit},...
        forGUI.troughLocs{iUnit}, qMetric.waveformDuration_peakTrough(iUnit), ...
        forGUI.spatialDecayPoints(iUnit,:), qMetric.spatialDecaySlope(iUnit), qMetric.waveformBaselineFlatness(iUnit), ....
        forGUI.tempWv(iUnit,:)] = bc_waveformShape(templateWaveforms,thisUnit, qMetric.maxChannels(thisUnit),...
        param.ephys_sample_rate, goodChannels, param.maxWvBaselineFraction, waveformBaselineWindow,...
        param.minThreshDetectPeaksTroughs, param.plotDetails); %QQ do we need tempWv ? 

    %% amplitude
    qMetric.rawAmplitude(iUnit) = bc_getRawAmplitude(rawWaveformsFull(iUnit,rawWaveformsPeakChan(iUnit),:), ...
        param.ephysMetaFile);

    %% distance metrics
    if param.computeDistanceMetrics
        [qMetric.isoD(iUnit), qMetric.Lratio(iUnit), qMetric.silhouetteScore(iUnit), ...
            forGUI.d2_mahal{iUnit}, forGUI.mahalobnis_Xplot{iUnit}, forGUI.mahalobnis_Yplot{iUnit}] = bc_getDistanceMetrics(pcFeatures, ...
            pcFeatureIdx, thisUnit, sum(spikeTemplates == thisUnit), spikeTemplates == thisUnit, theseSpikeTemplates, ...
            param.nChannelsIsoDist, param.plotDetails); %QQ make faster 
    end

end

%% get unit types and save data
qMetric.maxChannels = qMetric.maxChannels(uniqueTemplates)'; 
qMetric.signalToNoiseRatio = signalToNoiseRatio'; 


fprintf('Finished extracting quality metrics from %s \n', param.rawFile)
try
    qMetric = bc_saveQMetrics(param, qMetric, forGUI, savePath);
    fprintf('Saved quality metrics from %s to %s \n', param.rawFile, savePath)
catch
    warning on;
    warning('Warning, quality metrics from %s not saved \n', param.rawFile)
end

%% get some summary plots
unitType = bc_getQualityUnitType(param, qMetric);

bc_plotGlobalQualityMetric(qMetric, param, unitType, uniqueTemplates, forGUI.tempWv);



end
