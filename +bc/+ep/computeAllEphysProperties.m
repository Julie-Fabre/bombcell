function ephysProperties = computeAllEphysProperties(spikeTimes_samples, spikeTemplates, templateWaveforms,...
     templateAmplitudes, pcFeatures, channelPositions, paramEP, savePath)

ephysProperties = struct;

% get unit max channels
maxChannels = bc.qm.helpers.getWaveformMaxChannel(templateWaveforms);

% extract and save or load in raw waveforms
[rawWaveformsFull, rawWaveformsPeakChan, signalToNoiseRatio] = bc.qm.helpers.extractRawWaveformsFast(paramEP, ...
    spikeTimes_samples, spikeTemplates, paramEP.reextractRaw, savePath, paramEP.verbose); % takes ~10' for
% an average dataset, the first time it is run, <1min after that

% remove any duplicate spikes
[uniqueTemplates, ~, spikeTimes_samples, spikeTemplates, ~, ~, ~, ~, ~, ...
    ephysProperties.maxChannels] = ...
    bc.qm.removeDuplicateSpikes(spikeTimes_samples, spikeTemplates, templateAmplitudes, ...
    pcFeatures, rawWaveformsFull, rawWaveformsPeakChan, signalToNoiseRatio, ...
    maxChannels, paramEP.removeDuplicateSpikes, paramEP.duplicateSpikeWindow_s, ...
    paramEP.ephys_sample_rate, paramEP.saveSpikes_withoutDuplicates, savePath, paramEP.recomputeDuplicateSpikes);

spikeTimes = spikeTimes_samples ./ paramEP.ephys_sample_rate; %convert to seconds after using sample indices to extract raw waveforms

% Work in progress - divide recording into time chunks like in quality  metrics
% spikeTimes_seconds = spikeTimes_samples ./ param.ephys_sample_rate; %convert to seconds after using sample indices to extract raw waveforms
% if param.computeTimeChunks
%     timeChunks = [min(spikeTimes_seconds):param.deltaTimeChunk:max(spikeTimes_seconds), max(spikeTimes_seconds)];
% else
%     timeChunks = [min(spikeTimes_seconds), max(spikeTimes_seconds)];
% end

fprintf('\n Extracting ephys properties ... ')
for iUnit = 1:length(uniqueTemplates)
    clearvars thisUnit theseSpikeTimes

    thisUnit = uniqueTemplates(iUnit);
    ephysProperties.phy_clusterID(iUnit) = thisUnit - 1; % this is the cluster ID as it appears in phy
    ephysProperties.clusterID(iUnit) = thisUnit; % this is the cluster ID as it appears in phy, 1-indexed (adding 1)
    theseSpikeTimes = spikeTimes(spikeTemplates == thisUnit);

    %% ACG-based properties  
    ephysProperties.acg(iUnit, :) = bc.ep.computeACG(theseSpikeTimes, paramEP.ACGbinSize, paramEP.ACGduration, paramEP.plotDetails);

    [ephysProperties.postSpikeSuppression_ms(iUnit), ephysProperties.tauRise_ms(iUnit), ephysProperties.tauDecay_ms(iUnit),...
        ephysProperties.refractoryPeriod_ms(iUnit)] = bc.ep.computeACGprop(ephysProperties.acg(iUnit, :), paramEP.ACGbinSize, paramEP.ACGduration);
    
    %% ISI-based properties
    ISIs = diff(spikeTimes);

    [ephysProperties.propLongISI(iUnit), ephysProperties.coefficient_variation(iUnit),...
         ephysProperties.coefficient_variation2(iUnit),  ephysProperties.isi_skewness(iUnit)] = bc.ep.computeISIprop(ISIs, theseSpikeTimes);

    %% Waveform-based properties
    % Work in progress: add option to use mean raw waveform 
    [ephysProperties.waveformDuration_peakTrough_us(iUnit), ephysProperties.halfWidth_ms(iUnit), ...
        ephysProperties.peakTroughRatio(iUnit), ephysProperties.firstPeakTroughRatio(iUnit),...
        ephysProperties.nPeaks(iUnit), ephysProperties.nTroughs(iUnit)] =...
        bc.ep.computeWaveformProp(templateWaveforms,thisUnit, ephysProperties.maxChannels(thisUnit),...
        paramEP, channelPositions);

    %% Burstiness properties
    % Work in progress

    %% Spike properties
    [ephysProperties.mean_firingRate(iUnit), ephysProperties.fanoFactor(iUnit),...
        ephysProperties.max_FiringRate(iUnit), ephysProperties.min_FiringRate(iUnit)] = bc.ep.computeSpikeProp(theseSpikeTimes);
    

    %% Progress info
    if ((mod(iUnit, 100) == 0) || iUnit == length(uniqueTemplates)) && paramEP.verbose
       fprintf(['\n   Finished ', num2str(iUnit), ' / ', num2str(length(uniqueTemplates)), ' units.']);
    end
end

%% save ephys properties
ephysProperties.maxChannels = ephysProperties.maxChannels(uniqueTemplates)';

fprintf('\n Finished extracting ephys properties')
try
    ephysProperties = bc.ep.saveEphysProperties(paramEP, ephysProperties, savePath);
    fprintf('\n Saved ephys properties to %s \n', savePath)
catch
    warning('\n Warning, ephys properties not saved! \n')
end

%% get some summary plots - work in progress 


end