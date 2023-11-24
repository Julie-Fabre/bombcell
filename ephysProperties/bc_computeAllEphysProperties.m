function ephysProperties = bc_computeAllEphysProperties(spikeTimes_samples, spikeTemplates, templateWaveforms, paramEP, savePath)

ephysProperties = struct;
uniqueTemplates = unique(spikeTemplates);
spikeTimes = spikeTimes_samples ./ paramEP.ephys_sample_rate; %convert to seconds after using sample indices to extract raw waveforms
%timeChunks = min(spikeTimes):param.deltaTimeChunk:max(spikeTimes);
maxChannels = bc_getWaveformMaxChannelEP(templateWaveforms);
%% loop through units and get ephys properties
% QQ divide in time chunks , add plotThis 

fprintf('\n Extracting ephys properties ... ')

for iUnit = 1:length(uniqueTemplates)
    clearvars thisUnit theseSpikeTimes theseAmplis
    thisUnit = uniqueTemplates(iUnit);
    ephysProperties.clusterID(iUnit) = thisUnit;
    theseSpikeTimes = spikeTimes(spikeTemplates == thisUnit);

    %% ACG-based metrics  
    ephysProperties.acg(iUnit, :) = bc_computeACG(theseSpikeTimes, paramEP.ACGbinSize, paramEP.ACGduration, paramEP.plotThis);
    %units? -|> ms convert 
    [ephysProperties.postSpikeSuppression_ms(iUnit), ephysProperties.tauRise_ms(iUnit), ephysProperties.tauDecay_ms(iUnit),...
        ephysProperties.refractoryPeriod_ms(iUnit)] = bc_computeACGprop(ephysProperties.acg(iUnit, :), paramEP.ACGbinSize, paramEP.ACGduration);
    
    %% ISI-based metrics
     ISIs = diff(spikeTimes);

    % prop long isi 
    bc_computePropLongISI

    % Coefficient of Variation (CV) of ISI
    ISI_CV = std(ISIs) / mean(ISIs);

    % Coefficient of Variation 2 (CV2) of ISI
    ISI_CV2 = 2 * mean(abs(diff(ISIs))) / mean([ISIs(1:end-1); ISIs(2:end)]);

    % ISI Skewness
    ISI_Skewness = skewness(ISIs);

% Fano Factor for Spike Counts
% Assuming 'window' is the time window over which you want to compute Fano Factor
spikeCounts = histcounts(spikeTimes, 'BinWidth', window);
FanoFactor = var(spikeCounts) / mean(spikeCounts);


    %% Waveform-based metrics 
     ephysProperties.templateDuration(iUnit) = bc_computeTemplateWaveformDuration(templateWaveforms(thisUnit, :, maxChannels(iUnit)),...
        paramEP.ephys_sample_rate);
    
    %% Burstiness metrics 


    %% compute firing rate
    ephysProperties.spike_rateSimple(iUnit) = bc_computeFR(theseSpikeTimes);

 
    % Assuming 'waveform' is your waveform data vector
% And 'time' is a corresponding time vector

% Find the peak and trough
[peakAmplitude, peakIndex] = max(waveform);
[troughAmplitude, troughIndex] = min(waveform(peakIndex:end));
troughIndex = troughIndex + peakIndex - 1; % Adjust index

% Compute Peak-to-Trough Duration
peakToTroughDuration = time(troughIndex) - time(peakIndex);

% Compute Half-Width
halfAmplitude = peakAmplitude / 2;
aboveHalfIndices = find(waveform >= halfAmplitude);
halfWidthStartIndex = aboveHalfIndices(find(aboveHalfIndices < peakIndex, 1, 'last'));
halfWidthEndIndex = aboveHalfIndices(find(aboveHalfIndices > peakIndex, 1));
halfWidth = time(halfWidthEndIndex) - time(halfWidthStartIndex);

% Compute Rise Time
riseTime = time(peakIndex) - time(halfWidthStartIndex);

% Compute Decay Time
decayTime = time(halfWidthEndIndex) - time(peakIndex);

% Compute Rise Slope (Max Slope during the Rising Phase)
riseSlope = max(diff(waveform(halfWidthStartIndex:peakIndex)) ./ diff(time(halfWidthStartIndex:peakIndex)));

% Compute Decay Slope (Max Slope during the Falling Phase)
decaySlope = min(diff(waveform(peakIndex:halfWidthEndIndex)) ./ diff(time(peakIndex:halfWidthEndIndex)));




    %% cv, cv2

    %% Fano factor

    %% skewISI

    %% max firing rate

    %% bursting things
    if ((mod(iUnit, 100) == 0) || iUnit == length(uniqueTemplates)) && paramEP.verbose
       fprintf(['\n   Finished ', num2str(iUnit), ' / ', num2str(length(uniqueTemplates)), ' units.']);
    end
end

%% save ephys properties
fprintf('\n Finished extracting ephys properties')
try
    bc_saveEphysProperties(paramEP, ephysProperties, savePath);
    fprintf('\n Saved ephys properties to %s \n', savePath)
    %% get some summary plots
    
catch
    warning('\n Warning, ephys properties not saved! \n')
end
%% plot
paramEP.plotThis=0;
if paramEP.plotThis
    % QQ plot histograms of each metric with the cutoffs set in params
    figure();
    subplot(311)
    scatter(abs(ephysProperties.templateDuration), ephysProperties.postSpikeSuppression);
    xlabel('waveform duration (us)')
    ylabel('post spike suppression')
    makepretty;
    
    subplot(312)
    scatter(ephysProperties.postSpikeSuppression, ephysProperties.propLongISI);
    xlabel('post spike suppression')
    ylabel('prop long ISI')
    makepretty;

    subplot(313)
    scatter(abs(ephysProperties.templateDuration), ephysProperties.propLongISI);
    xlabel('waveform duration (us)')
    ylabel('prop long ISI')
    makepretty;
end


end