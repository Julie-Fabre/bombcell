function ephysProperties = bc_computeAllEphysProperties(spikeTimes_samples, spikeTemplates, templateWaveforms, paramEP, savePath)

ephysProperties = struct;
uniqueTemplates = unique(spikeTemplates);
spikeTimes = spikeTimes_samples ./ paramEP.ephys_sample_rate; %convert to seconds after using sample indices to extract raw waveforms
%timeChunks = min(spikeTimes):param.deltaTimeChunk:max(spikeTimes);
maxChannels = bc_getWaveformMaxChannel(templateWaveforms);
%% loop through units and get ephys properties
% QQ didvide in time chunks , add plotThis 

fprintf('\n Extracting ephys properties ... ')

for iUnit = 1:length(uniqueTemplates)
    clearvars thisUnit theseSpikeTimes theseAmplis
    thisUnit = uniqueTemplates(iUnit);
    ephysProperties.clusterID(iUnit) = thisUnit;
    theseSpikeTimes = spikeTimes(spikeTemplates == thisUnit);

    %% compute ACG
    [acg, ~] = CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
        ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', paramEP.ACGbinSize, 'duration', paramEP.ACGduration, 'norm', 'rate'); %function
    ephysProperties.acg(iUnit, :) = acg(:, 1, 1);
    
    %% compute post spike suppression
    ephysProperties.postSpikeSupression(iUnit) = bc_computePSS(acg(:, 1, 1));

    %% compute template duration
    ephysProperties.templateDuration(iUnit) = bc_computeTemplateWaveformDuration(templateWaveforms(thisUnit, :, maxChannels(iUnit)),...
        paramEP.ephys_sample_rate);
    
    %% compute firing rate
    ephysProperties.spike_rateSimple(iUnit) = bc_computeFR(theseSpikeTimes);

    %% compute proportion long ISIs
    ephysProperties.propLongISI(iUnit) = bc_computePropLongISI(theseSpikeTimes, paramEP.longISI);

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
    fprintf('\n Saved ephys properties to %s', savePath)
    %% get some summary plots
    
catch
    warning('\n Warning, ephys properties not saved! \n')
end
%% plot
if paramEP.plotThis
    % QQ plot histograms of each metric with the cutoffs set in params
    figure();
    subplot(311)
    scatter(abs(ephysProperties.templateDuration), ephysProperties.postSpikeSupression);
    xlabel('waveform duration (us)')
    ylabel('post spike suppression')
    makepretty;
    
    subplot(312)
    scatter(ephysProperties.postSpikeSupression, ephysProperties.propLongISI);
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