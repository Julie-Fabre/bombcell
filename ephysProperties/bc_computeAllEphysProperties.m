function ephysProperties = bc_computeAllEphysProperties(spikeTimes, spikeTemplates, templateWaveforms, param)

ephysProperties = struct;
uniqueTemplates = unique(spikeTemplates);
spikeTimes = spikeTimes ./ param.ephys_sample_rate; %convert to seconds after using sample indices to extract raw waveforms
timeChunks = min(spikeTimes):param.deltaTimeChunk:max(spikeTimes);
maxChannels = bc_getWaveformMaxChannel(templateWaveforms);
%% loop through units and get quality metrics
% QQ didvide in time chunks , add plotThis 

for iUnit = 1:length(uniqueTemplates)
    clearvars thisUnit theseSpikeTimes theseAmplis

    thisUnit = uniqueTemplates(iUnit);
    theseSpikeTimes = spikeTimes(spikeTemplates == thisUnit);

    %% compute ACG
    [acg, ~] = CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
        ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', param.ACGbinSize, 'duration', param.ACGduration, 'norm', 'rate'); %function
    ephysProperties.acg(iUnit, :) = acg(:, 1, 1);
    
    %% compute post spike suppression
    ephysProperties.postSpikeSupression(iUnit) = bc_computePSS(acg(:, 1, 1));

    %% compute template duration
    ephysProperties.templateDuration(iUnit) = bc_computeTemplateWaveformDuration(templateWaveforms(thisUnit, :, maxChannels(iUnit)),...
        param.ephys_sample_rate);
    
    %% compute firing rate
    ephysProperties.spike_rateSimple(iUnit) = bc_computeFR(theseSpikeTimes);

    %% compute proportion long ISIs
    ephysProperties.propLongISI(iUnit) = bc_computePropLongISI(theseSpikeTimes, param.longISI);

    %% cv, cv2

    %% Fano factor

    %% skewISI

    %% max firing rate

    %% bursting things
end

if param.plotThis
    % QQ plot histograms of each metric with the cutoffs set in params

end
end