function ephysProp = bc_computeAllEphysProperties(spikeTimes, spikeTemplates, templateWaveforms, param)
ephysProp = struct;
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
    [ccg, ~] = CCGBz([double(theseSpikeTimes); double(theseSpikeTimes)], [ones(size(theseSpikeTimes, 1), 1); ...
        ones(size(theseSpikeTimes, 1), 1) * 2], 'binSize', param.ACGbinSize, 'duration', param.ACGduration, 'norm', 'rate'); %function
    ephysProp.acg(iUnit, :) = ccg(:, 1, 1);
    
    %% compute post spike suppression
    ephysProp.pss(iUnit) = bc_computePSS(ccg(:, 1, 1));

    %% compute template duration
    ephysProp.templateDuration(iUnit) = bc_computeTemplateWaveformDuration(templateWaveforms(thisUnit, :, maxChannels(iUnit)),...
        param.ephys_sample_rate);
    
    %% compute firing rate
    ephysProp.spike_rateSimple(iUnit) = bc_computeFR(theseSpikeTimes);

    %% compute proportion long ISIs
    ephysProp.propLongISI(iUnit) = bc_computePropLongISI(theseSpikeTimes, param.longISI);

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