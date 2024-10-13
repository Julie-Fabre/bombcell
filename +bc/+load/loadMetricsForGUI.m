%get memmap

if exist('loadRawTraces', 'var') 
    if loadRawTraces 
        bc.viz.getRawMemMap;
    else
        memMapData =[];
    end
else
    loadRawTraces=0;
    memMapData =[];
end

if ~exist('param','var')
    param = paramBC;
end
% put ephys data into structure
ephysData = struct;
if exist('spikeTimes_samples','var')
    ephysData.spike_times_samples = spikeTimes_samples;
    ephysData.ephys_sample_rate = 30000;
    ephysData.spike_templates = spikeTemplates;
    ephysData.templates = templateWaveforms;
    ephysData.template_amplitudes = templateAmplitudes;
    ephysData.channel_positions = channelPositions;

else % load in unit match data 
    if iscell(sp)
        ephysData.spike_times_samples = sp{countid}.st(idx)*sp{countid}.sample_rate;
        ephysData.ephys_sample_rate = sp{countid}.sample_rate;
        ephysData.spike_templates = sp{countid}.spikeTemplates(idx)+1;
        ephysData.templates = sp{countid}.temps;
        ephysData.template_amplitudes = sp{countid}.tempScalingAmps(idx);
        ephysData.channel_positions = channelpostmp;
    else
        ephysData.spike_times_samples = sp.st(idx)*sp.sample_rate;
        ephysData.ephys_sample_rate = sp.sample_rate;
        ephysData.spike_templates = sp.spikeTemplates(idx)+1;
        ephysData.templates = sp.temps;
        ephysData.template_amplitudes = sp.tempScalingAmps(idx);
        ephysData.channel_positions = channelpostmp;
    end
end

ephysData.waveform_t = 1e3 * ((0:size(ephysData.templates, 2) - 1) / ephysData.ephys_sample_rate);
ephysParams = struct;
plotRaw = 1;
probeLocation = [];

% load raw waveforms 
if exist([fullfile(savePath, 'templates._bc_rawWaveforms.npy')], 'file')
    rawWaveforms.average = readNPY([fullfile(savePath, 'templates._bc_rawWaveforms.npy')]);
    rawWaveforms.peakChan = readNPY([fullfile(savePath, 'templates._bc_rawWaveformPeakChannels.npy')]);
else
    rawWaveforms.average = nan(max(ephysData.spike_templates),82,size(ephysData.templates,3));
    rawWaveforms.peakChan = ones(max(ephysData.spike_templates),1); 
end

% remove any duplicate spikes 
param.removeDuplicateSpikes =0; %disable for now, we don't need here. speed up :)
[uniqueTemplates, ~, ephysData.spike_times_samples, ephysData.spike_templates, ephysData.template_amplitudes, ...
    ~, rawWaveforms.average, rawWaveforms.peakChan, signalToNoiseRatio] = ...
    bc.qm.removeDuplicateSpikes(ephysData.spike_times_samples, ephysData.spike_templates, ephysData.template_amplitudes,...
    [], rawWaveforms.average, rawWaveforms.peakChan,[],...
    qMetric.maxChannels, param.removeDuplicateSpikes, param.duplicateSpikeWindow_s, ...
    param.ephys_sample_rate, param.saveSpikes_withoutDuplicates, savePath, param.recomputeDuplicateSpikes);

% convert spike times from samples to seconds 
ephysData.spike_times = ephysData.spike_times_samples ./ ephysData.ephys_sample_rate;

% load other gui stuffs 
if ~exist('forGUI', 'var') || ~isempty(dir([savePath, filesep, 'templates.qualityMetricDetailsforGUI.mat']))
    load([savePath, filesep, 'templates.qualityMetricDetailsforGUI.mat'])
else
    forGUI = struct;
     
    % get unique templates 
    uniqueTemplates = unique(spikeTemplates);
    
    % divide recording into time chunks 
    spikeTimes_seconds = spikeTimes_samples ./ param.ephys_sample_rate; %convert to seconds after using sample indices to extract raw waveforms
    if param.computeTimeChunks
        timeChunks = [min(spikeTimes_seconds):param.deltaTimeChunk:max(spikeTimes_seconds), max(spikeTimes_seconds)];
    else
        timeChunks = [min(spikeTimes_seconds), max(spikeTimes_seconds)];
    end
    
    %% loop through units and get quality metrics
    fprintf('Loading GUI data for %s ... \n', param.rawFile)
    
    for iUnit = 1:length(uniqueTemplates)
        
        clearvars thisUnit theseSpikeTimes theseAmplis theseSpikeTemplates
    
        % get this unit's attributes 
        thisUnit = uniqueTemplates(iUnit);
        theseSpikeTimes = spikeTimes_seconds(spikeTemplates == thisUnit);
        theseAmplis = templateAmplitudes(spikeTemplates == thisUnit);
    
        %% remove duplicate spikes 
    
        %% re-compute percentage spikes missing and fraction contamination on timechunks
        thisUnits_timesToUse = [qMetric.useTheseTimesStart(iUnit), qMetric.useTheseTimesStop(iUnit)];
        [~, ~, ...
            ~, forGUI.ampliBinCenters{iUnit}, forGUI.ampliBinCounts{iUnit}, ...
            forGUI.ampliGaussianFit{iUnit}] = bc.qm.percSpikesMissing(theseAmplis, theseSpikeTimes, ...
            thisUnits_timesToUse, param.plotDetails);
    
    
        %% waveform
    
        waveformBaselineWindow = [param.waveformBaselineWindowStart, param.waveformBaselineWindowStop];
        [~, ~, ~, forGUI.peakLocs{iUnit},...
            forGUI.troughLocs{iUnit}, ~, ...
            forGUI.spatialDecayPoints(iUnit,:), ~, ~, ....
            forGUI.tempWv(iUnit,:)] = bc.qm.waveformShape(templateWaveforms, thisUnit, qMetric.maxChannels(iUnit),...
            param.ephys_sample_rate, channelPositions, param.maxWvBaselineFraction, waveformBaselineWindow,...
            param.minThreshDetectPeaksTroughs, param.plotDetails); %do we need tempWv ? 
    
        %% distance metrics
        if param.computeDistanceMetrics
            [~, ~, ~, ...
                forGUI.d2_mahal{iUnit}, forGUI.mahalobnis_Xplot{iUnit}, forGUI.mahalobnis_Yplot{iUnit}] = bc.qm.getDistanceMetrics(pcFeatures, ...
                pcFeatureIdx, thisUnit, sum(spikeTemplates == thisUnit), spikeTemplates == thisUnit, theseSpikeTemplates, ...
                param.nChannelsIsoDist, param.plotDetails); %QQ
        end
    
    end


end
