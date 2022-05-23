function maxChannels = bc_getWaveformMaxChannel(templateWaveforms)
% JF, Get the max channel for all templates (channel with largest amplitude)
% ------
% Inputs
% ------
% templateWaveforms: nTemplates × nTimePoints × nChannels single matrix of
%   template waveforms for each template and channel
% ------
% Outputs
% ------
% maxChannels: nTemplates * 1 vector of max channels for each template
% 
% templateWaveforms_baselineSub = spikeMapMean - mean(spikeMapMean(:, 1:10), 2); %subtract baseline from template 
% 
% templateWaveforms_baselineSub_smooth = smoothdata(spikeMapMean, 1, 'gaussian', 5);
%   
[~, maxChannels] = max(max(abs(templateWaveforms), [], 2), [], 3);
end