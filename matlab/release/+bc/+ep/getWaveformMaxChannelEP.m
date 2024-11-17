function maxChannels = getWaveformMaxChannelEP(templateWaveforms)
% JF, Get the max channel for all templates (channel with largest amplitude)
% ------
% Inputs
% ------
% templateWaveforms: nTemplates × nTimePoints × nChannels single matrix of
%   template waveforms for each template and channel
% ------
% Outputs
% ------
% maxChannels: nTemplates × 1 vector of the channel with maximum amplitude
%   for each template 


% Get the waveform of all templates (channel with largest amplitude)
[~,maxChannels] = max(max(abs(templateWaveforms),[],2),[],3);


% figure();
% plot(squeeze(templateWaveforms_whitened(1,:,maxChannels(1))))
% hold on;
% plot(squeeze(templateWaveforms(1,:,maxChannels(1))))
end