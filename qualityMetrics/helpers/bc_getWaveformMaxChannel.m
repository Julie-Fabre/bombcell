function maxChannels = bc_getWaveformMaxChannelEP(templateWaveforms)
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

 
[~, maxChannels] = max(max(abs(templateWaveforms), [], 2), [], 3);
end