function maxChannels = bc_getWaveformMaxChannel(waveformTemplates)
% Get the waveform of all templates (channel with largest amplitude)
    [~, maxChannels] = max(max(abs(waveformTemplates), [], 2), [], 3);
%     templates_max = nan(size(waveformTemplates,1), size(waveformTemplates, 2));
%     for curr_template = 1:size(waveformTemplates,1)
%         templates_max(curr_template, :) = ...
%             waveformTemplates(curr_template, :, max_site(curr_template));
%     end
end