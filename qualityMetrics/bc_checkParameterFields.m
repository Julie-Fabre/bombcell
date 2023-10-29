function param_complete = bc_checkParameterFields(param)
% JF, Check input structure has all necessary fields + add them with
% defualt values if not. This is to ensure backcompatibility when any new
% paramaters are introduced. By default, any parameters not already present
% will be set so that the quality metrics are calculated in the same way as
% they were before these new parameters were introduced.
% ------
% Inputs
% ------



%% Default values for fields
% duplicate spikes
defaultValues.removeDuplicateSpikes = 0;
defaultValues.duplicateSpikeWindow_s = 0.0001;
defaultValues.saveSpikes_withoutDuplicates = 1;
defaultValues.recomputeDuplicateSpikes = 0;

% raw waveforms 
defaultValues.detrendWaveforms = 0;
defaultValues.extractRaw = 1;

% amplitude 
defaultValues.gain_to_uV = NaN;

% phy saving 
defaultValues.saveAsTSV = 0;
defaultValues.unitType_for_phy = 0;


%% Check for missing fields and add them with default value
[param_complete, missingFields] = bc_addMissingFieldsWithDefault(param, defaultValues);

%% Display result
if ~isempty(missingFields)
    disp('Missing param fields filled in with default values');
    disp(missingFields);
end

end
