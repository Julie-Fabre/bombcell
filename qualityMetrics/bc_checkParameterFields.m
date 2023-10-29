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
param.removeDuplicateSpikes = 0;
param.duplicateSpikeWindow_s = 0.000166;
param.saveSpikes_withoutDuplicates = 1;
param.recomputeDuplicateSpikes = 0;

% raw waveforms 
param.detrendWaveforms = 0;
param.extractRaw = 1;

% amplitude 
param.gain_to_uV = NaN;

% phy saving 
param.saveAsTSV = 0;
param.unitType_for_phy = 0;


%% Check for missing fields and add them with default value
[param_complete, missingFields] = bc_addMissingFieldsWithDefault(param, defaultValues);

%% Display result
disp('Missing param fields filled in with default values');
disp(missingFields);

end
