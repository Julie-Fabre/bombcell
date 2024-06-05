function param_complete = bc_checkParameterFields(param)
% JF, Check input structure has all necessary fields + add them with
% default values if not. This is to ensure backcompatibility when any new
% paramaters are introduced. By default, any parameters not already present
% will be set so that the quality metrics are calculated in the same way as
% they were before these new parameters were introduced. (i.e. this does
% not change anything for the user!).
% ------
% Inputs
% ------
% - param 
% ------
% Outputs
% ------
% - param_complete 



%% Default values for fields
% duplicate spikes
defaultValues.removeDuplicateSpikes = 0;
defaultValues.duplicateSpikeWindow_s = 0.0001;
defaultValues.saveSpikes_withoutDuplicates = 1;
defaultValues.recomputeDuplicateSpikes = 0;

% raw waveforms 
defaultValues.detrendWaveform = 0;
defaultValues.extractRaw = 1;

% amplitude 
defaultValues.gain_to_uV = NaN;

% phy saving 
defaultValues.saveAsTSV = 0;
defaultValues.unitType_for_phy = 0;

% separate good from mua in non-somatic
defaultValues.splitGoodAndMua_NonSomatic = 0;

% waveforms: first peak (before trough) to second peak (after trough) ratio
defaultValues.firstPeakRatio = 0; % if units have an initial peak before the trough,
    % it must be at least firstPeakRatio times larger than the peak after
    % the trough to qualify as a non-somatic unit. 0 means this value is
    % not used.
defaultValues.normalizeSpDecay = 0;% whether to normalize spatial decay points relative to 
% maximum - this makes the spatrial decay slop calculation more invariant to the 
% spike-sorting algorithm used
%% Check for missing fields and add them with default value
[param_complete, missingFields] = bc_addMissingFieldsWithDefault(param, defaultValues);

%% Display result
if ~isempty(missingFields)
    disp('Missing param fields filled in with default values');
    disp(missingFields);
end

end
