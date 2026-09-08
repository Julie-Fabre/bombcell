function param_complete = checkParameterFields(param)
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

%% Name changes
[~, param] = bc.qm.prettify_names([], param); % some names were changed for added clarity

%% Normalise to a struct
% loadSavedMetrics reads the param parquet as a 1-row table, and param is
% conceptually a struct everywhere downstream. Converting here also keeps
% addMissingFieldsWithDefault on its struct branch, whose table branch cannot
% represent an empty default (repmat([], 1, 1) stays 0-by-0 and the assignment
% fails, e.g. for contaminationValues).
if istable(param)
    param = table2struct(param(1, :));
end

%% Default values for fields
% duplicate spikes
defaultValues.removeDuplicateSpikes = 0;
defaultValues.duplicateSpikeWindow_s = 0.0001;
defaultValues.saveSpikes_withoutDuplicates = 1;
defaultValues.recomputeDuplicateSpikes = 0;

% raw waveforms 
defaultValues.detrendWaveform = 0;
defaultValues.extractRaw = 1;

defaultValues.computeSpatialDecay = 1;

% amplitude 
defaultValues.gain_to_uV = NaN;

% phy saving 
defaultValues.saveAsTSV = 0;
defaultValues.unitType_for_phy = 0;

% separate good from mua in non-somatic
defaultValues.splitGoodAndMua_NonSomatic = 0;

% refractory period violations
defaultValues.hillOrLlobetMethod = 1;  % legacy: 1=hill, 0=llobet
defaultValues.rpvMethod = 'hill';  % new: 'hill', 'llobet', or 'ibl_sliding'
defaultValues.tauR_values = 0.002;  % refractory period values to test (s)
defaultValues.contaminationValues = [];  % for ibl_sliding (default 0.5-35%)
defaultValues.confidenceThreshold = 0.9;  % for ibl_sliding

% waveform - noise
defaultValues.normalizeSpDecay = 0;% whether to normalize spatial decay points relative to 
% maximum - this makes the spatrial decay slop calculation more invariant to the 
% spike-sorting algorithm used
defaultValues.spDecayLinFit = 1;
defaultValues.minSpatialDecaySlopeExp = 0.01; % in a.u./um
defaultValues.maxSpatialDecaySlopeExp = 0.1; % in a.u./um
defaultValues.maxScndPeakToTroughRatio_noise = 0.8; % peak must be less than this x the trough 

% waveform - non-somatic
defaultValues.maxMainPeakToTroughRatio_nonSomatic = 0.8; % peak must be less than this x the trough 
defaultValues.minWidthFirstPeak_nonSomatic = 0; % in samples 
defaultValues.minWidthMainTrough_nonSomatic = 0; % in samples
defaultValues.minTroughToPeak2Ratio_nonSomatic = 0; % trough should be min 5 x bigger than 1rst peak to count as non-somatic 
defaultValues.maxPeak1ToPeak2Ratio_nonSomatic = Inf; % if units have an initial peak before the trough,
    % it must be at least this many times larger than the peak after the
    % trough to qualify as a non-somatic unit. Inf disables the test (no
    % ratio can exceed it); 0 would do the opposite and make it always pass.

%% Keep the non-somatic peak1/peak2 parameters coherent as a group
% The four parameters above are ANDed together in bc.qm.getQualityUnitType, so the
% back-compatibility values set above (0 widths, 0 ratio, Inf peak1/peak2) disable
% that test outright. That is what we want for a param set predating all four.
% It is not what we want when the user configured some of them: a param file using
% the old firstPeakRatio / minWidthFirstPeak / minWidthMainTrough names, for
% instance, is renamed by prettify_names above but still has no
% minTroughToPeak2Ratio_nonSomatic, and filling that one with 0 would silently
% switch off a test the user had deliberately set up. So when the group is only
% partly present, fill the gaps with the bc.qm.qualityParamValues defaults instead.
nonSomaticGroup = {'minTroughToPeak2Ratio_nonSomatic', 'minWidthFirstPeak_nonSomatic', ...
    'minWidthMainTrough_nonSomatic', 'maxPeak1ToPeak2Ratio_nonSomatic'};
groupIsSet = isfield(param, nonSomaticGroup);

if any(groupIsSet) && ~all(groupIsSet)
    % scale the width thresholds by spike width, as qualityParamValues does
    widthScale = 1;
    if all(isfield(param, {'spikeWidth', 'standardSpikeWidth'}))
        spikeWidth = param.spikeWidth;
        standardWidth = param.standardSpikeWidth;
        if isscalar(spikeWidth) && isscalar(standardWidth) && ...
                isfinite(spikeWidth) && isfinite(standardWidth) && standardWidth > 0
            widthScale = spikeWidth / standardWidth;
        end
    end
    defaultValues.minTroughToPeak2Ratio_nonSomatic = 5;
    defaultValues.minWidthFirstPeak_nonSomatic = max(2, round(4*widthScale));
    defaultValues.minWidthMainTrough_nonSomatic = max(3, round(5*widthScale));
    defaultValues.maxPeak1ToPeak2Ratio_nonSomatic = 3;

    warning('bombcell:partialNonSomaticParams', '%s', ...
        ['Some non-somatic peak1/peak2 parameters are set (', ...
        strjoin(nonSomaticGroup(groupIsSet), ', '), ') but others are missing (', ...
        strjoin(nonSomaticGroup(~groupIsSet), ', '), ').', newline, ...
        'Filling the missing ones with the current bombcell defaults rather than ', ...
        'with values that would disable the peak1/peak2 non-somatic test entirely.'])
end

%% Check for missing fields and add them with default value
[param_complete, missingFields] = bc.qm.addMissingFieldsWithDefault(param, defaultValues);

%% Display result
if ~isempty(missingFields)
    disp('Missing param fields filled in with default values');
    disp(missingFields);
end

end
