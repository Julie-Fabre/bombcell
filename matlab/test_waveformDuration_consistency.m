% Test script to verify waveformDuration_peakTrough consistency
% between qualityMetrics and ephysProperties

fprintf('Testing waveformDuration_peakTrough consistency...\n\n');

% Load test data if available
testDataPath = fullfile(fileparts(which('bc.qm.runAllQualityMetrics')), '..', 'demos', 'demo_data');
if exist(testDataPath, 'dir')
    fprintf('Loading demo data from: %s\n', testDataPath);
    % Try to load demo data
    try
        load(fullfile(testDataPath, 'demo_data.mat'));
    catch
        fprintf('Could not load demo data, creating synthetic test cases...\n');
        % Create synthetic test cases
        createSyntheticTestCases = true;
    end
else
    fprintf('Demo data directory not found, creating synthetic test cases...\n');
    createSyntheticTestCases = true;
end

% Create synthetic test cases if needed
if exist('createSyntheticTestCases', 'var') && createSyntheticTestCases
    % Test case 1: Simple waveform with clear peak and trough
    t = 0:0.001:0.003; % 3ms window at 1kHz
    waveform1 = [zeros(1,10), 0.5*sin(2*pi*500*t), zeros(1,10)];
    
    % Test case 2: Complex waveform with multiple peaks
    waveform2 = [zeros(1,5), 0.3, 0.2, 0.1, -0.8, -0.6, 0.9, 0.7, 0.4, zeros(1,5)];
    
    % Test case 3: Trough-first waveform
    waveform3 = [zeros(1,5), -1.0, -0.8, -0.4, 0.2, 0.6, 0.5, 0.3, zeros(1,5)];
    
    templateWaveforms = zeros(3, 30, 1);
    templateWaveforms(1, :, 1) = waveform1(1:30);
    templateWaveforms(2, :, 1) = waveform2;
    templateWaveforms(3, :, 1) = waveform3;
    
    % Create necessary parameters
    param.ephys_sample_rate = 30000;
    param.minThreshDetectPeaksTroughs = 0.2;
    param.plotDetails = false;
    param.computeSpatialDecay = false;
    param.normalizeSpDecay = false;
    param.spDecayLinFit = true;
    
    paramEP = param; % Same parameters for ephysProperties
    
    channelPositions = [0, 0]; % Single channel for simplicity
    maxChannel = 1;
end

% Test each waveform
numUnits = size(templateWaveforms, 1);
results = struct();

for thisUnit = 1:numUnits
    fprintf('\nTesting unit %d...\n', thisUnit);
    
    % Call qualityMetrics version (waveformShape)
    waveformBaselineWindow = NaN;
    [nPeaks_qm, nTroughs_qm, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, peakLocs_qm, troughLocs_qm, ...
        waveformDuration_qm, ~, thisWaveform_qm] = bc.qm.waveformShape(templateWaveforms, ...
        thisUnit, maxChannel, param, channelPositions, waveformBaselineWindow);
    
    % Call ephysProperties version (computeWaveformProp)
    [waveformDuration_ep, ~, ~, ~, nPeaks_ep, nTroughs_ep] = ...
        bc.ep.computeWaveformProp(templateWaveforms, thisUnit, maxChannel, paramEP, channelPositions);
    
    % Store results
    results(thisUnit).unit = thisUnit;
    results(thisUnit).waveformDuration_qm = waveformDuration_qm;
    results(thisUnit).waveformDuration_ep = waveformDuration_ep;
    results(thisUnit).difference = abs(waveformDuration_qm - waveformDuration_ep);
    results(thisUnit).nPeaks_qm = nPeaks_qm;
    results(thisUnit).nPeaks_ep = nPeaks_ep;
    results(thisUnit).nTroughs_qm = nTroughs_qm;
    results(thisUnit).nTroughs_ep = nTroughs_ep;
    
    % Display results
    fprintf('  QualityMetrics duration: %.2f µs\n', waveformDuration_qm);
    fprintf('  EphysProperties duration: %.2f µs\n', waveformDuration_ep);
    fprintf('  Difference: %.6f µs\n', results(thisUnit).difference);
    fprintf('  Peaks (QM/EP): %d / %d\n', nPeaks_qm, nPeaks_ep);
    fprintf('  Troughs (QM/EP): %d / %d\n', nTroughs_qm, nTroughs_ep);
    
    if results(thisUnit).difference > 1e-6
        fprintf('  WARNING: Durations do not match!\n');
    else
        fprintf('  ✓ Durations match!\n');
    end
end

% Summary
fprintf('\n\nSUMMARY:\n');
fprintf('Total units tested: %d\n', numUnits);
maxDiff = max([results.difference]);
fprintf('Maximum difference: %.6f µs\n', maxDiff);

if maxDiff < 1e-6
    fprintf('\n✅ SUCCESS: All waveformDuration_peakTrough calculations are now consistent!\n');
else
    fprintf('\n❌ FAILURE: Some discrepancies remain. Please check the implementation.\n');
    % Show units with discrepancies
    discrepantUnits = find([results.difference] > 1e-6);
    if ~isempty(discrepantUnits)
        fprintf('\nUnits with discrepancies: %s\n', num2str(discrepantUnits));
    end
end