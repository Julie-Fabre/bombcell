# Spike Width Analysis in MATLAB BombCell Code

## Summary of Findings

### 1. Where spike width is defined and used

#### Definition locations:
- **`+bc/+qm/qualityParamValues.m`** (lines 60-64):
  - Sets `param.spikeWidth = 61` for Kilosort 4
  - Sets `param.spikeWidth = 82` for Kilosort < 4
  
- **`+bc/+ep/ephysPropValues.m`** (line 64):
  - Sets `paramEP.spikeWidth = 82` (default, no Kilosort version check)

#### Usage locations:
- **`+bc/+qm/+helpers/extractRawWaveformsFast.m`**: Main file that uses spikeWidth for raw waveform extraction
- **`+bc/+qm/+helpers/loadRawWaveforms.m`**: Uses spikeWidth to initialize waveform arrays
- **`+bc/+qm/plotGlobalQualityMetric.m`**: Uses spikeWidth when template waveforms cannot be loaded
- **`+bc/+qm/+helpers/getWaveformPeakProperties.m`**: Checks waveform size (82 vs other) to adjust preprocessing

### 2. Current hardcoded checks

#### In `extractRawWaveformsFast.m` (lines 53-60):
```matlab
switch spikeWidth
    case 82
        % spikeWidth = 82: kilosort <4, baseline = 1:41
        halfWidth = spikeWidth / 2;
    case 61
        % spikeWidth = 61: kilosort 4, baseline = 1:20
        halfWidth = 20;
end
```

#### In `getWaveformPeakProperties.m` (lines 3-9):
```matlab
if size(thisWaveform, 2) == 82
    % < KS4 waveforms, remove the zero start values that can create
    % artificial peaks/troughs
    thisWaveform(1:24) = NaN;
else
    thisWaveform(1:4) = NaN;
end
```

#### In `qualityParamValues.m`:
- Lines 79-87: Different `waveformBaselineNoiseWindow` based on Kilosort version
- Lines 114-120: Different waveform baseline window start/stop based on Kilosort version

### 3. Spike width relationship to sampling rate

The spike width in samples corresponds to a fixed duration in milliseconds:
- **82 samples at 30kHz = 2.73ms** (Kilosort < 4)
- **61 samples at 30kHz = 2.03ms** (Kilosort 4)

The current implementation assumes a fixed 30kHz sampling rate (`param.ephys_sample_rate = 30000`).

### 4. Files that need modification for flexible spike width handling

1. **`+bc/+qm/qualityParamValues.m`**:
   - Remove hardcoded spike width values
   - Calculate spike width based on desired duration and actual sampling rate
   - Adjust baseline windows accordingly

2. **`+bc/+ep/ephysPropValues.m`**:
   - Similar changes as qualityParamValues.m

3. **`+bc/+qm/+helpers/extractRawWaveformsFast.m`**:
   - Replace switch statement with flexible calculation
   - Calculate halfWidth based on spike width percentage or fixed offset

4. **`+bc/+qm/+helpers/getWaveformPeakProperties.m`**:
   - Replace hardcoded size check (82) with flexible logic
   - Adjust NaN padding based on spike width percentage

5. **`+bc/+qm/+helpers/loadRawWaveforms.m`**:
   - Already uses spikeWidth parameter, no changes needed

6. **`+bc/+qm/plotGlobalQualityMetric.m`**:
   - Already uses param.spikeWidth, no changes needed

### 5. Recommended approach for flexible spike width

Instead of hardcoding spike widths, calculate them based on:
1. Desired waveform duration (e.g., 2.73ms or 2.03ms)
2. Actual sampling rate from metadata

Example calculation:
```matlab
% Define desired durations
if kilosortVersion == 4
    desired_duration_ms = 2.03;
else
    desired_duration_ms = 2.73;
end

% Calculate spike width from sampling rate
param.spikeWidth = round(desired_duration_ms * param.ephys_sample_rate / 1000);
```

This would make the code work correctly for any sampling rate, not just 30kHz.