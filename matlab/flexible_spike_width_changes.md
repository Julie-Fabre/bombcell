# Flexible Spike Width Implementation for BombCell

## Summary
BombCell now supports flexible spike widths based on actual sampling rates, removing the hardcoded 61/82 sample restrictions. This allows proper processing of data collected at non-standard sampling rates (e.g., 32kHz).

## Key Changes

### 1. New Helper Functions
- `calculateSpikeWidth.m`: Calculates appropriate spike width based on sampling rate and Kilosort version
- `calculateHalfWidth.m`: Calculates the half-width (samples before/after peak) based on spike width

### 2. Updated Parameter Files
- `qualityParamValues.m`: 
  - Automatically detects sampling rate from metadata
  - Calculates spike width dynamically
  - Scales baseline windows proportionally
  
- `ephysPropValues.m`:
  - Similar updates for ephys properties calculations
  - Auto-detects Kilosort version from template size

### 3. Updated Processing Functions
- `extractRawWaveformsFast.m`: Removed hardcoded switch statement, uses flexible half-width calculation
- `getWaveformPeakProperties.m`: Scales initial sample removal based on waveform width
- `plotGlobalQualityMetric.m`: Flexible x-axis limits based on spike width

## How It Works

### For 32kHz Data (User's Case)
- Kilosort 4 at 32kHz: 65 samples (maintains ~2.03ms window)
- Kilosort <4 at 32kHz: 87 samples (maintains ~2.73ms window)

### Automatic Scaling
All parameters that were previously hardcoded now scale proportionally:
- Baseline noise windows
- Peak/trough width thresholds  
- Waveform plot ranges

## Usage
No changes needed to existing code! The spike width is automatically calculated when you create parameters:

```matlab
% Your existing code works as-is
param = bc.qm.qualityParamValues(ephysMetaDir, rawFile, ephysKilosortPath, gain_to_uV, kilosortVersion);

% The param.spikeWidth is now automatically set based on your actual sampling rate
% For 32kHz data with KS4, param.spikeWidth will be 65 instead of 61
```

## Benefits
1. **No more manual edits**: Works out-of-the-box with any sampling rate
2. **Maintains temporal consistency**: Same time windows regardless of sampling rate
3. **Backwards compatible**: Standard 30kHz data behaves exactly as before
4. **Future-proof**: Easy to add support for new Kilosort versions or probe types

## Testing
Run `test_32khz_example.m` to see the calculations for different sampling rates and verify the behavior with your 32kHz data.