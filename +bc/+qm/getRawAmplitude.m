function rawAmplitude_uV = getRawAmplitude(rawWaveforms, peakChan, scalingFactors)
% JF, Get the amplitude of the mean raw waveform for a unit
% ------
% Inputs
% ------
% rawWaveforms: nTimePoints × 1 double vector of the mean raw waveform
%   for one unit in mV
% peakChan : 
% scalingFactors : 
% ------
% Outputs
% ------
% rawAmplitude_uV: raw amplitude in microVolts of the mean raw waveform for
%   this unit

% sanitize inputs
if nargin < 4 || isempty(probeType)
    probeType = 'NaN';
else
    probeType = num2str(probeType);
end

% get scaling factor for peak chan 
scalingFactor = scalingFactors(peakChan);
    
% scale waveforms to get amplitude in microVolts 
rawWaveforms = rawWaveforms .* scalingFactor;
rawAmplitude_uV = abs(max(rawWaveforms)) + abs(min(rawWaveforms));
end
