function param = qualityParamValues(ephysMetaDir, rawFile, ephysKilosortPath, gain_to_uV, kilosortVersion)
% JF, Load a parameter structure defining extraction and
% classification parameters
% ------
% Inputs
% ------
% ephysMetaDir: dir() structure of the path to your .meta or .oebin meta
%   file
% rawFile: character array defining the path where your uncompressed raw
%   ephys data is
% ------
% Outputs
% ------
% param: matlab structure defining extraction and
% classification parameters (see bc_qualityParamValues for required fields
% and suggested starting values)
% 
if nargin < 5 
    kilosortVersion = 2;
end
param = struct; %initialize structure 

%% calculating quality metrics parameters 
% plotting parameters 
param.plotDetails = 0; % generates a lot of plots, 
    % mainly good if you running through the code line by line to check things,
    % to debug, or to get nice plots for a presentation
param.plotGlobal = 1; % plot summary of quality metrics 
param.verbose = 1; % update user on progress
param.reextractRaw = 0; % re extract raw waveforms or not 

% saving parameters 
param.saveAsTSV = 1; % additionally save outputs in .tsv file - this is 
    % useful if you want to use phy after bombcell: each quality metric value
    % will appear as a column in the Cluster view
param.unitType_for_phy = 1; % whether to save the output of unitType in .tsv file for phy
if nargin < 3
    warning('no ephys kilosort path defined in bc_qualityParamValues, will save output tsv file in the savePath location')
else
    param.ephysKilosortPath = ephysKilosortPath;
end
param.saveMatFileForGUI = 1; % save certain outputs at .mat file - useful for GUI

% duplicate spikes parameters 
param.removeDuplicateSpikes = 1;
param.duplicateSpikeWindow_s = 0.00001; % in seconds 
param.saveSpikes_withoutDuplicates = 1;
param.recomputeDuplicateSpikes = 0;

% amplitude / raw waveform parameters
param.detrendWaveform = 1; % If this is set to 1, each raw extracted spike is
    % detrended (we remove the best straight-fit line from the spike)
    % using MATLAB's builtin function detrend.
param.nRawSpikesToExtract = 100; % how many raw spikes to extract for each unit 
param.saveMultipleRaw = 0; % If you wish to save the nRawSpikesToExtract as well, 
    % currently needed if you want to run unit match https://github.com/EnnyvanBeest/UnitMatch
    % to track chronic cells over days after this
param.decompressData = 0; % whether to decompress .cbin ephys data 
if kilosortVersion == 4
    param.spikeWidth = 61; % width in samples 
else
    param.spikeWidth = 82; % width in samples 
end
if strcmp(rawFile, "NaN")
    param.extractRaw = 0;
else
    param.extractRaw = 1; %whether to extract raw waveforms or not 
end
param.probeType = 1; % if you are using spikeGLX and your meta file does 
    % not contain information about your probe type for some reason
    % specify it here: '1' for 1.0 (3Bs) and '2' for 2.0 (single or 4-shanks)
    % For additional probe types, make a pull request with more
    % information.  If your spikeGLX meta file contains information about your probe
    % type, or if you are using open ephys, this paramater wil be ignored.
param.computeSpatialDecay = 1;

% signal to noise ratio
if kilosortVersion == 4
    param.waveformBaselineNoiseWindow = 10; %time in samples at beginning of times
        % extracted to computer the mean raw waveform - this needs to be before the
        % waveform starts 
else
    param.waveformBaselineNoiseWindow = 20; %time in samples at beginning of times
        % extracted to computer the mean raw waveform - this needs to be before the
        % waveform starts 
end

% refractory period parameters
param.tauR_valuesMin = 2/1000; % refractory period time (s), usually 0.0020. 
    % If this value is different than param.tauR_valuesMax, bombcell will
    % estimate the tauR value taking possible values between :
    % param.tauR_valuesMin:param.tauR_valuesStep:param.tauR_valuesMax
param.tauR_valuesStep = 0.5/1000; % refractory period time (s) steps. Only 
    % used if param.tauR_valuesMin is different from param.tauR_valuesMax
param.tauR_valuesMax = 2/1000; % refractory period time (s), usually 0.0020
param.tauC = 0.1/1000; % censored period time (s) - this is to prevent duplicate spikes 
param.hillOrLlobetMethod = 1; % 1 to use Hill et al method; 0 to use Llobet et al method

% percentage spikes missing parameters 
param.computeTimeChunks = 0; % compute fraction refractory period violations 
    % and percent spikes missing for different time chunks 
param.deltaTimeChunk = 360; %time in seconds 

% presence ratio 
param.presenceRatioBinSize = 60; % in seconds 

% drift estimate
param.driftBinSize = 60; % in seconds
param.computeDrift = 0; % whether to compute each units drift. this is a 
    % critically slow step that takes around 2seconds per unit 

% waveform parameters
if kilosortVersion == 4
    param.waveformBaselineWindowStart = 1;
    param.waveformBaselineWindowStop = 11; % in samples 
else
    param.waveformBaselineWindowStart = 20;
    param.waveformBaselineWindowStop = 30; % in samples 
end
param.minThreshDetectPeaksTroughs = 0.2; % this is multiplied by the max value 
    % in a units waveform to give the minimum prominence to detect peaks using
    % matlab's findpeaks function.
param.normalizeSpDecay = 1; % whether to normalize spatial decay points relative to 
% maximum - this makes the spatrial decay slop calculation more invariant to the 
% spike-sorting algorithm used
param.spDecayLinFit = 0; % if false, use an exponential fit

% recording parameters
param.ephys_sample_rate = 30000; % samples per second
param.nChannels = 385; %number of recorded channels (including any sync channels)
    % recorded in the raw data. This is usually 384 or 385 for neuropixels
    % recordings
param.nSyncChannels = 1;
if  ~isempty(ephysMetaDir)
    param.ephysMetaFile = [ephysMetaDir.folder, filesep, ephysMetaDir.name];
    param.gain_to_uV = NaN;
else
    param.ephysMetaFile = 'NaN';
    param.gain_to_uV = gain_to_uV;
end
param.rawFile = rawFile;

% distance metric parameters
param.computeDistanceMetrics = 0; % whether to compute distance metrics - this can be time consuming 
param.nChannelsIsoDist = 4; % number of nearby channels to use in distance metric computation 


%% classifying units into good/mua/noise parameters 
% whether to classify non-somatic units 
param.splitGoodAndMua_NonSomatic = 0;

% waveform - noise
param.maxNPeaks = 2; % maximum number of peaks
param.maxNTroughs = 1; % maximum number of troughs
param.somatic = 1; % keep only somatic units, and reject non-somatic ones
param.minWvDuration = 100; % in us
param.maxWvDuration = 1150; % in us
param.minSpatialDecaySlope = -0.008; % in a.u./um
param.minSpatialDecaySlopeExp = 0.01; % in a.u./um
param.maxSpatialDecaySlopeExp = 0.1; % in a.u./um
param.maxWvBaselineFraction = 0.3; % maximum absolute value in waveform baseline
    % should not exceed this fraction of the waveform's abolute peak value
param.maxScndPeakToTroughRatio_noise = 0.8; % peak must be less than this x the trough 

% waveform - non-somatic
param.maxPeak1ToPeak2Ratio_nonSomatic = 3; % if units have an initial peak before the trough,
    % it must be at least firstPeakRatio times larger than the peak after the trough to qualify as a non-somatic unit. 
param.maxMainPeakToTroughRatio_nonSomatic = 0.8; % peak must be less than this x the trough 
param.minWidthFirstPeak_nonSomatic = 4; % in samples 
param.minWidthMainTrough_nonSomatic = 5; % in samples
param.minTroughToPeak2Ratio_nonSomatic = 5; % trough should be min 5 x bigger than 1rst peak to count as non-somatic 

% distance metrics
param.isoDmin = 20; % minimum isolation distance value
param.lratioMax = 0.1; % maximum l-ratio value
param.ssMin = NaN; % minimum silhouette score 

% other classification params
param.minAmplitude = 20; % in uV
param.maxRPVviolations = 0.1; % fraction
param.maxPercSpikesMissing = 20; % in percentage
param.minNumSpikes = 300; % number of spikes
param.maxDrift = 100; % in micrometers 
param.minPresenceRatio = 0.7; % fraction
param.minSNR = 1;

end
