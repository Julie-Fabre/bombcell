import numpy as np
from pathlib import Path

from bombcell.loading_utils import get_gain_spikeglx

def get_default_parameters(
    kilosort_path,
    raw_file=None,
    kilosort_version=4,
    meta_file=None,
    gain_to_uV=None,
):
    """
    Creates the parameters dictionary

    Parameters
    ----------
    kilosort_path : str
        The path to the KiloSort directory
    raw_file : str, optional
        The path to the raw data, by default None
    kilosort_version : int, optional
        Changes parameters based on if KS4 or earlier version were used, by default None
    meta_file : str, optional
        The path to the meta file of the raw recording (.meta for SpikeGLX or .oebin for OpenEphys), by default None
    gain_to_uV : float, optional
        The gain to micro volts if needed to give manually, by default None

    Returns
    -------
    param : dictionary
        The full param dictionary need to run BombCell
    """
    param = {
        # Quality metric computation and display parameters
        ## general 
        "plotDetails": False,  # show step-by-step plots
        "plotGlobal": True,  # Summary plots of quality metrics
        "savePlots": False,  # If True will save plots to disk
        "plotsSaveDir": None,  # Directory to save plots to (if None, saves to kilosort_path/bombcell_plots/)
        "verbose": True,  # If True will update user on progress
        "reextractRaw": False,  # If True will re extract raw waveforms
        "saveAsTSV": True,  # save outputs as a .tsv file, useful for using phy after bombcell
        "unit_type_for_phy": True,  # save a unit_type .tsv file for phy
        "ephysKilosortPath": str(kilosort_path),  # path to the KiloSort directory

        ## Duplicate spike parameters
        "removeDuplicateSpikes": False,
        "duplicateSpikeWindow_s": 0.000034,  # in seconds
        "saveSpikes_withoutDuplicates": True,
        "recomputeDuplicateSpikes": False,

        ## Amplitude / raw waveform parameters
        "detrendWaveform": True,  # If True will linearly de-trend the average waveforms for BombCell
        "detrendForUnitMatch": False,  # If True will linearly de-trend raw waveforms saved for UnitMatch
        "nRawSpikesToExtract": 100,  # Number of raw spikes per unit
        "decompress_data": False,  # whether to decompress .cbin data
        "extractRaw": True,
        "probeType": 1,  # If you are using spikeGLX and your meta files does not
        # contain information on your probe type specify it here
        # '1' for 1.0 (3Bs) and '2' for 2.0 (single or 4-shanks)

        ## Refractory period parameters
        "tauR_valuesMin": 2 / 1000,  # refractory period time (s), usually 0.002 s
        "tauR_valuesMax": 2 / 1000,  # refractory period time (s)
        "tauR_valuesStep": 0.5 / 1000,  # if tauR_valuesMin and tauR_valuesMax are different
        # bombcell will estimate values in between using
        # tauR_valuesStep
        "tauC": 0.1 / 1000,  # censored period time (s), to prevent duplicate spikes
        "hillOrLlobetMethod": True,  # use hill if 1, else use Llobet et al.

        ## Percentage spikes missing parameters
        "computeTimeChunks": False,  # compute fraction refractory period violations and
        # percent spikes missing for different time chunks
        "deltaTimeChunk": 360,  # time in seconds

        ## Presence  ratio
        "presenceRatioBinSize": 60,  # in seconds

        ## Drift estimate
        "driftBinSize": 60,  # in seconds
        "computeDrift": False,  # If True computes drift per unit

        ## Waveform parameters
        "minThreshDetectPeaksTroughs": 0.2,  # this is multiples by the max value in a units
        # waveform to give the minimum prominence to detect peaks

        # it must be at least this many times larger than the peak after the trough
        # to qualify as a non-somatic unit
        "normalizeSpDecay": True,  # If True, will normalize spatial decay points relative to maximum
        # this makes the spatial decay more invariant to the spike-sorting
        "spDecayLinFit": False, # if True, use a linear fit for spatial decay. If false, use exponential (preferred)
        "computeSpatialDecay": True,

        ## Recording parameters - !WARNING! if you modify any of these after having already run bombcell, you 
        # will need to set 'reextractRaw' to true to update the raw waveforms
        "ephys_sample_rate": 30000,  # samples per second
        "nChannels": 385,  # Number of recorded channels (including any sync channels) in raw data
        "nSyncChannels": 1, # Number of recorded SYNC channels in raw data

        ## Distance metric parameters
        "computeDistanceMetrics": False,  # If True computes distance metics NOTE is slow in ML
        "nChannelsIsoDist": 4,  # Number of nearby channels to use in distance metric computation

        # Quality metric classification parameters
        "splitGoodAndMua_NonSomatic": False,  # whether to classify non-somatic units
        ## Waveform-based
        "maxNPeaks": 2,  # maximum number of peaks
        "maxNTroughs": 1,  # maximum number of troughs
        "minWvDuration": 100,  # in us
        "maxWvDuration": 1150,  # in us
        "minSpatialDecaySlope": -0.008,
        "minSpatialDecaySlopeExp": 0.01,  # in a.u / um
        "maxSpatialDecaySlopeExp": 0.1,  # in a.u / um
        "maxWvBaselineFraction": 0.3,  # maximum absolute value in waveform baseline should not
        # exceed this fraction of the waveforms's absolute peak
        "maxScndPeakToTroughRatio_noise": 0.8, 
        "minTroughToPeak2Ratio_nonSomatic": 5,
        "minWidthFirstPeak_nonSomatic": 4,
        "minWidthMainTrough_nonSomatic": 5,
        "maxPeak1ToPeak2Ratio_nonSomatic": 3,
        "maxMainPeakToTroughRatio_nonSomatic": 0.8,

        ## Distance metrics
        "isoDmin": 20,  # minimum isolation distance value
        "lratioMax": 0.3,  # maximum l-ratio value
        "ss_min": np.nan,  # minimum silhouette score, not currently implemented
        
        ## Other classification parameters
        "minAmplitude": 40,  # in uV
        "maxRPVviolations": 0.1,  # max fraction of refractory period violations
        "maxPercSpikesMissing": 20,  # max percentage of missing spikes
        "minNumSpikes": 300,  # minimum number of total spikes recorded
        "maxDrift": 100,  # in um
        "minPresenceRatio": 0.7,  # minimum fraction of time chunks unit must be present for
        "minSNR": 5,  # min SNR for a good unit
    }


    # Fetch metadata uV conversion factor
    if meta_file is not None and gain_to_uV is None:
        # Check if this is an OpenEphys file
        if '.oebin' in str(meta_file):
            # OpenEphys format - use hardcoded scaling factor
            # OpenEphys already applies standard gain (2.34 μV/bit for AP), 
            # so we only need the final 0.195 multiplier
            # Ref: https://open-ephys.atlassian.net/wiki/spaces/OEW/pages/166789121/Flat+binary+format
            gain_to_uV = 0.1949999928474426  
        else:
            # SpikeGLX format - read from meta file
            gain_to_uV = get_gain_spikeglx(meta_file)

    # Add to param dictionary
    if meta_file is not None:
        param["ephys_meta_file"] = str(meta_file)
        if gain_to_uV is not None and not np.isnan(gain_to_uV):
            param["gain_to_uV"] = gain_to_uV
        else:
            param["gain_to_uV"] = np.nan
    else:
        param["ephys_meta_file"] = None
        param["gain_to_uV"] = gain_to_uV

    if raw_file != None:
        param["raw_data_file"] = str(raw_file)
    else:
        param["raw_data_file"] = None

    if kilosort_version == 4:
        param["spike_width"] = 61 # width of spike in samples
        param["waveformBaselineNoiseWindow"] = 10  # time in samples at the beginning, with no signal
        param["waveform_baseline_window_start"] = 0  # 0-indexed, in samples
        param["waveform_baseline_window_stop"] = 10  # 0-indexed, in samples

    else:
        param["spike_width"] = 82 # width of spike in samples
        param["waveformBaselineNoiseWindow"] = 20  # time in samples at the beginning, with no signal
        param["waveform_baseline_window_start"] = 21  # in samples
        param["waveform_baseline_window_stop"] = 31  # in samples

    return param


def get_unit_match_parameters(
    kilosort_path,
    raw_file=None,
    kilosort_version=4,
    meta_file=None,
    gain_to_uV=None,
):
    """
    Creates the parameters dictionary optimized for UnitMatch
    
    Parameters
    ----------
    kilosort_path : str
        The path to the KiloSort directory
    raw_file : str, optional
        The path to the raw data, by default None
    kilosort_version : int, optional
        Changes parameters based on if KS4 or earlier version were used, by default None
    meta_file : str, optional
        The path to the meta file of the raw recording (.meta for SpikeGLX or .oebin for OpenEphys), by default None
    gain_to_uV : float, optional
        The gain to micro volts if needed to give manually, by default None

    Returns
    -------
    param : dictionary
        The full param dictionary optimized for UnitMatch
    """
    # Get defaults first
    param = get_default_parameters(kilosort_path, raw_file, kilosort_version, meta_file, gain_to_uV)
    
    # Unit match specific parameters
    param["detrendWaveform"] = True  # BombCell average waveforms should be detrended for quality metrics
    param["detrendForUnitMatch"] = False  # UnitMatch raw waveforms should not be detrended (it is done in-house)
    param["nRawSpikesToExtract"] = 1000  # inf if you don't encounter memory issues and want to load all spikes
    param["saveMultipleRaw"] = True  # If you wish to save the nRawSpikesToExtract as well,
                                     # currently needed if you want to run unit match https://github.com/EnnyvanBeest/UnitMatch
                                     # to track chronic cells over days after this
    param["decompress_data"] = True  # UnitMatch typically needs decompression enabled
    
    return param


