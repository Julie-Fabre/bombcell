import warnings

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
        # New recommended parameters:
        "rpv_method": "hill",  # Method for RPV computation: 'hill', 'llobet', or 'ibl_sliding'
        "tauR_values": np.array([0.002]),  # Refractory period values to test (in seconds)
                                            # For sweeping, use e.g. np.arange(0.001, 0.005, 0.0005)
        "tauC": 0.1 / 1000,  # Censored period time (s), to prevent duplicate spikes
        "contamination_values": None,  # For ibl_sliding: contamination values to test
                                       # If None, uses np.arange(0.5, 35, 0.5) / 100
        "confidence_threshold": 0.9,  # For ibl_sliding: confidence threshold for contamination estimate

        # Legacy parameters (still supported for backward compatibility):
        "tauR_valuesMin": 2 / 1000,  # refractory period time (s), usually 0.002 s
        "tauR_valuesMax": 2 / 1000,  # refractory period time (s)
        "tauR_valuesStep": 0.5 / 1000,  # step size for tauR sweep
        "hillOrLlobetMethod": True,  # use hill if 1, else use Llobet et al. (legacy)

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

        ## Recording parameters - !!WARNINGS!! :
        # 1. if you modify any of these after having already run bombcell, you 
        # will need to set 'reextractRaw' to true to update the raw waveforms
        # 2. if you also specify a meta file as input & you are using spikeGLX or 
        # OpenEphys to record your probes, these values will be over-ridden by the ones
        # in your provided meta file (bombcell reads them in)
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
            # so we only need the final ~0.195 multiplier
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



# Parameters that predate a given bombcell version are filled in with values chosen
# to reproduce the behaviour from before that parameter existed, so that reloading an
# old _bc_parameters._bc_qMetrics.parquet classifies the same way it did originally.
# Mirrors bc.qm.checkParameterFields on the MATLAB side.
_BACKCOMPAT_DEFAULTS = {
    "extractRaw": True,
    "computeSpatialDecay": True,
    "spDecayLinFit": True,
    "computeDrift": False,
    "computeDistanceMetrics": False,
    "splitGoodAndMua_NonSomatic": False,
    "minSpatialDecaySlopeExp": 0.01,
    "maxSpatialDecaySlopeExp": 0.1,
    "maxScndPeakToTroughRatio_noise": 0.8,
    "maxMainPeakToTroughRatio_nonSomatic": 0.8,
    # The four below are ANDed together in get_quality_unit_type, so these values
    # disable that test outright. See _NON_SOMATIC_GROUP.
    "minWidthFirstPeak_nonSomatic": 0,
    "minWidthMainTrough_nonSomatic": 0,
    "minTroughToPeak2Ratio_nonSomatic": 0,
    "maxPeak1ToPeak2Ratio_nonSomatic": np.inf,
}

# The peak1/peak2 non-somatic test is a conjunction of these four thresholds.
_NON_SOMATIC_GROUP = (
    "minTroughToPeak2Ratio_nonSomatic",
    "minWidthFirstPeak_nonSomatic",
    "minWidthMainTrough_nonSomatic",
    "maxPeak1ToPeak2Ratio_nonSomatic",
)

# Every parameter get_quality_unit_type reads. Anything here that is missing and has
# no back-compatibility default is a hard error rather than an invented threshold.
_CLASSIFICATION_KEYS = (
    "maxNPeaks", "maxNTroughs", "minWvDuration", "maxWvDuration",
    "maxWvBaselineFraction", "maxScndPeakToTroughRatio_noise",
    "computeSpatialDecay", "spDecayLinFit", "minSpatialDecaySlope",
    "minSpatialDecaySlopeExp", "maxSpatialDecaySlopeExp",
    "minTroughToPeak2Ratio_nonSomatic", "minWidthFirstPeak_nonSomatic",
    "minWidthMainTrough_nonSomatic", "maxPeak1ToPeak2Ratio_nonSomatic",
    "maxMainPeakToTroughRatio_nonSomatic", "maxPercSpikesMissing", "minNumSpikes",
    "maxRPVviolations", "minPresenceRatio", "extractRaw", "minAmplitude", "minSNR",
    "computeDrift", "maxDrift", "computeDistanceMetrics", "isoDmin", "lratioMax",
    "splitGoodAndMua_NonSomatic",
)


def check_parameter_fields(param, verbose=True):
    """
    Fill in parameters missing from an older param set, for back-compatibility.

    Missing fields are filled with values that reproduce the behaviour from before
    that field was introduced, so reloading an old parquet reclassifies as it did
    originally. The exception is the group of four thresholds behind the peak1/peak2
    non-somatic test: those are only inert as a set, so a param set holding some of
    them has the rest filled from the current defaults instead of being silently
    switched off. See bc.qm.checkParameterFields for the MATLAB equivalent.

    Parameters
    ----------
    param : dict
        Parameters, e.g. as returned by load_bc_results
    verbose : bool, optional
        Whether to report which fields were filled in, by default True

    Returns
    -------
    dict
        A copy of param with any missing fields added

    Raises
    ------
    KeyError
        If a parameter needed for classification is missing and has no
        back-compatibility default, rather than inventing a threshold for it
    """
    param = dict(param)
    defaults = dict(_BACKCOMPAT_DEFAULTS)

    group_is_set = [key in param for key in _NON_SOMATIC_GROUP]
    if any(group_is_set) and not all(group_is_set):
        # Some of the conjunction is configured, so filling the rest with the inert
        # values above would switch off a test the user did set up. Use the current
        # defaults instead, scaling the widths by spike width as get_default_parameters
        # does when a MATLAB-written param set gives us the spike width to do it with.
        width_scale = 1.0
        spike_width = param.get("spikeWidth")
        standard_width = param.get("standardSpikeWidth")
        if spike_width and standard_width and np.isfinite(spike_width) and standard_width > 0:
            width_scale = spike_width / standard_width
        defaults["minTroughToPeak2Ratio_nonSomatic"] = 5
        defaults["minWidthFirstPeak_nonSomatic"] = max(2, round(4 * width_scale))
        defaults["minWidthMainTrough_nonSomatic"] = max(3, round(5 * width_scale))
        defaults["maxPeak1ToPeak2Ratio_nonSomatic"] = 3

        present = [k for k, s in zip(_NON_SOMATIC_GROUP, group_is_set) if s]
        absent = [k for k, s in zip(_NON_SOMATIC_GROUP, group_is_set) if not s]
        warnings.warn(
            f"Some non-somatic peak1/peak2 parameters are set ({', '.join(present)}) "
            f"but others are missing ({', '.join(absent)}). Filling the missing ones "
            "with the current bombcell defaults rather than with values that would "
            "disable the peak1/peak2 non-somatic test entirely.",
            stacklevel=2,
        )

    filled = [key for key in defaults if key not in param]
    for key in filled:
        param[key] = defaults[key]

    still_missing = [key for key in _CLASSIFICATION_KEYS if key not in param]
    if still_missing:
        raise KeyError(
            "Parameters needed for classification are missing and have no "
            f"back-compatibility default: {', '.join(still_missing)}. Add them to "
            "param, or regenerate it with get_default_parameters."
        )

    if filled and verbose:
        print(f"Missing param fields filled in with default values: {', '.join(sorted(filled))}")

    return param
