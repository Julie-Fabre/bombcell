import numpy as np


def get_default_parameters(
    kilosort_path,
    raw_dir=None,
    kilosort_version=None,
    ephys_meta_dir=None,
    gain_to_uV=None,
):
    param = {
        # Quality metric computation and display parameters
        ## general 
        "show_detail_plots": False,  # show step-by-step plots
        "show_summary_plots": True,  # Summary plots of quality metrics
        "verbose": True,  # If True will update user on progress
        "re_extract_raw": False,  # If True will re extract raw waveforms
        "save_as_tsv": True,  # save outputs as a .tsv file, useful for using phy after bombcell
        "unit_type_for_phy": True,  # save a unit_type .tsv file for phy
        "ephys_kilosort_path": kilosort_path,  # path to the KiloSort directory
        "save_mat_file": False,  # TODO use scipy to save .mat file?

        ## Duplicate spike parameters
        "remove_duplicate_spike": True,
        "duplicate_spikes_window_s": 0.00001,  # in seconds
        "save_spike_without_duplicates": True,
        "recompute_duplicate_spike": False,

        ## Amplitude / raw waveform parameters
        "detrend_waveform": True,  # If True will linearly de-trend over time
        "n_raw_spikes_to_extract": 500,  # Number of raw spikes per unit
        "save_multiple_raw": False,  # TODO check if still needed for UM
        "decompress_data": False,  # whether to decompress .cbin data
        "extract_raw_waveforms": True,
        "probe_type": 1,  # If you are using spikeGLX and your meta files does not
        # contain information on your probe type specify it here
        # '1' for 1.0 (3Bs) and '2' for 2.0 (single or 4-shanks)

        ## Refractory period parameters
        "tauR_values_min": 2 / 1000,  # refractory period time (s), usually 0.002 s
        "tauR_values_max": 2 / 1000,  # refractory period time (s)
        "tauR_values_steps": 0.5
        / 1000,  # if tauR_values_min and tauR_values_max are different
        # bombcell will estimate values in between using
        # tauR_values_steps
        "tauC": 0.1 / 1000,  # censored period time (s), to prevent duplicate spikes
        "use_hill_method": True,  # use hill if 1, else use Llobet et al.

        ## Percentage spikes missing parameters
        "compute_time_chunks": True,  # compute fraction refractory period violations and
        # percent spikes missing for different time chunks
        "delta_time_chunk": 360,  # time in seconds

        ## Presence  ratio
        "presence_ratio_bin_size": 60,  # in seconds

        ## Drift estimate
        "drift_bin_size": 60,  # in seconds
        "compute_drift": False,  # If True computes drift per unit

        ## Waveform parameters
        "min_thresh_detect_peaks_troughs": 0.2,  # this is multiples by the max value in a units
        # waveform to give the minimum prominence to detect peaks

        # it must be at least this many times larger than the peak after the trough
        # to qualify as a non-somatic unit
        "normalize_spatial_decay": True,  # If True, will normalize spatial decay points relative to maximum
        # this makes the spatial decay more invariant to the spike-sorting
        "sp_decay_lin_fit": False, # if True, use a linear fit for spatial decay. If false, use exponential (preferred)
        "max_scnd_peak_to_trough_ratio_noise": 0.8,
        "min_trough_to_peak2_ratio_non_somatic": 5,
        "min_width_first_peak_non_somatic": 4,
        "min_width_main_trough_non_somatic": 5,
        "max_peak1_to_peak2_ratio_non_somatic": 3,
        "max_main_peak_to_trough_ratio_non_somatic": 0.8,

        ## Recording parameters
        "ephys_sample_rate": 30000,  # samples per second
        "n_channels": 385,  # Number of recorded channels (including any sync channels) recorded in raw data
        "n_sync_channels": 1,

        ## Distance metric parameters
        "compute_distance_metrics": False,  # If True computes distance metics NOTE is slow in ML
        "n_channels_iso_dist": 4,  # Number of nearby channels to use in distance metric computation

        # Quality metric classification parameters
        "split_good_and_mua_non_somatic": False,  # whether to classify non-somatic units
        ## Waveform-based
        "max_n_peaks": 2,  # maximum number of peaks
        "max_n_troughs": 1,  # maximum number of troughs
        "keep_only_somatic": True,  # keep only somatic units
        "min_wv_duration": 100,  # in us
        "max_wv_duration": 800,  # in us
        "min_spatial_decay_slope": -0.008,
        "min_spatial_decay_slope_exp": -0.01,  # in a.u / um
        "max_spatial_decay_slope_exp": -0.1,  # in a.u / um
        "max_wv_baseline_fraction": 0.3,  # maximum absolute value in waveform baseline should not
        # exceed this fraction of the waveforms's absolute peak
        "max_scnd_peak_to_trough_ratio_noise": 0.8, 
        "min_trough_to_peak2_ratio_non_somatic": 5,
        "min_width_first_peak_non_somatic": 4,
        "min_width_main_trough_non_somatic": 5,
        "max_peak1_to_peak2_ratio_non_somatic": 3,
        "max_main_peak_to_trough_ratio_non_somatic": 0.8,

        ## Distance metrics
        "iso_d_min": 20,  # minimum isolation distance value
        "lratio_max": 0.1,  # maximum l-ratio value
        "ss_min": np.nan,  # minimum silhouette score
        ## Other classification parameters
        "min_amplitude": 20,  # in uV
        "max_RPV": 0.1,  # max fraction of refractory period violations
        "max_perc_spikes_missing": 20,  # max percentage of missing spikes
        "min_num_spikes_total": 300,  # minimum number of total spikes recorded
        "max_drift": 100,  # in um
        "min_presence_ratio": 0.7,  # minimum fraction of time chunks unit must be present for
        "min_SNR": 0,  # min SNR for a good unit
    }

    if ephys_meta_dir is not None:
        param["ephys_meta_file"] = ephys_meta_dir
        if gain_to_uV is not None and not np.isnan(gain_to_uV):
            param["gain_to_uV"] = gain_to_uV
        else:
            param["gain_to_uV"] = np.NaN
    else:
        param["ephys_meta_file"] = None
        param["gain_to_uV"] = gain_to_uV

    if raw_dir != None:
        param["raw_data_dir"] = raw_dir

    if kilosort_version == 4:
        param["spike_width"] = (61,)  # width of spike in samples
        param["waveform_baseline_noise_window"] = (
            10  # time in samples at the beginning, with no signal
        )
        param["waveform_baseline_window_start"] = 0  # in samples
        param["waveform_baseline_window_stop"] = 10  # in samples
    else:
        param["spike_width"] = (82,)  # width of spike in samples
        param["waveform_baseline_noise_window"] = (
            20  # time in samples at the beginning, with no signal
        )
        param["waveform_baseline_window_start"] = 21  # in samples
        param["waveform_baseline_window_stop"] = 31  # in samples

    return param


def default_parameters_for_unitmatch(
    kilosort_path,
    raw_dir=None,
    kilosort_version=None,
    ephys_meta_dir=None,
    gain_to_uV=None,
):

    param = get_default_parameters(
    kilosort_path,
    raw_dir=None,
    kilosort_version=None,
    ephys_meta_dir=None,
    gain_to_uV=None,
    )
    param["detrend_waveform"] = 0
    param["detrend_waveform"] = True  # If True will linearly de-trend over time
    param["n_raw_spikes_to_extract"] = 1000  # Number of raw spikes per unit
    param["save_multiple_raw"] = True  # TODO check if still needed for UM
    param["decompress_data"] = True # whether to decompress .cbin data
        

    return param
