import time
from pathlib import Path
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from tqdm.auto import tqdm

from bombcell.extract_raw_waveforms import manage_data_compression, extract_raw_waveforms
from bombcell.loading_utils import load_ephys_data

# import matplotlib.pyplot as plt
import bombcell.quality_metrics as qm
from bombcell.save_utils import get_metric_keys, save_results
from bombcell.plot_functions import *


##TODO can add runtimes to optional steps here
def show_times(
    runtimes_spikes_missing_1,
    runtimes_RPV_1,
    runtimes_chunks_to_keep,
    runtimes_spikes_missing_2,
    runtimes_RPV_2,
    runtimes_presence_ratio,
    runtimes_max_drift,
    runtimes_waveform_shape,
):
    """
    Prints all the gathered run times for each step of the BombCell pipeline

    Parameters
    ----------
    runtimes_spikes_missing_1 : ndarray
        Spikes missing runtime for first call
    runtimes_RPV_1 : ndarray
        RPV runtime for first call
    runtimes_chunks_to_keep : ndarray
        Chunks to keep runtime
    runtimes_spikes_missing_2 : ndarray
        Spikes missing runtime for the second call
    runtimes_RPV_2 : ndarray
        RPV runtime for the second call
    runtimes_presence_ratio : ndarray
        Presence ratio runtime
    runtimes_max_drift : ndarray
        Drift runtime
    runtimes_waveform_shape : ndarray
        Waveforms shape runtime
    """
    print(f"The time the first spikes missing took: {runtimes_spikes_missing_1.sum()}")
    print(f"The time the first RPV took: {runtimes_RPV_1.sum()}")
    print(f"The time the time chunks took: {runtimes_chunks_to_keep.sum()}")
    print(f"The time the second spikes missing took: {runtimes_spikes_missing_2.sum()}")
    print(f"The time the second RPV took: {runtimes_RPV_2.sum()}")
    print(f"The time the presence ratio took: {runtimes_presence_ratio.sum()}")
    print(f"The time the max drift took: {runtimes_max_drift.sum()}")
    print(f"The time the waveform shapes took: {runtimes_waveform_shape.sum()}")


def print_unit_qm(quality_metrics, unit_idx, param, unit_type = None):
    """
    Prints all of the extracted quality metrics for a unit

    Parameters
    ----------
    quality_metrics : dict
        The full quality metrics dictionary
    unit_idx : int
        The id of the unit to look at 
    param : dict
        The param dictionary
    unit_type : ndarray, optional
        The array which contains what cell type every unit is classed as, by default None
    """
    print(f"For unit {unit_idx}:")
    print(
        f'nPeaks : {quality_metrics["nPeaks"][unit_idx]}, nTroughs : {quality_metrics["nTroughs"][unit_idx]}'
    )
    print(
        f'waveformDuration_peakTrough : {quality_metrics["waveformDuration_peakTrough"][unit_idx]:.3f}'
    )
    print(
        f'waveform_baseline : {quality_metrics["waveform_baseline"][unit_idx]}, spatialDecaySlope : {quality_metrics["spatialDecaySlope"][unit_idx]}'
    )
    print(
        f'percentageSpikesMissing_gaussian : {quality_metrics["percentageSpikesMissing_gaussian"][unit_idx]}, nSpikes : {quality_metrics["nSpikes"][unit_idx]}'
    )
    print(
        f'fractionRPVs : {quality_metrics["fractionRPVs_estimatedTauR"][unit_idx]}, presenceRatio : {quality_metrics["presenceRatio"][unit_idx]}'
    )

    if param["extractRaw"]:
        print(
            f'rawAmplitude : {quality_metrics["rawAmplitude"][unit_idx]:.3f}, signalToNoiseRatio : {quality_metrics["signalToNoiseRatio"][unit_idx]:.3f}'
        )
    if param["computeDistanceMetrics"]:
        print(
            f'maxDriftEstimate : {quality_metrics["maxDriftEstimate"][unit_idx]:.3f}'
        )

    print(f'Waveform IS somatic = {1 == quality_metrics["is_somatic"][unit_idx]}')

    if unit_type is not None:
        print(f"The Units is classed as {unit_type[unit_idx]}")


def print_qm_thresholds(param):
    """
    This function prints all of the thresholds used

    Parameters
    ----------
    param : dict
        The param dictionary
    """
    print("Current threshold params:")
    print(
        f'maxNPeaks = {param["maxNPeaks"]}, maxNTroughs = {param["maxNTroughs"]}'
    )
    print(
        f'minWvDuration = {param["minWvDuration"]}, maxWvDuration = {param["maxWvDuration"]}'
    )
    print(
        f'maxWvBaselineFraction = {param["maxWvBaselineFraction"]}, minSpatialDecaySlope = {param["minSpatialDecaySlope"]}'
    )
    print(
        f'maxPercSpikesMissing = {param["maxPercSpikesMissing"]}, minNumSpikes = {param["minNumSpikes"]}'
    )
    print(
        f'maxRPVviolations = {param["maxRPVviolations"]}, minPresenceRatio = {param["minPresenceRatio"]}'
    )

    if param["extractRaw"]:
        print(f'minAmplitude = {param["minAmplitude"]}, minSNR = {param["minSNR"]}')

    if param["computeDistanceMetrics"]:
        print(f'maxDrift = {param["maxDrift"]}')


def show_somatic(quality_metrics, unit_idx):
    """
    This function shows all of the information related to somatic/non-somatic classification

    Parameters
    ----------
    quality_metrics : dict
        The full quality metrics dictionary
    unit_idx : int
        The index of the unit to look at
    """
    print(f'The max trough is {quality_metrics["trough"][unit_idx]}')
    print(f'The main peak before is {quality_metrics["main_peak_before"][unit_idx]}')
    print(f'The main peak after is {quality_metrics["main_peak_after"][unit_idx]}')
    print(f'The first peak width is {quality_metrics["width_before"][unit_idx]}')
    print(f'The trough_width is {quality_metrics["mainTrough_width"][unit_idx]}')

def order_good_sites(good_sites, channel_pos):
    """
    Reorder channel positions so they are ordered from smallest to biggest

    Parameters
    ----------
    good_sites : ndarray
        The indexes of the good sites
    channel_pos : ndarray
        The channel positions

    Returns
    -------
    reordered_good_sites : ndarray
        The good sites indexes in order
    """
    # make it so it goes from biggest to smallest
    reordered_idx = np.argsort(-channel_pos[good_sites, 1])
    reordered_good_sites = good_sites[reordered_idx]

    # re-arange x-axis so it goes (smaller x, bigger x)
    for i in range(8):
        a, b = channel_pos[reordered_good_sites[[2 * i, 2 * i + 1]], 0]

        if a > b:
            # swap order
            reordered_good_sites[[2 * i + 1, 2 * i]] = reordered_good_sites[
                [2 * i, 2 * i + 1]
            ]

    return reordered_good_sites


def nearest_channels(quality_metrics, channel_positions, this_unit, unique_templates):
    """
    Finds the nearest 16 channels for plotting waveforms

    Parameters
    ----------
    quality_metrics : dict
        The quality metrics dictionary
    channel_positions : ndarray
        The channel positions 
    this_unit : int
        The index of the unit to look at
    unique_templates : ndarray
        The array which converts bombcell id to kilosort id

    Returns
    -------
    reordered_good_sites : ndarray
        The indexes of the nearest channels
    """

    unit_id = unique_templates[this_unit]  # JF: this function needs some cleaning up

    max_channel = quality_metrics["maxChannels"][unit_id]

    x, y = channel_positions[max_channel, :]

    #Includes adjacent columns 
    x_dist = np.abs(channel_positions[:, 0] - x)
    near_x_dist = np.min(x_dist[x_dist != 0])

    not_these_x = np.argwhere(x_dist > near_x_dist)

    y_dist = np.abs(channel_positions[:, 1] - y)
    y_dist[not_these_x] = (
        y_dist.max()
    )  # set the bad x_to max y, this keeps the shape of the array
    good_sites = np.argsort(y_dist)[:16]

    reordered_good_sites = order_good_sites(good_sites, channel_positions)

    return reordered_good_sites


def plot_raw_waveforms(
    quality_metrics, channel_positions, this_unit, waveform, unique_templates
):
    """
    Plots the raw waveforms of the unit on the max channel and nearby channels

    Parameters
    ----------
    quality_metrics : dict
        The quality metrics dictionary
    channel_positions : ndarray
        The channel positions
    this_unit : int
        The unit index
    waveform : ndarray
        The waveforms for each unit and channel 
    unique_templates : ndarray
        The array which converts bombcell id to kilosort id

    Returns
    -------
    fig : plot
        The plot of the waveforms
    """

    unit_id = unique_templates[this_unit]  

    fig = Figure(figsize=(4, 6), dpi=100)
    fig.set_tight_layout(False)

    main_ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])
    main_ax_offset = 0.2
    main_ax_scale = 0.8

    good_channels = nearest_channels(
        quality_metrics, channel_positions, this_unit, unique_templates
    ).squeeze()

    min_x, min_y = channel_positions[good_channels[-2], [0, 1]]
    max_x, maxy = channel_positions[good_channels[1], [0, 1]]
    delta_x = (max_x - min_x) / 2
    delta_y = (maxy - min_y) / 18

    # may want to change so it find this for both units and selects the most extreme arguments
    # however i dont think this will be necessary
    sub_min_y = np.nanmin(waveform[unit_id, :, good_channels])
    sub_max_y = np.nanmax(waveform[unit_id, :, good_channels])

    # shift each waveform so 0 is at the channel site, 1/9 is width of a y waveform plot
    waveform_y_offset = (
        (np.abs(sub_max_y) / (np.abs(sub_min_y) + np.abs(sub_max_y))) * 1 / 8
    )  

    main_ax.set_xlim(min_x - delta_x, max_x + delta_x)
    main_ax.set_ylim(min_y - delta_y, maxy + delta_y)

    rel_channel_positions = (
        (channel_positions - channel_positions[good_channels].min(axis=0))
        / (
            channel_positions[good_channels].max(axis=0)
            - channel_positions[good_channels].min(axis=0)
        )
        * 0.8
    )
    rel_channel_positions += main_ax_offset
    for i in range(8):
        for j in range(2):
            # may need to change this positioning if units sizes are irregular
            # if j == 0:
            #     #The peak in the waveform is not half way, so maths says the x axis should be starting at
            #     #0.1 and 0.6 so the middle is at 0.25/0.76 however chosen these values so it looks better by eye
            #     ax =  fig.add_axes([main_ax_offset + main_ax_scale*0.25, main_ax_offset + main_ax_scale*(i/9 - 1/18 + waveform_y_offset), main_ax_scale*0.25, main_ax_scale*1/9])
            # if j == 1:
            #     ax = fig.add_axes([main_ax_offset + main_ax_scale*0.75, main_ax_offset + main_ax_scale*(i/9 - 1/18 + waveform_y_offset), main_ax_scale*0.25, main_ax_scale*1/9])

            if j == 0:
                # The peak in the waveform is not half way, so maths says the x axis should be starting at
                # 0.1 and 0.6 so the middle is at 0.25/0.76 however chosen these values so it looks better by eye
                ax = fig.add_axes(
                    [
                        rel_channel_positions[good_channels, 0][i * 2 + j],
                        rel_channel_positions[good_channels, 1][i * 2 + j],
                        main_ax_scale * 0.25,
                        main_ax_scale * 1 / 8,
                    ]
                )
            if j == 1:
                ax = fig.add_axes(
                    [
                        rel_channel_positions[good_channels, 0][i * 2 + j],
                        rel_channel_positions[good_channels, 1][i * 2 + j],
                        main_ax_scale * 0.25,
                        main_ax_scale * 1 / 8,
                    ]
                )

            ax.plot(waveform[unit_id, :, good_channels[i * 2 + j]], color="g")

            ax.set_ylim(sub_min_y, sub_max_y)
            ax.set_axis_off()

    main_ax.spines.right.set_visible(False)
    main_ax.spines.top.set_visible(False)
    main_ax.set_xticks([min_x, max_x])
    main_ax.set_xlabel("Xpos ($\mu$m)", size=14)
    main_ax.set_ylabel("Ypos ($\mu$m)", size=14)

    return fig


def show_unit(
    template_waveforms,
    this_unit,
    unique_templates,
    quality_metrics,
    channel_positions,
    param,
    unit_type=None,
):
    """
    Shows the qualitymetrics and waveforms of a unit

    Parameters
    ----------
    template_waveforms : ndarray
        The waveforms for each unit and channel
    this_unit : int
        The id for the unit to look at
    unique_templates : ndarray
        The array which converts bombcell id to kilosort id
    quality_metrics : dict
        The quality metrics dictionary
    channel_positions : ndarray
        The channel positions
    param : dict
        The param dictionary
    unit_type : ndarray, optional
        The array which contains what cell type every unit is classed as, by default None

    Returns
    -------
    fig : plot
        The plot of the waveforms
    """
    print_unit_qm(quality_metrics, this_unit, param, unit_type=unit_type)

    fig = plot_raw_waveforms(
        quality_metrics,
        channel_positions,
        this_unit,
        template_waveforms,
        unique_templates,
    )
    return fig


def create_quality_metrics_dict(n_units, snr=None):
    """
    This function creates an quality_metrics dictionary with empty arrays to assign quality metric values to
    for each unit

    Parameters
    ----------
    n_units : int
        The number of units
    snr : ndarray, optional
        The SNR array if applicable, by default None

    Returns
    -------
    quality_metrics : dict
        The quality metrics dictionary
    """
    init_keys = [
        "nSpikes"
        ] + get_metric_keys()

    quality_metrics = {}
    quality_metrics["phy_clusterID"] = np.zeros(n_units).astype(int)
    for k in init_keys:
        quality_metrics[k] = np.full(n_units, np.nan)
    

    # Use passed snr values if found
    if isinstance(snr, np.ndarray):
        quality_metrics["signalToNoiseRatio"] = snr

    return quality_metrics


def set_unit_nan(unit_idx, quality_metrics, not_enough_spikes):
    """
    Set quality metrics to NaN for units with too few spikes.
    """
    metrics_keys = get_metric_keys()
    
    for k in metrics_keys:
        quality_metrics[k][unit_idx] = np.nan
    
    not_enough_spikes[unit_idx] = 1

    return quality_metrics, not_enough_spikes

def get_all_quality_metrics(
    unique_templates,
    spike_times_seconds,
    spike_clusters,
    template_amplitudes,
    time_chunks,
    pc_features,
    pc_features_idx,
    quality_metrics,
    raw_waveforms_full,
    channel_positions,
    template_waveforms,
    param,
    save_path,
):
    """
    This function runs all of the quality metric calculations

    Parameters
    ----------
    unique_templates : ndarray
        The array which converts bombcell id to kilosort id
    spike_times_seconds : ndarray
        The times of spikes in seconds
    spike_clusters : ndarray
        The id of each spike
    template_amplitudes : ndarray
        The amplitude for each spike
    time_chunks : ndarray
        The time chunks to use
    pc_features : ndarray
        The principal components of the data
    pc_features_idx : ndarray
        The unit and channel indexes for the principal components
    quality_metrics : dict
        The empty quality metrics dictionary
    raw_waveforms_full : ndarray
        The raw extracted waveforms
    channel_positions : ndarray
        The max channels of each unit
    template_waveforms : ndarray
        The template waveforms for each unit
    param : dict
        The dictionary of parameters
    save_path: str
        Bombcell results saving path

    Returns
    -------
    quality_metrics : dict
        The quality metrics for every unit
    runtimes : dict
        The runtimes for each sections
    """
    # Collect the time it takes to run each section
    runtimes_spikes_missing_1 = np.zeros(unique_templates.shape[0])
    runtimes_RPV_1 = np.zeros(unique_templates.shape[0])
    runtimes_chunks_to_keep = np.zeros(unique_templates.shape[0])
    runtimes_spikes_missing_2 = np.zeros(unique_templates.shape[0])
    runtimes_RPV_2 = np.zeros(unique_templates.shape[0])
    runtimes_presence_ratio = np.zeros(unique_templates.shape[0])
    runtimes_max_drift = np.zeros(unique_templates.shape[0])
    runtimes_waveform_shape = np.zeros(unique_templates.shape[0])
    runtime_dist_metrics = np.zeros(unique_templates.shape[0])
    RPV_tauR_estimate_units_NtauR = []

    not_enough_spikes = np.zeros(unique_templates.size)
    bad_units = 0
    bar_description = "Computing bombcell quality metrics: {percentage:3.0f}%|{bar:10}| {n}/{total} units"
    for unit_idx in tqdm(range(unique_templates.size), bar_format=bar_description):
        this_unit = unique_templates[unit_idx]
        quality_metrics["phy_clusterID"][unit_idx] = this_unit

        these_spike_times = spike_times_seconds[spike_clusters == this_unit]
        these_amplitudes = template_amplitudes[spike_clusters == this_unit]

        # number of spikes
        quality_metrics["nSpikes"][unit_idx] = these_spike_times.shape[0]

        # Ignoring for the moment as need to find a way to get the same shape as actual results for percent_missings
        # and fraction RPVs # JF: what?
        if these_spike_times.size < 50:
            quality_metrics, not_enough_spikes = set_unit_nan(
                unit_idx, quality_metrics, not_enough_spikes
            )
            bad_units += 1
            continue

        # percentage spikes missing
        time_tmp = time.time()
        (
            percent_missing_gaussian,
            percent_missing_symmetric,
        ) = qm.perc_spikes_missing(
            these_amplitudes, these_spike_times, time_chunks, param
        )
        runtimes_spikes_missing_1[unit_idx] = time.time() - time_tmp

        # fraction contamination
        time_tmp = time.time()
        fraction_RPVs, num_violations = qm.fraction_RP_violations(
            these_spike_times, these_amplitudes, time_chunks, param
        )
        runtimes_RPV_1[unit_idx] = time.time() - time_tmp

        # get time chunks to keep
        time_tmp = time.time()
        (
            these_spike_times,
            these_amplitudes,
            these_spike_clusters,
            quality_metrics["useTheseTimesStart"][unit_idx],
            quality_metrics["useTheseTimesStop"][unit_idx],
            quality_metrics["RPV_window_index"][unit_idx],
        ) = qm.time_chunks_to_keep(
            percent_missing_gaussian,
            fraction_RPVs,
            time_chunks,
            these_spike_times,
            these_amplitudes,
            spike_clusters,
            spike_times_seconds,
            param,
        )
        runtimes_chunks_to_keep[unit_idx] = time.time() - time_tmp

        use_these_times = np.array(
            (
                quality_metrics["useTheseTimesStart"][unit_idx],
                quality_metrics["useTheseTimesStop"][unit_idx],
            )
        )
        # re-compute percentage spikes missing and RPV on time chunks
        time_tmp = time.time()
        (
            quality_metrics["percentageSpikesMissing_gaussian"][unit_idx],
            quality_metrics["percentageSpikesMissing_symmetric"][unit_idx],
        ) = qm.perc_spikes_missing(
            these_amplitudes, these_spike_times, use_these_times, param, metric = True
        )
        runtimes_spikes_missing_2[unit_idx] = time.time() - time_tmp

        time_tmp = time.time()
        fraction_RPVs, num_violations = qm.fraction_RP_violations(
            these_spike_times,
            these_amplitudes,
            use_these_times,
            param)
        runtimes_RPV_2[unit_idx] = time.time() - time_tmp
        fraction_RPVs = fraction_RPVs[0] # only 'use_these_times', so single time chunk

        quality_metrics["fractionRPVs_estimatedTauR"][unit_idx] = fraction_RPVs[
            int(quality_metrics["RPV_window_index"][unit_idx])
        ]
        RPV_tauR_estimate_units_NtauR.append([unit_idx, fraction_RPVs])

        # get presence ratio
        time_tmp = time.time()
        quality_metrics["presenceRatio"][unit_idx] = qm.presence_ratio(
            these_spike_times,
            quality_metrics["useTheseTimesStart"][unit_idx],
            quality_metrics["useTheseTimesStop"][unit_idx],
            param,
        )
        runtimes_presence_ratio[unit_idx] = time.time() - time_tmp

        # maximum cumulative drift estimate
        time_tmp = time.time()
        (
            quality_metrics["maxDriftEstimate"][unit_idx],
            quality_metrics["cumDriftEstimate"][unit_idx],
        ) = qm.max_drift_estimate(
            pc_features,
            pc_features_idx,
            these_spike_clusters,
            these_spike_times,
            this_unit,
            channel_positions,
            param,
        )
        runtimes_max_drift[unit_idx] = time.time() - time_tmp

        # number of spikes
        quality_metrics["nSpikes"][unit_idx] = these_spike_times.shape[0]

        # waveform
        time_tmp = time.time()
        waveform_baseline_window = np.array(
            (
                param["waveform_baseline_window_start"],
                param["waveform_baseline_window_stop"],
            )
        )

        (
            quality_metrics["nPeaks"][unit_idx],
            quality_metrics["nTroughs"][unit_idx],
            quality_metrics["waveformDuration_peakTrough"][unit_idx],
            quality_metrics["spatialDecaySlope"][unit_idx],
            quality_metrics["waveformBaselineFlatness"][unit_idx],
            quality_metrics["scndPeakToTroughRatio"][unit_idx],
            quality_metrics["peak1ToPeak2Ratio"][unit_idx],
            quality_metrics["mainPeakToTroughRatio"][unit_idx],
            quality_metrics["troughToPeak2Ratio"][unit_idx],
            quality_metrics["mainPeak_before_width"][unit_idx],
            quality_metrics["mainTrough_width"][unit_idx],
            param,
        ) = qm.waveform_shape(
            template_waveforms,
            this_unit,
            quality_metrics["maxChannels"],
            channel_positions,
            waveform_baseline_window,
            param,
        )
        runtimes_waveform_shape[unit_idx] = time.time() - time_tmp

        # amplitude
        if raw_waveforms_full is not None and param["extractRaw"] and param['gain_to_uV'] is not None:
            quality_metrics["rawAmplitude"][unit_idx] = qm.get_raw_amplitude(
                raw_waveforms_full[unit_idx], param["gain_to_uV"]
            )
        else:
            quality_metrics["rawAmplitude"][unit_idx] = np.nan

        time_tmp = time.time()
        if param["computeDistanceMetrics"]:
            (
                quality_metrics["isolationDistance"][unit_idx],
                quality_metrics["Lratio"][unit_idx],
                quality_metrics["silhouetteScore"][unit_idx],
            ) = qm.get_distance_metrics(
                pc_features, pc_features_idx, this_unit, spike_clusters, param
            )
        runtime_dist_metrics = time.time() - time_tmp

    runtimes = {
        "times_spikes_missing_1": runtimes_spikes_missing_1, # JF: what is this?
        "times_RPV_1": runtimes_RPV_1,
        "times_chunks_to_keep": runtimes_chunks_to_keep,
        "times_spikes_missing_2": runtimes_spikes_missing_2,
        "times_RPV_2": runtimes_RPV_2,
        "times_presence_ratio": runtimes_presence_ratio,
        "times_max_drift": runtimes_max_drift,
        "times_waveform_shape": runtimes_waveform_shape,
    }

    # save tauR values as parquet file for bombcell GUI
    tauR_min = param["tauR_valuesMin"]
    tauR_max = param["tauR_valuesMax"]
    tauR_step = param["tauR_valuesStep"]
    tauR_window = np.arange(
        tauR_min, tauR_max + tauR_step, tauR_step
    )
    RPV_tauR_estimate_units_NtauR = np.array([el[1] for el in RPV_tauR_estimate_units_NtauR])
    df = pd.DataFrame(RPV_tauR_estimate_units_NtauR,
                      columns=tauR_window,
                      index=[el[0] for el in RPV_tauR_estimate_units_NtauR])
    df.to_parquet(Path(save_path) / "templates._bc_fractionRefractoryPeriodViolationsPerTauR.parquet")

    if param["computeDistanceMetrics"]:
        runtimes["time_dist_metrics"] = runtime_dist_metrics

    return quality_metrics, runtimes


def run_bombcell(ks_dir, save_path, param):
    """
    This function runs the entire bombcell pipeline from input data paths

    Parameters
    ----------
    ks_dir : string
        The path to the KiloSort (or equivalent) save directory
    save_path : string
        The path to the directory to save the bombcell results
    param : dict
        The param dictionary

    Returns
    -------
    quality_metrics : dict
        The quality metrics for each unit
    param : dict
        The parameters 
    unit_type : ndarray
        The unit classifications as numbers 
    unit_type_string: ndarray
        The unit classifications as names
    """
    (
        spike_times_samples,
        spike_clusters, # actually spike_templates, but they're the same in bombcell
        template_waveforms,
        template_amplitudes,
        pc_features,
        pc_features_idx,
        channel_positions,
    ) = load_ephys_data(ks_dir)

    # Extract or load in raw waveforms
    if param["raw_data_file"] is not None:
        (
        raw_waveforms_full,
        raw_waveforms_peak_channel,
        signal_to_noise_ratio,
        raw_waveforms_id_match,
        ) = extract_raw_waveforms(
            param,
            spike_clusters,
            spike_times_samples,
            param["reextractRaw"],
            save_path,
        )
    else:
        raw_waveforms_full = None
        raw_waveforms_peak_channel = None
        signal_to_noise_ratio = None
        raw_waveforms_id_match = None
        param["extractRaw"] = False  # No waveforms to extract!

    # pre-load peak channels
    maxChannels = qm.get_waveform_peak_channel(template_waveforms)

    # Remove duplicate spikes
    if param["removeDuplicateSpikes"]:
        (
            non_empty_units,
            duplicate_spike_idx,
            spike_times_samples,
            spike_clusters,
            template_amplitudes,
            pc_features,
            raw_waveforms_full,
            raw_waveforms_peak_channel,
            signal_to_noise_ratio,
            maxChannels,
        ) = qm.remove_duplicate_spikes(
            spike_times_samples,
            spike_clusters,
            template_amplitudes,
            maxChannels,
            save_path,
            param,
            pc_features=pc_features,
            raw_waveforms_full=raw_waveforms_full,
            raw_waveforms_peak_channel=raw_waveforms_peak_channel,
            signal_to_noise_ratio=signal_to_noise_ratio,
        )
    else:
        non_empty_units = np.unique(spike_clusters)

    # Divide recording into time chunks
    spike_times_seconds = spike_times_samples / param["ephys_sample_rate"]
    if param["computeTimeChunks"]:
        time_chunks = np.arange(
            np.min(spike_times_seconds),
            np.max(spike_times_seconds),
            param["deltaTimeChunk"],
        )
    else:
        time_chunks = np.array(
            (np.min(spike_times_seconds), np.max(spike_times_seconds))
        )

    unique_templates = non_empty_units # template ids are cluster ids, in bombcell
    param['unique_templates'] = unique_templates

    # Initialize quality metrics dictionary
    n_units = unique_templates.size
    quality_metrics = create_quality_metrics_dict(n_units, snr=signal_to_noise_ratio)
    quality_metrics["maxChannels"] = maxChannels

    # Complete with remaining quality metrics
    quality_metrics, times = get_all_quality_metrics(
        unique_templates,
        spike_times_seconds,
        spike_clusters,
        template_amplitudes,
        time_chunks,
        pc_features,
        pc_features_idx,
        quality_metrics,
        raw_waveforms_full,
        channel_positions,
        template_waveforms,
        param,
        save_path,
    )

    unit_type, unit_type_string = qm.get_quality_unit_type(
        param, quality_metrics
    )  # JF: this should be inside bc.get_all_quality_metrics

    plot_summary_data(quality_metrics, template_waveforms, unit_type, unit_type_string, param)

    save_results(
        quality_metrics,
        unit_type_string,
        unique_templates,
        param,
        raw_waveforms_full,
        raw_waveforms_peak_channel,
        raw_waveforms_id_match,
        save_path,
    )  # JF: this should be inside bc.get_all_quality_metrics

    return (
        quality_metrics,
        param,
        unit_type,
        unit_type_string,
    )


def make_qm_table(quality_metrics, param, unit_type_string):
    """
    Makes a table out of the quality metrics 

    Parameters
    ----------
    quality_metrics : dict
        The quality metrics for each unit
    param : dict
        The parameters
    unit_type : ndarray
        The cell type classifications for each unit
    unique_templates : ndarray
        The array which converts bombcell id to kilosort id

    Returns
    -------
    qm_table : DataFrame
        The quality metrics information and as pandas dataframe
    """
    unique_templates = param['unique_templates']

    qm_table_list = [unit_type_string, unique_templates]
    qm_table_names = ["unit_type", "Original ID"]

    nan_result = np.isnan(quality_metrics["nPeaks"])

    too_many_peaks = quality_metrics["nPeaks"] > param["maxNPeaks"]
    too_many_troughs = quality_metrics["nTroughs"] > param["maxNTroughs"]
    too_short_waveform = (
        quality_metrics["waveformDuration_peakTrough"] < param["minWvDuration"]
    )
    too_long_waveform = (
        quality_metrics["waveformDuration_peakTrough"] > param["maxWvDuration"]
    )
    duration = np.logical_or(too_short_waveform, too_long_waveform).squeeze()
    too_noisy_baseline = (
        quality_metrics["waveformBaselineFlatness"] > param["maxWvBaselineFraction"]
    )
    peak2_to_trough = quality_metrics["scndPeakToTroughRatio"] > param["maxScndPeakToTroughRatio_noise "]

    qm_table_list.extend([nan_result, too_many_peaks, too_many_troughs, duration, too_noisy_baseline, peak2_to_trough])
    qm_table_names.extend(["NaN result", "# peaks", "# troughs", "duration", "baseline flatness", "peak2 / trough"])

    if param["computeSpatialDecay"]:
        if param["computeSpatialDecay"] & param["spDecayLinFit"]:
            bad_spatial_decay = quality_metrics['spatialDecaySlope'] > param['minSpatialDecaySlope']
        elif param["computeSpatialDecay"]:
            too_shallow_decay = quality_metrics["spatialDecaySlope"] < param["minSpatialDecaySlopeExp"]
            too_steep_decay = (
                quality_metrics["spatialDecaySlope"] > param["maxSpatialDecaySlopeExp"]
            )  
            bad_spatial_decay = np.logical_or(too_shallow_decay, too_steep_decay).squeeze()
        
        qm_table_list.append(bad_spatial_decay)
        qm_table_names.append("spatial decay")
    # classify as mua
    # ALL or ANY?

    too_few_total_spikes = quality_metrics["nSpikes"] < param["minNumSpikes"]
    too_many_spikes_missing = (
        quality_metrics["percentageSpikesMissing_gaussian"] > param["maxPercSpikesMissing"]
    )
    too_low_presence_ratio = (
        quality_metrics["presenceRatio"] < param["minPresenceRatio"]
    )
    too_many_RPVs = quality_metrics["fractionRPVs_estimatedTauR"] > param["maxRPVviolations"]

    qm_table_list.extend([too_few_total_spikes, too_many_spikes_missing, too_low_presence_ratio, too_many_RPVs])
    qm_table_names.extend(["# spikes", "% spikes missing", "presence ratio", "fraction RPVs"])

    if param["extractRaw"]:
        too_small_amplitude = (
            quality_metrics["rawAmplitude"] < param["minAmplitude"]
        ) 
        too_small_SNR = (
            quality_metrics["signalToNoiseRatio"] < param["minSNR"]
        )
        qm_table_list.extend([too_small_amplitude, too_small_SNR])
        qm_table_names.extend(["amplitude", "SNR"])

    if param["computeDrift"]:
        too_large_drift = (
            quality_metrics["maxDriftEstimate"] > param["maxDrift"]
        )
        qm_table_list.append(too_large_drift)
        qm_table_names.append("drift")

    # determine if ALL unit is somatic or non-somatic
    is_non_somatic = (
        (quality_metrics["troughToPeak2Ratio"] < param["minTroughToPeak2Ratio_nonSomatic"]) &
        (quality_metrics["mainPeak_before_width"] < param["minWidthFirstPeak_nonSomatic"]) &
        (quality_metrics["mainTrough_width"] < param["minWidthMainTrough_nonSomatic"]) &
        (quality_metrics["peak1ToPeak2Ratio"] > param["maxPeak1ToPeak2Ratio_nonSomatic"]) |
        (quality_metrics["mainPeakToTroughRatio"] > param["maxMainPeakToTroughRatio_nonSomatic"])
    )
    peak_main_to_trough = quality_metrics["mainPeakToTroughRatio"] > param["maxMainPeakToTroughRatio_nonSomatic"]
    peak1_to_peak2 = quality_metrics["peak1ToPeak2Ratio"] > param["maxPeak1ToPeak2Ratio_nonSomatic"]

    qm_table_list.extend([is_non_somatic, peak_main_to_trough, peak1_to_peak2])
    qm_table_names.extend(["non somatic", "peak(main) / trough", "peak1 / peak2"])


    if param["computeDistanceMetrics"]:
        too_low_iso_dist = quality_metrics['isolationDistance'] < param["isoDmin "]
        too_high_lratio = quality_metrics["Lratio"] > param["lratioMax"]

        qm_table_list.extend([too_low_iso_dist, too_high_lratio])
        qm_table_names.extend(["iso dist", "l ratio"])


    # DO this for the optional params
    qm_table = pd.DataFrame(
        qm_table_list, qm_table_names
    ).T
    return qm_table