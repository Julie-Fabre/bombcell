import time
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from tqdm.auto import tqdm

from bombcell.extract_raw_waveforms import manage_data_compression, extract_raw_waveforms
from bombcell.loading_utils import get_gain_spikeglx, load_ephys_data

# import matplotlib.pyplot as plt
import bombcell.quality_metrics as qm
from bombcell.save_utils import get_metric_keys, save_results



def show_times(
    times_spikes_missing_1,
    times_RPV_1,
    times_chunks_to_keep,
    times_spikes_missing_2,
    times_RPV_2,
    times_presence_ratio,
    times_max_drift,
    times_waveform_shape,
):
    print(f"The time the first spikes missing took: {times_spikes_missing_1.sum()}")
    print(f"The time the first RPV took: {times_RPV_1.sum()}")
    print(f"The time the time chunks took: {times_chunks_to_keep.sum()}")
    print(f"The time the second spikes missing took: {times_spikes_missing_2.sum()}")
    print(f"The time the second RPV took: {times_RPV_2.sum()}")
    print(f"The time the presence ratio took: {times_presence_ratio.sum()}")
    print(f"The time the max drift took: {times_max_drift.sum()}")
    print(f"The time the waveform shapes took: {times_waveform_shape.sum()}")


def print_unit_qm(quality_metrics, unit_idx, param, unit_type=None):
    print(f"For unit {unit_idx}:")
    print(
        f'n_peaks : {quality_metrics["n_peaks"][unit_idx]}, n_troughs : {quality_metrics["n_troughs"][unit_idx]}'
    )
    print(
        f'waveform_duration_peak_trough : {quality_metrics["waveform_duration_peak_trough"][unit_idx]:.3f}'
    )
    print(
        f'waveform_baseline : {quality_metrics["waveform_baseline"][unit_idx]}, spatial_decay_slope : {quality_metrics["spatial_decay_slope"][unit_idx]}'
    )
    print(
        f'percent_missing_gaussian : {quality_metrics["percent_missing_gaussian"][unit_idx]}, n_spikes : {quality_metrics["n_spikes"][unit_idx]}'
    )
    print(
        f'fraction_RPVs : {quality_metrics["fraction_RPVs"][unit_idx]}, presence_ratio : {quality_metrics["presence_ratio"][unit_idx]}'
    )

    if param["extract_raw_waveforms"]:
        print(
            f'raw_amplitude : {quality_metrics["raw_amplitude"][unit_idx]:.3f}, signal_to_noise_ratio : {quality_metrics["signal_to_noise_ratio"][unit_idx]:.3f}'
        )
    if param["compute_distance_metrics"]:
        print(
            f'max_drift_estimate : {quality_metrics["max_drift_estimate"][unit_idx]:.3f}'
        )

    print(f'Waveform IS somatic = {1 == quality_metrics["is_somatic"][unit_idx]}')

    if unit_type is not None:
        print(f"The Units is classed as {unit_type[unit_idx]}")


def print_qm_thresholds(param):
    print("Current threshold params:")
    print(
        f'max_n_peaks = {param["max_n_peaks"]}, max_n_troughs = {param["max_n_troughs"]}'
    )
    print(
        f'min_wv_duration = {param["min_wv_duration"]}, max_wv_duration = {param["max_wv_duration"]}'
    )
    print(
        f'max_wv_baseline_fraction = {param["max_wv_baseline_fraction"]}, min_spatial_decay_slope = {param["min_spatial_decay_slope"]}'
    )
    print(
        f'max_perc_spikes_missing = {param["max_perc_spikes_missing"]}, min_num_spikes_total = {param["min_num_spikes_total"]}'
    )
    print(
        f'max_RPV = {param["max_RPV"]}, min_presence_ratio = {param["min_presence_ratio"]}'
    )

    if param["extract_raw_waveforms"]:
        print(f'min_amplitude = {param["min_amplitude"]}, min_SNR = {param["min_SNR"]}')

    if param["compute_distance_metrics"]:
        print(f'max_drift = {param["max_drift"]}')


def show_somatic(quality_metrics, unit, is1, is2, is3):
    print(f'The max trough is {quality_metrics["trough"][unit]}')
    print(f'The main peak before is {quality_metrics["main_peak_before"][unit]}')
    print(f'The main peak after is {quality_metrics["main_peak_after"][unit]}')
    print(f'The first peak width is {quality_metrics["width_before"][unit]}')
    print(f'The trough_width is {quality_metrics["trough_width"][unit]}')

    if is1[unit] == 0:
        print("The trough is to small rel to peaks")
    if is2[unit] == 0:
        print("The first peak is too big")
    if is3[unit] == 0:
        print("The peak size to width is wrong")


def order_good_sites(good_sites, channel_pos):
    # make it so it goes from biggest to smallest
    reordered_idx = np.argsort(-channel_pos[good_sites, 1].squeeze())
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

    unit_id = unique_templates[this_unit]  # JF: this function needs some cleaning up

    max_channel = quality_metrics["peak_channels"][unit_id].squeeze()

    x, y = channel_positions[max_channel, :]

    x_dist = np.abs(channel_positions[:, 0] - x)
    near_x_dist = np.min(x_dist[x_dist != 0])

    not_these_x = np.argwhere(x_dist > near_x_dist)

    y_dist = np.abs(channel_positions[:, 1] - y)
    y_dist[not_these_x] = (
        y_dist.max()
    )  # set the bad x_to max y, this keeps the shape of the array
    good_sites = np.argsort(y_dist)[:16]

    ####
    # x, y = channel_positions[max_channel,:]

    # x_dist = np.abs(channel_positions[:,0] - x)
    # near_x = np.argmin(x_dist)

    # good_x_sites = np.argwhere( np.logical_and((x-50 < channel_positions[:,0]) == True, (channel_positions[:,0] < x+50) == True))
    # y_values = channel_positions[good_x_sites,1]

    # y_dist_to_max_site = np.abs(y_values - channel_positions[max_channel,1])
    # good_sites = np.argsort(y_dist_to_max_site,axis = 0 )[:16]
    # good_sites = good_x_sites[good_sites]

    reordered_good_sites = order_good_sites(good_sites, channel_positions)

    # ###
    # channels_with_same_x = np.argwhere(np.logical_and(channel_positions[:,0] <= channel_positions[max_channel, 0] +33,
    #                                 channel_positions[:,0] >= channel_positions[max_channel, 0] -33)) #4 shank probe

    # current_max_channel = channel_positions[max_channel, :]
    # distance_to_current_channel = np.square(channel_positions - current_max_channel)
    # sum_euclid = np.sum(distance_to_current_channel, axis = 1)

    # #find nearest x channels
    # near_x_channels = np.argwhere(distance_to_current_channel[:,0] <= np.sort(distance_to_current_channel[:,0])[50]).squeeze()

    # #from these enar x channel find the nearest y channels

    # use_these_channels = near_x_channels[np.argwhere(distance_to_current_channel[near_x_channels,1] <= np.sort(distance_to_current_channel[near_x_channels,1])[16])].squeeze()

    # #reordered_good_sites = order_good_sites(use_these_channels, channel_positions)
    # re_ordered_good_sites = use_these_channels

    return reordered_good_sites


def plot_raw_waveforms(
    quality_metrics, channel_positions, this_unit, waveform, unique_templates
):

    unit_id = unique_templates[this_unit]  # JF: this function needs some cleaning up

    fig = Figure(figsize=(4, 6), dpi=100)
    fig.set_tight_layout(False)

    main_ax = fig.add_axes([0.2, 0.2, 0.8, 0.8])
    main_ax_offset = 0.2
    main_ax_scale = 0.8

    good_channels = nearest_channels(
        quality_metrics, channel_positions, this_unit, unique_templates
    ).squeeze()

    min_x, min_y = channel_positions[good_channels[-2], [0, 1]].squeeze()
    max_x, maxy = channel_positions[good_channels[1], [0, 1]].squeeze()
    delta_x = (max_x - min_x) / 2
    delta_y = (maxy - min_y) / 18

    # may want to change so it find this for both units and selects the most extreme arguments
    # however i dont think tis will be necessary
    sub_min_y = np.nanmin(waveform[unit_id, :, good_channels])
    sub_max_y = np.nanmax(waveform[unit_id, :, good_channels])

    # shift each waveform so 0 is at the channel site, 1/9 is width of a y waveform plot
    waveform_y_offset = (
        (np.abs(sub_max_y) / (np.abs(sub_min_y) + np.abs(sub_max_y))) * 1 / 8
    )  # JF: i don't think is used

    # make the main scatter positiose site as scatter with opacity
    # main_ax.scatter(channel_positions[good_channels,0], channel_positions[good_channels,1], c = 'grey', alpha = 0.3)
    # main_ax.set_xlim(min_x - delta_x, max_x + delta_x)
    # main_ax.set_ylim(min_y - delta_y, maxy + delta_y)

    # rel_channel_positions = (channel_positions - channel_positions[good_channels].squeeze().min(axis = 0))/ (channel_positions[good_channels.squeeze()].max(axis = 0)  - channel_positions[good_channels].squeeze().min(axis = 0)) * 0.8
    # rel_channel_positions += main_ax_offset
    # for i in range(9):
    #     for j in range(2):
    #         #may need to change this positioning if units sizes are irregular
    #         # if j == 0:
    #         #     #The peak in the waveform is not half way, so maths says the x axis should be starting at
    #         #     #0.1 and 0.6 so the middle is at 0.25/0.76 however chosen these values so it loks better by eye
    #         #     ax =  fig.add_axes([main_ax_offset + main_ax_scale*0.25, main_ax_offset + main_ax_scale*(i/9 - 1/18 + waveform_y_offset), main_ax_scale*0.25, main_ax_scale*1/9])
    #         # if j == 1:
    #         #     ax = fig.add_axes([main_ax_offset + main_ax_scale*0.75, main_ax_offset + main_ax_scale*(i/9 - 1/18 + waveform_y_offset), main_ax_scale*0.25, main_ax_scale*1/9])

    #         if j == 0:
    #             #The peak in the waveform is not half way, so maths says the x axis should be starting at
    #             #0.1 and 0.6 so the middle is at 0.25/0.76 however chosen these values so it loks better by eye
    #             ax =  fig.add_axes([rel_channel_positions[good_channels,0][i*2 + j],rel_channel_positions[good_channels,1][i*2 + j], main_ax_scale*0.2, main_ax_scale*1/9])
    #         if j == 1:
    #             ax = fig.add_axes([rel_channel_positions[good_channels,0][i*2 + j],rel_channel_positions[good_channels,1][i*2 + j], main_ax_scale*0.22, main_ax_scale*1/9])

    #         ax.plot(waveform[unit_id,:,good_channels[i*2 + j]].squeeze(), color = 'g')

    #         ax.set_ylim(sub_min_y,sub_max_y)
    #         ax.set_axis_off()

    main_ax.set_xlim(min_x - delta_x, max_x + delta_x)
    main_ax.set_ylim(min_y - delta_y, maxy + delta_y)

    rel_channel_positions = (
        (channel_positions - channel_positions[good_channels].squeeze().min(axis=0))
        / (
            channel_positions[good_channels.squeeze()].max(axis=0)
            - channel_positions[good_channels].squeeze().min(axis=0)
        )
        * 0.8
    )
    rel_channel_positions += main_ax_offset
    for i in range(8):
        for j in range(2):
            # may need to change this positioning if units sizes are irregular
            # if j == 0:
            #     #The peak in the waveform is not half way, so maths says the x axis should be starting at
            #     #0.1 and 0.6 so the middle is at 0.25/0.76 however chosen these values so it loks better by eye
            #     ax =  fig.add_axes([main_ax_offset + main_ax_scale*0.25, main_ax_offset + main_ax_scale*(i/9 - 1/18 + waveform_y_offset), main_ax_scale*0.25, main_ax_scale*1/9])
            # if j == 1:
            #     ax = fig.add_axes([main_ax_offset + main_ax_scale*0.75, main_ax_offset + main_ax_scale*(i/9 - 1/18 + waveform_y_offset), main_ax_scale*0.25, main_ax_scale*1/9])

            if j == 0:
                # The peak in the waveform is not half way, so maths says the x axis should be starting at
                # 0.1 and 0.6 so the middle is at 0.25/0.76 however chosen these values so it loks better by eye
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

            ax.plot(waveform[unit_id, :, good_channels[i * 2 + j]].squeeze(), color="g")

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
    print_unit_qm(quality_metrics, this_unit, param, unit_type=unit_type)
    unit_id = unique_templates[this_unit]  # JF: i don't think is used

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
    dict
        The quality metrics dictionary
    """
    init_keys = [
        "phy_cluster_id",
        "cluster_id",
        "n_spikes"
        ] + get_metric_keys()

    quality_metrics = {}
    for k in init_keys:
        quality_metrics[k] = np.full(n_units, np.nan)

    # Use passed snr values if found
    if isinstance(snr, np.ndarray):
        quality_metrics["signal_to_noise_ratio"] = snr

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
):
    """
    This function runs all of the quality metric calculations

    Parameters
    ----------
    unique_templates : ndarray
        An of unique id for each unit
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

    Returns
    -------
    (dict, dict)
        The quality_metrics dictionary and the times taken for each section
    """
    # Collect the time it takes to run each section
    times_spikes_missing_1 = np.zeros(unique_templates.shape[0])
    times_RPV_1 = np.zeros(unique_templates.shape[0])
    times_chunks_to_keep = np.zeros(unique_templates.shape[0])
    times_spikes_missing_2 = np.zeros(unique_templates.shape[0])
    times_RPV_2 = np.zeros(unique_templates.shape[0])
    times_presence_ratio = np.zeros(unique_templates.shape[0])
    times_max_drift = np.zeros(unique_templates.shape[0])
    times_waveform_shape = np.zeros(unique_templates.shape[0])
    time_dist_metrics = np.zeros(unique_templates.shape[0])

    not_enough_spikes = np.zeros(unique_templates.size)
    bad_units = 0
    bar_description = "Computing bombcell quality metrics: {percentage:3.0f}%|{bar:10}| {n}/{total} units"
    for unit_idx in tqdm(range(unique_templates.size), bar_format=bar_description):
        this_unit = unique_templates[unit_idx]
        quality_metrics["phy_cluster_id"][unit_idx] = this_unit
        quality_metrics["cluster_id"][unit_idx] = this_unit

        these_spike_times = spike_times_seconds[spike_clusters == this_unit]
        these_amplitudes = template_amplitudes[spike_clusters == this_unit]

        # number of spikes
        quality_metrics["n_spikes"][unit_idx] = these_spike_times.shape[0]

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
        times_spikes_missing_1[unit_idx] = time.time() - time_tmp

        # fraction contamination
        time_tmp = time.time()
        fraction_RPVs, num_violations = qm.fraction_RP_violations(
            these_spike_times, these_amplitudes, time_chunks, param
        )
        times_RPV_1[unit_idx] = time.time() - time_tmp

        # get time chunks to keep
        time_tmp = time.time()
        (
            these_spike_times,
            these_amplitudes,
            these_spike_clusters,
            quality_metrics["use_these_times_start"][unit_idx],
            quality_metrics["use_these_times_stop"][unit_idx],
            quality_metrics["RPV_use_tauR_est"][unit_idx],
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
        times_chunks_to_keep[unit_idx] = time.time() - time_tmp

        use_these_times = np.array(
            (
                quality_metrics["use_these_times_start"][unit_idx],
                quality_metrics["use_these_times_stop"][unit_idx],
            )
        )
        # re-compute percentage spikes missing and RPV on time chunks
        time_tmp = time.time()
        (
            quality_metrics["percent_missing_gaussian"][unit_idx],
            quality_metrics["percent_missing_symmetric"][unit_idx],
        ) = qm.perc_spikes_missing(
            these_amplitudes, these_spike_times, use_these_times, param
        )
        times_spikes_missing_2[unit_idx] = time.time() - time_tmp

        time_tmp = time.time()
        fraction_RPVs, num_violations = qm.fraction_RP_violations(
            these_spike_times,
            these_amplitudes,
            use_these_times,
            param,
            use_this_tauR=quality_metrics["RPV_use_tauR_est"][unit_idx],
        )
        times_RPV_2[unit_idx] = time.time() - time_tmp

        quality_metrics["fraction_RPVs"][unit_idx] = fraction_RPVs[
            quality_metrics["RPV_use_tauR_est"][unit_idx].astype(int)
        ]

        # get presence ratio
        time_tmp = time.time()
        quality_metrics["presence_ratio"][unit_idx] = qm.presence_ratio(
            these_spike_times,
            quality_metrics["use_these_times_start"][unit_idx],
            quality_metrics["use_these_times_stop"][unit_idx],
            param,
        )
        times_presence_ratio[unit_idx] = time.time() - time_tmp

        # maximum cumulative drift estimate
        time_tmp = time.time()
        (
            quality_metrics["max_drift_estimate"][unit_idx],
            quality_metrics["cumulative_drift_estimate"][unit_idx],
        ) = qm.max_drift_estimate(
            pc_features,
            pc_features_idx,
            these_spike_clusters,
            these_spike_times,
            this_unit,
            channel_positions,
            param,
        )
        times_max_drift[unit_idx] = time.time() - time_tmp

        # number of spikes
        quality_metrics["n_spikes"][unit_idx] = these_spike_times.shape[0]

        # waveform
        time_tmp = time.time()
        waveform_baseline_window = np.array(
            (
                param["waveform_baseline_window_start"],
                param["waveform_baseline_window_stop"],
            )
        )

        (
            quality_metrics["n_peaks"][unit_idx],
            quality_metrics["n_troughs"][unit_idx],
            quality_metrics["waveform_duration_peak_trough"][unit_idx],
            quality_metrics["spatial_decay_slope"][unit_idx],
            quality_metrics["waveform_baseline_flatness"][unit_idx],
            quality_metrics["scnd_peak_to_trough_ratio"][unit_idx],
            quality_metrics["peak1_to_peak2_ratio"][unit_idx],
            quality_metrics["main_peak_to_trough_ratio"][unit_idx],
            quality_metrics["trough_to_peak2_ratio"][unit_idx],
            quality_metrics["peak_before_width"][unit_idx],
            quality_metrics["trough_width"][unit_idx],
        ) = qm.waveform_shape(
            template_waveforms,
            this_unit,
            quality_metrics["peak_channels"],
            channel_positions,
            waveform_baseline_window,
            param,
        )
        times_waveform_shape[unit_idx] = time.time() - time_tmp

        # amplitude
        if raw_waveforms_full is not None and param["extract_raw_waveforms"]:
            quality_metrics["raw_amplitude"][unit_idx] = qm.get_raw_amplitude(
                raw_waveforms_full[unit_idx], param["gain_to_uV"]
            )
        else:
            quality_metrics["raw_amplitude"][unit_idx] = np.nan

        time_tmp = time.time()
        if param["compute_distance_metrics"]:
            (
                quality_metrics["isolation_dist"][unit_idx],
                quality_metrics["l_ratio"][unit_idx],
                quality_metrics["silhouette_score"][unit_idx],
            ) = qm.get_distance_metrics(
                pc_features, pc_features_idx, this_unit, spike_clusters, param
            )
        time_dist_metrics = time.time() - time_tmp

    times = {
        "times_spikes_missing_1": times_spikes_missing_1, # JF: what is this?
        "times_RPV_1": times_RPV_1,
        "times_chunks_to_keep": times_chunks_to_keep,
        "times_spikes_missing_2": times_spikes_missing_2,
        "times_RPV_2": times_RPV_2,
        "times_presence_ratio": times_presence_ratio,
        "times_max_drift": times_max_drift,
        "times_waveform_shape": times_waveform_shape,
    }

    if param["compute_distance_metrics"]:
        times["time_dist_metrics"] = time_dist_metrics

    return quality_metrics, times


def run_bombcell(ks_dir, raw_dir, save_path, param):
    """
    This function runs the entire bombcell pipeline from input data paths

    Parameters
    ----------
    ks_dir : string
        The path to the KiloSort (or equivalent) save directory
    raw_dir : string
        The path to the raw data directory
    save_path : string
        The path to the directory to save the bombcell results
    param : dict
        The param dictionary

    Returns
    -------
    _type_
        _description_
    """
    (
        spike_times_samples,
        spike_clusters,
        template_waveforms,
        template_amplitudes,
        pc_features,
        pc_features_idx,
        channel_positions,
        good_channels,
    ) = load_ephys_data(ks_dir)

    # Extract or load in raw waveforms
    if raw_dir != None:
        raw_waveforms_full, raw_waveforms_peak_channel, SNR = extract_raw_waveforms(
            param,
            spike_clusters.squeeze(),
            spike_times_samples.squeeze(),
            param["re_extract_raw"],
            save_path,
        )
    else:
        raw_waveforms_full = None
        raw_waveforms_peak_channel = None
        SNR = None
        param["extract_raw_waveforms"] = False  # No waveforms to extract!

    # pre-load peak channels
    peak_channels = qm.get_waveform_peak_channel(template_waveforms)

    # Remove duplicate spikes
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
        peak_channels,
    ) = qm.remove_duplicate_spikes(
        spike_times_samples,
        spike_clusters,
        template_amplitudes,
        peak_channels,
        save_path,
        param,
        pc_features=pc_features,
        raw_waveforms_full=raw_waveforms_full,
        raw_waveforms_peak_channel=raw_waveforms_peak_channel,
        signal_to_noise_ratio=SNR,
    )

    # Divide recording into time chunks
    spike_times_seconds = spike_times_samples / param["ephys_sample_rate"]
    if param["compute_time_chunks"]:
        time_chunks = np.arange(
            np.min(spike_times_seconds),
            np.max(spike_times_seconds),
            param["delta_time_chunk"],
        )
    else:
        time_chunks = np.array(
            (np.min(spike_times_seconds), np.max(spike_times_seconds))
        )

    unique_templates = non_empty_units

    # Initialize quality metrics dictionary
    n_units = unique_templates.size
    quality_metrics = create_quality_metrics_dict(n_units, snr=SNR)
    quality_metrics["peak_channels"] = peak_channels

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
    )

    unit_type, unit_type_string = qm.get_quality_unit_type(
        param, quality_metrics
    )  # JF: this should be inside bc.get_all_quality_metrics

    save_results(
        quality_metrics,
        unit_type_string,
        unique_templates,
        param,
        raw_waveforms_full,
        raw_waveforms_peak_channel,
        save_path,
    )  # JF: this should be inside bc.get_all_quality_metrics

    return (
        quality_metrics,
        param,
        unit_type,
        unit_type_string,
    )


def make_qm_table(quality_metrics, param, unique_templates, unit_type):
    # classify noise
    nan_result = np.isnan(quality_metrics["n_peaks"])

    too_many_peaks = quality_metrics["n_peaks"] > param["max_n_peaks"]

    too_many_troughs = quality_metrics["n_troughs"] > param["max_n_troughs"]

    too_short_waveform = (
        quality_metrics["waveform_duration_peak_trough"] < param["min_wv_duration"]
    )

    too_long_waveform = (
        quality_metrics["waveform_duration_peak_trough"] > param["max_wv_duration"]
    )

    too_noisy_baseline = (
        quality_metrics["waveform_baseline"] > param["max_wv_baseline_fraction"]
    )

    ##
    too_shallow_decay = quality_metrics["exp_decay"] > param["min_spatial_decay_slope"]
    to_steep_decay = (
        quality_metrics["exp_decay"] < param["max_spatial_decay_slope"]
    )  # JF: i don't think is used, but it should be?
    # classify as mua
    # ALL or ANY?

    too_few_total_spikes = quality_metrics["n_spikes"] < param["min_num_spikes_total"]

    too_many_spikes_missing = (
        quality_metrics["percent_missing_gaussian"] > param["max_perc_spikes_missing"]
    )

    too_low_presence_ratio = (
        quality_metrics["presence_ratio"] < param["min_presence_ratio"]
    )

    too_many_RPVs = quality_metrics["fraction_RPVs"] > param["max_RPV"]

    if param["extract_raw_waveforms"]:
        too_small_amplitude = (
            quality_metrics["raw_amplitude"] < param["min_amplitude"]
        )  # JF: i don't think is used, but it should be?

        too_small_SNR = (
            quality_metrics["signal_to_noise_ratio"] < param["min_SNR"]
        )  # JF: i don't think is used, but it should be?

    if param["compute_drift"]:
        too_large_drift = (
            quality_metrics["max_drift_estimate"] > param["max_drift"]
        )  # JF: i don't think is used, but it should be?

    # determine if ALL unit is somatic or non-somatic
    param["non_somatic_trough_peak_ratio"] = 1.25
    param["non_somatic_peak_before_to_after_ratio"] = 1.2
    # somatic == 0, non_somatic == 1
    is_somatic = np.ones(unique_templates.size)

    is_somatic[
        (
            quality_metrics["trough"]
            / np.max(
                (
                    quality_metrics["main_peak_before"],
                    quality_metrics["main_peak_after"],
                ),
                axis=0,
            )
        )
        < param["non_somatic_trough_peak_ratio"]
    ] = 0

    is_somatic[
        (quality_metrics["main_peak_before"] / quality_metrics["main_peak_after"])
        > param["non_somatic_peak_before_to_after_ratio"]
    ] = 0

    is_somatic[
        (
            quality_metrics["main_peak_before"] * param["first_peak_ratio"]
            > quality_metrics["main_peak_after"]
        )
        & (quality_metrics["width_before"] < param["min_width_first_peak"])
        & (
            quality_metrics["main_peak_before"] * param["min_main_peak_to_trough_ratio"]
            > quality_metrics["trough"]
        )
        & (quality_metrics["trough_width"] < param["min_width_main_trough"])
    ] = 0

    # is_somatic[np.isnan(quality_metrics['trough'])] = 0
    quality_metrics["is_somatic_new"] = is_somatic

    not_somatic = is_somatic == 1

    qm_table_array = np.array(
        (
            nan_result,
            too_many_peaks,
            too_many_troughs,
            too_short_waveform,
            too_long_waveform,
            too_noisy_baseline,
            too_shallow_decay,
            too_few_total_spikes,
            too_many_spikes_missing,
            too_many_RPVs,
            too_low_presence_ratio,
            not_somatic,
        )
    )

    qm_table_array = np.vstack((qm_table_array, unit_type))
    qm_table_array = np.vstack((unique_templates, qm_table_array))
    # DO this for the optional params
    qm_table = pd.DataFrame(
        qm_table_array,
        index=[
            "Original ID",
            "NaN result",
            "Peaks",
            "Troughs",
            "Waveform Min Length",
            "Waveform Max Length",
            "Baseline",
            "Spatial Decay",
            "Min Spikes",
            "Missing Spikes",
            "RPVs",
            "Presence Ratio",
            "Somatic",
            "Good Unit",
        ],
    ).T
    return qm_table


def manage_if_raw_data(raw_dir, gain_to_uV):
    """
    This function handles the decompression of raw data and extraction of gain if a raw_dir is given

    Parameters
    ----------
    raw_dir : str / None
        Either a string with the path to the raw data directory or None

    Returns
    -------
    tuple
        The raw data path the meta directory path and the gain if applicable
    """
    if raw_dir != None:
        ephys_raw_data, meta_path = manage_data_compression(
            raw_dir, decompressed_data_local=raw_dir
        )
        if gain_to_uV is not None and not np.isnan(gain_to_uV):
            gain_to_uV = get_gain_spikeglx(meta_path)
        else:
            gain_to_uV = np.nan
        return ephys_raw_data, meta_path, gain_to_uV
    else:
        return None, None, None
