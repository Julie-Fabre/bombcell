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


def clean_inf_values(quality_metrics, columns=None):
    """
    Convert inf values to nan across specified columns or all numeric columns.
    
    Parameters
    ----------
    quality_metrics : dict
        Dictionary containing quality metrics data
    columns : list, optional
        List of column names to process. If None, processes all columns that contain numeric data.
        
    Returns
    -------
    dict
        Dictionary with inf values converted to nan
    """
    # Create a copy to avoid modifying the original
    cleaned = quality_metrics.copy()
    
    # If no specific columns specified, find all numeric columns
    if columns is None:
        columns = []
        for key, value in cleaned.items():
            if isinstance(value, np.ndarray):
                # Check if the array contains numeric data
                if len(value) > 0 and np.issubdtype(value.dtype, np.number):
                    columns.append(key)
    
    # Convert inf to nan for specified columns
    for col in columns:
        if col in cleaned:
            cleaned[col] = np.where(cleaned[col] == np.inf, np.nan, cleaned[col])
    
    return cleaned


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
    main_ax.set_xlabel(r"Xpos ($\mu$m)", size=14)
    main_ax.set_ylabel(r"Ypos ($\mu$m)", size=14)

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



def _precompute_unit_gui_data(unit_idx, unit_id, template_waveforms, quality_metrics, 
                             spike_clusters, template_amplitudes, channel_positions, 
                             gui_data, param, per_bin_data=None):
    """Helper function to precompute GUI data for a single unit during quality metrics computation"""
    try:
        template = template_waveforms[unit_idx]
        max_ch = np.argmax(np.ptp(template, axis=0))
        waveform = template[:, max_ch]
        
        # Use exact same peak/trough detection as quality metrics (inlined to avoid scope issues)
        try:
            from scipy.signal import find_peaks
            
            # Use same parameters as waveform_shape
            min_thresh_detect_peaks_troughs = param.get("minThreshDetectPeaksTroughs", 0.25)
            
            # Handle baseline artifacts (same as waveform_shape)
            first_valid_index = 0
            if len(waveform) > 4 and np.any(np.abs(waveform[0:4]) > 2 * np.nanstd(waveform)):
                first_valid_index = 5
            
            # Same prominence calculation as waveform_shape
            min_prominence = min_thresh_detect_peaks_troughs * np.max(np.abs(waveform[first_valid_index:]))
            
            # Find troughs (same logic as waveform_shape)
            trough_locs, trough_dict = find_peaks(
                waveform[first_valid_index:] * -1, prominence=min_prominence, width=0
            )
            
            if trough_locs.size > 1:
                trough_locs += first_valid_index
                max_trough_idx = np.nanargmax(trough_dict["prominences"])
                trough_locs = trough_locs[max_trough_idx:max_trough_idx+1]  # Keep only main trough
            elif trough_locs.size == 1:
                trough_locs += first_valid_index
                trough_locs = np.atleast_1d(trough_locs)
            
            if trough_locs.size == 0:
                trough_locs = np.array([np.nanargmin(waveform[first_valid_index:]) + first_valid_index])
            
            # Get main trough location
            main_trough_idx = np.nanargmin(waveform[trough_locs])
            trough_loc = trough_locs[main_trough_idx]
            
            # Find peaks before and after trough (same logic as waveform_shape)
            peaks_before_locs = np.array([])
            peaks_after_locs = np.array([])
            
            # Peaks before trough
            if trough_loc > 2:
                peaks_before_locs, peaks_before_dict = find_peaks(
                    waveform[first_valid_index:trough_loc], prominence=min_prominence, width=0
                )
                peaks_before_locs += first_valid_index
                if peaks_before_locs.shape[0] > 1:
                    max_peak = np.nanargmax(peaks_before_dict["prominences"])
                    peaks_before_locs = peaks_before_locs[max_peak:max_peak+1]
            
            # Peaks after trough
            if waveform.shape[0] - trough_loc > 2:
                peaks_after_locs, peaks_after_dict = find_peaks(
                    waveform[trough_loc:], prominence=min_prominence, width=0
                )
                peaks_after_locs += trough_loc
                if peaks_after_locs.shape[0] > 1:
                    max_peak = np.nanargmax(peaks_after_dict["prominences"])
                    peaks_after_locs = peaks_after_locs[max_peak:max_peak+1]
            
            # Handle forced peaks (same logic as waveform_shape)
            used_max_before = False
            used_max_after = False
            
            if peaks_before_locs.size == 0:
                if trough_loc > 2:
                    peaks_before_locs, peaks_before_dict = find_peaks(
                        waveform[:trough_loc], prominence=0.01 * np.max(np.abs(waveform)), width=0
                    )
                    peaks_before_locs += first_valid_index
                    if peaks_before_locs.shape[0] > 1:
                        max_peak = np.nanargmax(peaks_before_dict["prominences"])
                        peaks_before_locs = peaks_before_locs[max_peak:max_peak+1]
                
                if peaks_before_locs.size == 0:
                    peaks_before_locs = np.array([np.nanargmax(waveform[first_valid_index:trough_loc]) + first_valid_index])
                used_max_before = True
            
            if peaks_after_locs.size == 0:
                if waveform.shape[0] - trough_loc > 2:
                    peaks_after_locs, peaks_after_dict = find_peaks(
                        waveform[trough_loc:], prominence=0.01 * np.max(np.abs(waveform)), width=0
                    )
                    if peaks_after_locs.shape[0] > 1:
                        max_peak = np.nanargmax(peaks_after_dict["prominences"])
                        peaks_after_locs = peaks_after_locs[max_peak:max_peak+1] + trough_loc
                    elif peaks_after_locs.shape[0] == 1:
                        peaks_after_locs += trough_loc
                
                if peaks_after_locs.size == 0:
                    peaks_after_locs = np.array([np.nanargmax(waveform[trough_loc:]) + trough_loc])
                used_max_after = True
            
            # Apply final filtering (same as waveform_shape)
            if used_max_before and used_max_after:
                if waveform[peaks_before_locs[0]] > waveform[peaks_after_locs[0]]:
                    used_max_before = False
                else:
                    used_max_after = False
            
            # Combine peaks and apply prominence filtering
            peaks_before_locs = np.atleast_1d(peaks_before_locs)
            peaks_after_locs = np.atleast_1d(peaks_after_locs)
            
            main_peak_before = np.max(waveform[peaks_before_locs]) if peaks_before_locs.size > 0 else 0
            main_peak_after = np.max(waveform[peaks_after_locs]) if peaks_after_locs.size > 0 else 0
            
            # Final peak selection (same logic as waveform_shape lines 1335-1344)
            if used_max_before and (main_peak_before < min_prominence * 0.5):
                final_peak_locs = peaks_after_locs
            elif used_max_after and (main_peak_after < min_prominence * 0.5):
                final_peak_locs = peaks_before_locs
            else:
                final_peak_locs = np.hstack((peaks_before_locs, peaks_after_locs))
            
            gui_data['peak_locations'][unit_id] = final_peak_locs.tolist()
            gui_data['trough_locations'][unit_id] = trough_locs.tolist()
            
        except ImportError:
            # Fallback without scipy
            gui_data['peak_locations'][unit_id] = []
            gui_data['trough_locations'][unit_id] = []
        
        # Waveform scaling info
        gui_data['waveform_scaling'][unit_id] = {
            'max_channel': max_ch,
            'scaling_factor': np.ptp(waveform) * 2.5 if waveform.size > 0 else 1.0
        }
        
        # Spatial decay fit if channel positions available
        if channel_positions is not None and max_ch < len(channel_positions):
            max_pos = channel_positions[max_ch]
            nearby_channels = []
            distances = []
            amplitudes = []
            
            for ch in range(template.shape[1]):
                if ch < len(channel_positions):
                    distance = np.sqrt(np.sum((channel_positions[ch] - max_pos)**2))
                    if distance < 100:  # Within 100μm
                        nearby_channels.append(ch)
                        distances.append(distance)
                        amplitudes.append(np.max(np.abs(template[:, ch])))
            
            gui_data['channel_arrangements'][unit_id] = nearby_channels
            
            # Spatial decay fit
            if len(distances) >= 3 and len(amplitudes) >= 3:
                try:
                    from scipy.optimize import curve_fit
                    
                    def exp_decay(x, a, b):
                        return a * np.exp(-b * x)
                    
                    valid_mask = (np.array(distances) > 0) & (np.array(amplitudes) > 0)
                    if np.sum(valid_mask) >= 3:
                        dist_fit = np.array(distances)[valid_mask]
                        amp_fit = np.array(amplitudes)[valid_mask]
                        
                        popt, _ = curve_fit(exp_decay, dist_fit, amp_fit, 
                                          p0=[np.max(amp_fit), 0.01], maxfev=1000)
                        
                        x_smooth = np.linspace(0, np.max(dist_fit), 100)
                        y_smooth = exp_decay(x_smooth, *popt)
                        
                        gui_data['spatial_decay_fits'][unit_id] = {
                            'distances': distances,
                            'amplitudes': amplitudes,
                            'fit_x': x_smooth,
                            'fit_y': y_smooth,
                            'fit_params': popt
                        }
                except:
                    pass  # Skip if fitting fails
        
        # Amplitude distribution fit
        spike_mask = spike_clusters == unit_id
        if np.sum(spike_mask) > 50:
            unit_amplitudes = template_amplitudes[spike_mask]
            try:
                hist, bin_edges = np.histogram(unit_amplitudes, bins=50, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                
                from scipy.optimize import curve_fit
                
                def gaussian(x, a, mu, sigma):
                    return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
                
                mu_init = np.mean(unit_amplitudes)
                sigma_init = np.std(unit_amplitudes)
                a_init = np.max(hist)
                
                popt, _ = curve_fit(gaussian, bin_centers, hist, 
                                  p0=[a_init, mu_init, sigma_init], maxfev=1000)
                
                x_smooth = np.linspace(np.min(unit_amplitudes), np.max(unit_amplitudes), 100)
                y_smooth = gaussian(x_smooth, *popt)
                
                cutoff_val = mu_init - 2 * sigma_init
                percent_missing = np.sum(unit_amplitudes < cutoff_val) / len(unit_amplitudes) * 100
                
                gui_data['amplitude_fits'][unit_id] = {
                    'amplitudes': unit_amplitudes,
                    'hist': hist,
                    'bin_centers': bin_centers,
                    'fit_x': x_smooth,
                    'fit_y': y_smooth,
                    'percent_missing': percent_missing,
                    'fit_params': popt
                }
            except:
                pass  # Skip if fitting fails
        
        # ACG placeholder (computed lazily in GUI)
        gui_data['acg_data'][unit_id] = None
        
        # Store per-bin data for time bin metrics plotting
        if per_bin_data:
            if 'per_bin_metrics' not in gui_data:
                gui_data['per_bin_metrics'] = {}
            gui_data['per_bin_metrics'][unit_id] = per_bin_data
        
    except Exception as e:
        if param.get("verbose", False):
            print(f"Warning: GUI data precomputation failed for unit {unit_id}: {e}")


def _save_gui_data(gui_data, save_path, unique_templates, param):
    """Helper function to save GUI data"""
    try:
        import pickle
        import os
        
        gui_folder = os.path.join(save_path, "for_GUI")
        os.makedirs(gui_folder, exist_ok=True)
        gui_data_path = os.path.join(gui_folder, "gui_data.pkl")
        
        with open(gui_data_path, 'wb') as f:
            pickle.dump(gui_data, f)
            
        if param.get("verbose", False):
            spatial_decay_count = len(gui_data['spatial_decay_fits'])
            amplitude_fits_count = len(gui_data['amplitude_fits'])
            print(f"GUI visualization data saved to: {gui_data_path}")
            print(f"   Generated spatial decay fits: {spatial_decay_count}/{len(unique_templates)} units")
            print(f"   Generated amplitude fits: {amplitude_fits_count}/{len(unique_templates)} units")
        
    except Exception as e:
        if param.get("verbose", False):
            print(f"GUI data saving failed (GUI will still work): {e}")


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
    gui_data=None,
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
    # Initialize GUI data structure for precomputation during quality metrics
    if gui_data is None:
        gui_data = {
            'peak_locations': {},
            'trough_locations': {},
            'peak_loc_for_duration': {},
            'trough_loc_for_duration': {},
            'peak_trough_labels': {},
            'duration_lines': {},
            'spatial_decay_fits': {},
            'amplitude_fits': {},
            'channel_arrangements': {},
            'waveform_scaling': {},
            'acg_data': {}
        }
    
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
            perc_missing_per_bin_data
        ) = qm.perc_spikes_missing(
            these_amplitudes, these_spike_times, time_chunks, param, return_per_bin=True
        )
        runtimes_spikes_missing_1[unit_idx] = time.time() - time_tmp

        # fraction contamination
        time_tmp = time.time()
        fraction_RPVs, num_violations, rpv_per_bin_data = qm.fraction_RP_violations(
            these_spike_times, these_amplitudes, time_chunks, param, return_per_bin=True
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
            drift_per_bin_data
        ) = qm.max_drift_estimate(
            pc_features,
            pc_features_idx,
            these_spike_clusters,
            these_spike_times,
            this_unit,
            channel_positions,
            param,
            return_per_bin=True
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
            peak_locs_gui,
            trough_locs_gui,
            peak_loc_for_duration_gui,
            trough_loc_for_duration_gui,
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

        # Store GUI-specific data if it's being precomputed
        if gui_data is not None:
            # Initialize GUI data keys if they don't exist
            if 'peak_locations' not in gui_data:
                gui_data['peak_locations'] = {}
            if 'trough_locations' not in gui_data:
                gui_data['trough_locations'] = {}
            if 'peak_loc_for_duration' not in gui_data:
                gui_data['peak_loc_for_duration'] = {}
            if 'trough_loc_for_duration' not in gui_data:
                gui_data['trough_loc_for_duration'] = {}
            
            # Store the data, handling numpy arrays and NaN values properly
            try:
                if hasattr(peak_locs_gui, '__len__') and len(peak_locs_gui) > 0:
                    gui_data['peak_locations'][this_unit] = peak_locs_gui.tolist() if hasattr(peak_locs_gui, 'tolist') else peak_locs_gui
                else:
                    gui_data['peak_locations'][this_unit] = []
                    
                if hasattr(trough_locs_gui, '__len__') and len(trough_locs_gui) > 0:
                    gui_data['trough_locations'][this_unit] = trough_locs_gui.tolist() if hasattr(trough_locs_gui, 'tolist') else trough_locs_gui
                else:
                    gui_data['trough_locations'][this_unit] = []
                    
                gui_data['peak_loc_for_duration'][this_unit] = peak_loc_for_duration_gui if not np.isnan(peak_loc_for_duration_gui) else None
                gui_data['trough_loc_for_duration'][this_unit] = trough_loc_for_duration_gui if not np.isnan(trough_loc_for_duration_gui) else None
            except Exception as e:
                # Fallback to empty if there's any issue
                gui_data['peak_locations'][this_unit] = []
                gui_data['trough_locations'][this_unit] = []
                gui_data['peak_loc_for_duration'][this_unit] = None
                gui_data['trough_loc_for_duration'][this_unit] = None

        # amplitude
        if raw_waveforms_full is not None and param["extractRaw"] and param['gain_to_uV'] is not None:
            # Use the template's peak channel for raw amplitude calculation
            template_peak_channel = quality_metrics["maxChannels"][unit_idx]
            quality_metrics["rawAmplitude"][unit_idx] = qm.get_raw_amplitude(
                raw_waveforms_full[unit_idx], param["gain_to_uV"], peak_channel=template_peak_channel
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

        # Precompute GUI data during quality metrics computation
        if unit_idx < len(template_waveforms):
            # Collect per-bin data for this unit
            unit_per_bin_data = {
                'perc_missing': perc_missing_per_bin_data,
                'rpv': rpv_per_bin_data,
                'drift': drift_per_bin_data
            }
            _precompute_unit_gui_data(unit_idx, this_unit, template_waveforms, quality_metrics, 
                                    spike_clusters, template_amplitudes, channel_positions, 
                                    gui_data, param, unit_per_bin_data)

    # Save GUI data after processing all units
    if param.get("verbose", False):
        print("\nSaving GUI visualization data...")
    _save_gui_data(gui_data, save_path, unique_templates, param)

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


def run_bombcell(ks_dir, save_path, param, save_figures=False, return_figures=False):
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
    save_figures : bool, optional
        If True, saves the generated figures to disk, by default False
    return_figures : bool, optional
        If True, returns the generated figures, by default False

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
    figures : dict, optional
        If return_figures=True, returns dictionary with figure objects:
        'waveforms_overlay', 'upset_plots', 'histograms'
    """
    
    if param.get("verbose", False):
        print("🚀 Starting BombCell quality metrics pipeline...")
        print(f"📁 Processing data from: {ks_dir}")
        print(f"Results will be saved to: {save_path}")
        print("\nLoading ephys data...")
    
    (
        spike_times_samples,
        spike_clusters, # actually spike_templates, but they're the same in bombcell
        template_waveforms,
        template_amplitudes,
        pc_features,
        pc_features_idx,
        channel_positions,
    ) = load_ephys_data(ks_dir)
    
    if param.get("verbose", False):
        print(f"Loaded ephys data: {len(np.unique(spike_clusters))} units, {len(spike_times_samples):,} spikes")

    # pre-load peak channels from templates before extracting raw waveforms
    maxChannels = qm.get_waveform_peak_channel(template_waveforms)

    # Extract or load in raw waveforms
    if param["raw_data_file"] is not None:
        # Handle data decompression if needed
        if param.get("decompress_data", False):
            if param.get("verbose", False):
                print("\n📦 Checking for compressed data...")
            from bombcell.extract_raw_waveforms import decompress_data_if_needed
            param["raw_data_file"] = decompress_data_if_needed(
                param["raw_data_file"], 
                decompress_data=param["decompress_data"]
            )
        
        if param.get("verbose", False):
            print("\n🔍 Extracting raw waveforms...")
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
            maxChannels,  # Pass template peak channels
        )
    else:
        raw_waveforms_full = None
        raw_waveforms_peak_channel = None
        signal_to_noise_ratio = None
        raw_waveforms_id_match = None
        param["extractRaw"] = False  # No waveforms to extract!

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
    if param.get("verbose", False):
        print(f"\n⚙️ Computing quality metrics for {n_units} units...")
        print("   (Progress bar will appear below)")
    
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

    if param.get("verbose", False):
        print("\n🏷️ Classifying units (good/MUA/noise/non-soma)...")
    
    unit_type, unit_type_string = qm.get_quality_unit_type(
        param, quality_metrics
    )  # JF: this should be inside bc.get_all_quality_metrics

    # Override param settings if save_figures is explicitly set
    if save_figures:
        param["savePlots"] = True
        if param.get("plotsSaveDir") is None:
            param["plotsSaveDir"] = str(Path(save_path) / "bombcell_plots")
    
    figures = None
    if param.get("verbose", False):
        print("\nGenerating summary plots...")
    
    # Call plot_summary_data with return_figures parameter if needed
    if return_figures:
        figures = plot_summary_data(quality_metrics, template_waveforms, unit_type, unit_type_string, param, return_figures=True)
    else:
        plot_summary_data(quality_metrics, template_waveforms, unit_type, unit_type_string, param)

    if param.get("verbose", False):
        print("\nSaving results...")
    
    save_results(
        quality_metrics,
        unit_type_string,
        unique_templates,
        param,
        raw_waveforms_full,
        raw_waveforms_peak_channel,
        raw_waveforms_id_match,
        save_path,
        ks_dir,
    )  

    if return_figures and figures is not None:
        return (
            quality_metrics,
            param,
            unit_type,
            unit_type_string,
            figures,
        )
    else:
        return (
            quality_metrics,
            param,
            unit_type,
            unit_type_string,
        )


def run_bombcell_unit_match(ks_dir, save_path, raw_file=None, meta_file=None, kilosort_version=4, gain_to_uV=None, save_figures=False, return_figures=False):
    """
    This function runs bombcell pipeline with parameters optimized for UnitMatch
    
    Parameters
    ----------
    ks_dir : string
        The path to the KiloSort (or equivalent) save directory
    save_path : string
        The path to the directory to save the bombcell results
    raw_file : string, optional
        The path to the raw data file
    meta_file : string, optional
        The path to the meta file
    kilosort_version : int, optional
        The kilosort version used (default: 4)
    gain_to_uV : float, optional
        The gain to microvolts conversion factor
    save_figures : bool, optional
        If True, saves the generated figures to disk, by default False
    return_figures : bool, optional
        If True, returns the generated figures, by default False
        
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
    figures : dict, optional
        If return_figures=True, returns dictionary with figure objects:
        'waveforms_overlay', 'upset_plots', 'histograms'
    """
    from bombcell.default_parameters import get_unit_match_parameters
    
    # Get unit match specific parameters
    param = get_unit_match_parameters(ks_dir, raw_file, kilosort_version, meta_file, gain_to_uV)
    
    if param.get("verbose", False):
        print("🚀 Running BombCell with UnitMatch parameters...")
        print(f"   - Extracting {param['nRawSpikesToExtract']} raw spikes per unit")
        print(f"   - Saving multiple raw waveforms: {param['saveMultipleRaw']}")
        print(f"   - Detrending waveforms: {param['detrendWaveform']}")
    
    # Run bombcell with unit match parameters
    return run_bombcell(ks_dir, save_path, param, save_figures=save_figures, return_figures=return_figures)


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

    # Only evaluate noise metrics for units actually classified as NOISE
    is_noise_unit = unit_type_string == 'NOISE'
    
    too_many_peaks = np.full(len(unit_type_string), False, dtype=bool)
    too_many_peaks[is_noise_unit] = quality_metrics["nPeaks"][is_noise_unit] > param["maxNPeaks"]
    
    too_many_troughs = np.full(len(unit_type_string), False, dtype=bool)
    too_many_troughs[is_noise_unit] = quality_metrics["nTroughs"][is_noise_unit] > param["maxNTroughs"]
    
    duration = np.full(len(unit_type_string), False, dtype=bool)
    if np.any(is_noise_unit):
        too_short_waveform = quality_metrics["waveformDuration_peakTrough"] < param["minWvDuration"]
        too_long_waveform = quality_metrics["waveformDuration_peakTrough"] > param["maxWvDuration"]
        duration[is_noise_unit] = (too_short_waveform | too_long_waveform)[is_noise_unit]
    
    too_noisy_baseline = np.full(len(unit_type_string), False, dtype=bool)
    too_noisy_baseline[is_noise_unit] = (
        quality_metrics["waveformBaselineFlatness"][is_noise_unit] > param["maxWvBaselineFraction"]
    )
    
    peak2_to_trough = np.full(len(unit_type_string), False, dtype=bool)
    peak2_to_trough[is_noise_unit] = (
        quality_metrics["scndPeakToTroughRatio"][is_noise_unit] > param["maxScndPeakToTroughRatio_noise"]
    )

    qm_table_list.extend([too_many_peaks, too_many_troughs, duration, too_noisy_baseline, peak2_to_trough])
    qm_table_names.extend(["# peaks", "# troughs", "waveform duration", "baseline flatness", "peak2 / trough"])

    if param["computeSpatialDecay"]:
        bad_spatial_decay = np.full(len(unit_type_string), False, dtype=bool)
        if np.any(is_noise_unit):
            if param["spDecayLinFit"]:
                bad_spatial_decay[is_noise_unit] = (
                    quality_metrics['spatialDecaySlope'][is_noise_unit] < param['minSpatialDecaySlope']
                )
            else:
                too_shallow_decay = quality_metrics["spatialDecaySlope"] < param["minSpatialDecaySlopeExp"]
                too_steep_decay = quality_metrics["spatialDecaySlope"] > param["maxSpatialDecaySlopeExp"]
                bad_spatial_decay[is_noise_unit] = (too_shallow_decay | too_steep_decay)[is_noise_unit]
        
        qm_table_list.append(bad_spatial_decay)
        qm_table_names.append("spatial decay")
    # classify as mua
    # ALL or ANY?

    # Only evaluate MUA metrics for units actually classified as MUA 
    is_mua_unit = (unit_type_string == 'MUA')
    
    too_many_spikes_missing = np.full(len(unit_type_string), False, dtype=bool)
    too_many_spikes_missing[is_mua_unit] = (
        quality_metrics["percentageSpikesMissing_gaussian"][is_mua_unit] > param["maxPercSpikesMissing"]
    )
    
    too_low_presence_ratio = np.full(len(unit_type_string), False, dtype=bool)
    too_low_presence_ratio[is_mua_unit] = (
        quality_metrics["presenceRatio"][is_mua_unit] < param["minPresenceRatio"]
    )
    
    too_few_total_spikes = np.full(len(unit_type_string), False, dtype=bool)
    too_few_total_spikes[is_mua_unit] = (
        quality_metrics["nSpikes"][is_mua_unit] < param["minNumSpikes"]
    )
    
    too_many_RPVs = np.full(len(unit_type_string), False, dtype=bool)
    too_many_RPVs[is_mua_unit] = (
        quality_metrics["fractionRPVs_estimatedTauR"][is_mua_unit] > param["maxRPVviolations"]
    )

    qm_table_list.extend([too_many_spikes_missing, too_low_presence_ratio, too_few_total_spikes, too_many_RPVs])
    qm_table_names.extend(["% spikes missing", "presence ratio", "# spikes", "fraction RPVs"])

    if param["extractRaw"]:
        too_small_amplitude = np.full(len(unit_type_string), False, dtype=bool)
        too_small_amplitude[is_mua_unit] = (
            quality_metrics["rawAmplitude"][is_mua_unit] < param["minAmplitude"]
        )
        
        too_small_SNR = np.full(len(unit_type_string), False, dtype=bool)
        too_small_SNR[is_mua_unit] = (
            quality_metrics["signalToNoiseRatio"][is_mua_unit] < param["minSNR"]
        )
        qm_table_list.extend([too_small_amplitude, too_small_SNR])
        qm_table_names.extend(["amplitude", "SNR"])

    if param["computeDrift"]:
        too_large_drift = np.full(len(unit_type_string), False, dtype=bool)
        too_large_drift[is_mua_unit] = (
            quality_metrics["maxDriftEstimate"][is_mua_unit] > param["maxDrift"]
        )
        qm_table_list.append(too_large_drift)
        qm_table_names.append("max. drift")

    # determine if ALL unit is somatic or non-somatic
    is_non_somatic = (
        (quality_metrics["troughToPeak2Ratio"] < param["minTroughToPeak2Ratio_nonSomatic"]) &
        (quality_metrics["mainPeak_before_width"] < param["minWidthFirstPeak_nonSomatic"]) &
        (quality_metrics["mainTrough_width"] < param["minWidthMainTrough_nonSomatic"]) &
        (quality_metrics["peak1ToPeak2Ratio"] > param["maxPeak1ToPeak2Ratio_nonSomatic"]) |
        (quality_metrics["mainPeakToTroughRatio"] > param["maxMainPeakToTroughRatio_nonSomatic"])
    )
    # Only evaluate non-somatic metrics for units actually classified as non-somatic
    is_non_somatic_unit = np.char.find(unit_type_string.astype(str), 'NON-SOMA') >= 0
    
    trough_to_peak2 = np.full(len(unit_type_string), False, dtype=bool)
    trough_to_peak2[is_non_somatic_unit] = (
        quality_metrics["troughToPeak2Ratio"][is_non_somatic_unit] < param["minTroughToPeak2Ratio_nonSomatic"]
    )
    
    peak1_to_peak2 = np.full(len(unit_type_string), False, dtype=bool)
    # For non-somatic units, check if peak1/peak2 criteria failed (only when first 3 criteria were met)
    if np.any(is_non_somatic_unit):
        first_three_criteria = (
            (quality_metrics["troughToPeak2Ratio"] < param["minTroughToPeak2Ratio_nonSomatic"]) &
            (quality_metrics["mainPeak_before_width"] < param["minWidthFirstPeak_nonSomatic"]) &
            (quality_metrics["mainTrough_width"] < param["minWidthMainTrough_nonSomatic"])
        )
        non_soma_with_first_three = is_non_somatic_unit & first_three_criteria
        peak1_to_peak2[non_soma_with_first_three] = (
            quality_metrics["peak1ToPeak2Ratio"][non_soma_with_first_three] > param["maxPeak1ToPeak2Ratio_nonSomatic"]
        )

    qm_table_list.extend([trough_to_peak2, peak1_to_peak2])
    qm_table_names.extend(["trough / peak2", "peak1 / peak2"])


    if param["computeDistanceMetrics"]:
        too_low_iso_dist = np.full(len(unit_type_string), False, dtype=bool)
        too_low_iso_dist[is_mua_unit] = (
            quality_metrics['isolationDistance'][is_mua_unit] < param["isoDmin"]
        )
        
        too_high_lratio = np.full(len(unit_type_string), False, dtype=bool)
        too_high_lratio[is_mua_unit] = (
            quality_metrics["Lratio"][is_mua_unit] > param["lratioMax"]
        )

        qm_table_list.extend([too_low_iso_dist, too_high_lratio])
        qm_table_names.extend(["isolation dist.", "L-ratio"])


    # DO this for the optional params
    qm_table = pd.DataFrame(
        qm_table_list, qm_table_names
    ).T
    return qm_table