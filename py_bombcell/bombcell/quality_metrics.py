import os

import numpy as np
from numba import njit

from scipy.optimize import curve_fit
from scipy.signal import medfilt, find_peaks
from scipy.stats import norm, chi2

import matplotlib.pyplot as plt

from bombcell.extract_raw_waveforms import path_handler


def get_waveform_peak_channel(template_waveforms):
    """
    Get the max channel for all templates (channel with largest amplitude)

    Parameters
    ----------
    template_waveforms : ndarray (n_templates, n_time_points, n_channels)
        The template waveforms for each template and channel

    Returns
    -------
    maxChannels : ndarray (n_templates)
        The channel with maximum amplitude for each template
    """
    max_value = np.max(template_waveforms, axis=1)
    min_value = np.min(template_waveforms, axis=1)
    maxChannels = np.nanargmax(max_value - min_value, axis=1)

    return maxChannels


@njit(cache=True)
def remove_duplicates(
    batch_spike_times_samples,
    batch_spike_clusters,
    batch_template_amplitudes,
    batch_spike_clusters_flat,
    maxChannels,
    duplicate_spike_window_samples,
):
    """
    This function uses batches of spike times and templates to find when spike overlap on the same channel:
        will remove the lowest amplitude spike for a pair the same unit
        will remove the most common spike of the batch if a pair of different units
    Parameters
    ----------
    batch_spike_times_samples : ndarray
        A batch of spike times in samples
    batch_spike_clusters : ndarray
        A batch of spike templates
    batch_template_amplitudes : ndarray
        A batch of spike template amplitudes
    batch_spike_clusters_flat : ndarray
        A batch of the flattened spike templates
    maxChannels : ndarray
        The max channel for each unit
    duplicate_spike_window_samples : int
        The length of time in samples which marks a pair of overlapping spike

    Returns
    -------
    remove_idx : ndarray
        An array which if 1 states that spike should be removed
    """

    num_spikes = batch_spike_times_samples.shape[0]
    remove_idx = np.zeros(num_spikes)
    # spike counts for the batch
    unit_spike_counts = np.bincount(batch_spike_clusters)

    # go through each spike in the batch
    for spike_idx1 in range(num_spikes):
        if remove_idx[spike_idx1] == 1:
            continue

        # go through each spike within +/- 25 idx's
        for spike_idx2 in np.arange(spike_idx1 - 25, min(spike_idx1 + 25, num_spikes)):
            # ignore self
            if spike_idx1 == spike_idx2:
                continue
            if remove_idx[spike_idx2] == 1:
                continue

            if (
                maxChannels[batch_spike_clusters_flat[spike_idx1]]
                != maxChannels[batch_spike_clusters_flat[spike_idx2]]
            ):
                continue
            # intra-unit removal
            if batch_spike_clusters[spike_idx1] == batch_spike_clusters[spike_idx2]:
                if (
                    np.abs(
                        batch_spike_times_samples[spike_idx1]
                        - batch_spike_times_samples[spike_idx2]
                    )
                    <= duplicate_spike_window_samples
                ):
                    # keep higher amplitude spike
                    if (
                        batch_template_amplitudes[spike_idx1]
                        < batch_template_amplitudes[spike_idx2]
                    ):
                        batch_spike_times_samples[spike_idx1] = np.nan
                        remove_idx[spike_idx1] = 1
                    else:
                        batch_spike_times_samples[spike_idx2] = np.nan
                        remove_idx[spike_idx2] = 1

            # inter-unit removal
            if batch_spike_clusters[spike_idx1] != batch_spike_clusters[spike_idx2]:
                if (
                    np.abs(
                        batch_spike_times_samples[spike_idx1]
                        - batch_spike_times_samples[spike_idx2]
                    )
                    <= duplicate_spike_window_samples
                ):
                    # keep spike from unit with less spikes
                    if (
                        unit_spike_counts[batch_spike_clusters[spike_idx1]]
                        < unit_spike_counts[batch_spike_clusters[spike_idx2]]
                    ):
                        batch_spike_times_samples[spike_idx1] = np.nan
                        remove_idx[spike_idx1] = 1
                    else:
                        batch_spike_times_samples[spike_idx2] = np.nan
                        remove_idx[spike_idx2] = 1

    return remove_idx


def remove_duplicate_spikes(
    spike_times_samples,
    spike_clusters,
    template_amplitudes,
    maxChannels,
    save_path,
    param,
    pc_features=None,
    raw_waveforms_full=None,
    raw_waveforms_peak_channel=None,
    signal_to_noise_ratio=None,
):
    """
    This function finds and removes spikes which have been recorded multiple times,
    by looking at overlap of spike on the same channels

    Parameters
    ----------
    spike_times_samples : ndarray
        The array of spike times in samples
    spike_clusters : ndarray
        The array which assigns each spike a id
    template_amplitudes : ndarray
        The array of amplitudes for each spike
    maxChannels : ndarray
        The max channel for each unit
    save_path : str
        The path to the save directory
    param : dict
        The param dictionary
    pc_features : ndarray, optional
        The pc features array, by default None
    raw_waveforms_full : ndarray, optional
        The array wth extracted raw waveforms, by default None
    raw_waveforms_peak_channel : ndarray, optional
        The peak channel for each extracted raw waveform, by default None
    signal_to_noise_ratio : ndarray, optional
        The signal to noise ratio, by default None

    Returns
    -------
    non_empty_units : ndarray
        Units which are not empty
    spike_times_samples : ndarray
        The array of spike times in samples
    spike_clusters : ndarray
        The array which assigns each spike a id
    template_amplitudes : ndarray
        The array of amplitudes for each spike
    pc_features : ndarray, optional
        The pc features array, by default None
    raw_waveforms_full : ndarray, optional
        The array wth extracted raw waveforms, by default None
    raw_waveforms_peak_channel : ndarray, optional
        The peak channel for each extracted raw waveform, by default None
    signal_to_noise_ratio : ndarray, optional
        The signal to noise ratio, by default None
    maxChannels : ndarray
        The max channel for each spike
    """
    # Create save_path if it does not exist
    save_path = path_handler(save_path)

    if param["removeDuplicateSpikes"]:
        # check if spikes are already extract or need to recompute
        if param["recomputeDuplicateSpikes"] or ~os.path.isdir(
            os.path.join(save_path, "spikes._bc_duplicateSpikes.npy")
        ):

            # parameters
            duplicate_spike_window_samples = (
                param["duplicateSpikeWindow_s"] * param["ephys_sample_rate"]
            )
            batch_size = 10000
            overlap_size = 100
            num_spikes_full = spike_times_samples.shape[0]

            # initialize and re-allocate
            duplicate_spike_idx = np.zeros(num_spikes_full)

            # rename the spike templates according to the remaining templates
            good_templates_idx = np.unique(spike_clusters)
            new_spike_idx = np.full(max(good_templates_idx) + 1, np.nan)
            new_spike_idx[good_templates_idx] = np.arange(good_templates_idx.shape[0])
            spike_clusters_flat = (
                new_spike_idx[spike_clusters].astype(np.int32)
            )

            # check for duplicate spikes in batches
            num_batches = (num_spikes_full) / (batch_size - overlap_size)
            start_idxs = np.arange(num_batches, dtype=np.int32) * (
                batch_size - overlap_size
            )
            for start_idx in start_idxs:
                end_idx = min(start_idx + batch_size, num_spikes_full)
                batch_spike_times_samples = spike_times_samples[
                    start_idx:end_idx
                ].astype(np.float32)
                batch_spike_clusters = spike_clusters[start_idx:end_idx]
                batch_template_amplitudes = template_amplitudes[start_idx:end_idx]
                batch_spike_clusters_flat = spike_clusters_flat[start_idx:end_idx]

                batch_remove_idx = remove_duplicates(
                    batch_spike_times_samples,
                    batch_spike_clusters,
                    batch_template_amplitudes,
                    batch_spike_clusters_flat,
                    maxChannels,
                    duplicate_spike_window_samples,
                )
                duplicate_spike_idx[start_idx:end_idx] = batch_remove_idx

            if param["saveSpikes_withoutDuplicates"]:
                np.save(
                    os.path.join(save_path, "spikes._bc_duplicateSpikes.npy"),
                    duplicate_spike_idx,
                )

        else:
            duplicate_spike_idx = np.load(
                os.path.join(save_path, "spikes._bc_duplicateSpikes.npy")
            )

        # check if there are any empty units
        unique_templates = np.unique(spike_clusters)
        non_empty_units = np.unique(spike_clusters[duplicate_spike_idx == 0])
        empty_unit_idx = np.isin(unique_templates, non_empty_units, invert=True)
        spike_idx_to_remove = np.argwhere(duplicate_spike_idx == 0).squeeze()

        # remove any empty units and duplicate spikes
        spike_times_samples = spike_times_samples[spike_idx_to_remove]
        spike_clusters = spike_clusters[spike_idx_to_remove]
        template_amplitudes = template_amplitudes[spike_idx_to_remove]

        if pc_features is not None:
            pc_features = pc_features[
                spike_idx_to_remove, :, :
            ]

        if raw_waveforms_full is not None:
            raw_waveforms_full = raw_waveforms_full[empty_unit_idx == False, :, :]
            raw_waveforms_peak_channel = raw_waveforms_peak_channel[
                empty_unit_idx == False
            ]

        if signal_to_noise_ratio is not None:
            signal_to_noise_ratio = signal_to_noise_ratio[empty_unit_idx == False]

        return (
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
        )


def gaussian_cut(bin_centers, A, u, s, c):
    """
    A gaussian curve with a cut of value c

    Parameters
    ----------
    bin_centers : ndarray
        The x values of the curve
    A : float
        The height of the gaussian
    u : float
        The center of the gaussian
    s : float
        The width of the gaussian
    c : float
        The cutoff values, less than this is set to 0

    Returns
    -------
    F : ndarray
        The cutoff gaussian as an array
    """
    F = A * np.exp(-((bin_centers - u) ** 2) / (2 * s**2))
    F[bin_centers < c] = 0
    return F


def is_peak_cutoff(spike_counts_per_amp_bin_gaussian):
    """
    Test to see if the amplitude distribution goes from the max to 0,
    this indicates less than half the spikes were recorded and estimating the spikes missing is unfeasible.

    Parameters
    ----------
    spike_counts_per_amp_bin_gaussian : ndarrray
        The spike counts per amplitude bin for the current unit

    Returns
    -------
    not_cutoff : bool
        False if the is amplitude distribution has a peak
    """
    # get max value
    max_val = np.max(spike_counts_per_amp_bin_gaussian)
    # see if the difference include the max value i.e goes from 0 to peak
    max_diff = np.max(np.diff(spike_counts_per_amp_bin_gaussian))
    if max_val == max_diff:
        not_cutoff = False
    else:
        not_cutoff = True
    return not_cutoff


def perc_spikes_missing(these_amplitudes, these_spike_times, time_chunks, param, metric = False, return_per_bin = False):
    """
    This function estimates the percentage of spike missing from a unit.

    Parameters
    ----------
    these_amplitudes : ndarray
        The amplitudes for the given unit
    these_spike_times : ndarray
        The spike times of the given unit
    time_chunks : ndarray
        The time chunks to consider
    param : dict
        The param dictionary
    metric : bool, optional
        If True will return the average percent spikes missing, by default False
    return_per_bin : bool, optional
        If True will return per-bin data for GUI plotting, by default False

    Returns
    -------
    percent_missing_gaussian : float or ndarray
        The estimated percentage of spikes missing when using a gaussian approximation
    percent_missing_symmetric : float or ndarray 
        The estimated percentage of spikes missing when using a symmetric approximation
    per_bin_data : dict, optional
        If return_per_bin=True, returns dict with 'time_bins', 'percent_missing_gaussian_per_bin', 'percent_missing_symmetric_per_bin'
    """

    percent_missing_gaussian = np.zeros(time_chunks.shape[0] - 1)
    percent_missing_symmetric = np.zeros(time_chunks.shape[0] - 1)
    ks_test_p_value = np.zeros(time_chunks.shape[0] - 1)  # NOT DONE CURRENTLY
    test = np.zeros(time_chunks.shape[0] - 1)
    fit_params_save = []
    for time_chunk_idx in range(time_chunks.shape[0] - 1):
        # amplitude histogram
        n_bins = 50
        chunk_idx = np.logical_and(
            these_spike_times >= time_chunks[time_chunk_idx],
            these_spike_times < time_chunks[time_chunk_idx + 1],
        )

        these_amplitudes_here = these_amplitudes[chunk_idx]
        
        if these_amplitudes_here.size == 0:
            percent_missing_gaussian[time_chunk_idx] = np.nan
            percent_missing_symmetric[time_chunk_idx] = np.nan
            amp_bin_gaussian = np.nan
            spike_counts_per_amp_bin_gaussian = np.nan
            gaussian_fit = np.nan
            continue
        # check for extreme outliers (see https://github.com/Julie-Fabre/bombcell/issues/179)
        # flagging for now but should we remove them entirely? have a separate function to this effect?
        iqr_threshold = 10
        quantiles = np.percentile(these_amplitudes_here, [1, 99])
        iqr = quantiles[1] - quantiles[0]
        outliers_iqr = these_amplitudes_here > (quantiles[1] + iqr_threshold * iqr)
        these_amplitudes_here = these_amplitudes_here[~outliers_iqr]

        spike_counts_per_amp_bin, bins = np.histogram(
            these_amplitudes_here, bins=n_bins
        )
        if np.sum(spike_counts_per_amp_bin) > 5:  # at least 5 spikes in time bin
            max_amp_bin = np.argwhere(
                spike_counts_per_amp_bin == spike_counts_per_amp_bin.max()
            )
            if max_amp_bin.size != 1:
                max_amp_bin = max_amp_bin.mean().astype(int)
                mode_seed = bins[max_amp_bin]
            else:
                mode_seed = bins[max_amp_bin].squeeze()
            bin_step = bins[1] - bins[0]

            ## Symmetric - not used currently
            # mirror the units amplitudes

            spike_counts_per_amp_bin_smooth = medfilt(spike_counts_per_amp_bin, 5)
            # first max value if there are multiple
            # necessarily has a 2 index peak
            max_amp_bins_smooth = np.nanargmax(spike_counts_per_amp_bin_smooth)
            surrogate_amplitudes = np.concatenate(
                (
                    spike_counts_per_amp_bin_smooth[-1:max_amp_bins_smooth:-1],
                    np.flip(spike_counts_per_amp_bin_smooth[-1:max_amp_bins_smooth:-1]),
                )
            )

            # start at the common last point the make steps back to fill the space
            end_point = bins[-1]
            start_point = (
                end_point - (surrogate_amplitudes.shape[0] - 1) * bin_step
            )  # -1 due to inclusive/exclusive limits
            surrogate_bins = np.linspace(
                start_point, end_point, surrogate_amplitudes.shape[0]
            )
            # plt.plot(np.pad(bins, (surrogate_bins.shape[0] - bins.shape[0], 0))) # plot confirms correct amplitude space

            # remove any negative bins
            surrogate_amplitudes = np.delete(
                surrogate_amplitudes, np.argwhere(surrogate_bins < 0)
            )
            surrogate_bins = np.delete(
                surrogate_amplitudes, np.argwhere(surrogate_bins < 0)
            )
            surrogate_area = np.sum(surrogate_amplitudes) * bin_step

            # estimate the percentage of missing spikes
            if surrogate_area == 0:
                p_missing = 0
            else:
                p_missing = (
                    (surrogate_area - np.sum(spike_counts_per_amp_bin) * bin_step)
                    / (surrogate_area) 
                ) * 100
            if p_missing < 0:  # If p_missing is -ve, the distribution is not symmetric
                p_missing = 0

            percent_missing_symmetric[time_chunk_idx] = p_missing

            ## Gaussian
            # make it cover all values from 0
            amp_bin_gaussian = bins[:-1] + bin_step / 2
            next_low_bin = amp_bin_gaussian[0] - bin_step
            # find the min bin which is a multiple of step size close to 0
            min_bin = next_low_bin - bin_step * np.ceil((next_low_bin / bin_step))
            add_points = np.arange(
                min_bin, next_low_bin + bin_step / 4, bin_step
            )  # have to add a little bit to the end for rounding problems
            amp_bin_gaussian = np.concatenate((add_points, amp_bin_gaussian))

            spike_counts_per_amp_bin_gaussian = np.concatenate(
                (np.zeros_like(add_points), spike_counts_per_amp_bin)
            )

            # test to see if the peak is cutoff
            not_cutoff = is_peak_cutoff(spike_counts_per_amp_bin_gaussian)

            if not_cutoff:
                # Testing for cut-off solves all of these issues
                p0 = np.array(
                    (
                        np.percentile(spike_counts_per_amp_bin_gaussian, 99),
                        mode_seed,
                        np.nanstd(these_amplitudes_here),
                        np.percentile(these_amplitudes_here, 1),
                    )
                )
                fit_params = curve_fit(
                    gaussian_cut,
                    amp_bin_gaussian,
                    spike_counts_per_amp_bin_gaussian,
                    p0=p0,
                    ftol=1e-3,
                    xtol=1e-3,
                    maxfev=10000,
                    method = 'trf'
                )[0]
                gaussian_fit = gaussian_cut(
                    amp_bin_gaussian, fit_params[0], fit_params[1], fit_params[2], p0[3]
                )

                norm_area = norm.cdf(
                    (fit_params[1] - fit_params[3]) / np.abs(fit_params[2])
                )
                fit_params_save.append(fit_params)
                percent_missing_gaussian[time_chunk_idx] = 100 * (1 - norm_area)
            else:
                percent_missing_gaussian[time_chunk_idx] = 100  # Use one as a fail here
                gaussian_fit = np.nan
        else:
            percent_missing_gaussian[time_chunk_idx] = np.nan
            percent_missing_symmetric[time_chunk_idx] = np.nan
            # ks_test = np.nan # when doing a ks test
            amp_bin_gaussian = np.nan
            spike_counts_per_amp_bin_gaussian = np.nan
            gaussian_fit = np.nan

        # TODO clean up plots
        if param["plotDetails"]:
            if np.sum(spike_counts_per_amp_bin_gaussian) > 0:
                plt.plot(
                    gaussian_fit,
                    amp_bin_gaussian,
                    label=f"gaussian fit time chunk {time_chunk_idx + 1}",
                )
                plt.plot(
                    spike_counts_per_amp_bin_gaussian,
                    amp_bin_gaussian,
                    label=f"spike amp time chunk {time_chunk_idx + 1} ",
                )
                plt.xlabel("count")
                plt.ylabel("amplitude")
                plt.legend()
                plt.show()
    
    # Store per-bin data for GUI
    per_bin_data = None
    if return_per_bin:
        per_bin_data = {
            'time_bins': time_chunks,
            'percent_missing_gaussian_per_bin': percent_missing_gaussian.copy(),
            'percent_missing_symmetric_per_bin': percent_missing_symmetric.copy()
        }

    if metric:
        percent_missing_gaussian = np.mean(percent_missing_gaussian)
        percent_missing_symmetric = np.mean(percent_missing_symmetric)

    if return_per_bin:
        return (
            percent_missing_gaussian,
            percent_missing_symmetric,
            per_bin_data
        )
    else:
        return (
            percent_missing_gaussian,
            percent_missing_symmetric
        )

def fraction_RP_violations(these_spike_times, these_amplitudes, time_chunks, param, return_per_bin = False):
    """
    This function estimates the fraction of refractory period violations for a given unit.

    Parameters
    ----------
    these_spike_times : ndarray
        The spike times for the given unit
    these_amplitudes : ndarray
        The spike amplitudes for the given unit
    time_chunks : ndarray
        The time chunks to consider
    param : dict
        The param dictionary containing:
        - tauR_valuesMin: minimum refractory period value
        - tauR_valuesMax: maximum refractory period value
        - tauR_valuesStep: step size for refractory period values
        - tauC: censored period
        - hillOrLlobetMethod: boolean to choose method
        - plotDetails: boolean to enable plotting (default: False)
        - RPV_tauR_estimate: index of tauR to use for plotting (optional)
    return_per_bin : bool, optional
        If True will return per-bin data for GUI plotting, by default False

    Returns
    -------
    fraction_RPVs : ndarray 
        The fraction of refractory period violations for the unit
    num_violations : ndarray
        The number of refractory period violations for the unit
    per_bin_data : dict, optional
        If return_per_bin=True, returns dict with 'time_bins', 'fraction_RPVs_per_bin'
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    tauR_min = param["tauR_valuesMin"]
    tauR_max = param["tauR_valuesMax"]
    tauR_step = param["tauR_valuesStep"]
    assert tauR_min <= tauR_max, "tauR_max is smaller than tauR_min! Check parameters!"

    # Create tauR window array (includes endpoint)
    tauR_window = np.arange(tauR_min, tauR_max + tauR_step, tauR_step)
    tauC = param["tauC"]
    
    # Set default value for plotting if not provided
    if "plotDetails" not in param:
        param["plotDetails"] = False

    # Initialize arrays
    fraction_RPVs = np.zeros((time_chunks.shape[0] - 1, tauR_window.shape[0]))
    overestimate_bool = np.zeros_like(fraction_RPVs)
    num_violations = np.zeros_like(fraction_RPVs)
    
    # Initialize raw fraction (violations/total spikes) for plotting
    RPV_fraction = np.zeros_like(fraction_RPVs)

    # Check if RPV_tauR_estimate is provided
    if "RPV_tauR_estimate" not in param:
        param["RPV_tauR_estimate"] = None

    # Create figure and gridspec if plotting is enabled
    if param["plotDetails"]:
        fig = plt.figure(figsize=(12, 8))
        gs = GridSpec(2, len(time_chunks) - 1, figure=fig)
        
        # Create the top panel that spans all columns
        ax_top = fig.add_subplot(gs[0, :])

    # Loop through each time chunk
    for time_chunk_idx in range(time_chunks.shape[0] - 1):
        # Get spikes in this chunk
        chunk_spike_times = these_spike_times[
            np.logical_and(
                these_spike_times >= time_chunks[time_chunk_idx],
                these_spike_times < time_chunks[time_chunk_idx + 1],
            )
        ]
        
        # Number of spikes in chunk
        n_chunk = chunk_spike_times.size
        
        # Calculate duration of this chunk
        duration_chunk = time_chunks[time_chunk_idx + 1] - time_chunks[time_chunk_idx]
        
        # Calculate chunk ISIs
        if n_chunk > 1:
            chunk_ISIs = np.diff(chunk_spike_times)
        else:
            chunk_ISIs = np.array([])
        
        # Loop through each tauR value
        for i_tau_r, tauR in enumerate(tauR_window):
            # Calculate number of refractory period violations
            if n_chunk > 1:
                num_violations[time_chunk_idx, i_tau_r] = np.sum(chunk_ISIs <= tauR)
                # Calculate raw fraction for plotting
                RPV_fraction[time_chunk_idx, i_tau_r] = num_violations[time_chunk_idx, i_tau_r] / n_chunk
            else:
                num_violations[time_chunk_idx, i_tau_r] = 0
                RPV_fraction[time_chunk_idx, i_tau_r] = 0
                
            # Apply either Hill or Llobet method
            if param["hillOrLlobetMethod"]:
                # Hill et al. method
                k = 2 * (tauR - tauC) * n_chunk**2
                T = duration_chunk
                
                if num_violations[time_chunk_idx, i_tau_r] == 0:
                    # No observed refractory period violations
                    fraction_RPVs[time_chunk_idx, i_tau_r] = 0
                    overestimate_bool[time_chunk_idx, i_tau_r] = 0
                else:
                    # Solve the quadratic equation
                    # Using coefficients [k, -k, nRPVs * T] to match MATLAB
                    rts = np.roots([k, -k, num_violations[time_chunk_idx, i_tau_r] * T])
                    
                    # Get minimum root and check if it's real
                    min_root = np.min(rts)
                    if np.isreal(min_root):
                        fraction_RPVs[time_chunk_idx, i_tau_r] = np.real(min_root)
                        overestimate_bool[time_chunk_idx, i_tau_r] = 0
                    else:
                        # If roots are complex, use approximation formula
                        if num_violations[time_chunk_idx, i_tau_r] < n_chunk:
                            fraction_RPVs[time_chunk_idx, i_tau_r] = num_violations[time_chunk_idx, i_tau_r] / (
                                2 * (tauR - tauC) * (n_chunk - num_violations[time_chunk_idx, i_tau_r])
                            )
                        else:
                            fraction_RPVs[time_chunk_idx, i_tau_r] = 1
                            overestimate_bool[time_chunk_idx, i_tau_r] = 1
                            
                    # Cap fraction at 1 (assumptions failing if > 1)
                    if fraction_RPVs[time_chunk_idx, i_tau_r] > 1:
                        fraction_RPVs[time_chunk_idx, i_tau_r] = 1
                        overestimate_bool[time_chunk_idx, i_tau_r] = 1
            else:
                # Llobet et al. method
                N = len(chunk_spike_times)
                isi_violations_sum = 0

                # Count all pair-wise violations
                if N > 1:
                    for i in range(N - 1):
                        isi_vec = chunk_spike_times[i+1:] - chunk_spike_times[i]
                        isi_violations_sum += np.sum((isi_vec <= tauR) & (isi_vec >= tauC))
                else:
                    isi_violations_sum = 0

                # Calculate fraction using Llobet equation
                if N > 0:  # Avoid division by zero
                    underRoot = 1 - (isi_violations_sum * (duration_chunk - 2 * N * tauC)) / (N**2 * (tauR - tauC))
                    if underRoot >= 0:
                        fraction_RPVs[time_chunk_idx, i_tau_r] = 1 - np.sqrt(underRoot)
                    else:
                        fraction_RPVs[time_chunk_idx, i_tau_r] = 1
                        overestimate_bool[time_chunk_idx, i_tau_r] = 1
                else:
                    fraction_RPVs[time_chunk_idx, i_tau_r] = 0
        
        # Plot ISI histogram if requested
        if param["plotDetails"]:
            # Create a subplot for each time chunk in the bottom row
            ax_bottom = fig.add_subplot(gs[1, time_chunk_idx])
            
            if n_chunk > 1:
                # Clean ISIs - removing duplicates (censored period)
                clean_isis = chunk_ISIs[chunk_ISIs >= tauC]
                
                # Convert to ms for plotting
                clean_isis_ms = clean_isis * 1000
                
                # Create histogram
                hist_values, edges = np.histogram(clean_isis_ms, bins=np.arange(0, 100.5, 0.5))
                centers = edges[:-1] + np.diff(edges)/2
                
                # Plot histogram
                ax_bottom.bar(centers, hist_values, width=0.5, color=[0, 0.35, 0.71], edgecolor=[0, 0.35, 0.71])
                
                # Add labels to first subplot only
                if time_chunk_idx == 0:
                    ax_bottom.set_xlabel('Interspike interval (ms)')
                    ax_bottom.set_ylabel('# of spikes')
                else:
                    ax_bottom.set_xticks([])
                    ax_bottom.set_yticks([])
                
                # Get y-limits for reference line
                ylims = ax_bottom.get_ylim()
                
                # Calculate baseline firing rate for reference line (similar to MATLAB's nanmean)
                long_isis = clean_isis_ms[(clean_isis_ms >= 400) & (clean_isis_ms <= 500)]
                if len(long_isis) > 0:
                    baseline = np.nanmean(np.histogram(long_isis, bins=np.arange(400, 500.5, 0.5))[0])
                    ax_bottom.plot([0, 10], [baseline, baseline], '--', color=[0.86, 0.2, 0.13])
                
                # Add refractory period lines
                RPV_tauR_estimate = param["RPV_tauR_estimate"]
                if RPV_tauR_estimate is None:
                    # Show min and max tauR if no specific estimate
                    if len(tauR_window) > 1:
                        for idx in [0, len(tauR_window)-1]:
                            ax_bottom.axvline(x=tauR_window[idx] * 1000, color=[0.86, 0.2, 0.13])
                        
                        # Add title with min/max RPV rates
                        ax_bottom.set_title(f"{fraction_RPVs[time_chunk_idx, 0]*100:.0f}% rpv\n"
                                  f"{fraction_RPVs[time_chunk_idx, -1]*100:.0f}% rpv")
                    else:
                        # Just one tauR value
                        ax_bottom.axvline(x=tauR_window[0] * 1000, color=[0.86, 0.2, 0.13])
                        ax_bottom.set_title(f"{fraction_RPVs[time_chunk_idx, 0]*100:.0f}% rpv\n"
                                  f"frac. rpv={RPV_fraction[time_chunk_idx, 0]:.3f}")
                else:
                    # Use the specified tauR estimate
                    if 0 <= RPV_tauR_estimate < len(tauR_window):
                        ax_bottom.axvline(x=tauR_window[RPV_tauR_estimate] * 1000, color=[0.86, 0.2, 0.13])
                        ax_bottom.set_title(f"{fraction_RPVs[time_chunk_idx, RPV_tauR_estimate]*100:.0f}% rpv\n"
                                  f"frac. rpv={RPV_fraction[time_chunk_idx, RPV_tauR_estimate]:.3f}")
    
    # Create top row plot with spike amplitudes vs time if plotting is enabled
    if param["plotDetails"]:
        # Plot spike amplitudes vs time in the top panel
        ax_top.scatter(these_spike_times, these_amplitudes, s=4, c=[0, 0.35, 0.71], alpha=0.7)
        
        # Add vertical lines for time chunks
        ylims = ax_top.get_ylim()
        for i, t in enumerate(time_chunks):
            ax_top.axvline(x=t, color=[0.7, 0.7, 0.7])
        
        ax_top.set_xlabel('time (s)')
        ax_top.set_ylabel('amplitude scaling\nfactor')
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
    
    # Store per-bin data for GUI
    per_bin_data = None
    if return_per_bin:
        per_bin_data = {
            'time_bins': time_chunks,
            'fraction_RPVs_per_bin': fraction_RPVs.copy()  # Shape: (n_time_chunks, n_tauR_values)
        }

    if return_per_bin:
        return fraction_RPVs, num_violations, per_bin_data
    else:
        return fraction_RPVs, num_violations


def time_chunks_to_keep(
    percent_missing_gaussian,
    fraction_RPVs,
    time_chunks,
    these_spike_times,
    these_amplitudes,
    spike_clusters,
    spike_times_seconds,
    param,
):
    """
    This function uses the percentage of missing spikes and the refractory period violations to find which time chunk the unit as properly recorded in.

    Parameters
    ----------
    percent_missing_gaussian : float
        The percentage of missing spikes
    fraction_RPVs : float
        The fraction of refractory period violations
    time_chunks : ndarray
        The time chunk to use for the unit
    these_spike_times : ndarray
        The spike times of this unit
    these_amplitudes : ndarray
        The spike amplitudes of this unit
    spike_clusters : ndarray
        The template waveforms of this unit
    spike_times_seconds : ndarray
        The spike times in seconds
    param : dict
        The param dictionary

    Returns
    -------
    these_spike_times : ndarray
        Good time chunks to use
    these_amplitudes : ndarray
        The amplitudes for the good time chunks
    these_spike_clusters : ndarray
        The spike times for these good time chunks
    use_this_time_start : float
        The start of the good time chunk
    use_this_time_end : float
        The end of the good time chunk
    use_tauR : float
        The index of the tau_R which has the smallest contamination
    """
    maxRPVviolationss = param["maxRPVviolations"]
    maxPercSpikesMissing = param["maxPercSpikesMissing"]

    sum_RPV = np.sum(fraction_RPVs, axis=0)
    use_tauR = np.where(sum_RPV == np.min(sum_RPV))[0][-1] # gives the last index of the tauR which has smallest contamination # CAUGHT BUG was argmax!!
    use_these_times_temp = np.zeros(time_chunks.shape[0] - 1)

    use_these_times_temp = np.argwhere(
        np.logical_and(
            percent_missing_gaussian < maxPercSpikesMissing,
            fraction_RPVs[:, use_tauR] < maxRPVviolationss,
        )
    ).squeeze()
    if use_these_times_temp.size > 0:
        use_these_times_temp = np.atleast_1d(
            use_these_times_temp
        )  # if there is only 1 value make it a 1-d array
        continuous_times = np.diff(
            use_these_times_temp
        )  # will be 1 for adjacent good chunks
        if np.any(continuous_times == 1):
            chunk_starts = np.concatenate(
                (
                    np.zeros(1),
                    np.atleast_1d((np.argwhere(continuous_times > 1) + 1).squeeze()),
                )
            )
            chunk_ends = np.concatenate(
                (
                    np.atleast_1d((np.argwhere(continuous_times > 1) + 1).squeeze()),
                    np.array((use_these_times_temp.size - 1,)),
                )
            )

            chunk_lengths = chunk_ends - chunk_starts + 1

            longest_chunk = np.max(chunk_lengths)  # JF: i don't think is used
            longest_chunk_idx = np.nanargmax(chunk_lengths)

            longest_chunk_start = use_these_times_temp[
                chunk_starts[longest_chunk_idx].astype(int)
            ]
            longest_chunk_end = use_these_times_temp[
                chunk_ends[longest_chunk_idx].astype(int)
            ]

            use_these_times = time_chunks[
                longest_chunk_start : longest_chunk_end + 1
            ]  # include end
        else:
            # no continuous time chunks so arbitrarily choose the first
            use_these_times = time_chunks[
                use_these_times_temp[0] : use_these_times_temp[0] + 2
            ]  # +2 as python not inclusive of upper end
    else:
        # if there are no good time chunks use all time chunks for subsequent computations
        use_these_times = time_chunks

    # select which ones to keep
    these_amplitudes = these_amplitudes[
        np.logical_and(
            these_spike_times >= use_these_times[0],
            these_spike_times <= use_these_times[-1],
        )
    ]
    these_spike_clusters = spike_clusters.copy().astype(np.int32)
    these_spike_clusters[
        np.logical_or(
            spike_times_seconds < use_these_times[0], 
            spike_times_seconds > use_these_times[-1],
        )
    ] = -1  # set to -1 for bad times

    these_spike_times = these_spike_times[
        np.logical_and(
            these_spike_times >= use_these_times[0],
            these_spike_times <= use_these_times[-1],
        )
    ]

    use_this_time_start = use_these_times[0]
    use_this_time_end = use_these_times[-1]

    return (
        these_spike_times,
        these_amplitudes,
        these_spike_clusters,
        use_this_time_start,
        use_this_time_end,
        use_tauR,
    )


def presence_ratio(these_spike_times, use_this_time_start, use_this_time_end, param):
    """
    Calculates the presence ratio of the unit in the good time range

    Parameters
    ----------
    these_spike_times : ndarray
        The spike times of the unit
    use_this_time_start : float
        The good start of the good times
    use_this_time_end : float
        The end of the good times
    param : dict
        The param dictionary

    Returns
    -------
    presence_ratio : float
        The presence ratio for the unit
    """

    presenceRatioBinSize = param["presenceRatioBinSize"]

    # divide recording into bins
    presence_ratio_bins = np.arange(
        use_this_time_start, use_this_time_end, presenceRatioBinSize
    )

    spikes_per_bin = np.array(
        [
            np.sum(
                np.logical_and(
                    these_spike_times >= presence_ratio_bins[x],
                    these_spike_times < presence_ratio_bins[x + 1],
                )
            )
            for x in range(presence_ratio_bins.shape[0] - 1)
        ]
    )
    
    full_bins = np.zeros_like(spikes_per_bin)
    threshold = 0.05 * np.percentile(spikes_per_bin, 90)
    
    # Both conditions: >= threshold AND > 0
    full_bins[np.logical_and(spikes_per_bin >= threshold, spikes_per_bin > 0)] = 1
    
    presence_ratio = full_bins.sum() / full_bins.shape[0]
    
    return presence_ratio


def max_drift_estimate(
    pc_features,
    pc_features_idx,
    spike_clusters,
    these_spike_times,
    this_unit,
    channel_positions,
    param,
    return_per_bin = False
):
    """
    Calculates the drift of the unit using the PC components for each channels

    Parameters
    ----------
    pc_features : ndarray
        The top 3 PC features for the 32 most active channels for each unit
    pc_features_idx : ndarray
        Which channels are used for each unit
    spike_clusters : ndarray
        The array which assigns each spike to a unit
    these_spike_times : ndarray
        The spike times for the current unit
    this_unit : ndarray
        The ID of the current unit
    channel_positions : ndarray
        The (x,y) positions of each channel
    param : dict
        The param dictionary
    return_per_bin : bool, optional
        If True will return per-bin data for GUI plotting, by default False

    Returns
    -------
    max_drift_estimate : float
        The maximum drift estimated from the unit 
    cumulative_drift_estimate : float
        The cumulative drift estimated over the recording session
    per_bin_data : dict, optional
        If return_per_bin=True, returns dict with 'time_bins', 'median_spike_depth_per_bin'
    """
    channel_positions_z = channel_positions[:, 1]
    driftBinSize = param["driftBinSize"]

    # good_times_spikes = np.ones_like(spike_clusters)
    # good_times_spikes[spike_clusters == -1] = 0
    # pc_features_drift = pc_features[good_times_spikes == 1, :, :]
    # spike_clusters_current = spike_clusters[good_times_spikes == 1].astype(np.int32)

    # pc_features_pc1 = pc_features_drift[spike_clusters_current == this_unit, 0, :]
    # pc_features_pc1[pc_features_pc1 < 0] = 0 # remove negative entries

    pc_features_pc1 = pc_features[spike_clusters == this_unit, 0, :]
    pc_features_pc1[pc_features_pc1 < 0] = 0  # remove negative entries

    # NOTE test with and without only getting this units pc feature idx here
    # this is just several thousand copies of the same 32/ n_pce_feature array
    # spike_pc_feature = pc_features_idx[spike_clusters[spike_clusters == this_unit], :] # get channel for each spike
    spike_pc_feature = pc_features_idx[this_unit, :]

    pc_channel_pos_weights = channel_positions_z[spike_pc_feature]

    spike_depth_in_channels = np.sum(
        pc_channel_pos_weights[np.newaxis, :] * pc_features_pc1**2, axis=1
    ) / (np.nansum(pc_features_pc1**2, axis=1) + 1e-3) #adding small value to stop dividing by zero

    # estimate cumulative drift

    # NOTE this allow units which are only active briefly to still have two bins
    if these_spike_times.max() - these_spike_times.min() < 2 * driftBinSize:
        driftBinSize = (these_spike_times.max() - these_spike_times.min()) / 2

    time_bins = np.arange(
        these_spike_times.min(), these_spike_times.max(), driftBinSize
    )

    median_spike_depth = np.zeros(time_bins.shape[0] - 1)
    for i, time_bin_start in enumerate(time_bins[:-1]):
        all_spike_depths = spike_depth_in_channels[
                np.logical_and(
                    these_spike_times >= time_bin_start,
                    these_spike_times < (time_bin_start + driftBinSize),
                )
            ]
        if all_spike_depths.size > 0:
            median_spike_depth[i] = np.nanmedian(all_spike_depths)
        else:
            median_spike_depth[i] = np.nan
    max_drift_estimate = np.nanmax(median_spike_depth) - np.nanmin(median_spike_depth)
    cumulative_drift_estimate = np.sum(
        np.abs(np.diff(median_spike_depth[~np.isnan(median_spike_depth)]))
    )

    # Store per-bin data for GUI
    per_bin_data = None
    if return_per_bin:
        per_bin_data = {
            'time_bins': time_bins,
            'median_spike_depth_per_bin': median_spike_depth.copy()
        }

    if return_per_bin:
        return max_drift_estimate, cumulative_drift_estimate, per_bin_data
    else:
        return max_drift_estimate, cumulative_drift_estimate


def linear_fit(x, m, c):
    """
    Standard equation for a line

    Parameters
    ----------
    x : ndarray
        X values
    m : float
        Gradient
    c : float
        intercept

    Returns
    -------
    y : ndarray
        The y-values
    """
    return m * x + c


def exp_fit(x, m, A):
    """
    Standard equation for an exponential

    Parameters
    ----------
    x : ndarray
        X values
    m : float
        The exponential pre-factor
    A : float
        The multiplicative pre-factor

    Returns
    -------
    y : ndarray
        The y-values
    """
    return A * np.exp(m * x)


def waveform_shape(
    template_waveforms,
    this_unit,
    maxChannels,
    channel_positions,
    waveform_baseline_window,
    param,
):
    """
    Calculates waveforms shape based quality metrics and unit information

    Parameters
    ----------
    template_waveforms : ndarray
        The template waveforms
    this_unit : int
        The current unit ID
    maxChannels : ndarray
        The max channel for each unit
    channel_positions : ndarray
        The (x,y) positions of each channel
    waveform_baseline_window : ndarray
        The waveform baseline start and end points
    param : dict
        The param dictionary

    Returns
    -------
    n_peaks : int
        The number of peaks 
    n_troughs : int
        The number of troughs
    waveform_duration_peak_trough : float
        The time from peak to trough
    spatial_decay_slope : float
        The spatial decay of the waveforms amplitude over space
    waveform_baseline : float
        A measure of noise in the waveforms starting values
    scnd_peak_to_trough_ratio : float
        The ratio of the second peak to the trough
    peak1_to_peak2_ratio : float
        The ratio of the first peak to the second peak
    main_peak_to_trough_ratio : float
        The ratio of the first peak to the trough
    trough_to_peak2_ratio : float
        The ratio of the trough to the first peak ratio #TODO check this
    peak_before_width : float
        The width of the first peak
    trough_width : float
        The width of the trough
    param : dict
        The parameters 
    """
    min_thresh_detect_peaks_troughs = param["minThreshDetectPeaksTroughs"]
    this_waveform = template_waveforms[this_unit, :, maxChannels[this_unit]]

    if param["spDecayLinFit"]:
        CHANNEL_TOLERANCE = 33 # need to make more restrictive. for most geometries, this includes all the channels.
        MIN_CHANNELS_FOR_FIT = 5
        NUM_CHANNELS_FOR_FIT = 6
    else:
        CHANNEL_TOLERANCE = 33
        MIN_CHANNELS_FOR_FIT = 8
        NUM_CHANNELS_FOR_FIT = 10
        
    if np.any(np.isnan(this_waveform)):
        n_peaks = np.nan
        n_troughs = np.nan
        is_somatic = np.nan
        peak_locs = np.nan
        trough_locs = np.nan
        waveform_duration_peak_trough = np.nan
        spatial_decay_points = np.full((1, NUM_CHANNELS_FOR_FIT), np.nan)
        spatial_decay_slope = np.nan
        waveform_baseline = np.nan
        scnd_peak_to_trough_ratio = np.nan
        peak1_to_peak2_ratio = np.nan
        main_peak_to_trough_ratio = np.nan
        trough_to_peak2_ratio = np.nan
        peak_before_width = np.nan
        mainTrough_width = np.nan
    else:
        this_waveform_fit = this_waveform
        if np.size(this_waveform) == 82:  # Checking if the waveform length is 82 (KS4)
            # For KS4 waveforms, set the first 24 values to NaN to avoid artificial peaks/troughs (MATLAB line 6)
            this_waveform = this_waveform.copy()  # Make copy to avoid modifying original
            this_waveform[:24] = np.nan
            first_valid_index = 24  # Start from index 24 (MATLAB 1-indexed = Python 0-indexed)
        else:
            # For other waveforms, set first 4 values to NaN (MATLAB line 8)
            this_waveform = this_waveform.copy()  # Make copy to avoid modifying original  
            this_waveform[:4] = np.nan
            first_valid_index = 4  # Start from index 4

        # Detect troughs (MATLAB line 16: findpeaks on inverted waveform)
        min_prominence = min_thresh_detect_peaks_troughs * np.nanmax(np.abs(this_waveform))  # Use full waveform max like MATLAB line 13

        # Find troughs by inverting waveform (MATLAB line 16)
        trough_locs, trough_dict = find_peaks(
            this_waveform * -1, prominence=min_prominence, width=0
        )
        TRS = (this_waveform * -1)[trough_locs]  # Get trough magnitudes
        
        # Adjust for prominence filtering like MATLAB (lines 18-21)
        # MATLAB only adjusts trough width but keeps ALL troughs in TRS
        if len(trough_dict["widths"]) > 1:
            max_prominence_idx = np.nanargmax(trough_dict["prominences"])
            trough_width = trough_dict["widths"][max_prominence_idx]
        elif len(trough_dict.get("widths", [])) == 1:
            trough_width = trough_dict["widths"][0]
        else:
            trough_width = np.nan
        # IMPORTANT: Keep ALL troughs in TRS, don't filter down to one!

        # If no trough detected, find the minimum (MATLAB lines 24-30)  
        if len(TRS) == 0:
            TRS = np.array([np.nanmin(this_waveform)])
            trough_locs = np.array([np.nanargmin(this_waveform)])
            n_troughs = 1
            trough_width = np.nan
        else:
            n_troughs = len(TRS)  # MATLAB line 29: numel(TRS)

        # Get the main trough location (MATLAB lines 33-40)
        if len(TRS) > 0 and not np.all(np.isnan(TRS)):
            mainTrough_idx = np.nanargmax(TRS)  # TRS contains trough magnitudes (positive since inverted)
            # nanargmax returns a scalar, no need to index it
            mainTrough_value = TRS[mainTrough_idx]
            trough_loc = trough_locs[mainTrough_idx]
        else:
            # Fallback if no valid troughs found
            mainTrough_idx = 0
            mainTrough_value = 0
            trough_loc = len(this_waveform) // 2  # Use middle of waveform as fallback
        
        # Store trough values for later use
        troughs = TRS  # These are the actual trough magnitudes

        # Find peaks before and after the trough (MATLAB lines 42-63)
        PKS_before = np.array([])
        peakLocs_before = np.array([])
        width_before = np.nan
        
        # Find peaks before trough (MATLAB lines 43-52)
        if trough_loc > 3:  # MATLAB line 43: need at least 3 samples
            peakLocs_before, peak_dict_before = find_peaks(
                this_waveform[:trough_loc], prominence=min_prominence, width=0
            )
            PKS_before = this_waveform[peakLocs_before]
            widths_before = peak_dict_before.get("widths", [])
            prominences_before = peak_dict_before.get("prominences", [])
            if len(widths_before) > 1:
                if len(prominences_before) > 0 and not np.all(np.isnan(prominences_before)):
                    max_peak_idx = np.nanargmax(prominences_before)
                    # nanargmax returns a scalar, no need to index it
                    width_before = widths_before[max_peak_idx]
                else:
                    width_before = widths_before[0]
            elif len(widths_before) == 1:
                width_before = widths_before[0]
            # IMPORTANT: Keep ALL peaks from PKS_before, don't reduce to single peak
        
        PKS_after = np.array([])
        peakLocs_after = np.array([])
        width_after = np.nan
        
        # Find peaks after trough (MATLAB lines 53-63)
        if len(this_waveform) - trough_loc > 3:  # MATLAB line 53
            peakLocs_after_temp, peak_dict_after = find_peaks(
                this_waveform[trough_loc:], prominence=min_prominence, width=0
            )
            PKS_after = this_waveform[trough_loc:][peakLocs_after_temp]
            peakLocs_after = peakLocs_after_temp + trough_loc  # MATLAB line 55: adjust for offset
            widths_after = peak_dict_after.get("widths", [])
            prominences_after = peak_dict_after.get("prominences", [])
            if len(widths_after) > 1:
                if len(prominences_after) > 0 and not np.all(np.isnan(prominences_after)):
                    max_peak_idx = np.nanargmax(prominences_after)
                    # nanargmax returns a scalar, no need to index it
                    width_after = widths_after[max_peak_idx]
                else:
                    width_after = widths_after[0]
            elif len(widths_after) == 1:
                width_after = widths_after[0]
            # IMPORTANT: Keep ALL peaks from PKS_after, don't reduce to single peak
                
        # Handle case where no peaks detected with min_prominence (MATLAB lines 65-91)
        usedMaxBefore = 0
        if len(PKS_before) == 0:  # MATLAB line 67
            if trough_loc > 3:  # MATLAB line 68
                peakLocs_before_temp, peak_dict_before_temp = find_peaks(
                    this_waveform[:trough_loc], prominence=0.015*np.nanmax(np.abs(this_waveform)), width=0
                )
                PKS_before = this_waveform[peakLocs_before_temp]
                peakLocs_before = peakLocs_before_temp
                widths_before_temp = peak_dict_before_temp.get("widths", [])
                prominences_before_temp = peak_dict_before_temp.get("prominences", [])
                if len(PKS_before) > 1:  # MATLAB lines 73-79
                    if len(prominences_before_temp) > 0 and not np.all(np.isnan(prominences_before_temp)):
                        max_peak_idx = np.nanargmax(prominences_before_temp)
                        # MATLAB: Keep only the peak with maximum prominence
                        peakLocs_before = np.array([peakLocs_before[max_peak_idx]])
                        PKS_before = np.array([PKS_before[max_peak_idx]]) 
                        width_before = widths_before_temp[max_peak_idx]
                    else:
                        width_before = widths_before_temp[0] if len(widths_before_temp) > 0 else np.nan
                elif len(widths_before_temp) == 1:
                    width_before = widths_before_temp[0]
                    
            if len(PKS_before) == 0:  # MATLAB lines 81-89
                width_before = np.nan
                # Handle case where all values might be NaN (MATLAB line 83: uses max, not nanargmax)
                waveform_segment = this_waveform[:trough_loc]
                if np.all(np.isnan(waveform_segment)) or len(waveform_segment) == 0:
                    PKS_before = np.array([0.0])
                    peakLocs_before = np.array([0])
                else:
                    max_idx = np.nanargmax(waveform_segment)
                    PKS_before = np.array([waveform_segment[max_idx]])
                    peakLocs_before = np.array([max_idx])
                
            usedMaxBefore = 1  # MATLAB line 90
            
        usedMaxAfter = 0  
        if len(PKS_after) == 0:  # MATLAB line 94
            if len(this_waveform) - trough_loc > 3:  # MATLAB line 95
                peakLocs_after_temp, peak_dict_after_temp = find_peaks(
                    this_waveform[trough_loc:], prominence=0.015*np.nanmax(np.abs(this_waveform)), width=0
                )
                PKS_after = this_waveform[trough_loc:][peakLocs_after_temp]
                peakLocs_after = peakLocs_after_temp + trough_loc  # MATLAB line 97
                widths_after_temp = peak_dict_after_temp.get("widths", [])
                prominences_after_temp = peak_dict_after_temp.get("prominences", [])
                if len(PKS_after) > 1:  # MATLAB lines 101-107
                    if len(prominences_after_temp) > 0 and not np.all(np.isnan(prominences_after_temp)):
                        max_peak_idx = np.nanargmax(prominences_after_temp)
                        # MATLAB: Keep only the peak with maximum prominence
                        peakLocs_after = np.array([peakLocs_after[max_peak_idx]])
                        PKS_after = np.array([PKS_after[max_peak_idx]])
                        width_after = widths_after_temp[max_peak_idx]
                    else:
                        width_after = widths_after_temp[0] if len(widths_after_temp) > 0 else np.nan
                elif len(widths_after_temp) == 1:
                    width_after = widths_after_temp[0]
                    
            if len(PKS_after) == 0:  # MATLAB lines 108-117
                width_after = np.nan
                # Handle case where all values might be NaN (MATLAB line 110: uses max, not nanargmax)
                waveform_segment = this_waveform[trough_loc:]
                if np.all(np.isnan(waveform_segment)) or len(waveform_segment) == 0:
                    PKS_after = np.array([0.0])
                    peakLocs_after = np.array([trough_loc])
                else:
                    max_idx = np.nanargmax(waveform_segment)
                    PKS_after = np.array([waveform_segment[max_idx]])
                    peakLocs_after = np.array([trough_loc + max_idx])
                
            usedMaxAfter = 1  # MATLAB line 118
            
        # If both forced peaks were used, keep only the larger one (MATLAB lines 121-128)
        if usedMaxAfter > 0 and usedMaxBefore > 0:  # MATLAB line 122
            if len(PKS_before) > 0 and len(PKS_after) > 0:
                if PKS_before[0] > PKS_after[0]:  # MATLAB line 123
                    usedMaxBefore = 0
                else:
                    usedMaxAfter = 0
                    
        # Get main peak values for ratios (MATLAB lines 131-135)
        # Use absolute values to find the largest peaks/troughs
        mainPeak_before_size = np.max(np.abs(PKS_before)) if len(PKS_before) > 0 else 0
        mainPeak_after_size = np.max(np.abs(PKS_after)) if len(PKS_after) > 0 else 0
        mainTrough_size = np.max(TRS) if len(TRS) > 0 else 0  # TRS is already magnitudes from inverted waveform
        
        # Combine peak information - final filtering (MATLAB lines 137-147)
        if usedMaxBefore == 1 and len(PKS_before) > 0 and mainPeak_before_size < min_prominence * 0.5:  # MATLAB line 138
            PKS = PKS_after
            peakLocs = peakLocs_after
        elif usedMaxAfter == 1 and len(PKS_after) > 0 and mainPeak_after_size < min_prominence * 0.5:  # MATLAB line 141
            PKS = PKS_before  
            peakLocs = peakLocs_before
        else:  # MATLAB lines 144-146
            PKS = np.concatenate([PKS_before, PKS_after]) if len(PKS_before) > 0 and len(PKS_after) > 0 else (PKS_before if len(PKS_before) > 0 else PKS_after)
            peakLocs = np.concatenate([peakLocs_before, peakLocs_after]) if len(peakLocs_before) > 0 and len(peakLocs_after) > 0 else (peakLocs_before if len(peakLocs_before) > 0 else peakLocs_after)
            
        # Get number of peaks and troughs (MATLAB lines 164-165)
        n_peaks = len(PKS)  # MATLAB line 164: numel(PKS) 
        n_troughs = len(TRS)  # Already set above to match MATLAB line 29

        max_waveform_location = np.nanargmax(np.abs(this_waveform))
        max_waveform_value = this_waveform[max_waveform_location]  # signed value
        if max_waveform_value > 0:  # positive peak
            peak_loc_for_duration = max_waveform_location
            trough_loc_for_duration = np.nanargmin(this_waveform[peak_loc_for_duration:])
            trough_loc_for_duration = (
                trough_loc_for_duration + peak_loc_for_duration
            )  # arg for truncated waveform
        if max_waveform_value < 0:  # positive peak
            trough_loc_for_duration = max_waveform_location
            peak_loc_for_duration = np.nanargmax(this_waveform[trough_loc_for_duration:])
            peak_loc_for_duration = (
                peak_loc_for_duration + trough_loc_for_duration
            )  # arg for truncated waveform

        # waveform duration in micro seconds
        waveform_duration_peak_trough = (
            10**6
            * np.abs(trough_loc_for_duration - peak_loc_for_duration)
            / param["ephys_sample_rate"]
        )

        # waveform ratios (MATLAB lines 158-162)
        scnd_peak_to_trough_ratio = get_ratio(mainPeak_after_size, mainTrough_size)
        peak1_to_peak2_ratio = get_ratio(mainPeak_before_size, mainPeak_after_size)
        main_peak_to_trough_ratio = get_ratio(max(mainPeak_before_size, mainPeak_after_size), mainTrough_size)
        trough_to_peak2_ratio = get_ratio(mainTrough_size, mainPeak_before_size)
        
        # Set width variables to match MATLAB return values (MATLAB lines 166-169)
        peak_before_width = width_before
        mainTrough_width = trough_width

        # plt.figure(figsize=(8, 6))
        # plt.plot(this_waveform, 'r-', linewidth=2)  
        # plt.show()  # This will display the plot

        # Initialize spatial decay slope
        spatial_decay_slope = np.nan

        if param["computeSpatialDecay"]:
            if np.min(np.diff(np.unique(channel_positions[:, 1]))) < 30:
                param["computeSpatialDecay"] = True
            else:
                param["computeSpatialDecay"] = False
        if param["computeSpatialDecay"]:
            # get waveforms spatial decay across channels
            max_channel = maxChannels[this_unit]

            
            x, y = channel_positions[max_channel, :]
            current_max_channel = channel_positions[max_channel, :]

            # Calculate distances from peak channel in x dimension
            x_dist = np.abs(channel_positions[:, 0] - x)

            # Find channels that are within CHANNEL_TOLERANCE distance in x
            valid_x_channels = np.argwhere(x_dist <= CHANNEL_TOLERANCE).flatten()

            if len(valid_x_channels) < MIN_CHANNELS_FOR_FIT:
                spatial_decay_slope = np.nan
            else: # Skip to next iteration if not enough channels

                # Calculate y distances only for channels that passed x distance check
                y_dist = np.abs(channel_positions[:, 1] - y)

                # Set y distances to max for channels that didn't pass x distance check
                y_dist[~np.isin(np.arange(len(y_dist)), valid_x_channels)] = y_dist.max()

                if param["spDecayLinFit"]:
                    use_these_channels = np.argsort(y_dist)[:NUM_CHANNELS_FOR_FIT] 

                    # Distance fomr the main channels
                    channel_distances = np.sqrt(
                        np.sum(
                            np.square(channel_positions[use_these_channels] - current_max_channel),
                            axis=1,
                        )
                    )

                    spatial_decay_points = np.max(
                        np.abs(template_waveforms[this_unit, :, use_these_channels]), axis=1
                    )

                    sort_idx = np.argsort(channel_distances)
                    channel_distances = channel_distances[sort_idx]
                    spatial_decay_points = spatial_decay_points[sort_idx]

                    if param["normalizeSpDecay"]:
                        spatial_decay_points = spatial_decay_points / np.max(spatial_decay_points)

                    # estimate initial paramters
                    intercept = np.max(
                        spatial_decay_points
                    )  # Take the max value of the max channel
                    grad = (spatial_decay_points[1] - spatial_decay_points[0]) / (
                        channel_distances[1] - channel_distances[0]
                    )

                    # Can add p0 to linear params, but not needed as easier to fit linear curve
                    out_linear = curve_fit(linear_fit, channel_distances, spatial_decay_points)[
                        0
                    ]  #
                    spatial_decay_slope = -out_linear[0]
                else:
                    use_these_channels = np.argsort(y_dist)[:NUM_CHANNELS_FOR_FIT]  

                    # Distance from the main channels
                    channel_distances = np.sqrt(
                        np.sum(
                            np.square(channel_positions[use_these_channels] - current_max_channel),
                            axis=1,
                        )
                    )

                    spatial_decay_points = np.max(
                        np.abs(template_waveforms[this_unit, :, use_these_channels]), axis=1
                    )

                    sort_idx = np.argsort(channel_distances)
                    channel_distances = channel_distances[sort_idx]
                    spatial_decay_points = spatial_decay_points[sort_idx]

                    if param["normalizeSpDecay"]:
                        spatial_decay_points = spatial_decay_points / np.max(spatial_decay_points)

                    # Initial parameters matching MATLAB
                    initial_guess = [0.1, 1]  # [A, b]

                    # Ensure inputs are float64 (equivalent to MATLAB double)
                    channel_distances = np.float64(channel_distances)
                    spatial_decay_points = np.float64(spatial_decay_points)

                    # Curve fit with same initial parameters as MATLAB
                    out_exp = curve_fit(
                        exp_fit,
                        channel_distances,
                        spatial_decay_points,
                        p0=initial_guess,
                        maxfev=5000
                    )[0]
                    spatial_decay_slope = -out_exp[0]

        # get waveform baseline fraction
        waveform_baseline = np.nan
        if waveform_baseline_window is not None and len(waveform_baseline_window) >= 2:
            if ~np.isnan(waveform_baseline_window[0]) and ~np.isnan(waveform_baseline_window[1]):
                baseline_segment = this_waveform_fit[
                    int(waveform_baseline_window[0]) : int(waveform_baseline_window[1])
                ]
                if len(baseline_segment) > 0:
                    waveform_baseline = np.max(np.abs(baseline_segment)) / np.max(np.abs(this_waveform_fit))

        # plt.plot(channel_distances, spatial_decay_points,'o', label = 'Data')
        # plt.plot(channel_distances, out_linear[1] + channel_distances * out_linear[0], label = f'linear fit grad = {out_linear[0]:.4f}')
        # plt.plot(channel_distances, exp_fit(channel_distances, out_exp[0], out_exp[1]), label = f'exp fit decay constant = {out_exp[0]:.4f}')
        # plt.xlabel('Distance um')
        # plt.ylabel('normalised amplitude')
        # plt.legend()

        trough = np.max(troughs)
    # Ensure GUI variables are always defined
    peak_locs_for_gui = locals().get('peakLocs', np.array([]))
    trough_locs_for_gui = locals().get('trough_locs', np.array([]))  # ALL trough locations for GUI
    peak_loc_for_duration_gui = locals().get('peak_loc_for_duration', np.nan)
    trough_loc_for_duration_gui = locals().get('trough_loc_for_duration', np.nan)
    
    return (
        n_peaks,
        n_troughs,
        waveform_duration_peak_trough,
        spatial_decay_slope,
        waveform_baseline,
        scnd_peak_to_trough_ratio,
        peak1_to_peak2_ratio,
        main_peak_to_trough_ratio,
        trough_to_peak2_ratio,
        peak_before_width,
        mainTrough_width,
        peak_locs_for_gui,
        trough_locs_for_gui,
        peak_loc_for_duration_gui,
        trough_loc_for_duration_gui,
        param,
    )

def get_ratio(numerator, denominator):
   if denominator == 0: return float('inf')
   if numerator in (None, 0): return 0.0
   return abs(numerator / denominator)

@njit(cache=True)
def custom_mahal_loop(test_spike_features, current_spike_features):
    """
    Calualtes the mahal distance for the pc of of other spikes against the
    ditrbution fomr all of the spike from this units
    #NOTE could optimse further as the cov, inv_covmat dont need to be calcualted for different spikes

    Parameters
    ----------
    test_spike_features : ndarray (n_spikes_other, pc_sie * n_channels)
        from the other unit
    current_spike_features : ndarray(n_spikes, pc_size * n_channels)
        _description_

    Returns
    -------
    mahal : ndarray (n_spikes_other)
        The mahal score for the test units against the current unit distribution
    """
    # inv covarriance metric from the current spike
    cov = np.cov(current_spike_features.T)
    inv_covmat = np.linalg.inv(cov)
    # numba cant do mean with axis =
    mean_data = np.zeros(current_spike_features.shape[1])
    for i in range(current_spike_features.shape[1]):
        mean_data[i] = np.mean(current_spike_features[:, i])

    test_features_mu = test_spike_features - mean_data[np.newaxis, :]
    mahal = np.zeros(test_spike_features.shape[0])

    # it is faster as loop over spikes than doing it vecotrised to avoid alrger matrix calcualtions!
    for i in range(test_spike_features.shape[0]):
        y_mu = test_features_mu[i, :]
        left = np.dot(y_mu, inv_covmat)
        mahal[i] = np.dot(left, y_mu.T)

    return mahal


def get_distance_metrics(
    pc_features, pc_features_idx, this_unit, spike_clusters, param
):
    """
    Generates functional distance based metrics, such as L-ratio mahalanobis distance

    Parameters
    ----------
    pc_features : ndarray
        The top 3 PC features for the 32 most active channels for each unit
    pc_features_idx : ndarray
        Which channels are used for each unit
    this_unit : int
        The current unit id
    spike_clusters : ndarray
        The array which assigns each spike to a unit
    param : dict
        The param dictionary

    Returns
    -------
    isolation_dist : float
        The isolation distance
    L_ratio : float
        The L ratio
    silhouette_score : float
        The silhouette score
    """
    # get distance metrics

    n_pcs = pc_features.shape[1]  # should be 3

    # get current unit max 'n_chans_to_use' chanels
    these_channels = pc_features_idx[this_unit, 0 : param["nChannelsIsoDist"]]

    # current units features
    this_unit_idx = spike_clusters == this_unit
    n_spikes = this_unit_idx.sum()
    these_features = np.reshape(
        pc_features[this_unit_idx, :, : param["nChannelsIsoDist"]],
        (n_spikes, -1),
    )

    # precompute unique identifiers and allocate space for outputs
    unique_ids = np.unique(spike_clusters) # np.unique(spike_clusters[spike_clusters > 0])
    mahalanobis_distance = np.zeros(unique_ids.size)  # JF: i don't think is used
    other_units_double = np.zeros(unique_ids.size)  # JF: i don't think is used
    # NOTE the first dimension here maybe the prbolem?
    other_features = np.zeros(
        (pc_features.shape[0], pc_features.shape[1], param["nChannelsIsoDist"])
    )
    other_features_ind = np.full(
        (pc_features.shape[0], param["nChannelsIsoDist"]), np.nan
    )
    n_count = 0  # ML/python difference

    for i, id in enumerate(unique_ids):
        if id == this_unit:
            continue

        # identify channels associated with the current ID
        current_channels = pc_features_idx[id, :]
        other_spikes = np.squeeze(spike_clusters == id)

        # process channels that are common between current channels and the unit of interest
        # NOTE This bit could likely be faster.
        for channel_idx in range(param["nChannelsIsoDist"]):
            if np.isin(these_channels[channel_idx], current_channels):
                common_channel_idx = np.squeeze(np.argwhere(current_channels == these_channels[channel_idx]))
                channel_spikes = pc_features[other_spikes, :, common_channel_idx]
                other_features[
                    n_count : n_count + channel_spikes.shape[0], :, channel_idx
                ] = channel_spikes
                other_features_ind[
                    n_count : n_count + channel_spikes.shape[0], channel_idx
                ] = id
                # n_count += channel_spikes.shape[0]

        # NOTE i think n_count shouldnt be in the each channel loop?
        if np.any(np.isin(these_channels, current_channels)):
            n_count = n_count + channel_spikes.shape[0]

        # #calculate mahalanobis distance if applicable
        # if np.any(np.isin(these_channels, current_channels)):
        #     row_indicies = np.argwhere(other_features_ind == id)[:,0]
        #     if np.logical_and(these_features.shape[0] > these_features.shape[1], row_indicies.size > these_features.shape[1]):
        #         other_features_reshaped = np.reshape(other_features[row_indicies], (row_indicies.size, n_pcs * param['nChannelsIsoDist']))
        #         #NOTE try using different functions
        #         mahalanobis_distance[i] = np.nanmean(custom_mahal_loop(other_features_reshaped, these_features))

    # predefine outputs
    isolation_dist = np.nan
    L_ratio = np.nan
    silhouette_score = np.nan
    mahal_D = np.nan  # JF: I don't think this is used
    histogram_mahal_units_counts = np.nan
    histogram_mahal_units_edges = np.nan
    histogram_mahal_noise_counts = np.nan
    histogram_mahal_noise_edges = np.nan

    # reshape features for the mahalanobis distance calc if there are other features
    # any other units have spikes at active channels and enough spikes to test
    other_features = other_features[~np.isnan(other_features_ind[:, 0]), :, :]
    other_features_ind = other_features_ind[~np.isnan(other_features_ind)]
    #####FROM HERE !!!!
    if np.logical_and(
        np.any(~np.isnan(other_features_ind)),
        n_spikes > param["nChannelsIsoDist"] * n_pcs,
    ):
        other_features = np.reshape(other_features, (other_features.shape[0], -1))

        mahal_sort = np.sort(custom_mahal_loop(other_features, these_features))
        L = np.sum(1 - chi2.cdf(mahal_sort, n_pcs * param["nChannelsIsoDist"]))
        L_ratio = L / n_spikes

        if np.logical_and(
            n_count > n_spikes, n_spikes > n_pcs * param["nChannelsIsoDist"]
        ):
            isolation_dist = mahal_sort[n_spikes]

        mahal_self = custom_mahal_loop(these_features, these_features)
        mahal_self_sort = np.sort(mahal_self)  # JF: I don't think this is used

    # plt.hist(mahal_self, bins = 50, range = (0, np.quantile(mahal_self, 0.995)), density = True, histtype = 'step', label = 'mahal')
    # plt.hist(mahal_sort, bins = 50, range = (0, np.quantile(mahal_sort, 0.995)), density = True, histtype = 'step', label = 'inter unit mahal')
    # plt.title(f' L-ratio = {L_ratio:.4f}')
    # plt.xlabel('mahalanobis_distance')
    # plt.ylabel('probability')
    # plt.legend()

    return (
        isolation_dist,
        L_ratio,
        silhouette_score,
    )


def get_raw_amplitude(this_raw_waveform, gain_to_uV, peak_channel=None):
    """
    The raw amplitude from extracted average waveforms

    Parameters
    ----------
    this_raw_waveform : ndarray
        The extracted raw average waveforms (n_channels, spike_width) or (spike_width,) if single channel
    gain_to_uV : float
        The waveform gain
    peak_channel : int, optional
        The peak channel to use for amplitude calculation. If None, uses all channels (old behavior)

    Returns
    -------
    raw_ampltitude : float
        The actual raw amplitude
    """
    this_raw_waveform_tmp = this_raw_waveform.copy()
    
    # If peak_channel is specified and waveform is multi-channel, use only that channel
    if peak_channel is not None and this_raw_waveform_tmp.ndim == 2:
        this_raw_waveform_tmp = this_raw_waveform_tmp[peak_channel, :]
    
    if ~np.isnan(gain_to_uV):
        this_raw_waveform_tmp *= gain_to_uV
        raw_amplitude = np.abs(np.nanmax(this_raw_waveform_tmp)) + np.abs(
            np.nanmin(this_raw_waveform_tmp)
        )
    else:
        raw_amplitude = np.nan

    return raw_amplitude


def get_quality_unit_type(param, quality_metrics):
    """
    Classifies neural units based on quality metrics.
    
    Unit Types:
    0: Noise units
    1: Good units
    2: MUA (Multi-Unit Activity)
    3: Non-somatic units (good if split)
    4: Non-somatic MUA (if split)

    Parameters
    ----------
    param : dict
        The parameters
    quality_metrics : dict
        The quality metrics
    
    Returns
    -------
    unit_type : ndarray
        The unit type classifaction as a number
    unit_type_string : ndarray
        The unit type classification as a string
    """
    n_units = len(quality_metrics["nPeaks"])
    unit_type = np.full(n_units, np.nan)
    
    # Noise classification
    noise_mask = (
        np.isnan(quality_metrics["nPeaks"]) |
        (quality_metrics["nPeaks"] > param["maxNPeaks"]) |
        (quality_metrics["nTroughs"] > param["maxNTroughs"]) |
        (quality_metrics["waveformDuration_peakTrough"] < param["minWvDuration"]) |
        (quality_metrics["waveformDuration_peakTrough"] > param["maxWvDuration"]) |
        (quality_metrics["waveformBaselineFlatness"] > param["maxWvBaselineFraction"]) |
        (quality_metrics["scndPeakToTroughRatio"] > param["maxScndPeakToTroughRatio_noise"])
    )

    if param["computeSpatialDecay"] & param["spDecayLinFit"]:
        noise_mask |= (quality_metrics["spatialDecaySlope"] < param["minSpatialDecaySlope"])
    elif param["computeSpatialDecay"]:
        noise_mask |= (
            (quality_metrics["spatialDecaySlope"] < param["minSpatialDecaySlopeExp"]) |
            (quality_metrics["spatialDecaySlope"] > param["maxSpatialDecaySlopeExp"])
        )
    
    unit_type[noise_mask] = 0
    
    # Non-somatic classification
    is_non_somatic = (
        (quality_metrics["troughToPeak2Ratio"] < param["minTroughToPeak2Ratio_nonSomatic"]) &
        (quality_metrics["mainPeak_before_width"] < param["minWidthFirstPeak_nonSomatic"]) &
        (quality_metrics["mainTrough_width"] < param["minWidthMainTrough_nonSomatic"]) &
        (quality_metrics["peak1ToPeak2Ratio"] > param["maxPeak1ToPeak2Ratio_nonSomatic"]) |
        (quality_metrics["mainPeakToTroughRatio"] > param["maxMainPeakToTroughRatio_nonSomatic"])
    )
    
    # MUA classification
    mua_mask = np.isnan(unit_type) & (
        (quality_metrics["percentageSpikesMissing_gaussian"] > param["maxPercSpikesMissing"]) |
        (quality_metrics["nSpikes"] < param["minNumSpikes"]) |
        (quality_metrics["fractionRPVs_estimatedTauR"] > param["maxRPVviolations"]) |
        (quality_metrics["presenceRatio"] < param["minPresenceRatio"])
    )
    
    if param["extractRaw"] and np.all(~np.isnan(quality_metrics['rawAmplitude'])):
        mua_mask |= np.isnan(unit_type) & (
            (quality_metrics["rawAmplitude"] < param["minAmplitude"]) |
            (quality_metrics["signalToNoiseRatio"] < param["minSNR"])
        )
    
    if param["computeDrift"]:
        mua_mask |= np.isnan(unit_type) & (quality_metrics["maxDriftEstimate"] > param["maxDrift"])
    
    if param["computeDistanceMetrics"]:
        mua_mask |= np.isnan(unit_type) & (
            (quality_metrics["isolationDistance"] < param["isoDmin"]) |
            (quality_metrics["Lratio"] > param["lratioMax"])
        )
    
    unit_type[mua_mask] = 2
    unit_type[np.isnan(unit_type)] = 1
    
    # Handle non-somatic classifications
    if param["splitGoodAndMua_NonSomatic"]:
        good_non_somatic = (unit_type == 1) & is_non_somatic
        mua_non_somatic = (unit_type == 2) & is_non_somatic
        unit_type[good_non_somatic] = 3
        unit_type[mua_non_somatic] = 4
    else:
        unit_type[(unit_type != 0) & is_non_somatic] = 3
    
    # Create string labels
    labels = {0: "NOISE", 1: "GOOD", 2: "MUA", 
             3: "NON-SOMA GOOD" if param["splitGoodAndMua_NonSomatic"] else "NON-SOMA",
             4: "NON-SOMA MUA"}
    
    unit_type_string = np.full(n_units, "", dtype=object)
    for code, label in labels.items():
        unit_type_string[unit_type == code] = label
    
    return unit_type, unit_type_string