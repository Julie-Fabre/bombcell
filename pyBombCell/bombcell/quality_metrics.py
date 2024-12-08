import os

import numpy as np
from numba import njit

from scipy.optimize import curve_fit
from scipy.signal import medfilt, find_peaks
from scipy.stats import norm, chi2

import matplotlib.pyplot as plt

# import bombcell.extract_raw_waveforms as erw
# import bombcell.loading_utils as led
# import bombcell.default_parameters as params
from bombcell.save_utils import path_handler


def get_waveform_max_channel(template_waveforms):
    """
    Get the max channel for all templates (channel with largest amplitude)

    Parameters
    ----------
    template_waveforms : ndarray (n_templates, n_time_points, n_channels)
        The template waveforms for each template and channel

    Returns
    -------
    ndarray (n_templates)
        The channel with maximum amplitude for each template
    """
    max_value = np.max(template_waveforms, axis=1)
    min_value = np.min(template_waveforms, axis=1)
    max_channels = np.argmax(max_value - min_value, axis=1)

    return max_channels


@njit(cache=True)
def remove_duplicates(
    batch_spike_times_samples,
    batch_spike_templates,
    batch_template_amplitudes,
    batch_spike_templates_flat,
    max_channels,
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
    batch_spike_templates : ndarray
        A batch of spike templates
    batch_template_amplitudes : ndarray
        A batch of spike template amplitudes
    batch_spike_templates_flat : ndarray
        A batch of the flattened spike templates
    max_channels : ndarray
        The max channel for each unit
    duplicate_spike_window_samples : int
        The length of time in samples which marks a pair of overlapping spike

    Returns
    -------
    ndarray
        An array which if 1 states that spike should be removed
    """

    num_spikes = batch_spike_times_samples.shape[0]
    remove_idx = np.zeros(num_spikes)
    # spike counts for the batch
    unit_spike_counts = np.bincount(batch_spike_templates[:, 0])

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
                max_channels[batch_spike_templates_flat[spike_idx1]]
                != max_channels[batch_spike_templates_flat[spike_idx2]]
            ):
                continue
            # intra-unit removal
            if batch_spike_templates[spike_idx1] == batch_spike_templates[spike_idx2]:
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
            if batch_spike_templates[spike_idx1] != batch_spike_templates[spike_idx2]:
                if (
                    np.abs(
                        batch_spike_times_samples[spike_idx1]
                        - batch_spike_times_samples[spike_idx2]
                    )
                    <= duplicate_spike_window_samples
                ):
                    # keep spike from unit with less spikes
                    if (
                        unit_spike_counts[batch_spike_templates[spike_idx1]]
                        < unit_spike_counts[batch_spike_templates[spike_idx2]]
                    ):
                        batch_spike_times_samples[spike_idx1] = np.nan
                        remove_idx[spike_idx1] = 1
                    else:
                        batch_spike_times_samples[spike_idx2] = np.nan
                        remove_idx[spike_idx2] = 1

    return remove_idx


def remove_duplicate_spikes(
    spike_times_samples,
    spike_templates,
    template_amplitudes,
    max_channels,
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
    spike_templates : ndarray
        The array which assigns each spike a id
    template_amplitudes : ndarray
        The array of amplitudes for each spike
    max_channels : ndarray
        The max channel for each spike
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
    tuple
        All of the arrays with duplicate spikes removed
    """

    # Create save_path if it does not exist
    save_path = path_handler(save_path)

    if param["remove_duplicate_spike"]:
        # check if spikes are already extract or need to recompute
        if param["recompute_duplicate_spike"] or ~os.path.isdir(
            os.path.join(save_path, "spikes._bc_duplicateSpikes.npy")
        ):

            # parameters
            duplicate_spike_window_samples = (
                param["duplicate_spikes_window_s"] * param["ephys_sample_rate"]
            )
            batch_size = 10000
            overlap_size = 100
            num_spikes_full = spike_times_samples.shape[0]

            # initialize and re-allocate
            duplicate_spike_idx = np.zeros(num_spikes_full)

            # rename the spike templates according to the remaining templates
            good_templates_idx = np.unique(spike_templates)
            new_spike_idx = np.full(max(good_templates_idx) + 1, np.nan)
            new_spike_idx[good_templates_idx] = np.arange(good_templates_idx.shape[0])
            spike_templates_flat = (
                new_spike_idx[spike_templates].squeeze().astype(np.int32)
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
                batch_spike_templates = spike_templates[start_idx:end_idx]
                batch_template_amplitudes = template_amplitudes[start_idx:end_idx]
                batch_spike_templates_flat = spike_templates_flat[start_idx:end_idx]

                batch_remove_idx = remove_duplicates(
                    batch_spike_times_samples,
                    batch_spike_templates,
                    batch_template_amplitudes,
                    batch_spike_templates_flat,
                    max_channels,
                    duplicate_spike_window_samples,
                )
                duplicate_spike_idx[start_idx:end_idx] = batch_remove_idx

            if param["save_spike_without_duplicates"]:
                np.save(
                    os.path.join(save_path, "spikes._bc_duplicateSpikes.npy"),
                    duplicate_spike_idx,
                )

        else:
            duplicate_spike_idx = np.load(
                os.path.join(save_path, "spikes._bc_duplicateSpikes.npy")
            )

        # check if there are any empty units
        unique_templates = np.unique(spike_templates)
        non_empty_units = np.unique(spike_templates[duplicate_spike_idx == 0])
        empty_unit_idx = np.isin(unique_templates, non_empty_units, invert=True)

        # remove any empty units and duplicate spikes
        spike_times_samples = spike_times_samples[np.argwhere(duplicate_spike_idx == 0)]
        spike_templates = spike_templates[np.argwhere(duplicate_spike_idx == 0)]
        template_amplitudes = template_amplitudes[np.argwhere(duplicate_spike_idx == 0)]

        if pc_features is not None:
            pc_features = pc_features[
                [np.argwhere(duplicate_spike_idx == 0)], :, :
            ].squeeze()

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
            spike_templates,
            template_amplitudes,
            pc_features,
            raw_waveforms_full,
            raw_waveforms_peak_channel,
            signal_to_noise_ratio,
            max_channels,
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
    ndarray
        The cutoff gaussian
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
    bool
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


def perc_spikes_missing(these_amplitudes, these_spike_times, time_chunks, param):
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

    Returns
    -------
    tuple
        The estimates of percentage missing spikes and the amplitude distributions
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

        spike_counts_per_amp_bin, bins = np.histogram(
            these_amplitudes[chunk_idx == 1], bins=n_bins
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
            max_amp_bins_smooth = np.argmax(spike_counts_per_amp_bin_smooth)
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
            p_missing = (
                (surrogate_area - np.sum(spike_counts_per_amp_bin) * bin_step)
                / surrogate_area
            ) * 100
            if p_missing < 0:  # If p_missing is -ve, the distribution is not symmetric
                p_missing = 0

            percent_missing_symmetric[time_chunk_idx] = p_missing

            # NOTE currently not doing KS test
            ## KS test, currently is skipped

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
                        np.max(spike_counts_per_amp_bin_gaussian),
                        mode_seed,
                        np.nanstd(these_amplitudes),
                        np.percentile(these_amplitudes, 1),
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
                )[0]
                gaussian_fit = gaussian_cut(
                    amp_bin_gaussian, fit_params[0], fit_params[1], fit_params[2], p0[3]
                )

                norm_area = norm.cdf(
                    (fit_params[1] - fit_params[3]) / np.abs(fit_params[2])
                )
                fit_params_save.append(fit_params)
                test[time_chunk_idx] = (fit_params[1] - fit_params[3]) / fit_params[2]
                percent_missing_gaussian[time_chunk_idx] = 100 * (1 - norm_area)
            else:
                percent_missing_gaussian[time_chunk_idx] = 1  # Use one as a fail here
                gaussian_fit = np.nan
        else:
            percent_missing_gaussian[time_chunk_idx] = np.nan
            percent_missing_symmetric[time_chunk_idx] = np.nan
            # ks_test = np.nan # when doing a ks test
            amp_bin_gaussian = np.nan
            spike_counts_per_amp_bin_gaussian = np.nan
            gaussian_fit = np.nan

        # TODO clean up plots
        if param["show_detail_plots"]:
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

    # #NOT done for each time chunk
    # if param['show_detail_plots']:
    #     plt.scatter(these_spike_times, these_amplitudes, s = 10)
    #     plt.xlim((0,these_spike_times.max()))
    #     plt.ylim((0, these_amplitudes.max()))
    #     plt.xlabel('time (s)')
    #     plt.ylabel('amplitude scaling')
    #     for time_chunk in time_chunks:
    #         plt.axvline(time_chunk, ls = '--')
    return (
        percent_missing_gaussian,
        percent_missing_symmetric
    )


# NOTE ML needs these_amplitudes only for a plot!
def fraction_RP_violations(
    these_spike_times, these_amplitudes, time_chunks, param, use_this_tauR=None
):
    """
    This function estimates the fraction of refractory period violations for a given unit.
    #NOTE this function/method could likely be improved (To many units have more than 1 fraction RPVs)

    Parameters
    ----------
    these_spike_times : ndarray
        The spike times for the given unit
    these_amplitudes : ndarray
        The spike amplitudes for the given unit
    time_chunks : ndarray
        The time chunks to consider
    param : dict
        The param dictionary
    use_this_tauR : float
        The Value of tauR to use after testing different values

    Returns
    -------
    tuple
        The fraction of refractory period violations and the number of violations
    """

    tauR_min = param["tauR_values_min"]
    tauR_max = param["tauR_values_max"]
    tauR_step = param["tauR_values_steps"]

    tauR_window = np.arange(
        tauR_min, tauR_max + tauR_step, tauR_step
    )  # arange doesn't include the end point!

    if use_this_tauR != None:
        # Get the index of the max tauR
        tauR_window = tauR_window[int(use_this_tauR)][
            np.newaxis
        ]  # Keep it as an array with a shape for the loop

    tauC = param["tauC"]

    # initialize arrays
    fraction_RPVs = np.zeros((time_chunks.shape[0] - 1, tauR_window.shape[0]))
    overestimate_bool = np.zeros_like(fraction_RPVs)
    num_violations = np.zeros_like(fraction_RPVs)

    # loop through each time chunk
    for time_chunk_idx in range(time_chunks.shape[0] - 1):
        # number of spikes in a chunks
        chunk_spike_times = these_spike_times[
            np.logical_and(
                these_spike_times >= time_chunks[time_chunk_idx],
                these_spike_times < time_chunks[time_chunk_idx + 1],
            )
        ]
        n_chunk = chunk_spike_times.size

        chunk_ISIs = np.diff(chunk_spike_times)

        duration_chunk = time_chunks[time_chunk_idx + 1] - time_chunks[time_chunk_idx]
        # total times at which refractory period violation can occur
        for i, tauR in enumerate(tauR_window):
            if param["use_hill_method"]:
                ## equivalent to the old code!
                a = (
                    2
                    * (tauR - tauC)
                    * n_chunk**2
                    / (time_chunks[time_chunk_idx + 1] - time_chunks[time_chunk_idx])
                )

                num_violations[time_chunk_idx, i] = np.sum(
                    np.diff(chunk_spike_times) <= tauR
                )

                if (
                    num_violations[time_chunk_idx, i] == 0
                ):  # NO observed refractory period violations
                    # this might be due to having no/few spikes in the region, use presence ratio
                    fraction_RPVs[time_chunk_idx, i] = 0
                    overestimate_bool[time_chunk_idx, i] = 0
                else:  # solve the eqn above
                    rts = np.roots((-1, 1, -num_violations[time_chunk_idx, i] / a))

                    if ~np.all(np.iscomplex(rts)):
                        fraction_RPVs[time_chunk_idx, i] = np.min(rts)
                        overestimate_bool[time_chunk_idx, i] = 0

                    # function returns imaginary number if r is too high: over-estimate number
                    else:
                        overestimate_bool[time_chunk_idx, i] = 1
                        if (
                            num_violations[time_chunk_idx, i] < n_chunk
                        ):  # to not get a negative number or a 0
                            fraction_RPVs[time_chunk_idx, i] = num_violations[
                                time_chunk_idx, i
                            ] / (
                                2
                                * (tauR - tauC)
                                * (n_chunk - num_violations[time_chunk_idx, i])
                            )
                            # fraction_RPVs[time_chunk_idx, i] = num_violations[time_chunk_idx, i] / ((n_chunk - num_violations[time_chunk_idx, i]))
                        else:
                            fraction_RPVs[time_chunk_idx, i] = 1

                    if (
                        fraction_RPVs[time_chunk_idx, i] > 1
                    ):  # A value above 1 makes no sense, the assumptions are failing
                        fraction_RPVs[time_chunk_idx, i] = 1
            else:
                num_violations = np.sum(
                    np.logical_and(chunk_ISIs <= tauR, chunk_ISIs > tauC)
                )  # number of violations

                under_root_value = 1 - num_violations * duration_chunk / (
                    n_chunk**2 * (tauR - tauC)
                )

                if under_root_value >= 0:
                    fraction_RPVs[time_chunk_idx, i] = 1 - np.sqrt(under_root_value)
                else:
                    # assumptions have broken down
                    fraction_RPVs[time_chunk_idx, i] = 1

    return fraction_RPVs, num_violations


def time_chunks_to_keep(
    percent_missing_gaussian,
    fraction_RPVs,
    time_chunks,
    these_spike_times,
    these_amplitudes,
    spike_templates,
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
    spike_templates : ndarray
        The template waveforms of this unit
    spike_times_seconds : ndarray
        The spike times in seconds
    param : dict
        The param dictionary

    Returns
    -------
    tuple
        The values to use for the different arrays
    """
    max_RPVs = param["max_RPV"]
    max_perc_spikes_missing = param["max_perc_spikes_missing"]

    sum_RPV = np.sum(fraction_RPVs, axis=0)
    use_tauR = np.argmax(
        sum_RPV
    )  # gives the index of the tauR which has smallest contamination
    use_these_times_temp = np.zeros(time_chunks.shape[0] - 1)

    use_these_times_temp = np.argwhere(
        np.logical_and(
            percent_missing_gaussian < max_perc_spikes_missing,
            fraction_RPVs[:, use_tauR] < max_RPVs,
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
            longest_chunk_idx = np.argmax(chunk_lengths)

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
    these_spike_templates = spike_templates.copy().astype(np.int32)
    these_spike_templates[
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
        these_spike_templates,
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
    float
        The presence ratio for the unit
    """

    presence_ratio_bin_size = param["presence_ratio_bin_size"]

    # divide recording into bins
    presence_ratio_bins = np.arange(
        use_this_time_start, use_this_time_end, presence_ratio_bin_size
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
    full_bins[spikes_per_bin >= 0.05 * np.percentile(spikes_per_bin, 90)] = 1
    # print(f'The presence threshold spike No. is: {0.05 * np.percentile(spikes_per_bin, 90)}')
    presence_ratio = full_bins.sum() / full_bins.shape[0]
    # print(f'The presence ratio is {presence_ratio}')
    # ADD presence ratio plots
    return presence_ratio


def max_drift_estimate(
    pc_features,
    pc_features_idx,
    spike_templates,
    these_spike_times,
    this_unit,
    channel_positions,
    param,
):
    """
    Calculates the drift of the unit using the PC components for each channels

    Parameters
    ----------
    pc_features : ndarray
        The top 3 PC features for the 32 most active channels for each unit
    pc_features_idx : ndarray
        Which channels are used for each unit
    spike_templates : ndarray
        The array which assigns each spike to a unit
    these_spike_times : ndarray
        The spike times for the current unit
    this_unit : ndarray
        The ID of the current unit
    channel_positions : ndarray
        The (x,y) positions of each channel
    param : dict
        The param dictionary

    Returns
    -------
    tuple
        The max and the cumulative drift estimates
    """
    channel_positions_z = channel_positions[:, 1]
    drift_bin_size = param["drift_bin_size"]

    # good_times_spikes = np.ones_like(spike_templates)
    # good_times_spikes[spike_templates == -1] = 0
    # pc_features_drift = pc_features[good_times_spikes.squeeze() == 1, :, :]
    # spike_templates_current = spike_templates[good_times_spikes == 1].astype(np.int32)

    # pc_features_pc1 = pc_features_drift[spike_templates_current.squeeze() == this_unit, 0, :]
    # pc_features_pc1[pc_features_pc1 < 0] = 0 # remove negative entries

    pc_features_pc1 = pc_features[spike_templates.squeeze() == this_unit, 0, :]
    pc_features_pc1[pc_features_pc1 < 0] = 0  # remove negative entries

    # NOTE test with and without only getting this units pc feature idx here
    # this is just several thousand copies of the same 32/ n_pce_feature array
    # spike_pc_feature = pc_features_idx[spike_templates[spike_templates == this_unit].squeeze(), :] # get channel for each spike
    spike_pc_feature = pc_features_idx[this_unit, :]

    pc_channel_pos_weights = channel_positions_z[spike_pc_feature]

    spike_depth_in_channels = np.sum(
        pc_channel_pos_weights[np.newaxis, :] * pc_features_pc1**2, axis=1
    ) / np.nansum(pc_features_pc1**2, axis=1)

    # estimate cumulative drift

    # NOTE this allow units which are only active briefly to still have two bins
    if these_spike_times.max() - these_spike_times.min() < 2 * drift_bin_size:
        drift_bin_size = (these_spike_times.max() - these_spike_times.min()) / 2

    time_bins = np.arange(
        these_spike_times.min(), these_spike_times.max(), drift_bin_size
    )

    median_spike_depth = np.zeros(time_bins.shape[0] - 1)
    for i, time_bin_start in enumerate(time_bins[:-1]):
        median_spike_depth[i] = np.nanmedian(
            spike_depth_in_channels[
                np.logical_and(
                    these_spike_times >= time_bin_start,
                    these_spike_times < (time_bin_start + drift_bin_size),
                )
            ]
        )
    max_drift_estimate = np.nanmax(median_spike_depth) - np.nanmin(median_spike_depth)
    cumulative_drift_estimate = np.sum(
        np.abs(np.diff(median_spike_depth[~np.isnan(median_spike_depth)]))
    )

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
    ndarray
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
    ndarray
        The y-values
    """
    return A * np.exp(m * x)


def waveform_shape(
    template_waveforms,
    this_unit,
    max_channels,
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
    max_channels : ndarray
        The max channel for each unit
    channel_positions : ndarray
        The (x,y) positions of each channel
    waveform_baseline_window : ndarray
        The waveform baseline start and end points
    param : dict
        The param dictionary

    Returns
    -------
    tuple
        Waveform shape based metrics
    """
    # neemd template_waveforms this_unit max_channel, ephys_sample_rate, channel_positions, baseline_thresh
    # waveform_base_line_window, min_thresh_detect_peaks_trough, first_peak_ratio, normalize_sp_decay, plothis
    min_thresh_detect_peaks_troughs = param["min_thresh_detect_peaks_troughs"]

    this_waveform = template_waveforms[this_unit, :, max_channels[this_unit]]

    # NOTE if using raw waveforms may need to change this !!
    if np.any(np.isnan(this_waveform)):
        n_peaks = np.nan
        n_troughs = np.nan
        is_somatic = np.nan
        peak_locs = np.nan
        trough_locs = np.nan
        waveform_duration_peak_trough = np.nan
        spatial_decay_points = np.full((1, 6), np.nan)
        spatial_decay_slope = np.nan
        waveform_baseline = np.nan
    else:
        # New finding peaks/trough for somatic/non-somatic
        min_prominence = min_thresh_detect_peaks_troughs * np.max(np.abs(this_waveform))

        trough_locs, trough_dict = find_peaks(
            this_waveform * -1, prominence=min_prominence, width=0
        )

        # more than 1 trough find the biggest
        if trough_locs.size > 1:
            max_trough_idx = np.argmax(
                trough_dict["prominences"]
            )  # prominence is =ve even though is trough
            trough_width = trough_dict["widths"][max_trough_idx]
        elif trough_locs.size == 1:
            trough_width = trough_dict["widths"]
            trough_locs = np.atleast_1d(trough_locs)

        if trough_locs.size == 0:
            trough_locs = np.argmin(this_waveform)
            trough_locs = np.atleast_1d(trough_locs)
            trough_width = np.nan
            n_troughs = 1
        else:
            n_troughs = trough_locs.shape[0]

        # get the main trough, if multiple trough have the same value choose first
        main_trough = np.min(this_waveform[trough_locs])  # JF: i don't think is used
        main_trough_idx = np.argmin(this_waveform[trough_locs])
        trough_loc = trough_locs[main_trough_idx]
        troughs = np.abs(this_waveform[trough_locs])

        # find peaks before the trough
        if trough_loc > 2:  # need at least 3 sample to get peak before
            peaks_before_locs, peaks_before_dict = find_peaks(
                this_waveform[:trough_loc], prominence=min_prominence, width=0
            )
            if peaks_before_locs.shape[0] > 1:
                max_peak = np.argmax(
                    peaks_before_dict["prominences"]
                )  # prominence is +ve even though is trough
                peak_before_width = peaks_before_dict["widths"][max_peak]
            else:
                peak_before_width = peaks_before_dict["widths"]
        else:
            peaks_before_locs = np.array(())

        # find peaks after trough
        if this_waveform.shape[0] - trough_loc > 2:
            peaks_after_locs, peaks_after_dict = find_peaks(
                this_waveform[trough_loc:], prominence=min_prominence, width=0
            )
            peaks_after_locs += trough_loc
            if peaks_after_locs.shape[0] > 1:
                max_peak = np.argmax(
                    peaks_after_dict["prominences"]
                )  # prominence is +ve even though is trough
                width_after = peaks_after_dict["widths"][max_peak]
            else:
                width_after = peaks_after_dict["widths"]
        else:
            peaks_after_locs = np.array(())

        # If no peaks found with the min_prominence
        used_max_before = False
        if peaks_before_locs.size == 0:
            if trough_loc > 2:  # need at least 3 sample to get peak before
                peaks_before_locs, peaks_before_dict = find_peaks(
                    this_waveform[:trough_loc],
                    prominence=0.01 * np.max(np.abs(this_waveform)),
                    width=0,
                )
                # only want the biggest of these picks, with smaller prominences
                if peaks_before_locs.shape[0] > 1:
                    max_peak = np.argmax(
                        peaks_before_dict["prominences"]
                    )  # prominence is +ve even though is trough
                    peaks_before_locs = peaks_before_locs[max_peak]
                    peak_before_width = peaks_before_dict["widths"][max_peak]
                else:
                    peak_before_width = np.nan # set to nan if there is no peak before
            else:
                peaks_before_locs = np.array(())

            if peaks_before_locs.size == 0:
                width_before = 0  # 0 if no width_before
                peaks_before_locs = np.argmax(this_waveform[:trough_loc])

            used_max_before = True

        # same for after the major trough
        used_max_after = False
        if peaks_after_locs.size == 0:
            if trough_loc > 2:  # need at least 3 sample to get peak before
                peaks_after_locs, peaks_after_dict = find_peaks(
                    this_waveform[trough_loc:],
                    prominence=0.01 * np.max(np.abs(this_waveform)),
                    width=0,
                )
                # only want the biggest of these picks, with smaller prominences
                if peaks_after_locs.shape[0] > 1:
                    max_peak = np.argmax(
                        peaks_after_dict["prominences"]
                    )  # prominence is +ve even though is trough
                    peaks_after_locs = peaks_after_locs[max_peak] + trough_loc
                    width_after = peaks_after_dict["widths"][max_peak]
                else:
                    peaks_after_widths = np.nan

            if peaks_after_locs.size == 0:
                width_after = 0  # JF: i don't think is used
                peaks_after_locs = np.argmax(this_waveform[trough_loc:]) + trough_loc

            used_max_after = True

        # if neither a peak before or after was detected with the min_prominence, the larger peak is the true peak
        if used_max_before & used_max_after:
            if this_waveform[peaks_before_locs] > this_waveform[peaks_after_locs]:
                used_max_before = False
            else:
                used_max_after = False

        # get the main peaks before and after the trough
        peaks_before_locs = np.atleast_1d(np.asarray(peaks_before_locs))
        main_peak_before = np.max(this_waveform[peaks_before_locs])
        main_peak_before_idx = np.argmax(this_waveform[peaks_before_locs])
        main_peak_before_loc = peaks_before_locs[
            main_peak_before_idx
        ]  # JF: i don't think is used

        peaks_after_locs = np.atleast_1d(np.asarray(peaks_after_locs))
        main_peak_after = np.max(this_waveform[peaks_after_locs])
        main_peak_after_idx = np.argmax(this_waveform[peaks_after_locs])
        main_peak_after_loc = peaks_after_locs[main_peak_after_idx]

        # combine peak information
        if used_max_before & (main_peak_before < min_prominence * 0.5):
            peaks = this_waveform[peaks_after_locs]
            peak_locs = peaks_after_locs
        elif used_max_after & (main_peak_after < min_prominence * 0.5):
            peaks = this_waveform[peaks_before_locs]
            peak_locs = peaks_before_locs
        else:
            peak_locs = np.hstack((peaks_before_locs, peaks_after_locs))
            peaks = this_waveform[peak_locs]

        n_peaks = peaks.size
        n_troughs = troughs.size

        # waveform peak to trough duration
        max_waveform_abs_value = np.max(
            np.abs(this_waveform)
        )  # JF: i don't think is used
        max_waveform_location = np.argmax(np.abs(this_waveform))
        max_waveform_value = this_waveform[max_waveform_location]  # signed value
        if max_waveform_value > 0:  # positive peak
            peak_loc_for_duration = max_waveform_location
            trough_loc_for_duration = np.argmin(this_waveform[peak_loc_for_duration:])
            trough_loc_for_duration = (
                trough_loc_for_duration + peak_loc_for_duration
            )  # arg for truncated waveform
        if max_waveform_value < 0:  # positive peak
            trough_loc_for_duration = max_waveform_location
            peak_loc_for_duration = np.argmax(this_waveform[trough_loc_for_duration:])
            peak_loc_for_duration = (
                peak_loc_for_duration + trough_loc_for_duration
            )  # arg for truncated waveform

        # waveform duration in micro seconds
        waveform_duration_peak_trough = (
            10**6
            * np.abs(trough_loc_for_duration - peak_loc_for_duration)
            / param["ephys_sample_rate"]
        )

        # waveform ratios
        # waveform ratios
        scnd_peak_to_trough_ratio = abs(main_peak_after / main_trough)
        peak1_to_peak2_ratio = abs(main_peak_before / main_peak_after)
        main_peak_to_trough_ratio = max(main_peak_before, main_peak_after) / abs(main_trough) 
        trough_to_peak2_ratio = abs(main_trough / main_peak_before)

        # get waveforms spatial decay across channels
        # DECIDE which fit
        # linear_fit = True
        max_channel = max_channels[this_unit]

        
        x, y = channel_positions[max_channel, :]
        current_max_channel = channel_positions[max_channel, :]

        x_dist = np.abs(channel_positions[:, 0] - x)
        near_x_dist = np.min(x_dist[x_dist != 0])

        not_these_x = np.argwhere(x_dist > near_x_dist)

        y_dist = np.abs(channel_positions[:, 1] - y)
        y_dist[not_these_x] = (
            y_dist.max()
        )  # set the bad x_to max y, this keeps the shape of the array
        # CHOOSE HOW MANY POINT

        use_these_channels = np.argsort(y_dist)[:6]  # Doing 12?

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

        if param["normalize_spatial_decay"]:
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

        # NOTE code is duplicated as linear and exponential need a different number of channels
        use_these_channels = np.argsort(y_dist)[:12]  # Doing 12?

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

        if param["normalize_spatial_decay"]:
            spatial_decay_points = spatial_decay_points / np.max(spatial_decay_points)

        # estimate initial paramters
        intercept = np.max(
            spatial_decay_points
        )  # Take the max value of the max channel
        grad = (spatial_decay_points[1] - spatial_decay_points[0]) / (
            channel_distances[1] - channel_distances[0]
        )

        out_exp = curve_fit(
            exp_fit,
            channel_distances,
            spatial_decay_points,
            p0=(grad, intercept),
            maxfev=2000,
        )[0]

        linear_spatial_decay = out_linear[0]
        exp_spaital_decay = out_exp[0]
        # CURRENTLY just use linear
        spatial_decay_slope = out_linear[0]
        # TRYING EXP
        spatial_decay_slope = out_exp[0]

        # get waveform baseline fraction
        if ~np.isnan(waveform_baseline_window)[0]:
            waveform_baseline = np.max(
                np.abs(
                    this_waveform[
                        waveform_baseline_window[0] : waveform_baseline_window[1]
                    ]
                )
            ) / np.max(np.abs(this_waveform))

        # plt.plot(channel_distances, spatial_decay_points,'o', label = 'Data')
        # plt.plot(channel_distances, out_linear[1] + channel_distances * out_linear[0], label = f'linear fit grad = {out_linear[0]:.4f}')
        # plt.plot(channel_distances, exp_fit(channel_distances, out_exp[0], out_exp[1]), label = f'exp fit decay constant = {out_exp[0]:.4f}')
        # plt.xlabel('Distance um')
        # plt.ylabel('normalised amplitude')
        # plt.legend()

        trough = np.max(troughs)
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
        trough_width,
    )


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
    ndarray (n_spikes_other)
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
    pc_features, pc_features_idx, this_unit, spike_templates, param
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
    spike_templates : ndarray
        The array which assigns each spike to a unit
    param : dict
        The param dictionary

    Returns
    -------
    tuple
        The distance based metrics
    """
    # get distance metrics

    n_pcs = pc_features.shape[1]  # should be 3

    # get current unit max 'n_chans_to_use' chanels
    these_channels = pc_features_idx[this_unit, 0 : param["n_channels_iso_dist"]]

    # current units features
    this_unit_idx = spike_templates == this_unit
    n_spikes = this_unit_idx.sum()
    these_features = np.reshape(
        pc_features[this_unit_idx.squeeze(), :, : param["n_channels_iso_dist"]],
        (n_spikes, -1),
    )

    # precompute unique identifiers and allocate space for outputs
    unique_ids = np.unique(spike_templates[spike_templates > 0])
    mahalanobis_distance = np.zeros(unique_ids.size)  # JF: i don't think is used
    other_units_double = np.zeros(unique_ids.size)  # JF: i don't think is used
    # NOTE the first dimension here maybe the prbolem?
    other_features = np.zeros(
        (pc_features.shape[0], pc_features.shape[1], param["n_channels_iso_dist"])
    )
    other_features_ind = np.full(
        (pc_features.shape[0], param["n_channels_iso_dist"]), np.nan
    )
    n_count = 0  # ML/python difference

    for i, id in enumerate(unique_ids):
        if id == this_unit:
            continue

        # identify channels associated with the current ID
        current_channels = pc_features_idx[id, :]
        other_spikes = np.squeeze(spike_templates == id)

        # process channels that are common between current channels and the unit of interest
        # NOTE This bit could likely be faster.
        for channel_idx in range(param["n_channels_iso_dist"]):
            if np.isin(these_channels[channel_idx], current_channels):
                common_channel_idx = np.argwhere(
                    current_channels == these_channels[channel_idx]
                )
                channel_spikes = pc_features[
                    other_spikes, :, common_channel_idx
                ].squeeze()
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
        #         other_features_reshaped = np.reshape(other_features[row_indicies], (row_indicies.size, n_pcs * param['n_channels_iso_dist']))
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
        n_spikes > param["n_channels_iso_dist"] * n_pcs,
    ):
        other_features = np.reshape(other_features, (other_features.shape[0], -1))

        mahal_sort = np.sort(custom_mahal_loop(other_features, these_features))
        L = np.sum(1 - chi2.cdf(mahal_sort, n_pcs * param["n_channels_iso_dist"]))
        L_ratio = L / n_spikes

        if np.logical_and(
            n_count > n_spikes, n_spikes > n_pcs * param["n_channels_iso_dist"]
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


def get_raw_amplitude(this_raw_waveform, gain_to_uV):
    """
    The raw amplitude from extracted average waveforms

    Parameters
    ----------
    this_raw_waveform : ndarray
        The extracted raw average waveforms
    gain_to_uV : float
        The waveform gain

    Returns
    -------
    float
        The actual raw amplitude
    """

    this_raw_waveform *= gain_to_uV
    raw_amplitude = np.abs(np.max(this_raw_waveform)) + np.abs(
        np.min(this_raw_waveform)
    )

    return raw_amplitude


def get_quality_unit_type(param, quality_metrics):
    """
    Assign each unit a type based of its' quality metrics.
    unit_type == 0 all noise units
    unit_type == 1 all good units
    unit_type == 2 all mua units
    unit_type == 3 all non-somatic units (if split somatic units its good non-somatic units)
    unit_type == 4 (if split somatic units its mua non-somatic units)


    Parameters
    ----------
    param : df
        The param dataframe from ML BombCell
    quality_metrics : df
        The quality metrics dataframefrom ML BombCell

    Returns
    -------
    tuple (np array, np array)
        Two array of the unit types one as number the other as strings
    """


    unit_type = np.full(quality_metrics["n_peaks"].shape[0], np.nan)

    # classify noise
    unit_type[np.isnan(quality_metrics["n_peaks"])] = 0
    unit_type[quality_metrics["n_peaks"] > param["max_n_peaks"]] = 0
    unit_type[quality_metrics["n_troughs"] > param["max_n_troughs"]] = 0
    unit_type[
        quality_metrics["waveform_duration_peak_trough"] < param["min_wv_duration"]
    ] = 0
    unit_type[
        quality_metrics["waveform_duration_peak_trough"] > param["max_wv_duration"]
    ] = 0
    unit_type[
        quality_metrics["waveform_baseline_flatness"] > param["max_wv_baseline_fraction"]
    ] = 0
    if param["sp_decay_lin_fit"]:
        unit_type[quality_metrics["spatial_decay_slope"] < param["min_spatial_decay_slope_exp"]] = 0
        unit_type[quality_metrics["spatial_decay_slope"] > param["max_spatial_decay_slope_exp"]] = 0
    else:
        unit_type[quality_metrics["spatial_decay_slope"] < param["min_spatial_decay_slope"]] = 0

    unit_type[quality_metrics["scnd_peak_to_trough_ratio"] > param["max_scnd_peak_to_trough_ratio_noise"]] = 0
    # classify non-somatic 

    is_non_somatic = (
    (quality_metrics["trough_to_peak2_ratio"] < param["min_trough_to_peak2_ratio_non_somatic"]) &
    (quality_metrics["peak_before_width"] < param["min_width_first_peak_non_somatic"]) &
    (quality_metrics["trough_width"] < param["min_width_main_trough_non_somatic"]) &
    (quality_metrics["peak1_to_peak2_ratio"] > param["max_peak1_to_peak2_ratio_non_somatic"])
    ) | (quality_metrics["main_peak_to_trough_ratio"] > param["max_main_peak_to_trough_ratio_non_somatic"])
    

    # classify MUA
    # ALL or ANY?
    unit_type[
        np.logical_and(
            quality_metrics["percent_missing_gaussian"]
            > param["max_perc_spikes_missing"],
            np.isnan(unit_type),
        )
    ] = 2
    unit_type[
        np.logical_and(
            quality_metrics["n_spikes"] < param["min_num_spikes_total"],
            np.isnan(unit_type),
        )
    ] = 2
    unit_type[
        np.logical_and(
            quality_metrics["fraction_RPVs"] > param["max_RPV"], np.isnan(unit_type)
        )
    ] = 2
    unit_type[
        np.logical_and(
            quality_metrics["presence_ratio"] < param["min_presence_ratio"],
            np.isnan(unit_type),
        )
    ] = 2

    if param["extract_raw_waveforms"]:
        unit_type[
            np.logical_and(
                quality_metrics["raw_amplitude"] < param["min_amplitude"],
                np.isnan(unit_type),
            )
        ] = 2
        unit_type[
            np.logical_and(
                quality_metrics["signal_to_noise_ratio"] < param["min_SNR"],
                np.isnan(unit_type),
            )
        ] = 2

    if param["compute_drift"]:
        unit_type[
            np.logical_and(
                quality_metrics["max_drift_estimate"] > param["max_drift"],
                np.isnan(unit_type),
            )
        ] = 2

    if param["compute_distance_metrics"]:
        unit_type[
            np.logical_and(
                quality_metrics["isolation_dist"] > param["iso_d_min"],
                np.isnan(unit_type),
            )
        ] = 2
        unit_type[
            np.logical_and(
                quality_metrics["l_ratio"] > param["lratio_max"], np.isnan(unit_type)
            )
        ] = 2

    unit_type[np.isnan(unit_type)] = 1  # SINGLE SEXY UNIT

    if param["split_good_and_mua_non_somatic"]:
        unit_type[np.logical_and(is_non_somatic == 1, unit_type == 1)] = (
            3  # Good non-somatic
        )
        unit_type[np.logical_and(is_non_somatic == 1, unit_type == 2)] = (
            4  # MUA non-somatic
        )
    else:
        unit_type[np.logical_and(is_non_somatic == 1, unit_type != 0)] = (
            3  # Good non-somatic
        )

    # Have unit types as strings as well
    unit_type_string = np.full(unit_type.size, "", dtype=object)
    unit_type_string[unit_type == 0] = "NOISE"
    unit_type_string[unit_type == 1] = "GOOD"
    unit_type_string[unit_type == 2] = "MUA"

    if param["split_good_and_mua_non_somatic"]:
        unit_type_string[unit_type == 3] = "NON-SOMA GOOD"
        unit_type_string[unit_type == 4] = "NON-SOMA MUA"
    else:
        unit_type_string[unit_type == 3] = "NON-SOMA"

    return unit_type, unit_type_string
