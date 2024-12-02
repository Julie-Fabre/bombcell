import os
from pathlib import Path
from joblib import Parallel, delayed

import numpy as np

from mtscomp import Reader
from scipy.signal import detrend
from scipy.ndimage import gaussian_filter

from bombcell.save_utils import path_handler


def read_meta(meta_path):
    """
    Read the meta file attached to a recording.

    Parameters
    ----------
    meta_path : path
        The path to the .meta file
    """
    meta_dict = {}
    with meta_path.open() as f:
        mdat_list = f.read().splitlines()
        # convert the list entries into key value pairs
        for m in mdat_list:
            cs_list = m.split(sep="=")
            if cs_list[0][0] == "~":
                curr_key = cs_list[0][1 : len(cs_list[0])]
            else:
                curr_key = cs_list[0]
            meta_dict.update({curr_key: cs_list[1]})

    return meta_dict


def process_a_unit(
    raw_data,
    spike_width,
    half_width,
    all_spikes_idxs,
    n_channels_rec,
    n_channels,
    n_sync_channels,
    cid,
    detrend_waveforms,
    save_multiple_raw,
    waveform_baseline_noise,
    save_path,
):
    """
    Reads in data from a unit, and processes the data.

    Parameters
    ----------
    raw_data : memmap
        The numpy memmap of the raw data
    spike_width : int
        The total numple of samples to take per unit
    half_width : int
        The number of samples before the spike starts
    all_spikes_idxs : ndarray (n_spike_to_extract)
        All of the spikes idnexs of spike to extract for the unit
    n_channels_rec : int
        The total number of channels in the recording
    n_channels : int
        The number of good recording channels (e.g non-sync channels)
    n_sync_channels : int
        The number of sync channel in the recording
    cid : int
        The id of the cluster
    detrend_waveforms : bool
        If True will lineraly detrend the waveforms over time
    save_multiple_raw : bool
        If true will prepare and save wavefrosm suitable for UnitMatch
    waveform_baseline_noise : int
        The number of samples before the waveform which are nosie
    save_path : path
        The path to the directory to save the UnitMatch data


    Returns
    -------
    cluster_raw_waveforms
        A dictionary of the necessary infomation extract from the raw data
    """
    # Create save_path if it does not exist
    save_path = path_handler(save_path)

    cluster_raw_waveforms = {}
    spike_idx = all_spikes_idxs[~np.isnan(all_spikes_idxs)]
    n_spikes_sampled = spike_idx.size

    # create empty array for the spike map
    spike_map = np.full(
        (n_channels - n_sync_channels, spike_width, n_spikes_sampled), np.nan
    )
    for i, sid in enumerate(spike_idx[~np.isnan(spike_idx)]):
        # get the raw data for each spike chosen
        tmp = raw_data[
            int(sid - half_width - 1) : int(sid + spike_width - half_width - 1),
            np.arange(n_channels_rec),
        ]
        # -1, to better fit with ML, adn python indexing!
        tmp.astype(np.float64)

        # option to remove a linear in time trends
        if detrend_waveforms:
            spike_map[:, :, i] = detrend(tmp[:, :-n_sync_channels], axis=0).swapaxes(
                0, 1
            )
        else:
            spike_map[:, :, i] = tmp[:, :-n_sync_channels].swapaxes(0, 1)

    # Save average waveforms for unitmatch
    # create the waveforms now, save later
    if save_multiple_raw:
        tmp_spike_map = spike_map.swapaxes(0, 1)  # allign with UnitMatch

        # smooth over axis at once
        tmp_spike_map = gaussian_filter(tmp_spike_map, axes=0, sigma=1, radius=2)
        # matches matlab smoothdata, EXCEPT at boundaries!
        tmp_spike_map -= np.mean(tmp_spike_map[:waveform_baseline_noise, :, :], axis=0)[
            np.newaxis, :, :
        ]

        # split into 2 CV for unitmatch!
        UM_CV_limit = np.floor(n_spikes_sampled / 2).astype(int)
        avg_waveforms = np.full((spike_width, n_channels - n_sync_channels, 2), np.nan)
        avg_waveforms[:, :, 0] = np.median(tmp_spike_map[:, :, :UM_CV_limit], axis=-1)
        avg_waveforms[:, :, 1] = np.median(tmp_spike_map[:, :, UM_CV_limit:], axis=-1)

        np.save(
            os.path.join(save_path, "RawWaveforms", (f"Unit{cid}_RawSpikes.npy")),
            avg_waveforms[:, :, :],
        )

    # get average, baseline-subtracted waveforms, Not smoothing as a mandatory processing step!
    spike_map_mean = np.nanmean(spike_map, axis=2)
    raw_waveforms_full = (
        spike_map_mean
        - spike_map_mean[:, :waveform_baseline_noise].mean(axis=1)[:, np.newaxis]
    )

    # Smoothing can be applied later
    # spike_map_smoothed = gaussian_filter(raw_waveforms[f'cluster_{cid}']['spike_map_mean'], axes = 1, sigma = 1, radius = 2)

    # find peak channel
    raw_waveforms_peak_channel = np.argmax(
        np.max(raw_waveforms_full[:, :], axis=1)
        - np.min(raw_waveforms_full[:, :], axis=1)
    )
    average_baseline = np.nanmean(
        spike_map[int(raw_waveforms_peak_channel), :waveform_baseline_noise, :], axis=1
    )

    del spike_map  # free up memory

    cluster_raw_waveforms["spike_map_mean"] = spike_map_mean
    cluster_raw_waveforms["average_baseline"] = average_baseline
    cluster_raw_waveforms["raw_waveforms_full"] = raw_waveforms_full
    cluster_raw_waveforms["raw_waveforms_peak_channel"] = raw_waveforms_peak_channel
    cluster_raw_waveforms["spike_idxs"] = spike_idx
    return cluster_raw_waveforms


def unpack_dicts(
    all_waveforms,
    spike_width,
    n_clusters,
    unique_clusters,
    clus_spike_times,
    n_channels,
    n_sync_channels,
    waveform_baseline_noise,
):
    """
    Unpacks the list of dictionaries from the parallel computation to a nested dictionary and arrays, like in the MatLab version

    Parameters
    ----------
    all_waveforms : list of dictionaries
        The result of the parallel extraction and processing of the raw data.
    spike_width : int
        The nymber of sampples in an extracted spike.
    n_clusters : int
        The number of unique clusters extracted
    unique_clusters : ndarray (n_clusters)
        The idxs of the clusters.
    clus_spike_times : ndarray (n_clusters, n_spikes_to_extract)
        All of the idxs of all of the spikes which were extracted.
    n_channels : int
        The number of good recording channels (e.g non-sync channels)
    n_sync_channels : int
        The number of sync channel in the recording
    waveform_baseline_noise : int
        The number of samples before the waveform which are nosie

    Returns
    -------
    raw_waveforms : dictionary
        A nested dicitonary which contains a dictionary of waveform properties for each cluster.
    raw_waveforms_full : ndarray (n_clusters, n_channles, spike_width)
        All extracted average wavefroms
    raw_waveforms_peak_channels : ndarray (n_clusters)
        The peak channel for each cluster
    average_baseline : ndarray (n_clusters, wavefroms_baseline_noise)
        The averge value for each clusters of the time samples before the signal

    """
    raw_waveforms = {}
    raw_waveforms_full = np.full(
        (n_clusters, n_channels - n_sync_channels, spike_width), np.nan
    )
    raw_waveforms_peak_channel = np.full((n_clusters), np.nan)
    average_baseline = np.full((n_clusters, waveform_baseline_noise), np.nan)

    for i, cid in enumerate(unique_clusters):
        raw_waveforms[f"cluster_{cid}"] = {}
        raw_waveforms[f"cluster_{cid}"]["spike_map_mean"] = all_waveforms[i][
            "spike_map_mean"
        ]
        raw_waveforms[f"cluster_{cid}"]["spike_idx_sampled"] = all_waveforms[i][
            "spike_idxs"
        ]
        raw_waveforms[f"cluster_{cid}"]["Cluster_idx"] = cid
        raw_waveforms[f"cluster_{cid}"]["spike_idxs"] = clus_spike_times[i]

        raw_waveforms_full[i, :, :] = all_waveforms[i]["raw_waveforms_full"]
        raw_waveforms_peak_channel[i] = all_waveforms[i]["raw_waveforms_peak_channel"]
        average_baseline[i] = all_waveforms[i]["average_baseline"]

    return (
        raw_waveforms,
        raw_waveforms_full,
        raw_waveforms_peak_channel,
        average_baseline,
    )


def extract_raw_waveforms(
    param, spike_templates, spike_times, re_extract_waveforms, save_path
):
    """
    Extracts avereage raw wavefroms from the raw data, can be used to get raw wavefroms for UnitMatch as well

    Parameters
    ----------
    param : dict
        The param dictionairy used in BombCell
    spike_templates : ndarry
        The array which assigns a spike to a cluster
    spike_times : ndarray
        The array which gives the sample number for each spike
    re_extract_waveforms : bool
        If True will re-extract waveforms if there are waveforms saved
    save_path : str
        The path to the direcotry where results will be saved
    """
    # Create save_path if it does not exist
    save_path = path_handler(save_path)

    # Get necessary info from param
    raw_data_path = param["raw_data_dir"]
    meta_path = Path(param["ephys_meta_file"])
    n_channels = param["n_channels"]
    n_sync_channels = param["n_sync_channels"]
    n_spikes_to_extract = param["n_raw_spikes_to_extract"]
    detrend_waveforms = param["detrend_waveform"]
    save_multiple_raw = param.get(
        "save_multiple_raw", False
    )  # get and save data for UnitMatch
    waveform_baseline_noise = param.get("waveform_baseline_noise", 20)

    raw_waveform_dir = os.path.join(save_path, "templates._bc_rawWaveforms.npy")

    if os.path.isdir(raw_waveform_dir):
        pass
    else:
        os.mkdir(raw_waveform_dir)

    # if data exists and re_extract_waveforms is false load in data
    if np.logical_and(
        os.stat(raw_waveform_dir).st_size != 0, re_extract_waveforms is False
    ):
        raw_waveforms_full = np.load(
            os.path.join(save_path, "templates._bc_rawWaveforms.npy")
        )
        raw_waveforms_peak_channel = np.load(
            os.path.join(save_path, "templates._bc_rawWaveformPeakChannels.npy")
        )

    # Extract the raw waveforms
    else:
        spike_width = 81  # is in param!
        # half_width is the number of sample before spike_time which are recorded, then will take spike_width - half_width after
        if spike_width == 81:
            # kilosort < 4, baseline 0:41
            half_width = spike_width / 2

        elif spike_width == 61:
            # kilosort = 4, baseline 0:20
            half_width = 20

        unique_clusters = np.unique(spike_templates)
        n_clusters = unique_clusters.shape[0]

        meta_dict = read_meta(meta_path)
        n_elements = (
            int(meta_dict["fileSizeBytes"]) / 2
        )  # int16 so 2byters per data point
        n_channels_rec = int(meta_dict["nSavedChans"])
        param["n_channels_rec"] = n_channels_rec

        raw_data = np.memmap(
            raw_data_path,
            dtype="int16",
            shape=(int(n_elements / n_channels_rec), n_channels_rec),
        )

        # filter so only spikes which have a full have width recordedcan be sampled
        spike_templates_filt = spike_templates[
            np.logical_or(
                half_width < spike_times,
                spike_times < raw_data.shape[0] - spike_width + half_width,
            )
        ]
        spike_times_filt = spike_times[
            np.logical_or(
                half_width < spike_times,
                spike_times < raw_data.shape[0] - spike_width + half_width,
            )
        ]

        # calculate all the spikes to sample
        all_spikes_idxs = np.zeros((n_clusters, n_spikes_to_extract))
        clus_spike_times = []
        # Process ALL unit
        for i, idx in enumerate(unique_clusters):
            clus_spike_times.append(spike_times_filt[spike_templates_filt == idx])
            if n_spikes_to_extract < len(clus_spike_times[i]):
                # -1 so can't index out of region
                all_spikes_idxs[i, :] = clus_spike_times[i][
                    np.linspace(
                        0, len(clus_spike_times[i]) - 1, n_spikes_to_extract, dtype=int
                    )
                ]
            else:
                all_spikes_idxs[i, : len(clus_spike_times[i])] = clus_spike_times[i]
                all_spikes_idxs[i, len(clus_spike_times[i]) :] = np.nan

        if save_multiple_raw & ~os.path.isdir(os.path.join(save_path, "RawWaveforms")):
            os.mkdir(os.path.join(save_path, "RawWaveforms"))

        raw_waveforms_full = np.full(
            (n_clusters, n_channels - n_sync_channels, spike_width), np.nan
        )
        raw_waveforms_peak_channel = np.full((n_clusters), np.nan)
        average_baseline = np.full((n_clusters, waveform_baseline_noise), np.nan)

        all_waveforms = Parallel(n_jobs=-1, verbose=20, mmap_mode="r", max_nbytes=None)(
            delayed(process_a_unit)(
                raw_data,
                spike_width,
                half_width,
                all_spikes_idxs[i],
                n_channels_rec,
                n_channels,
                n_sync_channels,
                cid,
                detrend_waveforms,
                save_multiple_raw,
                waveform_baseline_noise,
                save_path,
            )
            for i, cid in enumerate(unique_clusters)
        )

        (
            raw_waveforms,
            raw_waveforms_full,
            raw_waveforms_peak_channel,
            average_baseline,
        ) = unpack_dicts(
            all_waveforms,
            spike_width,
            n_clusters,
            unique_clusters,
            clus_spike_times,
            n_channels,
            n_sync_channels,
            waveform_baseline_noise,
        )

        # Final processing and saving data
        average_baseline_cat = average_baseline.reshape(-1)
        # NOTE not +1 !
        average_baseline_idx = np.hstack(
            [(np.ones(waveform_baseline_noise) * i) for i in range(n_clusters)]
        )

        # save both arrays
        np.save(
            os.path.join(save_path, "templates._bc_baselineNoiseAmplitude.npy"),
            average_baseline_cat,
        )
        np.save(
            os.path.join(save_path, "templates._bc_baselineNoiseAmplitudeIndex.npy"),
            average_baseline_idx,
        )

    # get SNR
    if os.path.isfile(
        os.path.join(save_path, "templates._bc_baselineNoiseAmplitude.npy")
    ) & os.path.isfile(
        os.path.join(save_path, "templates._bc_baselineNoiseAmplitudeIndex.npy")
    ):
        SNR = np.zeros(n_clusters)
        for cid in range(n_clusters):
            max_value = np.max(
                raw_waveforms_full[cid, raw_waveforms_peak_channel[cid].astype(int), :]
            )  # max value from peak channel
            # Mean average deviation
            noise_est = np.nanmean(
                np.abs(
                    average_baseline_cat[average_baseline_idx == cid]
                    - np.nanmean(average_baseline_cat[average_baseline_idx == cid])
                )
            )
            SNR[cid] = max_value / noise_est

    return raw_waveforms_full, raw_waveforms_peak_channel, SNR


def manage_data_compression(ephys_raw_dir, decompressed_data_local=None):
    """
    Tries to find the raw data in the given directory and handle decompression

    Parameters
    ----------
    ephys_raw_dir : str
        Path to raw data
    decompressed_data_local : str, optional
        File to decompress data to, by default None

    Returns
    -------
    tuple
        The path to the decompressed raw data and the meta file

    Raises
    ------
    Exception
        If the function could not find raw data
    """
    exts = []
    files = os.listdir(ephys_raw_dir)
    bc_decompressed_data = None
    decompressed_data = None
    compressed_data = None

    for file in files:
        ext = os.path.splitext(file)[1]
        exts.append(ext)

        # get paths to data
        if file == "_bc_decompressed*.bin":
            bc_decompressed_data = file
        if ext == ".bin":
            pre_ext = os.path.splitext(os.path.splitext(file)[0])[1]
            if pre_ext != ".lf":
                decompressed_data = file
        if ext == ".cbin":
            compressed_data = os.path.join(ephys_raw_dir, file)

        # get .ch / .meta paths
        if ext == ".ch":
            pre_ext = os.path.splitext(os.path.splitext(file)[0])[1]
            if pre_ext != ".lf":
                compressed_ch = os.path.join(ephys_raw_dir, file)
        if ext == ".meta":
            pre_ext = os.path.splitext(os.path.splitext(file)[0])[1]
            if pre_ext != ".lf":
                meta_path = os.path.join(ephys_raw_dir, file)

    # If bc_decomp data use it
    if bc_decompressed_data is not None:
        print(
            f"Using previously decompressed ephys data file {bc_decompressed_data} as raw data"
        )
        ephys_raw_data = bc_decompressed_data
    elif decompressed_data is not None:
        print(f"Using found decompressed data {decompressed_data}")
        ephys_raw_data = decompressed_data
    elif compressed_data is not None:
        if decompressed_data_local is not None:
            print(
                f"Decompressing ephys data file {compressed_data} to {decompressed_data_local}"
            )
            ephys_raw_data = decompress_data(
                compressed_data, compressed_ch, decompressed_data_local
            )
        else:
            print(
                f"Trying to decompress {compressed_data}, please re-run function with with \n decompressed_data_local = path directory to store decompressed data"
            )
            return

    else:
        raise Exception("Could not find compressed or decompressed data")

    # need full paths
    ephys_raw_data = os.path.join(ephys_raw_dir, ephys_raw_data)
    meta_path = os.path.join(ephys_raw_dir, meta_path)
    return ephys_raw_data, meta_path


def decompress_data(
    compressed_data,
    compressed_ch,
    decompressed_data_local,
    check_after_decompress=False,
):
    decompressed_data_path = os.path.join(
        decompressed_data_local, "_bc_decompressed.bin"
    )

    r = Reader(
        check_after_decompress=check_after_decompress
    )  # Can skip the verification check to save time
    r.open(compressed_data, compressed_ch)
    r.tofile(decompressed_data_path)
    r.close()

    return decompressed_data_path
