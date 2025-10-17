import os
import json
from pathlib import Path
from joblib import Parallel, delayed

import numpy as np

try:
    from mtscomp import Reader
    MTSCOMP_AVAILABLE = True
except ImportError:
    MTSCOMP_AVAILABLE = False
    print("Warning: mtscomp not available. Some raw data formats may not be supported.")
from scipy.signal import detrend
from scipy.ndimage import gaussian_filter
from tqdm.auto import tqdm

def path_handler(path: str) -> None:
    path = Path(path).expanduser()
    assert path.parent.exists(), f"{str(path.parent)} must exist to create {str(path)}."
    path.mkdir(exist_ok=True)
    return path

def read_meta(meta_path):
    """
    Read the meta file attached to a recording.

    Parameters
    ----------
    meta_path : path
        The path to the .meta file (SpikeGLX) or .oebin file (Open Ephys)
    
     Returns
    -------
    meta_dict : dict
        The meta file opened as a dictionary
    """
    meta_dict = {}
    
    if str(meta_path).endswith('.oebin'):
        # Open Ephys Binary format - JSON file
        with meta_path.open() as f:
            oebin_data = json.load(f)
        
        # Extract relevant metadata from Open Ephys format
        # We need to find the continuous data recording info
        if 'continuous' in oebin_data and len(oebin_data['continuous']) > 0:
            # Usually the first continuous recording is the neural data
            continuous_data = oebin_data['continuous'][0]
            
            # Get the number of channels from the continuous data
            n_channels = continuous_data.get('num_channels', 384)
            
            # Get sample rate
            sample_rate = continuous_data.get('sample_rate', 30000.0)
            
            # Get bit volts for scaling (microvolts per bit)
            # Default to 0.195 for Open Ephys if not specified
            bit_volts = 0.195
            if 'channels' in continuous_data and len(continuous_data['channels']) > 0:
                # Get bit_volts from first channel if available
                bit_volts = continuous_data['channels'][0].get('bit_volts', 0.195)
            
            # Calculate file size from the binary file
            # The binary file is usually in the same directory with name continuous.dat
            binary_file = meta_path.parent / continuous_data.get('folder_name', 'continuous') / 'continuous.dat'
            if binary_file.exists():
                file_size_bytes = os.path.getsize(binary_file)
            else:
                # Try to find any .dat file in the parent directory
                dat_files = list(meta_path.parent.glob('**/*.dat'))
                if dat_files:
                    file_size_bytes = os.path.getsize(dat_files[0])
                else:
                    # Default to a large value if we can't find the file
                    file_size_bytes = 0
            
            # Create meta_dict in SpikeGLX format for compatibility
            meta_dict['fileTimeSecs'] = file_size_bytes / (2 * n_channels * sample_rate) if file_size_bytes > 0 else 0
            meta_dict['fileSizeBytes'] = str(file_size_bytes)
            meta_dict['nSavedChans'] = str(n_channels)  # In Open Ephys, this is total channels
            meta_dict['sRateHz'] = str(sample_rate)
            meta_dict['imAiRangeMax'] = '1'  # Not used but kept for compatibility
            meta_dict['imAiRangeMin'] = '-1'  # Not used but kept for compatibility
            
            # For Open Ephys, we use the bit_volts directly
            # The conversion factor is bit_volts * 1e6 to get from volts to microvolts
            # But since bit_volts is already in microvolts, we use it directly
            meta_dict['imMaxInt'] = '32768'  # 2^15 for int16
            meta_dict['bitVolts'] = str(bit_volts)  # Store bit_volts for later use
            
        else:
            raise ValueError(f"Could not find continuous data in {meta_path}")
            
    else:
        # SpikeGLX format - key=value pairs
        with meta_path.open() as f:
            mdat_list = f.read().splitlines()
            # convert the list entries into key value pairs
            for m in mdat_list:
                cs_list = m.split(sep="=")
                if len(cs_list) >= 2:  # Ensure we have at least key=value
                    if cs_list[0][0] == "~":
                        curr_key = cs_list[0][1 : len(cs_list[0])]
                    else:
                        curr_key = cs_list[0]
                    # Handle cases where value might contain '=' 
                    meta_dict.update({curr_key: '='.join(cs_list[1:])})

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
    detrendWaveform,
    detrendForUnitMatch,
    waveform_baseline_noise,
    save_directory,
    save_multiple_raw,
    template_peak_channel=None
):
    """
    Reads in data from a unit, and processes the data.

    Parameters
    ----------
    raw_data : memmap
        The numpy memmap of the raw data
    spike_width : int
        The total number of samples to take per unit
    half_width : int
        The number of samples before the spike starts
    all_spikes_idxs : ndarray (n_spike_to_extract)
        All of the spikes indexes of spike to extract for the unit
    n_channels_rec : int
        The total number of channels in the recording
    n_channels : int
        The number of good recording channels (e.g non-sync channels)
    n_sync_channels : int
        The number of sync channel in the recording
    cid : int
        The id of the cluster
    detrendWaveform : bool
        If True will linearly de-trend the average waveforms for BombCell
    detrendForUnitMatch : bool
        If True will linearly de-trend raw waveforms saved for UnitMatch
    waveform_baseline_noise : int
        The number of samples before the waveform which are noise
    save_directory : pathlib.Path
        The path to the directory to save the UnitMatch data
    template_peak_channel : int, optional
        The peak channel from the template. If provided, this will be used instead of calculating from raw waveforms

    Returns
    -------
    cluster_raw_waveforms : ndarray
        A dictionary of the necessary information extract from the raw data for a unit
    """

    cluster_raw_waveforms = {}
    spike_idx = all_spikes_idxs[~np.isnan(all_spikes_idxs)]
    n_spikes_sampled = spike_idx.size

    # create empty array for the spike map
    # Extract only neural channels (nChannels - nSyncChannels)
    spike_map = np.full(
        (n_channels - n_sync_channels, spike_width, n_spikes_sampled), np.nan
    )
    for i, sid in enumerate(spike_idx[~np.isnan(spike_idx)]):
        # get the raw data for each spike chosen
        tmp = raw_data[
            int(sid - half_width - 1) : int(sid + spike_width - half_width - 1),
            np.arange(n_channels_rec),
        ]
        if tmp.shape[0] != spike_map.shape[1]:
            # if hit, spike overlaps with the beginning or the end of the binary file.
            continue
        # -1, to better fit with ML, adn python indexing!
        tmp.astype(np.float64)

        # option to remove a linear in time trends
        if detrendWaveform:
            if n_sync_channels > 0:
                detrended = detrend(tmp[:, :-n_sync_channels], axis=0).swapaxes(0, 1)
            else:
                detrended = detrend(tmp[:, :], axis=0).swapaxes(0, 1)
            
            spike_map[:, :, i] = detrended
        else:
            spike_map[:, :, i] = tmp[:, :-n_sync_channels].swapaxes(0, 1)

    # Save average waveforms for unitmatch
    # create the waveforms now, save later
    if save_multiple_raw:
        # Create separate spike map for UnitMatch with its own detrending control
        unitmatch_spike_map = np.full(
            (n_channels - n_sync_channels, spike_width, n_spikes_sampled), np.nan
        )
        
        for i, sid in enumerate(spike_idx[~np.isnan(spike_idx)]):
            # get the raw data for each spike chosen
            tmp = raw_data[
                int(sid - half_width - 1) : int(sid + spike_width - half_width - 1),
                np.arange(n_channels_rec),
            ]
            if tmp.shape[0] != spike_width:
                # if hit, spike overlaps with the beginning or the end of the binary file.
                continue
            tmp.astype(np.float64)

            # option to remove a linear in time trends for UnitMatch (separate from BombCell)
            if detrendForUnitMatch:
                detrended = detrend(tmp[:, :-n_sync_channels], axis=0).swapaxes(
                    0, 1
                )
                unitmatch_spike_map[:, :, i] = detrended
            else:
                unitmatch_spike_map[:, :, i] = tmp[:, :-n_sync_channels].swapaxes(0, 1)
        
        tmp_spike_map = unitmatch_spike_map.swapaxes(0, 1)  # align with UnitMatch

        # smooth over axis at once
        tmp_spike_map = gaussian_filter(tmp_spike_map, axes=0, sigma=1, radius=2)
        # matches matlab smoothdata, EXCEPT at boundaries!
        tmp_spike_map -= np.mean(tmp_spike_map[:waveform_baseline_noise, :, :], axis=0)[
            np.newaxis, :, :
        ]

        # split into 2 CV for unitmatch!
        UM_CV_limit = np.floor(n_spikes_sampled / 2).astype(int)
        avg_waveforms = np.full((spike_width, n_channels - n_sync_channels, 2), np.nan)
        avg_waveforms[:, :, 0] = np.nanmedian(tmp_spike_map[:, :, :UM_CV_limit], axis=-1)
        avg_waveforms[:, :, 1] = np.nanmedian(tmp_spike_map[:, :, UM_CV_limit:], axis=-1)

        np.save(
            save_directory / f"Unit{cid}_RawSpikes.npy",
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

    # use template peak channel if provided, otherwise calculate from raw waveforms
    if template_peak_channel is not None:
        raw_waveforms_peak_channel = template_peak_channel
    else:
        # find peak channel from raw waveforms (old behavior)
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
        The nymber of samples in an extracted spike.
    n_clusters : int
        The number of unique clusters extracted
    unique_clusters : ndarray (n_clusters)
        The indexes of the clusters.
    clus_spike_times : ndarray (n_clusters, n_spikes_to_extract)
        All of the indexes of all of the spikes which were extracted.
    n_channels : int
        The number of good recording channels (e.g non-sync channels)
    n_sync_channels : int
        The number of sync channel in the recording
    waveform_baseline_noise : int
        The number of samples before the waveform which are noise

    Returns
    -------
    raw_waveforms : dictionary
        A nested dicitonary which contains a dictionary of waveform properties for each cluster.
    raw_waveforms_full : ndarray (n_clusters, n_channles, spike_width)
        All extracted average waveforms
    raw_waveforms_peak_channels : ndarray (n_clusters)
        The peak channel for each cluster
    average_baseline : ndarray (n_clusters, waveforms_baseline_noise)
        The average value for each clusters of the time samples before the signal

    """
    raw_waveforms = {}
    raw_waveforms_full = np.full(
        (n_clusters, n_channels - n_sync_channels, spike_width), np.nan
    )
    raw_waveforms_peak_channel = np.full((n_clusters), np.nan)
    average_baseline = np.full((n_clusters, waveform_baseline_noise), np.nan)


    for i in range(unique_clusters.shape[0]):
        raw_waveforms[f"cluster_{i}"] = {}
        raw_waveforms[f"cluster_{i}"]["spike_map_mean"] = all_waveforms[i][
            "spike_map_mean"
        ]
        raw_waveforms[f"cluster_{i}"]["spike_idx_sampled"] = all_waveforms[i][
            "spike_idxs"
        ]
        raw_waveforms[f"cluster_{i}"]["Cluster_idx"] = i
        raw_waveforms[f"cluster_{i}"]["spike_idxs"] = clus_spike_times[i]

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
    param, spike_clusters, spike_times, re_extract_waveforms, save_path, template_peak_channels=None
):
    """
    Extracts average raw waveforms from the raw data, can be used to get raw waveforms for UnitMatch as well

    Parameters
    ----------
    param : dict
        The param dictionary used in BombCell
    spike_clusters : ndarray
        The array which assigns a spike to a cluster
    spike_times : ndarray
        The array which gives the sample number for each spike
    re_extract_waveforms : bool
        If True will re-extract waveforms if there are waveforms saved
    save_path : str
        The path to the directory where results will be saved
    template_peak_channels : ndarray, optional
        Array of peak channels from templates. If provided, raw waveforms will use these instead of calculating their own

    Returns
    -------
    raw_waveforms_full : ndarray (n_clusters, n_channles, spike_width)
        All extracted average waveforms
    raw_waveforms_peak_channels : ndarray (n_clusters)
        The peak channel for each cluster
    SNR : ndarray (n_clusters)
        The signal to noise ratio for each unit
    """
    # Create save_path if it does not exist
    save_path = path_handler(save_path)

    raw_waveforms_dir = save_path / "RawWaveforms"
    raw_waveforms_dir.mkdir(exist_ok = True)

    raw_waveforms_file = save_path / "templates._bc_rawWaveforms.npy"
    raw_waveforms_peak_channel_file = save_path / "templates._bc_rawWaveformPeakChannels.npy"
    snr_noise_file = save_path / "templates._bc_baselineNoiseAmplitude.npy"
    snr_noise_idx_file = save_path / "templates._bc_baselineNoiseAmplitudeIndex.npy"
    raw_waveforms_id_match_file = save_path / "_bc_rawWaveforms_kilosort_format.npy"

    # Cluster ids
    unique_clusters = np.unique(spike_clusters)
    n_clusters = unique_clusters.shape[0]
    max_cluster_id = unique_clusters[-1]

    # Get necessary info from param
    raw_data_file = param["raw_data_file"]
    meta_path = Path(param["ephys_meta_file"]) if param["ephys_meta_file"] is not None else None
    n_channels = param["nChannels"]
    n_sync_channels = param["nSyncChannels"]
    n_spikes_to_extract = param["nRawSpikesToExtract"]
    detrendWaveform = param["detrendWaveform"]  # For BombCell average waveforms
    detrendForUnitMatch = param.get("detrendForUnitMatch", False)  # For UnitMatch raw waveforms
    save_multiple_raw = param.get("saveMultipleRaw", False)  # get and save data for UnitMatch
    waveform_baseline_noise = param.get("waveformBaselineNoiseWindow", 20)
    spike_width = param["spike_width"]
    

    # if data exists and re_extract_waveforms is false, load in data
    recompute = re_extract_waveforms
    if raw_waveforms_file.exists() and not recompute:
        if raw_waveforms_peak_channel_file.exists() and snr_noise_file.exists() and snr_noise_idx_file.exists():
            print(f"Loading file {raw_waveforms_file}...", end='', flush=True)

            raw_waveforms_full = np.load(raw_waveforms_file)
            raw_waveforms_peak_channel = np.load(raw_waveforms_peak_channel_file)
            raw_waveforms_id_match = np.load(raw_waveforms_id_match_file)
            baseline_noise_all = np.load(snr_noise_file)
            baseline_noise_idx = np.load(snr_noise_idx_file)
            print(f"\rLoading file {raw_waveforms_file}... Done!") 

            check = check_extracted_waveforms(
                raw_waveforms_id_match, raw_waveforms_peak_channel, spike_clusters, spike_times, baseline_noise_all, param, save_path)

            if check != (None, None, None, None, None):
                raw_waveforms_id_match, raw_waveforms_peak_channel, raw_waveforms_full, baseline_noise_all, baseline_noise_idx = check
            # Check whether number of clusters changed
            # assumes that raw_waveforms_full has empty rows for jumps in unit indices
            if raw_waveforms_full.shape[0] != n_clusters:
                print("\rSome units' raw waveforms are not extracted. Extracting now ...") 
                recompute = True
        else:
            recompute = True
    else:
        recompute = True

    # Extract the raw waveforms
    if recompute:
        # half_width is the number of sample before spike_time which are recorded,
        # then will take spike_width - half_width after
        if spike_width == 82: # kilosort < 4, baseline 0:41
            half_width = 41  # Fixed to be an integer
        elif spike_width == 61: # kilosort = 4, baseline 0:20
            half_width = 20
        else:
            # For custom spike widths, use floor division to get the midpoint
            # This ensures we always get an integer value
            half_width = spike_width // 2

        if meta_path is not None and meta_path.exists():
            meta_dict = read_meta(meta_path)
            n_elements = (int(meta_dict["fileSizeBytes"]) / 2)  # int16 so 2 bytes per data point
            n_channels_rec = int(meta_dict["nSavedChans"])  # Total channels including sync
            param["n_channels_rec"] = n_channels_rec
            
            # For Open Ephys files, adjust the raw data file path if needed
            if str(meta_path).endswith('.oebin'):
                # In Open Ephys, nSavedChans is already the total channel count
                # Make sure we're using the right raw data file path
                if not Path(raw_data_file).exists():
                    # Try to find the continuous.dat file
                    possible_dat = meta_path.parent / 'continuous' / 'continuous.dat'
                    if possible_dat.exists():
                        raw_data_file = str(possible_dat)
                        param["raw_data_file"] = raw_data_file
                        print(f"Using Open Ephys raw data file: {raw_data_file}")
                    else:
                        # Look for any .dat file
                        dat_files = list(meta_path.parent.glob('**/*.dat'))
                        if dat_files:
                            raw_data_file = str(dat_files[0])
                            param["raw_data_file"] = raw_data_file
                            print(f"Using Open Ephys raw data file: {raw_data_file}")
        else:
            # Use default values when no metafile is available
            print("Warning: No meta file found. Using default parameters...")
            # Get file size directly from the raw data file
            import os
            file_size_bytes = os.path.getsize(raw_data_file)
            n_elements = file_size_bytes / 2  # int16 so 2 bytes per data point
            
            # When no metafile, nChannels already includes sync channels
            n_channels_rec = n_channels  # Should be 385 (384 neural + 1 sync)
            print(f"Using {n_channels_rec} total channels in recording")
            param["n_channels_rec"] = n_channels_rec

        raw_data = np.memmap(
            raw_data_file,
            dtype="int16",
            shape=(int(n_elements / n_channels_rec), n_channels_rec),
        )

        # filter so only spikes which have a full width recorded can be sampled
        spike_clusters_filt = spike_clusters[
            np.logical_and(
                half_width < spike_times,
                spike_times < raw_data.shape[0] - spike_width + half_width,
            )
        ]
        spike_times_filt = spike_times[
            np.logical_and(
                half_width < spike_times,
                spike_times < raw_data.shape[0] - spike_width + half_width,
            )
        ]

        # calculate all the spikes to sample
        all_spikes_idxs = np.zeros((n_clusters, n_spikes_to_extract))
        clus_spike_times = []
        # Process ALL unit
        for i, idx in enumerate(unique_clusters):
            clus_spike_times.append(spike_times_filt[spike_clusters_filt == idx])
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

        all_waveforms = Parallel(n_jobs=-1, verbose=10, mmap_mode="r", max_nbytes=None)(
            delayed(process_a_unit)(
                raw_data,
                spike_width,
                half_width,
                all_spikes_idxs[i],
                n_channels_rec,
                n_channels,
                n_sync_channels,
                cid,
                detrendWaveform,
                detrendForUnitMatch,
                waveform_baseline_noise,
                raw_waveforms_dir,
                save_multiple_raw,
                template_peak_channels[i] if template_peak_channels is not None and i < len(template_peak_channels) else None,
            )
            for i, cid in tqdm(enumerate(unique_clusters))
        )

        (raw_waveforms,
         raw_waveforms_full,
         raw_waveforms_peak_channel,
         baseline_noise_all) = unpack_dicts(
                            all_waveforms,
                            spike_width,
                            n_clusters,
                            unique_clusters,
                            clus_spike_times,
                            n_channels,
                            n_sync_channels,
                            waveform_baseline_noise)
        # Final processing and saving data
        # NOTE not +1 !
        baseline_noise_all = baseline_noise_all.reshape(-1)
        baseline_noise_idx = np.hstack([(np.ones(waveform_baseline_noise) * cid) for cid in unique_clusters])
        # save baseline noise arrays
        np.save(snr_noise_file, baseline_noise_all)
        np.save(snr_noise_idx_file, baseline_noise_idx)

    # Compute SNR
    SNR = np.zeros(n_clusters)
    for i, cid in enumerate(unique_clusters):
        mask = unique_clusters == i
        cluster_idx = np.where(mask)[0]
        peak_channel = raw_waveforms_peak_channel[cluster_idx].astype(int)

        # Maximum absolute value of the waveform (signal)
        peak_waveform = raw_waveforms_full[cluster_idx, peak_channel, :]
        signal = np.max(np.abs(np.squeeze(peak_waveform)))

        # Get baseline noise for this cluster
        baseline_mask = baseline_noise_idx  == i
        baseline = baseline_noise_all[baseline_mask]

        # Calculate MAD (noise) - Median Absolute Deviation
        noise = np.median(np.abs(baseline - np.median(baseline)))

        # Calculate SNR
        SNR[i] = signal / noise


    #Save a copy of the waveforms were the row number matches the cluster index
    raw_waveforms_id_match = np.full((max_cluster_id + 1, n_channels - n_sync_channels, spike_width), np.nan)
    for i, idx in enumerate(unique_clusters):
        raw_waveforms_id_match[idx] = raw_waveforms_full[i]

    # Save the extracted waveforms if they were recomputed
    if recompute:
        np.save(raw_waveforms_file, raw_waveforms_full)
        np.save(raw_waveforms_peak_channel_file, raw_waveforms_peak_channel)
        np.save(raw_waveforms_id_match_file, raw_waveforms_id_match)

    return raw_waveforms_full, raw_waveforms_peak_channel, SNR, raw_waveforms_id_match


def decompress_data_if_needed(raw_file_path, decompress_data=True):
    """
    Check if raw data needs decompression and decompress if necessary.
    
    Parameters
    ----------
    raw_file_path : str or Path
        Path to the raw data file (can be .bin or .cbin)
    decompress_data : bool, optional
        Whether to decompress compressed data, by default True
        
    Returns
    -------
    str
        Path to the usable raw data file (decompressed if needed)
        
    Raises
    ------
    Exception
        If compressed data is found but decompress_data is False
    """
    raw_file_path = Path(raw_file_path)
    ephys_raw_dir = raw_file_path.parent
    
    # If the file is already .bin and exists, just return it
    if raw_file_path.suffix == '.bin' and raw_file_path.exists():
        return str(raw_file_path)
    
    # Look for data files in the directory
    files = os.listdir(ephys_raw_dir)
    bc_decompressed_data = None
    decompressed_data = None
    compressed_data = None
    compressed_ch = None
    
    # Find compressed or decompressed binary files
    for file in files:
        ext = os.path.splitext(file)[1]
        
        if "_bc_decompressed" in file:
            bc_decompressed_data = file
        
        elif ext == ".bin":
            pre_ext = os.path.splitext(os.path.splitext(file)[0])[1]
            if pre_ext != ".lf":  # Exclude LFP data
                decompressed_data = file
        
        elif ext == ".cbin":
            compressed_data = file
        
        elif ext == ".ch":
            pre_ext = os.path.splitext(os.path.splitext(file)[0])[1]
            if pre_ext != ".lf":  # Exclude LFP data
                compressed_ch = file
    
    # Check for previously decompressed data with bombcell suffix
    if bc_decompressed_data is not None:
        print(f"Using previously decompressed ephys data file {bc_decompressed_data}")
        return str(ephys_raw_dir / bc_decompressed_data)
    
    # Check for regular decompressed data
    elif decompressed_data is not None:
        print(f"Using raw data {decompressed_data}")
        return str(ephys_raw_dir / decompressed_data)
    
    # Handle compressed data
    elif compressed_data is not None:
        if compressed_ch is None:
            raise Exception(f"Found compressed data file {compressed_data} but no matching .ch file!")
        
        if decompress_data:
            print(f"Decompressing ephys data file {compressed_data}...")
            decompressed_data_name = compressed_data.replace(".cbin", ".bin").split('.')
            decompressed_data_name[0] = decompressed_data_name[0] + '_bc_decompressed'
            decompressed_data_name = ".".join(decompressed_data_name)
            
            decompressed_path = _decompress_raw_data(
                ephys_raw_dir, compressed_data, compressed_ch, decompressed_data_name
            )
            return str(decompressed_path)
        else:
            raise Exception(f"Found compressed data at {ephys_raw_dir} but decompress_data=False. "
                          f"Set decompress_data=True to decompress {compressed_data}")
    else:
        raise Exception(f"Could not find any suitable ephys data (.bin, .cbin) at {ephys_raw_dir}!")


def manage_data_compression(ephys_raw_dir, param):
    """
    Tries to find the raw data in the given directory and handle decompression if necessary.

    Parameters
    ----------
    ephys_raw_dir : str
        Path to raw data
    decompressed_data_local : str, optional
        File to decompress data to, by default None

    Returns
    -------
    ephys_raw_data : str
        The path to the raw data file
    meta_path : str
        The path to the meta data file

    Raises
    ------
    Exception
        If the function could not find raw data
    """
    exts = []
    files = os.listdir(ephys_raw_dir)
    ephys_raw_dir = Path(ephys_raw_dir)
    bc_decompressed_data = None
    decompressed_data = None
    compressed_data = None
    
    # Find compressed or decompressed binary files at ephys_raw_dir
    for file in files:
        ext = os.path.splitext(file)[1]
        exts.append(ext)

        if "_bc_decompressed" in file:
            bc_decompressed_data = file

        if ext == ".bin":
            pre_ext = os.path.splitext(os.path.splitext(file)[0])[1]
            if pre_ext != ".lf":
                decompressed_data = file

        if ext == ".cbin":
            compressed_data = file

        if ext == ".ch":
            pre_ext = os.path.splitext(os.path.splitext(file)[0])[1]
            if pre_ext != ".lf":
                compressed_ch = file
        
        if ext == ".dat":
            decompressed_data = file

    # Assign raw data path after eventual decompression
    ephys_raw_data = None
    if bc_decompressed_data is not None:
        print(f"Using previously decompressed ephys data file {bc_decompressed_data} as raw data.")
        ephys_raw_data = ephys_raw_dir / bc_decompressed_data
    elif decompressed_data is not None:
        print(f"Using raw data {decompressed_data}.")
        ephys_raw_data = ephys_raw_dir / decompressed_data
    elif compressed_data is not None:
        assert compressed_ch is not None, f"Found compressed data file {compressed_data} but no matching .ch file!"
        if param['decompress_data']:
            print(f"Decompressing ephys data file {compressed_data}...")
            decompressed_data_name = compressed_data.replace(".cbin", ".bin").split('.')
            decompressed_data_name[0] = decompressed_data_name[0] + '_bc_decompressed'
            decompressed_data_name = ".".join(decompressed_data_name)
            ephys_raw_data = _decompress_raw_data(
                ephys_raw_dir, compressed_data, compressed_ch, decompressed_data_name
            )
        else:
            raise Exception(f"Could not find decompressed data at {ephys_raw_dir}! Set param['decompress_data'] to true to decompress data at {ephys_raw_dir}.")
    else:
        raise Exception(f"Could not find compressed or decompressed data at {ephys_raw_dir}!")

    # need full paths
    assert ephys_raw_data is not None, f"Raw data (.ap.bin, .dat, or .cbin file) not found at {ephys_raw_dir}!"

    return ephys_raw_data


def _decompress_raw_data(
    source_directory,
    compressed_data,
    compressed_ch,
    decompressed_data_name,
    check_after_decompress=False,
):
    """
    Decompresses compressed raw recordings to disk for further processing

    Parameters
    ----------
    source_directory: str
        Directory holding the compressed_data .cbin and compressed_ch .ch files
    compressed_data : str
        Name of .cbin compressed data file
    compressed_ch : str
        Name of .ch file
    decompressed_data_name : str
        Target name for decompressed .bin data
    check_after_decompress : bool, optional
        If True will go through the mtscomp extra checks , by default False

    Returns
    -------
    decompressed_data_path : str
        The path to the decompressed data
    """
    source_directory = Path(source_directory)
    compressed_data = str(source_directory / compressed_data)
    compressed_ch = str(source_directory / compressed_ch)
    decompressed_data_name = str(source_directory / decompressed_data_name)

    r = Reader(
        check_after_decompress=check_after_decompress
    )  # Can skip the verification check to save time
    r.open(compressed_data, compressed_ch)
    r.tofile(decompressed_data_name)
    r.close()

    return decompressed_data_name

def check_extracted_waveforms(raw_waveforms_id_match, raw_waveforms_peak_channel, spike_clusters, spike_times, baseline_noise_all, param, save_path):

    # get the current and old cluster indexes
    unique_id_new = np.unique(spike_clusters)
    unique_id_extracted = np.argwhere(np.isnan(raw_waveforms_id_match[:,0,0]) == False).squeeze()

    if np.all(unique_id_new == unique_id_extracted):
        print('No splits/merges detected')
        return None, None, None, None, None
    else:
        #find the different indexes
        new_indexes_to_get = unique_id_new[np.argwhere(np.isin(unique_id_new, unique_id_extracted) == False)]
        old_indexes_to_remove = unique_id_extracted[np.argwhere(np.isin(unique_id_extracted, unique_id_extracted) == False)]

        #create new waveforms array for the new indexes and remove old indexes
        n_new_clusters = unique_id_new[-1] - unique_id_extracted[-1]
        new_raw_waveforms_matching_ids = np.pad(raw_waveforms_id_match, ((0,n_new_clusters),(0,0),(0,0)))
        new_raw_waveforms_matching_ids[old_indexes_to_remove] = np.nan

        #create baseline 
        baseline_id_match = np.zeros(np.max(unique_id_extracted)+1)
        for i, idx in enumerate(unique_id_extracted):
            baseline_id_match[idx] = baseline_noise_all[i*waveform_baseline_noise:(i+1)*waveform_baseline_noise].mean()
        new_baseline = np.pad(baseline_id_match, (0,n_new_clusters))
        new_baseline[old_indexes_to_remove] = np.nan

        new_baseline_noise_idx = np.hstack([(np.ones(waveform_baseline_noise) * cid) for cid in unique_id_new])

        #create array for peak channels with matching row to id 
        peak_channels_id_match = np.zeros(np.max(unique_id_extracted)+1)
        for i, idx in enumerate(unique_id_extracted):
            peak_channels_id_match[idx] = raw_waveforms_peak_channel[i]

        new_peak_channels_matching_ids = np.pad(peak_channels_id_match, (0,n_new_clusters))
        new_peak_channels_matching_ids[old_indexes_to_remove] = np.nan

        print(f'Extracting unit index(s) {new_indexes_to_get}.. from detected splits')

        ##NOTE code here is repeated from extracting all units
        n_clusters = unique_id_new.size
        # Get necessary info from param
        raw_data_file = param["raw_data_file"]
        meta_path = Path(param["ephys_meta_file"]) if param["ephys_meta_file"] is not None else None
        n_channels = param["nChannels"]
        n_sync_channels = param["nSyncChannels"]
        n_spikes_to_extract = param["nRawSpikesToExtract"]
        detrendWaveform = param["detrendWaveform"]
        waveform_baseline_noise = param.get("waveformBaselineNoiseWindow", 20)
        spike_width = param["spike_width"]
        save_multiple_raw = param.get("saveMultipleRaw", False)  # get and save data for UnitMatch
    

        # half_width is the number of sample before spike_time which are recorded,
        # then will take spike_width - half_width after
        if spike_width == 82: # kilosort < 4, baseline 0:41 (fixed from 81 to match main function)
            half_width = 41  # Fixed to be an integer
        elif spike_width == 61: # kilosort = 4, baseline 0:20
            half_width = 20
        else:
            # For custom spike widths, use floor division to get the midpoint
            # This ensures we always get an integer value
            half_width = spike_width // 2

        if meta_path is not None and meta_path.exists():
            meta_dict = read_meta(meta_path)
            n_elements = (int(meta_dict["fileSizeBytes"]) / 2)  # int16 so 2 bytes per data point
            n_channels_rec = int(meta_dict["nSavedChans"])  # Total channels including sync
            param["n_channels_rec"] = n_channels_rec
            
            # For Open Ephys files, adjust the raw data file path if needed
            if str(meta_path).endswith('.oebin'):
                # In Open Ephys, nSavedChans is already the total channel count
                # Make sure we're using the right raw data file path
                if not Path(raw_data_file).exists():
                    # Try to find the continuous.dat file
                    possible_dat = meta_path.parent / 'continuous' / 'continuous.dat'
                    if possible_dat.exists():
                        raw_data_file = str(possible_dat)
                        param["raw_data_file"] = raw_data_file
                        print(f"Using Open Ephys raw data file: {raw_data_file}")
                    else:
                        # Look for any .dat file
                        dat_files = list(meta_path.parent.glob('**/*.dat'))
                        if dat_files:
                            raw_data_file = str(dat_files[0])
                            param["raw_data_file"] = raw_data_file
                            print(f"Using Open Ephys raw data file: {raw_data_file}")
        else:
            # Use default values when no metafile is available
            print("Warning: No meta file found. Using inputed parameters...")
            # Get file size directly from the raw data file
            import os
            file_size_bytes = os.path.getsize(raw_data_file)
            n_elements = file_size_bytes / 2  # int16 so 2 bytes per data point
            
            # When no metafile, nChannels already includes sync channels
            n_channels_rec = n_channels  # Should be 385 (384 neural + 1 sync)
            print(f"Using {n_channels_rec} total channels in recording")
            param["n_channels_rec"] = n_channels_rec

        raw_data = np.memmap(
            raw_data_file,
            dtype="int16",
            shape=(int(n_elements / n_channels_rec), n_channels_rec),
        )

        # filter so only spikes which have a full width recorded can be sampled
        spike_clusters_filt = spike_clusters[
            np.logical_and(
                half_width < spike_times,
                spike_times < raw_data.shape[0] - spike_width + half_width,
            )
        ]
        spike_times_filt = spike_times[
            np.logical_and(
                half_width < spike_times,
                spike_times < raw_data.shape[0] - spike_width + half_width,
            )
        ]

        # calculate all the spikes to sample
        all_spikes_idxs = np.zeros((n_clusters, n_spikes_to_extract))
        clus_spike_times = []
        # Process ALL unit
        bar_description = "Extracting raw waveforms: {percentage:3.0f}%|{bar:10}| {n}/{total} units"
        for i, idx in tqdm(enumerate(unique_id_new), bar_format=bar_description):
            clus_spike_times.append(spike_times_filt[spike_clusters_filt == idx])
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

        # Create the raw waveforms directory for saving multiple raw files
        raw_waveforms_dir = Path(save_path) / "RawWaveforms"
        raw_waveforms_dir.mkdir(exist_ok=True)
        
        for id in new_indexes_to_get:
            # Get the correct index in all_spikes_idxs array
            unit_idx = np.where(unique_id_new == id)[0][0]
            # Get template peak channel for this unit
            template_peak_ch = template_peak_channels[unit_idx] if template_peak_channels is not None and unit_idx < len(template_peak_channels) else None
            
            tmp_raw_waveform_info = process_a_unit(
                raw_data,
                spike_width,
                half_width,
                all_spikes_idxs[unit_idx],
                n_channels_rec,
                n_channels,
                n_sync_channels,
                id,
                detrendWaveform,
                detrendForUnitMatch,
                waveform_baseline_noise,
                raw_waveforms_dir,
                save_multiple_raw,
                template_peak_ch,
            )
            new_raw_waveforms_matching_ids[id] = tmp_raw_waveform_info['raw_waveforms_full']
            new_peak_channels_matching_ids[id] = tmp_raw_waveform_info['raw_waveforms_peak_channel']

        #remove all empty rows
        new_raw_waveform_full = new_raw_waveforms_matching_ids[~np.isnan(new_raw_waveforms_matching_ids).all(axis=(1,2)),:,:]
    return new_raw_waveforms_matching_ids, new_peak_channels_matching_ids, new_raw_waveform_full, new_baseline, new_baseline_noise_idx