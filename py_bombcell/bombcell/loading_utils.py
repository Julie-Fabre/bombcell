import os
from pathlib import Path

import numpy as np
import pandas as pd

import bombcell.extract_raw_waveforms as erw


def load_ephys_data(ephys_path):
    """
    This function loads the necessary data from the spike sorting to run BombCell

    Parameters
    ----------
    ephys_path : str
        The path to the KiloSorted output file

    Returns
    -------
    spike_times_samples : ndarray (n_spikes,)
        The array which gives each spike time in samples (*not* seconds)
    spike_clusters : ndarray (n_spikes,)
        The array which assigns a spike to a cluster
    template_waveforms : ndarray (m_templates, n_time_points, n_channels)
        The array of template waveforms for each templates and channel
    pc_features : ndarray (n_spikes, n_features_per_channel, n_pc_features)
        The array giving the PC values for each spike
    pc_feature_idx : ndarray (n_templates, n_pc_features)
        The array which specifies which channel contribute to each entry in dim 3 of the pc_features array
    channel_positions : ndarray (n_channels, 2)
        The array which gives the x and y coordinates of each channel
    good_channels: ndarray (n_channels,)
        The array defining the channels used by KiloSort, as some in-active channels are dropped during
        spike sorting
    """
    ephys_path = Path(ephys_path)

    # Note: removed +1 matlab indexing
    # Load spike templates, times, amplitudes
    spike_templates = np.load(ephys_path / "spike_templates.npy").squeeze()

    if (ephys_path / "spike_times_corrected.npy").exists():
        spike_times_samples = np.load(ephys_path / "spike_times_corrected.npy").squeeze()
    else:
        spike_times_samples = np.load(ephys_path / "spike_times.npy").squeeze()

    template_amplitudes = np.load(ephys_path / "amplitudes.npy").squeeze().astype(np.float64)

    # load and unwhiten templates
    templates_waveforms_whitened = np.load(ephys_path / "templates.npy")
    winv = np.load(ephys_path / "whitening_mat_inv.npy")
    templates_waveforms = np.zeros_like(templates_waveforms_whitened)
    for t in range(templates_waveforms.shape[0]):
        templates_waveforms[t, :, :] = templates_waveforms_whitened[t, :, :].squeeze() @ winv

    # Load pc features
    if (ephys_path / "pc_features.npy").exists():
        pc_features = np.load(ephys_path / "pc_features.npy").squeeze()
        pc_features_idx = np.load(ephys_path / "pc_feature_ind.npy").squeeze()
    else:
        pc_features = np.nan
        pc_features_idx = np.nan

    channel_positions = np.load(ephys_path / "channel_positions.npy").squeeze()

    # Handle Phy manual curation
    spike_templates, templates_waveforms, pc_features_idx = handle_manual_curation(
        ephys_path, spike_templates, templates_waveforms, pc_features_idx,
    )

    return (
        spike_times_samples,
        spike_templates,
        templates_waveforms,
        template_amplitudes,
        pc_features,
        pc_features_idx,
        channel_positions,
    )


def handle_manual_curation(ephys_path, spike_templates, templates_waveforms, pc_features_idx):
    # if manually curated data, template ids and cluster ids have diverged.
    # this function appends additional template waveforms to templates_waveforms,
    # and the units that do not exist anymore because they were merged remain as dead rows
    found_pc_features = not np.all(np.isnan(pc_features_idx))

    if (ephys_path / 'spike_clusters.npy').exists():
        spike_clusters = np.load(ephys_path / 'spike_clusters.npy').squeeze().astype(int)
        new_templates = np.unique(spike_clusters[~np.isin(spike_clusters, spike_templates)])
        n_new_units = len(new_templates)
        
        if n_new_units > 0:
            # initialize templates and pc features
            # TODO currently, if unit id jumps from 300 to 600,
            # there will be 300 empty rows in padded templates_waveforms.
            # this is inefficient and should be changed in the future.
            assert templates_waveforms.shape[0] == pc_features_idx.shape[0]

            new_units_max_index = max(new_templates)
            n_old_units = templates_waveforms.shape[0]
            n_new_rows = int(new_units_max_index - n_old_units + 1)

            padding = np.zeros((n_new_rows, 
                            templates_waveforms.shape[1], 
                            templates_waveforms.shape[2]))
            templates_waveforms = np.vstack([templates_waveforms, padding])
            
            if found_pc_features:
                pc_features_idx = np.vstack([
                                    pc_features_idx, 
                                    np.zeros((n_new_rows,
                                              pc_features_idx.shape[1]))
                                        ])
            
            for u in new_templates:
                # find corresponding pre merge/split templates and PCs
                oldTemplates = spike_templates[spike_clusters == u]
                merged_unit = len(np.unique(oldTemplates)) > 1
                
                if merged_unit:  # average if merge
                    newWaveform = np.mean(templates_waveforms[np.unique(oldTemplates), :, :], axis=0)
                else:  # just take value if split
                    newWaveform = templates_waveforms[np.unique(oldTemplates), :, :]
                templates_waveforms[u, :, :] = newWaveform
                
                if found_pc_features:
                    if merged_unit:
                        newPcFeatureIdx = np.mean(pc_features_idx[np.unique(oldTemplates), :], axis=0)
                    else:
                        newPcFeatureIdx = pc_features_idx[np.unique(oldTemplates), :]
                    pc_features_idx[u, :] = newPcFeatureIdx
    
    spike_templates = spike_templates.astype(int)
    if found_pc_features:
        pc_features_idx = pc_features_idx.astype(int)

    return spike_clusters, templates_waveforms, pc_features_idx


def get_gain_spikeglx(meta_path):
    """
    This function calculates the scaling factor to convert 16-bit analog values to microvolts.

    Uses the SpikeGLX formula: V = i * Vmax / Imax / gain

    For NP1/3A/3B probes:
        - Imax = imMaxInt (typically 512)
        - Vmax = imAiRangeMax (typically 0.6V)
        - gain = imChan0apGain (typically 500)

    For NP2/NP2.1/NP2.4 probes (type 21, 24):
        - Imax = 8192
        - Vmax = imAiRangeMax (typically 0.6V)
        - gain = 80 (fixed)

    Parameters
    ----------
    meta_path : str
        The path to the meta data file

    Returns
    -------
    scaling_factor : float
        The scaling factor to convert from int16 to microvolts (µV/bit)

    Raises
    ------
    Exception
        If the probe type is not handled or required meta fields are missing
    """
    meta_dict = erw.read_meta(Path(meta_path))

    # Check if this is an Open Ephys file
    if str(meta_path).endswith('.oebin'):
        # For Open Ephys files, the bit_volts value is already the scaling factor
        if 'bitVolts' in meta_dict:
            # bitVolts is already in microvolts per bit
            return float(meta_dict['bitVolts'])
        else:
            raise Exception(
                "Open Ephys meta file missing 'bitVolts' field. "
                "Cannot determine scaling factor."
            )

    # Determine probe type
    if "imDatPrb_type" in meta_dict:
        probeType = meta_dict["imDatPrb_type"]
    elif "imProbeOpt" in meta_dict:
        probeType = meta_dict["imProbeOpt"]
    else:
        raise Exception(
            "Cannot find imDatPrb_type or imProbeOpt in meta file. "
            "Cannot determine probe type."
        )

    # NP1, 3A, 3B and similar probes
    probeType_1 = np.array(
        (
            "0",
            "1",
            "3",
            "1020",
            "1030",
            "1100",
            "1110",
            "1120",
            "1121",
            "1122",
            "1123",
            "1200",
            "1300",
        )
    )
    # NP2, NP2.1, NP2.4 probes
    probeType_2 = np.array(
        ("21", "24", "2003", "2004", "2013", "2014", "2020")
    )

    # Get Vmax from meta file (in Volts), convert to microvolts
    if "imAiRangeMax" not in meta_dict:
        raise Exception(
            "Meta file missing 'imAiRangeMax' field. "
            "Cannot determine voltage range."
        )
    Vmax_uV = float(meta_dict["imAiRangeMax"]) * 1e6  # Convert V to µV

    if np.isin(probeType, probeType_1):
        # NP1/3A/3B: Read Imax and gain from meta file
        if "imMaxInt" not in meta_dict:
            raise Exception(
                f"Meta file missing 'imMaxInt' field for probe type {probeType}. "
                "Cannot determine Imax."
            )
        Imax = int(meta_dict["imMaxInt"])

        if "imChan0apGain" not in meta_dict:
            raise Exception(
                f"Meta file missing 'imChan0apGain' field for probe type {probeType}. "
                "Cannot determine gain."
            )
        gain = float(meta_dict["imChan0apGain"])

    elif np.isin(probeType, probeType_2):
        # NP2/NP2.1/NP2.4: Fixed Imax=8192 and gain=80
        Imax = 8192
        gain = 80
    else:
        raise Exception(
            f"Probe type '{probeType}' is not recognized. "
            "Please raise a GitHub issue or add the gain_to_uv manually."
        )

    # Calculate scaling factor: V = i * Vmax / Imax / gain
    scaling_factor = Vmax_uV / Imax / gain

    return scaling_factor


def load_bc_results(bc_path):
    """
    Loads saved BombCell results

    Parameters
    ----------
    bc_path : string
        The absolute path to the directory which has the saved BombCell results

    Returns
    -------
    param : dict
        The parameters which were used to run BombCell
    quality_metrics : dict
        The quality metrics extracted
    fraction_RPVs_all_taur : dict
        All of the values fro refractory period violations for each tau_R and each unit
    """
    # Files
    # BombCell params ML
    param_path = os.path.join(bc_path, "_bc_parameters._bc_qMetrics.parquet")
    if os.path.exists(param_path):
        param_df = pd.read_parquet(param_path)
        # Convert DataFrame to dictionary for compatibility with quality functions
        if len(param_df) == 1:
            # Single row - convert to dictionary using iloc[0]
            param = param_df.iloc[0].to_dict()
        else:
            # Multiple rows - use first row
            param = param_df.iloc[0].to_dict()
    else:
        print("Parameter file not found")
        param = None

    # BombCell quality metrics
    quality_metrics_path = os.path.join(bc_path, "templates._bc_qMetrics.parquet")
    if os.path.exists(quality_metrics_path):
        quality_metrics = pd.read_parquet(quality_metrics_path)
    else:
        print("Quality Metrics file not found")

    # BombCell fration RPVS all TauR
    fractions_RPVs_all_taur_path = os.path.join(
        bc_path, "templates._bc_fractionRefractoryPeriodViolationsPerTauR.parquet"
    )
    if os.path.exists(fractions_RPVs_all_taur_path):
        fractions_RPVs_all_taur = pd.read_parquet(fractions_RPVs_all_taur_path)
    else:
        print("Fraction RPVs all TauR file not found")
        fractions_RPVs_all_taur = None

    return param, quality_metrics, fractions_RPVs_all_taur
