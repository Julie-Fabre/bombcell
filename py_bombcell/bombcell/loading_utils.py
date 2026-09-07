import os
import re
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


def get_ap_gain_from_imro(meta_dict, probe_type):
    """
    Read the AP gain of each channel out of the imro table.

    NP1/3A/3B probes let the user set the AP gain per channel, so it is stored
    in the imro table rather than in a dedicated meta field. Meta files written
    before SpikeGLX added `imChan0apGain` have no other record of it.

    Mirrors ChanGainsIM in SpikeGLX's own SGLX_readMeta.

    Parameters
    ----------
    meta_dict : dict
        The meta file read into a dictionary
    probe_type : str
        The `imDatPrb_type` (or `imProbeOpt`) value for this recording

    Returns
    -------
    gains : ndarray or None
        AP gain per channel, or None if the imro table is absent or its
        format is not one this function knows how to read
    """
    imro = meta_dict.get("imroTbl", "").strip()
    if imro == "":
        return None

    # imro tables are a run of parenthesised groups: a header, then one entry
    # per channel, e.g. "(0,384)(0 0 0 500 250 1)(1 0 0 500 250 1)..."
    groups = re.findall(r"\(([^)]*)\)", imro)
    if len(groups) == 0:
        return None

    if probe_type == "1110":
        # Active UHD probes carry a single gain for the whole probe, in the
        # imro header: (type, ref, ..., apGain, lfGain)
        header = groups[0].replace(",", " ").split()
        if len(header) < 5:
            return None
        try:
            return np.array([float(header[3])])
        except ValueError:
            return None

    # Every other NP1-like probe: one entry per channel, laid out as
    # (channel bank refid apGain lfGain [apFilt]). 3A tables omit apFilt.
    gains = []
    for entry in groups[1:]:
        fields = entry.replace(",", " ").split()
        if len(fields) < 5:
            return None
        try:
            gains.append(float(fields[3]))
        except ValueError:
            return None

    if len(gains) == 0:
        return None
    return np.array(gains)


def get_gain_spikeglx(meta_path):
    """
    This function calculates the scaling factor to convert 16-bit analog values to microvolts.

    Uses the SpikeGLX formula: V = i * Vmax / Imax / gain

    Vmax is read from `imAiRangeMax` in the meta file.

    Imax is determined with the following fallback chain:
        1. Read 'imMaxInt' from meta file (preferred)
        2. Fall back to probe-type-specific defaults:
           - NXT probes: no fallback, 'imMaxInt' is required (see below)
           - NP1/3A/3B probes: 512 (10-bit ADC)
           - NP2/NP2.1/NP2.4 probes: 2048 (commercial, 12-bit ADC) or
             8192 (pre-commercial, 14-bit ADC)
        3. Fall back to commercial Neuropixels default (512 for NP1, 2048 for NP2)

    The AP gain is determined with the following fallback chain:
        1. Read `imChan0apGain` from the meta file
        2. Fall back to probe-type-specific sources:
            - NXT probes: no fallback, `imChan0apGain` is required (see below)
            - NP1/3A/3B probes: read the per-channel gain from `imroTbl`, as the
              gain is user-configurable and cannot be assumed
            - NP2 pre-commercial (21, 24): 80
            - NP2 commercial (all other `2...` imDatPrb_type codes): 100

    For NP1/3A/3B probes:
        - Imax = imMaxInt (typically 512)
        - Vmax = imAiRangeMax (typically 0.6V, i.e. 1.2 Vpp)
        - gain = imChan0apGain, else channel 0's gain in imroTbl (typically 500)

    For NXT probes (identified by `imDatPrb_tech`, as SpikeGLX has not published
    `imDatPrb_type` codes for them):
        - Imax = imMaxInt, required
        - Vmax = imAiRangeMax (typically 0.67V for the active parts)
        - gain = imChan0apGain, required (typically 100 for the active parts)
        The passive NP3000 is NP1-like (10-bit, user-set gain) while the active
        NP30xx parts are 12-bit at gain 100, so neither value is assumed.

    For NP2/NP2.1/NP2.4 probes:
        - Imax = imMaxInt (typically 2048 for commercial, 8192 for pre-commercial)
        - Vmax = imAiRangeMax (typically 0.62V for commercial, 0.5V for
          pre-commercial)
        - gain = imChan0apGain (typically 100 for commercial, 80 for pre-commercial)

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
        (
            "21",
            "24",
            "2003",
            "2004",
            "2005",
            "2006",
            "2013",
            "2014",
            "2020",
            "2021",
            "2022",
            "2300",
        )
    )

    # Get Vmax from meta file (in Volts), convert to microvolts
    if "imAiRangeMax" not in meta_dict:
        raise Exception(
            "Meta file missing 'imAiRangeMax' field. "
            "Cannot determine voltage range."
        )
    Vmax_uV = float(meta_dict["imAiRangeMax"]) * 1e6  # Convert V to µV

    if np.isin(probeType, probeType_1):
        # NP1/3A/3B: Read Imax from meta file, fallback to 512 (commercial default)
        if "imMaxInt" in meta_dict:
            Imax = int(meta_dict["imMaxInt"])
        else:
            Imax = 512  # Commercial NP1 default (10-bit ADC: 2^10 / 2)

        if "imChan0apGain" in meta_dict:
            gain = float(meta_dict["imChan0apGain"])
        else:
            # Meta files written before SpikeGLX added imChan0apGain keep the
            # per-channel AP gain in the imro table instead.
            imro_gains = get_ap_gain_from_imro(meta_dict, probeType)
            if imro_gains is None:
                raise Exception(
                    f"Meta file missing 'imChan0apGain' field for probe type {probeType}, "
                    "and the AP gain could not be read from 'imroTbl'. "
                    "Cannot determine gain."
                )
            gain = float(imro_gains[0])

            if np.unique(imro_gains).size > 1:
                import warnings
                warnings.warn(
                    f"AP gain varies across channels in 'imroTbl' "
                    f"(values: {np.unique(imro_gains)}). "
                    f"Using channel 0's gain ({gain}) for every channel."
                )

    elif np.isin(probeType, probeType_2):
        # NP2/NP2.1/NP2.4: Read Imax from meta file, fallback to 2048 (commercial default)
        if "imMaxInt" in meta_dict:
            Imax = int(meta_dict["imMaxInt"])
        else:
            Imax = 2048  # Commercial NP2 default (12-bit ADC: 2^12 / 2)

        # AP gain: prefer meta file's imChan0apGain. Only fall back to subtype defaults
        # when the field is absent.
        if "imChan0apGain" in meta_dict:
            gain = float(meta_dict["imChan0apGain"])
        elif probeType in ("21", "24"):
            gain = 80.0    # Pre-commercial probes
        else:
            gain = 100.0   # Commercial NP2 probes

    elif meta_dict.get("imDatPrb_tech") == "nxt":
        # Neuropixels NXT. SpikeGLX does not publish imDatPrb_type codes for
        # these yet, so they are identified by imDatPrb_tech instead. The NXT
        # family is not electrically uniform -- the passive NP3000 is NP1-like
        # (10-bit, user-configurable gain) while the active NP30xx probes are
        # 12-bit with a fixed gain of 100 -- so both values are required from
        # the meta file rather than assumed. Every NXT recording has them.
        missing = [f for f in ("imMaxInt", "imChan0apGain") if f not in meta_dict]
        if len(missing) > 0:
            raise Exception(
                f"Meta file for an NXT probe (imDatPrb_pn "
                f"'{meta_dict.get('imDatPrb_pn', 'unknown')}') is missing "
                f"{missing}. These cannot be assumed for NXT probes, as the "
                "passive and active parts differ in both ADC depth and gain."
            )
        Imax = int(meta_dict["imMaxInt"])
        gain = float(meta_dict["imChan0apGain"])

    else:
        # Unknown probe type: try to read from meta, fallback to commercial NP2 default
        if "imMaxInt" in meta_dict:
            Imax = int(meta_dict["imMaxInt"])
        else:
            Imax = 2048  # Commercial default

        if "imChan0apGain" in meta_dict:
            gain = float(meta_dict["imChan0apGain"])
        else:
            gain = 100.0  # NP2 default gain

        import warnings
        warnings.warn(
            f"Probe type '{probeType}' is not recognized. "
            f"Using Imax={Imax} and gain={gain}. "
            "Please raise a GitHub issue to add support for this probe type."
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
        quality_metrics = None

    # Repopulate unique_templates / empty_unit_idx in param — these are normally
    # set as a side-effect of make_qualityMetrics, but downstream plotting code
    # (e.g. plot_waveforms_overlay) reads them from param.
    if param is not None and quality_metrics is not None:
        if "phy_clusterID" in quality_metrics.columns:
            param["unique_templates"] = quality_metrics["phy_clusterID"].to_numpy().astype(int)
        else:
            param["unique_templates"] = np.arange(len(quality_metrics))
        if "nSpikes" in quality_metrics.columns:
            param["empty_unit_idx"] = (quality_metrics["nSpikes"].to_numpy() == 0)
        else:
            param["empty_unit_idx"] = np.zeros(len(quality_metrics), dtype=bool)

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
