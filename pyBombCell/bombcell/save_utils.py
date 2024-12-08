import os
from pathlib import Path

import numpy as np
import pandas as pd


def path_handler(path: str) -> None:
    path = Path(path).expanduser()
    assert path.parent.exists(), f"{str(path.parent)} must exist to create {str(path)}."
    path.mkdir(exist_ok=True)
    return path


def save_qmetric_tsv(metric, unique_templates, save_path, file_name, column_titles):
    """
    This function save and array (a quality metric) as a .tsv file

    Parameters
    ----------
    metric : ndarray
        A quality metric
    unique_templates : ndarray
        The array of the IDs of the units
    save_path : str
        The path to the save directory
    file_name : str
        The name of the file
    column_titles : tuple
        A tuple whihc contains the column title for the tsv file
    """
    # Create save_path if it does not exist
    save_path = path_handler(save_path)

    file_path = os.path.join(save_path, file_name)
    data_to_save = pd.DataFrame(
        data=np.array((unique_templates, metric)).T, columns=column_titles
    )
    data_to_save.to_csv(file_path, sep="\t", index=False)


def save_quality_metrics_as_tsvs(
    quality_metrics, unit_type_string, unique_templates, save_path
):
    """
    This function saves the most used quality metrics as a .tsv file

    Parameters
    ----------
    quality_metrics : dict
        The quality metrics dictionary
    unit_type_string : ndarray
        The array which contains the units labels
    unique_templates : ndarray
        The array of the IDs of the units
    save_path : str
        The path to the save directory
    """
    # Create save_path if it does not exist
    save_path = path_handler(save_path)

    save_qmetric_tsv(
        unit_type_string,
        unique_templates,
        save_path,
        r"cluster_bc_unitType.tsv",
        ("cluster_id", "bc_unitType"),
    )
    save_qmetric_tsv(
        quality_metrics["fraction_RPVs"],
        unique_templates,
        save_path,
        r"cluster_frac_RPVs.tsv",
        ("cluster_id", "frac_RPVs"),
    )
    save_qmetric_tsv(
        quality_metrics["is_somatic"],
        unique_templates,
        save_path,
        r"cluster_is_somatic.tsv",
        ("cluster_id", "is_somatic"),
    )
    save_qmetric_tsv(
        quality_metrics["max_drift_estimate"],
        unique_templates,
        save_path,
        r"cluster_max_drift.tsv",
        ("cluster_id", "max_drift"),
    )
    save_qmetric_tsv(
        quality_metrics["n_peaks"],
        unique_templates,
        save_path,
        r"cluster_n_peaks.tsv",
        ("cluster_id", "n_peaks"),
    )
    save_qmetric_tsv(
        quality_metrics["n_troughs"],
        unique_templates,
        save_path,
        r"cluster_n_troughs.tsv",
        ("cluster_id", "n_troughs"),
    )
    save_qmetric_tsv(
        quality_metrics["percent_missing_gaussian"],
        unique_templates,
        save_path,
        r"cluster_percentageSpikesMissing_gaussian.tsv",
        ("cluster_id", "percentageSpikesMissing_gaussian"),
    )
    save_qmetric_tsv(
        quality_metrics["presence_ratio"],
        unique_templates,
        save_path,
        r"cluster_presence_ratio.tsv",
        ("cluster_id", "presence_ratio"),
    )
    save_qmetric_tsv(
        quality_metrics["signal_to_noise_ratio"],
        unique_templates,
        save_path,
        r"cluster_SNR.tsv",
        ("cluster_id", "SNR"),
    )
    save_qmetric_tsv(
        quality_metrics["exp_decay"],
        unique_templates,
        save_path,
        r"cluster_spatial_decay_slope.tsv",
        ("cluster_id", "spatial_decay_slope"),
    )
    save_qmetric_tsv(
        quality_metrics["waveform_duration_peak_trough"],
        unique_templates,
        save_path,
        r"cluster_waveform_dur.tsv",
        ("cluster_id", "waveform_dur"),
    )
    save_qmetric_tsv(
        quality_metrics["waveform_baseline"],
        unique_templates,
        save_path,
        r"cluster_wv_baseline_flatness.tsv",
        ("cluster_id", "wv_baseline_flatness"),
    )


def save_quality_metrics_as_parquet(
    quality_metrics, save_path, file_name="templates._bc_qMetrics.parquet"
):
    """
    This function save the whole quality metrics dictionary as a parquet file

    Parameters
    ----------
    quality_metrics : dict
        The quality metrics dictionary
    save_path : str
        The path to the save directory
    file_name : str, optional
        The name of the file, by default 'templates._bc_qMetrics.parquet'
    """
    # Create save_path if it does not exist
    save_path = path_handler(save_path)

    file_path = os.path.join(save_path, file_name)
    quality_metrics_save = quality_metrics.copy()
    quality_metrics_save["max_channels"] = quality_metrics["max_channels"][
        quality_metrics["cluster_id"].astype(int)
    ]
    quality_metrics_df = pd.DataFrame.from_dict(quality_metrics_save)
    quality_metrics_df.to_parquet(file_path)


def save_params_as_parquet(
    param, save_path, file_name="_bc_parameters._bc_qMetrics.parquet"
):
    """
    This function save the whole param dictionary as a parquet file

    Parameters
    ----------
    param : dict
        The param dictionary
    save_path : str
        The path to the save directory
    file_name : str, optional
        The name of the file, by default '_bc_parameters._bc_qMetrics.parquet'
    """
    # Create save_path if it does not exist
    save_path = path_handler(save_path)

    # PyArrow cant save Path type objects as a parquet
    param_save = param.copy()
    param_save["ephys_kilosort_path"] = str(param["ephys_kilosort_path"])

    file_path = os.path.join(save_path, file_name)
    param_df = pd.DataFrame.from_dict(param_save)
    param_df.to_parquet(file_path)


def save_waveforms_as_npy(raw_waveforms_full, raw_waveforms_peak_channel, save_path):
    """
    This function saves the raw waveform information as npy arrays

    Parameters
    ----------
    raw_waveforms_full : ndarray
        The (n_units, n_channels, time) array of extracted average raw waveforms
    raw_waveforms_peak_channel : ndarray
        The peak channels of the extracted raw waveforms
    save_path : str
        The path to the save directory
    """
    # Create save_path if it does not exist
    save_path = path_handler(save_path)

    file_path_raw_waveforms = os.path.join(save_path, "templates._bc_rawWaveforms.npy")
    np.save(file_path_raw_waveforms, raw_waveforms_full)

    file_path_peak_channels = os.path.join(
        save_path, "templates._bc_rawWaveformsPeakChannels.npy"
    )
    np.save(file_path_peak_channels, raw_waveforms_peak_channel)


def save_results(
    quality_metrics,
    unit_type_string,
    unique_templates,
    param,
    raw_waveforms_full,
    raw_waveforms_peak_channel,
    save_path,
):
    """
    This function saves all of the BombCell data to the given save directory

    Parameters
    ----------
    quality_metrics : dict
        The quality metrics dictionary
    unit_type_string : ndarray
        The array which contains the units labels
    unique_templates : ndarray
        The array of the IDs of the units
    param : dict
        The param dictionary
    raw_waveforms_full : ndarray
        The (n_units, n_channels, time) array of extracted average raw waveforms
    raw_waveforms_peak_channel : ndarray
        The peak channels of the extracted raw waveforms
    save_path : str
        The path to the save directory
    """
    # Create save_path if it does not exist
    save_path = path_handler(save_path)

    save_quality_metrics_as_tsvs(
        quality_metrics, unit_type_string, unique_templates, save_path
    )
    save_quality_metrics_as_parquet(
        quality_metrics, save_path, file_name="templates._bc_qMetrics.parquet"
    )
    save_params_as_parquet(
        param, save_path, file_name="_bc_parameters._bc_qMetrics.parquet"
    )
    save_waveforms_as_npy(raw_waveforms_full, raw_waveforms_peak_channel, save_path)
