import os
from pathlib import Path

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from numpy.typing import NDArray

from cachecache import Cacher, distributed_cacher
__cachedir__ = "~/.bombcell"
global_bc_cacher = Cacher(__cachedir__)


def path_handler(path: str) -> None:
    path = Path(path).expanduser()
    assert path.parent.exists(), f"{str(path.parent)} must exist to create {str(path)}."
    path.mkdir(exist_ok=True)
    return path

def get_metric_keys():
    return [
            # noise metrics
            "nPeaks",
            "nTroughs",
            "waveformDuration_peakTrough",
            "spatialDecaySlope",
            "waveformBaselineFlatness",
            "scndPeakToTroughRatio",
            # non-somatic metrics
            "mainPeakToTroughRatio",
            "peak1ToPeak2Ratio",
            "troughToPeak2Ratio",
            "mainPeak_before_width",
            "mainTrough_width",
            # MUA metrics 
            "percentageSpikesMissing_gaussian",
            "percentageSpikesMissing_symmetric",
            "RPV_window_index",
            "fractionRPVs_estimatedTauR",
            "presenceRatio",
            "maxDriftEstimate",
            "cumDriftEstimate",
            "rawAmplitude",
            "signalToNoiseRatio",
            "isolationDistance",
            "Lratio",
            "silhouetteScore",
            # time metrics
            "useTheseTimesStart",
            "useTheseTimesStop",
            ]


def save_quality_metric_tsv(metric_data, template_ids, output_dir, filename, column_names):
    """
    Save a quality metric array as a TSV file.

    Parameters
    ----------
    metric_data : array
        The quality metric data to save
    template_ids : array
        Array of unit template IDs
    output_dir : str
        Directory path for saving the file
    filename : str
        Name of the output file
    column_names : tuple
        Column headers for the TSV file (template ID column, metric column)
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Construct full file path
    file_path = os.path.join(output_dir, filename)

    # Create DataFrame and save as TSV
    df = pd.DataFrame(
        data=np.array((template_ids, metric_data)).T,
        columns=column_names
    )
    df.to_csv(file_path, sep="\t", index=False)

def save_all_quality_metrics(quality_metrics, unit_types, template_ids, output_dir, param, ks_dir):
    """
    Save all quality metrics as separate TSV files.

    Parameters
    ----------
    quality_metrics : dict
        Dictionary containing various quality metrics arrays
    unit_types : array
        Array containing unit type labels
    template_ids : array
        Array of unit template IDs
    output_dir : str
        Directory path for saving BombCell results (used for internal files)
    ks_dir : str
        Directory path to the original Kilosort directory (for TSV files - Phy compatibility)
    """
    if param["saveAsTSV"]:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # IMPORTANT: TSV files must be saved in the original Kilosort directory for Phy compatibility
        # Always use ks_dir (the original Kilosort directory) for TSV files
        tsv_save_dir = ks_dir
        
        # Additional safety check: if ks_dir somehow contains 'bombcell' subdirectory, correct it
        if 'bombcell' in str(ks_dir) and os.path.basename(ks_dir) == 'bombcell':
            tsv_save_dir = os.path.dirname(ks_dir)
            if param.get("verbose", False):
                print(f"📁 Corrected TSV location from bombcell subdirectory to: {tsv_save_dir}")
        
        if param.get("verbose", False):
            print(f"📁 Saving TSV files to Kilosort directory: {tsv_save_dir}")

        if param["unit_type_for_phy"]:
            # Save unit types to Kilosort directory (not bombcell subdirectory)
            save_quality_metric_tsv(
                unit_types,
                template_ids,
                tsv_save_dir,  # Use corrected directory
                "cluster_bc_unitType.tsv",
                ("cluster_id", "bc_unitType")
            )

        # Get all metric keys
        metric_keys = get_metric_keys()

        # Generate filename and column headers for each metric
        for metric_name in metric_keys:
            if metric_name in quality_metrics:
                # Convert metric name to filename format
                filename = f"cluster_{metric_name}.tsv"
                
                # Create column headers
                column_names = ("cluster_id", metric_name)
                
                # Save the metric to Kilosort directory (not bombcell subdirectory)
                save_quality_metric_tsv(
                    quality_metrics[metric_name],
                    template_ids,
                    tsv_save_dir,  # Use corrected directory
                    filename,
                    column_names
                )
            else:
                print(f"Warning: Metric '{metric_name}' not found in quality_metrics dictionary")

def save_quality_metrics_and_verify(quality_metrics, unit_types, template_ids, output_dir, param, ks_dir):
    """
    Save all quality metrics and verify that all expected metrics were saved.

    Parameters
    ----------
    quality_metrics : dict
        Dictionary containing various quality metrics arrays
    unit_types : array
        Array containing unit type labels
    template_ids : array
        Array of unit template IDs
    output_dir : str
        Directory path for saving the files
    """
    # Save all metrics
    save_all_quality_metrics(quality_metrics, unit_types, template_ids, output_dir, param, ks_dir)
    
    # Verify all metrics were saved
    expected_metrics = set(get_metric_keys())
    saved_metrics = set(quality_metrics.keys())
    missing_metrics = expected_metrics - saved_metrics
    
    if missing_metrics:
        print("Warning: The following metrics were not found in the quality_metrics dictionary:")
        for metric in sorted(missing_metrics):
            print(f"  - {metric}")
    else:
        print("All expected metrics were successfully saved.")


def save_dict_as_parquet_and_csv(
    dic, save_path, file_name="templates._bc_qMetrics"
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

    file_path = str(save_path / file_name)
    quality_metrics_df = pd.DataFrame.from_dict(dic)
    quality_metrics_df.to_parquet(file_path + ".parquet")
    quality_metrics_df.to_csv(file_path + ".csv")


def save_params_as_parquet(
    param, save_path, file_name="_bc_parameters._bc_qMetrics"
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
    for key, value in param.items():
        if key == 'ephysKilosortPath':
            param_save[key] = str(value)
        if type(value) == Path:
            param_save[key] = str(value)

    file_path = save_path / file_name
    param_df = pd.DataFrame.from_dict([param_save])
    param_df.to_parquet(str(file_path) + ".parquet")


def save_waveforms_as_npy(raw_waveforms_full, raw_waveforms_peak_channel, raw_waveforms_id_match, save_path):
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

    file_path_raw_waveforms = save_path / "templates._bc_rawWaveforms.npy"
    np.save(file_path_raw_waveforms, raw_waveforms_full)

    file_path_peak_channels = save_path / "templates._bc_rawWaveformPeakChannels.npy"
    np.save(file_path_peak_channels, raw_waveforms_peak_channel)

    file_path_raw_waveforms_id_match = save_path / "_bc_rawWaveforms_kilosort_format"
    np.save(file_path_raw_waveforms_id_match, raw_waveforms_id_match)


def save_results(
    quality_metrics,
    unit_type_string,
    unique_templates,
    param,
    raw_waveforms_full,
    raw_waveforms_peak_channel,
    raw_waveforms_id_match,
    save_path,
    ks_dir,
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

    save_quality_metrics_and_verify(quality_metrics, unit_type_string, unique_templates, save_path, param, ks_dir)

    # Get rid of peak channels of empty rows, which were kept for convenient indexing up to here
    quality_metrics_save = quality_metrics.copy()
    quality_metrics_save["maxChannels"] = quality_metrics["maxChannels"][
        quality_metrics["phy_clusterID"].astype(int)
    ]

    # Save full quality metrics table
    save_dict_as_parquet_and_csv(
        quality_metrics_save, save_path, file_name="templates._bc_qMetrics"
    )
    save_params_as_parquet(
        param, save_path, file_name="_bc_parameters._bc_qMetrics"
    )

    # Save waveforms
    save_waveforms_as_npy(raw_waveforms_full, raw_waveforms_peak_channel, raw_waveforms_id_match, save_path)
