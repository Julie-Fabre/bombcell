import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import matplotlib.axes
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path
import os
from typing import List, Tuple, Optional, Dict
try:
    from upsetplot import UpSet, from_indicators
    UPSETPLOT_AVAILABLE = True
except ImportError:
    UPSETPLOT_AVAILABLE = False
    print("Warning: upsetplot-bombcell not available. Some plotting functions may not work.")

from bombcell import helper_functions as hf

#V0.9 of upset plots have pandas future warning so, suppressing them
import warnings

from collections import namedtuple

######################################################
# Default color scheme for unit types
######################################################
DEFAULT_UNIT_COLORS = {
    'NOISE': '#8B0000',          # Dark red
    'GOOD': '#228B22',           # Forest green
    'MUA': '#DAA520',            # Goldenrod
    'NON-SOMA': '#4169E1',       # Royal blue
    'NON-SOMA GOOD': '#4169E1',  # Royal blue
    'NON-SOMA MUA': '#87CEEB',   # Light sky blue
    # Alternative labels (lowercase)
    'noise': '#8B0000',
    'somatic, good': '#228B22',
    'somatic, MUA': '#DAA520',
    'non-somatic': '#4169E1',
    'non-somatic, good': '#4169E1',
    'non-somatic, MUA': '#87CEEB',
}

######################################################
# Top Level Functions
# These functions represent the original interface
# to BombCell's plotting functionality, and are 
# geared towards generating plots in the notebook
# environment
######################################################
def plot_summary_data(quality_metrics, template_waveforms, unit_type, unit_type_string, param, return_figures=False):
    """
    This function plots summary figure to visualize bombcell's results

    Parameters
    ----------
    quality_metrics : dict
        The dictionary containing all quality metrics
    template_waveforms : ndarray
        The array containing all waveforms for each unit
    unit_type : ndarray
        The bombcell integer unit classification
    unit_type_string : ndarray
        The bombcell string unit classification
    unique_templates : ndarray
        The array which converts to the original unit ID's
    param : dict
        The dictionary of all bomcell parameters
    return_figures : bool, optional
        If True, returns a dictionary of figure objects, by default False

    Returns
    -------
    dict or None
        If return_figures is True, returns a dictionary with keys:
        'waveforms_overlay', 'upset_plots', 'histograms'
        Otherwise returns None
    """
    figures = {}
    
    if param["plotGlobal"]:
        # Get save directory if saving is enabled
        save_dir = None
        if param.get("savePlots", False):
            if param.get("plotsSaveDir"):
                save_dir = Path(param["plotsSaveDir"])
            else:
                save_dir = Path(param["ephysKilosortPath"]) / "bombcell_plots"
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Plot waveforms overlay
        fig_waveforms = plot_waveforms_overlay(quality_metrics, template_waveforms, unit_type, param, save_dir=save_dir)
        if return_figures:
            figures['waveforms_overlay'] = fig_waveforms
            
        # Plot upset plots
        fig_upset_list = upset_plots(quality_metrics, unit_type_string, param, save_dir=save_dir)
        if return_figures:
            figures['upset_plots'] = fig_upset_list
            
        # Plot histograms
        fig_histograms = plot_histograms(quality_metrics, param, save_dir=save_dir)
        if return_figures:
            figures['histograms'] = fig_histograms
    
    if return_figures:
        return figures
    return None


def plot_waveforms_overlay(quality_metrics, template_waveforms, unit_type, param, save_dir=None):
    """
    This function plots overlaid waveforms for each of bombcell's unit classification types (e.g Noise, MUA..)

    Parameters
    ----------
    quality_metrics : dict
        The dictionary containing all quality metrics
    template_waveforms : ndarray
        The array containing all waveforms for each unit
    unit_type : ndarray
        The bombcell integer unit classification
    unique_templates : ndarray
        The array which converts to the original unit ID's
    param : dict
        The dictionary of all bomcell parameters
    save_dir : Path or str, optional
        Directory to save the figure to, by default None

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    #One figure all of a unit type
    #if split into 4 unit types
    unique_templates = param['unique_templates']

    # set labels based on param["splitGoodAndMua_NonSomatic"]
    if param["splitGoodAndMua_NonSomatic"]:
        labels = {
            0: "noise", 
            1: "somatic, good", 
            2: "somatic, MUA", 
            3: "non-somatic, good",
            4: "non-somatic, MUA"
        }
    else:
        labels = {
            0: "noise",
            1: "somatic, good",
            2: "somatic, MUA",
            3: "non-somatic"
        }
    
    n_categories = len(labels.keys())
    if n_categories < 5:
        nrows = 2
        ncols = 2
        img_pos = [[0,0], [0,1], [1,0], [1,1]]
        n_plots = 4
    else:
        nrows = 3
        ncols = 2
        img_pos = [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]]
        n_plots = 5
    
    #TODO change alpha to be inversly proprotional to n units
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols)
    for plot_idx in range(nrows * ncols):
        if plot_idx < n_plots:
            unit_type_ = plot_idx
            unit_type_str = labels[unit_type_]
            ax = axs[img_pos[plot_idx][0]][img_pos[plot_idx][1]]
            generate_waveform_overlay(param, quality_metrics, unit_type_str, template_waveforms, ax)        
        else:
            # Hide the unused subplot (6th subplot when n_plots=5)
            ax = axs[img_pos[plot_idx][0]][img_pos[plot_idx][1]]
            ax.spines[["right", "top", "bottom", "left"]].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
    
    # Save figure if requested
    if save_dir is not None:
        save_path = Path(save_dir) / "waveforms_overlay.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if param.get("verbose", True):
            print(f"Saved waveforms overlay figure to {save_path}")
    
    # Return the figure object
    return fig


def upset_plots(quality_metrics, unit_type_string, param, save_dir=None):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    """
    This function plots three upset plots, showing how each metric is connected

    Parameters
    ----------
    quality_metrics : dict
        The dictionary containing all quality metrics
    unit_type_string : ndarray
        The bombcell string unit classification
    unique_templates : ndarray
        The array which converts to the original unit ID's
    param : dict
        The dictionary of all bomcell parameters
    save_dir : Path or str, optional
        Directory to save the figures to, by default None

    Returns
    -------
    list of matplotlib.figure.Figure
        List of figure objects for each upset plot
    """

    qm_table = hf.make_qm_table(quality_metrics, param, unit_type_string)
    
    figures = []
    
    # Determine which unit types to plot based on param["splitGoodAndMua_NonSomatic"]
    if param["splitGoodAndMua_NonSomatic"]:
        # When splitting non-somatic into good and MUA
        unit_types = ["NOISE", "NON-SOMA GOOD", "NON-SOMA MUA", "MUA"]
    else:
        # Original behavior: all non-somatic together
        unit_types = ["NOISE", "NON-SOMA", "MUA"]
    
    for unit_type in unit_types:
        fig = plt.figure()
        generate_upset_plot(qm_table, unit_type, fig=fig)
        figures.append(fig)
        
        # Save figure if requested
        if save_dir is not None:
            # Replace spaces with underscores in filename
            filename_unit_type = unit_type.lower().replace(" ", "_")
            save_path = Path(save_dir) / f"upset_plot_{filename_unit_type}.png"
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            if param.get("verbose", True):
                print(f"Saved {unit_type} upset plot to {save_path}")
    
    return figures


def plot_histograms(quality_metrics, param, save_dir=None):
    """
    This function find what metrics have been extracted and plots histograms for each metric

    Parameters
    ----------
    quality_metrics : dict
        The dictionary containing all quality metrics
    param : dict
        The dictionary of all bomcell parameters
    save_dir : Path or str, optional
        Directory to save the figure to, by default None

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """

    from .plotting_utils import get_color_from_matrix, get_metric_info_list

    # get valid metrics, i.e., metrics present in quality_metrics dictionary
    metric_info = get_metric_info_list(param, quality_metrics)
    valid_metric_info = [mi for mi in metric_info if mi.plot_condition and (mi.name in quality_metrics)]

    # Calculate grid layout
    num_subplots = len(valid_metric_info)
    num_rows = int(np.floor(np.sqrt(num_subplots)))
    num_cols = int(np.ceil(num_subplots / num_rows))

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 9))
    fig.patch.set_facecolor('white')
    
    if num_rows == 1:
        axs = axs.reshape(1, -1)
    elif num_cols == 1:
        axs = axs.reshape(-1, 1)

    # loop through valid metrics, plotting histogram for each one
    for idx, vmi in enumerate(valid_metric_info):
        row_id = idx // num_cols
        col_id = idx % num_cols
        ax = axs[row_id, col_id]

        # get color from color matrix (use modulus operator for wraparound)
        bar_color = get_color_from_matrix(idx)
        
        # plotting function goes here
        include_y_label = (idx==0)
        generate_histogram(vmi.name, quality_metrics, param, bar_color, ax, include_y_label)

    # Hide unused subplots
    for i in range(len(valid_metric_info), num_rows * num_cols):
        row_id = i // num_cols
        col_id = i % num_cols
        axs[row_id, col_id].set_visible(False)

    plt.tight_layout()
    
    # Save figure if requested
    if save_dir is not None:
        save_path = Path(save_dir) / "quality_metrics_histograms.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if param.get("verbose", True):
            print(f"Saved quality metrics histograms to {save_path}")
    
    # Only show if not saving (to avoid backend issues)
    if save_dir is None:
        plt.show()
    
    # Return the figure object
    return fig



######################################################
# Individual Plot Generation Functions
# These functions generate the individual plots that
# make up the output of the top-level functions
######################################################
def generate_waveform_overlay(
        param: dict, 
        quality_metrics: dict, 
        unit_type_str: str, 
        template_waveforms: np.ndarray=None,
        ax: matplotlib.axes.Axes = None,
    ):
    if template_waveforms is None:
        from .loading_utils import load_ephys_data
        ks_dir = param["ephysKilosortPath"]
        _, _, template_waveforms, _, _, _, _ = load_ephys_data(ks_dir)
    
    from .quality_metrics import get_quality_unit_type
    unit_types_all, _ = get_quality_unit_type(param, quality_metrics)

    # get unit_type integer
    if param["splitGoodAndMua_NonSomatic"]:
        try:
            unit_type = {
                "noise": 0,
                "somatic, good": 1,
                "somatic, MUA": 2,
                "non-somatic, good": 3,
                "non-somatic, MUA": 4,
            }[unit_type_str]
        except KeyError:
            raise(f"Invalid unit type {unit_type_str} - permitted values are 'noise', 'somatic, good', 'somatic, MUA', 'non-somatic, good', 'non-somatic, MUA'")
    else:
        try:
            unit_type = {
                "noise": 0,
                "somatic, good": 1,
                "somatic, MUA": 2,
                "non-somatic": 3,
            }[unit_type_str]
        except KeyError:
            raise(f"Invalid unit type {unit_type_str} - permitted values are 'noise', 'somatic, good', 'somatic, MUA', 'non-somatic'")

    # get unique templates
    unique_templates = param["unique_templates"]
    unit_type_template_ids = unique_templates[unit_types_all==unit_type]
    n_units_of_type = unit_type_template_ids.size

    # initialize figure, axis handles

    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = plt.gcf() # placeholder???

    # if the current unit type has more than 0 units, generate a plot
    if n_units_of_type > 0:
        for template_id in unit_type_template_ids:
            max_channel_id = quality_metrics["maxChannels"][template_id]
            template_max_waveform = template_waveforms[template_id, 0:, max_channel_id] # template waveforms comes from load_ephys_data
            ax.plot(template_max_waveform, color="black", alpha=0.1)
            ax.spines[["right", "top", "bottom", "left"]].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{unit_type_str} units (n = {n_units_of_type})")
    
    else:
        ax.spines[["right", "top", "bottom", "left"]].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"No {unit_type_str} units (n = 0)")
        
    return fig, ax


def generate_upset_plot(
        qm_table: pd.DataFrame, 
        unit_type_str: str,
        fig: matplotlib.figure.Figure = None,
):
    try:
        # ensure upper case
        unit_type_str = unit_type_str.upper()

        # get metrics relevant to chosen unit type
        if unit_type_str=="NOISE":
            unit_type_metrics = ["# peaks", "# troughs", "waveform duration", "spatial decay", "baseline flatness", "peak2 / trough"] #Duration is peak to trough duration
        elif unit_type_str=="NON-SOMA" or unit_type_str=="NON-SOMA GOOD" or unit_type_str=="NON-SOMA MUA":
            unit_type_metrics = ["trough / peak2", "peak1 / peak2"]
        elif unit_type_str=="MUA":
            unit_type_metrics = ["SNR", "amplitude", "presence ratio", "# spikes", "% spikes missing", "fraction RPVs", "max. drift", "isolation dist.", "L-ratio"]
        else:
            raise ValueError(f"Invalid unit type {unit_type_str} - allowed values are 'NOISE', 'NON-SOMA', 'NON-SOMA GOOD', 'NON-SOMA MUA', 'MUA'")
        
        # filter out uncomputed metrics
        unit_type_metrics = [m for m in unit_type_metrics if m in qm_table.columns]

        # generate mask for the chosen unit type and filter the data from qm_table
        # For NON-SOMA unit types
        if unit_type_str == "NON-SOMA":
            unit_type_mask = qm_table['unit_type'].str.startswith("NON-SOMA")
        elif unit_type_str == "NON-SOMA GOOD":
            unit_type_mask = qm_table['unit_type'] == "NON-SOMA GOOD"
        elif unit_type_str == "NON-SOMA MUA":
            unit_type_mask = qm_table['unit_type'] == "NON-SOMA MUA"
        else:
            unit_type_mask = qm_table['unit_type'].str.startswith(unit_type_str)
        unit_type_data = qm_table.loc[unit_type_mask, unit_type_metrics]

        # in the unit_type_data dataframe, some metric names should be replaced with more comprehensible "display names"
        display_names = {
            "SNR": "signal/noise (SNR)",
            "fraction RPVs": "refractory period viol. (RPV)",
            "amplitude": "amplitude"
        }
        metric_display_names = [display_names.get(metric_name, metric_name) for metric_name in unit_type_metrics]
        unit_type_data.columns = metric_display_names

        # count the number of units of the chosen type
        n_unit_type = unit_type_mask.sum()
        
        # count the total number of units
        n_total_units = len(qm_table)
        
        # check how many columns have True values
        n_cols_with_true_vals = (unit_type_data.sum() > 0).sum()
        if n_cols_with_true_vals >= 1 and n_unit_type > 0:
            upset = UpSet(from_indicators(metric_display_names, data=unit_type_data), min_degree=1)
            upset.plot(fig=fig)
            plt.suptitle(f"Units classified as {unit_type_str.lower()} (n = {n_unit_type}/{n_total_units})")
        elif n_unit_type > 0:
            print(f"{unit_type_str.capitalize()} upset plot skipped: no metrics have failures")
    except (AttributeError, ValueError) as e:
        import warnings
        warnings.warn(f"Could not create {unit_type_str.lower()} upset plot due to library compatibility: {e}", RuntimeWarning)

def generate_histogram(
        metric_name, 
        quality_metrics, 
        param, 
        bar_color=None, 
        ax=None, 
        include_y_label=False
):

    if ax is None:
        fig, ax = plt.subplots(1,1)
    else:
        fig = plt.gcf() # placeholder?

    if bar_color is None:
        bar_color='k'

    from .plotting_utils import get_metric_info_dict
    from .helper_functions import clean_inf_values

    vmi = get_metric_info_dict(param, quality_metrics)[metric_name]
    
    # Use the shared utility to clean inf values
    cleaned_metrics = clean_inf_values(quality_metrics, [vmi.name])
    metric_data = cleaned_metrics[vmi.name]
    
    # Remove NaN and inf values for all metrics
    metric_data = metric_data[~np.isnan(metric_data)]
    metric_data = metric_data[~np.isinf(metric_data)]
    
    if len(metric_data) > 0:
        # Plot histogram with probability normalization (MATLAB style)
        if vmi.name in ['nPeaks', 'nTroughs']:
            # Use integer bins for discrete metrics
            bins = np.arange(np.min(metric_data), np.max(metric_data) + 2) - 0.5
        elif vmi.name == 'waveformDuration_peakTrough':
            # Use fewer bins for waveform duration like MATLAB
            bins = 20
        else:
            bins = 40
            
        n, bins_out, patches = ax.hist(metric_data, bins=bins, density=True, 
                                        color=bar_color, alpha=0.7)
    
        if vmi.name in ['nPeaks', 'nTroughs']:
            binsize_offset = 0.5
        else:
            binsize_offset = (bins_out[1] - bins_out[0]) / 2 if len(bins_out) > 1 else 0
        
        # Convert to probability (like MATLAB's 'Normalization', 'probability')
        if vmi.name not in ['nPeaks', 'nTroughs']:
            bin_width = bins_out[1] - bins_out[0]
            for patch in patches:
                patch.set_height(patch.get_height() * bin_width)
        
        # Add threshold lines above histogram at 0.9
        x_lim = ax.get_xlim()
        # Extend x-axis to make room for text labels
        x_range = x_lim[1] - x_lim[0]
        ax.set_xlim([x_lim[0] - 0.1*x_range, x_lim[1] + 0.1*x_range])
        x_lim = ax.get_xlim()
        line_y = 0.9  # Position lines at 0.9
        
        line_colors = vmi.line_colors.reshape(3, 3)

        # add vertical lines for thresholds at value 0.5*bin_width
        if vmi.min_threshold is not None:
            ax.axvline(vmi.min_threshold + binsize_offset, color='k', linewidth=2)
        if vmi.max_threshold is not None:
            ax.axvline(vmi.max_threshold + binsize_offset, color='k', linewidth=2)
        
        # add horizontal colored lines at value + 0.5*bin_width
        if vmi.min_threshold is not None and vmi.max_threshold is not None:
            ax.plot([x_lim[0], vmi.min_threshold + binsize_offset], [line_y, line_y], color=line_colors[0], linewidth=6,) # left
            ax.plot([vmi.min_threshold + binsize_offset, vmi.max_threshold + binsize_offset], [line_y, line_y], color=line_colors[1], linewidth=6,) # middle
            ax.plot([vmi.max_threshold + binsize_offset, x_lim[1]], [line_y, line_y], color=line_colors[2], linewidth=6,) # right

        elif (vmi.min_threshold is not None and vmi.max_threshold is None) \
            or (vmi.min_threshold is None and vmi.max_threshold is not None):

            threshold = vmi.min_threshold if vmi.min_threshold is not None else vmi.max_threshold
            ax.plot([x_lim[0], threshold + binsize_offset], [line_y, line_y], color = line_colors[0], linewidth=6,) # left
            ax.plot([threshold + binsize_offset, x_lim[1]], [line_y, line_y], color = line_colors[1], linewidth=6,) # right
        
        # set up labels for histogram's horizontal ranges -- first is "bad" label, then is "good" label
        labels = {
            "noise": ("Noise", "Neuronal"),
            "nonsomatic": ("Non-Somatic", "Somatic"),
            "mua": ("MUA", "Good"),
        }[vmi.metric_type]

        bad_label = labels[0]
        good_label = labels[1]
        
        horizontal_markers = []
        horizontal_markers.append(x_lim[0])
        if vmi.min_threshold is not None: 
            horizontal_markers.append(vmi.min_threshold)
        if vmi.max_threshold is not None: 
            horizontal_markers.append(vmi.max_threshold)
        horizontal_markers.append(x_lim[1])
        
        text_x = [ (a + b) / 2 for a, b in zip(horizontal_markers[:-1], horizontal_markers[1:])] # calculate midpoints between horizontal markers
        text_y = 0.95

        if vmi.min_threshold is not None and vmi.max_threshold is not None:
            ax.text(text_x[0], text_y, f"  {bad_label}  ", ha="center", fontsize=10, color=line_colors[0], weight="bold",) # left -- bad
            ax.text(text_x[1], text_y, f"  {good_label}  ", ha="center", fontsize=10, color=line_colors[1], weight="bold",) # middle -- good
            ax.text(text_x[2], text_y, f"  {bad_label}  ", ha="center", fontsize=10, color=line_colors[2], weight="bold",) # right -- bad

        elif vmi.min_threshold is not None and vmi.max_threshold is None:
            ax.text(text_x[0], text_y, f"  {bad_label}  ", ha="center", fontsize=10, color=line_colors[0], weight="bold",) # left -- bad
            ax.text(text_x[1], text_y, f"  {good_label}  ", ha="center", fontsize=10, color=line_colors[1], weight="bold",) # right -- good

        elif vmi.min_threshold is None and vmi.max_threshold is not None:
            ax.text(text_x[0], text_y, f"  {good_label}  ", ha="center", fontsize=10, color=line_colors[0], weight="bold",) # left -- good
            ax.text(text_x[1], text_y, f"  {bad_label}  ", ha="center", fontsize=10, color=line_colors[1], weight="bold",) # right -- bad

        # Set histogram limits from 0 to 1
        ax.set_ylim([0, 1])

        ax.set_xlabel(vmi.short_name, fontsize=13)
        if include_y_label:
            ax.set_ylabel('frac. units', fontsize=13)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize=12)

    return fig, ax


######################################################
# Supplementary Figure Generation
# These functions generate publication-ready figures
# summarizing BombCell quality control results
######################################################

def _load_single_dataset_for_supp(bc_path: str) -> Tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load BombCell results from a single dataset path.

    Parameters
    ----------
    bc_path : str
        Path to the BombCell results directory

    Returns
    -------
    param : dict
        BombCell parameters
    quality_metrics : dict
        Quality metrics as a dictionary
    template_waveforms : ndarray
        Template waveforms array
    unit_type : ndarray
        Integer unit type classification
    unit_type_string : ndarray
        String unit type classification
    """
    from .loading_utils import load_bc_results, load_ephys_data
    from .quality_metrics import get_quality_unit_type

    param, quality_metrics_df, _ = load_bc_results(bc_path)

    # Convert DataFrame to dict for compatibility
    quality_metrics = {}
    for col in quality_metrics_df.columns:
        quality_metrics[col] = np.array(quality_metrics_df[col])

    # Load template waveforms from kilosort path
    ks_path = param.get('ephysKilosortPath', bc_path)
    _, _, template_waveforms, _, _, _, _ = load_ephys_data(ks_path)

    # Set unique_templates if not present
    if 'unique_templates' not in param:
        if 'phy_clusterID' in quality_metrics:
            param['unique_templates'] = np.array(quality_metrics['phy_clusterID']).astype(int)
        else:
            param['unique_templates'] = np.arange(len(next(iter(quality_metrics.values()))))

    # Get unit classifications
    unit_type, unit_type_string = get_quality_unit_type(param, quality_metrics)

    return param, quality_metrics, template_waveforms, unit_type, unit_type_string


def _aggregate_dataset_statistics(
    all_dataset_paths: List[str],
    split_nonsomatic: bool
) -> dict:
    """
    Compute unit type proportions across all datasets.

    Parameters
    ----------
    all_dataset_paths : list of str
        Paths to BombCell results directories
    split_nonsomatic : bool
        Whether to split non-somatic into good and MUA

    Returns
    -------
    stats : dict
        Dictionary with statistics for each unit type category.
        Keys are category names, values are dicts with:
        - 'proportions': list of per-dataset proportions
        - 'mean': mean proportion
        - 'se': standard error
        - 'n_datasets': number of datasets
        - 'counts': list of counts per dataset
    """
    # Define unit type categories based on splitGoodAndMua_NonSomatic
    if split_nonsomatic:
        categories = ['NOISE', 'GOOD', 'MUA', 'NON-SOMA GOOD', 'NON-SOMA MUA']
    else:
        categories = ['NOISE', 'GOOD', 'MUA', 'NON-SOMA']

    # Initialize storage
    all_proportions = {cat: [] for cat in categories}
    all_counts = {cat: [] for cat in categories}
    total_units_per_dataset = []

    for path in all_dataset_paths:
        try:
            param, qm, _, unit_type, unit_type_string = _load_single_dataset_for_supp(path)
            n_units = len(unit_type_string)
            total_units_per_dataset.append(n_units)

            for cat in categories:
                count = int(np.sum(unit_type_string == cat))
                proportion = count / n_units if n_units > 0 else 0
                all_proportions[cat].append(proportion)
                all_counts[cat].append(count)
        except Exception as e:
            warnings.warn(f"Could not load {path}: {e}")
            continue

    # Compute statistics
    stats = {}
    for cat in categories:
        props = np.array(all_proportions[cat])
        counts = all_counts[cat]
        n_datasets = len(props)
        stats[cat] = {
            'proportions': props,
            'mean': np.mean(props) if n_datasets > 0 else 0,
            'se': np.std(props, ddof=1) / np.sqrt(n_datasets) if n_datasets > 1 else 0,
            'n_datasets': n_datasets,
            'counts': counts,
            'total_count': sum(counts),
        }

    stats['_meta'] = {
        'n_datasets': len(total_units_per_dataset),
        'total_units': sum(total_units_per_dataset),
        'units_per_dataset': total_units_per_dataset
    }

    return stats


def _create_supp_figure_layout(
    n_unit_types: int,
    n_histogram_metrics: int,
    figsize: Tuple[float, float]
) -> Tuple[matplotlib.figure.Figure, dict]:
    """
    Create figure with GridSpec layout for supplementary figure.

    Layout:
    - (a) Histograms at the top
    - (b) Waveforms on the left (2/3 width) below histograms
    - (c) Bar chart on the right (1/3 width) below histograms

    Parameters
    ----------
    n_unit_types : int
        Number of unit type categories (4 or 5)
    n_histogram_metrics : int
        Number of histogram metrics to plot
    figsize : tuple
        Figure size (width, height) - height will be scaled based on content

    Returns
    -------
    fig : Figure
        The matplotlib figure
    axes : dict
        Dictionary with keys 'waveforms', 'histograms', 'bar_chart'
    """
    # Calculate histogram rows needed
    n_hist_rows = int(np.ceil(n_histogram_metrics / 4))
    n_hist_cols = min(4, n_histogram_metrics)

    # Scale figure height based on content
    # Height for histograms (~2.2 inches per row) + waveforms/bar chart (~2.5 inches)
    fig_height = (n_hist_rows * 2.2) + 2.5

    # 2-row layout: histograms on top, waveforms + bar chart on bottom
    fig = plt.figure(figsize=(figsize[0], fig_height), constrained_layout=False)

    gs_main = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[n_hist_rows * 1.5, 1],
                                 hspace=0.25, top=0.95, bottom=0.06)

    # Panel (a): Histograms on top
    gs_histograms = gridspec.GridSpecFromSubplotSpec(n_hist_rows, n_hist_cols,
                                                      subplot_spec=gs_main[0],
                                                      wspace=0.35, hspace=0.5)

    # Bottom row: waveforms (left ~2/3) + bar chart (right ~1/3)
    gs_bottom = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_main[1],
                                                  width_ratios=[2, 1], wspace=0.15)

    # Panel (b): Waveforms (left, 2/3 width)
    n_waveform_cols = 5 if n_unit_types > 4 else 4
    gs_waveforms = gridspec.GridSpecFromSubplotSpec(1, n_waveform_cols, subplot_spec=gs_bottom[0],
                                                     wspace=0.1)

    # Panel (c): Bar chart (right, 1/3 width, vertically centered)
    gs_bar = gridspec.GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_bottom[1],
                                               height_ratios=[1, 2, 1])
    bar_ax = fig.add_subplot(gs_bar[1])  # Middle row

    axes = {
        'waveforms': [fig.add_subplot(gs_waveforms[i]) for i in range(n_unit_types)],
        'histograms': [fig.add_subplot(gs_histograms[i // n_hist_cols, i % n_hist_cols])
                       for i in range(n_histogram_metrics)],
        'bar_chart': bar_ax
    }

    return fig, axes


def _plot_supp_waveforms_panel(
    axes: List[matplotlib.axes.Axes],
    quality_metrics: dict,
    template_waveforms: np.ndarray,
    unit_type: np.ndarray,
    param: dict,
    colors: dict
):
    """Plot overlaid waveforms for each unit type in the supplementary figure."""
    unique_templates = param.get('unique_templates', np.arange(len(unit_type)))

    # Set labels based on param["splitGoodAndMua_NonSomatic"]
    if param.get("splitGoodAndMua_NonSomatic", False):
        labels = {
            0: ("NOISE", "Noise"),
            1: ("GOOD", "Good"),
            2: ("MUA", "MUA"),
            3: ("NON-SOMA GOOD", "Non-somatic Good"),
            4: ("NON-SOMA MUA", "Non-somatic MUA")
        }
    else:
        labels = {
            0: ("NOISE", "Noise"),
            1: ("GOOD", "Good"),
            2: ("MUA", "MUA"),
            3: ("NON-SOMA", "Non-somatic")
        }

    for idx, (unit_code, (cat_key, display_label)) in enumerate(labels.items()):
        if idx >= len(axes):
            break
        ax = axes[idx]

        # Get units of this type
        unit_mask = unit_type == unit_code
        unit_ids = unique_templates[unit_mask]
        n_units = len(unit_ids)

        color = colors.get(cat_key, 'black')

        if n_units > 0:
            # Set alpha inversely proportional to n_units
            alpha = max(0.03, min(0.3, 15 / n_units))

            for unit_id in unit_ids:
                if unit_id < len(quality_metrics.get("maxChannels", [])):
                    max_ch = int(quality_metrics["maxChannels"][unit_id])
                    if unit_id < template_waveforms.shape[0] and max_ch < template_waveforms.shape[2]:
                        waveform = template_waveforms[unit_id, :, max_ch]
                        ax.plot(waveform, color=color, alpha=alpha, linewidth=0.8)

        # Clean axis styling
        ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{display_label}\n(n = {n_units})", fontsize=10, fontweight='bold')

    # Add panel label
    axes[0].text(-0.15, 1.15, 'b', transform=axes[0].transAxes,
                 fontsize=14, fontweight='bold', va='top')


def _plot_supp_histograms_panel(
    axes: List[matplotlib.axes.Axes],
    quality_metrics: dict,
    param: dict,
    metric_names: List[str]
):
    """Plot histograms of quality metrics for the supplementary figure.

    Uses the existing generate_histogram function for consistent styling
    with threshold lines and color-coded pass/fail regions.
    """
    from .plotting_utils import get_metric_info_dict, get_color_from_matrix

    # Get metric info to check which metrics are valid
    metric_info_dict = get_metric_info_dict(param, quality_metrics)

    for idx, metric_name in enumerate(metric_names):
        if idx >= len(axes):
            break
        ax = axes[idx]

        if metric_name not in metric_info_dict:
            ax.set_visible(False)
            continue

        bar_color = get_color_from_matrix(idx)

        # Use the existing generate_histogram function for consistent styling
        include_y_label = (idx == 0)
        generate_histogram(metric_name, quality_metrics, param, bar_color, ax, include_y_label)

    # Add panel label
    if len(axes) > 0:
        axes[0].text(-0.25, 1.15, 'a', transform=axes[0].transAxes,
                     fontsize=14, fontweight='bold', va='top')


def _plot_supp_proportions_bar_chart(
    ax: matplotlib.axes.Axes,
    stats: dict,
    colors: dict,
    split_nonsomatic: bool
):
    """
    Plot bar chart showing mean +/- SE unit type proportions.

    This creates a compact, clean bar chart with error bars.
    """
    if split_nonsomatic:
        categories = ['GOOD', 'MUA', 'NOISE', 'NON-SOMA GOOD', 'NON-SOMA MUA']
        display_labels = ['Good', 'MUA', 'Noise', 'Non-soma\nGood', 'Non-soma\nMUA']
    else:
        categories = ['GOOD', 'MUA', 'NOISE', 'NON-SOMA']
        display_labels = ['Good', 'MUA', 'Noise', 'Non-somatic']

    x_positions = np.arange(len(categories))
    means = [stats[cat]['mean'] * 100 for cat in categories]  # Convert to percentage
    ses = [stats[cat]['se'] * 100 for cat in categories]
    bar_colors = [colors.get(cat, 'gray') for cat in categories]

    # Create compact bars with thinner width
    bar_width = 0.5
    bars = ax.bar(x_positions, means, width=bar_width, yerr=ses, capsize=3,
                  color=bar_colors, edgecolor='white', linewidth=0.5,
                  error_kw={'elinewidth': 1, 'capthick': 1, 'color': 'black'})

    # Clean, minimal styling
    ax.set_xticks(x_positions)
    ax.set_xticklabels(display_labels, fontsize=9)
    ax.set_ylabel('% units', fontsize=9)
    max_val = max(means) + max(ses) if ses else max(means)
    ax.set_ylim(0, max_val * 1.15)
    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.tick_params(axis='x', length=0)  # Hide x-axis ticks
    ax.tick_params(axis='y', labelsize=8)

    # Subtle n_datasets annotation
    n_datasets = stats['_meta']['n_datasets']
    ax.text(0.98, 0.92, f'n={n_datasets}',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color='gray')

    # Panel label
    ax.text(-0.08, 1.08, 'c', transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top')


def _generate_supp_methods_text(
    param: dict,
    stats: dict,
    quality_metrics: dict = None
) -> str:
    """
    Generate methods section text with actual numbers from the analysis.

    Parameters
    ----------
    param : dict
        BombCell parameters
    stats : dict
        Aggregated statistics from _aggregate_dataset_statistics
    quality_metrics : dict, optional
        Quality metrics for the example dataset

    Returns
    -------
    methods_text : str
        Ready-to-use methods section text
    """
    from .methods_text import generate_methods_text

    # Get base methods text
    base_text, references, _ = generate_methods_text(param, quality_metrics, citation_style='inline')

    # Add dataset-specific statistics
    meta = stats['_meta']
    n_datasets = meta['n_datasets']
    total_units = meta['total_units']

    # Build summary statistics paragraph
    summary_lines = [
        f"\n\n--- Summary Statistics ---\n",
        f"Across {n_datasets} recording session{'s' if n_datasets > 1 else ''} "
        f"({total_units:,} total units), the following proportions were observed "
        f"(mean +/- SE):"
    ]

    # Determine categories based on param
    if param.get("splitGoodAndMua_NonSomatic", False):
        categories = [
            ('GOOD', 'good single units'),
            ('MUA', 'multi-unit activity'),
            ('NOISE', 'noise'),
            ('NON-SOMA GOOD', 'non-somatic good units'),
            ('NON-SOMA MUA', 'non-somatic MUA')
        ]
    else:
        categories = [
            ('GOOD', 'good single units'),
            ('MUA', 'multi-unit activity'),
            ('NOISE', 'noise'),
            ('NON-SOMA', 'non-somatic units')
        ]

    for cat_key, cat_label in categories:
        if cat_key in stats:
            mean_pct = stats[cat_key]['mean'] * 100
            se_pct = stats[cat_key]['se'] * 100
            total_count = stats[cat_key]['total_count']
            summary_lines.append(
                f"  - {cat_label}: {mean_pct:.1f} +/- {se_pct:.1f}% "
                f"({total_count:,} units total)"
            )

    summary_text = '\n'.join(summary_lines)

    # Add references section
    ref_text = "\n\nReferences:\n" + "\n".join(f"  - {ref}" for ref in references)

    # Combine
    full_methods = base_text + summary_text + ref_text

    return full_methods


def generate_supplementary_figure(
    example_dataset_path: str,
    all_dataset_paths: List[str],
    param: Optional[dict] = None,
    figsize: Tuple[float, float] = (14, 10),
    dpi: int = 300,
    save_path: Optional[str] = None,
    histogram_metrics: Optional[List[str]] = None,
    color_scheme: Optional[dict] = None,
    show_legend: bool = True,
) -> Tuple[matplotlib.figure.Figure, str]:
    """
    Generate a publication-ready supplementary figure for BombCell quality control.

    Creates a multi-panel figure with:
    (a) Histograms of quality metric distributions for the example dataset
    (b) Overlaid waveforms by unit type for an example dataset (left, 2/3 width)
    (c) Bar chart showing mean +/- SE unit type proportions across all datasets (right, 1/3 width)

    Also generates a methods section text with actual statistics.

    Parameters
    ----------
    example_dataset_path : str
        Path to the BombCell results directory for the example dataset.
        This dataset is used for panels (a) and (b).
    all_dataset_paths : List[str]
        List of paths to BombCell results directories for all datasets.
        These are used to compute statistics in panel (c) and methods text.
    param : dict, optional
        BombCell parameter dictionary. If None, loads from example_dataset_path.
    figsize : tuple, optional
        Figure size in inches (width, height). Default (14, 10).
    dpi : int, optional
        Resolution for saved figure. Default 300.
    save_path : str, optional
        If provided, saves figure to this path (PNG format).
    histogram_metrics : List[str], optional
        List of metric names to include in histograms. If None, uses defaults:
        nPeaks, waveformDuration_peakTrough, fractionRPVs_estimatedTauR,
        presenceRatio, percentageSpikesMissing_gaussian, signalToNoiseRatio
    color_scheme : dict, optional
        Custom colors for unit types. Keys should be unit type names
        (e.g., 'GOOD', 'MUA', 'NOISE', 'NON-SOMA').
        If None, uses default BombCell colors.
    show_legend : bool, optional
        Whether to include a legend. Default True.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated multi-panel figure.
    methods_text : str
        Ready-to-use methods section text with actual statistics.

    Examples
    --------
    >>> from bombcell import generate_supplementary_figure
    >>>
    >>> # Single example dataset for detailed panels
    >>> example_path = "/path/to/recording1/bombcell"
    >>>
    >>> # All datasets for statistics
    >>> all_paths = [
    ...     "/path/to/recording1/bombcell",
    ...     "/path/to/recording2/bombcell",
    ...     "/path/to/recording3/bombcell",
    ... ]
    >>>
    >>> # Generate figure and methods text
    >>> fig, methods_text = generate_supplementary_figure(
    ...     example_dataset_path=example_path,
    ...     all_dataset_paths=all_paths,
    ...     save_path="supplementary_figure_qc.png"
    ... )
    >>>
    >>> # Print methods section
    >>> print(methods_text)
    """
    # 1. Load example dataset
    if param is None:
        param, quality_metrics, template_waveforms, unit_type, unit_type_string = \
            _load_single_dataset_for_supp(example_dataset_path)
    else:
        _, quality_metrics, template_waveforms, unit_type, unit_type_string = \
            _load_single_dataset_for_supp(example_dataset_path)
        # Ensure unique_templates is set
        if 'unique_templates' not in param:
            if 'phy_clusterID' in quality_metrics:
                param['unique_templates'] = np.array(quality_metrics['phy_clusterID']).astype(int)
            else:
                param['unique_templates'] = np.arange(len(unit_type))

    # 2. Aggregate statistics across all datasets
    split_nonsomatic = param.get("splitGoodAndMua_NonSomatic", False)
    stats = _aggregate_dataset_statistics(all_dataset_paths, split_nonsomatic)

    # 3. Set up colors
    colors = color_scheme if color_scheme else DEFAULT_UNIT_COLORS

    # 4. Determine layout parameters
    n_unit_types = 5 if split_nonsomatic else 4

    if histogram_metrics is None:
        # Use the same logic as plot_histograms to get ALL valid metrics
        from .plotting_utils import get_metric_info_list
        metric_info = get_metric_info_list(param, quality_metrics)
        valid_metric_info = [mi for mi in metric_info if mi.plot_condition and (mi.name in quality_metrics)]
        histogram_metrics = [mi.name for mi in valid_metric_info]

    # Ensure we have at least some metrics
    if len(histogram_metrics) == 0:
        histogram_metrics = [k for k in quality_metrics.keys()
                           if not k.startswith('_') and k not in ['phy_clusterID', 'maxChannels']]

    n_histogram_metrics = len(histogram_metrics)

    # 5. Create figure layout
    fig, axes = _create_supp_figure_layout(n_unit_types, n_histogram_metrics, figsize)

    # 6. Plot panels
    _plot_supp_waveforms_panel(axes['waveforms'], quality_metrics, template_waveforms,
                               unit_type, param, colors)

    _plot_supp_histograms_panel(axes['histograms'], quality_metrics, param, histogram_metrics)

    _plot_supp_proportions_bar_chart(axes['bar_chart'], stats, colors, split_nonsomatic)

    # 7. Add legend if requested
    if show_legend:
        if split_nonsomatic:
            legend_items = [
                ('Good', colors.get('GOOD', '#228B22')),
                ('MUA', colors.get('MUA', '#DAA520')),
                ('Noise', colors.get('NOISE', '#8B0000')),
                ('Non-soma Good', colors.get('NON-SOMA GOOD', '#4169E1')),
                ('Non-soma MUA', colors.get('NON-SOMA MUA', '#87CEEB')),
            ]
        else:
            legend_items = [
                ('Good', colors.get('GOOD', '#228B22')),
                ('MUA', colors.get('MUA', '#DAA520')),
                ('Noise', colors.get('NOISE', '#8B0000')),
                ('Non-somatic', colors.get('NON-SOMA', '#4169E1')),
            ]

        from matplotlib.patches import Patch
        legend_handles = [Patch(facecolor=color, edgecolor='black', label=label)
                         for label, color in legend_items]
        fig.legend(handles=legend_handles, loc='upper right',
                   bbox_to_anchor=(0.98, 0.98), fontsize=9, framealpha=0.9)

    # 8. Generate methods text
    methods_text = _generate_supp_methods_text(param, stats, quality_metrics)

    # 9. Save if requested
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved supplementary figure to {save_path}")

        # Also save methods text
        methods_path = save_path.with_suffix('.txt')
        methods_path.write_text(methods_text)
        print(f"Saved methods text to {methods_path}")

    return fig, methods_text