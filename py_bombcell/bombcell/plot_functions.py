import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
try:
    from upsetplot import UpSet, from_indicators
    UPSETPLOT_AVAILABLE = True
except ImportError:
    UPSETPLOT_AVAILABLE = False
    print("Warning: upsetplot not available. Some plotting functions may not work.")

from bombcell import helper_functions as hf

#V0.9 of upset plots have pandas future warning so, suppressing them
import warnings

from collections import namedtuple


def plot_summary_data(quality_metrics, template_waveforms, unit_type, unit_type_string, param):
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
    """
    if param["plotGlobal"]:
        plot_waveforms_overlay(quality_metrics, template_waveforms, unit_type, param) 
        upset_plots(quality_metrics, unit_type_string, param)
        plot_all_histograms(quality_metrics, param)

def generate_upset_plot(qm_table: pd.DataFrame, unit_type_str: str):
    try:
        # ensure upper case
        unit_type_str = unit_type_str.upper()

        # get metrics relevant to chosen unit type
        if unit_type_str=="NOISE":
            unit_type_metrics = ["# peaks", "# troughs", "waveform duration", "spatial decay", "baseline flatness", "peak2 / trough"] #Duration is peak to trough duration
        elif unit_type_str=="NON-SOMATIC":
            unit_type_metrics = ["trough / peak2", "peak1 / peak2"]
        elif unit_type_str=="MUA":
            unit_type_metrics = ["SNR", "amplitude", "presence ratio", "# spikes", "% spikes missing", "fraction RPVs", "max. drift", "isolation dist.", "L-ratio"]
        else:
            raise ValueError(f"Invalid unit type {unit_type_str} - allowed values are 'NOISE', 'NON-SOMATIC', 'MUA'")
        
        # filter out uncomputed metrics
        unit_type_metrics = [m for m in unit_type_metrics if m in qm_table.columns]

        # generate mask for the chosen unit type and filter the data from qm_table        
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
            upset.plot()
            plt.suptitle(f"Units classified as {unit_type_str.lower()} (n = {n_unit_type}/{n_total_units})")
            plt.show()
        elif n_unit_type > 0:
            print(f"{unit_type_str.capitalize()} upset plot skipped: no metrics have failures")
    except (AttributeError, ValueError) as e:
        print(f"Warning: Could not create noise upset plot due to library compatibility: {e}")


def upset_plots(quality_metrics, unit_type_string, param):
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
    """

    qm_table = hf.make_qm_table(quality_metrics, param, unit_type_string)
    
    generate_upset_plot(qm_table, "NOISE")
    generate_upset_plot(qm_table, "NON-SOMATIC")
    generate_upset_plot(qm_table, "MUA")

def plot_waveforms_overlay(quality_metrics, template_waveforms, unit_type, param):
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
    
    n_categories = np.unique(unit_type).size
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
        unit_type_template_ids = unique_templates[unit_type==plot_idx]
        n_units_of_type = unit_type_template_ids.size
        ax = axs[img_pos[plot_idx][0]][img_pos[plot_idx][1]]

        # if the current unit type has more than 0 units, generate a plot
        if n_units_of_type > 0:
            for template_id in unit_type_template_ids:
                max_channel_id = quality_metrics["maxChannels"][template_id]
                ax.plot(template_waveforms[template_id, 0:, max_channel_id], color="black", alpha=0.1)
                ax.spines[["right", "top", "bottom", "left"]].set_visible(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{labels[plot_idx]} units (n = {n_units_of_type})")

        # if the current unit type has no units, or if this is an "extra" plot, do this instead
        elif (n_units_of_type==0) or (plot_idx==n_plots):
            ax.spines[["right", "top", "bottom", "left"]].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            # if it's not an "extra" plot, add a title signifying 0 units
            if plot_idx < n_plots:
                ax.set_title(f"No {labels[plot_idx]} units (n = 0)")
 

def plot_all_histograms(quality_metrics, param):
    """
    This function find what metrics have been extracted and plots histograms for each metric

    Parameters
    ----------
    quality_metrics : dict
        The dictionary containing all quality metrics
    param : dict
        The dictionary of all bomcell parameters
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
        plot_histogram(vmi.name, quality_metrics, param, bar_color, ax, include_y_label)

    # Hide unused subplots
    for i in range(len(valid_metric_info), num_rows * num_cols):
        row_id = i // num_cols
        col_id = i % num_cols
        axs[row_id, col_id].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_histogram(metric_name, quality_metrics, param, bar_color=None, ax=None, include_y_label=False):

    if ax is None:
        fig, ax = plt.subplots(1,1)

    if bar_color is None:
        bar_color='k'

    from .plotting_utils import get_metric_info_dict

    vmi = get_metric_info_dict(param, quality_metrics)[metric_name]
    metric_data = quality_metrics[vmi.name]
    metric_data = np.where(metric_data==np.inf, np.nan, metric_data) # previously, this was done explicitly for peak1ToPeak2Ratio, troughToPeak2Ratio
    
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