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
        plot_histograms(quality_metrics, param)

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
 

def plot_histograms(quality_metrics, param):
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

    # Create copies to avoid SettingWithCopyWarning
    if 'peak1ToPeak2Ratio' in quality_metrics:
        quality_metrics['peak1ToPeak2Ratio'] = np.where(
            quality_metrics['peak1ToPeak2Ratio'] == np.inf, 
            np.nan, 
            quality_metrics['peak1ToPeak2Ratio']
        )
    if 'troughToPeak2Ratio' in quality_metrics:
        quality_metrics['troughToPeak2Ratio'] = np.where(
            quality_metrics['troughToPeak2Ratio'] == np.inf, 
            np.nan, 
            quality_metrics['troughToPeak2Ratio']
        )

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

    for idx, vmi in enumerate(valid_metric_info):
        row_id = idx // num_cols
        col_id = idx % num_cols
        ax = axs[row_id, col_id]

        # get color from color matrix (use modulus operator for wraparound)
        color = get_color_from_matrix(idx)
        
        metric_data = quality_metrics[vmi.name]
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
                                         color=color, alpha=0.7)
            
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
            
            thresh1 = vmi.threshold_1
            thresh2 = vmi.threshold_2
            line_colors = vmi.line_colors.reshape(3, 3)
            
            if vmi.name in ['nPeaks', 'nTroughs']:
                binsize_offset = 0.5
            else:
                binsize_offset = (bins_out[1] - bins_out[0]) / 2 if len(bins_out) > 1 else 0
            
            if thresh1 is not None or thresh2 is not None:
                if thresh1 is not None and thresh2 is not None:
                    # Add vertical lines for thresholds at value + 0.5*bin_width
                    ax.axvline(thresh1 + binsize_offset, color='k', linewidth=2)
                    ax.axvline(thresh2 + binsize_offset, color='k', linewidth=2)
                    # Add horizontal colored lines at value + 0.5*bin_width
                    ax.plot([x_lim[0], thresh1 + binsize_offset], 
                           [line_y, line_y], color=line_colors[0], linewidth=6)
                    ax.plot([thresh1 + binsize_offset, thresh2 + binsize_offset], 
                           [line_y, line_y], color=line_colors[1], linewidth=6)
                    ax.plot([thresh2 + binsize_offset, x_lim[1]], 
                           [line_y, line_y], color=line_colors[2], linewidth=6)
                    
                    # Add classification labels with arrows
                    midpoint1 = (x_lim[0] + thresh1) / 2
                    midpoint2 = (thresh1 + thresh2) / 2
                    midpoint3 = (thresh2 + x_lim[1]) / 2
                    text_y = 0.95  # Position text at 0.95
                    
                    # Determine metric type based on metric name
                    noise_metrics = ['nPeaks', 'nTroughs', 'waveformBaselineFlatness', 'waveformDuration_peakTrough', 'scndPeakToTroughRatio', 'spatialDecaySlope']
                    nonsomatic_metrics = ['peak1ToPeak2Ratio', 'mainPeakToTroughRatio']

                    if vmi.name in noise_metrics:
                        # Noise metrics: both thresholds -> Noise, Neuronal, Noise
                        ax.text(midpoint1, text_y, '  Noise  ', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '  Neuronal  ', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                        ax.text(midpoint3, text_y, '  Noise  ', ha='center', fontsize=10, 
                               color=line_colors[2], weight='bold')
                    elif vmi.name in nonsomatic_metrics:
                        # Non-somatic metrics: both thresholds -> Non-somatic, Somatic, Non-somatic
                        ax.text(midpoint1, text_y, '  Non-somatic  ', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '  Somatic  ', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                        ax.text(midpoint3, text_y, '  Non-somatic  ', ha='center', fontsize=10, 
                               color=line_colors[2], weight='bold')
                    else:
                        # MUA metrics: both thresholds -> MUA, Good, MUA
                        ax.text(midpoint1, text_y, '  MUA  ', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '  Good  ', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                        ax.text(midpoint3, text_y, '  MUA  ', ha='center', fontsize=10, 
                               color=line_colors[2], weight='bold')
                    
                elif thresh1 is not None:
                    # Add vertical line for threshold at value + 0.5*bin_width
                    ax.axvline(thresh1 + binsize_offset, color='k', linewidth=2)
                    # Add horizontal colored lines at value + 0.5*bin_width
                    ax.plot([x_lim[0], thresh1 + binsize_offset], 
                           [line_y, line_y], color=line_colors[0], linewidth=6)
                    ax.plot([thresh1 + binsize_offset, x_lim[1]], 
                           [line_y, line_y], color=line_colors[1], linewidth=6)
                    
                    # Add classification labels for single threshold
                    midpoint1 = (x_lim[0] + thresh1) / 2
                    midpoint2 = (thresh1 + x_lim[1]) / 2
                    text_y = 0.95  # Position text at 0.95
                    
                    # Determine metric type based on metric name
                    noise_metrics = ['nPeaks', 'nTroughs', 'waveformBaselineFlatness', 'waveformDuration_peakTrough', 'scndPeakToTroughRatio', 'spatialDecaySlope']
                    nonsomatic_metrics = ['peak1ToPeak2Ratio', 'mainPeakToTroughRatio']
                    
                    if vmi.name in noise_metrics:
                        # Noise metrics: thresh1 only -> Neuronal, Noise
                        ax.text(midpoint1, text_y, '  Neuronal  ', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '  Noise  ', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                    elif vmi.name in nonsomatic_metrics:
                        # Non-somatic metrics: thresh1 only -> Somatic, Non-somatic
                        ax.text(midpoint1, text_y, '  Somatic  ', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '  Non-somatic  ', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                    else:
                        # MUA metrics: thresh1 only
                        if vmi.name in ['isolationDistance', 'rawAmplitude']:
                            # For isolation distance and rawAmplitude: MUA on left, Good on right
                            ax.text(midpoint1, text_y, '  MUA  ', ha='center', fontsize=10, 
                                   color=line_colors[0], weight='bold')
                            ax.text(midpoint2, text_y, '  Good  ', ha='center', fontsize=10, 
                                   color=line_colors[1], weight='bold')
                        else:
                            # For other MUA metrics: Good on left, MUA on right
                            ax.text(midpoint1, text_y, '  Good  ', ha='center', fontsize=10, 
                                   color=line_colors[0], weight='bold')
                            ax.text(midpoint2, text_y, '  MUA  ', ha='center', fontsize=10, 
                                   color=line_colors[1], weight='bold')
                    
                elif thresh2 is not None:
                    # Add vertical line for threshold at value + 0.5*bin_width
                    ax.axvline(thresh2 + binsize_offset, color='k', linewidth=2)
                    # Add horizontal colored lines at value + 0.5*bin_width
                    ax.plot([x_lim[0], thresh2 + binsize_offset], 
                           [line_y, line_y], color=line_colors[0], linewidth=6)
                    ax.plot([thresh2 + binsize_offset, x_lim[1]], 
                           [line_y, line_y], color=line_colors[1], linewidth=6)
                    
                    # Add classification labels for threshold 2 only
                    midpoint1 = (x_lim[0] + thresh2) / 2
                    midpoint2 = (thresh2 + x_lim[1]) / 2
                    text_y = 0.95  # Position text at 0.95
                    
                    # Determine metric type based on metric name
                    noise_metrics = ['nPeaks', 'nTroughs', 'waveformBaselineFlatness', 'waveformDuration_peakTrough', 'scndPeakToTroughRatio', 'spatialDecaySlope']
                    nonsomatic_metrics = ['peak1ToPeak2Ratio', 'mainPeakToTroughRatio']
                    
                    if vmi.name in noise_metrics:
                        # Noise metrics: thresh2 only -> Noise, Neuronal
                        ax.text(midpoint1, text_y, '  Noise  ', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '  Neuronal  ', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                    elif vmi.name in nonsomatic_metrics:
                        # Non-somatic metrics: thresh2 only -> Non-somatic, Somatic
                        ax.text(midpoint1, text_y, '  Non-somatic  ', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '  Somatic  ', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                    else:
                        # MUA metrics: thresh2 only
                        if vmi.name in ['nSpikes', 'presenceRatio', 'signalToNoiseRatio']:
                            # For nSpikes, presenceRatio, and signalToNoiseRatio: MUA on left, Good on right
                            ax.text(midpoint1, text_y, '  MUA  ', ha='center', fontsize=10, 
                                   color=line_colors[0], weight='bold')
                            ax.text(midpoint2, text_y, '  Good  ', ha='center', fontsize=10, 
                                   color=line_colors[1], weight='bold')
                        else:
                            # For L-ratio: Good on left, MUA on right
                            ax.text(midpoint1, text_y, '  Good  ', ha='center', fontsize=10, 
                                   color=line_colors[0], weight='bold')
                            ax.text(midpoint2, text_y, '  MUA  ', ha='center', fontsize=10, 
                                   color=line_colors[1], weight='bold')

            # Set histogram limits from 0 to 1
            ax.set_ylim([0, 1])
            
        ax.set_xlabel(vmi.short_name, fontsize=13)
        if idx == 0:
            ax.set_ylabel('frac. units', fontsize=13)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize=12)

    # Hide unused subplots
    for i in range(len(valid_metric_info), num_rows * num_cols):
        row_id = i // num_cols
        col_id = i % num_cols
        axs[row_id, col_id].set_visible(False)

    plt.tight_layout()
    plt.show()