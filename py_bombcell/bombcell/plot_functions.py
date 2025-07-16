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

    noise_metrics = ["# peaks", "# troughs", "waveform duration", "spatial decay", "baseline flatness", "peak2 / trough"] #Duration is peak to trough duration
    non_somatic_metrics = ["trough / peak2", "peak1 / peak2"]
    mua_metrics = ["SNR", "amplitude", "presence ratio", "# spikes", "% spikes missing", "fraction RPVs", "max. drift", "isolation dist.", "L-ratio"]
    
    # Create display names mapping for better plot labels
    display_names = {
        "SNR": "signal/noise (SNR)",
        "fraction RPVs": "refractory period viol. (RPV)",
        "amplitude": "amplitude"
    }

    # Eventually filter out uncomputed metrics
    noise_metrics = [m for m in noise_metrics if m in qm_table.columns]
    non_somatic_metrics = [m for m in non_somatic_metrics if m in qm_table.columns]
    mua_metrics = [m for m in mua_metrics if m in qm_table.columns]

    # Get total number of units
    total_units = len(qm_table)
    
    # Plot upset plots with error handling for library compatibility
    try:
        # plot noise metrics upset plot - only include NOISE units
        noise_units_mask = qm_table['unit_type'] == 'NOISE'
        noise_data = qm_table.loc[noise_units_mask, noise_metrics].astype(bool)
        n_noise = noise_units_mask.sum()
        # Check how many columns have True values
        cols_with_true = (noise_data.sum() > 0).sum()
        if cols_with_true >= 1 and n_noise > 0:
            upset = UpSet(from_indicators(noise_metrics, data=noise_data), min_degree=1)
            upset.plot()
            plt.suptitle(f"Units classified as noise (n = {n_noise}/{total_units})")
            plt.show()
        elif n_noise > 0:
            print(f"Noise upset plot skipped: no metrics have failures")
    except (AttributeError, ValueError) as e:
        print(f"Warning: Could not create noise upset plot due to library compatibility: {e}")

    try:
        # plot non-somatic metrics upset plot - only include NON-SOMA units
        non_somatic_units_mask = qm_table['unit_type'].str.startswith('NON-SOMA')
        non_somatic_data = qm_table.loc[non_somatic_units_mask, non_somatic_metrics].astype(bool)
        n_non_somatic = non_somatic_units_mask.sum()
        # Check how many columns have True values
        cols_with_true = (non_somatic_data.sum() > 0).sum()
        if cols_with_true >= 1 and n_non_somatic > 0:
            upset = UpSet(from_indicators(non_somatic_metrics, data=non_somatic_data), min_degree=1)
            upset.plot()
            plt.suptitle(f"Units classified as non-somatic (n = {n_non_somatic}/{total_units})")
            plt.show()
        elif n_non_somatic > 0:
            print(f"Non-somatic upset plot skipped: no metrics have failures")
    except (AttributeError, ValueError) as e:
        print(f"Warning: Could not create non-somatic upset plot due to library compatibility: {e}")

    try:
        # plot MUA metrics upset plot - only include MUA units
        mua_units_mask = qm_table['unit_type'] == 'MUA'
        mua_data = qm_table.loc[mua_units_mask, mua_metrics].astype(bool).copy()
        # Rename columns for better display
        mua_display_names = [display_names.get(m, m) for m in mua_metrics]
        mua_data.columns = mua_display_names
        
        n_mua = mua_units_mask.sum()
        # Check how many columns have True values
        cols_with_true = (mua_data.sum() > 0).sum()
        if cols_with_true >= 1 and n_mua > 0:
            upset = UpSet(from_indicators(mua_display_names, data=mua_data), min_degree=1)
            upset.plot()
            plt.suptitle(f"Units classified as MUA (n = {n_mua}/{total_units})")
            plt.show()
        elif n_mua > 0:
            print(f"MUA upset plot skipped: no metrics have failures")
    except (AttributeError, ValueError) as e:
        print(f"Warning: Could not create MUA upset plot due to library compatibility: {e}")

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

    n_categories = np.unique(unit_type).size
    labels = {0: "noise", 1: "somatic, good", 2: "somatic, MUA", 
                3: "non-somatic, good" if param["splitGoodAndMua_NonSomatic"] else "non-somatic",
                4: "non-somatic, MUA", 5: ""}
    #TODO change alpha to be inversly proprotional to n units
    if n_categories < 5:
        fig, axs = plt.subplots(nrows = 2, ncols=2)
        img_pos = [[0,0], [0,1], [1,0], [1,1]]
        for i in range(4):
            og_id = unique_templates[unit_type == i]
            n_units_in_cat = og_id.size
            if n_units_in_cat !=0:
                for id in og_id:
                    axs[img_pos[i][0]][img_pos[i][1]].plot(template_waveforms[id, 0:, quality_metrics['maxChannels'][id]], color = 'black', alpha = 0.1)
                    axs[img_pos[i][0]][img_pos[i][1]].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
                    axs[img_pos[i][0]][img_pos[i][1]].set_xticks([])
                    axs[img_pos[i][0]][img_pos[i][1]].set_yticks([])
                    axs[img_pos[i][0]][img_pos[i][1]].set_title(f"{labels[i]} units (n = {n_units_in_cat})")
            else:
                axs[img_pos[i][0]][img_pos[i][1]].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
                axs[img_pos[i][0]][img_pos[i][1]].set_xticks([])
                axs[img_pos[i][0]][img_pos[i][1]].set_yticks([])
                axs[img_pos[i][0]][img_pos[i][1]].set_title(f"No {labels[i]} units (n = 0)")

    elif n_categories == 5:
        fig, axs = plt.subplots(nrows = 3, ncols=2)
        img_pos = [[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]]
        for i in range(6):
            og_id = unique_templates[np.argwhere(unit_type == i).squeeze()]
            n_units_in_cat = og_id.size
            if n_units_in_cat !=0:
                for id in og_id:
                    axs[img_pos[i][0]][img_pos[i][1]].plot(template_waveforms[id, 0:, quality_metrics['maxChannels'][id]], color = 'black', alpha = 0.1)
                    axs[img_pos[i][0]][img_pos[i][1]].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
                    axs[img_pos[i][0]][img_pos[i][1]].set_xticks([])
                    axs[img_pos[i][0]][img_pos[i][1]].set_yticks([])
                    axs[img_pos[i][0]][img_pos[i][1]].set_title(f"{labels[i]} units (n = {n_units_in_cat})")
            else:
                if i == 5:
                    axs[img_pos[i][0]][img_pos[i][1]].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
                    axs[img_pos[i][0]][img_pos[i][1]].set_xticks([])
                    axs[img_pos[i][0]][img_pos[i][1]].set_yticks([])
                else:
                    axs[img_pos[i][0]][img_pos[i][1]].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
                    axs[img_pos[i][0]][img_pos[i][1]].set_xticks([])
                    axs[img_pos[i][0]][img_pos[i][1]].set_yticks([])
                    axs[img_pos[i][0]][img_pos[i][1]].set_title(f"No {labels[i]} units (n = 0)")

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