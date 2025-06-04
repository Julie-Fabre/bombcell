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
    non_somatic_metrics = ["peak(main) / trough", "peak1 / peak2"]
    mua_metrics = ["SNR", "amplitude", "# spikes", "presence ratio", "% spikes missing", "fraction RPVs", "max. drift", "isolation dist.", "L-Ratio"]

    # Eventually filter out uncomputed metrics
    noise_metrics = [m for m in noise_metrics if m in qm_table.columns]
    non_somatic_metrics = [m for m in non_somatic_metrics if m in qm_table.columns]
    mua_metrics = [m for m in mua_metrics if m in qm_table.columns]

    # Plot upset plots with error handling for library compatibility
    try:
        # plot noise metrics upset plot
        noise_data = qm_table[noise_metrics].astype(bool)
        if len(noise_metrics) > 1:
            upset = UpSet(from_indicators(noise_metrics, data=noise_data), min_degree=1)
            upset.plot()
            plt.suptitle("Units classified as noise")
            plt.show()
    except (AttributeError, ValueError) as e:
        print(f"Warning: Could not create noise upset plot due to library compatibility: {e}")

    try:
        # plot non-somatic metrics upset plot  
        non_somatic_data = qm_table[non_somatic_metrics].astype(bool)
        if len(non_somatic_metrics) > 1:
            upset = UpSet(from_indicators(non_somatic_metrics, data=non_somatic_data), min_degree=1)
            upset.plot()
            plt.suptitle("Units classified as non-somatic")
            plt.show()
    except (AttributeError, ValueError) as e:
        print(f"Warning: Could not create non-somatic upset plot due to library compatibility: {e}")

    try:
        # plot MUA metrics upset plot
        mua_data = qm_table[mua_metrics].astype(bool)
        if len(mua_metrics) > 1:
            upset = UpSet(from_indicators(mua_metrics, data=mua_data), min_degree=1)
            upset.plot()
            plt.suptitle("Units classified as MUA")
            plt.show()
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
    labels = {0: "NOISE", 1: "GOOD", 2: "MUA", 
                3: "NON-SOMA GOOD" if param["splitGoodAndMua_NonSomatic"] else "NON-SOMA",
                4: "NON-SOMA MUA", 5: ""}
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
                    axs[img_pos[i][0]][img_pos[i][1]].set_title(f"{labels[i]} Units")
            else:
                axs[img_pos[i][0]][img_pos[i][1]].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
                axs[img_pos[i][0]][img_pos[i][1]].set_xticks([])
                axs[img_pos[i][0]][img_pos[i][1]].set_yticks([])
                axs[img_pos[i][0]][img_pos[i][1]].set_title(f"No {labels[i]} Units")

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
                    axs[img_pos[i][0]][img_pos[i][1]].set_title(f"{labels[i]} Units")
            else:
                if i == 5:
                    axs[img_pos[i][0]][img_pos[i][1]].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
                    axs[img_pos[i][0]][img_pos[i][1]].set_xticks([])
                    axs[img_pos[i][0]][img_pos[i][1]].set_yticks([])
                else:
                    axs[img_pos[i][0]][img_pos[i][1]].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
                    axs[img_pos[i][0]][img_pos[i][1]].set_xticks([])
                    axs[img_pos[i][0]][img_pos[i][1]].set_yticks([])
                    axs[img_pos[i][0]][img_pos[i][1]].set_title(f"No {labels[i]} Units")

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
    quality_metrics['peak1ToPeak2Ratio'][quality_metrics['peak1ToPeak2Ratio'] == np.inf] = np.nan
    quality_metrics['troughToPeak2Ratio'][quality_metrics['troughToPeak2Ratio'] == np.inf] = np.nan

    # Define MATLAB-style color matrices
    red_colors = np.array([
        [0.8627, 0.0784, 0.2353],  # Crimson
        [1.0000, 0.1412, 0.0000],  # Scarlet
        [0.7255, 0.0000, 0.0000],  # Cherry
        [0.5020, 0.0000, 0.1255],  # Burgundy
        [0.5020, 0.0000, 0.0000],  # Maroon
        [0.8039, 0.3608, 0.3608],  # Indian Red
    ])

    blue_colors = np.array([
        [0.2549, 0.4118, 0.8824],  # Royal Blue
        [0.0000, 0.0000, 0.5020],  # Navy Blue
    ])

    darker_yellow_orange_colors = np.array([
        [0.7843, 0.7843, 0.0000],  # Dark Yellow
        [0.8235, 0.6863, 0.0000],  # Dark Golden Yellow
        [0.8235, 0.5294, 0.0000],  # Dark Orange
        [0.8039, 0.4118, 0.3647],  # Dark Coral
        [0.8235, 0.3176, 0.2275],  # Dark Tangerine
        [0.8235, 0.6157, 0.6510],  # Dark Salmon
        [0.7882, 0.7137, 0.5765],  # Dark Goldenrod
        [0.8235, 0.5137, 0.3922],  # Dark Light Coral
        [0.7569, 0.6196, 0.0000],  # Darker Goldenrod
        [0.8235, 0.4510, 0.0000],  # Darker Orange
    ])

    color_mtx = np.vstack([red_colors, blue_colors, darker_yellow_orange_colors])

    # Define metrics in MATLAB order
    metric_names = ['nPeaks', 'nTroughs', 'waveformBaselineFlatness', 'waveformDuration_peakTrough', 
                   'scndPeakToTroughRatio', 'spatialDecaySlope', 'peak1ToPeak2Ratio', 'mainPeakToTroughRatio',
                   'rawAmplitude', 'signalToNoiseRatio', 'fractionRPVs_estimatedTauR', 'nSpikes', 
                   'presenceRatio', 'percentageSpikesMissing_gaussian', 'maxDriftEstimate', 
                   'isolationDistance', 'Lratio']

    metric_names_short = ['# peaks', '# troughs', 'baseline flatness', 'waveform duration',
                         'peak_2/trough', 'spatial decay', 'peak_1/peak_2', 'peak_{main}/trough',
                         'amplitude', 'SNR', 'frac. RPVs', '# spikes',
                         'presence ratio', '% spikes missing', 'maximum drift',
                         'isolation dist.', 'L-ratio']

    # Define thresholds
    metric_thresh1 = [param.get('maxNPeaks'), param.get('maxNTroughs'), param.get('maxWvBaselineFraction'),
                     param.get('minWvDuration'), param.get('maxScndPeakToTroughRatio_noise'),
                     param.get('minSpatialDecaySlope') if param.get('spDecayLinFit') else param.get('minSpatialDecaySlopeExp'),
                     param.get('maxPeak1ToPeak2Ratio_nonSomatic'), param.get('maxMainPeakToTroughRatio_nonSomatic'),
                     None, None, param.get('maxRPVviolations'), None, None, param.get('maxPercSpikesMissing'),
                     param.get('maxDrift'), param.get('isoDmin'), None]

    metric_thresh2 = [None, None, None, param.get('maxWvDuration'), None,
                     None if param.get('spDecayLinFit') else param.get('maxSpatialDecaySlopeExp'),
                     None, None, param.get('minAmplitude'), param.get('min_SNR'),
                     None, param.get('minNumSpikes'), param.get('minPresenceRatio'), None, None,
                     None, param.get('lratioMax')]

    # Define plot conditions
    plot_conditions = [True, True, True, True, True,
                      param.get('computeSpatialDecay', False),
                      True, True,
                      param.get('extractRaw', False) and np.all(~np.isnan(quality_metrics.get('rawAmplitude', [np.nan]))),
                      param.get('extractRaw', False) and np.all(~np.isnan(quality_metrics.get('signalToNoiseRatio', [np.nan]))),
                      True, True, True, True,
                      param.get('computeDrift', False),
                      param.get('computeDistanceMetrics', False),
                      param.get('computeDistanceMetrics', False)]

    # Define line colors for thresholds (MATLAB style)
    metric_line_cols = np.array([
        [0.2, 0.2, 0.2, 1, 0, 0, 0, 0, 0],  # nPeaks
        [0.2, 0.2, 0.2, 1, 0, 0, 0, 0, 0],  # nTroughs
        [0.2, 0.2, 0.2, 1, 0, 0, 1, 0, 0],  # baseline flatness
        [1, 0, 0, 0.2, 0.2, 0.2, 1, 0, 0],  # waveform duration
        [0.2, 0.2, 0.2, 1, 0, 0, 1, 0, 0],  # peak2/trough
        [1, 0, 0, 0.2, 0.2, 0.2, 1, 0, 0],  # spatial decay
        [0.2, 0.2, 0.2, 0.25, 0.41, 0.88, 0, 0, 0],  # peak1/peak2
        [0.2, 0.2, 0.2, 0.25, 0.41, 0.88, 0, 0, 0],  # peak_main/trough
        [1.0, 0.5469, 0, 0, 0.5, 0, 0, 0, 0],  # amplitude
        [1.0, 0.5469, 0, 0, 0.5, 0, 0, 0, 0],  # SNR
        [0, 0.5, 0, 1.0, 0.5469, 0, 0, 0, 0],  # frac RPVs
        [1.0, 0.5469, 0, 0, 0.5, 0, 0, 0, 0],  # nSpikes
        [1.0, 0.5469, 0, 0, 0.5, 0, 0, 0, 0],  # presence ratio
        [0, 0.5, 0, 1.0, 0.5469, 0, 0, 0, 0],  # % spikes missing
        [0, 0.5, 0, 1.0, 0.5469, 0, 0, 0, 0],  # max drift
        [1.0, 0.5469, 0, 0, 0.5, 0, 0, 0, 0],  # isolation dist
        [0, 0.5, 0, 1.0, 0.5469, 0, 0, 0, 0],  # L-ratio
    ])

    # Filter metrics that should be plotted
    valid_metrics = []
    valid_colors = []
    valid_labels = []
    valid_thresh1 = []
    valid_thresh2 = []
    valid_line_cols = []
    
    for i, (metric_name, condition) in enumerate(zip(metric_names, plot_conditions)):
        if condition and metric_name in quality_metrics:
            valid_metrics.append(metric_name)
            valid_colors.append(color_mtx[i % len(color_mtx)])
            valid_labels.append(metric_names_short[i])
            valid_thresh1.append(metric_thresh1[i])
            valid_thresh2.append(metric_thresh2[i])
            valid_line_cols.append(metric_line_cols[i])

    # Calculate grid layout
    num_subplots = len(valid_metrics)
    num_rows = int(np.floor(np.sqrt(num_subplots)))
    num_cols = int(np.ceil(num_subplots / num_rows))

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 9))
    fig.patch.set_facecolor('white')
    
    if num_rows == 1:
        axs = axs.reshape(1, -1)
    elif num_cols == 1:
        axs = axs.reshape(-1, 1)

    for i, metric_name in enumerate(valid_metrics):
        row_id = i // num_cols
        col_id = i % num_cols
        ax = axs[row_id, col_id]
        
        metric_data = quality_metrics[metric_name]
        metric_data = metric_data[~np.isnan(metric_data)]
        
        if len(metric_data) > 0:
            # Plot histogram with probability normalization (MATLAB style)
            if metric_name in ['nPeaks', 'nTroughs']:
                # Use integer bins for discrete metrics
                bins = np.arange(np.min(metric_data), np.max(metric_data) + 2) - 0.5
            elif metric_name == 'waveformDuration_peakTrough':
                # Use fewer bins for waveform duration like MATLAB
                bins = 20
            else:
                bins = 40
                
            n, bins_out, patches = ax.hist(metric_data, bins=bins, density=True, 
                                         color=valid_colors[i], alpha=0.7)
            
            # Convert to probability (like MATLAB's 'Normalization', 'probability')
            if metric_name not in ['nPeaks', 'nTroughs']:
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
            
            thresh1 = valid_thresh1[i]
            thresh2 = valid_thresh2[i]
            line_colors = valid_line_cols[i].reshape(3, 3)
            
            if metric_name in ['nPeaks', 'nTroughs']:
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
                    
                    if metric_name in noise_metrics:
                        # Noise metrics: both thresholds -> Noise, Neuronal, Noise
                        ax.text(midpoint1, text_y, '↓ Noise', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '↓ Neuronal', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                        ax.text(midpoint3, text_y, '↓ Noise', ha='center', fontsize=10, 
                               color=line_colors[2], weight='bold')
                    elif metric_name in nonsomatic_metrics:
                        # Non-somatic metrics: both thresholds -> Non-somatic, Somatic, Non-somatic
                        ax.text(midpoint1, text_y, '↓ Non-somatic', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '↓ Somatic', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                        ax.text(midpoint3, text_y, '↓ Non-somatic', ha='center', fontsize=10, 
                               color=line_colors[2], weight='bold')
                    else:
                        # MUA metrics: both thresholds -> MUA, Good, MUA
                        ax.text(midpoint1, text_y, '↓ MUA', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '↓ Good', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                        ax.text(midpoint3, text_y, '↓ MUA', ha='center', fontsize=10, 
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
                    
                    if metric_name in noise_metrics:
                        # Noise metrics: thresh1 only -> Neuronal, Noise
                        ax.text(midpoint1, text_y, '↓ Neuronal', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '↓ Noise', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                    elif metric_name in nonsomatic_metrics:
                        # Non-somatic metrics: thresh1 only -> Somatic, Non-somatic
                        ax.text(midpoint1, text_y, '↓ Somatic', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '↓ Non-somatic', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                    else:
                        # MUA metrics: thresh1 only -> Good, MUA
                        ax.text(midpoint1, text_y, '↓ Good', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '↓ MUA', ha='center', fontsize=10, 
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
                    
                    if metric_name in noise_metrics:
                        # Noise metrics: thresh2 only -> Noise, Neuronal
                        ax.text(midpoint1, text_y, '↓ Noise', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '↓ Neuronal', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                    elif metric_name in nonsomatic_metrics:
                        # Non-somatic metrics: thresh2 only -> Non-somatic, Somatic
                        ax.text(midpoint1, text_y, '↓ Non-somatic', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '↓ Somatic', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')
                    else:
                        # MUA metrics: thresh2 only -> MUA, Good
                        ax.text(midpoint1, text_y, '↓ MUA', ha='center', fontsize=10, 
                               color=line_colors[0], weight='bold')
                        ax.text(midpoint2, text_y, '↓ Good', ha='center', fontsize=10, 
                               color=line_colors[1], weight='bold')

            # Set histogram limits from 0 to 1
            ax.set_ylim([0, 1])
            
        ax.set_xlabel(valid_labels[i], fontsize=13)
        if i == 0:
            ax.set_ylabel('frac. units', fontsize=13)
        
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize=12)

    # Hide unused subplots
    for i in range(len(valid_metrics), num_rows * num_cols):
        row_id = i // num_cols
        col_id = i % num_cols
        axs[row_id, col_id].set_visible(False)

    plt.tight_layout()
    plt.show()