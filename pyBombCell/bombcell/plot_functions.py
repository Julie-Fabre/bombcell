import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from upsetplot import UpSet, from_indicators

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

    noise_metrics = ["# peaks", "# troughs", "duration", "spatial decay", "baseline flatness", "peak2 / trough"] #Duration is peak to trough duration
    non_somatic_metrics = ["peak(main) / trough", "peak1 / peak2"]
    mua_metrics = ["SNR", "amplitude", "# spikes", "presence ratio", "% spikes missing", "fraction RPVs"]

    # Eventually filter out uncomputed metrics
    noise_metrics = [m for m in noise_metrics if m in qm_table.columns]
    non_somatic_metrics = [m for m in non_somatic_metrics if m in qm_table.columns]
    mua_metrics = [m for m in mua_metrics if m in qm_table.columns]

    # plot noise metrics upset plot
    upset = UpSet(from_indicators(noise_metrics, data = qm_table.astype(bool)), min_degree = 1)
    upset.plot()
    plt.suptitle("Units classified as noise")
    plt.show()

    # plot non-somatic metrics upset plot
    upset = UpSet(from_indicators(non_somatic_metrics, data = qm_table.astype(bool)), min_degree = 1)
    upset.plot()
    plt.suptitle("Units classified as non-somatic")
    plt.show()

    # plot MUA metrics upset plot
    upset = UpSet(from_indicators(mua_metrics, data = qm_table.astype(bool)), min_degree = 1)
    upset.plot()
    plt.suptitle("Units classified as MUA")
    plt.show()

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
                    axs[img_pos[i][0]][img_pos[i][1]].plot(template_waveforms[id, 20:, quality_metrics['maxChannels'][id]], color = 'black', alpha = 0.1)
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
                    axs[img_pos[i][0]][img_pos[i][1]].plot(template_waveforms[id, 20:, quality_metrics['maxChannels'][id]], color = 'black', alpha = 0.1)
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

    #find which metrics to add
    color_pass = {"Noise" : "k", "Somatic" : "k", "MUA" : 'g'}
    color_fail = {"Noise" : "r", "Somatic" : "b", "MUA" : 'orange'}

    plot_metric_keys = ["nPeaks", "nTroughs", "waveformBaselineFlatness", "waveformDuration_peakTrough", "scndPeakToTroughRatio"]
    metric_types = ["Noise", "Noise", "Noise", "Noise", "Noise"]
    is_continous = [False, False, True, True, True]
    plot_metric_thresholds_lower_bound = [None, None, None, "minWvDuration", None]
    plot_metric_thresholds_upper_bound = ["maxNPeaks", "maxNTroughs", "maxWvBaselineFraction", "maxWvDuration", "maxScndPeakToTroughRatio_noise "]
    x_axis_labels = ["# peaks", "# troughs", "baseline flatness", "waveform duration", "peak2/trough"]

    #Add correct type of spatial decay if spatial decay is calculated
    if param["computeSpatialDecay"] & param["spDecayLinFit"]:
        plot_metric_keys.append("spatialDecaySlope")
        metric_types.append("Noise")
        is_continous.append(True)
        plot_metric_thresholds_lower_bound.append("maxSpatialDecaySlopeExp")
        plot_metric_thresholds_upper_bound.append(None)
        x_axis_labels.append("spatial decay")
    elif param["computeSpatialDecay"]:
        plot_metric_keys.append("spatialDecaySlope")
        metric_types.append("Noise")
        is_continous.append(True)
        plot_metric_thresholds_lower_bound.append("maxSpatialDecaySlopeExp")
        plot_metric_thresholds_upper_bound.append("minSpatialDecaySlopeExp")
        x_axis_labels.append("spatial decay")

    #add rest of core metrics
    plot_metric_keys.extend(["peak1ToPeak2Ratio", "mainPeakToTroughRatio",
                        "fractionRPVs_estimatedTauR", "presenceRatio", "percentageSpikesMissing_gaussian", "nSpikes"])
    metric_types.extend(["Somatic", "Somatic",
                    "MUA", "MUA", "MUA", "MUA"])
    is_continous.extend([True, True,
                    True, True, True, True])
    plot_metric_thresholds_lower_bound.extend([None, None,
                                        None, "minPresenceRatio", None, "minNumSpikes"])
    plot_metric_thresholds_upper_bound.extend(["maxPeak1ToPeak2Ratio_nonSomatic", "maxMainPeakToTroughRatio_nonSomatic",
                                        "maxRPVviolations", None, "maxPercSpikesMissing", None])
    x_axis_labels.extend([ "peak1/peak2", "peak(main)/trough",
                        "frac. RPVs", "presence ratio", "% spikes missing", "# spikes"])

    #add optional metrics
    if param["extractRaw"] and np.all(~np.isnan(quality_metrics['rawAmplitude'])):
        plot_metric_keys.extend(["rawAmplitude", "signalToNoiseRatio"])
        metric_types.extend(["MUA", "MUA"])
        is_continous.extend([True, True])
        plot_metric_thresholds_lower_bound.extend(["minAmplitude", "min_SNR"])
        plot_metric_thresholds_upper_bound.extend([None, None])
        x_axis_labels.extend(["amplitude", "SNR"])

    if param["computeDrift"]:
        plot_metric_keys.append("maxDriftEstimate")
        metric_types.append("MUA")
        is_continous.append(True)
        plot_metric_thresholds_lower_bound.append(None)
        plot_metric_thresholds_upper_bound.append("maxDrift")
        x_axis_labels.append("max drift")

    if param["computeDistanceMetrics"]:
        plot_metric_keys.extend(["isolationDistance", "Lratio"])
        metric_types.extend(["MUA", "MUA"])
        is_continous.extend([True, True])
        plot_metric_thresholds_lower_bound.extend(["isoDmin ", None])
        plot_metric_thresholds_upper_bound.extend([None, "lratioMax"])
        x_axis_labels.extend(["isolation dist", "L ratio"])

    #plot all histograms
    n_metrics = len(plot_metric_keys) + 1
    n_rows = int(np.floor(np.sqrt(n_metrics)))
    n_columns = int(np.ceil(n_metrics / n_rows))

    fig, axs = plt.subplots(nrows = n_rows, ncols = n_columns, layout = "constrained", figsize = (12,8))
    for i in range(n_rows * n_columns):
        row_id = int(np.floor(i / n_columns))
        column_id = i % n_columns
        if (i+1) < n_metrics:
            #get the quality metric and the bounds
            metric = quality_metrics[plot_metric_keys[i]]
            if plot_metric_thresholds_lower_bound[i] is not None:
                lower_threshold = param[plot_metric_thresholds_lower_bound[i]]
            else:
                lower_threshold = np.nanmin(metric) - 1e-6
            if plot_metric_thresholds_upper_bound[i] is not None:
                upper_threshold = param[plot_metric_thresholds_upper_bound[i]]
            else:
                upper_threshold = np.nanmax(metric) + 1e-6
            
            current_metric_type = metric_types[i]
            pass_color = color_pass[current_metric_type]
            fail_color = color_fail[current_metric_type]

            #plot histogram with 40 bins
            if is_continous[i]:

                good_idx = np.logical_and(metric > lower_threshold, metric < upper_threshold)
                bins = np.histogram_bin_edges(metric[~np.isnan(metric)], bins = 40)
                axs[row_id][column_id].hist(metric[good_idx], bins, color = pass_color)
                axs[row_id][column_id].hist(metric[~good_idx], bins, color = fail_color)

                if plot_metric_thresholds_lower_bound[i] is not None:
                    axs[row_id][column_id].axvline(lower_threshold, color = 'k')
                if plot_metric_thresholds_upper_bound[i] is not None:
                    axs[row_id][column_id].axvline(upper_threshold, color = 'k')

                axs[row_id][column_id].set_xlabel(f"{x_axis_labels[i]}")
            #plot bar chart for discrete data
            else:
                good_idx = np.logical_and(metric > lower_threshold, metric < upper_threshold)
                points, counts = np.unique(metric, return_counts=True)

                colors = [fail_color if upper_threshold < x or x < lower_threshold  else pass_color for x in points ]
                axs[row_id][column_id].bar(points, counts, color = colors)

                if plot_metric_thresholds_lower_bound[i] is not None:
                    axs[row_id][column_id].axvline(lower_threshold + 0.5, color = 'k')
                if plot_metric_thresholds_upper_bound[i] is not None:
                    axs[row_id][column_id].axvline(upper_threshold + 0.5, color = 'k')

                axs[row_id][column_id].set_xlabel(f"{x_axis_labels[i]}")
            
            if i == 0:
                axs[row_id][column_id].set_ylabel("N. units")
            axs[row_id][column_id].spines[['right', 'top']].set_visible(False)
        else:
            axs[row_id][column_id].spines[['right', 'top', 'bottom', 'left']].set_visible(False)
            axs[row_id][column_id].set_xticks([])
            axs[row_id][column_id].set_yticks([])
    plt.show()