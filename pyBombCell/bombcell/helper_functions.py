import time
import numpy as np

from matplotlib.figure import Figure

# import bombcell.extract_raw_waveforms as erw
# import bombcell.loading_utils as led
# import bombcell.default_parameters as params
# import matplotlib.pyplot as plt
import bombcell.quality_metrics as qm


def show_times(times_spikes_missing_1, times_RPV_1, times_chunks_to_keep, times_spikes_missing_2, times_RPV_2, times_pressence_ratio, times_max_drift, times_waveform_shape):
    print(f'The time the first spikes missing took: {times_spikes_missing_1.sum()}')
    print(f'The time the first RPV took: {times_RPV_1.sum()}')
    print(f'The time the time chunks took: {times_chunks_to_keep.sum()}')
    print(f'The time the second spikes missing took: {times_spikes_missing_2.sum()}')
    print(f'The time the second RPV took: {times_RPV_2.sum()}')
    print(f'The time the presence ratio took: {times_pressence_ratio.sum()}')
    print(f'The time the max drift took: {times_max_drift.sum()}')
    print(f'The time the waveform shapes took: {times_waveform_shape.sum()}')

def print_unit_qm(quality_metrics, unit_idx, param, unit_type = None):
    print(f'For unit {unit_idx}:')
    print(f'n_peaks : {quality_metrics["n_peaks"][unit_idx]}, n_troughs : {quality_metrics["n_troughs"][unit_idx]}')
    print(f'waveform_duration_peak_trough : {quality_metrics["waveform_duration_peak_trough"][unit_idx]:.3f}')
    print(f'waveform_baseline : {quality_metrics["waveform_baseline"][unit_idx]}, spatial_decay_slope : {quality_metrics["spatial_decay_slope"][unit_idx]}')
    print(f'percent_missing_gaussian : {quality_metrics["percent_missing_gaussian"][unit_idx]}, n_spikes : {quality_metrics["n_spikes"][unit_idx]}')
    print(f'fraction_RPVs : {quality_metrics["fraction_RPVs"][unit_idx]}, presence_ratio : {quality_metrics["presence_ratio"][unit_idx]}')

    if param['extract_raw_waveforms']:
        print(f'raw_amplitude : {quality_metrics["raw_amplitude"][unit_idx]:.3f}, signal_to_noise_ratio : {quality_metrics["signal_to_noise_ratio"][unit_idx]:.3f}')
    if param['compute_distance_metrics']:
        print(f'max_drift_estimate : {quality_metrics["max_drift_estimate"][unit_idx]:.3f}')

    print(f'Waveform IS somatic = {1 == quality_metrics["is_somatic"][unit_idx]}')

    if unit_type is not None:
        print(f'The Units is classed as {unit_type[unit_idx]}')

def print_qm_thresholds(param):
    print('Current threshold params:')
    print(f'max_n_peaks = {param["max_n_peaks"]}, max_n_troughs = {param["max_n_troughs"]}')
    print(f'min_wave_duration = {param["min_wave_duration"]}, max_wave_duration = {param["max_wave_duration"]}')
    print(f'max_wave_baseline_fraction = {param["max_wave_baseline_fraction"]}, min_spatial_decay_slope = {param["min_spatial_decay_slope"]}')
    print(f'max_perc_spikes_missing = {param["max_perc_spikes_missing"]}, min_num_spikes_total = {param["min_num_spikes_total"]}')
    print(f'max_RPV = {param["max_RPV"]}, min_presence_ratio = {param["min_presence_ratio"]}')

    if param['extract_raw_waveforms']:
        print(f'min_amplitude = {param["min_amplitude"]}, min_SNR = {param["min_SNR"]}')
    
    if param['compute_distance_metrics']:
        print(f'max_drift = {param["max_drift"]}')

def show_somatic(quality_metrics, unit, is1, is2, is3):
    print(f'The max trough is {quality_metrics["trough"][unit]}')
    print(f'The main peak before is {quality_metrics["main_peak_before"][unit]}')
    print(f'The main peak after is {quality_metrics["main_peak_after"][unit]}')
    print(f'The first peak width is {quality_metrics["width_before"][unit]}')
    print(f'The trough_width is {quality_metrics["trough_width"][unit]}')

    if is1[unit] == 0:
        print('The trough is to small rel to peaks')
    if is2[unit] == 0:
        print('The first peak is too big')
    if is3[unit] == 0:
        print('The peak size to width is wrong')

def order_good_sites(good_sites, channel_pos):
    # make it so it goes from biggest to smallest
    reordered_idx = np.argsort(-channel_pos[good_sites,1].squeeze())
    reordered_good_sites = good_sites[reordered_idx]

    #re-arange x-axis so it goes (smaller x, bigger x)
    for i in range(8):
        a,b = channel_pos[reordered_good_sites[[2*i, 2*i+1]],0]

        if a > b:
            #swap order
            reordered_good_sites[[2*i + 1, 2*i]] = reordered_good_sites[[2*i, 2*i+1]]

    return reordered_good_sites

def nearest_channels(quality_metrics, channel_positions, this_unit, unique_templates):

    unit_id = unique_templates[this_unit]
    max_channel = quality_metrics['max_channels'][unit_id].squeeze()


    x, y = channel_positions[max_channel,:]

    x_dist = np.abs(channel_positions[:,0] - x)
    near_x_dist = np.min(x_dist[x_dist != 0])

    not_these_x = np.argwhere( x_dist > near_x_dist)

    y_dist = np.abs(channel_positions[:,1] - y)
    y_dist[not_these_x] = y_dist.max() # set the bad x_to max y, this keeps the shape of the array
    good_sites = np.argsort(y_dist)[:16]    

    ####
    # x, y = channel_positions[max_channel,:]

    # x_dist = np.abs(channel_positions[:,0] - x)
    # near_x = np.argmin(x_dist)

    # good_x_sites = np.argwhere( np.logical_and((x-50 < channel_positions[:,0]) == True, (channel_positions[:,0] < x+50) == True))
    # y_values = channel_positions[good_x_sites,1]

    # y_dist_to_max_site = np.abs(y_values - channel_positions[max_channel,1])
    # good_sites = np.argsort(y_dist_to_max_site,axis = 0 )[:16]
    # good_sites = good_x_sites[good_sites]

    reordered_good_sites = order_good_sites(good_sites, channel_positions)

    # ###
    # channels_with_same_x = np.argwhere(np.logical_and(channel_positions[:,0] <= channel_positions[max_channel, 0] +33,
    #                                 channel_positions[:,0] >= channel_positions[max_channel, 0] -33)) #4 shank probe
        
    # current_max_channel = channel_positions[max_channel, :]
    # distance_to_current_channel = np.square(channel_positions - current_max_channel)
    # sum_euclid = np.sum(distance_to_current_channel, axis = 1)

    # #find nearest x channels
    # near_x_channels = np.argwhere(distance_to_current_channel[:,0] <= np.sort(distance_to_current_channel[:,0])[50]).squeeze()

    # #from these enar x channel find the nearest y channels

    # use_these_channels = near_x_channels[np.argwhere(distance_to_current_channel[near_x_channels,1] <= np.sort(distance_to_current_channel[near_x_channels,1])[16])].squeeze()

    # #reordered_good_sites = order_good_sites(use_these_channels, channel_positions)
    # re_ordered_good_sites = use_these_channels
    
    return reordered_good_sites


def plot_raw_waveforms(quality_metrics, channel_positions, this_unit, waveform, unique_templates):

    unit_id = unique_templates[this_unit]

    fig = Figure(figsize=(4,6), dpi = 100)
    fig.set_tight_layout(False)

    main_ax = fig.add_axes([0.2,0.2,0.8,0.8])
    main_ax_offset = 0.2
    main_ax_scale = 0.8


    good_channels = nearest_channels(quality_metrics, channel_positions, this_unit, unique_templates).squeeze()

    min_x, min_y = channel_positions[good_channels[-2],[0,1]].squeeze()
    max_x, maxy = channel_positions[good_channels[1],[0,1]].squeeze()
    delta_x = (max_x - min_x) / 2
    delta_y = (maxy - min_y) / 18

    #may want to change so it find this for both units and selects the most extreme arguments
    #however i dont think tis will be necessary
    sub_min_y = np.nanmin(waveform[unit_id,:,good_channels])
    sub_max_y = np.nanmax(waveform[unit_id,:,good_channels])

    # shift each waveform so 0 is at the channel site, 1/9 is width of a y waveform plot
    waveform_y_offset = (np.abs(sub_max_y) / (np.abs(sub_min_y) + np.abs(sub_max_y)) ) * 1/8

    #make the main scatter positiose site as scatter with opacity 
    # main_ax.scatter(channel_positions[good_channels,0], channel_positions[good_channels,1], c = 'grey', alpha = 0.3)
    # main_ax.set_xlim(min_x - delta_x, max_x + delta_x)
    # main_ax.set_ylim(min_y - delta_y, maxy + delta_y)

    # rel_channel_positions = (channel_positions - channel_positions[good_channels].squeeze().min(axis = 0))/ (channel_positions[good_channels.squeeze()].max(axis = 0)  - channel_positions[good_channels].squeeze().min(axis = 0)) * 0.8
    # rel_channel_positions += main_ax_offset
    # for i in range(9):
    #     for j in range(2):
    #         #may need to change this positioning if units sizes are irregular
    #         # if j == 0:
    #         #     #The peak in the waveform is not half way, so maths says the x axis should be starting at
    #         #     #0.1 and 0.6 so the middle is at 0.25/0.76 however chosen these values so it loks better by eye
    #         #     ax =  fig.add_axes([main_ax_offset + main_ax_scale*0.25, main_ax_offset + main_ax_scale*(i/9 - 1/18 + waveform_y_offset), main_ax_scale*0.25, main_ax_scale*1/9])
    #         # if j == 1:
    #         #     ax = fig.add_axes([main_ax_offset + main_ax_scale*0.75, main_ax_offset + main_ax_scale*(i/9 - 1/18 + waveform_y_offset), main_ax_scale*0.25, main_ax_scale*1/9])

    #         if j == 0:
    #             #The peak in the waveform is not half way, so maths says the x axis should be starting at
    #             #0.1 and 0.6 so the middle is at 0.25/0.76 however chosen these values so it loks better by eye
    #             ax =  fig.add_axes([rel_channel_positions[good_channels,0][i*2 + j],rel_channel_positions[good_channels,1][i*2 + j], main_ax_scale*0.2, main_ax_scale*1/9])
    #         if j == 1:
    #             ax = fig.add_axes([rel_channel_positions[good_channels,0][i*2 + j],rel_channel_positions[good_channels,1][i*2 + j], main_ax_scale*0.22, main_ax_scale*1/9])

    #         ax.plot(waveform[unit_id,:,good_channels[i*2 + j]].squeeze(), color = 'g')             

            
    #         ax.set_ylim(sub_min_y,sub_max_y)
    #         ax.set_axis_off()


    main_ax.set_xlim(min_x - delta_x, max_x + delta_x)
    main_ax.set_ylim(min_y - delta_y, maxy + delta_y)

    rel_channel_positions = (channel_positions - channel_positions[good_channels].squeeze().min(axis = 0))/ (channel_positions[good_channels.squeeze()].max(axis = 0)  - channel_positions[good_channels].squeeze().min(axis = 0)) * 0.8
    rel_channel_positions += main_ax_offset
    for i in range(8):
        for j in range(2):
            #may need to change this positioning if units sizes are irregular
            # if j == 0:
            #     #The peak in the waveform is not half way, so maths says the x axis should be starting at
            #     #0.1 and 0.6 so the middle is at 0.25/0.76 however chosen these values so it loks better by eye
            #     ax =  fig.add_axes([main_ax_offset + main_ax_scale*0.25, main_ax_offset + main_ax_scale*(i/9 - 1/18 + waveform_y_offset), main_ax_scale*0.25, main_ax_scale*1/9])
            # if j == 1:
            #     ax = fig.add_axes([main_ax_offset + main_ax_scale*0.75, main_ax_offset + main_ax_scale*(i/9 - 1/18 + waveform_y_offset), main_ax_scale*0.25, main_ax_scale*1/9])

            if j == 0:
                #The peak in the waveform is not half way, so maths says the x axis should be starting at
                #0.1 and 0.6 so the middle is at 0.25/0.76 however chosen these values so it loks better by eye
                ax =  fig.add_axes([rel_channel_positions[good_channels,0][i*2 + j],rel_channel_positions[good_channels,1][i*2 + j], main_ax_scale*0.25, main_ax_scale*1/8])
            if j == 1:
                ax = fig.add_axes([rel_channel_positions[good_channels,0][i*2 + j],rel_channel_positions[good_channels,1][i*2 + j], main_ax_scale*0.25, main_ax_scale*1/8])

            ax.plot(waveform[unit_id,:,good_channels[i*2 + j]].squeeze(), color = 'g')             

            
            ax.set_ylim(sub_min_y,sub_max_y)
            ax.set_axis_off()



    main_ax.spines.right.set_visible(False)
    main_ax.spines.top.set_visible(False)
    main_ax.set_xticks([min_x, max_x])
    main_ax.set_xlabel('Xpos ($\mu$m)', size = 14)
    main_ax.set_ylabel('Ypos ($\mu$m)', size = 14)
    
    return fig

def show_unit(template_waveforms, this_unit, unique_templates, quality_metrics, channel_positions, param, unit_type = None):
    print_unit_qm(quality_metrics, this_unit, param, unit_type = unit_type)
    unit_id = unique_templates[this_unit]

    fig = plot_raw_waveforms(quality_metrics, channel_positions, this_unit, template_waveforms, unique_templates)
    return fig

def create_quality_metrics_dict(n_units, snr = None):
    """
    This function creates an quality_metrics dictionary with empty arrays to assign quality metric values to
    for each unit

    Parameters
    ----------
    n_units : int
        The number of units
    snr : ndarray, optional
        The SNR array if applicable, by default None

    Returns
    -------
    dict
        The quality metrics dictionary
    """

    quality_metrics = {}
    quality_metrics['phy_cluster_id'] = np.full(n_units, np.nan)
    quality_metrics['cluster_id'] = np.full(n_units, np.nan)
    quality_metrics['use_these_times_start'] = np.full(n_units, np.nan)
    quality_metrics['use_these_times_stop'] = np.full(n_units, np.nan)
    quality_metrics['RPV_use_tauR_est'] = np.full(n_units, np.nan)

    quality_metrics['percent_missing_gaussian'] = np.full(n_units, np.nan)
    quality_metrics['percent_missing_symmetric'] = np.full(n_units, np.nan)
    quality_metrics['fraction_RPVs'] = np.full(n_units, np.nan)
    quality_metrics['max_drift_estimate'] = np.full(n_units, np.nan)
    quality_metrics['cumulatve_drift_estimate'] = np.full(n_units, np.nan)
    quality_metrics['n_spikes'] = np.full(n_units, np.nan)
    quality_metrics['presence_ratio'] = np.full(n_units, np.nan)

    quality_metrics['n_peaks'] = np.full(n_units, np.nan)
    quality_metrics['n_troughs'] = np.full(n_units, np.nan)
    quality_metrics['is_somatic'] = np.full(n_units, np.nan)
    quality_metrics['waveform_duration_peak_trough'] = np.full(n_units, np.nan)
    #quality_metrics['spatial_decay_slope'] = np.zeros(unique_templates.shape[0])
    quality_metrics['waveform_baseline'] = np.full(n_units, np.nan)
    quality_metrics['linear_decay'] = np.full(n_units, np.nan)
    quality_metrics['exp_decay'] = np.full(n_units, np.nan)
    quality_metrics['waveform_baseline'] = np.full(n_units, np.nan)
    quality_metrics['trough'] = np.full(n_units, np.nan)
    quality_metrics['main_peak_before'] = np.full(n_units, np.nan)
    quality_metrics['main_peak_after'] = np.full(n_units, np.nan)
    quality_metrics['width_before'] = np.full(n_units, np.nan)
    quality_metrics['trough_width'] = np.full(n_units, np.nan)

    quality_metrics['raw_amplitude'] = np.full(n_units, np.nan)

    quality_metrics['isolation_dist'] = np.full(n_units, np.nan)
    quality_metrics['l_ratio'] = np.full(n_units, np.nan)
    quality_metrics['silhouette_score'] = np.full(n_units, np.nan)

    if isinstance(snr, np.ndarray):
        quality_metrics['signal_to_noise_ratio'] = snr
    
    return quality_metrics

def set_unit_nan(unit_idx, quality_metrics, not_enough_spikes):
        not_enough_spikes[unit_idx] = 1
        quality_metrics['use_these_times_start'][unit_idx] = np.nan
        quality_metrics['use_these_times_stop'][unit_idx] = np.nan
        quality_metrics['RPV_use_tauR_est'][unit_idx] = np.nan
        quality_metrics['percent_missing_gaussian'][unit_idx] = np.nan
        quality_metrics['percent_missing_symmetric'][unit_idx] = np.nan
        quality_metrics['fraction_RPVs'][unit_idx] = np.nan
        quality_metrics['max_drift_estimate'][unit_idx] = np.nan
        quality_metrics['cumulatve_drift_estimate'][unit_idx] = np.nan
        quality_metrics['n_peaks'][unit_idx] = np.nan
        quality_metrics['n_troughs'][unit_idx] = np.nan
        #quality_metrics['is_somatic'][unit_idx] = np.nan
        quality_metrics['waveform_duration_peak_trough'][unit_idx] = np.nan
        #quality_metrics['spatial_decay_slope'][unit_idx] = np.nan
        quality_metrics['waveform_baseline'][unit_idx] = np.nan
        quality_metrics['raw_amplitude'][unit_idx] = np.nan

        quality_metrics['linear_decay'][unit_idx] = np.nan
        quality_metrics['exp_decay'][unit_idx] = np.nan
        quality_metrics['trough'][unit_idx] = np.nan
        quality_metrics['main_peak_before'][unit_idx] = np.nan
        quality_metrics['main_peak_after'][unit_idx] = np.nan
        quality_metrics['width_before'][unit_idx] = np.nan
        quality_metrics['trough_width'][unit_idx] = np.nan

        quality_metrics['is_somatic'][unit_idx] = np.nan

        return quality_metrics, not_enough_spikes

def get_all_quality_metrics(unique_templates, spike_times_seconds, spike_templates, template_amplitudes, time_chunks, 
                            pc_features, pc_features_idx, quality_metrics, raw_waveforms_full, channel_positions, template_waveforms, param):
    #Collect the time it takes to run each section
    times_spikes_missing_1 = np.zeros(unique_templates.shape[0])
    times_RPV_1 = np.zeros(unique_templates.shape[0])
    times_chunks_to_keep = np.zeros(unique_templates.shape[0])
    times_spikes_missing_2 = np.zeros(unique_templates.shape[0])
    times_RPV_2 = np.zeros(unique_templates.shape[0]) 
    times_pressence_ratio = np.zeros(unique_templates.shape[0])
    times_max_drift = np.zeros(unique_templates.shape[0])
    times_waveform_shape = np.zeros(unique_templates.shape[0])
    time_dist_metrics = np.zeros(unique_templates.shape[0])

    not_enough_spikes = np.zeros(unique_templates.size)
    bad_units = 0
    for unit_idx in range(unique_templates.size):
        this_unit = unique_templates[unit_idx]
        quality_metrics['phy_cluster_id'][unit_idx] = this_unit
        quality_metrics['cluster_id'][unit_idx] = this_unit

        these_spike_times = spike_times_seconds[spike_templates == this_unit]
        these_amplitudes = template_amplitudes[spike_templates == this_unit]

        # number of spikes
        quality_metrics['n_spikes'][unit_idx] = these_spike_times.shape[0]

        #Ignoring for the moment as need to find a way to get the same shape as actual results for percent_missings
        # and fraction RPVs
        if these_spike_times.size < 50:
            quality_metrics, not_enough_spikes = set_unit_nan(unit_idx, quality_metrics, not_enough_spikes)
            bad_units +=1
            continue

        # percentage spikes missing 
        time_tmp = time.time()
        percent_missing_gaussian, percent_missing_symmetric, amp_bin_gaussian, spike_counts_per_amp_bin_gaussian, \
            gaussian_fit = qm.perc_spikes_missing(these_amplitudes, these_spike_times, time_chunks, param)
        times_spikes_missing_1[unit_idx] = time.time() - time_tmp

        # fraction contamination
        time_tmp = time.time()
        fraction_RPVs, num_violations = qm.fraction_RP_violations(these_spike_times, these_amplitudes, time_chunks, param)
        times_RPV_1[unit_idx] = time.time() - time_tmp

        # get time chunks to keep
        time_tmp = time.time()
        these_spike_times, these_amplitudes, these_spike_templates, \
            quality_metrics['use_these_times_start'][unit_idx], quality_metrics['use_these_times_stop'][unit_idx], quality_metrics['RPV_use_tauR_est'][unit_idx] = \
            qm.time_chunks_to_keep(percent_missing_gaussian, fraction_RPVs, time_chunks, these_spike_times, these_amplitudes, spike_templates, spike_times_seconds, param)
        times_chunks_to_keep[unit_idx] = time.time() - time_tmp

        use_these_times = np.array((quality_metrics['use_these_times_start'][unit_idx], quality_metrics['use_these_times_stop'][unit_idx]))
        #re-compute percentage spikes missing and RPV on time chunks
        time_tmp = time.time()
        quality_metrics['percent_missing_gaussian'][unit_idx], quality_metrics['percent_missing_symmetric'][unit_idx], amp_bin_gaussian, spike_counts_per_amp_bin_gaussian, \
                gaussian_fit = qm.perc_spikes_missing(these_amplitudes, these_spike_times, use_these_times, param)
        times_spikes_missing_2[unit_idx] = time.time() - time_tmp

        time_tmp = time.time()
        fraction_RPVs, num_violations = qm.fraction_RP_violations(these_spike_times, these_amplitudes, use_these_times, param, use_this_tauR = quality_metrics['RPV_use_tauR_est'][unit_idx])
        times_RPV_2[unit_idx] = time.time() - time_tmp

        quality_metrics['fraction_RPVs'][unit_idx] = fraction_RPVs[quality_metrics['RPV_use_tauR_est'][unit_idx].astype(int)]

        #get presence ratio
        time_tmp = time.time()
        quality_metrics['presence_ratio'][unit_idx] = qm.presence_ratio(these_spike_times, quality_metrics['use_these_times_start'][unit_idx], quality_metrics['use_these_times_stop'][unit_idx], param)
        times_pressence_ratio[unit_idx] = time.time() - time_tmp

        # maximum cumulative drift estimate
        time_tmp = time.time()
        quality_metrics['max_drift_estimate'][unit_idx], quality_metrics['cumulatve_drift_estimate'][unit_idx] = qm.max_drift_estimate(pc_features, pc_features_idx, these_spike_templates, these_spike_times, this_unit, channel_positions, param)
        times_max_drift[unit_idx] = time.time() - time_tmp

        # number of spikes
        quality_metrics['n_spikes'][unit_idx] = these_spike_times.shape[0]

        # waveform
        time_tmp = time.time()
        waveform_baseline_window = np.array((param['waveform_baseline_window_start'], param['waveform_baseline_window_stop']))
        quality_metrics['n_peaks'][unit_idx], quality_metrics['n_troughs'][unit_idx] , peak_locs, trough_locs,\
            quality_metrics['waveform_duration_peak_trough'][unit_idx] , spatial_decay_points, quality_metrics['linear_decay'][unit_idx], quality_metrics['exp_decay'][unit_idx],\
            quality_metrics['waveform_baseline'][unit_idx], this_waveform, quality_metrics['trough'][unit_idx], quality_metrics['main_peak_before'][unit_idx], \
            quality_metrics['main_peak_after'][unit_idx], quality_metrics['width_before'][unit_idx], quality_metrics['trough_width'][unit_idx], quality_metrics['is_somatic'][unit_idx] = qm.waveform_shape(template_waveforms, this_unit, quality_metrics['max_channels'], channel_positions, waveform_baseline_window, param)
        times_waveform_shape[unit_idx] = time.time() - time_tmp

        # amplitude
        gain_to_uV = 1
        if param['extract_raw_waveforms']:
            quality_metrics['raw_amplitude'][unit_idx] = qm.get_raw_amplitude(raw_waveforms_full[unit_idx], gain_to_uV)
        else:
            quality_metrics['raw_amplitude'][unit_idx] = np.nan

        time_tmp = time.time()
        if param['compute_distance_metrics']:
            quality_metrics['isolation_dist'][unit_idx], quality_metrics['l_ratio'][unit_idx], quality_metrics['silhouette_score'][unit_idx], histrogram_mahal_units_counts, histrogram_mahal_units_edges,\
                histrogram_mahal_noise_counts, histrogram_mahal_noise_edges = qm.get_distance_metrics(pc_features, pc_features_idx, this_unit, spike_templates, param)
        time_dist_metrics = time.time() - time_tmp

        if (unit_idx+1) % 10 == 0 or unit_idx == unique_templates.shape[0] - 1:
            print(f'done unit idx {unit_idx + 1} out of {unique_templates.shape[0]}')
    
    times = {'times_spikes_missing_1' : times_spikes_missing_1, 'times_RPV_1' : times_RPV_1,
             'times_chunks_to_keep' : times_chunks_to_keep, 'times_spikes_missing_2' : times_spikes_missing_2,
             'times_RPV_2' : times_RPV_2, 'times_pressence_ratio' : times_pressence_ratio,
             'times_max_drift' : times_max_drift, 'times_waveform_shape' : times_waveform_shape}
    
    if param['compute_distance_metrics']:
        times['time_dist_metrics'] = time_dist_metrics

    return quality_metrics, times

def get_quality_unit_type(param, quality_metrics):
    """
    Assign each unit a type based of its' quality metrics.
    unit_type == 0 all noise units
    unit_type == 1 all good units
    unit_type == 2 all mua units
    unit_type == 3 all non-somatic units (if split somatic units its good non-somatic units)
    unit_type == 4 (if split somatic units its mua non-somatic units)


    Parameters
    ----------
    param : df
        The param dataframe from ML BombCell 
    quality_metrics : df
        The quality metrics dataframefrom ML BombCell

    Returns
    -------
    tuple (np array, np array)
        Two array of the unit types one as number the other as strings
    """
    
    #Testing for non-somatic waveforms
    is_non_somatic = np.zeros(quality_metrics['n_peaks'].shape[0])

    is_non_somatic[(quality_metrics['trough'] / np.max((quality_metrics['main_peak_before'] , quality_metrics['main_peak_after']), axis = 0)) < param['non_somatic_trough_peak_ratio']] = 1 

    is_non_somatic[(quality_metrics['main_peak_before'] / quality_metrics['main_peak_after'])  > param['non_somatic_peak_before_to_after_ratio']] = 1

    is_non_somatic[(quality_metrics['main_peak_before'] * param['first_peak_ratio'] > quality_metrics['main_peak_after']) & (quality_metrics['width_before'] < param['min_width_first_peak']) \
    & (quality_metrics['main_peak_before'] * param['min_main_peak_to_trough_ratio'] > quality_metrics['trough']) & (quality_metrics['trough_width'] < param['min_width_main_trough'])] = 1

    #Test all quality metrics
    ## categorise units
    # unit_type == 0 all noise units
    # unit_type == 1 all good units
    # unit_type == 2 all mua units
    # unit_type == 3 all non-somatic units (if split somatic units its good non-somatic units)
    # unit_type == 4 (if split somatic units its mua non-somatic units)

    unit_type = np.full(quality_metrics['n_peaks'].shape[0], np.nan)

    # classify noise
    unit_type[np.isnan(quality_metrics['n_peaks'])] = 0
    unit_type[quality_metrics['n_peaks']  > param['max_n_peaks']] = 0
    unit_type[quality_metrics['n_troughs'] > param['max_n_troughs']] = 0
    unit_type[quality_metrics['waveform_duration_peak_trough'] < param['min_wave_duration']] = 0
    unit_type[quality_metrics['waveform_duration_peak_trough'] > param['max_wave_duration']] = 0
    unit_type[quality_metrics['waveform_baseline'] > param['max_wave_baseline_fraction']] = 0
    unit_type[quality_metrics['exp_decay'] > param['min_spatial_decay_slope']] = 0
    unit_type[quality_metrics['exp_decay'] < param['max_spatial_decay_slope']] = 0

    # classify as mua
    #ALL or ANY?
    unit_type[np.logical_and(quality_metrics['percent_missing_gaussian'] > param['max_perc_spikes_missing'], np.isnan(unit_type))] = 2
    unit_type[np.logical_and(quality_metrics['n_spikes'] < param['min_num_spikes_total'] , np.isnan(unit_type))] = 2
    unit_type[np.logical_and(quality_metrics['fraction_RPVs']> param['max_RPV'], np.isnan(unit_type))] = 2
    unit_type[np.logical_and(quality_metrics['presence_ratio'] < param['min_presence_ratio'] , np.isnan(unit_type))] = 2

    if param['extract_raw_waveforms']:
        unit_type[np.logical_and(quality_metrics['raw_amplitude'] < param['min_amplitude'] , np.isnan(unit_type))] = 2
        unit_type[np.logical_and(quality_metrics['signal_to_noise_ratio'] < param['min_SNR'] , np.isnan(unit_type))] = 2

    if param['compute_drift']:
        unit_type[np.logical_and(quality_metrics['max_drift_estimate'] > param['max_drift'] , np.isnan(unit_type))] = 2

    if param['compute_distance_metrics']:
        unit_type[np.logical_and(quality_metrics['isolation_dist'] > param['iso_d_min'] , np.isnan(unit_type))] = 2
        unit_type[np.logical_and(quality_metrics['l_ratio'] > param['lratio_max'] , np.isnan(unit_type))] = 2

    unit_type[np.isnan(unit_type)] = 1 # SINGLE SEXY UNIT

    if param['split_good_and_mua_non_somatic']:
        unit_type[np.logical_and(is_non_somatic == 1, unit_type == 1)] = 3 # Good non-somatic
        unit_type[np.logical_and(is_non_somatic == 1, unit_type == 2)] = 4 # MUA non-somatic
    else:
        unit_type[np.logical_and(is_non_somatic == 1, unit_type != 0)] = 3 # Good non-somatic

    #Have unit types as strings as well
    unit_type_string = np.full(unit_type.size, '', dtype = object)
    unit_type_string[unit_type == 0] = 'NOISE'
    unit_type_string[unit_type == 1] = 'GOOD'
    unit_type_string[unit_type == 2] = 'MUA'

    if param['split_good_and_mua_non_somatic']:
        unit_type_string[unit_type == 3] = 'NON-SOMA GOOD'
        unit_type_string[unit_type == 4] = 'NON-SOMA MUA'
    else:
        unit_type_string[unit_type == 3] = 'NON-SOMA'
    
    return unit_type, unit_type_string