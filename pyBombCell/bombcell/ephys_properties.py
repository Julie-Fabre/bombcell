import numpy as np
import pandas as pd
from scipy import signal, optimize, stats
from scipy.sparse import csr_matrix
import os

# Path handling utility function
def path_handler(path):
    """Simple path handler utility"""
    return Path(path)


def run_all_ephys_properties(ephys_path, param=None, save_path=None):
    """
    Main function to compute all ephys properties and classify cells
    
    Parameters
    ----------
    ephys_path : str
        Path to ephys data directory
    param : dict, optional
        Parameters dictionary
    save_path : str, optional
        Path to save results
    
    Returns
    -------
    ephys_properties : dict
        Dictionary containing all computed ephys properties
    cell_types : dict
        Dictionary containing cell type classifications
    """
    if param is None:
        from bombcell.default_parameters import get_default_parameters
        param = get_default_parameters(ephys_path)
    
    if save_path is None:
        save_path = ephys_path
    
    # Set ephys properties parameters
    param = ephys_prop_values(param)
    
    # Compute all ephys properties
    ephys_properties = compute_all_ephys_properties(ephys_path, param, save_path)
    
    # Classify cells
    cell_types = classify_cells(ephys_properties, param)
    
    # Save results
    save_ephys_properties(ephys_properties, cell_types, save_path, param)
    
    return ephys_properties, param


def ephys_prop_values(param):
    """
    Set default parameters for ephys properties computation
    
    Parameters
    ----------
    param : dict
        Existing parameters dictionary
        
    Returns
    -------
    param : dict
        Updated parameters dictionary with ephys property defaults
    """
    # ACG parameters
    param.setdefault('acg_binSize', 0.0005)  # 0.5ms bins
    param.setdefault('acg_duration_ms', 100)  # 100ms window
    param.setdefault('refractory_period_ms', 2)  # 2ms refractory period
    
    # ISI parameters  
    param.setdefault('longISI_threshold', 2.0)  # 2 seconds
    param.setdefault('cv_threshold', 2.0)
    param.setdefault('cv2_window', 1.0)  # 1 second window for CV2
    
    # Waveform parameters
    param.setdefault('wf_duration_method', 'peak_to_trough')
    param.setdefault('half_width_method', 'fwhm')  # full width at half maximum
    
    # Firing rate parameters
    param.setdefault('fr_bin_size', 60)  # 60 second bins for firing rate
    param.setdefault('min_spikes_for_stats', 100)  # minimum spikes for reliable stats
    
    # Classification thresholds (striatum)
    param.setdefault('msn_waveform_duration_max', 0.0005)  # 0.5ms
    param.setdefault('fsi_waveform_duration_max', 0.0004)  # 0.4ms
    param.setdefault('fsi_firing_rate_min', 10)  # 10 Hz
    param.setdefault('tan_cv_max', 1.0)
    param.setdefault('tan_firing_rate_min', 2)  # 2 Hz
    
    # Classification thresholds (cortex)
    param.setdefault('narrow_waveform_duration_max', 0.0004)  # 0.4ms
    param.setdefault('wide_waveform_duration_min', 0.0005)  # 0.5ms
    
    # Brain region
    param.setdefault('brain_region', 'striatum')  # 'striatum' or 'cortex'
    
    return param


def compute_all_ephys_properties(ephys_path, param, save_path):
    """
    Compute all ephys properties for all units
    
    Parameters
    ----------
    ephys_path : str
        Path to ephys data
    param : dict
        Parameters dictionary
    save_path : str
        Path to save results
        
    Returns
    -------
    ephys_properties : dict
        Dictionary containing all computed properties
    """
    from bombcell.loading_utils import load_ephys_data
    
    # Load spike data
    (
        spike_times_samples,
        spike_clusters,
        template_waveforms,
        template_amplitudes,
        pc_features,
        pc_features_idx,
        channel_positions,
    ) = load_ephys_data(ephys_path)
    
    # Convert to seconds
    spike_times = spike_times_samples / param.get('ephys_sample_rate', 30000)
    
    # Get unique units
    unique_units = np.unique(spike_clusters)
    n_units = len(unique_units)
    
    # Initialize properties dictionary
    ephys_properties = []
    
    for i in range(n_units):
        properties = {
            # ACG properties
            'acg_pss_ratio': np.nan,
            'acg_tau_rise': np.nan,
            'acg_tau_decay': np.nan,
            # ISI properties
            'isi_cv': np.nan,
            'isi_cv2': np.nan,
            'isi_skewness': np.nan,
            'prop_long_isi': np.nan,
            # Waveform properties
            'waveform_duration_peak_trough': np.nan,
            'waveform_half_width': np.nan,
            'peak_to_trough_ratio': np.nan,
            'n_peaks': np.nan,
            'n_troughs': np.nan,
            # Spike properties
            'firing_rate_mean': np.nan,
            'firing_rate_std': np.nan,
            'fano_factor': np.nan,
        }
        ephys_properties.append(properties)
    
    print(f"Computing ephys properties for {n_units} units...")
    
    # Get sampling rate
    sampling_rate = param.get('ephys_sample_rate', 30000)
    
    # Convert spike times to seconds if needed
    if np.max(spike_times) > 1e6:  # Likely in samples
        spike_times_sec = spike_times / sampling_rate
    else:
        spike_times_sec = spike_times
    
    # Compute properties for each unit
    from tqdm.auto import tqdm
    
    bar_description = "Computing ephys properties: {percentage:3.0f}%|{bar:10}| {n}/{total} units"
    for i, unit_id in enumerate(tqdm(unique_units, bar_format=bar_description)):
        
        # Get spikes for this unit
        unit_spikes = spike_times_sec[spike_clusters == unit_id]
        
        if len(unit_spikes) < param['min_spikes_for_stats']:
            continue
            
        # Get template for this unit (use template_waveforms loaded from kilosort)
        unit_template = template_waveforms[unit_id]
        
        # Compute ACG properties
        acg_props = compute_acg_properties(unit_spikes, param)
        ephys_properties[i]['acg_pss_ratio'] = acg_props.get('post_spike_suppression_ratio', np.nan)
        ephys_properties[i]['acg_tau_rise'] = acg_props.get('tau_rise_ms', np.nan)
        ephys_properties[i]['acg_tau_decay'] = acg_props.get('tau_decay_ms', np.nan)
        
        # Compute ISI properties
        isi_props = compute_isi_properties(unit_spikes, param)
        ephys_properties[i]['isi_cv'] = isi_props.get('cv', np.nan)
        ephys_properties[i]['isi_cv2'] = isi_props.get('cv2', np.nan)
        ephys_properties[i]['isi_skewness'] = isi_props.get('isi_skewness', np.nan)
        ephys_properties[i]['prop_long_isi'] = isi_props.get('prop_long_isi', np.nan)
        
        # Compute waveform properties
        wf_props = compute_waveform_properties(unit_template, param, sampling_rate)
        ephys_properties[i]['waveform_duration_peak_trough'] = wf_props.get('waveform_duration_us', np.nan)
        ephys_properties[i]['waveform_half_width'] = wf_props.get('half_width_ms', np.nan)
        ephys_properties[i]['peak_to_trough_ratio'] = wf_props.get('peak_to_trough_ratio', np.nan)
        ephys_properties[i]['n_peaks'] = wf_props.get('n_peaks', np.nan)
        ephys_properties[i]['n_troughs'] = wf_props.get('n_troughs', np.nan)
        
        # Compute spike properties
        spike_props = compute_spike_properties(unit_spikes, param)
        ephys_properties[i]['firing_rate_mean'] = spike_props.get('mean_firing_rate', np.nan)
        ephys_properties[i]['firing_rate_std'] = spike_props.get('std_firing_rate', np.nan)
        ephys_properties[i]['fano_factor'] = spike_props.get('fano_factor', np.nan)
    
    print("Ephys properties computation complete!")
    return ephys_properties


def compute_acg_properties(spike_times, param):
    """
    Compute auto-correlogram based properties
    
    Parameters
    ----------
    spike_times : array
        Spike times in seconds
    param : dict
        Parameters dictionary
        
    Returns
    -------
    acg_props : dict
        Dictionary containing ACG properties
    """
    # Compute ACG
    acg, lags = compute_acg(spike_times, param['acg_binSize'], param['acg_duration_ms']/1000)
    
    # Initialize output
    acg_props = {
        'post_spike_suppression_ratio': np.nan,
        'tau_rise_ms': np.nan,
        'tau_decay_ms': np.nan
    }
    
    if len(acg) == 0 or len(spike_times) < 10:
        return acg_props
    
    # Ensure ACG has some counts
    if np.sum(acg) == 0:
        return acg_props
    
    # Debug: print ACG info
    if len(spike_times) > 100:  # Only for units with enough spikes
        print(f"  ACG computed: {len(acg)} bins, total counts: {np.sum(acg)}, max: {np.max(acg)}")
    
    # Find center bin
    center_idx = len(acg) // 2
    
    # Post-spike suppression: ratio of minimum in first 10ms to baseline
    post_bins = max(1, int(0.01 / param['acg_binSize']))  # 10ms in bins, at least 1
    baseline_start = max(post_bins + 5, int(0.05 / param['acg_binSize']))  # Start baseline at 50ms
    
    # Make sure we have enough bins
    if center_idx + baseline_start + 10 < len(acg) and post_bins > 0:
        # Suppression region: 1-10ms after center (exclude 0 lag to avoid refractory artifact)
        suppression_start = center_idx + 1
        suppression_end = min(center_idx + post_bins + 1, len(acg))
        suppression_region = acg[suppression_start:suppression_end]
        
        # Baseline region: 50ms to end
        baseline_start_idx = center_idx + baseline_start
        baseline_region = acg[baseline_start_idx:]
        
        if len(suppression_region) > 0 and len(baseline_region) > 0:
            # Use minimum in suppression region
            min_val = np.min(suppression_region)
            baseline_mean = np.mean(baseline_region)
            
            if baseline_mean > 0:
                acg_props['post_spike_suppression_ratio'] = min_val / baseline_mean
            else:
                # If baseline is 0, check if suppression is also 0
                acg_props['post_spike_suppression_ratio'] = 0.0 if min_val == 0 else np.nan
    
    # Simple tau estimation based on ACG shape in first 20ms
    try:
        if center_idx + int(0.02/param['acg_binSize']) < len(acg):
            fit_region = acg[center_idx+1:center_idx+int(0.02/param['acg_binSize'])]  # 0-20ms region
            
            if len(fit_region) > 5:
                # Find rise time: time to reach half maximum
                max_val = np.max(fit_region)
                half_max = max_val * 0.5
                
                if max_val > 0:
                    # Find first bin that exceeds half max
                    rise_idx = np.where(fit_region >= half_max)[0]
                    if len(rise_idx) > 0:
                        acg_props['tau_rise_ms'] = rise_idx[0] * param['acg_binSize'] * 1000
                
                # Simple decay estimation: fit exponential to second half
                if len(fit_region) > 10:
                    decay_region = fit_region[len(fit_region)//2:]
                    x = np.arange(len(decay_region))
                    
                    if len(decay_region) > 3 and np.max(decay_region) > 0:
                        # Simple exponential fit
                        try:
                            # Log-linear fit for exponential decay
                            y_log = np.log(np.maximum(decay_region, np.max(decay_region)*0.01))
                            coeffs = np.polyfit(x, y_log, 1)
                            tau_bins = -1 / coeffs[0] if coeffs[0] < 0 else np.nan
                            acg_props['tau_decay_ms'] = tau_bins * param['acg_binSize'] * 1000
                        except:
                            pass
    except:
        pass
    
    return acg_props


def fast_acg(spike_times, bin_size_ms=0.5, win_size_ms=100):
    """
    Fast auto-correlogram computation using vectorized operations
    
    Parameters
    ----------
    spike_times : array
        Spike times in seconds
    bin_size_ms : float
        Bin size in milliseconds
    win_size_ms : float
        Window size in milliseconds
        
    Returns
    -------
    acg : array
        Auto-correlogram counts
    lags : array
        Lag times in milliseconds
    """
    if len(spike_times) < 2:
        return np.array([]), np.array([])
    
    # Convert to milliseconds for computation
    spike_times_ms = spike_times * 1000
    n_bins = int(win_size_ms / bin_size_ms)
    if n_bins % 2 == 0:
        n_bins += 1
    
    half_bins = n_bins // 2
    max_lag_ms = half_bins * bin_size_ms
    
    # Create bins
    bins = np.arange(-max_lag_ms, max_lag_ms + bin_size_ms, bin_size_ms)
    
    # For very large spike trains, use a more memory-efficient approach
    if len(spike_times) > 10000:
        # Use a chunked approach for large spike trains
        acg = np.zeros(len(bins) - 1)
        chunk_size = 5000
        
        for i in range(0, len(spike_times_ms), chunk_size):
            chunk = spike_times_ms[i:i+chunk_size]
            # Compute differences for this chunk vs all spikes
            for spike_time in chunk:
                diffs = spike_times_ms - spike_time
                # Remove self-correlation
                diffs = diffs[diffs != 0]
                # Only keep diffs within window
                valid_diffs = diffs[np.abs(diffs) <= max_lag_ms]
                if len(valid_diffs) > 0:
                    hist, _ = np.histogram(valid_diffs, bins=bins)
                    acg += hist
    else:
        # Vectorized computation for smaller spike trains
        spike_diffs = spike_times_ms[:, None] - spike_times_ms[None, :]
        # Remove diagonal (self-correlations)
        mask = ~np.eye(len(spike_times_ms), dtype=bool)
        spike_diffs = spike_diffs[mask]
        
        # Histogram the differences
        acg, _ = np.histogram(spike_diffs, bins=bins)
    
    lags = bins[:-1] + bin_size_ms / 2  # Center of bins
    
    return acg, lags


def compute_acg(spike_times, bin_size, duration):
    """
    Wrapper for fast ACG computation - maintains interface compatibility
    
    Parameters
    ----------
    spike_times : array
        Spike times in seconds
    bin_size : float
        Bin size in seconds
    duration : float
        Total duration in seconds (centered around 0)
        
    Returns
    -------
    acg : array
        Auto-correlogram counts
    lags : array
        Lag times in seconds
    """
    bin_size_ms = bin_size * 1000  # Convert to ms
    duration_ms = duration * 1000  # Convert to ms
    
    acg, lags_ms = fast_acg(spike_times, bin_size_ms, duration_ms)
    lags = lags_ms / 1000  # Convert back to seconds
    
    return acg, lags


def compute_isi_properties(spike_times, param):
    """
    Compute inter-spike interval based properties
    
    Parameters
    ----------
    spike_times : array
        Spike times in seconds
    param : dict
        Parameters dictionary
        
    Returns
    -------
    isi_props : dict
        Dictionary containing ISI properties
    """
    isi_props = {
        'prop_long_isi': np.nan,
        'cv': np.nan,
        'cv2': np.nan,
        'isi_skewness': np.nan
    }
    
    if len(spike_times) < 2:
        return isi_props
    
    # Compute ISIs
    isis = np.diff(spike_times)
    
    # Proportion of long ISIs
    long_isis = isis > param['longISI_threshold']
    isi_props['prop_long_isi'] = np.mean(long_isis)
    
    # Coefficient of variation
    if len(isis) > 0:
        isi_props['cv'] = np.std(isis) / np.mean(isis)
        isi_props['isi_skewness'] = stats.skew(isis)
    
    # CV2 (local coefficient of variation)
    if len(isis) > 1:
        cv2_values = []
        for i in range(len(isis) - 1):
            mean_isi = (isis[i] + isis[i+1]) / 2
            diff_isi = abs(isis[i+1] - isis[i])
            cv2_values.append(diff_isi / mean_isi)
        isi_props['cv2'] = np.mean(cv2_values)
    
    return isi_props


def compute_waveform_properties(template, param, sampling_rate):
    """
    Compute waveform-based properties
    
    Parameters
    ----------
    template : array
        Template waveform (samples x channels)
    param : dict
        Parameters dictionary
    sampling_rate : float
        Sampling rate in Hz
        
    Returns
    -------
    wf_props : dict
        Dictionary containing waveform properties
    """
    wf_props = {
        'waveform_duration_us': np.nan,
        'half_width_ms': np.nan,
        'peak_to_trough_ratio': np.nan,
        'first_peak_to_trough_ratio': np.nan,
        'n_peaks': np.nan,
        'n_troughs': np.nan
    }
    
    # Get max channel
    max_channel = np.argmax(np.max(np.abs(template), axis=0))
    waveform = template[:, max_channel]
    
    # Find peaks and troughs
    peaks, _ = signal.find_peaks(waveform, height=np.max(waveform) * 0.1)
    troughs, _ = signal.find_peaks(-waveform, height=np.max(-waveform) * 0.1)
    
    wf_props['n_peaks'] = len(peaks)
    wf_props['n_troughs'] = len(troughs)
    
    # Peak-to-trough duration
    if len(peaks) > 0 and len(troughs) > 0:
        main_peak = peaks[np.argmax(waveform[peaks])]
        main_trough = troughs[np.argmax(-waveform[troughs])]
        
        duration_samples = abs(main_peak - main_trough)
        wf_props['waveform_duration_us'] = (duration_samples / sampling_rate) * 1e6
        
        # Peak-to-trough ratio
        wf_props['peak_to_trough_ratio'] = waveform[main_peak] / abs(waveform[main_trough])
        
        # First peak-to-trough ratio
        if len(peaks) > 1:
            first_peak = peaks[0] if peaks[0] < main_trough else main_peak
            wf_props['first_peak_to_trough_ratio'] = waveform[first_peak] / abs(waveform[main_trough])
    
    # Half width (simplified)
    try:
        half_max = np.max(waveform) / 2
        indices = np.where(waveform >= half_max)[0]
        if len(indices) > 1:
            half_width_samples = indices[-1] - indices[0]
            wf_props['half_width_ms'] = (half_width_samples / sampling_rate) * 1000
    except:
        pass
    
    return wf_props


def compute_spike_properties(spike_times, param):
    """
    Compute spike timing based properties
    
    Parameters
    ----------
    spike_times : array
        Spike times in seconds
    param : dict
        Parameters dictionary
        
    Returns
    -------
    spike_props : dict
        Dictionary containing spike properties
    """
    spike_props = {
        'mean_firing_rate': np.nan,
        'std_firing_rate': np.nan,
        'fano_factor': np.nan,
        'max_firing_rate': np.nan,
        'min_firing_rate': np.nan
    }
    
    if len(spike_times) < 2:
        return spike_props
    
    # Mean firing rate
    duration = spike_times[-1] - spike_times[0]
    spike_props['mean_firing_rate'] = len(spike_times) / duration
    
    # Firing rate in bins for Fano factor and percentiles
    bin_size = param['fr_bin_size']
    bins = np.arange(spike_times[0], spike_times[-1] + bin_size, bin_size)
    
    if len(bins) > 1:
        spike_counts, _ = np.histogram(spike_times, bins)
        firing_rates = spike_counts / bin_size
        
        if len(firing_rates) > 1:
            # Fano factor
            if np.mean(firing_rates) > 0:
                spike_props['fano_factor'] = np.var(firing_rates) / np.mean(firing_rates)
            
            # Standard deviation and percentiles
            spike_props['std_firing_rate'] = np.std(firing_rates)
            spike_props['max_firing_rate'] = np.percentile(firing_rates, 95)
            spike_props['min_firing_rate'] = np.percentile(firing_rates, 5)
    
    return spike_props


def classify_cells(ephys_properties, param):
    """
    Classify cells based on ephys properties
    
    Parameters
    ----------
    ephys_properties : list
        List of dictionaries containing ephys properties
    param : dict
        Parameters dictionary
        
    Returns
    -------
    cell_types : list
        List containing cell classifications
    """
    n_units = len(ephys_properties)
    
    if param['brain_region'] == 'striatum':
        cell_types = classify_striatum_cells(ephys_properties, param)
    elif param['brain_region'] == 'cortex':
        cell_types = classify_cortex_cells(ephys_properties, param)
    else:
        # Generic classification
        cell_types = ['Unknown'] * n_units
    
    return cell_types


def classify_striatum_cells(ephys_properties, param):
    """
    Classify striatal cell types (MSN, FSI, TAN, UIN)
    
    Parameters
    ----------
    ephys_properties : list
        List of dictionaries containing ephys properties
    param : dict
        Parameters dictionary
        
    Returns
    -------
    cell_types : list
        List containing cell classifications
    """
    n_units = len(ephys_properties)
    cell_types = ['Unknown'] * n_units
    
    for i in range(n_units):
        wf_duration = ephys_properties[i].get('waveform_duration_peak_trough', np.nan)
        firing_rate = ephys_properties[i].get('firing_rate_mean', np.nan)
        cv = ephys_properties[i].get('isi_cv', np.nan)
        
        # Skip if missing critical properties
        if np.isnan(wf_duration) or np.isnan(firing_rate):
            continue
        
        # Note: waveform_duration_peak_trough is already in microseconds
        # Convert to seconds for comparison
        wf_duration_sec = wf_duration / 1e6 if not np.isnan(wf_duration) else np.nan
        
        # FSI: narrow waveform + high firing rate
        if (not np.isnan(wf_duration_sec) and 
            wf_duration_sec < param['fsi_waveform_duration_max'] and 
            firing_rate > param['fsi_firing_rate_min']):
            cell_types[i] = 'FSI'
        
        # TAN: regular firing (low CV) + moderate firing rate
        elif (not np.isnan(cv) and cv < param['tan_cv_max'] and 
              firing_rate > param['tan_firing_rate_min']):
            cell_types[i] = 'TAN'
        
        # MSN: wide waveform + low firing rate
        elif (not np.isnan(wf_duration_sec) and 
              wf_duration_sec > param['msn_waveform_duration_max']):
            cell_types[i] = 'MSN'
        
        # Unidentified interneuron
        else:
            cell_types[i] = 'UIN'
    
    return cell_types


def classify_cortex_cells(ephys_properties, param):
    """
    Classify cortical cell types (narrow-spiking, wide-spiking)
    
    Parameters
    ----------
    ephys_properties : list
        List of dictionaries containing ephys properties
    param : dict
        Parameters dictionary
        
    Returns
    -------
    cell_types : list
        List containing cell classifications
    """
    n_units = len(ephys_properties)
    cell_types = ['Unknown'] * n_units
    
    for i in range(n_units):
        wf_duration = ephys_properties[i].get('waveform_duration_peak_trough', np.nan)
        
        if np.isnan(wf_duration):
            continue
        
        # Convert to seconds
        wf_duration_sec = wf_duration / 1e6
        
        # Narrow-spiking (putative interneurons)
        if wf_duration_sec < param['narrow_waveform_duration_max']:
            cell_types[i] = 'Narrow-spiking'
        
        # Wide-spiking (putative pyramidal)
        elif wf_duration_sec > param['wide_waveform_duration_min']:
            cell_types[i] = 'Wide-spiking'
        
        # Intermediate
        else:
            cell_types[i] = 'Intermediate'
    
    return cell_types


def save_ephys_properties(ephys_properties, cell_types, save_path, param):
    """
    Save ephys properties and cell classifications to file
    
    Parameters
    ----------
    ephys_properties : list
        List of dictionaries containing ephys properties
    cell_types : list
        List containing cell classifications
    save_path : str
        Path to save results
    param : dict
        Parameters dictionary
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Create DataFrame
    df_ephys = pd.DataFrame(ephys_properties)
    df_ephys['cell_type'] = cell_types
    
    # Save to parquet
    ephys_file = os.path.join(save_path, 'templates._bc_ephysProperties.parquet')
    df_ephys.to_parquet(ephys_file, index=False)
    
    # Save parameters
    param_df = pd.DataFrame([param])
    param_file = os.path.join(save_path, '_bc_ephysParameters.parquet')
    param_df.to_parquet(param_file, index=False)
    
    print(f"Ephys properties saved to: {ephys_file}")
    print(f"Parameters saved to: {param_file}")


def load_ephys_properties(save_path):
    """
    Load saved ephys properties and cell classifications
    
    Parameters
    ----------
    save_path : str
        Path to saved files
        
    Returns
    -------
    ephys_properties : pd.DataFrame
        DataFrame containing ephys properties and cell types
    param : dict
        Parameters used for computation
    """
    ephys_file = os.path.join(save_path, 'templates._bc_ephysProperties.parquet')
    param_file = os.path.join(save_path, '_bc_ephysParameters.parquet')
    
    ephys_properties = pd.read_parquet(ephys_file)
    param_df = pd.read_parquet(param_file)
    param = param_df.iloc[0].to_dict()
    
    return ephys_properties, param