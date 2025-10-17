import numpy as np
import pandas as pd
from scipy import signal, optimize, stats
from scipy.sparse import csr_matrix
import os
import psutil
import gc
from pathlib import Path
from tqdm.auto import tqdm
from .ccg_fast import acg as compute_acg_fast

__all__ = [
    'run_all_ephys_properties',
    'get_ephys_parameters', 
    'ephys_prop_values',
    'compute_all_ephys_properties',
    'compute_acg_properties',
    'compute_isi_properties',
    'compute_waveform_properties',
    'save_ephys_properties'
]

# Path handling utility function
def path_handler(path):
    """Simple path handler utility"""
    return Path(path)


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def log_memory_usage(step_name, verbose=True):
    """Log memory usage at different steps"""
    # Memory logging removed per user request
    pass


def run_all_ephys_properties(ephys_path, param=None, save_path=None):
    """
    Main function to compute all ephys properties - like MATLAB runAllEphysProperties
    
    Parameters
    ----------
    ephys_path : str
        Path to ephys data directory
    param : dict, optional
        Quality metrics parameters dictionary (for compatibility)
    save_path : str, optional
        Path to save results
    
    Returns
    -------
    ephys_properties : list
        List of dictionaries containing ephys properties for each unit
    ephys_param : dict
        Ephys properties parameters (separate from quality metrics)
    """
    if save_path is None:
        save_path = Path(ephys_path) / "bombcell"
    
    # Get SEPARATE ephys properties parameters (like MATLAB ephysPropValues)
    raw_file = param.get('raw_data_file', None) if param else None
    meta_file = param.get('ephys_meta_file', None) if param else None
    ephys_param = get_ephys_parameters(ephys_path, raw_file, meta_file)
    
    # Compute all ephys properties using ephys_param
    ephys_properties = compute_all_ephys_properties(ephys_path, ephys_param, save_path)
    
    # Save results
    save_ephys_properties(ephys_properties, save_path, ephys_param)
    
    return ephys_properties, ephys_param


def get_ephys_parameters(ephys_path, raw_file=None, meta_file=None):
    """
    Get ephys properties parameters - SEPARATE from quality metrics parameters
    Mimics MATLAB ephysPropValues.m exactly
    
    Parameters
    ----------
    ephys_path : str or Path
        Path to ephys data directory
    raw_file : str, optional
        Path to raw data file
    meta_file : str, optional
        Path to meta file
        
    Returns
    -------
    ephys_param : dict
        Ephys properties parameters dictionary
    """
    ephys_param = {}
    
    # Basic settings
    ephys_param['plotDetails'] = False
    ephys_param['verbose'] = True
    
    # Recording parameters
    ephys_param['ephys_sample_rate'] = 30000
    ephys_param['nChannels'] = 385
    ephys_param['nSyncChannels'] = 1
    ephys_param['ephysMetaFile'] = meta_file if meta_file else 'NaN'
    ephys_param['rawFile'] = raw_file if raw_file else 'NaN'
    ephys_param['gain_to_uV'] = np.nan
    
    # Duplicate spikes parameters
    ephys_param['removeDuplicateSpikes'] = False
    ephys_param['duplicateSpikeWindow_s'] = 0.00001
    ephys_param['saveSpikes_withoutDuplicates'] = True
    ephys_param['recomputeDuplicateSpikes'] = False
    
    # Raw waveform parameters
    ephys_param['detrendWaveform'] = True
    ephys_param['nRawSpikesToExtract'] = 100
    ephys_param['saveMultipleRaw'] = False
    ephys_param['decompressData'] = False
    ephys_param['spikeWidth'] = 82
    ephys_param['extractRaw'] = False
    ephys_param['probeType'] = 1
    ephys_param['detrendWaveforms'] = False
    ephys_param['reextractRaw'] = False
    
    # ACG parameters - EXACT MATLAB values
    ephys_param['ACGbinSize'] = 0.001  # 1ms bins
    ephys_param['ACGduration'] = 1.0   # 1 second duration
    
    # Proportion Long ISI
    ephys_param['longISI'] = 2.0  # 2 seconds
    
    # Waveform parameters
    ephys_param['minThreshDetectPeaksTroughs'] = 0.2
    ephys_param['maxWvBaselineFraction'] = 0.3
    ephys_param['normalizeSpDecay'] = True
    ephys_param['spDecayLinFit'] = True
    ephys_param['minWidthFirstPeak'] = 4
    ephys_param['minMainPeakToTroughRatio'] = 10
    ephys_param['minWidthMainTrough'] = 5
    ephys_param['firstPeakRatio'] = 3
    
    # Cell classification parameters - EXACT MATLAB values
    # Striatum
    ephys_param['propISI_CP_threshold'] = 0.1
    ephys_param['templateDuration_CP_threshold'] = 400  # microseconds
    ephys_param['postSpikeSup_CP_threshold'] = 40       # milliseconds
    
    # Cortex
    ephys_param['templateDuration_Ctx_threshold'] = 400  # microseconds
    
    # Analysis parameters
    ephys_param['min_spikes_for_stats'] = 100  # minimum spikes needed for analysis
    ephys_param['fr_bin_size'] = 1.0  # bin size in seconds for firing rate analysis
    ephys_param['min_recording_duration'] = 60.0  # minimum recording duration in seconds
    
    # Memory management parameters
    ephys_param['max_spikes_acg'] = 8000  # maximum spikes for ACG computation
    ephys_param['acg_chunk_size'] = 2000  # chunk size for ACG processing
    
    # Classification parameters for striatum (default values)
    ephys_param['fsi_waveform_duration_max'] = 400e-6  # 400 microseconds in seconds
    ephys_param['fsi_firing_rate_min'] = 10.0  # Hz
    ephys_param['tan_cv_max'] = 0.5
    ephys_param['tan_firing_rate_min'] = 2.0  # Hz
    ephys_param['msn_waveform_duration_max'] = 400e-6  # 400 microseconds in seconds
    
    # Classification parameters for cortex
    ephys_param['narrow_waveform_duration_max'] = 400e-6  # 400 microseconds in seconds
    
    # Brain region (can be set by user)
    ephys_param['brain_region'] = 'unknown'  # 'striatum', 'cortex', or 'unknown'
    
    return ephys_param


def ephys_prop_values(param):
    """
    DEPRECATED: Use get_ephys_parameters() instead
    This function exists for backward compatibility
    """
    # Convert quality metrics param to ephys param
    ephys_path = param.get('ephysKilosortPath', '.')
    raw_file = param.get('raw_data_file', None)
    meta_file = param.get('ephys_meta_file', None)
    
    return get_ephys_parameters(ephys_path, raw_file, meta_file)


def compute_all_ephys_properties(ephys_path, param, save_path):
    """
    Compute all ephys properties for all units
    
    
    Parameters
    ----------
    ephys_path : str
        Path to ephys data
    param : dict
        Parameters dictionary with memory management settings
    save_path : str
        Path to save results
        
    Returns
    -------
    ephys_properties : list
        List of dictionaries containing all computed properties
    """
    import gc
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
    
    for i, unit_id in enumerate(unique_units):
        properties = {
            # Unit ID 
            'unit_id': unit_id,
            # ACG properties
            'postSpikeSuppression': np.nan,
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
            # MATLAB-compatible names (initialize now, assign later)
            'postSpikeSuppression_ms': np.nan,
            'propLongISI': np.nan,
            'waveformDuration_peakTrough_us': np.nan,
        }
        ephys_properties.append(properties)
    
    print(f"Computing ephys properties for {n_units} units ...")
    log_memory_usage("Initial memory", param.get('verbose', True))
    
    # Get sampling rate
    sampling_rate = param.get('ephys_sample_rate', 30000)
    
    # Convert spike times to seconds if needed
    if np.max(spike_times) > 1e6:  # Likely in samples
        spike_times_sec = spike_times / sampling_rate
    else:
        spike_times_sec = spike_times
    
    log_memory_usage("After spike time conversion", param.get('verbose', True))

    
    # Unit-by-unit processing with progress tracking
    for i, unit_id in enumerate(tqdm(unique_units, desc="Computing ephys properties")):
        
        # Get spikes for this unit
        unit_spikes = spike_times_sec[spike_clusters == unit_id]
        
        if len(unit_spikes) < param['min_spikes_for_stats']:
            # Clean up and continue to next unit
            del unit_spikes
            gc.collect()
            continue
            
        # Get template for this unit - avoid loading all at once
        unit_template = template_waveforms[unit_id].copy()  # Copy to avoid memory issues

        
        # Initialize variables for cleanup
        acg_props = {}
        isi_props = {}
        wf_props = {}
        spike_props = {}
        
        try:
            # Compute ACG properties
            acg_props = compute_acg_properties(unit_spikes, param)
            ephys_properties[i]['postSpikeSuppression'] = acg_props.get('post_spike_suppression_ratio', np.nan)
            ephys_properties[i]['acg_tau_rise'] = acg_props.get('tau_rise_ms', np.nan)
            ephys_properties[i]['acg_tau_decay'] = acg_props.get('tau_decay_ms', np.nan)
            
            # MATLAB-compatible property names
            pss_ms = acg_props.get('post_spike_suppression_ms', np.nan)
            ephys_properties[i]['postSpikeSuppression_ms'] = pss_ms
            
            # Clean up ACG computation results immediately
            del acg_props
            gc.collect()
            
            # Compute ISI properties
            isi_props = compute_isi_properties(unit_spikes, param)
            ephys_properties[i]['isi_cv'] = isi_props.get('cv', np.nan)
            ephys_properties[i]['isi_cv2'] = isi_props.get('cv2', np.nan)
            ephys_properties[i]['isi_skewness'] = isi_props.get('isi_skewness', np.nan)
            ephys_properties[i]['prop_long_isi'] = isi_props.get('prop_long_isi', np.nan)
            
            # MATLAB-compatible property names
            ephys_properties[i]['propLongISI'] = isi_props.get('prop_long_isi', np.nan)
            
            # Clean up ISI computation results immediately
            del isi_props
            gc.collect()
            
            # Compute waveform properties
            wf_props = compute_waveform_properties(unit_template, param, sampling_rate)
            ephys_properties[i]['waveform_duration_peak_trough'] = wf_props.get('waveform_duration_us', np.nan)
            ephys_properties[i]['waveform_half_width'] = wf_props.get('half_width_ms', np.nan)
            ephys_properties[i]['peak_to_trough_ratio'] = wf_props.get('peak_to_trough_ratio', np.nan)
            
            # MATLAB-compatible property names (assign AFTER computation)
            ephys_properties[i]['waveformDuration_peakTrough_us'] = ephys_properties[i]['waveform_duration_peak_trough']
            ephys_properties[i]['n_peaks'] = wf_props.get('n_peaks', np.nan)
            ephys_properties[i]['n_troughs'] = wf_props.get('n_troughs', np.nan)
            
            # Clean up waveform computation results immediately
            del wf_props
            gc.collect()
            
            # Compute spike properties
            spike_props = compute_spike_properties(unit_spikes, param)
            ephys_properties[i]['firing_rate_mean'] = spike_props.get('mean_firing_rate', np.nan)
            ephys_properties[i]['firing_rate_std'] = spike_props.get('std_firing_rate', np.nan)
            ephys_properties[i]['fano_factor'] = spike_props.get('fano_factor', np.nan)
            
            # Clean up spike computation results immediately
            del spike_props
            gc.collect()
            
        except (MemoryError, np.core._exceptions._ArrayMemoryError) as e:
            print(f"Error processing unit {unit_id}: {e}")
            print("Skipping this unit and continuing...")
            # Set all properties to NaN for this unit
            for key in ephys_properties[i].keys():
                if key != 'unit_id':
                    ephys_properties[i][key] = np.nan
            # Force aggressive cleanup
            del acg_props, isi_props, wf_props, spike_props
            gc.collect()
            gc.collect()  # Double collection for aggressive cleanup
        except Exception as e:
            print(f"Unexpected error processing unit {unit_id}: {e}")
            print("Skipping this unit and continuing...")
            # Set all properties to NaN for this unit
            for key in ephys_properties[i].keys():
                if key != 'unit_id':
                    ephys_properties[i][key] = np.nan
            # Clean up variables
            del acg_props, isi_props, wf_props, spike_props
            gc.collect()
        
        # Immediate cleanup after each unit - CRITICAL for memory management
        del unit_spikes, unit_template
        gc.collect()
        
        # Memory monitoring every 10 units
        if (i + 1) % 10 == 0:
            current_memory = get_memory_usage()
            
            # Force additional cleanup if memory is getting high
            if current_memory > 3000:  # > 3GB
                gc.collect()
                gc.collect()  # Double collection for aggressive cleanup
    
    print("Ephys properties computation complete!")
    log_memory_usage("Final memory", param.get('verbose', True))
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
    # Compute ACG using MATLAB parameter names
    acg_bin_size = param.get('ACGbinSize', 0.001)
    acg_duration = param.get('ACGduration', 1.0)
    acg, lags = compute_acg(spike_times, acg_bin_size, acg_duration, param)
    
    # Initialize output
    acg_props = {
        'post_spike_suppression_ratio': np.nan,
        'post_spike_suppression_ms': np.nan,
        'tau_rise_ms': np.nan,
        'tau_decay_ms': np.nan
    }
    
    if len(acg) == 0 or len(spike_times) < 10:
        return acg_props
    
    # Ensure ACG has some counts
    if np.sum(acg) == 0:
        return acg_props
    
    # Find center bin
    center_idx = len(acg) // 2
    
    # Post-spike suppression: ratio of minimum in first 10ms to baseline
    bin_size_sec = acg_bin_size
    post_bins = max(1, int(0.01 / bin_size_sec))  # 10ms in bins, at least 1
    baseline_start_bins = min(max(20, int(0.05 / bin_size_sec)), len(acg)//4)  # Start baseline at 50ms, but adapt to ACG length
    
    # Make sure we have enough bins for both regions
    if center_idx + baseline_start_bins + 10 < len(acg):
        # Suppression region: 1-10ms after center (skip center bin to avoid refractory period)
        supp_start = center_idx + 1
        supp_end = min(center_idx + post_bins + 1, len(acg))
        suppression_region = acg[supp_start:supp_end]
        
        # Baseline region: 50ms to end (use symmetric baseline on both sides)
        baseline_left = acg[max(0, center_idx-baseline_start_bins):center_idx-post_bins]
        baseline_right = acg[center_idx+baseline_start_bins:]
        baseline_region = np.concatenate([baseline_left, baseline_right]) if len(baseline_left) > 0 else baseline_right
        
        if len(suppression_region) > 0 and len(baseline_region) > 5:
            min_val = np.min(suppression_region)
            baseline_mean = np.mean(baseline_region)
            
            if baseline_mean > 0:
                postSpikeSuppression_ratio = min_val / baseline_mean
                acg_props['post_spike_suppression_ratio'] = postSpikeSuppression_ratio
                
                # Compute post-spike suppression EXACTLY as in MATLAB Nature paper
                # MATLAB: postSpikeSup = find(thisACG(500:1000) >= nanmean(thisACG(600:900)));
                # Convert MATLAB indices to Python (MATLAB uses 1-based indexing)
                
                # Calculate equivalent indices for our ACG
                # MATLAB ACG is 1000 bins (1s duration, 1ms bins), center at 500
                # Our ACG has different binning, so scale appropriately
                acg_duration_sec = acg_duration
                total_bins = len(acg)
                
                if total_bins >= 100:  # Need enough bins for the computation
                    # Map MATLAB indices to our indices
                    # MATLAB 500:1000 = 0:500ms post-spike (500 bins)
                    # MATLAB 600:900 = 100:400ms post-spike (300 bins for baseline)
                    
                    # Scale to our bin structure
                    post_start_idx = center_idx  # Start at center (0ms)
                    post_end_idx = min(center_idx + int(0.5 / bin_size_sec), len(acg))  # Up to 500ms
                    
                    baseline_start_idx = center_idx + int(0.1 / bin_size_sec)  # 100ms post-spike
                    baseline_end_idx = min(center_idx + int(0.4 / bin_size_sec), len(acg))  # 400ms post-spike
                    
                    if (post_end_idx > post_start_idx and 
                        baseline_end_idx > baseline_start_idx and
                        baseline_end_idx <= len(acg)):
                        
                        # Compute baseline as in MATLAB
                        baseline_region = acg[baseline_start_idx:baseline_end_idx]
                        baseline_mean_matlab = np.nanmean(baseline_region)
                        
                        # Find first point where ACG >= baseline mean (as in MATLAB)
                        post_region = acg[post_start_idx:post_end_idx]
                        recovery_indices = np.where(post_region >= baseline_mean_matlab)[0]
                        
                        if len(recovery_indices) > 0:
                            # Time to first recovery point (in ms)
                            recovery_bin = recovery_indices[0]  # First index in post_region
                            postSpikeSup_ms = recovery_bin * bin_size_sec * 1000  # Convert to ms
                            acg_props['post_spike_suppression_ms'] = postSpikeSup_ms
                        else:
                            acg_props['post_spike_suppression_ms'] = np.nan
                    else:
                        acg_props['post_spike_suppression_ms'] = np.nan
                else:
                    acg_props['post_spike_suppression_ms'] = np.nan
            else:
                acg_props['post_spike_suppression_ratio'] = 0.0
                acg_props['post_spike_suppression_ms'] = 50.0  # Default
    
    # Simple tau estimation based on ACG shape in first 20ms
    try:
        # Use correct parameter name
        acg_bin_size_used = param.get('ACGbinSize', 0.001)
        if center_idx + int(0.02/acg_bin_size_used) < len(acg):
            fit_region = acg[center_idx+1:center_idx+int(0.02/acg_bin_size_used)]  # 0-20ms region
            
            if len(fit_region) > 5:
                # Find rise time: time to reach half maximum
                max_val = np.max(fit_region)
                half_max = max_val * 0.5
                
                if max_val > 0:
                    # Find first bin that exceeds half max
                    rise_idx = np.where(fit_region >= half_max)[0]
                    if len(rise_idx) > 0:
                        acg_props['tau_rise_ms'] = rise_idx[0] * acg_bin_size_used * 1000
                
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
                            acg_props['tau_decay_ms'] = tau_bins * acg_bin_size_used * 1000
                        except:
                            pass
    except:
        pass
    
    return acg_props



def compute_acg(spike_times, bin_size, duration, param=None):
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
    param : dict, optional
        Parameters dictionary containing memory management settings
        
    Returns
    -------
    acg : array
        Auto-correlogram counts
    lags : array
        Lag times in seconds
    """
    bin_size_ms = bin_size * 1000  # Convert to ms
    duration_ms = duration * 1000  # Convert to ms
    
    # Get sampling rate from param or use default
    fs = param.get('ephys_sampling_rate', 30000) if param is not None else 30000
    
    # Convert spike times from seconds to samples
    spike_times_samples = (spike_times * fs).astype(np.uint64)
    
    # Use acg from ccg_fast
    acg_result = compute_acg_fast(spike_times_samples, cbin=bin_size_ms, cwin=duration_ms, 
                                  fs=fs, normalize="counts", cache_results=False)
    
    # Generate lag times to match expected output
    n_bins = len(acg_result)
    lags_ms = np.linspace(-duration_ms/2, duration_ms/2, n_bins)
    lags = lags_ms / 1000  # Convert back to seconds
    
    return acg_result, lags


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
    
    # Proportion of long ISIs - use MATLAB parameter name
    long_isi_threshold = param.get('longISI', 2.0)  # MATLAB: paramEP.longISI = 2
    long_isis = isis > long_isi_threshold
    isi_props['prop_long_isi'] = np.mean(long_isis)
    
    # Coefficient of variation
    if len(isis) > 0:
        isi_props['cv'] = np.std(isis) / np.mean(isis)
        isi_props['isi_skewness'] = stats.skew(isis)
    
    # CV2 (local coefficient of variation) - vectorized for speed
    if len(isis) > 1:
        mean_isis = (isis[:-1] + isis[1:]) / 2
        diff_isis = np.abs(isis[1:] - isis[:-1])
        valid_mask = mean_isis > 0
        if np.any(valid_mask):
            cv2_values = diff_isis[valid_mask] / mean_isis[valid_mask]
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
    
    # Peak-to-trough duration using quality_metrics approach
    # Find the maximum absolute value location in the waveform
    max_waveform_location = np.nanargmax(np.abs(waveform))
    max_waveform_value = waveform[max_waveform_location]  # signed value
    
    if max_waveform_value > 0:  # positive peak
        peak_loc_for_duration = max_waveform_location
        # Find the minimum (trough) after the peak
        trough_loc_for_duration = np.nanargmin(waveform[peak_loc_for_duration:])
        trough_loc_for_duration = trough_loc_for_duration + peak_loc_for_duration  # adjust for truncated waveform
    else:  # negative peak (trough is the max absolute value)
        trough_loc_for_duration = max_waveform_location
        # Find the maximum (peak) after the trough
        peak_loc_for_duration = np.nanargmax(waveform[trough_loc_for_duration:])
        peak_loc_for_duration = peak_loc_for_duration + trough_loc_for_duration  # adjust for truncated waveform
    
    # Calculate duration in microseconds
    wf_props['waveform_duration_us'] = (
        10**6
        * np.abs(trough_loc_for_duration - peak_loc_for_duration)
        / sampling_rate
    )
    
    # Peak-to-trough ratio using the identified peak and trough
    if not np.isnan(peak_loc_for_duration) and not np.isnan(trough_loc_for_duration):
        wf_props['peak_to_trough_ratio'] = abs(waveform[peak_loc_for_duration]) / abs(waveform[trough_loc_for_duration])
    
    # For compatibility, still find peaks and troughs for counting
    if len(peaks) > 0 and len(troughs) > 0:
        # First peak-to-trough ratio
        if len(peaks) > 1:
            main_trough = troughs[np.argmax(-waveform[troughs])]
            first_peak = peaks[0] if peaks[0] < main_trough else peaks[np.argmax(waveform[peaks])]
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
    Compute spike timing based properties - memory optimized
    
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
    if duration > 0:
        spike_props['mean_firing_rate'] = len(spike_times) / duration
    
    # Firing rate in bins for Fano factor and percentiles
    bin_size = param.get('fr_bin_size', 1.0)
    n_bins = max(1, int(duration / bin_size))
    
    # Limit number of bins to avoid memory issues
    max_bins = 10000
    if n_bins > max_bins:
        bin_size = duration / max_bins
        n_bins = max_bins
    
    if n_bins > 1:
        bins = np.linspace(spike_times[0], spike_times[-1], n_bins + 1)
        spike_counts, _ = np.histogram(spike_times, bins)
        firing_rates = spike_counts / bin_size
        
        if len(firing_rates) > 1:
            # Fano factor
            mean_fr = np.mean(firing_rates)
            if mean_fr > 0:
                spike_props['fano_factor'] = np.var(firing_rates) / mean_fr
            
            # Standard deviation and percentiles
            spike_props['std_firing_rate'] = np.std(firing_rates)
            spike_props['max_firing_rate'] = np.percentile(firing_rates, 95)
            spike_props['min_firing_rate'] = np.percentile(firing_rates, 5)
    
    return spike_props



def save_ephys_properties(ephys_properties, save_path, ephys_param):
    """
    Save ephys properties and parameters to file
    
    Parameters
    ----------
    ephys_properties : list
        List of dictionaries containing ephys properties
    save_path : str
        Path to save results
    ephys_param : dict
        Ephys parameters dictionary
    """
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)
    
    # Create DataFrame
    df_ephys = pd.DataFrame(ephys_properties)
    
    # Save to parquet
    ephys_file = os.path.join(save_path, 'templates._bc_ephysProperties.parquet')
    df_ephys.to_parquet(ephys_file, index=False)
    
    # Save parameters
    param_df = pd.DataFrame([ephys_param])
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