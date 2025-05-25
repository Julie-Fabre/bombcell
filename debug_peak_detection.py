#!/usr/bin/env python3
"""
Debug script to analyze peak/trough detection step by step
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import sys
import os

# Add the bombcell module to path
sys.path.append('/home/jf5479/Dropbox/Python/bombcell/pyBombCell')

from bombcell.loading_utils import load_ephys_data
from bombcell.quality_metrics import waveform_shape
from bombcell.default_parameters import get_default_parameters

def debug_single_waveform(waveform, unit_id, param):
    """Debug peak/trough detection for a single waveform"""
    print(f"\n=== Debugging Unit {unit_id} ===")
    print(f"Waveform shape: {waveform.shape}")
    print(f"Waveform length: {len(waveform)}")
    print(f"Waveform min: {np.min(waveform):.4f}, max: {np.max(waveform):.4f}")
    print(f"Non-NaN values: {np.sum(~np.isnan(waveform))}")
    
    min_thresh_detect_peaks_troughs = param["minThreshDetectPeaksTroughs"]
    
    # Apply the same preprocessing as the main function
    if np.size(waveform) == 82:  # KS4
        waveform_proc = waveform.copy()
        waveform_proc[:24] = np.nan
        first_valid_index = 24
        print(f"KS4 waveform detected, masking first 24 samples")
    else:
        waveform_proc = waveform.copy()
        waveform_proc[:4] = np.nan
        first_valid_index = 4
        print(f"Standard waveform, masking first 4 samples")
    
    print(f"After masking - Non-NaN values: {np.sum(~np.isnan(waveform_proc))}")
    
    # Calculate prominence threshold - use nanmax to handle NaN values
    max_abs_value = np.nanmax(np.abs(waveform_proc))
    min_prominence = min_thresh_detect_peaks_troughs * max_abs_value
    print(f"Min prominence threshold: {min_prominence:.6f}")
    print(f"Max abs value: {max_abs_value:.6f}")
    
    # Find troughs first (invert waveform)
    inverted = waveform_proc * -1
    print(f"Inverted waveform min: {np.nanmin(inverted):.4f}, max: {np.nanmax(inverted):.4f}")
    
    trough_locs, trough_dict = find_peaks(inverted, prominence=min_prominence, width=0)
    TRS = inverted[trough_locs]
    print(f"Found {len(trough_locs)} troughs with prominence >= {min_prominence:.6f}")
    print(f"Trough locations: {trough_locs}")
    print(f"Trough values (inverted): {TRS}")
    print(f"Trough prominences: {trough_dict.get('prominences', [])}")
    
    # If no troughs, use minimum
    if len(TRS) == 0:
        TRS = np.array([np.nanmin(waveform_proc)])
        trough_locs = np.array([np.nanargmin(waveform_proc)])
        print(f"No troughs found, using minimum at {trough_locs[0]} with value {TRS[0]:.4f}")
    
    n_troughs = len(TRS)
    
    # Get main trough
    if len(TRS) > 0:
        mainTrough_idx = np.nanargmax(TRS)
        trough_loc = trough_locs[mainTrough_idx]
        print(f"Main trough at index {trough_loc}, value {waveform_proc[trough_loc]:.4f}")
    else:
        trough_loc = len(waveform_proc) // 2
        print(f"Using fallback trough location: {trough_loc}")
    
    # Find peaks before trough
    PKS_before = np.array([])
    peakLocs_before = np.array([])
    
    if trough_loc > 3:
        peakLocs_before, peak_dict_before = find_peaks(
            waveform_proc[:trough_loc], prominence=min_prominence, width=0
        )
        PKS_before = waveform_proc[peakLocs_before]
        print(f"Found {len(PKS_before)} peaks before trough (locations: {peakLocs_before})")
        print(f"Peak values before: {PKS_before}")
        print(f"Peak prominences before: {peak_dict_before.get('prominences', [])}")
    
    # Find peaks after trough
    PKS_after = np.array([])
    peakLocs_after = np.array([])
    
    if len(waveform_proc) - trough_loc > 3:
        peakLocs_after_temp, peak_dict_after = find_peaks(
            waveform_proc[trough_loc:], prominence=min_prominence, width=0
        )
        PKS_after = waveform_proc[trough_loc:][peakLocs_after_temp]
        peakLocs_after = peakLocs_after_temp + trough_loc
        print(f"Found {len(PKS_after)} peaks after trough (locations: {peakLocs_after})")
        print(f"Peak values after: {PKS_after}")
        print(f"Peak prominences after: {peak_dict_after.get('prominences', [])}")
    
    # Handle forced peaks if needed
    usedMaxBefore = 0
    usedMaxAfter = 0
    
    if len(PKS_before) == 0:
        print("No peaks before trough, trying lower prominence...")
        if trough_loc > 3:
            lower_prominence = 0.01 * np.nanmax(np.abs(waveform_proc))
            print(f"Using lower prominence: {lower_prominence:.6f}")
            peakLocs_before_temp, peak_dict_before_temp = find_peaks(
                waveform_proc[:trough_loc], prominence=lower_prominence, width=0
            )
            PKS_before = waveform_proc[peakLocs_before_temp]
            peakLocs_before = peakLocs_before_temp
            print(f"Found {len(PKS_before)} peaks with lower prominence")
            
        if len(PKS_before) == 0:
            print("Still no peaks, using maximum value before trough")
            waveform_segment = waveform_proc[:trough_loc]
            if not np.all(np.isnan(waveform_segment)) and len(waveform_segment) > 0:
                max_idx = np.nanargmax(waveform_segment)
                PKS_before = np.array([waveform_segment[max_idx]])
                peakLocs_before = np.array([max_idx])
                print(f"Forced peak before at {max_idx} with value {PKS_before[0]:.4f}")
        usedMaxBefore = 1
    
    if len(PKS_after) == 0:
        print("No peaks after trough, trying lower prominence...")
        if len(waveform_proc) - trough_loc > 3:
            lower_prominence = 0.01 * np.nanmax(np.abs(waveform_proc))
            peakLocs_after_temp, peak_dict_after_temp = find_peaks(
                waveform_proc[trough_loc:], prominence=lower_prominence, width=0
            )
            PKS_after = waveform_proc[trough_loc:][peakLocs_after_temp]
            peakLocs_after = peakLocs_after_temp + trough_loc
            print(f"Found {len(PKS_after)} peaks with lower prominence")
            
        if len(PKS_after) == 0:
            print("Still no peaks, using maximum value after trough")
            waveform_segment = waveform_proc[trough_loc:]
            if not np.all(np.isnan(waveform_segment)) and len(waveform_segment) > 0:
                max_idx = np.nanargmax(waveform_segment)
                PKS_after = np.array([waveform_segment[max_idx]])
                peakLocs_after = np.array([trough_loc + max_idx])
                print(f"Forced peak after at {trough_loc + max_idx} with value {PKS_after[0]:.4f}")
        usedMaxAfter = 1
    
    # Combine peaks
    if len(PKS_before) > 0 and len(PKS_after) > 0:
        PKS = np.concatenate([PKS_before, PKS_after])
        peakLocs = np.concatenate([peakLocs_before, peakLocs_after])
    elif len(PKS_before) > 0:
        PKS = PKS_before
        peakLocs = peakLocs_before
    else:
        PKS = PKS_after
        peakLocs = peakLocs_after
    
    n_peaks = len(PKS)
    
    print(f"\nFINAL RESULTS:")
    print(f"Number of peaks: {n_peaks}")
    print(f"Number of troughs: {n_troughs}")
    print(f"Peak locations: {peakLocs}")
    print(f"Peak values: {PKS}")
    print(f"Trough locations: {trough_locs}")
    print(f"Trough values (original): {waveform_proc[trough_locs] if len(trough_locs) > 0 else []}")
    
    return n_peaks, n_troughs, peakLocs, trough_locs, PKS, TRS

def main():
    # Load the toy data
    data_dir = "/home/jf5479/Dropbox/Python/bombcell/pyBombCell/Demos/toy_data"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        return
    
    print("Loading ephys data...")
    (spike_times_samples, spike_templates, template_waveforms, 
     template_amplitudes, pc_features, pc_features_idx, channel_positions) = load_ephys_data(data_dir)
    
    # Get parameters - use a dummy path since we only need the defaults
    param = get_default_parameters(data_dir)
    
    # Compute maxChannels
    maxChannels = np.nanargmax(
        np.max(np.abs(template_waveforms), axis=1), axis=1
    )
    
    
    print(f"Template waveforms shape: {template_waveforms.shape}")
    print(f"Number of units: {template_waveforms.shape[0]}")
    
    # Debug first few units in detail
    units_to_debug = [0, 1, 2, 7, 14]  # Sample different units
    
    results = {}
    for unit_id in units_to_debug:
        if unit_id >= template_waveforms.shape[0]:
            continue
            
        waveform = template_waveforms[unit_id, :, maxChannels[unit_id]]
        n_peaks, n_troughs, peak_locs, trough_locs, PKS, TRS = debug_single_waveform(
            waveform, unit_id, param
        )
        
        results[unit_id] = {
            'n_peaks': n_peaks,
            'n_troughs': n_troughs,
            'peak_locs': peak_locs,
            'trough_locs': trough_locs,
            'waveform': waveform
        }
        
        # Create a plot for this waveform
        plt.figure(figsize=(12, 6))
        plt.plot(waveform, 'k-', label='Original waveform', alpha=0.7)
        
        # Mark peaks
        if len(peak_locs) > 0:
            plt.plot(peak_locs, waveform[peak_locs], 'ro', markersize=8, label=f'Peaks ({len(peak_locs)})')
        
        # Mark troughs 
        if len(trough_locs) > 0:
            plt.plot(trough_locs, waveform[trough_locs], 'bo', markersize=8, label=f'Troughs ({len(trough_locs)})')
        
        plt.title(f'Unit {unit_id}: {n_peaks} peaks, {n_troughs} troughs')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'/home/jf5479/Dropbox/Python/bombcell/debug_unit_{unit_id}.png', dpi=150)
        plt.close()
    
    print(f"\n=== SUMMARY ===")
    for unit_id, result in results.items():
        print(f"Unit {unit_id}: {result['n_peaks']} peaks, {result['n_troughs']} troughs")
    
    # Check if all units have the same counts
    peak_counts = [result['n_peaks'] for result in results.values()]
    trough_counts = [result['n_troughs'] for result in results.values()]
    
    print(f"\nPeak count variation: {set(peak_counts)}")
    print(f"Trough count variation: {set(trough_counts)}")
    
    if len(set(peak_counts)) == 1:
        print("⚠️  WARNING: All units have the same peak count!")
    if len(set(trough_counts)) == 1:
        print("⚠️  WARNING: All units have the same trough count!")

if __name__ == "__main__":
    main()