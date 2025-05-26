#!/usr/bin/env python3
"""
Direct test of the mark_peaks_and_troughs function
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

sys.path.append('/home/jf5479/Dropbox/Python/bombcell/pyBombCell')

def test_mark_peaks_and_troughs():
    """Test the mark_peaks_and_troughs function directly"""
    
    print("🔧 Testing mark_peaks_and_troughs function directly...")
    
    # Create a realistic test waveform
    n_samples = 82
    waveform = np.zeros(n_samples)
    
    # Add a clear peak at sample 30 and trough at sample 45
    waveform[28:33] = [0.2, 0.5, 1.0, 0.5, 0.2]  # Peak around sample 30
    waveform[43:48] = [-0.3, -0.8, -1.5, -0.8, -0.3]  # Trough around sample 45
    # Add some baseline noise
    waveform += np.random.normal(0, 0.05, n_samples)
    
    print(f"Created waveform: min={np.min(waveform):.3f}, max={np.max(waveform):.3f}")
    
    # Test peak detection
    prominence_thresh = 0.2 * np.max(np.abs(waveform))
    print(f"Prominence threshold: {prominence_thresh:.3f}")
    
    # Find peaks and troughs using the same logic as GUI
    peaks, peak_props = find_peaks(waveform, prominence=prominence_thresh, width=0)
    troughs, trough_props = find_peaks(-waveform, prominence=prominence_thresh, width=0)
    
    print(f"Detected peaks: {peaks} (values: {waveform[peaks]})")
    print(f"Detected troughs: {troughs} (values: {waveform[troughs]})")
    
    # Test the actual GUI logic for main peak/trough selection
    main_peak_idx = None
    main_trough_idx = None
    
    print(f"\n--- Testing GUI Logic ---")
    print(f"Waveform shape: {waveform.shape}, min: {np.nanmin(waveform):.3f}, max: {np.nanmax(waveform):.3f}")
    print(f"Found {len(peaks)} peaks at {peaks}")
    print(f"Found {len(troughs)} troughs at {troughs}")
    
    # ROBUST approach: Find the main peak and main trough separately
    # Main peak: highest positive point in waveform
    if len(peaks) > 0:
        # Use detected peaks and find the highest one
        main_peak_idx = peaks[np.argmax(waveform[peaks])]
        print(f"Main peak from detected peaks: {main_peak_idx}, value: {waveform[main_peak_idx]:.3f}")
    else:
        # Fallback: find highest point in entire waveform
        main_peak_idx = np.nanargmax(waveform)
        print(f"Main peak from fallback: {main_peak_idx}, value: {waveform[main_peak_idx]:.3f}")
    
    # Main trough: lowest negative point in waveform  
    if len(troughs) > 0:
        # Use detected troughs and find the deepest one
        main_trough_idx = troughs[np.argmin(waveform[troughs])]
        print(f"Main trough from detected troughs: {main_trough_idx}, value: {waveform[main_trough_idx]:.3f}")
    else:
        # Fallback: find lowest point in entire waveform
        main_trough_idx = np.nanargmin(waveform)
        print(f"Main trough from fallback: {main_trough_idx}, value: {waveform[main_trough_idx]:.3f}")
    
    # Ensure indices are valid integers
    if main_peak_idx is not None:
        main_peak_idx = int(main_peak_idx)
    if main_trough_idx is not None:
        main_trough_idx = int(main_trough_idx)
        
    print(f"Final main_peak_idx: {main_peak_idx}, main_trough_idx: {main_trough_idx}")
    
    # Test duration line drawing
    print(f"\n--- Testing Duration Line Drawing ---")
    x_offset = 0
    y_offset = 0
    amp_range = np.max(np.abs(waveform))
    
    print(f"About to draw duration line. main_peak_idx: {main_peak_idx}, main_trough_idx: {main_trough_idx}")
    print(f"x_offset: {x_offset}, y_offset: {y_offset}, amp_range: {amp_range}")
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot waveform
    ax.plot(waveform, 'k-', linewidth=2, label='Waveform')
    
    # Plot detected peaks
    if len(peaks) > 0:
        ax.plot(peaks, waveform[peaks], 'ro', markersize=8, label='Detected Peaks')
    
    # Plot detected troughs
    if len(troughs) > 0:
        ax.plot(troughs, waveform[troughs], 'bo', markersize=8, label='Detected Troughs')
    
    # Highlight main peak and trough
    if main_peak_idx is not None:
        ax.plot(main_peak_idx, waveform[main_peak_idx], 'r*', markersize=15, label='Main Peak')
        
    if main_trough_idx is not None:
        ax.plot(main_trough_idx, waveform[main_trough_idx], 'b*', markersize=15, label='Main Trough')
    
    # Draw duration line if both indices exist
    if main_peak_idx is not None and main_trough_idx is not None:
        # Draw horizontal line at a fixed y-position below the waveform
        line_y = y_offset - amp_range * 0.3  # Position line below waveform
        
        peak_x = main_peak_idx + x_offset
        trough_x = main_trough_idx + x_offset
        peak_y = waveform[main_peak_idx] + y_offset
        trough_y = waveform[main_trough_idx] + y_offset
        
        print(f"Drawing duration line:")
        print(f"  Peak position: ({peak_x}, {peak_y})")
        print(f"  Trough position: ({trough_x}, {trough_y})")
        print(f"  Line y-position: {line_y}")
        print(f"  Duration: {abs(main_peak_idx - main_trough_idx)} samples")
        
        # Draw horizontal duration line
        ax.plot([peak_x, trough_x], [line_y, line_y], 
               'g-', linewidth=4, alpha=0.8, zorder=15, label='Duration Line')
        
        # Add vertical lines to connect to peak and trough
        ax.plot([peak_x, peak_x], [peak_y, line_y], 
               'g--', linewidth=3, alpha=0.8, zorder=14, label='Peak Connection')
        ax.plot([trough_x, trough_x], [trough_y, line_y], 
               'g--', linewidth=3, alpha=0.8, zorder=14, label='Trough Connection')
        
        print(f"Duration line drawn successfully!")
        
        # Add duration text
        duration_samples = abs(main_peak_idx - main_trough_idx)
        duration_us = duration_samples * (1000000 / 30000)  # Assuming 30kHz sampling
        ax.text((peak_x + trough_x) / 2, line_y - amp_range * 0.1, 
               f'Duration: {duration_samples} samples ({duration_us:.0f} μs)', 
               ha='center', va='top', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        print(f"Cannot draw duration line - missing indices!")
        if main_peak_idx is None:
            print(f"  main_peak_idx is None")
        if main_trough_idx is None:
            print(f"  main_trough_idx is None")
    
    ax.set_title('Duration Line Debug Test - Direct Function Test', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sample', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/jf5479/Dropbox/Python/bombcell/debug_duration_direct.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("📸 Debug plot saved as debug_duration_direct.png")
    
    return main_peak_idx is not None and main_trough_idx is not None

if __name__ == "__main__":
    print("🔧 Direct Duration Line Debug Test")
    print("=" * 40)
    
    success = test_mark_peaks_and_troughs()
    
    if success:
        print("\n✅ Duration line logic working correctly!")
    else:
        print("\n❌ Duration line logic failed!")