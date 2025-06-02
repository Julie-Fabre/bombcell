"""
Fast cross-correlogram computation using C extension
Python version of MATLAB CCGBz function
"""

import numpy as np
try:
    # Try relative import first
    from . import ccg_heart
    CCG_HEART_AVAILABLE = True
except ImportError:
    try:
        # Try direct import
        import ccg_heart
        CCG_HEART_AVAILABLE = True
    except ImportError:
        CCG_HEART_AVAILABLE = False
        print("Warning: CCG C extension not available. Use 'python ccg_setup.py build_ext --inplace' to build it.")

def ccg_bz(times, groups=None, bin_size=0.001, duration=2.0, fs=1/20000, norm='counts'):
    """
    Compute multiple cross- and auto-correlograms
    
    Python version of MATLAB CCGBz function with C backend for speed
    
    Parameters:
    -----------
    times : array-like or list of arrays
        Spike times in seconds. Can be:
        - 1D array of all spike times with corresponding groups
        - List of arrays, one per unit (groups will be auto-generated)
    groups : array-like or None
        Unit IDs for each spike in times (1-indexed integers)
        If None and times is a list, groups will be auto-generated
    bin_size : float, default=0.001
        Bin size in seconds
    duration : float, default=2.0
        Duration of each cross-correlogram in seconds
    fs : float, default=1/20000
        Sampling frequency for time discretization
    norm : str, default='counts'
        Normalization: 'counts' or 'rate'
        'counts': raw spike counts
        'rate': spikes per second
        
    Returns:
    --------
    ccg : ndarray
        3D array [n_bins, n_groups, n_groups] where ccg[t,i,j] is the
        number (or rate) of spikes from group j at time lag t relative
        to reference spikes from group i
    t : ndarray
        Time lag vector in seconds
        
    Notes:
    ------
    - Requires ccg_heart C extension for speed (falls back to slow Python if not available)
    - Groups must be positive integers (no zeros allowed)
    - Spikes will be automatically sorted by time
    """
    
    # Handle cell array input (list of spike time arrays)
    if isinstance(times, (list, tuple)) and groups is None:
        num_cells = len(times)
        new_groups = []
        for i, cell_times in enumerate(times):
            new_groups.extend([i + 1] * len(cell_times))  # 1-indexed
        times = np.concatenate(times)
        groups = np.array(new_groups, dtype=np.uint32)
    else:
        times = np.asarray(times, dtype=np.float64)
        if groups is None:
            groups = np.ones(len(times), dtype=np.uint32)
        else:
            groups = np.asarray(groups, dtype=np.uint32)
    
    # Sort by time
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    groups = groups[sort_idx]
    
    # Validate inputs
    if len(times) != len(groups):
        raise ValueError("times and groups must have the same length")
    
    if np.any(groups == 0):
        raise ValueError("groups must be positive integers (no zeros)")
        
    if len(times) <= 1:
        n_groups = int(np.max(groups)) if len(groups) > 0 else 1
        half_bins = int(round(duration / bin_size / 2))
        n_bins = 2 * half_bins + 1
        t = np.arange(-half_bins, half_bins + 1) * bin_size
        ccg = np.zeros((n_bins, n_groups, n_groups), dtype=np.uint16)
        return ccg, t
    
    # Calculate parameters
    n_groups = int(np.max(groups))
    half_bins = int(round(duration / bin_size / 2))
    n_bins = 2 * half_bins + 1
    t = np.arange(-half_bins, half_bins + 1) * bin_size
    
    # Convert times to integer units for C computation
    times_int = np.round(times / fs).astype(np.float64)
    bin_size_int = round(bin_size / fs)
    
    if CCG_HEART_AVAILABLE:
        # Use fast C implementation
        try:
            result = ccg_heart.ccg_heart(times_int, groups, bin_size_int, half_bins, False)
            counts_flat = result[0] if isinstance(result, tuple) else result
            counts = counts_flat.astype(np.double).reshape((n_bins, n_groups, n_groups))
        except Exception as e:
            print(f"Warning: C extension failed ({e}), falling back to Python implementation")
            counts = _ccg_python(times_int, groups, bin_size_int, half_bins, n_groups, n_bins)
    else:
        # Fall back to Python implementation
        counts = _ccg_python(times_int, groups, bin_size_int, half_bins, n_groups, n_bins)
    
    # Handle normalization
    if norm == 'rate':
        for g in range(1, n_groups + 1):
            num_ref_spikes = np.sum(groups == g)
            if num_ref_spikes > 0:
                counts[:, g-1, :] = counts[:, g-1, :] / num_ref_spikes / bin_size
    
    # Flip to match MATLAB output (most negative lags first)
    ccg = np.flipud(counts)
    
    return ccg, t

def _ccg_python(times, groups, bin_size, half_bins, n_groups, n_bins):
    """
    Python fallback implementation of cross-correlogram computation
    Much slower than C version but provides same results
    """
    n_spikes = len(times)
    furthest_edge = bin_size * (half_bins + 0.5)
    
    # Initialize count array
    counts = np.zeros((n_bins, n_groups, n_groups), dtype=np.double)
    
    for center_spike in range(n_spikes):
        mark1 = groups[center_spike]
        time1 = times[center_spike]
        
        # Go backward
        for second_spike in range(center_spike - 1, -1, -1):
            time2 = times[second_spike]
            
            if abs(time1 - time2) > furthest_edge:
                break
                
            bin_idx = half_bins + int(np.floor(0.5 + (time2 - time1) / bin_size))
            if 0 <= bin_idx < n_bins:
                mark2 = groups[second_spike]
                counts[bin_idx, mark1-1, mark2-1] += 1
        
        # Go forward
        for second_spike in range(center_spike + 1, n_spikes):
            time2 = times[second_spike]
            
            if abs(time1 - time2) >= furthest_edge:
                break
                
            bin_idx = half_bins + int(np.floor(0.5 + (time2 - time1) / bin_size))
            if 0 <= bin_idx < n_bins:
                mark2 = groups[second_spike]
                counts[bin_idx, mark1-1, mark2-1] += 1
    
    return counts

def build_ccg_extension():
    """
    Build the CCG C extension if not already available
    
    Call this function in a notebook cell to build the fast C extension:
    
    ```python
    import bombcell as bc
    bc.build_ccg_extension()
    ```
    
    Then restart your kernel to use the fast implementation.
    """
    if CCG_HEART_AVAILABLE:
        print("✅ CCG C extension already available and working!")
        return True
        
    try:
        import subprocess
        import os
        import sys
        
        # Get the package directory
        package_dir = os.path.dirname(__file__)
        
        # Try building with the standalone setup first
        setup_file = os.path.join(package_dir, 'ccg_setup.py')
        if os.path.exists(setup_file):
            print("🔨 Building CCG C extension...")
            result = subprocess.run([
                sys.executable, setup_file, 'build_ext', '--inplace'
            ], cwd=package_dir, capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ CCG C extension built successfully!")
                print("🔄 Please restart your Python kernel to use the fast C implementation")
                print("   In Jupyter: Kernel -> Restart")
                return True
            else:
                print(f"❌ Build failed: {result.stderr}")
        
        # Fallback: try building with main setup.py
        main_setup = os.path.join(package_dir, '..', 'setup.py')
        if os.path.exists(main_setup):
            print("🔨 Trying alternative build method...")
            result = subprocess.run([
                sys.executable, main_setup, 'build_ext', '--inplace'
            ], cwd=os.path.dirname(main_setup), capture_output=True, text=True)
            
            if result.returncode == 0:
                print("✅ CCG C extension built successfully!")
                print("🔄 Please restart your Python kernel to use the fast C implementation")
                return True
                
        print("❌ Failed to build CCG extension")
        print("ℹ️  The CCG function will use a slower Python fallback")
        return False
            
    except Exception as e:
        print(f"❌ Error building CCG extension: {e}")
        print("ℹ️  The CCG function will use a slower Python fallback")
        return False

# Test function
def test_ccg():
    """
    Test the CCG implementation with simple data
    """
    # Generate test data
    np.random.seed(42)
    
    # Two units with different firing patterns
    times1 = np.sort(np.random.exponential(0.1, 100))
    times2 = np.sort(np.random.exponential(0.15, 80)) + 0.002  # slight offset
    
    times = np.concatenate([times1, times2])
    groups = np.concatenate([np.ones(len(times1)), np.full(len(times2), 2)]).astype(np.uint32)
    
    # Compute CCG
    ccg, t = ccg_bz(times, groups, bin_size=0.001, duration=0.1)
    
    print(f"CCG shape: {ccg.shape}")
    print(f"Time lags: {len(t)} bins from {t[0]:.3f} to {t[-1]:.3f} s")
    print(f"Peak auto-correlation unit 1: {np.max(ccg[:, 0, 0])}")
    print(f"Peak auto-correlation unit 2: {np.max(ccg[:, 1, 1])}")
    print(f"Max cross-correlation: {np.max(ccg[:, 0, 1])}")
    
    return ccg, t

if __name__ == "__main__":
    # Test the implementation
    test_ccg()