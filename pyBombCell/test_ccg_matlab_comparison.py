"""
Test script to compare Python CCG implementation with MATLAB
"""
import numpy as np
import sys
sys.path.insert(0, 'bombcell')
from ccg_fast import ccg_bz

def test_ccg_matlab_comparison():
    """
    Test with data similar to what MATLAB would use
    """
    print("Testing Python CCG implementation...")
    
    # Test 1: Simple two-unit case
    print("\n=== Test 1: Two units with known timing ===")
    times1 = np.array([0.001, 0.003, 0.005, 0.007, 0.009])  # 500 Hz
    times2 = np.array([0.002, 0.004, 0.006, 0.008, 0.010])  # 500 Hz, offset
    
    # Test both input formats
    # Format 1: Cell array (list of arrays)
    times_cells = [times1, times2]
    ccg1, t1 = ccg_bz(times_cells, bin_size=0.001, duration=0.01)
    
    # Format 2: Concatenated with groups
    times_concat = np.concatenate([times1, times2])
    groups_concat = np.concatenate([np.ones(len(times1)), np.full(len(times2), 2)]).astype(np.uint32)
    ccg2, t2 = ccg_bz(times_concat, groups_concat, bin_size=0.001, duration=0.01)
    
    print(f"Cell format CCG shape: {ccg1.shape}")
    print(f"Concat format CCG shape: {ccg2.shape}")
    print(f"Results match: {np.allclose(ccg1, ccg2)}")
    
    # Test 2: Rate normalization
    print("\n=== Test 2: Rate normalization ===")
    ccg_counts, _ = ccg_bz(times_cells, bin_size=0.001, duration=0.01, norm='counts')
    ccg_rate, _ = ccg_bz(times_cells, bin_size=0.001, duration=0.01, norm='rate')
    
    print(f"Max count (unit 1 auto): {np.max(ccg_counts[:, 0, 0])}")
    print(f"Max rate (unit 1 auto): {np.max(ccg_rate[:, 0, 0]):.2f} spikes/s")
    
    # Test 3: Performance test
    print("\n=== Test 3: Performance with larger dataset ===")
    np.random.seed(42)
    
    # Generate realistic spike trains
    n_units = 5
    duration_sec = 60.0  # 1 minute
    times_list = []
    
    for i in range(n_units):
        # Different firing rates for each unit
        rate = 10 + i * 5  # 10-30 Hz
        n_spikes = int(rate * duration_sec)
        spike_times = np.sort(np.random.exponential(1.0/rate, n_spikes))
        # Rescale to duration
        spike_times = spike_times * duration_sec / spike_times[-1]
        times_list.append(spike_times)
        print(f"Unit {i+1}: {len(spike_times)} spikes, {len(spike_times)/duration_sec:.1f} Hz")
    
    import time
    start_time = time.time()
    ccg_perf, t_perf = ccg_bz(times_list, bin_size=0.001, duration=0.1)
    end_time = time.time()
    
    total_spikes = sum(len(times) for times in times_list)
    print(f"\nProcessed {total_spikes} spikes in {(end_time-start_time)*1000:.2f} ms")
    print(f"CCG shape: {ccg_perf.shape}")
    print(f"Performance: {total_spikes/(end_time-start_time):.0f} spikes/second")
    
    # Test 4: Edge cases
    print("\n=== Test 4: Edge cases ===")
    
    # Empty input
    try:
        ccg_empty, _ = ccg_bz([], bin_size=0.001, duration=0.01)
        print(f"Empty input handled: shape {ccg_empty.shape}")
    except Exception as e:
        print(f"Empty input error: {e}")
    
    # Single spike
    ccg_single, _ = ccg_bz([np.array([0.001])], bin_size=0.001, duration=0.01)
    print(f"Single spike handled: shape {ccg_single.shape}")
    
    print("\n=== All tests completed ===")
    return ccg1, ccg_perf

def compare_with_matlab_output():
    """
    Compare specific cases that can be verified against MATLAB
    """
    print("\n=== MATLAB Comparison Cases ===")
    
    # Case matching MATLAB documentation example
    times = np.array([0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5])
    groups = np.array([1, 1, 2, 1, 2, 1, 2], dtype=np.uint32)
    
    ccg, t = ccg_bz(times, groups, bin_size=0.01, duration=0.2)
    
    print(f"MATLAB-style input:")
    print(f"Times: {times}")
    print(f"Groups: {groups}")
    print(f"CCG shape: {ccg.shape}")
    print(f"Time bins: {len(t)}")
    print(f"Center bin (zero lag): {np.argmax(t >= 0)}")
    
    # Check autocorrelation peaks
    center_bin = len(t) // 2
    print(f"Unit 1 autocorr at zero lag: {ccg[center_bin, 0, 0]}")
    print(f"Unit 2 autocorr at zero lag: {ccg[center_bin, 1, 1]}")
    print(f"Cross-corr 1->2 at zero lag: {ccg[center_bin, 0, 1]}")
    print(f"Cross-corr 2->1 at zero lag: {ccg[center_bin, 1, 0]}")

if __name__ == "__main__":
    # Run the tests
    ccg1, ccg_perf = test_ccg_matlab_comparison()
    compare_with_matlab_output()
    
    print(f"\nPython CCG implementation ready!")
    print(f"Import with: from bombcell import ccg_bz")
    print(f"Or: import bombcell as bc; bc.ccg_bz(...)")