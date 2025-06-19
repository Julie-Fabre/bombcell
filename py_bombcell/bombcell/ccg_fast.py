"""
Fast cross-correlogram functions.
"""

import numpy as np
from bombcell.save_utils import global_bc_cacher


def acg(spike_train, cbin, cwin,
        fs=30000, normalize="hertz",
        log_window_end=None, n_log_bins=10,
        again=False, cache_results=True, cache_path=None):
    """
    Compute auto-correlogram for a single spike train.
    
    Parameters:
    -----------
    spike_train : array
        Array of spike times, in SAMPLES (unsigned integers!).
    cbin : float
        Bin size in milliseconds.
    cwin : float
        Window size in milliseconds.
    fs : int, optional
        Sampling frequency in Hz. Default is 30000 Hz.
    normalize : str, optional
        Normalization method. Options are:
        - 'counts': raw spike counts (default)
        - 'hertz': spikes per second
        - 'pearson': Pearson correlation
        - 'zscore': Z-score normalization,
                    using mean and std defined on the 80% outermost portion of each cross-correlogram.
                    To use different z-scoring parameters, apply z-scoring yourself to the output in counts or hertz.
    log_window_end : float, optional
        End of the cross-correlation window for logarithmic binning, in milliseconds.
        If provided, defines the end of the window, yielding `n_log_bins` logarithmic bins 
        between 0 and `log_window_end`.
    n_log_bins : int, optional
        Number of logarithmic bins to use if `log_window_end` is provided. Default is 10.
    again : bool, optional
        If True, recompute the cross-correlograms even if they are already cached.
        Default is False.
    cache_results : bool, optional
        If True, cache the results to disk for faster future access.
        Default is True.
    cache_path : str, optional
        Path to the cache directory. If None, uses the global '~/.bombcell' cache path.
        
    Returns:
    --------
    crosscorrelograms : ndarray
        A 1D array of shape (n_bins, ) holding the autocorrelation histogram.

    Notes:
    ------
    Function cached with cachecache (https://github.com/m-beau/cachecache).
    Author: Maxime Beau
    """
    
    correlogram = ccg([spike_train], cbin, cwin,
                       fs, normalize,
                       log_window_end, n_log_bins,
                       again, cache_results, cache_path)
    
    return correlogram.squeeze()


@global_bc_cacher
def ccg(spike_trains, cbin, cwin,
        fs=30000, normalize="hertz",
        log_window_end=None, n_log_bins=10,
        again=False, cache_results=True, cache_path=None):
    """
    Compute cross-correlograms for across spike trains.
    
    Parameters:
    -----------
    spike_trains : list of arrays
        List of spike times for each unit, in SAMPLES (unsigned integers!).
    cbin : float
        Bin size in milliseconds.
    cwin : float
        Window size in milliseconds.
    fs : int, optional
        Sampling frequency in Hz. Default is 30000 Hz.
    normalize : str, optional
        Normalization method. Options are:
        - 'counts': raw spike counts (default)
        - 'hertz': spikes per second
        - 'pearson': Pearson correlation
        - 'zscore': Z-score normalization,
                    using mean and std defined on the 80% outermost portion of each cross-correlogram.
                    To use different z-scoring parameters, apply z-scoring yourself to the output in counts or hertz.
    log_window_end : float, optional
        End of the cross-correlation window for logarithmic binning, in milliseconds.
        If provided, defines the end of the window, yielding `n_log_bins` logarithmic bins 
        between 0 and `log_window_end`.
    n_log_bins : int, optional
        Number of logarithmic bins to use if `log_window_end` is provided. Default is 10.
    again : bool, optional
        If True, recompute the cross-correlograms even if they are already cached.
        Default is False.
    cache_results : bool, optional
        If True, cache the results to disk for faster future access.
        Default is True.
    cache_path : str, optional
        Path to the cache directory. If None, uses the global '~/.bombcell' cache path.
        
    Returns:
    --------
    crosscorrelograms : ndarray
        A 3D array of shape (n_units, n_units, n_bins) holding all crosscorrelation histograms.
        Each entry (i, j, :) represents the crosscorrelogram of unit j to triggered on spikes of unit i.
        Order of units is determined by the order of spike trains is spike_trains.

    Notes:
    ------
    Function cached with cachecache (https://github.com/m-beau/cachecache).
    Author: Maxime Beau
    """

    # Check input parameters
    assert len(spike_trains) > 0, "At least one spike train (then, will only compute autocorrelogram) is required."
    assert hasattr(spike_trains[0], '__iter__'), "spike_trains should be a list of spike times arrays."
    assert all([len(st) >= 2 for st in spike_trains]), "Each spike train should contain at least two spike times."
    
    # Convert spike trains to a single array of spike times and corresponding unit ids
    # (artificial unit ids: 0, 1, 2, ... for each spike train)
    spike_times = np.concatenate(spike_trains).astype(np.uint64)
    spike_clusters = np.concatenate([np.zeros(len(st)) + i for i, st in enumerate(spike_trains)]).astype(np.int32)
    argsort = np.argsort(spike_times)  # Sort spike times and respectively clusters by spike time
    spike_times = spike_times[argsort]
    spike_clusters = spike_clusters[argsort]
    
    # Compute cross-correlograms using the crosscorrelate function
    crosscorrelograms =  crosscorrelate(spike_times,
                                        spike_clusters,
                                        cwin,
                                        cbin,
                                        fs,
                                        True,
                                        log_window_end,
                                        n_log_bins)
    
    assert normalize in ['counts', 'hertz', 'pearson', 'zscore'], \
        "WARNING ccg() 'normalize' argument should be either 'counts', 'hertz', 'pearson', or 'zscore'."
    
    # Apply normalization
    n_units = len(spike_trains)
    if normalize in ['hertz', 'pearson', 'zscore']:
        crosscorrelograms = crosscorrelograms.astype(np.float64)
        for i1 in range(n_units):
            n_spikes_1 = len(spike_trains[i1])
            for i2 in range(n_units):
                n_spikes_2 = len(spike_trains[i2])
                c = crosscorrelograms[i1, i2, :]
                if normalize == 'hertz':
                    crosscorrelograms[i1, i2, :] = c * 1. / (n_spikes_1 * cbin * 1. / 1000)
                elif normalize == 'pearson':
                    crosscorrelograms[i1, i2, :] = c * 1. / np.sqrt(n_spikes_1 * n_spikes_2)
                elif normalize=='zscore':
                    crosscorrelograms[i1, i2, :] = zscore(c, 4. / 5)
    
    return crosscorrelograms


def crosscorrelate(spike_times,
                   spike_clusters,
                   win_size,
                   bin_size,
                   fs=30000,
                   symmetrize=True,
                   log_window_end=None,
                   n_log_bins=10):
    '''
    Computes crosscorrelation histograms between all pairs of unit spike trains.

    Parameters:
    -----------
    spike_times : (n_spikes,) array of non-negative integers
        Array of spike times for all units, in samples (non-negative integers!),
        sorted by time.
    spike_clusters : (n_spikes,) array of non-negative integers
        Array of unit indices matching each spike time in spike_times.
    win_size : float
        Full (not half) window size of crosscorrelogram, in milliseconds.
    bin_size : float
        Bin size for histogram, in milliseconds.
    fs : int, optional
        Sampling rate in Hertz. Default is 30000 Hz.
    symmetrize : bool, optional
        Whether to symmetrize the semi-correlograms (corr[i, j, half:] = corr[j, i, :half][::-1]).
        Default is True.
    log_window_end : float, optional
        End of the crosscorrelation window for logarithmic binning, in milliseconds.
        If provided, defines the end of the window, yielding `n_log_bins` logarithmic bins 
        between 0 and `log_window_end`.
    n_log_bins : int, optional
        Number of logarithmic bins to use if `log_window_end` is provided. Default is 10.

    Returns:
    --------
    correlograms : numpy.ndarray
        A 3D array of shape (n_units, n_units, n_bins) holding all crosscorrelation histograms.
        Each entry (i, j, :) represents the crosscorrelogram of unit j to triggered on spikes of unit i.
        Order of units is determined by the unique values in `spike_clusters` (sorted numerically)

    Notes:
    ------
    - When `log_window_end` is provided, logarithmic binning is used instead of linear binning.
    - The correlograms are symmetrized to ensure that the crosscorrelations are consistent 
      irrespective of the order of input neurons.

    Authors: Cyrille Rossant, Maxime Beau
    '''

    # Parameter check
    units = np.unique(spike_clusters)
    n_units = len(units)

    assert fs > 0, "Sampling frequency must be positive."
    assert win_size > 0, "Window size must be positive."
    assert bin_size > 0, "Bin size must be positive."

    assert spike_times.dtype in _INT_DTYPES
    assert spike_times.ndim == 1, "spike_times must be a 1D array."
    assert spike_clusters.dtype in _INT_DTYPES
    assert spike_clusters.ndim == 1, "spike_clusters must be a 1D array."
    assert len(spike_times) == len(spike_clusters), \
        "spike_times and spike_clusters must have the same length."

    # Convert win_size and bin_size to samples
    if log_window_end is None:
        bin_size = np.clip(bin_size, 1000 * 1. / fs, 1e8)  # in milliseconds
        win_size = np.clip(win_size, 1e-2, 1e8)  # in milliseconds
        winsize_bins = 2 * int(.5 * win_size * 1. / bin_size) + 1  # Both in millisecond
        assert winsize_bins >= 1
        assert winsize_bins % 2 == 1

        samples_per_bin = int(np.ceil(fs * bin_size * 1. / 1000))  # in samples
        assert samples_per_bin >= 1  # Cannot be smaller than a sample time

        correlograms = np.zeros((n_units, n_units, winsize_bins // 2 + 1), dtype=np.int32)

    else:
        log_bins = get_log_bins_samples(log_window_end, n_log_bins, fs)
        assert np.all(log_bins>=1), "log bins can only be superior to 1 (positive half of CCG window)"
        correlograms = np.zeros((n_units, n_units, len(log_bins)), dtype=np.int32)

    # Iterate over spikes
    # starting from the smallest time differences (the neighbouring spike, shift=1)
    # working the way up until no neighbouring spike
    # is close enough to to be included in the CCG window
    # (stops when mask is only False because of mask[:-shift][spike_diff_b > (winsize_bins // 2)] = False)
    shift = 1
    mask = np.ones_like(spike_times, dtype=bool)
    while mask[:-shift].any():

        # Compute delta_Ts between each spike and the closest spike in the past
        # (so delta_Ts are always positive integers).
        # No need to do the same looking in the future, as these would be the same delta_Ts, but negative
        spike_times = as_array(spike_times) # TODO: ensure if really necessary in loop
        spike_diff = spike_times[shift:] - spike_times[:-shift]
        # convert delta from samples to number of CCG bins
        # and naturally 'bin' by using integer division (floats rounded down to int)
        # if bins are linear, simple division.
        # if bins are log or anything else, no other option but to compare to bin edges
        # (conveniently, can be computed analytically if bins are linear)
        if log_window_end is None:
            spike_diff_b = spike_diff // samples_per_bin
        else:
            bin_position = spike_diff < log_bins[:, None]
            outside_of_win = ~np.any(bin_position, axis=0)
            spike_diff_b = np.argmax(bin_position, axis=0)-1
            # on log scale 0s (same time) do not have a bin.
            # we artificially put them in the smallest bin.
            spike_diff_b[spike_diff_b==-1] = 0 

        # Mask out spikes whose the closest spike in the past
        # is further than the correlogram half window.
        # This works because shift starts from the lowest possible delta_t (1 sample)
        # and because spikes are sorted
        # so spikes with no close spike for shift = 1 will never have a close spike for future shifts
        if log_window_end is None:
            mask[:-shift][spike_diff_b > (winsize_bins // 2)] = False
        else:
            # (because using argmax, cannot be larger than window edge)
            mask[:-shift][outside_of_win] = False
        m = mask[:-shift].copy()

        # Extract delta_Ts within half CCG window
        delta_t_bins_filtered = spike_diff_b[m]

        # Find units indices matching the delta_ts
        spike_clusters_i = index_of(spike_clusters, units)
        end_units_i=spike_clusters_i[:-shift][m]
        start_units_i=spike_clusters_i[+shift:][m]
        indices = np.ravel_multi_index((end_units_i, start_units_i, delta_t_bins_filtered), correlograms.shape)
        indices = as_array(indices)

        # Count histogram of delta_Ts for this set of neighbours (shift), binwise
        # bbins becomes a 3D array, Nunits x Nunits x Nbins
        bbins = np.bincount(indices)  # would turn [0,5,2,2,3]ms into [1,0,2,1,0,1]

        # Increment the matching spikes in the correlograms array.
        arr = as_array(correlograms.ravel())  # Alias -> modif of arr will apply to correlograms
        arr[:len(bbins)] += bbins  # increments the NunitsxNunits histograms at the same time

        shift += 1

    # Remove ACG values at 0 (perfect correlation, necessarily)
    correlograms[np.arange(n_units),
                 np.arange(n_units),
                 0] = 0

    if symmetrize==True:
        nu, _, n_bins = correlograms.shape
        assert nu == _
        correlograms[..., 0] = np.maximum(correlograms[..., 0],
                                          correlograms[..., 0].T)
        sym = correlograms[..., 1:][..., ::-1]
        sym = np.transpose(sym, (1, 0, 2))
        correlograms = np.dstack((sym, correlograms))

    return correlograms


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
    - Pure Python implementation 
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
    
    # Convert times to integer units for computation
    times_int = np.round(times / fs).astype(np.float64)
    bin_size_int = round(bin_size / fs)
    
    # Use Python implementation
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

#%% utilities

_ACCEPTED_ARRAY_DTYPES = (np.float32, np.float64,
                          np.int8, np.int16, np.uint8, np.uint16,
                          np.int32, np.int64, np.uint32, np.uint64,
                          bool)
_INT_DTYPES = (np.int8, np.int16, np.int32, np.int64,
               np.uint8, np.uint16, np.uint32, np.uint64)

def as_array(arr, dtype=None):
    """
    Convert an object to a numerical NumPy array.
    Avoid a copy if possible, unlike stock np.asarray.
    """
    if arr is None:
        return None
    if isinstance(arr, np.ndarray) and dtype is None:
        return arr
    if isinstance(arr, (int, float)):
        arr = [arr]
    out = np.asarray(arr)
    if dtype is not None:
        if out.dtype != dtype:
            out = out.astype(dtype)
    if out.dtype not in _ACCEPTED_ARRAY_DTYPES:
        raise ValueError("'arr' seems to have an invalid dtype: "
                         "{0:s}".format(str(out.dtype)))
    return out

def index_of(arr, lookup):
    """
    Replace scalars in an array by their indices in a lookup table.
    Implicitely assume that:
    * All elements of arr and lookup are non-negative integers.
    * All elements or arr belong to lookup.
    This is not checked for performance reasons.
    """
    # Equivalent of np.digitize(arr, lookup) - 1, but much faster.
    lookup = np.asarray(lookup, dtype=np.int32)
    m = (lookup.max() if len(lookup) else 0) + 1
    tmp = np.zeros(m + 1, dtype=np.int64)
    # Ensure that -1 values are kept.
    tmp[-1] = -1
    if len(lookup):
        tmp[lookup] = np.arange(len(lookup))
    return tmp[arr]

def get_log_bins_samples(log_window_end, n_log_bins, fs):
    "log_window_end in ms, fs is sampling rate - output in samples."
    log_window_end = log_window_end * fs/1000 # convert to samples
    assert isinstance(n_log_bins, (int)), "n_log_bins must be an integer!"
    log_bins = np.logspace(np.log10(1),np.log10(log_window_end), n_log_bins+1)
    return log_bins

def zscore(arr, frac=None, mn_ext=None, sd_ext=None):
    '''
    Returns z-scored (centered, reduced) array using outer edges of array to compute mean and std.
    Arguments:
        - arr: 1D np array
        - frac: ]0-1] float, outer fraction of array used to compute mean and standard deviation
        - mn_ext: optional, provide mean computed outside of function
        - sd_ext: optional, provide standard deviation computed outside of function
    '''
    if frac is not None:
        assert 0 < frac <= 1, 'Z-score fraction should be between 0 and 1!'
        left = int(len(arr) * frac / 2)
        right = int(len(arr) * (1 - frac / 2))
        baseline_arr = np.append(arr[:left], arr[right:])
    else:
        baseline_arr = arr

    mn = np.mean(baseline_arr) if mn_ext is None else mn_ext
    sd = np.std(baseline_arr) if sd_ext is None else sd_ext
    if sd == 0: sd = 1

    return (arr - mn) * 1. / sd

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