"""
Sliding Refractory Period Violations for BombCell.

Implements the sliding refractory period algorithm from:
  Roth et al. / SteinmetzLab (https://github.com/SteinmetzLab/slidingRefractory)
  Also implemented in SpikeInterface.

This method is robust to variation in a neuron's refractory period duration
without requiring prior knowledge. It uses Poisson statistics to control
the false acceptance rate.

The "sliding" aspect tests multiple refractory period values simultaneously
and returns the minimum contamination estimate with >=90% confidence.
"""

import numpy as np
from scipy import stats


def fraction_RP_violations_sliding(
    these_spike_times,
    these_amplitudes,
    time_chunks,
    param,
    return_per_bin=False,
):
    """
    Compute sliding refractory period violations.

    Uses the same tauR window as Hill/Llobet (tauR_valuesMin/Max/Step) and
    returns contamination estimates for each tauR value, so the downstream
    code (time_chunks_to_keep, RPV_window_index) works identically.

    Parameters
    ----------
    these_spike_times : ndarray
        Spike times in seconds for this unit.
    these_amplitudes : ndarray
        Spike amplitudes (not used, kept for interface compatibility).
    time_chunks : ndarray
        Time chunk boundaries in seconds.
    param : dict
        Parameter dict. Uses:
        - tauR_valuesMin/Max/Step: RP window to evaluate (shared with Hill/Llobet)
        - ephys_sample_rate: Sample rate in Hz (default 30000)
        - slidingRP_confThresh: Confidence threshold 0-1 (default 0.9)

    Returns
    -------
    fraction_RPVs : ndarray
        Shape (n_time_chunks, n_tauR_values). Contamination estimate per
        chunk per tauR value — same shape as Hill/Llobet output.
    num_violations : ndarray
        Shape (n_time_chunks, n_tauR_values). ISI violation counts.
    per_bin_data : dict, optional
        If return_per_bin=True, dict with detailed per-bin data.
    """
    sample_rate = param.get("ephys_sample_rate", 30000)
    conf_thresh = param.get("slidingRP_confThresh", 0.9)

    # Use the same tauR window as Hill/Llobet
    tauR_min = param["tauR_valuesMin"]
    tauR_max = param["tauR_valuesMax"]
    tauR_step = param["tauR_valuesStep"]
    tauR_window = np.arange(tauR_min, tauR_max + tauR_step, tauR_step)
    n_tauR = len(tauR_window)

    # Internal sliding RP grid: fine-grained RP values at sample resolution
    bin_size = 1.0 / sample_rate
    rp_grid = np.arange(bin_size, tauR_max + bin_size, bin_size)

    # Contamination values to test (0.5% to 34.5% in 0.5% steps)
    # Matches SteinmetzLab/SpikeInterface reference implementations
    cont_values = np.arange(0.5, 35, 0.5) / 100.0

    n_chunks = len(time_chunks) - 1
    fraction_RPVs = np.full((n_chunks, n_tauR), np.nan)
    num_violations = np.zeros((n_chunks, n_tauR))

    # Per-bin storage
    all_conf_matrices = []

    for chunk_idx in range(n_chunks):
        t_start = time_chunks[chunk_idx]
        t_stop = time_chunks[chunk_idx + 1]
        duration = t_stop - t_start

        chunk_mask = (these_spike_times >= t_start) & (these_spike_times < t_stop)
        chunk_spikes = these_spike_times[chunk_mask]
        n_spikes = len(chunk_spikes)

        if n_spikes < 2:
            all_conf_matrices.append(None)
            continue

        isis = np.diff(chunk_spikes)
        isis_sorted = np.sort(isis)
        firing_rate = n_spikes / duration if duration > 0 else 0

        # Count violations at each fine-grained RP (one-sided, from ISIs)
        acg_cumsum = np.searchsorted(isis_sorted, rp_grid, side='right')

        # Compute confidence matrix over the full fine grid
        conf_matrix = _compute_confidence_matrix(
            acg_cumsum, rp_grid, cont_values, n_spikes, firing_rate, conf_thresh
        )
        all_conf_matrices.append(conf_matrix)

        # For each tauR value, find the minimum contamination using
        # only RP grid points up to that tauR
        for t_idx, tauR in enumerate(tauR_window):
            # Mask: RP grid points <= this tauR
            rp_mask = rp_grid <= tauR + 1e-12  # small epsilon for float comparison
            if not rp_mask.any():
                continue

            num_violations[chunk_idx, t_idx] = np.sum(isis <= tauR)

            # Find minimum contamination where confidence >= threshold
            # at any RP point up to this tauR
            sub_conf = conf_matrix[:, rp_mask]  # (n_cont, n_rp_sub)
            thresh_pct = conf_thresh * 100
            passes = np.any(sub_conf >= thresh_pct, axis=1)
            if passes.any():
                first_idx = np.argmax(passes)
                fraction_RPVs[chunk_idx, t_idx] = cont_values[first_idx]
            else:
                fraction_RPVs[chunk_idx, t_idx] = 1.0

    if return_per_bin:
        per_bin_data = {
            "time_bins": time_chunks,
            "fraction_RPVs_per_bin": fraction_RPVs.copy(),
            "confidence_matrices": all_conf_matrices,
            "contamination_values_tested": cont_values,
            "rp_grid": rp_grid,
        }
        return fraction_RPVs, num_violations, per_bin_data

    return fraction_RPVs, num_violations


def _compute_confidence_matrix(
    acg_cumsum,
    rp_values,
    cont_values,
    n_spikes,
    firing_rate,
    conf_thresh,
):
    """
    Compute confidence matrix for contamination x refractory period.

    Uses Poisson statistics: given a contamination level, what's the probability
    that we'd observe this few (or fewer) violations?

    Parameters
    ----------
    acg_cumsum : ndarray
        Cumulative ACG counts at each RP value (one-sided).
    rp_values : ndarray
        Refractory period values in seconds.
    cont_values : ndarray
        Contamination proportion values to test.
    n_spikes : int
        Number of spikes.
    firing_rate : float
        Firing rate in Hz.
    conf_thresh : float
        Confidence threshold (e.g., 0.9 for 90%).

    Returns
    -------
    conf_matrix : ndarray
        Shape (n_cont, n_rp). Confidence values (0-100).
    """
    # Vectorized: build full (n_cont, n_rp) grids in one shot
    # expected_viol = firing_rate * contamination * 2 * rp_duration * n_spikes
    # Factor of 2 accounts for both sides of the ACG (observed counts are one-sided)
    # Matches SteinmetzLab/SpikeInterface reference implementations
    expected_viol = (firing_rate * n_spikes * 2) * np.outer(cont_values, rp_values)
    obs = acg_cumsum[np.newaxis, :]  # (1, n_rp) broadcast

    # Confidence = 1 - CDF(observed, expected) where expected > 0
    conf_matrix = np.zeros_like(expected_viol)
    pos = expected_viol > 0
    obs_full = np.broadcast_to(obs, expected_viol.shape)
    conf_matrix[pos] = 1.0 - stats.poisson.cdf(obs_full[pos], expected_viol[pos])
    # Where expected == 0: confidence is 1 if no violations, 0 otherwise
    zero = ~pos
    conf_matrix[zero] = np.where(obs_full[zero] == 0, 1.0, 0.0)

    # Scale to percentage
    conf_matrix *= 100

    return conf_matrix
