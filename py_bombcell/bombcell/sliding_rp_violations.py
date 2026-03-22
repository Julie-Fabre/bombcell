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
    Compute sliding refractory period violations - drop-in replacement for
    fraction_RP_violations when using the sliding method.

    This uses the same interface as the Hill/Llobet method for easy switching.

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
        - ephys_sample_rate: Sample rate in Hz (default 30000)
        - tauC: Censored period in seconds (default 0.0001)
        - slidingRP_minRP: Min RP to consider in seconds (default 0.0005)
        - slidingRP_maxRP: Max RP to consider in seconds (default 0.01)
        - slidingRP_binSize: ACG bin size in seconds (default 1/30000)
        - slidingRP_confThresh: Confidence threshold 0-1 (default 0.9)

    Returns
    -------
    fraction_RPVs : ndarray
        Shape (n_time_chunks, 1). Minimum contamination estimate per chunk.
        Note: returns shape (n_chunks, 1) to match Hill/Llobet interface.
    num_violations : ndarray
        Shape (n_time_chunks, 1). Total ISI violations per chunk.
    per_bin_data : dict, optional
        If return_per_bin=True, dict with detailed per-bin data.
    """
    sample_rate = param.get("ephys_sample_rate", 30000)
    tau_c = param.get("tauC", 0.0001)
    rp_min = param.get("slidingRP_minRP", 0.0005)  # 0.5ms
    rp_max = param.get("slidingRP_maxRP", 0.01)    # 10ms
    bin_size = param.get("slidingRP_binSize", 1.0 / sample_rate)
    conf_thresh = param.get("slidingRP_confThresh", 0.9)

    # Contamination values to test (0.5% to 35% in 0.5% steps)
    cont_values = np.arange(0.5, 35.5, 0.5) / 100.0

    n_chunks = len(time_chunks) - 1
    # Return shape matches Hill/Llobet: (n_chunks, n_tauR_values)
    # For sliding RP, we only have 1 "tauR" value (the optimal one found)
    fraction_RPVs = np.zeros((n_chunks, 1))
    num_violations = np.zeros((n_chunks, 1))

    # Store extra info for per-bin data
    rp_at_min_cont = np.full(n_chunks, np.nan)
    all_conf_matrices = []

    for chunk_idx in range(n_chunks):
        t_start = time_chunks[chunk_idx]
        t_stop = time_chunks[chunk_idx + 1]
        duration = t_stop - t_start

        # Get spikes in this chunk
        chunk_mask = (these_spike_times >= t_start) & (these_spike_times < t_stop)
        chunk_spikes = these_spike_times[chunk_mask]
        n_spikes = len(chunk_spikes)

        if n_spikes < 2:
            fraction_RPVs[chunk_idx, 0] = np.nan
            all_conf_matrices.append(None)
            continue

        # Compute ISIs and count violations at each RP threshold
        isis = np.diff(chunk_spikes)

        # Create RP values from bin_size to rp_max
        rp_values = np.arange(bin_size, rp_max + bin_size, bin_size)

        # Filter to only RPs >= rp_min
        rp_mask = rp_values >= rp_min
        rp_values_filtered = rp_values[rp_mask]

        if len(rp_values_filtered) == 0:
            fraction_RPVs[chunk_idx, 0] = np.nan
            all_conf_matrices.append(None)
            continue

        # Count cumulative violations at each RP (×2 for symmetric ACG)
        acg_cumsum = np.array([np.sum(isis <= rp) * 2 for rp in rp_values])
        acg_cumsum_filtered = acg_cumsum[rp_mask]

        # Store total violations for output
        num_violations[chunk_idx, 0] = np.sum(isis <= rp_max)

        # Compute confidence matrix and find minimum contamination
        firing_rate = n_spikes / duration if duration > 0 else 0

        conf_matrix, min_cont, rp_at_min = _compute_confidence_matrix(
            acg_cumsum_filtered,
            rp_values_filtered,
            cont_values,
            n_spikes,
            firing_rate,
            conf_thresh,
            tau_c,
        )

        fraction_RPVs[chunk_idx, 0] = min_cont if not np.isnan(min_cont) else 1.0
        rp_at_min_cont[chunk_idx] = rp_at_min
        all_conf_matrices.append(conf_matrix)

    if return_per_bin:
        per_bin_data = {
            "time_bins": time_chunks,
            "fraction_RPVs_per_bin": fraction_RPVs.copy(),
            "rp_at_min_contamination": rp_at_min_cont,
            "confidence_matrices": all_conf_matrices,
            "contamination_values_tested": cont_values,
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
    tau_c,
):
    """
    Compute confidence matrix for contamination × refractory period.

    Uses Poisson statistics: given a contamination level, what's the probability
    that we'd observe this few (or fewer) violations?

    Parameters
    ----------
    acg_cumsum : ndarray
        Cumulative ACG counts at each RP value.
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
    tau_c : float
        Censored period in seconds.

    Returns
    -------
    conf_matrix : ndarray
        Shape (n_cont, n_rp). Confidence values (0-100).
    min_contamination : float
        Minimum contamination with confidence >= conf_thresh.
    rp_at_min : float
        RP value where minimum contamination was found.
    """
    n_cont = len(cont_values)
    n_rp = len(rp_values)
    conf_matrix = np.zeros((n_cont, n_rp))

    for i, cont in enumerate(cont_values):
        for j, rp in enumerate(rp_values):
            # Effective refractory period (excluding censored period)
            effective_rp = max(0, rp - tau_c)

            # Expected violations under Poisson model:
            # contamination_rate × 2×RP_window × n_spikes
            cont_rate = firing_rate * cont
            expected_viol = cont_rate * 2 * effective_rp * n_spikes

            if expected_viol > 0:
                # Confidence = P(observed >= expected | Poisson(expected))
                # = 1 - P(observed < expected) = 1 - CDF(observed - 1)
                # Higher value = more confident unit is cleaner than this cont level
                obs = acg_cumsum[j]
                conf_matrix[i, j] = 1.0 - stats.poisson.cdf(obs, expected_viol)
            else:
                conf_matrix[i, j] = 1.0 if acg_cumsum[j] == 0 else 0.0

    # Scale to percentage
    conf_matrix *= 100

    # Find minimum contamination where confidence >= threshold at any RP
    min_contamination = np.nan
    rp_at_min = np.nan

    for i, cont in enumerate(cont_values):
        if np.any(conf_matrix[i, :] >= conf_thresh * 100):
            min_contamination = cont
            rp_idx = np.argmax(conf_matrix[i, :])
            rp_at_min = rp_values[rp_idx]
            break

    return conf_matrix, min_contamination, rp_at_min


def compute_sliding_rp_all_units(
    spike_times_seconds,
    spike_clusters,
    unique_templates,
    quality_metrics,
    param,
):
    """
    Compute sliding RP for all units using their selected time windows.

    Called after time_chunks_to_keep has determined good time periods.

    Parameters
    ----------
    spike_times_seconds : ndarray
        All spike times in seconds.
    spike_clusters : ndarray
        Cluster ID for each spike.
    unique_templates : ndarray
        Array of unique cluster IDs.
    quality_metrics : dict
        Must contain useTheseTimesStart/Stop.
    param : dict
        Parameter dictionary.

    Returns
    -------
    quality_metrics : dict
        Updated with 'slidingRP_contamination' per unit.
    """
    n_units = len(unique_templates)
    sliding_rp_cont = np.full(n_units, np.nan)

    t_starts = quality_metrics.get("useTheseTimesStart", np.full(n_units, np.nan))
    t_stops = quality_metrics.get("useTheseTimesStop", np.full(n_units, np.nan))

    for unit_idx in range(n_units):
        t_start = t_starts[unit_idx]
        t_stop = t_stops[unit_idx]

        if np.isnan(t_start) or np.isnan(t_stop):
            continue

        this_unit = unique_templates[unit_idx]
        these_spikes = spike_times_seconds[spike_clusters == this_unit]
        mask = (these_spikes >= t_start) & (these_spikes < t_stop)
        these_spikes = these_spikes[mask]

        if len(these_spikes) < 2:
            continue

        time_chunks = np.array([t_start, t_stop])
        # Dummy amplitudes (not used by sliding method)
        dummy_amps = np.ones(len(these_spikes))

        fraction_RPVs, _ = fraction_RP_violations_sliding(
            these_spikes, dummy_amps, time_chunks, param
        )

        sliding_rp_cont[unit_idx] = fraction_RPVs[0, 0]

    quality_metrics["slidingRP_contamination"] = sliding_rp_cont

    if param.get("verbose", False):
        valid = sliding_rp_cont[~np.isnan(sliding_rp_cont)]
        if len(valid) > 0:
            print(f"  Sliding RP contamination — mean: {np.mean(valid):.4f}, "
                  f"median: {np.median(valid):.4f} (n={len(valid)} units)")

    return quality_metrics
