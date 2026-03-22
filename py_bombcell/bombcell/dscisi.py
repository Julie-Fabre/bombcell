"""
DCISIv (Decoding Cross-contamination using ISI violations) for BombCell.

Implements the inhomogeneous model from:
  Bhagat, Bhatt & Bhalla (2024). "Decoding cross-contamination in spike-sorted
  neurophysiology data using inter-spike-interval violations."
  eNeuro 11(8), ENEURO.0554-23.2024.

This predicts False Discovery Rate (FDR) — the proportion of a unit's spikes
that are cross-contaminated (from other neurons) — using ISI violation rates
and time-varying firing rates across the population.

To remove: delete this file and remove the call in helper_functions.py.
"""

import warnings

import numpy as np
from random import sample


def compute_dscisi_fdr(spike_times_seconds, spike_clusters, unique_templates,
                       quality_metrics, param):
    """
    Compute per-unit False Discovery Rate using the DCISIv inhomogeneous model.

    Bins each unit's spike train into time bins to capture firing rate
    variation over time, then uses cross-neuron firing pattern comparisons
    to estimate FDR for N=1 and N=inf contaminant neurons. Falls back to the
    homogeneous model if fewer than 2 valid units are available.

    Parameters
    ----------
    spike_times_seconds : ndarray
        All spike times in seconds.
    spike_clusters : ndarray
        Cluster ID for each spike.
    unique_templates : ndarray
        Array of unique cluster IDs.
    quality_metrics : dict
        The quality metrics dict (must already contain useTheseTimesStart/Stop).
    param : dict
        Parameter dict. Uses:
        - tauR_valuesMin (s): refractory period for ISI violation counting
        - tauC (s): censor period
        - dcisi_bin_size (s): time bin for firing rate vectors (default 10)

    Returns
    -------
    quality_metrics : dict
        Updated with 'dcisi_fdr' (mean FDR across N=1 and N=inf per unit).

    Notes
    -----
    Per the paper's recommendation, per-unit FDR values have high stochasticity
    and should NOT be used as inclusion/exclusion criteria. Population-level
    summary statistics (mean/median across units) are more reliable.
    """
    tau = param["tauR_valuesMin"]       # refractory period in seconds
    tau_c = param["tauC"]               # censor period in seconds
    tau_e = tau - tau_c                  # effective refractory period
    bin_size = param.get("dcisi_bin_size", 10.0)  # seconds

    N_values = (1, float("inf"))
    max_compare = 10  # groups to sample for finite N (matches original DCISIv)

    n_units = len(unique_templates)
    fdr_values = np.full(n_units, np.nan)

    # --- Common time base for all units ---
    t_starts = quality_metrics["useTheseTimesStart"]
    t_stops = quality_metrics["useTheseTimesStop"]
    valid_times = ~np.isnan(t_starts) & ~np.isnan(t_stops)
    if not valid_times.any():
        quality_metrics["dcisi_fdr"] = fdr_values
        return quality_metrics

    t_global_start = np.nanmin(t_starts)
    t_global_stop = np.nanmax(t_stops)
    time_bins = np.arange(t_global_start, t_global_stop + bin_size, bin_size)
    n_time_bins = len(time_bins) - 1
    bin_durations = np.diff(time_bins)

    # --- Build firing rate matrix and ISI violation rates ---
    f_t_matrix = np.zeros((n_units, n_time_bins))
    isi_v = np.full(n_units, np.nan)
    valid_units = np.zeros(n_units, dtype=bool)

    for unit_idx in range(n_units):
        if not valid_times[unit_idx]:
            continue
        this_unit = unique_templates[unit_idx]
        t_start = t_starts[unit_idx]
        t_stop = t_stops[unit_idx]

        these_spike_times = spike_times_seconds[spike_clusters == this_unit]
        mask = (these_spike_times >= t_start) & (these_spike_times < t_stop)
        spks = these_spike_times[mask]

        if spks.size < 50:
            continue

        # ISI violation rate
        isi_v[unit_idx] = np.sum(np.diff(spks) < tau) / len(spks)

        # Firing rate vector (Hz per time bin)
        counts, _ = np.histogram(spks, bins=time_bins)
        f_t_matrix[unit_idx, :] = counts / bin_durations

        if np.linalg.norm(f_t_matrix[unit_idx, :]) > 0:
            valid_units[unit_idx] = True

    valid_idx = np.where(valid_units)[0]
    n_valid = len(valid_idx)

    if n_valid < 2:
        # Not enough neurons for inhomogeneous model — fall back to homogeneous
        for unit_idx in valid_idx:
            f_t_avg = np.mean(f_t_matrix[unit_idx, :])
            if f_t_avg <= 0:
                continue
            fdrs_for_N = [_fdr_homo(f_t_avg, isi_v[unit_idx], tau_e, N)
                          for N in N_values]
            fdr_values[unit_idx] = np.mean(fdrs_for_N)
        quality_metrics["dcisi_fdr"] = fdr_values
        _print_summary(fdr_values, param)
        return quality_metrics

    # --- Precompute unit vectors for all valid neurons ---
    f_t_unit = np.zeros_like(f_t_matrix)
    for idx in valid_idx:
        norm = np.linalg.norm(f_t_matrix[idx, :])
        if norm > 0:
            f_t_unit[idx, :] = f_t_matrix[idx, :] / norm

    # --- Inhomogeneous FDR for each valid neuron ---
    # Map valid_idx to a contiguous 0..n_valid-1 index for the "other" array
    for unit_idx in valid_idx:
        other_mask = valid_idx != unit_idx
        other_f_t_unit = f_t_unit[valid_idx[other_mask], :]
        n_other = other_mask.sum()

        fdrs_for_N = []
        for N in N_values:
            if N == float("inf"):
                # Contaminant pattern = sum of all other neurons' unit vectors
                f_FP_unit = np.sum(other_f_t_unit, axis=0)
                f_FP_norm = np.linalg.norm(f_FP_unit)
                if f_FP_norm == 0:
                    fdrs_for_N.append(1.0)
                    continue
                f_FP_unit = f_FP_unit / f_FP_norm

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    fdrs_for_N.append(_fdr_inhomo(
                        f_t_matrix[unit_idx], isi_v[unit_idx],
                        tau_e, N, f_FP_unit))
            else:
                # Sample groups of N contaminant neurons, average FDR
                N_int = int(N)
                if N_int * max_compare <= n_other:
                    comparisons = max_compare
                else:
                    comparisons = max(1, int(np.floor(n_other / N_int)))

                if n_other == N_int * comparisons:
                    FP_idx = [list(range(n_other))]
                else:
                    sampled = sample(range(n_other), N_int * comparisons)
                    FP_idx = [sampled[x:x + N_int]
                              for x in range(0, len(sampled), N_int)]

                k_fdrs = []
                for group in FP_idx:
                    f_FP_unit_k = np.sum(other_f_t_unit[group, :], axis=0)
                    f_FP_norm = np.linalg.norm(f_FP_unit_k)
                    if f_FP_norm == 0:
                        k_fdrs.append(N / (N + 1))
                        continue
                    f_FP_unit_k = f_FP_unit_k / f_FP_norm

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        k_fdrs.append(_fdr_inhomo(
                            f_t_matrix[unit_idx], isi_v[unit_idx],
                            tau_e, N, np.squeeze(f_FP_unit_k)))

                fdrs_for_N.append(np.mean(k_fdrs) if k_fdrs else N / (N + 1))

        fdr_values[unit_idx] = np.mean(fdrs_for_N)

    quality_metrics["dcisi_fdr"] = fdr_values
    _print_summary(fdr_values, param)
    return quality_metrics


def _fdr_inhomo(f_t, isi_v, tau_e, N, f_FP_unit):
    """
    DCISIv inhomogeneous model closed-form FDR for a single unit.

    Parameters
    ----------
    f_t : ndarray
        Firing rate vector (Hz per time bin) for this unit.
    isi_v : float
        ISI violation rate (violations / total spikes).
    tau_e : float
        Effective refractory period in seconds (tau - tau_c).
    N : int or float
        Number of assumed contaminant neurons (float("inf") for infinite).
    f_FP_unit : ndarray
        Unit vector of contaminant firing pattern (same length as f_t).

    Returns
    -------
    fdr : float
        Estimated false discovery rate.
    """
    f_t_mag = np.linalg.norm(f_t)
    if f_t_mag == 0:
        return 1.0 if N == float("inf") else N / (N + 1)

    f_t_unit = f_t / f_t_mag
    f_t_avg = np.average(f_t)
    if f_t_avg <= 0:
        return 1.0 if N == float("inf") else N / (N + 1)

    n = len(f_t)
    D = np.dot(f_t_unit, f_FP_unit)

    if N == float("inf"):
        N_corr1 = 1.0
        N_corr2 = 1.0
        max_fdr = 1.0
    else:
        N_corr1 = N / (N + 1)
        N_corr2 = (N + 1) / N
        max_fdr = N / (N + 1)

    sqrt1 = (f_t_mag ** 2) * (D ** 2)
    sqrt2 = (N_corr2 * f_t_avg * isi_v * n) / tau_e

    under_sqrt = sqrt1 - sqrt2
    if under_sqrt < 0:
        return max_fdr

    f_FP_mag = N_corr1 * (f_t_mag * D - np.sqrt(under_sqrt))
    f_FP_avg = np.average(f_FP_mag * f_FP_unit)
    fdr = f_FP_avg / f_t_avg

    if np.isnan(fdr):
        return max_fdr

    return fdr


def _fdr_homo(f_t, isi_v, tau_e, N):
    """
    DCISIv homogeneous model closed-form FDR (fallback for <2 neurons).
    """
    if N == float("inf"):
        N_corr1 = 1.0
        N_corr2 = 1.0
        max_fdr = 1.0
    else:
        N_corr1 = N / (N + 1)
        N_corr2 = (N + 1) / N
        max_fdr = N / (N + 1)

    under_sqrt = 1.0 - (N_corr2 * isi_v) / (f_t * tau_e)
    if under_sqrt < 0:
        return max_fdr

    fdr = N_corr1 * (1.0 - np.sqrt(under_sqrt))
    if np.isnan(fdr):
        return max_fdr

    return fdr


def _print_summary(fdr_values, param):
    """Print and store population-level FDR summary."""
    valid = fdr_values[~np.isnan(fdr_values)]
    if valid.size > 0:
        pop_mean = float(np.mean(valid))
        pop_median = float(np.median(valid))
        param["dcisi_population_mean"] = pop_mean
        param["dcisi_population_median"] = pop_median
        if param.get("verbose", False):
            print(f"\n  DCISIv population FDR (inhomogeneous) — mean: {pop_mean:.4f}, "
                  f"median: {pop_median:.4f} (n={valid.size} units)")
    else:
        param["dcisi_population_mean"] = np.nan
        param["dcisi_population_median"] = np.nan


def compute_combined_contamination(
    spike_times_seconds,
    spike_clusters,
    unique_templates,
    quality_metrics,
    param,
):
    """
    Compute combined contamination estimate using sliding RP and DCISIv.

    Both methods are computed using the same time bins (dcisi_bin_size) for
    consistency. The final estimate is a weighted combination:
    - Sliding RP (70%): More reliable for individual unit contamination
    - DCISIv (30%): Provides cross-population context

    Parameters
    ----------
    spike_times_seconds : ndarray
        All spike times in seconds.
    spike_clusters : ndarray
        Cluster ID for each spike.
    unique_templates : ndarray
        Array of unique cluster IDs.
    quality_metrics : dict
        Quality metrics dict (must contain useTheseTimesStart/Stop).
    param : dict
        Parameter dictionary.

    Returns
    -------
    quality_metrics : dict
        Updated with:
        - 'slidingRP_contamination_per_bin': Per-bin sliding RP estimates
        - 'dcisi_fdr': DCISIv FDR per unit
        - 'combined_contamination': Weighted combination of both methods
    """
    from bombcell.sliding_rp_violations import fraction_RP_violations_sliding

    n_units = len(unique_templates)
    bin_size = param.get("dcisi_bin_size", 10.0)

    # Get valid time ranges
    t_starts = quality_metrics.get("useTheseTimesStart", np.full(n_units, np.nan))
    t_stops = quality_metrics.get("useTheseTimesStop", np.full(n_units, np.nan))
    valid_times = ~np.isnan(t_starts) & ~np.isnan(t_stops)

    if not valid_times.any():
        quality_metrics["combined_contamination"] = np.full(n_units, np.nan)
        quality_metrics["slidingRP_contamination_per_bin"] = None
        return quality_metrics

    # Create shared time bins
    t_global_start = np.nanmin(t_starts)
    t_global_stop = np.nanmax(t_stops)
    time_bins = np.arange(t_global_start, t_global_stop + bin_size, bin_size)
    n_bins = len(time_bins) - 1

    # Storage for per-bin sliding RP
    sliding_rp_per_bin = np.full((n_units, n_bins), np.nan)
    sliding_rp_mean = np.full(n_units, np.nan)

    # Compute sliding RP per time bin for each unit
    for unit_idx in range(n_units):
        if not valid_times[unit_idx]:
            continue

        this_unit = unique_templates[unit_idx]
        these_spikes = spike_times_seconds[spike_clusters == this_unit]
        t_start = t_starts[unit_idx]
        t_stop = t_stops[unit_idx]
        mask = (these_spikes >= t_start) & (these_spikes < t_stop)
        these_spikes = these_spikes[mask]

        if len(these_spikes) < 2:
            continue

        # Use shared time bins
        dummy_amps = np.ones(len(these_spikes))
        fraction_RPVs, _ = fraction_RP_violations_sliding(
            these_spikes, dummy_amps, time_bins, param
        )

        sliding_rp_per_bin[unit_idx, :] = fraction_RPVs[:, 0]
        sliding_rp_mean[unit_idx] = np.nanmean(fraction_RPVs[:, 0])

    quality_metrics["slidingRP_contamination_per_bin"] = sliding_rp_per_bin

    # Compute DCISIv (uses same bin size internally)
    quality_metrics = compute_dscisi_fdr(
        spike_times_seconds, spike_clusters, unique_templates,
        quality_metrics, param
    )

    # Combine estimates
    dcisi_fdr = quality_metrics.get("dcisi_fdr", np.full(n_units, np.nan))
    combined = np.full(n_units, np.nan)

    for unit_idx in range(n_units):
        rp_est = sliding_rp_mean[unit_idx]
        dcisi_est = dcisi_fdr[unit_idx]

        if np.isnan(rp_est) and np.isnan(dcisi_est):
            continue
        elif np.isnan(rp_est):
            combined[unit_idx] = dcisi_est
        elif np.isnan(dcisi_est):
            combined[unit_idx] = rp_est
        else:
            # Weighted combination: sliding RP is more validated
            combined[unit_idx] = 0.7 * rp_est + 0.3 * dcisi_est

    quality_metrics["combined_contamination"] = combined

    if param.get("verbose", False):
        valid = combined[~np.isnan(combined)]
        if len(valid) > 0:
            print(f"  Combined contamination (sliding RP + DCISIv) — "
                  f"mean: {np.mean(valid):.4f}, median: {np.median(valid):.4f}")

    return quality_metrics
