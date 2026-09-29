"""
BombCell quality control pipeline for SpikeInterface SortingAnalyzer objects.

This module is the SpikeInterface entry point to BombCell: it decides which quality
metrics to compute (and computes their prerequisite extensions), runs the BombCell
labeling, makes the diagnostic plots and saves the results.

The labeling itself lives in SpikeInterface, as ``spikeinterface.curation.bombcell_label_units``.
This module orchestrates around it, which is why it carries the BombCell-flavored
defaults (which metrics, which parameters) rather than SpikeInterface.

SpikeInterface is an optional dependency of bombcell, needed only for this module::

    pip install "bombcell[spikeinterface]"

For spike-sorted data on disk rather than a SortingAnalyzer, use bombcell's own
pipeline instead (see ``bombcell.run_bombcell``).

Functions
---------
get_default_qc_params
    Get default parameters for quality metrics and BombCell labeling.
run_bombcell_qc
    Compute quality metrics, run BombCell labeling, and generate plots.

See Also
--------
spikeinterface.curation.bombcell_get_default_thresholds : Default classification thresholds.
spikeinterface.curation.bombcell_label_units : Core labeling function.
"""

from __future__ import annotations
import warnings
from pathlib import Path

_SPIKEINTERFACE_INSTALL_HINT = (
    "This needs SpikeInterface, which is an optional dependency of bombcell. "
    'Install it with: pip install "bombcell[spikeinterface]"'
)


def _require_spikeinterface():
    """Import the SpikeInterface pieces this module needs, with a clear error if missing."""
    try:
        from spikeinterface.curation import (
            bombcell_get_default_thresholds,
            bombcell_label_units,
            save_bombcell_results,
        )
    except ImportError as err:
        raise ImportError(_SPIKEINTERFACE_INSTALL_HINT) from err
    return bombcell_get_default_thresholds, bombcell_label_units, save_bombcell_results


def _require_spikeinterface_full():
    """Import spikeinterface.full, with a clear error if SpikeInterface is missing."""
    try:
        import spikeinterface.full as si
    except ImportError as err:
        raise ImportError(_SPIKEINTERFACE_INSTALL_HINT) from err
    return si

# The two refractory-period-violation metrics BombCell supports. The user picks
# one by including it as a key under thresholds["mua"] — that single choice
# determines which metric is computed AND thresholded. No other knob selects it.
_RPV_METRICS = ("sliding_rp_violation", "rp_contamination")

# Length of the time chunks valid periods are built from. SpikeInterface defaults to 30 s;
# this is BombCell's own deltaTimeChunk, so the pipeline behaves like BombCell out of the box.
_DEFAULT_TIME_CHUNK_S = 360


def _valid_periods_table(sorting_analyzer):
    """Per-time-chunk contamination and missing-spike rates, as a DataFrame.

    The ``valid_unit_periods`` extension evaluates every candidate time chunk and keeps the
    ones that pass. It stores the rates for all of them, so this is BombCell's per-chunk view:
    one row per unit and chunk, with the false-positive rate (refractory-period violations),
    the false-negative rate (missing spikes), and whether the chunk was kept.
    """
    import pandas as pd

    extension = sorting_analyzer.get_extension("valid_unit_periods")
    if extension is None:
        return None

    all_periods = extension.data["all_periods"]
    fps = extension.data["fps"]
    fns = extension.data["fns"]
    sampling_frequency = sorting_analyzer.sampling_frequency
    unit_ids = sorting_analyzer.unit_ids

    # The stored valid periods are merged spans, so they cannot be matched back to individual
    # chunks. Re-apply the criterion the extension used instead.
    kept = (fps < extension.params["fp_threshold"]) & (fns < extension.params["fn_threshold"])

    return pd.DataFrame(
        {
            "unit_id": [unit_ids[i] for i in all_periods["unit_index"]],
            "segment_index": all_periods["segment_index"],
            "start_s": all_periods["start_sample_index"] / sampling_frequency,
            "end_s": all_periods["end_sample_index"] / sampling_frequency,
            "fraction_rpv": fps,
            "fraction_spikes_missing": fns,
            "kept": kept,
        }
    )


def _valid_periods_params(thresholds: dict, params: dict) -> dict:
    """Build the ``valid_unit_periods`` parameters for a BombCell run.

    The false-positive and false-negative thresholds that select valid periods are the same
    quantities BombCell thresholds on (refractory-period violations and missing spikes), so
    they are derived from ``thresholds["mua"]`` rather than left to drift apart from it.
    Anything in ``params["valid_periods_params"]`` overrides all of this.
    """
    mua = thresholds.get("mua", {})
    vp_params = {
        "period_mode": "absolute",
        "period_duration_s_absolute": _DEFAULT_TIME_CHUNK_S,
    }

    fp_threshold = mua.get(_resolve_rpv_metric(thresholds), {}).get("less", None)
    if fp_threshold is not None:
        vp_params["fp_threshold"] = fp_threshold

    fn_threshold = mua.get("amplitude_cutoff", {}).get("less", None)
    if fn_threshold is not None:
        vp_params["fn_threshold"] = fn_threshold

    vp_params.update(params.get("valid_periods_params") or {})
    return vp_params


def _resolve_rpv_metric(thresholds: dict) -> str:
    """Return the single RPV metric name the user selected in thresholds["mua"].

    Raises if the user specified both (ambiguous) or neither (no RPV check).
    """
    mua = thresholds.get("mua", {})
    picked = [m for m in _RPV_METRICS if m in mua]
    if len(picked) == 0:
        raise ValueError(
            f"thresholds['mua'] must include exactly one of {_RPV_METRICS} "
            "to select the refractory-period-violation method."
        )
    if len(picked) > 1:
        raise ValueError(
            f"thresholds['mua'] contains multiple RPV metrics ({picked}). " "Pick exactly one: remove the other entry."
        )
    return picked[0]


def _warn_if_valid_periods_mismatch(sorting_analyzer, thresholds: dict) -> None:
    """Warn if a pre-existing valid_unit_periods extension was built with fp/fn
    thresholds that disagree with the bombcell RPV / amplitude_cutoff thresholds.

    The point is to surface the mismatch (not fix it) — the user picked those
    valid-periods params on purpose, but if their bombcell thresholds have
    since drifted, the labels below won't match the periods above.
    """
    ext = sorting_analyzer.get_extension("valid_unit_periods")
    if ext is None:
        return
    vp_params = ext.params or {}
    mua = thresholds.get("mua", {})

    # Match the RPV metric that bombcell will threshold
    rpv_bc = None
    for name in ("sliding_rp_violation", "rp_contamination"):
        if name in mua:
            rpv_bc = mua[name].get("less", None)
            break
    fp_vp = vp_params.get("fp_threshold", None)
    if rpv_bc is not None and fp_vp is not None and fp_vp != rpv_bc:
        warnings.warn(
            f"valid_unit_periods was computed with fp_threshold={fp_vp}, but bombcell "
            f"RPV threshold is {rpv_bc}. The valid periods were chosen with a different "
            "contamination criterion than the one bombcell will use for labeling. "
            "Recompute valid_unit_periods if you want them to line up."
        )

    ac_bc = mua.get("amplitude_cutoff", {}).get("less", None)
    fn_vp = vp_params.get("fn_threshold", None)
    if ac_bc is not None and fn_vp is not None and fn_vp != ac_bc:
        warnings.warn(
            f"valid_unit_periods was computed with fn_threshold={fn_vp}, but bombcell "
            f"amplitude_cutoff threshold is {ac_bc}. The valid periods were chosen with "
            "a different missing-spikes criterion than bombcell will use for labeling. "
            "Recompute valid_unit_periods if you want them to line up."
        )


def get_default_preprocessing_params():
    """
    Get default parameters for preprocessing and SortingAnalyzer creation.

    Returns a dictionary that can be modified and passed to preprocess_for_bombcell().

    Returns
    -------
    dict
        Dictionary with the following keys:

        **Highpass Filtering**

        freq_min : float, default: 300.0
            Highpass filter cutoff frequency in Hz. Removes low-frequency noise
            and LFP signals. Standard value for spike detection is 300 Hz.
            Lower values (150-250 Hz) may be used if spikes have significant
            low-frequency components.

        **Bad Channel Detection**

        detect_bad_channels : bool, default: True
            Whether to automatically detect and remove bad channels before
            further processing. Recommended to leave enabled.

        bad_channel_method : str, default: "coherence+psd"
            Method for detecting bad channels:
            - "coherence+psd": Combines coherence with neighbors and power
              spectral density analysis. Best for Neuropixels. (recommended)
            - "mad": Median absolute deviation of signal amplitude.
            - "std": Standard deviation based detection.

        **Phase Shift Correction**

        apply_phase_shift : bool, default: True
            Whether to apply inter-sample phase shift correction. Essential for
            Neuropixels probes where ADCs are multiplexed and channels are sampled
            at slightly different times. Corrects for timing offsets that can
            affect spike waveform shapes. Disable for non-Neuropixels probes.

        **Common Reference**

        apply_cmr : bool, default: True
            Whether to apply common reference to remove correlated noise.
            Highly recommended for Neuropixels recordings.

        cmr_reference : str, default: "global"
            Type of common reference:
            - "global": Use all channels (recommended for Neuropixels).
            - "local": Use nearby channels only (for probes with distinct groups).
            - "single": Reference to a single channel.

        cmr_operator : str, default: "median"
            Operation for computing reference signal:
            - "median": More robust to outliers (recommended).
            - "average": Standard mean reference.

        **Motion (drift) Correction**

        apply_motion_correction : bool, default: False
            Whether to estimate and correct probe drift. Off by default, and deliberately so:
            drift correction has to happen *before* sorting to be meaningful, and the
            recommended sorter, Kilosort4, already does its own. SpikeInterface runs Kilosort4
            with ``do_correction=True``, so leaving this False is right for a Kilosort4
            workflow - turning both on corrects the same drift twice.

            Turn this on when your sorter does not correct drift itself, and note that the
            ``sorting`` you pass in must have been produced from the corrected recording:
            correcting here does not move spikes that were sorted from uncorrected traces.

        motion_preset : str, default: "dredge_fast"
            SpikeInterface motion-correction preset, e.g. "dredge", "dredge_fast",
            "nonrigid_accurate", "rigid_fast", "kilosort_like", "medicine". Only used when
            ``apply_motion_correction`` is True.

        save_motion_info : bool, default: True
            Save the estimated motion next to the analyzer, so drift can be inspected later.

        **SortingAnalyzer Settings**

        sparse : bool, default: True
            Use sparse waveform representation, storing only channels near each
            unit. Significantly reduces memory usage for high-channel-count probes.
            Recommended True for Neuropixels.

        return_in_uV : bool, default: True
            Convert waveforms to microvolts using gain/offset from probe metadata.
            Required for amplitude-based quality metrics to be meaningful.

        **Waveform Extraction**

        max_spikes_per_unit : int, default: 500
            Maximum number of spikes to extract per unit for waveform analysis.
            Higher values give better templates but use more memory/time.
            500 is typically sufficient for stable template estimation.

        ms_before : float, default: 3.0
            Milliseconds before spike peak to extract. 3.0 ms captures the
            pre-spike baseline and any pre-depolarization.

        ms_after : float, default: 3.0
            Milliseconds after spike peak to extract. 3.0 ms captures the
            repolarization and afterhyperpolarization.

        **Template Computation**

        template_operators : list, default: ["average", "median", "std"]
            Statistics to compute for templates:
            - "average": Mean waveform (standard template).
            - "median": Median waveform (robust to outliers).
            - "std": Standard deviation (waveform variability).

        include_multi_channel_metrics : bool, default: True
            Compute template metrics across multiple channels, including:
            - exp_decay: Exponential decay of amplitude across channels.
            - velocity: Propagation velocity estimate.
            Required for BombCell noise detection. Leave True.

    Examples
    --------
    >>> params = get_default_preprocessing_params()
    >>> params["freq_min"] = 250.0  # Lower cutoff for some cell types
    >>> params["max_spikes_per_unit"] = 1000  # More spikes for better templates
    >>> analyzer, rec, bad_chs = preprocess_for_bombcell(recording, sorting, "analyzer.zarr", params=params)
    """
    return {
        # Highpass filtering
        "freq_min": 300.0,
        # Bad channel detection
        "detect_bad_channels": True,
        "bad_channel_method": "coherence+psd",
        # Phase shift (Neuropixels)
        "apply_phase_shift": True,
        # Common reference
        "apply_cmr": True,
        "cmr_reference": "global",
        "cmr_operator": "median",
        # Motion (drift) correction
        "apply_motion_correction": False,
        "motion_preset": "dredge_fast",
        "save_motion_info": True,
        # SortingAnalyzer
        "sparse": True,
        "return_in_uV": True,
        # Waveforms
        "max_spikes_per_unit": 500,
        "ms_before": 3.0,
        "ms_after": 3.0,
        # Templates
        "template_operators": ["average", "median", "std"],
        "include_multi_channel_metrics": True,
    }


def preprocess_for_bombcell(
    recording,
    sorting,
    analyzer_folder: str | Path,
    params: dict | None = None,
    rerun_extensions: bool = False,
    n_jobs: int = -1,
    progress_bar: bool = True,
):
    """
    Preprocess recording and create SortingAnalyzer with extensions for BombCell.

    This function applies standard preprocessing steps (filtering, bad channel
    removal, phase shift correction, common reference) and creates a SortingAnalyzer
    with all extensions required for BombCell quality control.

    Parameters
    ----------
    recording : BaseRecording
        Raw recording to preprocess. Typically loaded with si.read_spikeglx()
        or si.read_openephys().
    sorting : BaseSorting
        Spike sorting result. Can be loaded with si.read_sorter_folder() or
        any other sorting loader.
    analyzer_folder : str or Path
        Path to save the SortingAnalyzer. Will be created in zarr format.
        If folder exists, loads existing analyzer instead of creating new one.
    params : dict or None, default: None
        Preprocessing parameters from get_default_preprocessing_params().
        If None, uses all default values.
    rerun_extensions : bool, default: False
        If True, recompute all extensions even if they already exist.
        Useful after changing parameters.
    n_jobs : int, default: -1
        Number of parallel jobs for computation. -1 uses all available CPUs.
    progress_bar : bool, default: True
        Show progress bars during computation.

    Returns
    -------
    analyzer : SortingAnalyzer
        SortingAnalyzer saved to analyzer_folder with computed extensions:
        random_spikes, waveforms, templates, noise_levels, unit_locations,
        spike_locations, template_metrics. Note: spike_amplitudes is computed
        on-demand by run_bombcell_qc() if compute_amplitude_cutoff=True.
    rec_preprocessed : BaseRecording
        Preprocessed recording (lazy chain, not saved to disk).
        Can be used for further analysis or passed to other functions.
    bad_channel_ids : list or None
        List of channel IDs that were detected as bad and removed.
        None if detect_bad_channels=False.

    Examples
    --------
    Basic usage with defaults:

    >>> analyzer, rec, bad_chs = preprocess_for_bombcell(recording, sorting, "analyzer.zarr")

    With custom parameters:

    >>> params = get_default_preprocessing_params()
    >>> params["freq_min"] = 250.0
    >>> params["detect_bad_channels"] = False  # Already cleaned
    >>> analyzer, rec, bad_chs = preprocess_for_bombcell(
    ...     recording, sorting, "analyzer.zarr", params=params
    ... )

    Rerun extensions after parameter change:

    >>> analyzer, rec, bad_chs = preprocess_for_bombcell(
    ...     recording, sorting, "analyzer.zarr", rerun_extensions=True
    ... )
    """
    si = _require_spikeinterface_full()

    if params is None:
        params = get_default_preprocessing_params()

    analyzer_folder = Path(analyzer_folder)
    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=progress_bar)

    # Preprocess
    rec = si.highpass_filter(recording, freq_min=params["freq_min"])

    bad_channel_ids = None
    if params["detect_bad_channels"]:
        bad_channel_ids, _ = si.detect_bad_channels(rec, method=params["bad_channel_method"])
        bad_channel_ids = list(bad_channel_ids)
        if len(bad_channel_ids) > 0:
            rec = rec.remove_channels(bad_channel_ids)

    if params["apply_phase_shift"]:
        # Only recordings that carry ADC timing metadata (Neuropixels, via SpikeGLX/OpenEphys)
        # can be phase-shifted. Skip rather than fail for probes that do not, since this is
        # on by default.
        if "inter_sample_shift" in rec.get_property_keys():
            rec = si.phase_shift(rec)
        else:
            warnings.warn(
                "Skipping phase shift correction: this recording has no 'inter_sample_shift' "
                "property, which only probes with multiplexed ADCs (e.g. Neuropixels read with "
                "SpikeGLX or OpenEphys) provide. Set apply_phase_shift=False to silence this."
            )

    if params["apply_cmr"]:
        rec = si.common_reference(rec, reference=params["cmr_reference"], operator=params["cmr_operator"])

    # Motion (drift) correction, last so it acts on filtered, referenced traces. Off by default:
    # it is slow, and it only makes sense before sorting. If the sorting passed in was produced
    # from an uncorrected recording, correcting here would misalign it - see the docstring.
    if params["apply_motion_correction"]:
        rec = si.correct_motion(
            rec,
            preset=params["motion_preset"],
            folder=(Path(analyzer_folder).parent / "motion") if params["save_motion_info"] else None,
            **job_kwargs,
        )

    rec_preprocessed = rec

    # Create or load analyzer
    if analyzer_folder.exists():
        analyzer = si.load_sorting_analyzer(analyzer_folder)
        if not analyzer.has_recording():
            analyzer.set_temporary_recording(rec_preprocessed)
    else:
        analyzer = si.create_sorting_analyzer(
            sorting=sorting,
            recording=rec_preprocessed,
            sparse=params["sparse"],
            format="zarr",
            folder=analyzer_folder,
            return_in_uV=params["return_in_uV"],
        )

    # Compute extensions
    def _compute(name, **kwargs):
        if analyzer.has_extension(name) and not rerun_extensions:
            return
        if analyzer.has_extension(name):
            analyzer.delete_extension(name)
        analyzer.compute(name, **kwargs)

    _compute("random_spikes", method="uniform", max_spikes_per_unit=params["max_spikes_per_unit"])
    _compute("waveforms", ms_before=params["ms_before"], ms_after=params["ms_after"], **job_kwargs)
    _compute("templates", operators=list(params["template_operators"]))
    _compute("noise_levels")
    # spike_amplitudes computed on-demand by run_bombcell_qc if compute_amplitude_cutoff=True
    _compute("unit_locations")
    _compute("spike_locations", **job_kwargs)
    _compute("template_metrics", include_multi_channel_metrics=params["include_multi_channel_metrics"])

    return analyzer, rec_preprocessed, bad_channel_ids


def get_default_qc_params():
    """
    Get default parameters for quality metrics and BombCell labeling.

    Returns a dictionary that can be modified and passed to run_bombcell_qc().

    Returns
    -------
    dict
        Dictionary with the following keys:

        **Quality Metrics Selection**

        compute_amplitude_cutoff : bool, default: False
            Whether to compute amplitude_cutoff metric (estimated percentage of
            missing spikes). Requires spike_amplitudes extension which is slow
            to compute for large recordings. When enabled, spike_amplitudes will
            be computed automatically if not already present.

        compute_distance_metrics : bool, default: False
            Whether to compute isolation_distance and l_ratio metrics.
            These require PCA computation and are slow for large datasets.
            Useful for chronic recordings where cluster stability matters.
            Not recommended for acute recordings with expected drift.

        compute_drift : bool, default: True
            Whether to compute drift metrics (drift_ptp, drift_std, drift_mad).
            Measures how much units move over the recording. Important for
            acute recordings. drift_ptp (peak-to-peak drift in um) is used
            by BombCell MUA thresholds.

        **BombCell Labeling Options**

        Note: the refractory-period-violation method (``sliding_rp_violation``
        or ``rp_contamination``) and its threshold are selected in a single
        place — the thresholds dict (``thresholds["mua"]``). Whichever RPV
        key the user puts there determines which metric is computed AND
        thresholded. Extra metric-specific parameters can be passed via
        ``params["metric_params"]`` (e.g. ``{"sliding_rp_violation":
        {"exclude_ref_period_below_ms": 0.5}}``).

        split_non_somatic : bool, default: False
            If True, split non-somatic units into "non_soma_good" and
            "non_soma_mua" based on whether they pass MUA thresholds.
            If False, all non-somatic units are labeled "non_soma".
            To skip non-somatic labeling entirely, empty or omit the
            ``"non-somatic"`` section of ``thresholds``.

        compute_valid_periods : bool, default: False
            If True, identify valid time periods per unit (where the unit
            has stable amplitude and low refractory violations) and compute
            quality metrics only on those periods. Useful for recordings
            with unstable periods. Requires amplitude_scalings extension.
            If the ``valid_unit_periods`` extension is already present on the
            analyzer, it is reused as-is (no recompute); if its ``fp_threshold``
            / ``fn_threshold`` differ from the bombcell RPV / amplitude_cutoff
            thresholds, a warning is emitted.

            This is the SpikeInterface equivalent of BombCell's time chunks: the
            recording is split into chunks and only those where the unit looks
            well isolated are kept. The chunk length defaults to BombCell's
            ``deltaTimeChunk`` (360 s) rather than the SpikeInterface default of
            30 s, and the ``fp_threshold`` / ``fn_threshold`` that decide whether
            a chunk is kept are taken from ``thresholds["mua"]`` (the RPV metric
            and ``amplitude_cutoff``), so they cannot drift apart from the
            thresholds used for labeling.

        valid_periods_params : dict or None, default: None
            Parameters forwarded to the ``valid_unit_periods`` extension, overriding
            the ones derived above. For example
            ``{"period_duration_s_absolute": 120, "minimum_valid_period_duration": 60}``.
            Only used when the extension is computed here; an extension already on
            the analyzer is reused untouched.

        **Presence Ratio Parameters**

        presence_ratio_bin_duration_s : float, default: 60
            Bin duration in seconds for computing presence ratio.
            Presence ratio = fraction of bins containing at least one spike.
            60s bins are standard; shorter bins are stricter.

        **Drift Parameters**

        drift_interval_s : float, default: 60
            Interval in seconds for computing drift. Unit positions are
            estimated in each interval and drift is the movement across intervals.

        drift_min_spikes : int, default: 100
            Minimum spikes required per interval to estimate position.
            Intervals with fewer spikes are skipped.

        **Plotting Options**

        plot_histograms : bool, default: True
            Generate histograms of all metrics with threshold lines.
            Saved as "metric_histograms.png".

        plot_waveforms : bool, default: True
            Generate waveform overlay plots grouped by label (good, mua, noise, etc.).
            Saved as "waveforms_by_label.png".

        plot_upset : bool, default: True
            Generate UpSet plots showing which metrics fail together.
            Useful for understanding why units are labeled noise/mua.
            Requires 'upsetplot' package. Saved as "upset_plot_*.png".

        waveform_ylims : tuple or None, default: (-300, 100)
            Y-axis limits for waveform plots in microvolts.
            None for automatic scaling.

        figsize_histograms : tuple, default: (15, 10)
            Figure size (width, height) in inches for histogram plot.

    Examples
    --------
    >>> params = get_default_qc_params()
    >>> # Stricter for chronic recordings
    >>> params["compute_distance_metrics"] = True
    >>> params["compute_drift"] = False  # Less relevant for chronic
    >>> labels, metrics, figs = run_bombcell_qc(analyzer, params=params)
    """
    return {
        # Which metrics to compute
        "compute_amplitude_cutoff": False,  # slow - requires spike_amplitudes
        "compute_distance_metrics": False,
        "compute_drift": True,
        # BombCell labeling options
        "split_non_somatic": False,
        "compute_valid_periods": False,
        "valid_periods_params": None,
        # Presence ratio
        "presence_ratio_bin_duration_s": 60,
        # Drift
        "drift_interval_s": 60,
        "drift_min_spikes": 100,
        # Plotting
        "plot_histograms": True,
        "plot_waveforms": True,
        "plot_upset": True,
        "waveform_ylims": (-300, 100),
        "figsize_histograms": (15, 10),
    }


def run_bombcell_qc(
    sorting_analyzer,
    output_folder: str | Path = "bombcell",
    params: dict | str | Path | None = None,
    thresholds: dict | str | Path | None = None,
    rerun_quality_metrics: bool = False,
    rerun_pca: bool = False,
    rerun_amplitude_scalings: bool = False,
    n_jobs: int = -1,
    progress_bar: bool = True,
):
    """
    Compute quality metrics and run BombCell unit labeling.

    This function computes quality metrics on the SortingAnalyzer, runs the
    BombCell labeling algorithm to classify units as good/mua/noise/non_soma,
    generates diagnostic plots, and saves results.

    Parameters
    ----------
    sorting_analyzer : SortingAnalyzer
        Analyzer with template_metrics extension computed.
    output_folder : str or Path, default: "bombcell"
        Folder to save results (CSV files and plots). Set to None to skip saving.
        Created if it doesn't exist.
    params : dict, str, Path, or None, default: None
        QC parameters from get_default_qc_params(), or a path to a JSON file
        containing such a dict. If None, uses defaults.

        To override the default metric list built from the compute_* flags, set
        params["metric_names"] to an explicit list of metric names. To override
        the default metric params, set params["metric_params"] to a dict mapping
        metric name -> param dict. Any metric you add this way must correspond to
        a valid SpikeInterface quality metric.
    thresholds : dict, str, Path, or None, default: None
        BombCell classification thresholds from bombcell_get_default_thresholds(),
        or a path to a JSON file containing such a dict. If None, uses defaults.
        Structure:

        - "noise": Thresholds for waveform quality. Failing ANY -> "noise".
        - "mua": Thresholds for spike quality. Failing ANY -> "mua".
        - "non-somatic": Thresholds for waveform shape. Determines non-somatic units.

        Each threshold is {"greater": value, "less": value}. Use None to disable.
        See bombcell_get_default_thresholds() docstring for all thresholds.

        Refractory-period violations: ``thresholds["mua"]`` must contain exactly
        ONE of ``"sliding_rp_violation"`` or ``"rp_contamination"``. That single
        entry is the ONE place where the RPV method is selected — it determines
        both which RPV metric is computed AND the threshold applied to it.

    rerun_quality_metrics : bool, default: False
        Force recomputation of quality metrics even if they exist.
    rerun_pca : bool, default: False
        Force recomputation of PCA (only relevant if compute_distance_metrics=True).
    rerun_amplitude_scalings : bool, default: False
        Force recomputation of amplitude_scalings (used as a prerequisite for
        amplitude_cutoff and valid periods).
    n_jobs : int, default: -1
        Number of parallel jobs.
    progress_bar : bool, default: True
        Show progress bars.

    Returns
    -------
    labels : pd.DataFrame
        DataFrame with unit_ids as index and "bombcell_label" column.
        Possible labels: "good", "mua", "noise", "non_soma"
        (or "non_soma_good"/"non_soma_mua" if split_non_somatic=True).
    metrics : pd.DataFrame
        Combined DataFrame of all quality metrics and template metrics.
        Index is unit_ids, columns are metric names.
    figures : dict
        Dictionary of matplotlib figures:
        - "histograms": Metric histograms with threshold lines.
        - "waveforms": Waveform overlays grouped by label.
        - "upset": List of UpSet plot figures (one per label type).

    Saved Files (in output_folder)
    ------------------------------
    - labeling_results_wide.csv: One row per unit with all metrics and label.
    - labeling_results_narrow.csv: One row per unit-metric with pass/fail status.
    - thresholds.json: Thresholds used for classification (reproducibility).
    - bombcell_config.json: bombcell-specific options (split_non_somatic,
      compute_valid_periods).
      Note: quality metric params are stored on the analyzer via the quality_metrics extension.
    - metric_histograms.png: Histogram of each metric with threshold lines.
    - waveforms_by_label.png: Waveform overlays for each label category.
    - upset_plot_*.png: UpSet plots showing metric failure combinations.

    Note on valid periods
    ---------------------
    When ``params["compute_valid_periods"]=True``, the valid time periods per unit
    are stored by the ``valid_unit_periods`` extension on the analyzer itself.
    Access them via ``sorting_analyzer.get_extension("valid_unit_periods").get_data()``.
    If you want to customize the ``valid_unit_periods`` parameters (``fp_threshold``,
    ``fn_threshold``, ``period_mode``, etc.), compute the extension upstream and
    the pipeline will reuse it. A warning is raised if its fp/fn differ from the
    bombcell RPV / amplitude_cutoff thresholds.

    Examples
    --------
    Basic usage with defaults:

    >>> labels, metrics, figs = run_bombcell_qc(analyzer)

    With custom parameters and thresholds:

    >>> params = get_default_qc_params()
    >>> params["compute_distance_metrics"] = True  # For chronic recordings
    >>>
    >>> thresholds = bombcell_get_default_thresholds()
    >>> thresholds["mua"]["sliding_rp_violation"]["less"] = 0.05  # Stricter RP violations
    >>> thresholds["mua"]["num_spikes"]["greater"] = 100  # Lower spike threshold
    >>>
    >>> labels, metrics, figs = run_bombcell_qc(
    ...     analyzer,
    ...     output_folder="qc_results",
    ...     params=params,
    ...     thresholds=thresholds,
    ... )

    Get good units for downstream analysis:

    >>> good_units = labels[labels["bombcell_label"] == "good"].index.tolist()
    >>> mua_units = labels[labels["bombcell_label"] == "mua"].index.tolist()
    """
    import json

    bombcell_get_default_thresholds, bombcell_label_units, save_bombcell_results = _require_spikeinterface()

    # Resolve params (dict, JSON path, or None)
    if params is None:
        params = get_default_qc_params()
    elif isinstance(params, (str, Path)):
        with open(params, "r") as f:
            params = json.load(f)

    # Resolve thresholds (dict, JSON path, or None)
    if thresholds is None:
        thresholds = bombcell_get_default_thresholds()
    elif isinstance(thresholds, (str, Path)):
        with open(thresholds, "r") as f:
            thresholds = json.load(f)

    job_kwargs = dict(n_jobs=n_jobs, chunk_duration="1s", progress_bar=progress_bar)

    if output_folder is not None:
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

    # The RPV method is selected by whichever key the user puts in thresholds["mua"]
    # (either "sliding_rp_violation" or "rp_contamination"). This is the ONE place
    # the choice lives — no separate pipeline-level parameter.
    rpv_metric = _resolve_rpv_metric(thresholds)

    qm_params = {
        "presence_ratio": {"bin_duration_s": params["presence_ratio_bin_duration_s"]},
        "drift": {
            "interval_s": params["drift_interval_s"],
            "min_spikes_per_interval": params["drift_min_spikes"],
        },
    }

    # Build metric names (user can override via params["metric_names"])
    if "metric_names" in params and params["metric_names"] is not None:
        metric_names = list(params["metric_names"])
    else:
        metric_names = ["amplitude_median", "snr", "num_spikes", "presence_ratio", "firing_rate"]

        if params["compute_amplitude_cutoff"]:
            metric_names.append("amplitude_cutoff")

        metric_names.append(rpv_metric)

        if params["compute_drift"]:
            metric_names.append("drift")

        if params["compute_distance_metrics"]:
            metric_names.append("mahalanobis")

    compute_valid_periods = params["compute_valid_periods"]

    # Ensure prerequisite extensions are computed for whichever metrics are requested
    # amplitude_cutoff uses amplitude_scalings (not spike_amplitudes) in this pipeline
    needs_amplitude_scalings = "amplitude_cutoff" in metric_names or compute_valid_periods
    if needs_amplitude_scalings:
        if rerun_amplitude_scalings or not sorting_analyzer.has_extension("amplitude_scalings"):
            sorting_analyzer.compute("amplitude_scalings", **job_kwargs)

    if "mahalanobis" in metric_names:
        if not sorting_analyzer.has_extension("principal_components") or rerun_pca:
            # principal_components is computed from the waveforms, so that extension must exist.
            # We deliberately do not compute it here: computing "waveforms" invalidates every
            # extension derived from it, including "templates" and "template_metrics", and
            # recomputing those would silently drop whatever parameters the user chose.
            if not sorting_analyzer.has_extension("waveforms"):
                raise ValueError(
                    "compute_distance_metrics=True needs the 'waveforms' extension, which this "
                    "analyzer does not have. It cannot be computed here, because computing "
                    "'waveforms' invalidates the extensions derived from it ('templates', "
                    "'template_metrics'), discarding the parameters you computed them with. "
                    "Compute it yourself, before 'template_metrics':\n"
                    "    analyzer.compute(['random_spikes', 'noise_levels', 'waveforms', 'templates'])\n"
                    "    analyzer.compute('template_metrics', include_multi_channel_metrics=True)"
                )
            sorting_analyzer.compute("principal_components", n_components=5, mode="by_channel_local", **job_kwargs)

    # amplitude_median needs spike_amplitudes and drift needs spike_locations. Without these
    # SpikeInterface skips those metrics, and their thresholds are then dropped from the labeling.
    if "amplitude_median" in metric_names and not sorting_analyzer.has_extension("spike_amplitudes"):
        sorting_analyzer.compute("spike_amplitudes", **job_kwargs)

    if "drift" in metric_names and not sorting_analyzer.has_extension("spike_locations"):
        sorting_analyzer.compute("spike_locations", **job_kwargs)

    # User-provided metric_params override the defaults built above
    if "metric_params" in params and params["metric_params"] is not None:
        for metric, mp in params["metric_params"].items():
            qm_params[metric] = mp

    # Valid unit periods (pipeline-level, since it affects which QMs bombcell sees).
    # - If the extension already exists, reuse it as-is, but warn if its fp/fn thresholds
    #   don't line up with the bombcell RPV / amplitude_cutoff thresholds.
    # - If it doesn't exist, compute it using defaults (users who want custom fp/fn
    #   or period params should compute it upstream themselves).
    if compute_valid_periods:
        if sorting_analyzer.has_extension("valid_unit_periods"):
            _warn_if_valid_periods_mismatch(sorting_analyzer, thresholds)
        else:
            sorting_analyzer.compute("valid_unit_periods", **_valid_periods_params(thresholds, params), **job_kwargs)

    # Compute quality metrics
    if not sorting_analyzer.has_extension("quality_metrics") or rerun_quality_metrics:
        sorting_analyzer.compute(
            "quality_metrics",
            metric_names=metric_names,
            metric_params=qm_params,
            use_valid_periods=compute_valid_periods,
            **job_kwargs,
        )

    # Run BombCell (pure labeler — no extension mutation, no valid-periods magic)
    labels = bombcell_label_units(
        sorting_analyzer=sorting_analyzer,
        thresholds=thresholds,
        split_non_somatic=params["split_non_somatic"],
    )

    metrics = sorting_analyzer.get_metrics_extension_data()

    # Plots
    figures = {}
    if params["plot_histograms"] or params["plot_waveforms"] or params["plot_upset"]:
        import spikeinterface.widgets as sw

        if params["plot_histograms"]:
            w = sw.plot_metric_histograms(sorting_analyzer, thresholds, figsize=params["figsize_histograms"])
            figures["histograms"] = w.figure

        if params["plot_waveforms"]:
            w = sw.plot_unit_labels(sorting_analyzer, labels["bombcell_label"], ylims=params["waveform_ylims"])
            figures["waveforms"] = w.figure

        if params["plot_upset"]:
            w = sw.plot_bombcell_labels_upset(
                sorting_analyzer,
                unit_labels=labels["bombcell_label"],
                thresholds=thresholds,
                unit_labels_to_plot=["noise", "mua"],
            )
            figures["upset"] = w.figures

    # Save
    if output_folder is not None:
        import json

        save_bombcell_results(
            metrics=metrics,
            unit_label=labels["bombcell_label"].values,
            thresholds=thresholds,
            folder=output_folder,
        )
        # Save thresholds and bombcell-specific config so the run is reproducible
        # (quality metric params are stored on the analyzer itself via the extension)
        with open(output_folder / "thresholds.json", "w") as f:
            json.dump(thresholds, f, indent=2)
        bombcell_config = {
            "split_non_somatic": params["split_non_somatic"],
            "compute_valid_periods": compute_valid_periods,
        }
        with open(output_folder / "bombcell_config.json", "w") as f:
            json.dump(bombcell_config, f, indent=2)
        # Valid periods themselves live on the analyzer (analyzer.get_extension("valid_unit_periods")),
        # but the per-chunk rates behind them are worth having next to the results, the way
        # BombCell saves useTheseTimesStart/Stop.
        if compute_valid_periods:
            valid_periods_table = _valid_periods_table(sorting_analyzer)
            if valid_periods_table is not None:
                valid_periods_table.to_csv(output_folder / "valid_periods.csv", index=False)
        if "histograms" in figures:
            figures["histograms"].savefig(output_folder / "metric_histograms.png", dpi=150, bbox_inches="tight")
        if "waveforms" in figures:
            figures["waveforms"].savefig(output_folder / "waveforms_by_label.png", dpi=150, bbox_inches="tight")
        if "upset" in figures:
            for i, fig in enumerate(figures["upset"]):
                fig.savefig(output_folder / f"upset_plot_{i}.png", dpi=150, bbox_inches="tight")

    print(f"Labeled {len(labels)} units:")
    print(labels["bombcell_label"].value_counts().to_string())

    return labels, metrics, figures
