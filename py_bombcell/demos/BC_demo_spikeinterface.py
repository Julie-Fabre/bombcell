# %% [markdown]
# # BombCell on a SpikeInterface SortingAnalyzer
#
# This demo runs the full BombCell quality control pipeline on a
# [SpikeInterface](https://spikeinterface.readthedocs.io/) `SortingAnalyzer`, rather than on a
# Kilosort output folder (for that, see the other demos in this folder).
#
# It runs end to end on simulated data, so you can execute it as-is with no data and no paths to
# edit. To use your own data, replace the "Simulated data" cell with your recording and sorting.
#
# **Requirements**: SpikeInterface is an optional dependency of bombcell:
#
# ```
# pip install "bombcell[spikeinterface]"
# ```
#
# **How the pieces fit together**
#
# - `bombcell.preprocess_for_bombcell` (this repo) preprocesses the recording and builds a
#   `SortingAnalyzer` with the extensions bombcell needs.
# - `bombcell.run_bombcell_qc` (this repo) decides which quality metrics to compute, computes the
#   extensions they depend on, runs the labeling, plots and saves. It carries the BombCell-flavored
#   defaults.
# - `spikeinterface.curation.bombcell_label_units` (SpikeInterface) is the labeler itself: it reads
#   metrics and applies thresholds. It computes nothing.

# %%
import warnings
from pathlib import Path
from pprint import pprint

import spikeinterface.full as si
import spikeinterface.curation as sc
import spikeinterface.widgets as sw

import bombcell

# %% [markdown]
# ## Simulated data
#
# We generate a small ground-truth recording and its sorting, so this notebook runs with no data
# and nothing installed beyond bombcell and SpikeInterface. Replace this cell with your own
# recording, e.g. `si.read_spikeglx(...)`.

# %%
recording, sorting = si.generate_ground_truth_recording(
    num_channels=16,
    num_units=15,
    durations=[300.0],
    seed=0,
)
print(recording)

# %% [markdown]
# ## Spike sorting
#
# The recommended sorter is **Kilosort4**:
#
# ```python
# import spikeinterface.sorters as ss
# sorting = ss.run_sorter("kilosort4", recording, folder="ks4_output")
# ```
#
# It needs `pip install kilosort` (and a GPU to be quick), so this notebook uses the simulated
# ground-truth sorting instead of running it — that keeps the demo executable anywhere. Everything
# downstream is identical either way: bombcell only ever sees a sorting.
#
# **Drift correction.** SpikeInterface runs Kilosort4 with `do_correction=True`, so Kilosort4
# corrects drift itself. That is why `apply_motion_correction` below defaults to `False`: turning
# it on as well would correct the same drift twice. Only enable it for a sorter that does not
# correct drift, and remember it has to happen *before* sorting to mean anything — correcting the
# recording afterwards does not move spikes that were already sorted from uncorrected traces.

# %% [markdown]
# ## Preprocessing
#
# `preprocess_for_bombcell` applies the preprocessing bombcell expects and builds the
# `SortingAnalyzer` with the extensions it needs:
#
# highpass filter → detect and drop bad channels → phase shift → common median reference,
# then `random_spikes`, `waveforms`, `templates`, `noise_levels`, `unit_locations`,
# `spike_locations` and `template_metrics`.
#
# Every step is a parameter, so you can turn off whatever your data does not need. Two notes for
# this simulated recording: bad-channel detection wants a realistic probe geometry, and phase
# shift only applies to probes with multiplexed ADCs (Neuropixels read through SpikeGLX or
# OpenEphys) — it is skipped with a warning otherwise.
#
# Already have an analyzer? Skip this cell and use `si.load_sorting_analyzer(...)`, but make sure
# `template_metrics` is computed: that is where the waveform-shape metrics come from.

# %%
preprocessing_params = bombcell.get_default_preprocessing_params()
preprocessing_params["detect_bad_channels"] = False  # simulated data has no probe geometry
preprocessing_params["apply_phase_shift"] = False  # simulated data has no ADC metadata
pprint(preprocessing_params)

# %%
analyzer, recording_preprocessed, bad_channel_ids = bombcell.preprocess_for_bombcell(
    recording,
    sorting,
    analyzer_folder=Path("bombcell_si_demo_output") / "analyzer.zarr",
    params=preprocessing_params,
    n_jobs=1,
    progress_bar=False,
)

print(f"{analyzer.unit_ids.size} units")
print("bad channels:", bad_channel_ids)
print("extensions:", sorted(analyzer.get_loaded_extension_names()))

# %% [markdown]
# ## QC parameters
#
# `get_default_qc_params()` returns the knobs controlling *which* metrics get computed and how.
# The pipeline computes prerequisite extensions for you: `spike_amplitudes` for `amplitude_median`,
# `spike_locations` for drift, `amplitude_scalings` for `amplitude_cutoff`, and principal
# components for the distance metrics.

# %%
qc_params = bombcell.get_default_qc_params()

# --- Metrics to compute ---
qc_params["compute_amplitude_cutoff"] = True   # estimate missing spikes
qc_params["compute_drift"] = True              # position changes over time
qc_params["compute_distance_metrics"] = True   # isolation distance & L-ratio. Slow, and not drift-robust,
                                               # so recommended for stable/chronic recordings.
                                               # Needs the waveforms extension, which the
                                               # preprocessing step above computed.

# --- Labeling options ---
# The refractory-period-violation method is NOT set here: it is chosen by which key you put in
# thresholds["mua"] below. That single entry picks both the metric and its threshold.
qc_params["split_non_somatic"] = False    # if True, non-somatic units split into good/mua subcategories
qc_params["compute_valid_periods"] = False  # restrict metrics to each unit's stable periods

# --- Metric parameters ---
qc_params["presence_ratio_bin_duration_s"] = 60  # bin size (s) for "does the unit fire throughout?"
qc_params["drift_interval_s"] = 60               # time bin (s) for position over time
qc_params["drift_min_spikes"] = 100              # min spikes per bin to estimate position

# --- Plotting ---
qc_params["plot_histograms"] = True
qc_params["plot_waveforms"] = True
qc_params["plot_upset"] = True

pprint(qc_params)

# %% [markdown]
# To bypass the `compute_*` flags entirely and name the metrics yourself:
#
# ```python
# qc_params["metric_names"] = [
#     "amplitude_median", "snr", "num_spikes", "presence_ratio",
#     "firing_rate", "sliding_rp_violation", "drift",
# ]
# qc_params["metric_params"] = {"drift": {"interval_s": 30}}
# ```

# %% [markdown]
# ## Classification thresholds
#
# Each threshold is `{"greater": min, "less": max}` and a unit passes when `min < value < max`.
# Use `None` to disable a bound, and add `"abs": True` to compare on absolute value.
#
# There are three sections:
#
# - **noise** — waveform quality. Failing *any* threshold means `noise`.
# - **mua** — spike quality, applied only to units that passed noise. Failing *any* means `mua`.
# - **non-somatic** — waveform shape, used to detect axonal/dendritic units.
#
# **Refractory-period violations**: `thresholds["mua"]` should contain exactly one of
# `sliding_rp_violation` or `rp_contamination`. That one entry selects both which metric gets
# computed and the threshold applied to it.

# %%
thresholds = sc.bombcell_get_default_thresholds()
pprint(thresholds)

# %% [markdown]
# Adjust any of them before running. A few examples:
#
# ```python
# thresholds["mua"]["sliding_rp_violation"] = {"greater": None, "less": 0.05}  # stricter
# thresholds["mua"]["num_spikes"] = {"greater": 100, "less": None}             # lower spike count
# thresholds["mua"]["drift_ptp"] = {"greater": None, "less": None}             # disable a threshold
# ```
#
# You can also add **any** metric present in the analyzer's `quality_metrics` or `template_metrics`
# to any section, and it is applied like the built-in ones:
#
# ```python
# thresholds["mua"]["firing_rate"] = {"greater": 0.1, "less": None}
# thresholds["noise"]["half_width"] = {"greater": 0.05e-3, "less": 0.6e-3}
# ```
#
# Metrics that were never computed are skipped, with a warning saying which. `isolation_distance`
# and `l_ratio` need principal components, so they only appear because `compute_distance_metrics`
# is set above.

# %% [markdown]
# ## Run the pipeline
#
# This computes the quality metrics, labels every unit, makes the figures and writes everything to
# `output_folder`. Pass `output_folder=None` to skip saving.
#
# `params` and `thresholds` also accept a path to a JSON file. After each run the thresholds and
# the BombCell config are written next to the results, so a run can be reproduced.

# %%
output_folder = Path("bombcell_si_demo_output")

labels, metrics, figures = bombcell.run_bombcell_qc(
    sorting_analyzer=analyzer,
    output_folder=output_folder,
    params=qc_params,
    thresholds=thresholds,
    n_jobs=1,
    progress_bar=False,
)

# %% [markdown]
# ## Results

# %%
print(f"{metrics.shape[0]} units x {metrics.shape[1]} metrics\n")
print(labels["bombcell_label"].value_counts().to_string())

good_units = labels[labels["bombcell_label"] == "good"].index.tolist()
mua_units = labels[labels["bombcell_label"] == "mua"].index.tolist()
noise_units = labels[labels["bombcell_label"] == "noise"].index.tolist()

print(f"\ngood  ({len(good_units)}): {good_units}")
print(f"mua   ({len(mua_units)}): {mua_units}")
print(f"noise ({len(noise_units)}): {noise_units}")

# %%
sorted(p.name for p in output_folder.iterdir())

# %% [markdown]
# ## Figures
#
# The pipeline returns the figures it made, and saves them alongside the CSVs. Metric histograms
# with the thresholds drawn on:

# %%
figures["histograms"]

# %% [markdown]
# Waveforms grouped by label:

# %%
figures["waveforms"]

# %% [markdown]
# The UpSet plot shows *which combinations* of metrics units failed on, which is the quickest way
# to see whether one over-strict threshold is doing all the work:

# %%
figures["upset"][0] if figures.get("upset") else "no upset plot (a single metric explained every failure)"

# %% [markdown]
# You can also plot the labeled waveforms directly from the analyzer:

# %%
_ = sw.plot_unit_labels(analyzer, labels["bombcell_label"], ylims=(-300, 100))

# %% [markdown]
# ## Using the labeler on its own
#
# `bombcell_label_units` applies thresholds to metrics and nothing else - no computation, no
# plotting, no saving. Use it when you have already computed what you need, or when your metrics
# come from somewhere other than an analyzer.

# %%
labels_direct = sc.bombcell_label_units(
    sorting_analyzer=analyzer,
    thresholds=thresholds,
)
print((labels_direct["bombcell_label"] == labels["bombcell_label"]).all(), "- same labels as the pipeline")

# %% [markdown]
# It also takes a plain DataFrame of metrics, indexed by unit id:
#
# ```python
# import pandas as pd
# my_metrics = pd.read_csv("my_metrics.csv", index_col=0)
# labels_direct = bombcell_label_units(external_metrics=my_metrics, thresholds=thresholds)
# ```
#
# Because it computes nothing, labels restricted to time chunks need the metrics prepared first -
# see the time chunks section below.

# %% [markdown]
# ## Time chunks (valid time periods)
#
# BombCell can split the recording into time chunks and compute a unit's metrics only on the
# chunks where it is well isolated, which gives a fairer picture of units that drift or drop out
# partway through. In SpikeInterface these are the **valid periods** of the `valid_unit_periods`
# extension, and turning them on is a single flag: `compute_valid_periods`.
#
# A chunk is kept when both its refractory-period violations and its fraction of missing spikes
# are under threshold. Those limits are taken from your BombCell thresholds (the RPV metric and
# `amplitude_cutoff` under `thresholds["mua"]`), so the chunks are selected with the same criteria
# the labels use.
#
# Every unit in the simulated recording is equally clean from start to finish, so to show what
# time chunks do we spoil one good unit for the first minute only: we add a second spike 0.7 ms
# after each of its spikes, which puts them inside the refractory period. That mimics a unit that
# is contaminated by a neighbour for part of the session.

# %%
import numpy as np

contaminated_unit = good_units[0]
fs = sorting.sampling_frequency

spike_trains = {unit_id: sorting.get_unit_spike_train(unit_id) for unit_id in sorting.unit_ids}
spike_train = spike_trains[contaminated_unit]
first_minute = spike_train[spike_train < 60 * fs]
doublets = first_minute + int(0.0007 * fs)  # 0.7 ms later: inside the refractory period
spike_trains[contaminated_unit] = np.sort(np.concatenate([spike_train, doublets]))
sorting_contaminated = si.NumpySorting.from_unit_dict(spike_trains, fs)

print(f"unit {contaminated_unit}: added {doublets.size} refractory violations in the first 60 s")

analyzer_contaminated, _, _ = bombcell.preprocess_for_bombcell(
    recording,
    sorting_contaminated,
    analyzer_folder=Path("bombcell_si_demo_output_time_chunks") / "analyzer.zarr",
    params=preprocessing_params,
    n_jobs=1,
    progress_bar=False,
)

# %% [markdown]
# Now run the pipeline twice on it: once on the whole recording, once with time chunks.
#
# The chunk length defaults to 360 s, BombCell's `deltaTimeChunk`. This simulated recording is only
# 300 s long, so we use 60 s chunks here. Anything in `valid_periods_params` is passed straight to
# the extension and overrides the defaults.

# %%
qc_params_chunks = dict(qc_params)
qc_params_chunks["compute_distance_metrics"] = False  # not needed here, and keeps this quick
qc_params_chunks["plot_histograms"] = False
qc_params_chunks["plot_waveforms"] = False
qc_params_chunks["plot_upset"] = False

# Whole recording
qc_params_chunks["compute_valid_periods"] = False
labels_whole, metrics_whole, _ = bombcell.run_bombcell_qc(
    sorting_analyzer=analyzer_contaminated,
    output_folder=None,
    params=qc_params_chunks,
    thresholds=thresholds,
    n_jobs=1,
    progress_bar=False,
)

# Time chunks
qc_params_chunks["compute_valid_periods"] = True
qc_params_chunks["valid_periods_params"] = {
    "period_duration_s_absolute": 60,      # chunk length (s); default 360
    "minimum_valid_period_duration": 120,  # a unit needs at least this much good time (s)
}
output_folder_chunks = Path("bombcell_si_demo_output_time_chunks")
labels_chunks, metrics_chunks, _ = bombcell.run_bombcell_qc(
    sorting_analyzer=analyzer_contaminated,
    output_folder=output_folder_chunks,
    params=qc_params_chunks,
    thresholds=thresholds,
    n_jobs=1,
    progress_bar=False,
)

vp_params = analyzer_contaminated.get_extension("valid_unit_periods").params
print("chunk length (s):", vp_params["period_duration_s_absolute"])
print("max RPV fraction (fp_threshold):", vp_params["fp_threshold"])
print("max fraction of spikes missing (fn_threshold):", vp_params["fn_threshold"])

# %% [markdown]
# Every chunk of every unit, with the rates it was judged on and whether it was kept. The same
# table is saved as `valid_periods.csv` next to the other results. Here are the contaminated
# unit's chunks: the contaminated first minute is dropped, and so is the next chunk, whose
# amplitudes are judged together with its neighbours.

# %%
import pandas as pd

chunks = pd.read_csv(output_folder_chunks / "valid_periods.csv")
chunks[chunks["unit_id"].astype(str) == str(contaminated_unit)]

# %%
kept_per_unit = chunks.groupby("unit_id")["kept"].agg(["sum", "count"])
kept_per_unit.columns = ["chunks_kept", "chunks_total"]
kept_per_unit.T

# %% [markdown]
# The kept periods drawn over the spike amplitudes, for the contaminated unit and a clean one:

# %%
clean_unit = good_units[1]
_ = sw.plot_valid_unit_periods(analyzer_contaminated, unit_ids=[contaminated_unit, clean_unit])

# %% [markdown]
# On the whole recording the contaminated unit is labelled `mua`. Its `sliding_rp_violation` is
# NaN: the contamination is too high to estimate, which counts as failing the threshold. With
# time chunks its metrics come only from the chunks that were kept, and it is `good` again. The
# clean units are unaffected.

# %%
rpv_metric = next(m for m in ("sliding_rp_violation", "rp_contamination") if m in thresholds["mua"])
comparison = pd.DataFrame(
    {
        f"{rpv_metric} (whole)": metrics_whole[rpv_metric],
        f"{rpv_metric} (chunks)": metrics_chunks[rpv_metric],
        "amplitude_cutoff (whole)": metrics_whole["amplitude_cutoff"],
        "amplitude_cutoff (chunks)": metrics_chunks["amplitude_cutoff"],
        "label (whole)": labels_whole["bombcell_label"],
        "label (chunks)": labels_chunks["bombcell_label"],
    }
)
print(pd.crosstab(comparison["label (whole)"], comparison["label (chunks)"]))
comparison[comparison["label (whole)"] != comparison["label (chunks)"]]

# %% [markdown]
# **Tuning the chunks yourself.** To choose the criteria directly instead of deriving them from
# your BombCell thresholds, compute the extension before running the pipeline. The pipeline then
# reuses it as-is, and warns if its `fp_threshold`/`fn_threshold` disagree with your BombCell
# thresholds:
#
# ```python
# analyzer.compute("amplitude_scalings")
# analyzer.compute(
#     "valid_unit_periods",
#     period_mode="absolute",
#     period_duration_s_absolute=360,
#     fp_threshold=0.1,   # match your bombcell RPV threshold
#     fn_threshold=0.1,   # match your bombcell amplitude_cutoff threshold
#     minimum_valid_period_duration=180,
# )
# qc_params["compute_valid_periods"] = True
# ```
#
# The periods themselves live on the analyzer: `analyzer.get_extension("valid_unit_periods").get_data()`.

# %% [markdown]
# ## Tuning by recording type
#
# **Chronic** — stable recordings make the distance metrics reliable, so set
# `compute_distance_metrics = True`. Drift is usually minimal and less informative.
#
# **Acute** — drift artificially lowers `isolation_distance` and `l_ratio`, so leave
# `compute_distance_metrics = False` (the default) and keep the drift threshold strict.
#
# **Cerebellum** — complex spikes can trip the noise detector; relax `num_positive_peaks`.
#
# **Striatum** — MSNs fire sparsely, so lower the spike-count and presence-ratio thresholds.
