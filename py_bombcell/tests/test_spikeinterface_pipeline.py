"""Smoke tests for the SpikeInterface entry point (bombcell.run_bombcell_qc).

These guard against SpikeInterface API drift: the pipeline calls SpikeInterface's public
API to compute extensions and metrics, so a rename or signature change upstream breaks it
silently. Everything here runs on a small simulated recording, so no data is needed.

SpikeInterface is an optional dependency, so the whole module is skipped when it is absent.
"""

import matplotlib
import pytest

matplotlib.use("Agg")

si = pytest.importorskip("spikeinterface.full", reason="spikeinterface is an optional dependency")
sc = pytest.importorskip("spikeinterface.curation", reason="spikeinterface is an optional dependency")

import bombcell

# Labels bombcell_label_units can produce.
VALID_LABELS = {"good", "mua", "noise", "non_soma", "non_soma_good", "non_soma_mua"}


@pytest.fixture(scope="module")
def analyzer():
    """A small simulated SortingAnalyzer with the extensions the pipeline expects up front."""
    recording, sorting = si.generate_ground_truth_recording(
        num_channels=8,
        num_units=10,
        durations=[300.0],  # long enough for several drift bins at the default 60 s interval
        seed=0,
    )
    analyzer = si.create_sorting_analyzer(sorting, recording, sparse=True)
    analyzer.compute({"random_spikes": {"seed": 0}, "noise_levels": {}, "templates": {}})
    analyzer.compute("template_metrics", include_multi_channel_metrics=True)
    return analyzer


@pytest.fixture(scope="module")
def qc_result(analyzer, tmp_path_factory):
    """Run the pipeline once and share the result across tests (it is slow)."""
    output_folder = tmp_path_factory.mktemp("bombcell_qc")
    params = bombcell.get_default_qc_params()
    # On by default it is slow, but here it is what exercises the amplitude_scalings
    # prerequisite path. Distance metrics stay off: they need PCA and are slower still.
    params["compute_amplitude_cutoff"] = True
    labels, metrics, figures = bombcell.run_bombcell_qc(
        sorting_analyzer=analyzer,
        output_folder=output_folder,
        params=params,
        n_jobs=1,
        progress_bar=False,
    )
    return labels, metrics, figures, output_folder


def test_get_default_qc_params_without_spikeinterface():
    """The params dict is pure data and must not need SpikeInterface to build."""
    params = bombcell.get_default_qc_params()
    for key in ("compute_amplitude_cutoff", "compute_distance_metrics", "compute_drift", "split_non_somatic"):
        assert key in params


def test_labels_cover_every_unit(analyzer, qc_result):
    labels, _, _, _ = qc_result
    assert len(labels) == analyzer.unit_ids.size
    assert set(labels["bombcell_label"]).issubset(VALID_LABELS)


def test_prerequisite_extensions_are_computed(analyzer, qc_result):
    """amplitude_median needs spike_amplitudes and drift needs spike_locations.

    Without these SpikeInterface skips those metrics and their thresholds are dropped from
    the labeling, so the pipeline has to compute them itself.
    """
    assert analyzer.has_extension("spike_amplitudes")
    assert analyzer.has_extension("spike_locations")
    assert analyzer.has_extension("amplitude_scalings")  # prerequisite for amplitude_cutoff
    assert analyzer.has_extension("quality_metrics")


def test_thresholded_metrics_are_present_and_computed(qc_result):
    """Every metric the default thresholds reference should actually make it into the metrics.

    The exception is the PCA-derived pair, which needs compute_distance_metrics=True.
    """
    _, metrics, _, _ = qc_result
    thresholds = sc.bombcell_get_default_thresholds()
    pca_only = {"isolation_distance", "l_ratio"}

    expected = {m for section in thresholds.values() for m in section} - pca_only
    missing = sorted(m for m in expected if m not in metrics.columns)
    assert missing == [], f"thresholded metrics absent from the metrics table: {missing}"

    # amplitude_median and drift_ptp regressed once by being requested without their
    # prerequisite extension, which left them entirely NaN rather than missing.
    for name in ("amplitude_median", "drift_ptp"):
        assert metrics[name].notna().any(), f"{name} was computed but is all-NaN"


def test_results_are_saved(qc_result):
    _, _, _, output_folder = qc_result
    written = {p.name for p in output_folder.iterdir()}
    for name in (
        "labeling_results_wide.csv",
        "labeling_results_narrow.csv",
        "thresholds.json",
        "bombcell_config.json",
    ):
        assert name in written, f"{name} not written (found: {sorted(written)})"


def test_figures_are_returned(qc_result):
    _, _, figures, _ = qc_result
    assert "histograms" in figures
    assert "waveforms" in figures


def test_labeler_alone_matches_the_pipeline(analyzer, qc_result):
    """bombcell_label_units on the analyzer the pipeline just populated must agree with it."""
    labels, _, _, _ = qc_result
    direct = sc.bombcell_label_units(sorting_analyzer=analyzer)
    assert (direct["bombcell_label"] == labels["bombcell_label"]).all()


def test_distance_metrics_need_waveforms(analyzer):
    """Without the waveforms extension the pipeline must refuse, not compute it.

    Computing "waveforms" invalidates "templates" and "template_metrics", so computing it
    here would silently discard the parameters the caller used for those.
    """
    params = bombcell.get_default_qc_params()
    params["compute_distance_metrics"] = True
    with pytest.raises(ValueError, match="waveforms"):
        bombcell.run_bombcell_qc(analyzer, output_folder=None, params=params, n_jobs=1, progress_bar=False)


def test_distance_metrics_with_waveforms(tmp_path):
    """With waveforms computed up front, the PCA metrics come through."""
    recording, sorting = si.generate_ground_truth_recording(
        num_channels=8, num_units=10, durations=[300.0], seed=0
    )
    a = si.create_sorting_analyzer(sorting, recording, sparse=True)
    a.compute({"random_spikes": {"seed": 0}, "noise_levels": {}, "waveforms": {}, "templates": {}})
    a.compute("template_metrics", include_multi_channel_metrics=True)

    params = bombcell.get_default_qc_params()
    params["compute_distance_metrics"] = True
    _, metrics, _ = bombcell.run_bombcell_qc(a, output_folder=None, params=params, n_jobs=1, progress_bar=False)

    assert a.has_extension("principal_components")
    assert a.has_extension("template_metrics"), "computing PCA must not invalidate template_metrics"
    for name in ("isolation_distance", "l_ratio"):
        assert name in metrics.columns, f"{name} missing despite compute_distance_metrics=True"
        assert metrics[name].notna().any(), f"{name} is all-NaN"


def test_rpv_metric_selection_is_validated():
    """The RPV method is chosen by which key sits in thresholds["mua"]; neither or both is an error."""
    from bombcell.spikeinterface_pipeline import _resolve_rpv_metric

    assert _resolve_rpv_metric({"mua": {"sliding_rp_violation": {"less": 0.1}}}) == "sliding_rp_violation"
    assert _resolve_rpv_metric({"mua": {"rp_contamination": {"less": 0.1}}}) == "rp_contamination"

    with pytest.raises(ValueError):
        _resolve_rpv_metric({"mua": {"snr": {"greater": 5}}})

    with pytest.raises(ValueError):
        _resolve_rpv_metric({"mua": {"sliding_rp_violation": {"less": 0.1}, "rp_contamination": {"less": 0.1}}})
