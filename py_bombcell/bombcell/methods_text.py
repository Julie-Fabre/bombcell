"""
Methods text generator for bombcell.

Generates a ready-to-use methods section paragraph for scientific papers,
with appropriate citations, based on the parameters used to run bombcell.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ========================================================================
# Citation database
# ========================================================================

CITATIONS = {
    "bombcell": {
        "inline": "Fabre et al., 2023",
        "full": (
            "Fabre, J.M.J., van Beest, E.H., Peters, A.J., Carandini, M., "
            "& Harris, K.D. (2023). Bombcell: automated curation and cell "
            "classification of spike-sorted electrophysiology data. Zenodo. "
            "https://doi.org/10.5281/zenodo.8172821"
        ),
        "bibtex": (
            "@misc{fabre2023bombcell,\n"
            "  author = {Fabre, Julie M. J. and van Beest, Enny H. and Peters, Andrew J. and Carandini, Matteo and Harris, Kenneth D.},\n"
            "  title = {Bombcell: automated curation and cell classification of spike-sorted electrophysiology data},\n"
            "  year = {2023},\n"
            "  publisher = {Zenodo},\n"
            "  doi = {10.5281/zenodo.8172821},\n"
            "  url = {https://doi.org/10.5281/zenodo.8172821}\n"
            "}"
        ),
    },
    "hill2011": {
        "inline": "Hill et al., 2011",
        "full": (
            "Hill, D.N., Mehta, S.B., & Kleinfeld, D. (2011). Quality metrics "
            "to accompany spike sorting of extracellular signals. Journal of "
            "Neuroscience, 31(24), 8699-8705."
        ),
        "bibtex": (
            "@article{hill2011quality,\n"
            "  author = {Hill, Daniel N. and Mehta, Samar B. and Kleinfeld, David},\n"
            "  title = {Quality metrics to accompany spike sorting of extracellular signals},\n"
            "  journal = {Journal of Neuroscience},\n"
            "  volume = {31},\n"
            "  number = {24},\n"
            "  pages = {8699--8705},\n"
            "  year = {2011}\n"
            "}"
        ),
    },
    "llobet2022": {
        "inline": "Llobet et al., 2022",
        "full": (
            "Llobet, V., Wyngaard, D., Bhatt, D.K., Marx, M., & Bhatt, S. (2022). "
            "A sliding window framework for optimal refractory period estimation. "
            "bioRxiv."
        ),
        "bibtex": (
            "@article{llobet2022sliding,\n"
            "  author = {Llobet, Vincent and Wyngaard, Darik and Bhatt, Devika K. and Marx, Marius and Bhatt, Shashwat},\n"
            "  title = {A sliding window framework for optimal refractory period estimation},\n"
            "  journal = {bioRxiv},\n"
            "  year = {2022}\n"
            "}"
        ),
    },
    "deligkaris2016": {
        "inline": "Deligkaris et al., 2016",
        "full": (
            "Deligkaris, K., Bullmann, T., & Frey, U. (2016). Extracellularly "
            "recorded somatic and neuritic signal shapes and classification "
            "algorithms for high-density microelectrode array electrophysiology. "
            "Frontiers in Neuroscience, 10, 421. "
            "https://doi.org/10.3389/fnins.2016.00421"
        ),
        "bibtex": (
            "@article{deligkaris2016extracellularly,\n"
            "  author = {Deligkaris, Kosmas and Bullmann, Torsten and Frey, Urs},\n"
            "  title = {Extracellularly recorded somatic and neuritic signal shapes and classification algorithms for high-density microelectrode array electrophysiology},\n"
            "  journal = {Frontiers in Neuroscience},\n"
            "  volume = {10},\n"
            "  pages = {421},\n"
            "  year = {2016},\n"
            "  doi = {10.3389/fnins.2016.00421}\n"
            "}"
        ),
    },
    "siegle2021": {
        "inline": "Siegle et al., 2021",
        "full": (
            "Siegle, J.H., Jia, X., Durand, S., et al. (2021). Survey of spiking "
            "in the mouse visual system reveals functional hierarchy. Nature, 592, "
            "86-92. https://doi.org/10.1038/s41586-020-03171-x"
        ),
        "bibtex": (
            "@article{siegle2021survey,\n"
            "  author = {Siegle, Joshua H. and Jia, Xiaoxuan and Durand, S{\\'e}verine and others},\n"
            "  title = {Survey of spiking in the mouse visual system reveals functional hierarchy},\n"
            "  journal = {Nature},\n"
            "  volume = {592},\n"
            "  pages = {86--92},\n"
            "  year = {2021},\n"
            "  doi = {10.1038/s41586-020-03171-x}\n"
            "}"
        ),
    },
}


# ========================================================================
# Citation tracker
# ========================================================================

class _CitationTracker:
    """Tracks which citations have been used and formats them."""

    def __init__(self, style="inline"):
        self.style = style
        self._used_keys = []
        self._key_set = set()

    def cite(self, key):
        """Return formatted parenthetical citation: (Author et al., year) or [N]."""
        if key not in self._key_set:
            self._key_set.add(key)
            self._used_keys.append(key)
        if self.style == "inline":
            return f"({CITATIONS[key]['inline']})"
        else:
            idx = self._used_keys.index(key) + 1
            return f"[{idx}]"

    def textcite(self, key):
        """Return narrative citation: Author et al. (year) or Author et al. [N].
        Use when the author name is part of the sentence."""
        if key not in self._key_set:
            self._key_set.add(key)
            self._used_keys.append(key)
        inline = CITATIONS[key]["inline"]  # e.g. "Hill et al., 2011"
        if self.style == "inline":
            # "Hill et al., 2011" -> "Hill et al. (2011)"
            parts = inline.rsplit(", ", 1)
            return f"{parts[0]} ({parts[1]})"
        else:
            idx = self._used_keys.index(key) + 1
            parts = inline.rsplit(", ", 1)
            return f"{parts[0]} [{idx}]"

    def get_references(self):
        """Return ordered list of full reference strings."""
        refs = []
        for i, key in enumerate(self._used_keys):
            if self.style == "numbered":
                refs.append(f"[{i + 1}] {CITATIONS[key]['full']}")
            else:
                refs.append(CITATIONS[key]["full"])
        return refs

    def get_bibtex(self):
        """Return BibTeX entries for all cited references."""
        return [CITATIONS[key]["bibtex"] for key in self._used_keys]


# ========================================================================
# Section generators
# ========================================================================

def _intro_section(param, tracker):
    if param.get("splitGoodAndMua_NonSomatic", False):
        categories = (
            "noise, non-somatic good, non-somatic multi-unit activity "
            "(MUA), multi-unit activity (MUA), and good single units"
        )
    else:
        categories = (
            "noise, non-somatic, multi-unit activity (MUA), and good "
            "single units"
        )
    return (
        f"Automated spike sorting quality control was performed using "
        f"BombCell {tracker.cite('bombcell')}. Each unit was sequentially "
        f"evaluated and classified as {categories}, based on the quality "
        f"metrics described below."
    )


def _peak_trough_detection_section(param, tracker):
    thresh = param.get("minThreshDetectPeaksTroughs", 0.2)
    return (
        "Noise and non-somatic quality metrics were derived from each "
        "unit's template waveform on its peak channel (the channel with "
        "the largest amplitude). Peaks and troughs were identified using "
        "a prominence-based peak-finding algorithm, with the minimum "
        f"prominence threshold set to {thresh} times the waveform's "
        "absolute maximum. The main trough was defined as the most "
        "prominent trough, and peaks were detected separately on either "
        "side of it."
    )


def _noise_section(param, tracker):
    max_n_peaks = param.get("maxNPeaks", 2)
    max_n_troughs = param.get("maxNTroughs", 1)
    min_wv_dur = param.get("minWvDuration", 100)
    max_wv_dur = param.get("maxWvDuration", 1150)
    max_baseline = param.get("maxWvBaselineFraction", 0.3)
    max_scnd_peak = param.get("maxScndPeakToTroughRatio_noise", 0.8)

    text = (
        "Units were first evaluated for noise-like waveform features. "
        "A unit was classified as noise if any of the following criteria "
        "were met: the waveform contained more than "
        f"{max_n_peaks} peaks or more than {max_n_troughs} "
        f"{'trough' if max_n_troughs == 1 else 'troughs'}; "
        "the peak-to-trough duration (time between the largest peak and "
        f"largest trough) fell outside {min_wv_dur}\u2013{max_wv_dur} "
        "\u00b5s; the waveform baseline fraction (maximum absolute value "
        "in the baseline period preceding the spike, relative to the "
        f"waveform's absolute peak) exceeded {max_baseline}; or the "
        "ratio of the second peak amplitude to the main trough amplitude "
        f"exceeded {max_scnd_peak}."
    )

    if param.get("computeSpatialDecay", True):
        if param.get("spDecayLinFit", False):
            fit_type = "linear"
            n_channels = 6
        else:
            fit_type = "exponential"
            n_channels = 10
        sp_text = (
            " Additionally, the spatial decay of the waveform was "
            "quantified by measuring the absolute peak amplitude on the "
            f"{n_channels} nearest channels within 33 \u00b5m in the "
            "x-dimension, as a function of Euclidean distance (in \u00b5m) "
            "from the peak channel"
        )
        if param.get("normalizeSpDecay", True):
            sp_text += ". Amplitudes were normalized to the peak channel"
        sp_text += (
            f", and {fit_type == 'exponential' and 'an' or 'a'} "
            f"{fit_type} fit was applied to obtain the decay slope"
        )
        if param.get("spDecayLinFit", False):
            min_slope = param.get("minSpatialDecaySlope", -0.008)
            sp_text += (
                f". Units with a decay slope less than {min_slope} were "
                "classified as noise."
            )
        else:
            min_slope = param.get("minSpatialDecaySlopeExp", 0.01)
            max_slope = param.get("maxSpatialDecaySlopeExp", 0.1)
            sp_text += (
                f". Units with a decay slope outside the range "
                f"{min_slope}\u2013{max_slope} were classified as noise."
            )
        text += sp_text

    return text


def _nonsomatic_section(param, tracker):
    max_main_ptr = param.get("maxMainPeakToTroughRatio_nonSomatic", 0.8)
    max_p1p2 = param.get("maxPeak1ToPeak2Ratio_nonSomatic", 3)
    min_t2p2 = param.get("minTroughToPeak2Ratio_nonSomatic", 5)
    min_w_peak = param.get("minWidthFirstPeak_nonSomatic", 4)
    min_w_trough = param.get("minWidthMainTrough_nonSomatic", 5)

    text = (
        "Non-noise units were then evaluated for non-somatic (e.g., axonal) "
        f"waveform characteristics {tracker.cite('deligkaris2016')}. "
        "A unit was classified as non-somatic if any of the following "
        "criteria were met: "
        "the ratio of the largest peak amplitude (before or after the "
        f"trough) to the main trough amplitude exceeded {max_main_ptr}; "
        "the ratio of the pre-trough peak amplitude to the post-trough "
        f"peak amplitude exceeded {max_p1p2}; "
        "the ratio of the main trough amplitude to the pre-trough peak "
        f"amplitude was less than {min_t2p2}; "
        "the half-prominence width of the pre-trough peak was less than "
        f"{min_w_peak} samples; "
        "or the half-prominence width of the main trough was less than "
        f"{min_w_trough} samples."
    )
    return text


def _mua_section(param, tracker):
    tau_r_min = param.get("tauR_valuesMin", 0.002) * 1000
    tau_r_max = param.get("tauR_valuesMax", 0.002) * 1000
    tau_c = param.get("tauC", 0.0001) * 1000
    max_rpv = param.get("maxRPVviolations", 0.1)
    max_pct = param.get("maxPercSpikesMissing", 20)
    min_spikes = param.get("minNumSpikes", 300)
    min_pr = param.get("minPresenceRatio", 0.7)
    bin_size = param.get("presenceRatioBinSize", 60)

    text = (
        "The remaining units were then assessed for single-unit isolation "
        "quality, and classified as multi-unit activity (MUA) if any of "
        "the following criteria were met."
    )

    # RPV
    if param.get("hillOrLlobetMethod", True):
        rpv_method = f"the method of {tracker.textcite('hill2011')}"
    else:
        rpv_method = f"the method of {tracker.textcite('llobet2022')}"
    rpv_text = (
        " The fraction of refractory period violations was estimated "
        f"using {rpv_method}"
    )
    if tau_r_min == tau_r_max:
        rpv_text += (
            f", with a refractory period of {tau_r_min:.1f} ms and a "
            f"censored period of {tau_c:.1f} ms"
        )
    else:
        tau_r_step = param.get("tauR_valuesStep", 0.0005) * 1000
        rpv_text += (
            f", testing refractory period values from {tau_r_min:.1f} to "
            f"{tau_r_max:.1f} ms in steps of {tau_r_step:.1f} ms (with a "
            f"censored period of {tau_c:.1f} ms), and selecting the "
            "optimal value per unit"
        )
    rpv_text += (
        f"; units exceeding {max_rpv * 100:.0f}% violations were "
        "classified as MUA."
    )
    text += rpv_text

    # Spikes missing
    text += (
        " The percentage of missing spikes was estimated by fitting a "
        "Gaussian to each unit's amplitude distribution and computing "
        "the fraction falling below the detection threshold; units with "
        f"more than {max_pct}% missing spikes were classified as MUA."
    )

    if param.get("computeTimeChunks", False):
        delta = param.get("deltaTimeChunk", 360)
        text += (
            f" Both the refractory period violation rate and the "
            f"percentage of missing spikes were additionally computed in "
            f"{delta}-second time chunks to identify the most stable "
            "recording period for each unit."
        )

    # Spike count
    text += (
        f" Units with fewer than {min_spikes} total spikes were "
        "classified as MUA."
    )

    # Presence ratio
    text += (
        f" The presence ratio was computed by dividing the recording into "
        f"{bin_size}-second bins and calculating the fraction of bins in "
        "which the unit's spike count exceeded 5% of the 90th-percentile "
        f"bin count (inspired by {tracker.textcite('siegle2021')}); units "
        f"with a presence ratio below {min_pr} were classified as MUA."
    )

    # Amplitude and SNR
    if param.get("extractRaw", True):
        min_amp = param.get("minAmplitude", 40)
        min_snr = param.get("minSNR", 5)
        n_spikes = param.get("nRawSpikesToExtract", 100)
        raw_text = (
            f" The raw waveform amplitude (peak-to-trough voltage in "
            f"\u00b5V) was computed from the mean of {n_spikes} randomly "
            "sampled raw waveforms"
        )
        if param.get("detrendWaveform", True):
            raw_text += " after linear de-trending"
        raw_text += (
            f"; units with an amplitude below {min_amp} \u00b5V were "
            "classified as MUA. The signal-to-noise ratio (SNR) was "
            "defined as the waveform amplitude divided by the standard "
            f"deviation of the baseline noise; units with an SNR below "
            f"{min_snr} were classified as MUA."
        )
        text += raw_text

    # Drift
    if param.get("computeDrift", False):
        max_drift = param.get("maxDrift", 100)
        drift_bin = param.get("driftBinSize", 60)
        text += (
            " Electrode drift was estimated by computing, in "
            f"{drift_bin}-second bins, the spike depth as a weighted "
            "center of mass of the first principal component features "
            "across channels. Maximum drift was defined as the difference "
            "between the maximum and minimum median spike depth across "
            f"bins; units with drift exceeding {max_drift} \u00b5m were "
            "classified as MUA."
        )

    # Distance metrics
    if param.get("computeDistanceMetrics", False):
        n_chan = param.get("nChannelsIsoDist", 4)
        iso_min = param.get("isoDmin", 20)
        lratio_max = param.get("lratioMax", 0.3)
        text += (
            " Cluster isolation was assessed using principal component "
            f"features from the {n_chan} nearest channels. Isolation "
            "distance (the Mahalanobis distance to the nearest non-member "
            f"spike) was required to be at least {iso_min}. The L-ratio "
            "(a contamination estimate based on the chi-squared "
            "distribution of Mahalanobis distances) was required to be "
            f"below {lratio_max}. Units failing either criterion were "
            "classified as MUA."
        )

    return text


def _good_section(param, tracker, quality_metrics=None):
    text = (
        "All remaining units that passed the above criteria were "
        "classified as good single units."
    )

    # Optional unit count summary
    if quality_metrics is not None:
        try:
            import numpy as np
            unit_types = quality_metrics.get("bc_unitType", None)
            if unit_types is None:
                from bombcell.quality_metrics import get_quality_unit_type
                _, unit_type_string = get_quality_unit_type(param, quality_metrics)
                unit_types = unit_type_string
            if unit_types is not None:
                n_total = len(unit_types)
                counts = {}
                for label in ["GOOD", "MUA", "NOISE", "NON-SOMA",
                              "NON-SOMA GOOD", "NON-SOMA MUA"]:
                    c = np.sum(unit_types == label)
                    if c > 0:
                        counts[label] = int(c)
                if counts:
                    parts = [f"{v} {k.lower()}" for k, v in counts.items()]
                    text += (
                        f" Of {n_total} total units, "
                        + ", ".join(parts)
                        + "."
                    )
        except Exception:
            pass

    return text


# ========================================================================
# Main public API
# ========================================================================

def generate_methods_text(
    param: dict,
    quality_metrics: Optional[dict] = None,
    citation_style: str = "inline",
) -> Tuple[str, List[str], List[str]]:
    """
    Generate a methods section paragraph from bombcell parameters.

    Parameters
    ----------
    param : dict
        The bombcell parameter dictionary (from get_default_parameters or
        loaded from a saved parquet file).
    quality_metrics : dict, optional
        The quality metrics dictionary. If provided, a summary of unit
        classification counts is appended.
    citation_style : str, optional
        "inline" for author-year citations (default) or "numbered" for
        bracketed numbers.

    Returns
    -------
    methods_text : str
        The generated methods paragraph.
    references : list of str
        The list of formatted references cited in the text.
    bibtex_entries : list of str
        The list of BibTeX entries for the cited references.
    """
    tracker = _CitationTracker(style=citation_style)

    sections = [
        _intro_section(param, tracker),
        _peak_trough_detection_section(param, tracker),
        _noise_section(param, tracker),
        _nonsomatic_section(param, tracker),
        _mua_section(param, tracker),
        _good_section(param, tracker, quality_metrics),
    ]

    # Filter empty sections and join with line breaks between sections
    text = "\n\n".join(s for s in sections if s)
    references = tracker.get_references()
    bibtex_entries = tracker.get_bibtex()

    return text, references, bibtex_entries


def print_methods_text(
    param: dict,
    quality_metrics: Optional[dict] = None,
    citation_style: str = "inline",
) -> None:
    """
    Print the generated methods text and references to stdout.

    Parameters
    ----------
    param : dict
        The bombcell parameter dictionary.
    quality_metrics : dict, optional
        The quality metrics dictionary.
    citation_style : str, optional
        "inline" or "numbered".
    """
    text, refs, bibtex = generate_methods_text(param, quality_metrics, citation_style)
    print("=" * 70)
    print("METHODS")
    print("=" * 70)
    print(text)
    print()
    print("=" * 70)
    print("REFERENCES")
    print("=" * 70)
    for ref in refs:
        print(f"  {ref}")
    print()
    print("=" * 70)
    print("BIBTEX")
    print("=" * 70)
    print("\n\n".join(bibtex))
    print()


def save_methods_text(
    param: dict,
    save_path: str,
    quality_metrics: Optional[dict] = None,
    citation_style: str = "inline",
) -> None:
    """
    Save the generated methods text and references to a text file.

    Parameters
    ----------
    param : dict
        The bombcell parameter dictionary.
    save_path : str
        Path to the output text file.
    quality_metrics : dict, optional
        The quality metrics dictionary.
    citation_style : str, optional
        "inline" or "numbered".
    """
    text, refs, bibtex = generate_methods_text(param, quality_metrics, citation_style)
    save_path = Path(save_path)

    # Save methods text with formatted references
    output = f"METHODS\n{'=' * 70}\n{text}\n\n"
    output += f"REFERENCES\n{'=' * 70}\n"
    for ref in refs:
        output += f"  {ref}\n"
    save_path.write_text(output)

    # Save BibTeX file alongside
    bib_path = save_path.with_suffix(".bib")
    bib_path.write_text("\n\n".join(bibtex) + "\n")
