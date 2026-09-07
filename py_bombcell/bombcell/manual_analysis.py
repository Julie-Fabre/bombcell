"""
Manual classification analysis functions for BombCell

Functions for analyzing concordance between manual and automatic BombCell classifications,
and suggesting parameter threshold adjustments based on disagreements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Import BombCell modules needed for the wrapper function  
from .loading_utils import load_bc_results
from .quality_metrics import get_quality_unit_type


def load_manual_classifications(save_path):
    """
    Load manual classifications from CSV file
    
    Parameters
    ----------
    save_path : str or Path
        Path to BombCell output directory containing manual_unit_classifications.csv
        
    Returns
    -------
    manual_df : pd.DataFrame or None
        DataFrame with columns ['unit_id', 'manual_classification'] or None if file not found
    """
    save_path = Path(save_path)
    manual_file = save_path / "manual_unit_classifications.csv"
    
    if manual_file.exists():
        manual_df = pd.read_csv(manual_file)

        # The GUI writes -1 for units you have not classified yet. Counting those as
        # labels makes every summary below meaningless - they are not data.
        n_rows = len(manual_df)
        manual_df = manual_df[manual_df['manual_classification'] >= 0].reset_index(drop=True)
        n_unclassified = n_rows - len(manual_df)

        print(f"📂 Found manual classifications for {len(manual_df)} units")
        if n_unclassified > 0:
            print(f"   ({n_unclassified} units in the file are not classified yet - ignoring them)")
        if len(manual_df) == 0:
            print("❌ No units have been classified yet.")
            return None
        return manual_df
    else:
        print("❌ No manual classifications found. Please use the GUI to manually classify some units first.")
        print(f"Expected file: {manual_file}")
        return None


def analyze_classification_concordance(manual_df, quality_metrics_table, save_path=None, param=None):
    """
    Analyze concordance between manual and BombCell classifications
    
    Parameters
    ----------
    manual_df : pd.DataFrame
        Manual classifications with columns ['unit_id', 'manual_classification']
    quality_metrics_table : pd.DataFrame
        BombCell quality metrics table with 'phy_clusterID' column
    save_path : str or Path, optional
        Path to BombCell output directory containing unit type file
    param : dict, optional
        BombCell parameters. Used to know whether non-somatic units were split into
        good and MUA (param['splitGoodAndMua_NonSomatic']). If not given, this is
        inferred from the BombCell unit type labels.
        
    Returns
    -------
    merged_df : pd.DataFrame
        Merged dataframe with both manual and BombCell classifications
    confusion_df : pd.DataFrame
        Confusion matrix as crosstab
    concordance_stats : dict
        Dictionary with concordance statistics
    """
    if manual_df is None or len(manual_df) == 0:
        print("❌ No manual classifications available")
        return None, None, None
    
    # Load BombCell unit types from separate file
    bombcell_types = None
    if save_path is not None:
        save_path = Path(save_path)
        unit_type_file = save_path / "cluster_bc_unitType.tsv"
        if unit_type_file.exists():
            unit_types_df = pd.read_csv(unit_type_file, sep='\t')
            # Rename columns to match expected format
            unit_types_df = unit_types_df.rename(columns={'cluster_id': 'phy_clusterID', 'bc_unitType': 'Bombcell_unit_type'})
            bombcell_types = unit_types_df
        else:
            print(f"❌ BombCell unit types file not found: {unit_type_file}")
    
    # Try to get BombCell types from quality metrics table if not loaded separately
    if bombcell_types is None:
        if 'Bombcell_unit_type' in quality_metrics_table.columns:
            bombcell_types = quality_metrics_table[['phy_clusterID', 'Bombcell_unit_type']]
        else:
            print("❌ No BombCell unit types found in quality metrics table or separate file")
            return None, None, None
    
    # Merge manual and BombCell classifications
    merged_df = manual_df.merge(
        bombcell_types[['phy_clusterID', 'Bombcell_unit_type']], 
        left_on='unit_id', 
        right_on='phy_clusterID', 
        how='inner'
    )
    
    if len(merged_df) == 0:
        print("❌ No matching units found between manual and BombCell classifications")
        return None, None, None
    
    # Non-somatic units are split into good and MUA when splitGoodAndMua_NonSomatic is
    # set. Fall back to inferring this from the BombCell labels when param isn't given.
    if param is not None:
        split_nonsomatic = bool(param.get('splitGoodAndMua_NonSomatic', False))
    else:
        split_nonsomatic = merged_df['Bombcell_unit_type'].isin(
            ['NON-SOMA GOOD', 'NON-SOMA MUA']).any()
    
    # Create mapping of manual classification numbers to names
    reverse_mapping = {
        0: 'Noise',
        1: 'Good',
        2: 'MUA',
        3: 'Non-somatic good' if split_nonsomatic else 'Non-somatic',
        4: 'Non-somatic MUA',
    }
    
    # Convert manual classifications to BombCell format names
    merged_df['manual_type_name'] = merged_df['manual_classification'].map(reverse_mapping)
    
    # Handle any unmapped classifications
    merged_df['manual_type_name'] = merged_df['manual_type_name'].fillna('Unknown')
    
    # Normalize case for comparison - convert BombCell types to match manual format
    bc_case_mapping = {
        'NOISE': 'Noise',
        'GOOD': 'Good', 
        'MUA': 'MUA',
        'NON-SOMA': 'Non-somatic',
        'NON-SOMA GOOD': 'Non-somatic good',
        'NON-SOMA MUA': 'Non-somatic MUA'
    }
    merged_df['Bombcell_unit_type_normalized'] = merged_df['Bombcell_unit_type'].map(bc_case_mapping)
    merged_df['Bombcell_unit_type_normalized'] = merged_df['Bombcell_unit_type_normalized'].fillna(merged_df['Bombcell_unit_type'])
    
    # Calculate overall concordance (correct classifications)
    total_units = len(merged_df)
    concordant_units = (merged_df['manual_type_name'] == merged_df['Bombcell_unit_type_normalized']).sum()
    overall_concordance = concordant_units / total_units * 100
    
    print(f"📊 Classification Concordance Analysis")
    print(f"{'='*50}")
    print(f"Total classified units: {total_units}")
    print(f"Concordant classifications: {concordant_units}")
    print(f"Overall concordance: {overall_concordance:.1f}%")
    print(f"{'='*50}")
    
    # Create confusion matrix (BombCell as rows, Manual as columns)
    confusion_df = pd.crosstab(
        merged_df['Bombcell_unit_type_normalized'], 
        merged_df['manual_type_name'], 
        margins=True
    )
    print("\nConfusion Matrix (rows=BombCell, columns=Manual):")
    print(confusion_df)
    
    # Calculate per-class concordance (precision for BombCell classifications)
    print("\nPer-class concordance (BombCell classification accuracy):")
    concordance_by_class = {}
    for bc_type in confusion_df.index[:-1]:  # Exclude 'All' row
        if bc_type in confusion_df.columns:
            correct = confusion_df.loc[bc_type, bc_type]
            total_bc = confusion_df.loc[bc_type, 'All']
            concordance = correct / total_bc * 100 if total_bc > 0 else 0
            concordance_by_class[bc_type] = concordance
            print(f"  {bc_type}: {concordance:.1f}% ({correct}/{total_bc})")
        else:
            concordance_by_class[bc_type] = 0.0
            total_bc = confusion_df.loc[bc_type, 'All']
            print(f"  {bc_type}: 0.0% (0/{total_bc}) - no manual examples")
    
    # Calculate per-class recall (manual classification accuracy)
    print("\nPer-class recall (Manual classification coverage):")
    recall_by_class = {}
    for manual_type in confusion_df.columns[:-1]:  # Exclude 'All' column
        if manual_type in confusion_df.index:
            correct = confusion_df.loc[manual_type, manual_type]
            total_manual = confusion_df.loc['All', manual_type]
            recall = correct / total_manual * 100 if total_manual > 0 else 0
            recall_by_class[manual_type] = recall
            print(f"  {manual_type}: {recall:.1f}% ({correct}/{total_manual})")
        else:
            recall_by_class[manual_type] = 0.0
            total_manual = confusion_df.loc['All', manual_type]
            print(f"  {manual_type}: 0.0% (0/{total_manual}) - no BombCell examples")
    
    concordance_stats = {
        'overall_concordance': overall_concordance,
        'total_units': total_units,
        'concordant_units': concordant_units,
        'precision_by_class': concordance_by_class,
        'recall_by_class': recall_by_class
    }
    
    return merged_df, confusion_df, concordance_stats


# Threshold criteria applied by get_quality_unit_type, described as
# (quality metric, parameter, rejected side, classification stage).
#
# 'side' is the side of the threshold on which units are *rejected*: 'below' means
# units with metric < param fail the criterion, 'above' means metric > param fails.
# 'stage' is which decision the criterion contributes to: noise metrics separate
# NOISE from everything else, mua metrics separate GOOD from MUA among non-noise units.
_THRESHOLD_CRITERIA = [
    # metric, param, side, stage, integer-valued
    ("nPeaks", "maxNPeaks", "above", "noise", True),
    ("nTroughs", "maxNTroughs", "above", "noise", True),
    ("waveformDuration_peakTrough", "minWvDuration", "below", "noise", False),
    ("waveformDuration_peakTrough", "maxWvDuration", "above", "noise", False),
    ("waveformBaselineFlatness", "maxWvBaselineFraction", "above", "noise", False),
    ("scndPeakToTroughRatio", "maxScndPeakToTroughRatio_noise", "above", "noise", False),
    ("spatialDecaySlope", "minSpatialDecaySlope", "below", "noise", False),
    ("spatialDecaySlope", "minSpatialDecaySlopeExp", "below", "noise", False),
    ("spatialDecaySlope", "maxSpatialDecaySlopeExp", "above", "noise", False),
    ("percentageSpikesMissing_gaussian", "maxPercSpikesMissing", "above", "mua", False),
    ("nSpikes", "minNumSpikes", "below", "mua", True),
    ("fractionRPVs_estimatedTauR", "maxRPVviolations", "above", "mua", False),
    ("presenceRatio", "minPresenceRatio", "below", "mua", False),
    ("rawAmplitude", "minAmplitude", "below", "mua", False),
    ("signalToNoiseRatio", "minSNR", "below", "mua", False),
    ("maxDriftEstimate", "maxDrift", "above", "mua", False),
    ("isolationDistance", "isoDmin", "below", "mua", False),
    ("Lratio", "lratioMax", "above", "mua", False),
]

# Manual labels that count as passing each stage. Non-somatic units are a separate
# axis from quality, so 'Non-somatic good'/'Non-somatic MUA' join good/MUA. The
# unsplit 'Non-somatic' label conflates the two and is left out of the MUA stage.
_STAGE_LABELS = {
    "noise": {
        "pass": ["Good", "MUA", "Non-somatic", "Non-somatic good", "Non-somatic MUA"],
        "fail": ["Noise"],
    },
    "mua": {
        "pass": ["Good", "Non-somatic good"],
        "fail": ["MUA", "Non-somatic MUA"],
    },
}


def _criterion_is_active(metric, param_name, param):
    """Whether a threshold criterion is actually used given the current parameters."""
    if param_name not in param:
        return False
    if np.all(np.isnan(np.atleast_1d(np.asarray(param[param_name], dtype=float)))):
        return False

    if metric == "spatialDecaySlope":
        if not param.get("computeSpatialDecay", True):
            return False
        # Only one of the linear / exponential fit parameters is in play
        lin_fit = param.get("spDecayLinFit", False)
        if lin_fit != (param_name == "minSpatialDecaySlope"):
            return False
    if metric in ("rawAmplitude", "signalToNoiseRatio") and not param.get("extractRaw", False):
        return False
    if metric == "maxDriftEstimate" and not param.get("computeDrift", False):
        return False
    if metric in ("isolationDistance", "Lratio") and not param.get("computeDistanceMetrics", False):
        return False
    return True


def _passes(values, threshold, side):
    """Apply a threshold the same way get_quality_unit_type does."""
    with np.errstate(invalid='ignore'):
        return values >= threshold if side == "below" else values <= threshold


def _criterion_failures(full_data, param):
    """
    Which units each active criterion currently rejects.

    BombCell rejects a unit if *any* criterion fails, so tuning one threshold means
    asking what the whole classification looks like with only that threshold moved.
    That needs the other criteria's verdicts, which is what this returns.
    """
    failures = {}
    for metric, param_name, side, stage, _ in _THRESHOLD_CRITERIA:
        if metric not in full_data.columns or not _criterion_is_active(metric, param_name, param):
            continue
        values = full_data[metric].to_numpy(dtype=float)
        # A NaN metric cannot fail a threshold - BombCell handles those separately
        failures[(metric, param_name, stage)] = (
            ~_passes(values, float(param[param_name]), side) & np.isfinite(values))
    return failures


def _failures_of_others(failures, stage, this_key, n_units):
    """Units already rejected by the other criteria in this stage, whatever we do here."""
    others = np.zeros(n_units, dtype=bool)
    for key, fails in failures.items():
        if key[2] == stage and key != this_key:
            others |= fails
    return others


def _rejection_matrix(values, candidates, side, others_fail):
    """
    Unit x threshold table of which units the classifier rejects.

    Column j is the full stage verdict when this criterion's threshold is
    candidates[j]: rejected either by this criterion or by one of the others.
    """
    if side == "below":
        fails_here = values[:, None] < candidates[None, :]
    else:
        fails_here = values[:, None] > candidates[None, :]
    return others_fail[:, None] | fails_here


def _balanced_accuracy(rejected, should_reject):
    """
    Mean of sensitivity and specificity, per candidate threshold.

    Taking the mean rather than plain accuracy stops a threshold winning by rejecting
    (or keeping) everything when one class is much larger than the other.
    """
    axis = 0 if rejected.ndim > 1 else None
    sensitivity = rejected[should_reject].mean(axis=axis)
    specificity = (~rejected[~should_reject]).mean(axis=axis)
    return 0.5 * (sensitivity + specificity)


def _candidate_thresholds(values):
    """Midpoints between observed values, plus thresholds that accept / reject everything."""
    unique_values = np.unique(values)
    if unique_values.size < 2:
        return np.array([])
    midpoints = (unique_values[:-1] + unique_values[1:]) / 2
    span = unique_values[-1] - unique_values[0]
    return np.concatenate([[unique_values[0] - span], midpoints, [unique_values[-1] + span]])


def _best_threshold(rejected, should_reject, candidates, tie_break_towards):
    """Threshold maximising balanced accuracy, ties broken towards tie_break_towards."""
    scores = _balanced_accuracy(rejected, should_reject)
    tied = np.flatnonzero(scores >= scores.max() - 1e-12)
    closest = tied[np.argmin(np.abs(candidates[tied] - tie_break_towards))]
    return candidates[closest], scores[closest]


def _bootstrap_threshold(rejected, should_reject, candidates, tie_break_towards,
                         n_bootstrap, rng):
    """
    Bootstrap the optimal threshold to see how much it depends on individual units.

    Resampling is stratified so both classes survive every resample. Returns the median
    optimal threshold, a 68% interval, and an out-of-bag accuracy - each resample's
    threshold scored on the units that resample left out. Scoring a threshold on the
    units it was chosen from always flatters it, which is how a metric carrying no
    information can appear to beat the current setting.
    """
    reject_idx = np.flatnonzero(should_reject)
    keep_idx = np.flatnonzero(~should_reject)
    thresholds = np.empty(n_bootstrap)
    oob_accuracies = []

    for i in range(n_bootstrap):
        resampled = np.concatenate([
            rng.choice(reject_idx, size=reject_idx.size, replace=True),
            rng.choice(keep_idx, size=keep_idx.size, replace=True),
        ])
        threshold, _ = _best_threshold(
            rejected[resampled], should_reject[resampled], candidates, tie_break_towards)
        thresholds[i] = threshold

        held_out = np.setdiff1d(np.arange(should_reject.size), resampled)
        if held_out.size and 0 < should_reject[held_out].sum() < held_out.size:
            column = np.flatnonzero(candidates == threshold)[0]
            oob_accuracies.append(_balanced_accuracy(
                rejected[held_out, column], should_reject[held_out]))

    return (np.median(thresholds), np.percentile(thresholds, 16),
            np.percentile(thresholds, 84),
            float(np.mean(oob_accuracies)) if oob_accuracies else np.nan)


def _round_threshold(threshold, values, side, is_integer):
    """Round to a value that is readable without moving the decision boundary."""
    if is_integer:
        # Keep the same units on each side of the boundary
        return int(np.ceil(threshold)) if side == "below" else int(np.floor(threshold))

    magnitude = np.nanmax(np.abs(values))
    if magnitude == 0 or not np.isfinite(magnitude):
        return float(threshold)
    decimals = max(0, 3 - int(np.floor(np.log10(magnitude))) - 1)
    return float(np.round(threshold, decimals))


def suggest_parameter_adjustments(merged_df, quality_metrics_table, param,
                                  min_units_per_class=20, min_improvement=0.05,
                                  n_bootstrap=200, random_seed=0, return_details=False):
    """
    Suggest parameter threshold adjustments based on manually classified units

    Each BombCell threshold is tuned independently, by sweeping every possible
    threshold for its quality metric and keeping the one that best reproduces the
    manual labels. "Best" is balanced accuracy - the mean of sensitivity and
    specificity - so a threshold cannot win by simply accepting (or rejecting) every
    unit when one class is much larger than the other.

    A threshold is only suggested when all of the following hold, which keeps single
    outliers and mislabelled units from moving a parameter:

    - both classes have at least `min_units_per_class` manually labelled units
    - the new threshold beats the current one by at least `min_improvement`, judged on
      units held out of the fit rather than on the units it was chosen from
    - the current value falls outside the bootstrap interval of the optimum, i.e. the
      labelled units can actually tell the two apart

    The suggested value is the bootstrap median rather than the single best
    threshold, so it reflects where the boundary sits across resamples.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged dataframe with both manual and BombCell classifications
    quality_metrics_table : pd.DataFrame
        Full quality metrics table
    param : dict
        Current BombCell parameters
    min_units_per_class : int, optional
        Minimum manually labelled units needed on each side of a threshold, by default 20
    min_improvement : float, optional
        Minimum gain in balanced accuracy needed to suggest a change, by default 0.05

    The two defaults above are set for precision rather than coverage: telling you to
    change a threshold your labels give no reason to change is far more costly than
    staying quiet, because you would apply it to every session. Lower them if you would
    rather see marginal suggestions and judge them yourself.
    n_bootstrap : int, optional
        Bootstrap resamples used to estimate threshold stability, by default 200
    random_seed : int, optional
        Seed for the bootstrap, so suggestions are reproducible, by default 0
    return_details : bool, optional
        Also return the full per-parameter analysis, by default False

    Returns
    -------
    suggestions : list
        List of suggested parameter changes, as 'param: current → suggested' strings
    details : pd.DataFrame
        Per-parameter analysis, only returned when `return_details` is True. Includes
        parameters that were left alone and why.
    """
    empty_details = pd.DataFrame(columns=[
        'parameter', 'metric', 'stage', 'current', 'suggested', 'status',
        'n_pass', 'n_fail', 'accuracy_current', 'accuracy_suggested',
        'ci_low', 'ci_high'])

    if merged_df is None or len(merged_df) == 0:
        print("❌ No classification data available for parameter suggestions")
        return ([], empty_details) if return_details else []

    print(f"\n🔧 Parameter Threshold Suggestions")
    print(f"{'='*60}")

    # Merge with full quality metrics
    full_data = merged_df.merge(
        quality_metrics_table,
        left_on='unit_id',
        right_on='phy_clusterID',
        how='left'
    )

    print(f"Analyzing {len(full_data)} units with manual classifications...")

    rng = np.random.default_rng(random_seed)
    suggestions = []
    rows = []

    # Every criterion's current verdict, needed to judge one threshold in the context
    # of the others rather than on its own
    failures = _criterion_failures(full_data, param)
    n_units = len(full_data)
    manual = full_data['manual_type_name'].to_numpy()
    predicted_noise = _failures_of_others(failures, 'noise', None, n_units)

    for metric, param_name, side, stage, is_integer in _THRESHOLD_CRITERIA:
        key = (metric, param_name, stage)
        if key not in failures:
            continue

        current_threshold = float(param[param_name])
        labels = _STAGE_LABELS[stage]
        values_all = full_data[metric].to_numpy(dtype=float)

        # Units this stage's decision actually applies to. The MUA criteria only ever
        # see units that were not already thrown out as noise, so neither should we.
        evaluable = np.isin(manual, labels['pass'] + labels['fail']) & np.isfinite(values_all)
        if stage == 'mua':
            evaluable &= ~predicted_noise

        values = values_all[evaluable]
        should_reject = np.isin(manual[evaluable], labels['fail'])
        n_reject, n_keep = int(should_reject.sum()), int((~should_reject).sum())

        row = {'parameter': param_name, 'metric': metric, 'stage': stage,
               'current': current_threshold, 'suggested': np.nan, 'status': '',
               'n_pass': n_keep, 'n_fail': n_reject,
               'accuracy_current': np.nan, 'accuracy_suggested': np.nan,
               'ci_low': np.nan, 'ci_high': np.nan}

        if n_reject < min_units_per_class or n_keep < min_units_per_class:
            row['status'] = 'too few labels'
            rows.append(row)
            continue

        candidates = _candidate_thresholds(values)
        if candidates.size == 0:
            row['status'] = 'no variation'
            rows.append(row)
            continue

        # Score the classification the whole stage would produce, moving only this
        # threshold. A criterion that is already catching the right units cannot be
        # improved on, however well its metric happens to correlate with quality.
        others_fail = _failures_of_others(failures, stage, key, n_units)[evaluable]
        rejected = _rejection_matrix(values, candidates, side, others_fail)
        row['accuracy_current'] = _balanced_accuracy(
            others_fail | ~_passes(values, current_threshold, side), should_reject)

        median_threshold, ci_low, ci_high, accuracy_suggested = _bootstrap_threshold(
            rejected, should_reject, candidates, current_threshold, n_bootstrap, rng)
        suggested = _round_threshold(median_threshold, values, side, is_integer)

        row.update({'suggested': suggested, 'accuracy_suggested': accuracy_suggested,
                    'ci_low': ci_low, 'ci_high': ci_high})

        # A threshold outside the range of the data accepts every unit, which says the
        # criterion is only costing you units here - useful to know, but the number
        # itself is meaningless, so it is reported rather than offered to paste in
        accepts_everything = (suggested <= values.min() if side == 'below'
                              else suggested >= values.max())

        # If the current threshold falls inside the bootstrap interval, the labelled
        # units cannot tell it apart from the optimum - moving it would be churn
        if ci_low <= current_threshold <= ci_high:
            row['status'] = 'inconclusive'
        elif accuracy_suggested - row['accuracy_current'] < min_improvement:
            row['status'] = 'ok as is'
        elif accepts_everything:
            row['status'] = 'rejects only good units'
        else:
            row['status'] = 'suggested'
            suggestions.append(f"{param_name}: {param[param_name]} → {suggested}")

        rows.append(row)

    details = pd.DataFrame(rows, columns=empty_details.columns)

    if len(details) > 0:
        print(f"\n{'parameter':<32}{'current':>10}{'suggested':>11}"
              f"{'acc now':>9}{'acc new':>9}  status")
        for _, row in details.iterrows():
            suggested = '-' if np.isnan(row['suggested']) else f"{row['suggested']:g}"
            accuracy_current = ('-' if np.isnan(row['accuracy_current'])
                                else f"{row['accuracy_current']:.2f}")
            accuracy_suggested = ('-' if np.isnan(row['accuracy_suggested'])
                                  else f"{row['accuracy_suggested']:.2f}")
            marker = '👉' if row['status'] == 'suggested' else '  '
            print(f"{marker}{row['parameter']:<30}{row['current']:>10g}{suggested:>11}"
                  f"{accuracy_current:>9}{accuracy_suggested:>9}  {row['status']}")
        print("   acc = balanced accuracy of the resulting classification against your "
              "manual labels;\n   'acc new' is measured on held-out units.")

        unused = details[details['status'] == 'rejects only good units']
        if len(unused) > 0:
            print(f"\n⚠️  {', '.join(unused['parameter'])}: every unit these reject is one "
                  f"your labels kept.\n   Consider relaxing or disabling them for this data.")

        n_skipped = int((details['status'] == 'too few labels').sum())
        if n_skipped > 0:
            print(f"\n⚠️  {n_skipped} parameter(s) skipped for lack of labelled units - "
                  f"each needs {min_units_per_class} units on both sides of the threshold.")

    # Show the units where BombCell and the manual labels disagree, which is where any
    # remaining threshold problems will be
    disagreements = full_data[full_data['manual_type_name'] != full_data['Bombcell_unit_type_normalized']]

    if len(disagreements) > 0:
        print(f"\n🔍 {len(disagreements)} disagreements between BombCell and manual labels:")
        for _, row in disagreements.head(15).iterrows():
            bc_type = row.get('Bombcell_unit_type', 'Unknown')
            manual_type = row.get('manual_type_name', 'Unknown')
            print(f"  Unit {row['unit_id']}: BombCell={bc_type} → Manual={manual_type}")
        if len(disagreements) > 15:
            print(f"  ... and {len(disagreements) - 15} more")

    if suggestions:
        print(f"\n🎯 Summary of Suggested Parameter Changes:")
        print(f"{'='*60}")
        for i, suggestion in enumerate(suggestions, 1):
            print(f"{i}. {suggestion}")
        print(f"\n💡 To apply these changes, see the section below")
    else:
        print("\n✅ No parameter adjustments recommended - current thresholds match "
              "your manual labels as well as any others would.")

    return (suggestions, details) if return_details else suggestions


def plot_classification_comparison(merged_df, quality_metrics_table):
    """
    Create visualizations comparing manual vs BombCell classifications
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged dataframe with both manual and BombCell classifications
    quality_metrics_table : pd.DataFrame
        Full quality metrics table
    """
    if merged_df is None or len(merged_df) == 0:
        print("❌ No classification data available for plotting")
        return
    
    # Get full metrics for all classified units
    plot_data = merged_df.merge(
        quality_metrics_table, 
        left_on='unit_id', 
        right_on='phy_clusterID'
    )
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Manual vs BombCell Classification Analysis', fontsize=16)
    
    # Key metrics to plot
    metrics = [
        ('fractionRPVs_estimatedTauR', 'Fraction RPV Violations'),
        ('percentageSpikesMissing_gaussian', '% Spikes Missing'),
        ('presenceRatio', 'Presence Ratio'),
        ('nSpikes', 'Number of Spikes'),
        ('spatialDecaySlope', 'Spatial Decay Slope'),
        ('waveformDuration_peakTrough', 'Waveform Duration (μs)')
    ]
    
    colors = {'Good': 'green', 'MUA': 'orange', 'NOISE': 'red', 'NON-SOMA': 'blue',
              'Noise': 'red', 'Non-somatic': 'blue', 'Non-somatic good': 'blue',
              'Non-somatic MUA': 'darkmagenta'}
    
    # Determine which BombCell column to use
    bc_col = 'Bombcell_unit_type_normalized' if 'Bombcell_unit_type_normalized' in plot_data.columns else 'Bombcell_unit_type'
    
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i//3, i%3]
        
        # Plot BombCell classifications
        for bc_type in plot_data[bc_col].unique():
            if bc_type in colors:
                mask = plot_data[bc_col] == bc_type
                data_subset = plot_data[mask]
                if len(data_subset) > 0:
                    ax.scatter(data_subset[metric], [0.1]*len(data_subset), 
                              c=colors[bc_type], alpha=0.7, s=50, 
                              label=f'BC: {bc_type}', marker='o')
        
        # Plot Manual classifications  
        for manual_type in plot_data['manual_type_name'].unique():
            if manual_type in colors:
                mask = plot_data['manual_type_name'] == manual_type
                data_subset = plot_data[mask]
                if len(data_subset) > 0:
                    ax.scatter(data_subset[metric], [0.2]*len(data_subset), 
                              c=colors[manual_type], alpha=0.7, s=50, 
                              label=f'Manual: {manual_type}', marker='^')
        
        # Highlight disagreements
        disagreements = plot_data[plot_data['manual_type_name'] != plot_data[bc_col]]
        if len(disagreements) > 0:
            ax.scatter(disagreements[metric], [0.15]*len(disagreements), 
                      c='black', s=100, marker='x', alpha=0.8, 
                      label='Disagreements')
        
        ax.set_xlabel(title)
        ax.set_ylabel('Classification')
        ax.set_yticks([0.1, 0.15, 0.2])
        ax.set_yticklabels(['BombCell', 'Disagreement', 'Manual'])
        ax.grid(True, alpha=0.3)
        
        if i == 0:  # Only show legend for first subplot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.show()
    
    # Summary statistics plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Count agreements vs disagreements by type
    agreement_counts = {}
    disagreement_counts = {}
    
    for bc_type in plot_data[bc_col].unique():
        bc_mask = plot_data[bc_col] == bc_type
        bc_subset = plot_data[bc_mask]
        
        agreements = (bc_subset['manual_type_name'] == bc_subset[bc_col]).sum()
        total = len(bc_subset)
        disagreements = total - agreements
        
        agreement_counts[bc_type] = agreements
        disagreement_counts[bc_type] = disagreements
    
    # Create stacked bar chart
    types = list(agreement_counts.keys())
    agreements = [agreement_counts[t] for t in types]
    disagreements = [disagreement_counts[t] for t in types]
    
    ax.bar(types, agreements, label='Agreements', color='lightgreen', alpha=0.8)
    ax.bar(types, disagreements, bottom=agreements, label='Disagreements', color='lightcoral', alpha=0.8)
    
    ax.set_ylabel('Number of Units')
    ax.set_title('Classification Agreements vs Disagreements by Type')
    ax.legend()
    
    # Add percentage labels
    for i, (agree, disagree) in enumerate(zip(agreements, disagreements)):
        total = agree + disagree
        if total > 0:
            pct = agree / total * 100
            ax.text(i, total + 0.1, f'{pct:.1f}%', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def analyze_manual_vs_bombcell(save_path, quality_metrics_table, param, make_plots=False):
    """
    Complete analysis of manual vs BombCell classifications with suggestions
    
    Parameters
    ----------
    save_path : str or Path
        Path to BombCell output directory
    quality_metrics_table : pd.DataFrame
        BombCell quality metrics table
    param : dict
        BombCell parameters
    make_plots : bool, optional
        Whether to generate comparison plots, by default False (plots removed due to poor quality)
        
    Returns
    -------
    results : dict
        Dictionary containing analysis results, suggestions, and statistics
    """
    # Load manual classifications
    manual_df = load_manual_classifications(save_path)
    if manual_df is None:
        return None
    
    # Analyze concordance
    merged_df, confusion_df, stats = analyze_classification_concordance(manual_df, quality_metrics_table, save_path, param)
    if merged_df is None:
        return None
    
    # Get parameter suggestions
    suggestions, suggestion_details = suggest_parameter_adjustments(
        merged_df, quality_metrics_table, param, return_details=True)
    
    results = {
        'manual_df': manual_df,
        'merged_df': merged_df,
        'confusion_matrix': confusion_df,
        'concordance_stats': stats,
        'parameter_suggestions': suggestions,
        'parameter_suggestion_details': suggestion_details
    }
    
    return results


def compare_manual_vs_bombcell(save_path):
    """
    Simple function to compare manual vs BombCell classifications
    
    Parameters
    ----------
    save_path : str or Path
        Path to BombCell output directory
        
    Returns
    -------
    None
        Prints analysis results and parameter suggestions
    """
    save_path = Path(save_path)
    print(f"📊 Comparing manual vs BombCell classifications from: {save_path}")
    
    try:
        # Load BombCell results automatically
        param, quality_metrics, _ = load_bc_results(save_path)
        unit_type, unit_type_string = get_quality_unit_type(param, quality_metrics)
        quality_metrics_df = pd.DataFrame(quality_metrics)
        quality_metrics_df.insert(0, 'Bombcell_unit_type', unit_type_string)
        
        print(f"✅ Loaded BombCell results: {len(quality_metrics)} units")
        
        # Run concordance analysis
        analysis_results = analyze_manual_vs_bombcell(
            save_path=save_path,
            quality_metrics_table=quality_metrics_df, 
            param=param,
            make_plots=False
        )
        
        if analysis_results is not None:
            stats = analysis_results['concordance_stats']
            suggestions = analysis_results['parameter_suggestions']
            
            print(f"\n📈 Analysis Summary:")
            print(f"  Overall concordance: {stats['overall_concordance']:.1f}%")
            print(f"  Concordant units: {stats['concordant_units']}/{stats['total_units']}")
            
            print(f"\nConfusion Matrix (rows=BombCell, columns=Manual):")
            print(analysis_results['confusion_matrix'])
            
            # Parameter suggestions
            if len(suggestions) > 0:
                print(f"\n🔧 Suggested parameter adjustments:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f"  {i}. {suggestion}")
                    
                print(f"\n💡 To apply suggestions:")
                print(f"  1. Load your parameters: param, _, _ = bc.load_bc_results(save_path)")
                print(f"  2. Update param with suggested values (e.g., param['maxNPeaks'] = 1)")
                print(f"  3. Re-run bc.run_bombcell(ks_dir, save_path, param)")
                
                # Show example for first suggestion
                if len(suggestions) > 0:
                    first_suggestion = suggestions[0]
                    if '→' in first_suggestion and ':' in first_suggestion:
                        param_name = first_suggestion.split(':')[0].strip()
                        new_value_str = first_suggestion.split('→')[1].strip()
                        try:
                            if '.' in new_value_str:
                                new_value = float(new_value_str)
                            else:
                                new_value = int(new_value_str)
                            print(f"\n📋 Example: param['{param_name}'] = {new_value}")
                        except ValueError:
                            pass
            else:
                print(f"\n✅ No parameter adjustments recommended - parameters look good!")
                
        else:
            print("❌ No manual classifications found.")
            print("   Use the GUI to manually classify some units first:")
            print("   bc.unit_quality_gui(ks_dir, quality_metrics, unit_types, param, save_path)")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("   Make sure you have run BombCell analysis first and saved results.")