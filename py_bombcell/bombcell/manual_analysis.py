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
        print(f"📂 Found manual classifications for {len(manual_df)} units")
        return manual_df
    else:
        print("❌ No manual classifications found. Please use the GUI to manually classify some units first.")
        print(f"Expected file: {manual_file}")
        return None


def analyze_classification_concordance(manual_df, quality_metrics_table, save_path=None):
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
    
    # Create mapping of classification names to numbers and vice versa
    class_mapping = {'Noise': 0, 'Good': 1, 'MUA': 2, 'Non-somatic': 3}
    reverse_mapping = {v: k for k, v in class_mapping.items()}
    
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
    
    # Convert manual classifications to BombCell format names
    merged_df['manual_type_name'] = merged_df['manual_classification'].map(reverse_mapping)
    
    # Handle any unmapped classifications
    merged_df['manual_type_name'] = merged_df['manual_type_name'].fillna('Unknown')
    
    # Normalize case for comparison - convert BombCell types to match manual format
    bc_case_mapping = {
        'NOISE': 'Noise',
        'GOOD': 'Good', 
        'MUA': 'MUA',
        'NON-SOMA': 'Non-somatic'
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


def suggest_parameter_adjustments(merged_df, quality_metrics_table, param):
    """
    Suggest parameter threshold adjustments based on analysis of all quality metrics
    
    This function analyzes each unit's quality metrics to determine what parameter 
    thresholds would best match the user's manual classifications, then compares
    these optimal thresholds to current parameters to suggest improvements.
    
    Parameters
    ----------
    merged_df : pd.DataFrame
        Merged dataframe with both manual and BombCell classifications
    quality_metrics_table : pd.DataFrame
        Full quality metrics table
    param : dict
        Current BombCell parameters
        
    Returns
    -------
    suggestions : list
        List of suggested parameter changes
    """
    if merged_df is None or len(merged_df) == 0:
        print("❌ No classification data available for parameter suggestions")
        return []
    
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
    
    # Define key quality metrics and their parameter mappings
    quality_metrics = {
        'nSpikes': {'param': 'minNumSpikes', 'direction': 'min'},
        'presenceRatio': {'param': 'minPresenceRatio', 'direction': 'min'}, 
        'fractionRPVs_estimatedTauR': {'param': 'maxRPVviolations', 'direction': 'max'},
        'percentageSpikesMissing_gaussian': {'param': 'maxPercSpikesMissing', 'direction': 'max'},
        'nPeaks': {'param': 'maxNPeaks', 'direction': 'max'},
        'nTroughs': {'param': 'maxNTroughs', 'direction': 'max'},
        'waveformDuration_peakTrough': {'param': 'maxWvDuration', 'direction': 'max'},
        'spatialDecaySlope': {'param': 'minSpatialDecaySlope', 'direction': 'min'}
    }
    
    suggestions = []
    
    # Analyze each quality metric
    for metric, info in quality_metrics.items():
        if metric not in full_data.columns:
            continue
            
        param_name = info['param']
        if param_name not in param:
            continue
            
        current_threshold = param[param_name]
        direction = info['direction']
        
        # Get manually classified units
        good_units = full_data[full_data['manual_type_name'] == 'Good']
        noise_units = full_data[full_data['manual_type_name'] == 'Noise']
        
        if len(good_units) == 0 and len(noise_units) == 0:
            continue
            
        # Calculate optimal threshold based on manual classifications
        good_values = good_units[metric].dropna()
        noise_values = noise_units[metric].dropna()
        
        if len(good_values) == 0 or len(noise_values) == 0:
            continue
        
        # For 'min' parameters: threshold should be below the worst good unit
        # For 'max' parameters: threshold should be above the worst good unit
        if direction == 'min':
            # For minNumSpikes, minPresenceRatio: set to accommodate worst good unit
            optimal_threshold = good_values.min()
            needs_adjustment = current_threshold > optimal_threshold
            suggestion_value = optimal_threshold
        else:
            # For maxRPVviolations, maxPercSpikesMissing, maxNPeaks: set to exclude worst noise unit
            if len(noise_values) > 0:
                # Set threshold to be stricter than the best noise unit
                optimal_threshold = noise_values.min()
                needs_adjustment = current_threshold > optimal_threshold
                suggestion_value = optimal_threshold
            else:
                # No noise units, use good units as reference
                optimal_threshold = good_values.max()
                needs_adjustment = current_threshold < optimal_threshold
                suggestion_value = optimal_threshold
        
        # Special handling for nPeaks - if any noise units have >1 peak, suggest maxNPeaks=1
        if metric == 'nPeaks' and len(noise_values) > 0:
            if any(noise_values > 1) and current_threshold > 1:
                suggestions.append(f"maxNPeaks: {current_threshold} → 1")
                print(f"📊 {metric}: Noise units have >1 peak → suggest maxNPeaks=1")
                continue
        
        # Check if adjustment is needed and beneficial
        if needs_adjustment:
            # Calculate current performance
            if direction == 'min':
                good_pass_current = (good_values >= current_threshold).sum()
                good_pass_optimal = (good_values >= suggestion_value).sum()
            else:
                good_pass_current = (good_values <= current_threshold).sum()
                good_pass_optimal = (good_values <= suggestion_value).sum()
                
            # Only suggest if it improves classification of good units
            if good_pass_optimal > good_pass_current:
                # Format suggestion value appropriately
                if metric in ['nPeaks', 'nTroughs', 'nSpikes']:
                    suggestion_value = int(suggestion_value)
                else:
                    suggestion_value = round(suggestion_value, 3)
                    
                suggestions.append(f"{param_name}: {current_threshold} → {suggestion_value}")
                
                print(f"📊 {metric}: Current={current_threshold}, Optimal≈{suggestion_value}")
                print(f"   Good units passing: {good_pass_current}/{len(good_values)} → {good_pass_optimal}/{len(good_values)}")
    
    # Additional analysis for disagreements
    disagreements = full_data[full_data['manual_type_name'] != full_data['Bombcell_unit_type_normalized']]
    
    if len(disagreements) > 0:
        print(f"\n🔍 Analyzing {len(disagreements)} specific disagreements:")
        
        for _, row in disagreements.iterrows():
            unit_id = row['unit_id']
            bc_type = row.get('Bombcell_unit_type', 'Unknown')
            manual_type = row.get('manual_type_name', 'Unknown')
            
            print(f"\n📋 Unit {unit_id}: BombCell={bc_type} → Manual={manual_type}")
            
            # Show key metrics
            npeaks = row.get('nPeaks', 'N/A')
            rpv = row.get('fractionRPVs_estimatedTauR', 'N/A')
            spikes_missing = row.get('percentageSpikesMissing_gaussian', 'N/A')
            if rpv != 'N/A' and spikes_missing != 'N/A':
                print(f"   📊 nPeaks={npeaks}, RPV={rpv:.3f}, SpikesMissing={spikes_missing:.1f}%")
    
    # Summarize suggestions
    if suggestions:
        print(f"\n🎯 Summary of Suggested Parameter Changes:")
        print(f"{'='*60}")
        unique_suggestions = list(set(suggestions))
        for i, suggestion in enumerate(unique_suggestions, 1):
            print(f"{i}. {suggestion}")
        
        print(f"\n💡 To apply these changes, see the section below")
    else:
        print("\n✅ No specific parameter adjustments recommended based on current disagreements.")
    
    return suggestions


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
    
    colors = {'Good': 'green', 'MUA': 'orange', 'NOISE': 'red', 'NON-SOMA': 'blue'}
    
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
    merged_df, confusion_df, stats = analyze_classification_concordance(manual_df, quality_metrics_table, save_path)
    if merged_df is None:
        return None
    
    # Get parameter suggestions
    suggestions = suggest_parameter_adjustments(merged_df, quality_metrics_table, param)
    
    results = {
        'manual_df': manual_df,
        'merged_df': merged_df,
        'confusion_matrix': confusion_df,
        'concordance_stats': stats,
        'parameter_suggestions': suggestions
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