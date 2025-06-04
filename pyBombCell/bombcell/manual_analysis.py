"""
Manual classification analysis functions for BombCell

Functions for analyzing concordance between manual and automatic BombCell classifications,
and suggesting parameter threshold adjustments based on disagreements.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


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
    Suggest parameter threshold adjustments based on disagreements between manual and BombCell classifications
    
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
    
    # Find disagreements (using normalized comparison)
    disagreements = merged_df[merged_df['manual_type_name'] != merged_df['Bombcell_unit_type_normalized']].copy()
    
    if len(disagreements) == 0:
        print("✅ Perfect concordance! No parameter adjustments needed.")
        return []
    
    print(f"Analyzing {len(disagreements)} disagreements out of {len(merged_df)} units...")
    
    # Get full quality metrics for disagreement units
    # Always merge with quality metrics table to ensure we have all columns
    disagreement_metrics = disagreements.merge(
        quality_metrics_table, 
        left_on='unit_id', 
        right_on='phy_clusterID',
        how='left'
    )
    
    suggestions = []
    
    # Analyze specific disagreement patterns
    for _, row in disagreement_metrics.iterrows():
        unit_id = row['unit_id']
        
        # Get BombCell classification (use original uppercase format for parameter logic)
        bc_type = row.get('Bombcell_unit_type', 'Unknown')
        manual_type = row.get('manual_type_name', 'Unknown')
        
        print(f"\n📋 Unit {unit_id}: BombCell={bc_type} → Manual={manual_type}")
        
        # Show relevant metrics for this unit
        npeaks = row.get('nPeaks', 'N/A')
        ntroughs = row.get('nTroughs', 'N/A')
        rpv = row.get('fractionRPVs_estimatedTauR', 'N/A')
        spikes_missing = row.get('percentageSpikesMissing_gaussian', 'N/A')
        print(f"   📊 Metrics: nPeaks={npeaks}, nTroughs={ntroughs}, RPV={rpv:.3f}, SpikesMissing={spikes_missing:.1f}%")
        
        # Generate suggestions based on disagreement type
        if bc_type == 'NOISE' and manual_type in ['Good', 'MUA']:
            # BombCell too conservative (calling good units noise)
            print("   💡 BombCell is too conservative - good unit classified as noise")
            
            # Check which metrics failed and suggest relaxing thresholds
            if 'nSpikes' in row and row['nSpikes'] < param.get('minNumSpikes', 0):
                new_val = int(row['nSpikes'])
                print(f"      • minNumSpikes: {row['nSpikes']:.0f} < {param['minNumSpikes']} → Consider lowering to {new_val}")
                suggestions.append(f"minNumSpikes: {param['minNumSpikes']} → {new_val}")
            
            if 'presenceRatio' in row and row['presenceRatio'] < param.get('minPresenceRatio', 0):
                new_val = round(row['presenceRatio'], 3)
                print(f"      • minPresenceRatio: {row['presenceRatio']:.3f} < {param['minPresenceRatio']} → Consider lowering to {new_val}")
                suggestions.append(f"minPresenceRatio: {param['minPresenceRatio']} → {new_val}")
            
            if 'fractionRPVs_estimatedTauR' in row and row['fractionRPVs_estimatedTauR'] > param.get('maxRPVviolations', 1):
                new_val = round(row['fractionRPVs_estimatedTauR'], 3)
                print(f"      • maxRPVviolations: {row['fractionRPVs_estimatedTauR']:.3f} > {param['maxRPVviolations']} → Consider raising to {new_val}")
                suggestions.append(f"maxRPVviolations: {param['maxRPVviolations']} → {new_val}")
            
            if 'percentageSpikesMissing_gaussian' in row and row['percentageSpikesMissing_gaussian'] > param.get('maxPercSpikesMissing', 100):
                new_val = round(row['percentageSpikesMissing_gaussian'], 1)
                print(f"      • maxPercSpikesMissing: {row['percentageSpikesMissing_gaussian']:.1f}% > {param['maxPercSpikesMissing']}% → Consider raising to {new_val}%")
                suggestions.append(f"maxPercSpikesMissing: {param['maxPercSpikesMissing']} → {new_val}")
        
        elif bc_type in ['GOOD', 'MUA'] and manual_type == 'Noise':
            # BombCell too permissive (keeping bad units)
            print("   ⚠️  BombCell is too permissive - noise unit classified as good/MUA")
            
            # Check waveform shape metrics
            if 'nPeaks' in row and row['nPeaks'] > 1:
                current_max = param.get('maxNPeaks', 999)
                if current_max > 1:
                    print(f"      • maxNPeaks: current={current_max} → Consider setting to 1 (unit has {row['nPeaks']} peaks)")
                    suggestions.append(f"maxNPeaks: {current_max} → 1")
            
            # Suggest tightening thresholds
            if 'fractionRPVs_estimatedTauR' in row and row['fractionRPVs_estimatedTauR'] > 0.05:
                new_val = round(min(param.get('maxRPVviolations', 1), row['fractionRPVs_estimatedTauR'] * 0.8), 3)
                print(f"      • maxRPVviolations: {param.get('maxRPVviolations', 1)} → {new_val} (tighten)")
                suggestions.append(f"maxRPVviolations: {param.get('maxRPVviolations', 1)} → {new_val}")
            
            if 'percentageSpikesMissing_gaussian' in row and row['percentageSpikesMissing_gaussian'] > 30:
                new_val = round(max(param.get('maxPercSpikesMissing', 0), row['percentageSpikesMissing_gaussian'] * 0.8), 1)
                print(f"      • maxPercSpikesMissing: {param.get('maxPercSpikesMissing', 100)} → {new_val}% (tighten)")
                suggestions.append(f"maxPercSpikesMissing: {param.get('maxPercSpikesMissing', 100)} → {new_val}")
        
        elif bc_type == 'MUA' and manual_type == 'Good':
            # MUA→Good: could relax thresholds slightly
            print("   📈 Unit could be upgraded from MUA to Good")
            
            # Check if waveform shape is good (single peak)
            if 'nPeaks' in row and row['nPeaks'] == 1:
                current_max = param.get('maxNPeaks', 999)
                if current_max < 1:
                    print(f"      • maxNPeaks: current={current_max} → Consider setting to 1 (unit has {row['nPeaks']} peak)")
                    suggestions.append(f"maxNPeaks: {current_max} → 1")
                else:
                    print(f"      • Waveform shape looks good (nPeaks={row['nPeaks']}), check other metrics")
            
            if 'fractionRPVs_estimatedTauR' in row and row['fractionRPVs_estimatedTauR'] < param.get('maxRPVviolations', 1) * 1.2:
                print(f"      • RPV rate acceptable ({row['fractionRPVs_estimatedTauR']:.3f}), other metrics may need adjustment")
        
        elif bc_type == 'GOOD' and manual_type == 'MUA':
            # Good→MUA: tighten thresholds
            print("   📉 Unit should be downgraded from Good to MUA")
            
            if 'fractionRPVs_estimatedTauR' in row and row['fractionRPVs_estimatedTauR'] > param.get('maxRPVviolations', 1) * 0.7:
                new_val = round(row['fractionRPVs_estimatedTauR'] * 0.9, 3)
                print(f"      • maxRPVviolations: {param.get('maxRPVviolations', 1)} → {new_val} (tighten)")
                suggestions.append(f"maxRPVviolations: {param.get('maxRPVviolations', 1)} → {new_val}")
    
    # Summarize suggestions
    if suggestions:
        print(f"\n🎯 Summary of Suggested Parameter Changes:")
        print(f"{'='*60}")
        unique_suggestions = list(set(suggestions))
        for i, suggestion in enumerate(unique_suggestions, 1):
            print(f"{i}. {suggestion}")
        
        print(f"\n💡 To apply these changes:")
        print("   1. Update your parameter dictionary with the suggested values")
        print("   2. Re-run BombCell with the updated parameters")
        print("   3. Compare the new classifications with your manual labels")
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