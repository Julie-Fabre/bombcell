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


def analyze_classification_concordance(manual_df, quality_metrics_table):
    """
    Analyze concordance between manual and BombCell classifications
    
    Parameters
    ----------
    manual_df : pd.DataFrame
        Manual classifications with columns ['unit_id', 'manual_classification']
    quality_metrics_table : pd.DataFrame
        BombCell quality metrics table with 'phy_clusterID' and 'Bombcell_unit_type' columns
        
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
    
    # Merge manual and BombCell classifications
    merged_df = manual_df.merge(
        quality_metrics_table[['phy_clusterID', 'Bombcell_unit_type']], 
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
    
    # Calculate overall concordance (correct classifications)
    total_units = len(merged_df)
    concordant_units = (merged_df['manual_type_name'] == merged_df['Bombcell_unit_type']).sum()
    overall_concordance = concordant_units / total_units * 100
    
    print(f"📊 Classification Concordance Analysis")
    print(f"{'='*50}")
    print(f"Total classified units: {total_units}")
    print(f"Concordant classifications: {concordant_units}")
    print(f"Overall concordance: {overall_concordance:.1f}%")
    print(f"{'='*50}")
    
    # Create confusion matrix (BombCell as rows, Manual as columns)
    confusion_df = pd.crosstab(
        merged_df['Bombcell_unit_type'], 
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
    
    # Find disagreements
    disagreements = merged_df[merged_df['manual_type_name'] != merged_df['Bombcell_unit_type']].copy()
    
    if len(disagreements) == 0:
        print("✅ Perfect concordance! No parameter adjustments needed.")
        return []
    
    print(f"Analyzing {len(disagreements)} disagreements out of {len(merged_df)} units...")
    
    # Get full quality metrics for disagreement units
    # Check if disagreements already has quality metrics columns
    if 'nSpikes' in disagreements.columns:
        # Disagreements already has quality metrics from merged_df
        disagreement_metrics = disagreements.copy()
    else:
        # Need to merge with quality metrics table
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
        
        # Get BombCell classification (should be in disagreements from merged_df)
        bc_type = row.get('Bombcell_unit_type', 'Unknown')
        manual_type = row.get('manual_type_name', 'Unknown')
        
        print(f"\n📋 Unit {unit_id}: BombCell={bc_type} → Manual={manual_type}")
        
        # Generate suggestions based on disagreement type
        if bc_type == 'NOISE' and manual_type in ['Good', 'MUA']:
            # BombCell too conservative (calling good units noise)
            print("   💡 BombCell is too conservative - good unit classified as noise")
            
            # Check which metrics failed and suggest relaxing thresholds
            if row['nSpikes'] < param['minNumSpikes']:
                new_val = int(row['nSpikes'])
                print(f"      • minNumSpikes: {row['nSpikes']:.0f} < {param['minNumSpikes']} → Consider lowering to {new_val}")
                suggestions.append(f"minNumSpikes: {param['minNumSpikes']} → {new_val}")
            
            if row['presenceRatio'] < param['minPresenceRatio']:
                new_val = round(row['presenceRatio'], 3)
                print(f"      • minPresenceRatio: {row['presenceRatio']:.3f} < {param['minPresenceRatio']} → Consider lowering to {new_val}")
                suggestions.append(f"minPresenceRatio: {param['minPresenceRatio']} → {new_val}")
            
            if row['fractionRPVs_estimatedTauR'] > param['maxRPVviolations']:
                new_val = round(row['fractionRPVs_estimatedTauR'], 3)
                print(f"      • maxRPVviolations: {row['fractionRPVs_estimatedTauR']:.3f} > {param['maxRPVviolations']} → Consider raising to {new_val}")
                suggestions.append(f"maxRPVviolations: {param['maxRPVviolations']} → {new_val}")
            
            if row['percentageSpikesMissing_gaussian'] > param['maxPercSpikesMissing']:
                new_val = round(row['percentageSpikesMissing_gaussian'], 1)
                print(f"      • maxPercSpikesMissing: {row['percentageSpikesMissing_gaussian']:.1f}% > {param['maxPercSpikesMissing']}% → Consider raising to {new_val}%")
                suggestions.append(f"maxPercSpikesMissing: {param['maxPercSpikesMissing']} → {new_val}")
        
        elif bc_type in ['GOOD', 'MUA'] and manual_type == 'Noise':
            # BombCell too permissive (keeping bad units)
            print("   ⚠️  BombCell is too permissive - noise unit classified as good/MUA")
            
            # Suggest tightening thresholds
            if row['fractionRPVs_estimatedTauR'] > 0.05:
                new_val = round(min(param['maxRPVviolations'], row['fractionRPVs_estimatedTauR'] * 0.8), 3)
                print(f"      • maxRPVviolations: {param['maxRPVviolations']} → {new_val} (tighten)")
                suggestions.append(f"maxRPVviolations: {param['maxRPVviolations']} → {new_val}")
            
            if row['percentageSpikesMissing_gaussian'] > 30:
                new_val = round(max(param['maxPercSpikesMissing'], row['percentageSpikesMissing_gaussian'] * 0.8), 1)
                print(f"      • maxPercSpikesMissing: {param['maxPercSpikesMissing']} → {new_val}% (tighten)")
                suggestions.append(f"maxPercSpikesMissing: {param['maxPercSpikesMissing']} → {new_val}")
        
        elif bc_type == 'MUA' and manual_type == 'Good':
            # MUA→Good: could relax thresholds slightly
            print("   📈 Unit could be upgraded from MUA to Good")
            
            if row['fractionRPVs_estimatedTauR'] < param['maxRPVviolations'] * 1.2:
                print(f"      • RPV rate acceptable ({row['fractionRPVs_estimatedTauR']:.3f}), other metrics may need adjustment")
        
        elif bc_type == 'GOOD' and manual_type == 'MUA':
            # Good→MUA: tighten thresholds
            print("   📉 Unit should be downgraded from Good to MUA")
            
            if row['fractionRPVs_estimatedTauR'] > param['maxRPVviolations'] * 0.7:
                new_val = round(row['fractionRPVs_estimatedTauR'] * 0.9, 3)
                print(f"      • maxRPVviolations: {param['maxRPVviolations']} → {new_val} (tighten)")
                suggestions.append(f"maxRPVviolations: {param['maxRPVviolations']} → {new_val}")
    
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
    
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i//3, i%3]
        
        # Plot BombCell classifications
        for bc_type in plot_data['Bombcell_unit_type'].unique():
            if bc_type in colors:
                mask = plot_data['Bombcell_unit_type'] == bc_type
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
        disagreements = plot_data[plot_data['manual_type_name'] != plot_data['Bombcell_unit_type']]
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
    
    for bc_type in plot_data['Bombcell_unit_type'].unique():
        bc_mask = plot_data['Bombcell_unit_type'] == bc_type
        bc_subset = plot_data[bc_mask]
        
        agreements = (bc_subset['manual_type_name'] == bc_subset['Bombcell_unit_type']).sum()
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


def analyze_manual_vs_bombcell(save_path, quality_metrics_table, param, make_plots=True):
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
        Whether to generate comparison plots, by default True
        
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
    merged_df, confusion_df, stats = analyze_classification_concordance(manual_df, quality_metrics_table)
    if merged_df is None:
        return None
    
    # Get parameter suggestions
    suggestions = suggest_parameter_adjustments(merged_df, quality_metrics_table, param)
    
    # Create plots if requested
    if make_plots:
        plot_classification_comparison(merged_df, quality_metrics_table)
    
    results = {
        'manual_df': manual_df,
        'merged_df': merged_df,
        'confusion_matrix': confusion_df,
        'concordance_stats': stats,
        'parameter_suggestions': suggestions
    }
    
    return results