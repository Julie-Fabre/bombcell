import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


def simple_unit_browser(quality_metrics, unit_types=None, ephys_properties=None, start_unit=0):
    """
    Simple unit browser that works reliably in Jupyter notebooks
    
    Parameters
    ----------
    quality_metrics : dict
        Quality metrics from bombcell
    unit_types : array, optional
        Unit type classifications
    ephys_properties : list, optional
        Ephys properties if available
    start_unit : int, optional
        Unit index to start with
    """
    
    # Get unit information
    n_units = len(quality_metrics['phy_clusterID'])
    unit_ids = quality_metrics['phy_clusterID']
    
    def show_unit(unit_idx):
        """Display information for a single unit"""
        if unit_idx < 0 or unit_idx >= n_units:
            print(f"Unit index {unit_idx} out of range (0-{n_units-1})")
            return
            
        unit_id = unit_ids[unit_idx]
        
        # Get unit type
        unit_type_str = "UNKNOWN"
        if unit_types is not None and unit_idx < len(unit_types):
            type_map = {0: 'NOISE', 1: 'GOOD', 2: 'MUA', 3: 'NON-SOMA'}
            unit_type_str = type_map.get(unit_types[unit_idx], 'UNKNOWN')
        
        print("=" * 60)
        print(f"UNIT {unit_id} ({unit_type_str}) - Index {unit_idx+1}/{n_units}")
        print("=" * 60)
        
        # Quality metrics
        print("\nQUALITY METRICS:")
        print("-" * 20)
        
        key_metrics = [
            ('nSpikes', 'Number of Spikes'),
            ('presenceRatio', 'Presence Ratio'),
            ('fractionRPVs_estimatedTauR', 'Fraction RPVs'),
            ('waveformDuration_peakTrough', 'Waveform Duration (μs)'),
            ('spatialDecaySlope', 'Spatial Decay Slope'),
            ('percentageSpikesMissing_gaussian', '% Spikes Missing'),
            ('maxDriftEstimate', 'Max Drift Estimate')
        ]
        
        for key, label in key_metrics:
            if key in quality_metrics:
                value = quality_metrics[key][unit_idx] if unit_idx < len(quality_metrics[key]) else 'N/A'
                if isinstance(value, float):
                    if abs(value) > 1000 or (abs(value) < 0.001 and value != 0):
                        print(f"  {label}: {value:.2e}")
                    else:
                        print(f"  {label}: {value:.3f}")
                else:
                    print(f"  {label}: {value}")
        
        # Ephys properties if available
        if ephys_properties and unit_idx < len(ephys_properties):
            ephys = ephys_properties[unit_idx]
            print("\nEPHYS PROPERTIES:")
            print("-" * 20)
            
            ephys_metrics = [
                ('firing_rate_mean', 'Firing Rate (Hz)'),
                ('isi_cv', 'ISI CV'),
                ('postSpikeSuppression_ms', 'Post-spike Suppression (ms)'),
                ('propLongISI', 'Prop Long ISI'),
                ('waveformDuration_peakTrough_us', 'Waveform Duration (μs)')
            ]
            
            for key, label in ephys_metrics:
                if key in ephys:
                    value = ephys[key]
                    if isinstance(value, float) and not np.isnan(value):
                        print(f"  {label}: {value:.3f}")
                    elif not np.isnan(value):
                        print(f"  {label}: {value}")
                    else:
                        print(f"  {label}: N/A")
        
        # Navigation help
        print("\n" + "=" * 60)
        print("NAVIGATION:")
        print("  show_unit(index)     - Go to specific unit")
        print("  next_good()          - Next good unit")
        print("  next_mua()           - Next MUA unit") 
        print("  next_noise()         - Next noise unit")
        print("  summary()            - Show summary statistics")
        print("=" * 60)
    
    def next_good():
        """Go to next good unit"""
        if unit_types is not None:
            good_indices = np.where(unit_types == 1)[0]
            current_idx = getattr(next_good, 'current_idx', start_unit)
            next_good_idx = good_indices[good_indices > current_idx]
            if len(next_good_idx) > 0:
                next_good.current_idx = next_good_idx[0]
                show_unit(next_good.current_idx)
            else:
                print("No more good units found")
        else:
            print("Unit types not available")
    
    def next_mua():
        """Go to next MUA unit"""
        if unit_types is not None:
            mua_indices = np.where(unit_types == 2)[0]
            current_idx = getattr(next_mua, 'current_idx', start_unit)
            next_mua_idx = mua_indices[mua_indices > current_idx]
            if len(next_mua_idx) > 0:
                next_mua.current_idx = next_mua_idx[0]
                show_unit(next_mua.current_idx)
            else:
                print("No more MUA units found")
        else:
            print("Unit types not available")
    
    def next_noise():
        """Go to next noise unit"""
        if unit_types is not None:
            noise_indices = np.where(unit_types == 0)[0]
            current_idx = getattr(next_noise, 'current_idx', start_unit)
            next_noise_idx = noise_indices[noise_indices > current_idx]
            if len(next_noise_idx) > 0:
                next_noise.current_idx = next_noise_idx[0]
                show_unit(next_noise.current_idx)
            else:
                print("No more noise units found")
        else:
            print("Unit types not available")
    
    def summary():
        """Show summary statistics"""
        print("=" * 60)
        print("DATASET SUMMARY")
        print("=" * 60)
        print(f"Total units: {n_units}")
        
        if unit_types is not None:
            type_counts = pd.Series(unit_types).value_counts()
            type_map = {0: 'NOISE', 1: 'GOOD', 2: 'MUA', 3: 'NON-SOMA'}
            print("\nUnit type distribution:")
            for type_val, count in type_counts.items():
                type_name = type_map.get(type_val, 'UNKNOWN')
                percentage = (count / n_units) * 100
                print(f"  {type_name}: {count} ({percentage:.1f}%)")
        
        # Quality metrics summary
        if 'nSpikes' in quality_metrics:
            spikes = quality_metrics['nSpikes']
            spikes_clean = [s for s in spikes if not np.isnan(s)]
            if spikes_clean:
                print(f"\nSpike count statistics:")
                print(f"  Mean: {np.mean(spikes_clean):.0f}")
                print(f"  Median: {np.median(spikes_clean):.0f}")
                print(f"  Range: {np.min(spikes_clean):.0f} - {np.max(spikes_clean):.0f}")
        
        if ephys_properties:
            firing_rates = [ep.get('firing_rate_mean', np.nan) for ep in ephys_properties]
            firing_rates_clean = [fr for fr in firing_rates if not np.isnan(fr)]
            if firing_rates_clean:
                print(f"\nFiring rate statistics (Hz):")
                print(f"  Mean: {np.mean(firing_rates_clean):.2f}")
                print(f"  Median: {np.median(firing_rates_clean):.2f}")
                print(f"  Range: {np.min(firing_rates_clean):.2f} - {np.max(firing_rates_clean):.2f}")
    
    # Make functions available globally for easy access
    import builtins
    builtins.show_unit = show_unit
    builtins.next_good = next_good
    builtins.next_mua = next_mua
    builtins.next_noise = next_noise
    builtins.summary = summary
    
    # Show initial unit
    print("Simple Unit Browser Loaded!")
    print("Functions available: show_unit(), next_good(), next_mua(), next_noise(), summary()")
    print()
    show_unit(start_unit)
    
    return {
        'show_unit': show_unit,
        'next_good': next_good,
        'next_mua': next_mua,
        'next_noise': next_noise,
        'summary': summary
    }


def plot_unit_waveforms(quality_metrics, template_waveforms, unit_idx, unit_types=None):
    """
    Plot waveforms for a specific unit
    
    Parameters
    ----------
    quality_metrics : dict
        Quality metrics from bombcell
    template_waveforms : array
        Template waveforms
    unit_idx : int
        Unit index to plot
    unit_types : array, optional
        Unit type classifications
    """
    if unit_idx >= len(quality_metrics['phy_clusterID']):
        print(f"Unit index {unit_idx} out of range")
        return
    
    unit_id = quality_metrics['phy_clusterID'][unit_idx]
    
    # Get unit type
    unit_type_str = "UNKNOWN"
    if unit_types is not None and unit_idx < len(unit_types):
        type_map = {0: 'NOISE', 1: 'GOOD', 2: 'MUA', 3: 'NON-SOMA'}
        unit_type_str = type_map.get(unit_types[unit_idx], 'UNKNOWN')
    
    # Plot template waveform
    if unit_idx < len(template_waveforms):
        template = template_waveforms[unit_idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot all channels
        time_axis = np.arange(template.shape[0]) / 30.0  # Convert to ms
        
        # Plot peak channel waveform
        peak_chan = np.argmin(np.min(template, axis=0))
        waveform = template[:, peak_chan]
        
        axes[0].plot(time_axis, waveform, 'k-', linewidth=2)
        axes[0].set_title(f'Unit {unit_id} ({unit_type_str}) - Peak Channel')
        axes[0].set_xlabel('Time (ms)')
        axes[0].set_ylabel('Amplitude (μV)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot all channels
        for ch in range(min(10, template.shape[1])):  # Limit to 10 channels
            axes[1].plot(time_axis, template[:, ch] + ch * 100, alpha=0.7)
        
        axes[1].set_title(f'Unit {unit_id} - Multiple Channels')
        axes[1].set_xlabel('Time (ms)')
        axes[1].set_ylabel('Channel + Offset')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"No template waveform available for unit {unit_idx}")


# Add to module exports
__all__ = ['simple_unit_browser', 'plot_unit_waveforms']