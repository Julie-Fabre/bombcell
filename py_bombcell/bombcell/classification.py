import numpy as np
import matplotlib.pyplot as plt

def classify_striatum_cells(ephys_properties, param):
    """
    Classify striatal cell types exactly matching MATLAB classifyCells.m
    
    MATLAB Classification rules:
    - MSN: waveformDuration_peakTrough_us > templateDuration_CP_threshold & postSpikeSuppression_ms < postSpikeSup_CP_threshold
    - FSI: waveformDuration_peakTrough_us <= templateDuration_CP_threshold & propLongISI <= propISI_CP_threshold  
    - TAN: waveformDuration_peakTrough_us > templateDuration_CP_threshold & postSpikeSuppression_ms >= postSpikeSup_CP_threshold
    - UIN: waveformDuration_peakTrough_us <= templateDuration_CP_threshold & propLongISI > propISI_CP_threshold
    
    Parameters from MATLAB:
    - templateDuration_CP_threshold = 400 (microseconds)
    - postSpikeSup_CP_threshold = 40 (milliseconds) 
    - propISI_CP_threshold = 0.1
    """
    n_units = len(ephys_properties)
    cell_types = ['Unknown'] * n_units  # Initialize as Unknown
    
    # MATLAB parameter values
    templateDuration_CP_threshold = 400  # microseconds
    postSpikeSup_CP_threshold = 40       # milliseconds  
    propISI_CP_threshold = 0.1
    
    for i in range(n_units):
        # Use exact MATLAB variable names
        waveformDuration_peakTrough_us = ephys_properties[i].get('waveformDuration_peakTrough_us', np.nan)
        postSpikeSuppression_ms = ephys_properties[i].get('postSpikeSuppression_ms', np.nan)
        propLongISI = ephys_properties[i].get('propLongISI', np.nan)
        
        # If any required property is missing, assign Unknown
        if (np.isnan(waveformDuration_peakTrough_us) or 
            np.isnan(postSpikeSuppression_ms) or 
            np.isnan(propLongISI)):
            cell_types[i] = 'Unknown'
            continue
        
        # Apply exact MATLAB classification logic
        # MSN: wide waveform AND short post-spike suppression
        if (waveformDuration_peakTrough_us > templateDuration_CP_threshold and 
            postSpikeSuppression_ms < postSpikeSup_CP_threshold):
            cell_types[i] = 'MSN'
            
        # FSI: narrow waveform AND low proportion of long ISIs
        elif (waveformDuration_peakTrough_us <= templateDuration_CP_threshold and
              propLongISI <= propISI_CP_threshold):
            cell_types[i] = 'FSI'
            
        # TAN: wide waveform AND long post-spike suppression  
        elif (waveformDuration_peakTrough_us > templateDuration_CP_threshold and
              postSpikeSuppression_ms >= postSpikeSup_CP_threshold):
            cell_types[i] = 'TAN'
            
        # UIN: narrow waveform AND high proportion of long ISIs
        elif (waveformDuration_peakTrough_us <= templateDuration_CP_threshold and
              propLongISI > propISI_CP_threshold):
            cell_types[i] = 'UIN'
            
        else:
            # Should not reach here with valid data
            cell_types[i] = 'Unknown'
    
    return cell_types


def classify_cortex_cells(ephys_properties, param):
    """
    Classify cortical cell types exactly matching MATLAB classifyCells.m
    
    MATLAB Classification rules:
    - Wide-spiking: waveformDuration_peakTrough_us > templateDuration_Ctx_threshold
    - Narrow-spiking: waveformDuration_peakTrough_us <= templateDuration_Ctx_threshold
    
    Parameters from MATLAB:
    - templateDuration_Ctx_threshold = 400 (microseconds)
    """
    n_units = len(ephys_properties)
    cell_types = ['Unknown'] * n_units  # Initialize as Unknown
    
    # MATLAB parameter values
    templateDuration_Ctx_threshold = 400  # microseconds
    
    for i in range(n_units):
        # Use exact MATLAB variable names
        waveformDuration_peakTrough_us = ephys_properties[i].get('waveformDuration_peakTrough_us', np.nan)
        
        # If required property is missing, assign Unknown
        if np.isnan(waveformDuration_peakTrough_us):
            cell_types[i] = 'Unknown'
            continue
        
        # Apply exact MATLAB classification logic
        if waveformDuration_peakTrough_us > templateDuration_Ctx_threshold:
            cell_types[i] = 'Wide-spiking'
        else:
            cell_types[i] = 'Narrow-spiking'
    
    return cell_types


def plot_striatum_classification(ephys_properties, cell_types):
    """
    Plot striatum classification: waveform duration x PSS x prop long ISI (3D)
    """
    # Extract properties with more flexible data handling
    wf_durations = []
    pss_values = []
    prop_long_isis = []
    colors = []
    labels = []
    
    color_map = {'MSN': 'blue', 'FSI': 'red', 'TAN': 'green', 'UIN': 'orange', 'Unknown': 'gray'}
    
    for i, props in enumerate(ephys_properties):
        # Use exact MATLAB variable names
        waveformDuration_peakTrough_us = props.get('waveformDuration_peakTrough_us', np.nan)
        postSpikeSuppression_ms = props.get('postSpikeSuppression_ms', np.nan)
        propLongISI = props.get('propLongISI', np.nan)
        
        # Only include points with all valid properties
        if (not np.isnan(waveformDuration_peakTrough_us) and 
            not np.isnan(postSpikeSuppression_ms) and 
            not np.isnan(propLongISI)):
            wf_durations.append(waveformDuration_peakTrough_us)
            pss_values.append(postSpikeSuppression_ms)
            prop_long_isis.append(propLongISI)
            colors.append(color_map.get(cell_types[i], 'gray'))
            labels.append(cell_types[i])
    
    if len(wf_durations) == 0:
        print("No valid data points for striatum classification plot")
        return
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot by cell type using stem plot for better visibility
    for cell_type, color in color_map.items():
        mask = np.array(labels) == cell_type
        if np.any(mask):
            wf_subset = np.array(wf_durations)[mask]
            pss_subset = np.array(pss_values)[mask]
            prop_subset = np.array(prop_long_isis)[mask]
            
            # Create stem plot from z=0 to each point
            for j in range(len(wf_subset)):
                ax.plot([wf_subset[j], wf_subset[j]], 
                       [pss_subset[j], pss_subset[j]], 
                       [0, prop_subset[j]], 
                       color=color, alpha=0.7, linewidth=2)
            
            # Add scatter points at the top
            ax.scatter(wf_subset, pss_subset, prop_subset, 
                      c=color, label=f'{cell_type} (n={np.sum(mask)})', 
                      s=80, alpha=1.0, edgecolors='black', linewidth=1)
    
    ax.set_xlabel('Waveform Duration (μs)', fontsize=11)
    ax.set_ylabel('Post-Spike Suppression (ms)', fontsize=11)
    ax.set_zlabel('Proportion Long ISI', fontsize=11)
    ax.set_title('Striatum Cell Type Classification', fontsize=12, pad=20)
    
    # Add classification boundaries as reference
    ax.axvline(x=400, color='black', linestyle='--', alpha=0.5, label='400μs threshold')
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Print data summary
    print(f"Plotted {len(wf_durations)} units:")
    print(f"  Waveform duration range: {np.min(wf_durations):.0f}-{np.max(wf_durations):.0f} μs")
    print(f"  Post-spike suppression range: {np.nanmin(pss_values):.1f}-{np.nanmax(pss_values):.1f} ms") 
    print(f"  Prop long ISI range: {np.nanmin(prop_long_isis):.3f}-{np.nanmax(prop_long_isis):.3f}")
    


def plot_cortex_classification(ephys_properties, cell_types):
    """
    Plot cortex classification: waveform duration x firing rate (2D)
    """
    # Extract properties with flexible data handling
    wf_durations = []
    firing_rates = []
    labels = []
    
    color_map = {'Narrow-spiking': 'red', 'Wide-spiking': 'blue', 'Unknown': 'gray'}
    
    for i, props in enumerate(ephys_properties):
        # Use exact MATLAB variable names
        waveformDuration_peakTrough_us = props.get('waveformDuration_peakTrough_us', np.nan)
        firing_rate = props.get('firing_rate_mean', np.nan)
        
        # Only include points with valid waveform duration (firing rate can be missing for cortex)
        if not np.isnan(waveformDuration_peakTrough_us):
            wf_durations.append(waveformDuration_peakTrough_us)
            firing_rates.append(firing_rate if not np.isnan(firing_rate) else np.nan)
            labels.append(cell_types[i])
    
    if len(wf_durations) == 0:
        print("No valid data points for cortex classification plot")
        return
    
    # Create 2D plot
    plt.figure(figsize=(10, 7))
    
    # Plot by cell type for clear visualization
    for cell_type, color in color_map.items():
        mask = np.array(labels) == cell_type
        if np.any(mask):
            wf_subset = np.array(wf_durations)[mask]
            fr_subset = np.array(firing_rates)[mask]
            
            plt.scatter(wf_subset, fr_subset, 
                       c=color, label=f'{cell_type} (n={np.sum(mask)})', 
                       s=60, alpha=0.8, edgecolors='black', linewidth=0.5)
    
    # Add classification boundary
    plt.axvline(x=400, color='black', linestyle='--', alpha=0.7, label='400μs threshold')
    
    plt.xlabel('Waveform Duration (μs)', fontsize=11)
    plt.ylabel('Firing Rate (Hz)', fontsize=11)
    plt.title('Cortex Cell Type Classification\n(2D: Duration × Firing Rate)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Print data summary
    print(f"Plotted {len(wf_durations)} units:")
    print(f"  Waveform duration range: {np.min(wf_durations):.0f}-{np.max(wf_durations):.0f} μs")
    print(f"  Firing rate range: {np.nanmin(firing_rates):.1f}-{np.nanmax(firing_rates):.1f} Hz")


def debug_ephys_properties(ephys_properties):
    """Debug function to check what properties are available"""
    if len(ephys_properties) > 0:
        print("Available properties in first unit:")
        for key, value in ephys_properties[0].items():
            print(f"  {key}: {value}")
        print(f"\nTotal units: {len(ephys_properties)}")
        
        # Check key properties
        for i in range(min(3, len(ephys_properties))):
            waveform = ephys_properties[i].get('waveformDuration_peakTrough_us', 'MISSING')
            pss = ephys_properties[i].get('postSpikeSuppression_ms', 'MISSING')
            prop_isi = ephys_properties[i].get('propLongISI', 'MISSING')
            print(f"Unit {i}: waveform={waveform}, pss={pss}, propISI={prop_isi}")


def classify_and_plot_brain_region(ephys_properties, param, brain_region):
    """
    Main function for brain region classification and automatic plot generation
    
    Parameters:
    -----------
    ephys_properties : list
        List of ephys property dictionaries for each unit
    param : dict
        Parameter dictionary with classification thresholds
    brain_region : str
        Brain region to classify ('cortex' or 'striatum')
        
    Returns:
    --------
    list or None
        Cell type classifications, or None if region not supported
    """
    brain_region = brain_region.lower().strip()
    
    # Basic validation
    if len(ephys_properties) == 0:
        print("No ephys properties provided")
        return None
    
    if brain_region == 'striatum':
        cell_types = classify_striatum_cells(ephys_properties, param)
        plot_striatum_classification(ephys_properties, cell_types)
        
        # Summary with all possible cell types
        all_cell_types = ['MSN', 'FSI', 'TAN', 'UIN', 'Unknown']
        
    elif brain_region == 'cortex':
        cell_types = classify_cortex_cells(ephys_properties, param)
        plot_cortex_classification(ephys_properties, cell_types)
        
        # Summary with all possible cell types
        all_cell_types = ['Wide-spiking', 'Narrow-spiking', 'Unknown']
        
    else:
        print(f"We cannot do that yet! '{brain_region}' is not supported.")
        print("Currently supported regions: 'cortex' and 'striatum'")
        return None
    
    # Print classification summary
    import pandas as pd
    cell_type_counts = pd.Series(cell_types).value_counts()
    print(f"\n{brain_region.capitalize()} cell type distribution:")
    
    for cell_type in all_cell_types:
        count = cell_type_counts.get(cell_type, 0)
        percentage = (count / len(cell_types)) * 100
        print(f"  {cell_type}: {count} ({percentage:.1f}%)")
    
    return cell_types