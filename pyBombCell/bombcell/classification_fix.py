import numpy as np

def classify_striatum_cells_fixed(ephys_properties, param):
    """
    Classify striatal cell types (MSN, FSI, TAN, UIN) - FIXED VERSION
    """
    n_units = len(ephys_properties)
    cell_types = ['Unknown'] * n_units
    
    for i in range(n_units):
        wf_duration = ephys_properties[i].get('waveform_duration_peak_trough', np.nan)
        firing_rate = ephys_properties[i].get('firing_rate_mean', np.nan)
        cv = ephys_properties[i].get('isi_cv', np.nan)
        
        # Skip if missing critical properties
        if np.isnan(wf_duration) or np.isnan(firing_rate):
            continue
        
        # Convert to seconds for comparison
        wf_duration_sec = wf_duration / 1e6 if not np.isnan(wf_duration) else np.nan
        
        # FSI: narrow waveform + high firing rate
        if (not np.isnan(wf_duration_sec) and 
            wf_duration_sec < param['fsi_waveform_duration_max'] and 
            firing_rate > param['fsi_firing_rate_min']):
            cell_types[i] = 'FSI'
        
        # TAN: regular firing (low CV) + moderate firing rate
        elif (not np.isnan(cv) and cv < param['tan_cv_max'] and 
              firing_rate > param['tan_firing_rate_min']):
            cell_types[i] = 'TAN'
        
        # MSN: wide waveform + low firing rate
        elif (not np.isnan(wf_duration_sec) and 
              wf_duration_sec > param['msn_waveform_duration_max']):
            cell_types[i] = 'MSN'
        
        # Unidentified interneuron
        else:
            cell_types[i] = 'UIN'
    
    return cell_types


def classify_cortex_cells_fixed(ephys_properties, param):
    """
    Classify cortical cell types (narrow-spiking, wide-spiking) - FIXED VERSION
    """
    n_units = len(ephys_properties)
    cell_types = ['Unknown'] * n_units
    
    for i in range(n_units):
        wf_duration = ephys_properties[i].get('waveform_duration_peak_trough', np.nan)
        
        if np.isnan(wf_duration):
            continue
        
        # Convert to seconds
        wf_duration_sec = wf_duration / 1e6
        
        # Narrow-spiking (putative interneurons)
        if wf_duration_sec < param['narrow_waveform_duration_max']:
            cell_types[i] = 'Narrow-spiking'
        
        # Wide-spiking (putative pyramidal)
        elif wf_duration_sec > param['wide_waveform_duration_min']:
            cell_types[i] = 'Wide-spiking'
        
        # Intermediate
        else:
            cell_types[i] = 'Intermediate'
    
    return cell_types