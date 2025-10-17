import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
from pathlib import Path
import pandas as pd
import pickle
import os

from bombcell.ccg_fast import acg, ccg

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    
    # Only print once when running in main process
    import multiprocessing
    if multiprocessing.current_process().name == 'MainProcess':
        print("✅ ipywidgets available - interactive GUI ready")
except ImportError:
    widgets = None
    display = None
    clear_output = None
    print("❌ ERROR: ipywidgets not available!")
    print("📦 The BombCell GUI requires ipywidgets. Please install it:")
    print("   pip install ipywidgets")
    print("   OR")
    print("   conda install ipywidgets")
    print("💡 Then restart your Jupyter kernel and try again.")

def precompute_gui_data(ephys_data, quality_metrics, param, save_path=None):
    """
    Pre-compute all visualization data for fast GUI loading (like MATLAB BombCell)
    
    Parameters:
    -----------
    ephys_data : dict
        Ephys data dictionary
    quality_metrics : dict  
        Quality metrics dictionary
    param : dict
        Parameters dictionary
    save_path : str, optional
        Path to save pre-computed data. If directory, saves as 'for_GUI/gui_data.pkl'.
        If None, attempts to save to param['ephysKilosortPath']/bombcell/for_GUI/
    
    Returns:
    --------
    gui_data : dict
        Dictionary containing all pre-computed visualization data
    """
    if param.get("verbose", False):
        print("Pre-computing GUI visualization data...")
    
    unique_units = np.unique(ephys_data['spike_clusters'])
    n_units = len(unique_units)
    
    gui_data = {
        'peak_locations': {},
        'trough_locations': {},
        'peak_loc_for_duration': {},
        'trough_loc_for_duration': {},
        'peak_trough_labels': {},
        'duration_lines': {},
        'spatial_decay_fits': {},
        'amplitude_fits': {},
        'channel_arrangements': {},
        'waveform_scaling': {},
          }
    
    # Import here to avoid dependency issues
    try:
        from scipy.signal import find_peaks
        from scipy.optimize import curve_fit
        from scipy import stats
        SCIPY_AVAILABLE = True
    except ImportError:
        SCIPY_AVAILABLE = False
        print("Warning: SciPy not available, using simplified computations")
    
    for unit_idx in range(n_units):
        if unit_idx % 50 == 0 and param.get("verbose", False):
            print(f"Processing unit {unit_idx}/{n_units}")
            
        unit_id = unique_units[unit_idx]
        
        # Get unit data
        if 'templates' in ephys_data and unit_idx < len(ephys_data['templates']):
            template = ephys_data['templates'][unit_idx]
        elif 'template_waveforms' in ephys_data and unit_idx < len(ephys_data['template_waveforms']):
            template = ephys_data['template_waveforms'][unit_idx]
        else:
            continue
            
        # Get max channel
        if 'maxChannels' in quality_metrics and unit_idx < len(quality_metrics['maxChannels']):
            max_ch = int(quality_metrics['maxChannels'][unit_idx])
        else:
            max_ch = 0
            
        # Pre-compute peak/trough detection using quality metrics function
        max_ch_waveform = None
        if template.size > 0 and len(template.shape) > 1 and max_ch < template.shape[1]:
            max_ch_waveform = template[:, max_ch]  # Define here for use later
            
            # Use the same waveform_shape function as quality metrics to get consistent results
            try:
                from .quality_metrics import waveform_shape
                
                # Get max channels for all units (quality metrics has this)
                if 'maxChannels' in quality_metrics and len(quality_metrics['maxChannels']) > unit_idx:
                    all_max_channels = quality_metrics['maxChannels']
                else:
                    # Fallback: compute max channels  
                    all_max_channels = np.argmax(np.max(np.abs(ephys_data['template_waveforms']), axis=1), axis=1)
                
                # Get waveform baseline window from param
                baseline_window = [param.get('waveform_baseline_window_start', 21), 
                                 param.get('waveform_baseline_window_stop', 31)]
                
                # Run waveform_shape to get the actual peak/trough locations
                result = waveform_shape(
                    template_waveforms=ephys_data['template_waveforms'],
                    this_unit=unit_idx,
                    maxChannels=all_max_channels,
                    channel_positions=ephys_data.get('channel_positions', np.array([[0, 0]])),
                    waveform_baseline_window=baseline_window,
                    param=param
                )
                
                # Extract peak/trough locations from quality metrics result
                peak_locs_for_gui = result[11]  # Index 11 in return tuple  
                trough_locs_for_gui = result[12]  # Index 12 in return tuple
                peak_loc_for_duration_gui = result[13]  # Index 13
                trough_loc_for_duration_gui = result[14]  # Index 14
                
                gui_data['peak_locations'][unit_idx] = peak_locs_for_gui
                gui_data['trough_locations'][unit_idx] = trough_locs_for_gui
                
                # Store duration indices from quality metrics
                if not np.isnan(peak_loc_for_duration_gui) and not np.isnan(trough_loc_for_duration_gui):
                    gui_data['peak_loc_for_duration'][unit_idx] = int(peak_loc_for_duration_gui)
                    gui_data['trough_loc_for_duration'][unit_idx] = int(trough_loc_for_duration_gui)
                
            except Exception as e:
                if param.get("verbose", False):
                    print(f"Warning: Could not compute quality metrics peaks/troughs for unit {unit_idx}, falling back to simple detection: {e}")
                
                # Fallback to simple detection if quality metrics fail
                if SCIPY_AVAILABLE:
                    max_ch_waveform = template[:, max_ch]
                    waveform_range = np.max(max_ch_waveform) - np.min(max_ch_waveform)
                    
                    peaks, _ = find_peaks(max_ch_waveform, height=np.max(max_ch_waveform) * 0.5, distance=10, prominence=waveform_range * 0.1)
                    troughs, _ = find_peaks(-max_ch_waveform, height=-np.min(max_ch_waveform) * 0.5, distance=10, prominence=waveform_range * 0.1)
                    
                    gui_data['peak_locations'][unit_idx] = peaks
                    gui_data['trough_locations'][unit_idx] = troughs
                    
                    # Pre-compute duration lines for fallback
                    if len(peaks) > 0 and len(troughs) > 0:
                        max_ch_waveform = template[:, max_ch]
                        main_peak_idx = peaks[np.argmax(max_ch_waveform[peaks])]
                        main_trough_idx = troughs[np.argmin(max_ch_waveform[troughs])]
                        gui_data['peak_loc_for_duration'][unit_idx] = int(main_peak_idx)
                        gui_data['trough_loc_for_duration'][unit_idx] = int(main_trough_idx)
            
        # Pre-compute spatial decay information
        if ('channel_positions' in ephys_data and 
            'spatialDecaySlope' in quality_metrics and 
            unit_idx < len(quality_metrics['spatialDecaySlope']) and
            not np.isnan(quality_metrics['spatialDecaySlope'][unit_idx])):
            
            positions = ephys_data['channel_positions']
            if max_ch < len(positions):
                max_pos = positions[max_ch]
                
                # Find nearby channels (within 100μm)
                nearby_channels = []
                distances = []
                amplitudes = []
                
                for ch in range(template.shape[1]):
                    if ch < len(positions):
                        distance = np.sqrt(np.sum((positions[ch] - max_pos)**2))
                        if distance < 100:
                            nearby_channels.append(ch)
                            distances.append(distance)
                            amplitudes.append(np.max(template[:, ch]))
                
                if len(nearby_channels) > 1:
                    distances = np.array(distances)
                    amplitudes = np.array(amplitudes)
                    
                    # Normalize amplitudes
                    max_amp = np.max(amplitudes)
                    if max_amp > 0:
                        amplitudes = amplitudes / max_amp
                        
                        # Fit exponential decay
                        valid_idx = (distances > 0) & (amplitudes > 0.05)
                        if np.sum(valid_idx) > 1:
                            try:
                                x_fit = distances[valid_idx]
                                y_fit = amplitudes[valid_idx]
                                log_y = np.log(y_fit + 1e-10)
                                coeffs = np.polyfit(x_fit, log_y, 1)
                                
                                # Generate fitted curve
                                x_smooth = np.linspace(0, np.max(distances)*1.1, 100)
                                y_smooth = np.exp(np.polyval(coeffs, x_smooth))
                                
                                gui_data['spatial_decay_fits'][unit_idx] = {
                                    'channels': nearby_channels,
                                    'distances': distances,
                                    'amplitudes': amplitudes,
                                    'fit_x': x_smooth,
                                    'fit_y': y_smooth
                                }
                            except:
                                pass
        
        # Pre-compute amplitude fit
        spike_mask = ephys_data['spike_clusters'] == unit_id
        if 'template_amplitudes' in ephys_data and np.sum(spike_mask) > 10:
            amplitudes = ephys_data['template_amplitudes'][spike_mask]
            
            if SCIPY_AVAILABLE and len(amplitudes) > 10:
                try:
                    # Create histogram
                    n_bins = min(50, int(len(amplitudes) / 10))
                    hist_counts, bin_edges = np.histogram(amplitudes, bins=n_bins)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    
                    # Fit cutoff Gaussian (like BombCell)
                    def gaussian_cut(x, a, x0, sigma, xcut):
                        g = a * np.exp(-(x - x0)**2 / (2 * sigma**2))
                        g[x < xcut] = 0
                        return g
                    
                    p0 = [np.max(hist_counts), np.median(amplitudes), 
                          np.std(amplitudes), np.min(amplitudes)]
                    bounds = ([0, np.min(amplitudes), 0, np.min(amplitudes)],
                             [np.inf, np.max(amplitudes), np.ptp(amplitudes), np.median(amplitudes)])
                    
                    popt, _ = curve_fit(gaussian_cut, bin_centers, hist_counts, 
                                      p0=p0, bounds=bounds, maxfev=5000)
                    
                    # Generate fit curve
                    y_smooth = np.linspace(np.min(amplitudes), np.max(amplitudes), 200)
                    x_smooth = gaussian_cut(y_smooth, *popt)
                    
                    # Calculate percentage missing
                    norm_area_ndtr = stats.norm.cdf((popt[1] - popt[3]) / popt[2])
                    percent_missing = 100 * (1 - norm_area_ndtr)
                    
                    gui_data['amplitude_fits'][unit_idx] = {
                        'hist_counts': hist_counts,
                        'bin_centers': bin_centers,
                        'fit_x': x_smooth,
                        'fit_y': y_smooth,
                        'percent_missing': percent_missing,
                        'fit_params': popt
                    }
                except:
                    pass
        
        # Pre-compute channel arrangements
        if 'channel_positions' in ephys_data and max_ch < len(ephys_data['channel_positions']):
            positions = ephys_data['channel_positions']
            max_pos = positions[max_ch]
            
            # Find channels within 100μm
            channels_to_plot = []
            for ch in range(template.shape[1]):
                if ch < len(positions):
                    distance = np.sqrt(np.sum((positions[ch] - max_pos)**2))
                    if distance < 100:
                        channels_to_plot.append(ch)
            
            # Limit to 20 channels
            if len(channels_to_plot) > 20:
                distances = [(ch, np.sqrt(np.sum((positions[ch] - max_pos)**2))) 
                           for ch in channels_to_plot]
                distances.sort(key=lambda x: x[1])
                channels_to_plot = [ch for ch, _ in distances[:20]]
            
            gui_data['channel_arrangements'][unit_idx] = {
                'channels': channels_to_plot,
                'max_channel': max_ch,
                'scaling_factor': np.ptp(max_ch_waveform) * 2.5 if max_ch_waveform is not None else 1.0
            }
        
    
    # Save pre-computed data in "for_GUI" subfolder
    # Determine save path
    if save_path is None and 'ephysKilosortPath' in param:
        # Auto-save to bombcell/for_GUI/ folder
        bombcell_dir = os.path.join(param['ephysKilosortPath'], 'bombcell')
        save_path = bombcell_dir
        
    if save_path:
        # Create for_GUI subfolder
        if os.path.isdir(save_path) or not os.path.splitext(save_path)[1]:
            # If save_path is a directory, create for_GUI subfolder
            gui_folder = os.path.join(save_path, "for_GUI")
            os.makedirs(gui_folder, exist_ok=True)
            final_save_path = os.path.join(gui_folder, "gui_data.pkl")
        else:
            # If save_path is a file path, put it in for_GUI subfolder
            parent_dir = os.path.dirname(save_path)
            gui_folder = os.path.join(parent_dir, "for_GUI")
            os.makedirs(gui_folder, exist_ok=True)
            filename = os.path.basename(save_path)
            final_save_path = os.path.join(gui_folder, filename)
        
        try:
            with open(final_save_path, 'wb') as f:
                pickle.dump(gui_data, f)
            if param.get("verbose", False):
                print(f"Pre-computed GUI data saved to: {final_save_path}")
        except Exception as e:
            if param.get("verbose", False):
                print(f"Failed to save GUI data: {e}")
    else:
        if param.get("verbose", False):
            print("No save path provided - GUI data not saved")
    
    if param.get("verbose", False):
        print("Pre-computation complete!")
    return gui_data

def load_gui_data(load_path):
    """
    Load pre-computed GUI data
    
    Parameters:
    -----------
    load_path : str
        Path to load GUI data from. Can be:
        - Direct path to .pkl file
        - Directory containing for_GUI/gui_data.pkl
        - Directory where for_GUI/ subfolder will be checked
    
    Returns:
    --------
    gui_data : dict or None
        Pre-computed GUI data, or None if not found
    """
    # Try different path options
    possible_paths = []
    
    if load_path.endswith('.pkl'):
        # Direct path to file
        possible_paths.append(load_path)
        # Also try in for_GUI subfolder
        parent_dir = os.path.dirname(load_path)
        gui_folder = os.path.join(parent_dir, "for_GUI", os.path.basename(load_path))
        possible_paths.append(gui_folder)
    else:
        # Directory path - look for for_GUI/gui_data.pkl
        possible_paths.append(os.path.join(load_path, "for_GUI", "gui_data.pkl"))
        possible_paths.append(os.path.join(load_path, "gui_data.pkl"))
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    gui_data = pickle.load(f)
                print(f"Loaded GUI data from: {path}")
                return gui_data
            except Exception as e:
                print(f"Failed to load GUI data from {path}: {e}")
                continue
    
    print(f"GUI data file not found. Tried: {possible_paths}")
    return None


class InteractiveUnitQualityGUI:
    """
    Interactive GUI using ipywidgets for Jupyter notebooks
    """
    
    def __init__(self, ephys_data, quality_metrics, ephys_properties=None, 
                 raw_waveforms=None, param=None, unit_types=None, gui_data=None, save_path=None, layout='auto', auto_advance=True):
        """
        Initialize the interactive GUI
        
        Parameters:
        -----------
        gui_data : dict, optional
            Pre-computed GUI visualization data from precompute_gui_data()
            If provided, will use pre-computed results for faster display
        layout : str, optional
            Layout mode (portrait mode removed, always uses landscape)
            Default: 'auto'
        auto_advance : bool, optional
            Whether to automatically advance to next unit after manual classification
            Default: True
        """
        self.ephys_data = ephys_data
        self.quality_metrics = quality_metrics
        self.ephys_properties = ephys_properties or []
        self.raw_waveforms = raw_waveforms
        self.param = param or {}
        self.unit_types = unit_types
        self.save_path = save_path
        self.auto_advance = auto_advance
        
        # Determine layout mode
        self.layout_mode = self._determine_layout(layout)
        
        # Auto-load GUI data if not provided but param has path info
        if gui_data is None and param and 'ephysKilosortPath' in param:
            # Try to auto-load from standard bombcell location
            import os
            ks_path = param['ephysKilosortPath']
            possible_paths = [
                os.path.join(ks_path, 'bombcell', 'for_GUI', 'gui_data.pkl'),
                os.path.join(ks_path, 'for_GUI', 'gui_data.pkl'),
                os.path.join(ks_path, 'gui_data.pkl'),
                os.path.join(save_path, 'for_GUI', 'gui_data.pkl'),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    try:
                        gui_data = load_gui_data(os.path.dirname(path))
                        if gui_data:
                            print(f"🚀 Auto-loaded GUI data from: {path}")
                            break
                    except Exception as e:
                        continue
        
        self.gui_data = gui_data  # Pre-computed visualization data
        
        # Detailed gui_data loading feedback
        if gui_data:
            print(f"GUI data loaded successfully!")
            print(f"   Data types available: {list(gui_data.keys())}")
            
            # Check each data type
            data_status = []
            if 'peak_locations' in gui_data:
                count = len(gui_data['peak_locations'])
                data_status.append(f"Peak/trough detection: {count} units")
                
            if 'spatial_decay_fits' in gui_data:
                count = len(gui_data['spatial_decay_fits'])
                if count > 0:
                    data_status.append(f"Spatial decay fits: {count} units")
                else:
                    data_status.append(f"Spatial decay fits: {count} units (none available)")
                    
            if 'amplitude_fits' in gui_data:
                count = len(gui_data['amplitude_fits'])
                if count > 0:
                    data_status.append(f"Amplitude fits: {count} units")
                else:
                    data_status.append(f"Amplitude fits: {count} units (none available)")
                    
            
            for status in data_status:
                print(f"   {status}")
        else:
            print("No pre-computed GUI data found - will compute everything real-time")
        
        # Get unique units
        self.unique_units = np.unique(ephys_data['spike_clusters'])
        self.n_units = len(self.unique_units)
        print(f"Total units: {self.n_units}")
        self.current_unit_idx = 0
        
        # Initialize manual classifications (separate from bombcell unit_types)
        self._initialize_manual_classifications()
        
        # Setup widgets and display
        self.setup_widgets()
        
        self.compile_class_variables() # do it once here so that self.display_gui() doesn't break
        
        self.display_gui() 

        self.compile_class_variables() # do it here again to make sure all variable changes are properly reflected

        
        
    def _initialize_manual_classifications(self):
        """Initialize manual classification system - separate from bombcell classifications"""
        # Store original bombcell unit_types separately
        self.bombcell_unit_types = self.unit_types.copy() if self.unit_types is not None else None
        
        # Initialize manual classifications as a separate array
        self.manual_unit_types = self._load_manual_classifications()
        
        if self.manual_unit_types is not None:
            print("📂 Loaded existing manual classifications")
        else:
            # Initialize all as unclassified
            self.manual_unit_types = np.full(self.n_units, -1, dtype=int)  # -1 = unclassified
            print("📝 Initialized manual classification system (no previous classifications found)")
        
        # For display purposes, prioritize manual classifications over bombcell ones
        self.unit_types = self.manual_unit_types.copy()
        
        # Show auto-advance status
        if self.auto_advance:
            print("🚀 Auto-advance enabled: will automatically go to next unit after classification")
        else:
            print("👆 Auto-advance disabled: use navigation buttons to move between units")
        
        self.compile_class_variables()
    
    def export_manual_classifications(self, export_path=None):
        """
        Export manual classifications for parameter optimization
        
        Parameters:
        -----------
        export_path : str, optional
            Path to save exported classifications. If None, uses self.save_path
        
        Returns:
        --------
        dict : Dictionary with classification statistics and file paths
        """
        if export_path is None:
            export_path = self.save_path
            
        if export_path is None:
            print("⚠️  No save path provided for export")
            return None
            
        try:
            import os
            import pandas as pd
            
            os.makedirs(export_path, exist_ok=True)
            
            # Get classification counts for manual classifications
            class_names = {-1: "Unclassified", 0: "Noise", 1: "Good", 2: "MUA", 3: "Non-somatic"}
            manual_counts = {}
            for class_id, class_name in class_names.items():
                manual_counts[class_name] = np.sum(self.manual_unit_types == class_id)
            
            # Create detailed export with both manual and bombcell data
            bombcell_types = self.bombcell_unit_types if self.bombcell_unit_types is not None else np.full(self.n_units, -1)
            
            df = pd.DataFrame({
                'unit_id': self.unique_units,
                'unit_index': range(self.n_units),
                'manual_classification': self.manual_unit_types,
                'bombcell_classification': bombcell_types,
                'manual_classification_name': [class_names.get(t, 'Unknown') for t in self.manual_unit_types],
                'bombcell_classification_name': [class_names.get(t, 'Unknown') for t in bombcell_types]
            })
            
            # Save files
            csv_file = os.path.join(export_path, 'manual_classifications_for_optimization.csv')
            tsv_file = os.path.join(export_path, 'manual_classifications_for_optimization.tsv')
            
            df.to_csv(csv_file, index=False)
            df.to_csv(tsv_file, sep='\t', index=False)
            
            # Create summary
            summary = {
                'total_units': self.n_units,
                'manual_classification_counts': manual_counts,
                'files_saved': [csv_file, tsv_file],
                'manually_classified_units': np.sum(self.manual_unit_types != -1),
                'unclassified_units': np.sum(self.manual_unit_types == -1)
            }
            
            print("🎯 Manual Classification Export Summary:")
            print(f"   Total units: {summary['total_units']}")
            print(f"   Manually classified: {summary['manually_classified_units']}")
            print(f"   Unclassified: {summary['unclassified_units']}")
            print("   Manual classification breakdown:")
            for class_name, count in manual_counts.items():
                if count > 0:
                    print(f"     {class_name}: {count}")
            print(f"   Exported to: {csv_file}")
            print(f"   Note: File includes both manual and BombCell classifications for comparison")
            
            return summary
            
        except Exception as e:
            print(f"⚠️  Error exporting classifications: {str(e)}")
            return None
    
    def get_classification_comparison(self):
        """
        Get statistics comparing manual vs bombcell classifications
        
        Returns:
        --------
        dict : Dictionary with comparison statistics
        """
        if self.bombcell_unit_types is None:
            print("⚠️  No BombCell classifications available for comparison")
            return None
        
        class_names = {-1: "Unclassified", 0: "Noise", 1: "Good", 2: "MUA", 3: "Non-somatic"}
        
        # Count agreements and disagreements
        manual_classified = self.manual_unit_types != -1
        n_manual = np.sum(manual_classified)
        
        if n_manual == 0:
            print("⚠️  No manual classifications available for comparison")
            return None
        
        # For units that have manual classifications, compare with bombcell
        agreements = 0
        disagreements = 0
        comparison_details = {}
        
        for i in range(self.n_units):
            if self.manual_unit_types[i] != -1:  # Has manual classification
                manual_class = self.manual_unit_types[i]
                bombcell_class = self.bombcell_unit_types[i]
                
                if manual_class == bombcell_class:
                    agreements += 1
                else:
                    disagreements += 1
                    
                # Track specific disagreements
                manual_name = class_names.get(manual_class, 'Unknown')
                bombcell_name = class_names.get(bombcell_class, 'Unknown')
                key = f"{bombcell_name} -> {manual_name}"
                comparison_details[key] = comparison_details.get(key, 0) + 1
        
        agreement_rate = agreements / n_manual if n_manual > 0 else 0
        
        stats = {
            'total_units': self.n_units,
            'manually_classified': n_manual,
            'agreements': agreements,
            'disagreements': disagreements,
            'agreement_rate': agreement_rate,
            'disagreement_details': comparison_details
        }
        
        print("📊 Manual vs BombCell Classification Comparison:")
        print(f"   Units with manual classification: {n_manual}/{self.n_units}")
        print(f"   Agreements: {agreements}")
        print(f"   Disagreements: {disagreements}")
        print(f"   Agreement rate: {agreement_rate:.1%}")
        
        if comparison_details:
            print("   Disagreement patterns:")
            for pattern, count in comparison_details.items():
                print(f"     {pattern}: {count}")
        
        return stats
    
    def _determine_layout(self, layout):
        """Always return landscape mode - portrait mode removed"""
        return 'landscape'
        
    def setup_widgets(self):
        """Setup interactive widgets"""
        # Unit navigation
        self.unit_slider = widgets.IntSlider(
            value=0,
            min=0,
            max=self.n_units-1,
            step=1,
            description='Unit:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='500px')
        )
        self.unit_slider.observe(self.on_unit_change, names='value')
        
        # Unit number input
        self.unit_input = widgets.IntText(
            value=0, min=0, max=self.n_units-1,
            description='Go to:', placeholder='Enter unit #'
        )
        self.goto_unit_btn = widgets.Button(description='Go', button_style='primary')
        
        # Navigation buttons (bold arrows, bigger)
        self.prev_btn = widgets.Button(description='◀', button_style='info', 
                                      layout=widgets.Layout(width='70px', height='32px'))
        self.next_btn = widgets.Button(description='▶', button_style='info',
                                      layout=widgets.Layout(width='70px', height='32px'))
        
        # Unit type navigation - both directions (slightly smaller for good/mua/noise)
        self.goto_prev_good_btn = widgets.Button(description='◀ good', button_style='success',
                                                 layout=widgets.Layout(width='80px', height='32px'))
        self.goto_good_btn = widgets.Button(description='good ▶', button_style='success',
                                           layout=widgets.Layout(width='80px', height='32px'))
        self.goto_prev_mua_btn = widgets.Button(description='◀ mua', button_style='warning',
                                               layout=widgets.Layout(width='75px', height='32px'))
        self.goto_mua_btn = widgets.Button(description='mua ▶', button_style='warning',
                                          layout=widgets.Layout(width='75px', height='32px'))
        self.goto_prev_noise_btn = widgets.Button(description='◀ noise', button_style='danger',
                                                  layout=widgets.Layout(width='85px', height='32px'))
        self.goto_noise_btn = widgets.Button(description='noise ▶', button_style='danger',
                                            layout=widgets.Layout(width='85px', height='32px'))
        self.goto_prev_nonsomatic_btn = widgets.Button(description='◀ non-somatic', button_style='primary',
                                                      layout=widgets.Layout(width='150px', height='32px'))
        self.goto_nonsomatic_btn = widgets.Button(description='non-somatic ▶', button_style='primary',
                                                  layout=widgets.Layout(width='150px', height='32px'))
        
        # Unit info display
        self.unit_info = widgets.HTML(value="")
        
        # Classification toggle buttons - match widths with navigation buttons above
        self.classify_good_btn = widgets.Button(
            description='mark as good', 
            button_style='success',
            layout=widgets.Layout(width='160px', height='35px')  # Match ◀good + good▶ = 80+80
        )
        self.classify_mua_btn = widgets.Button(
            description='mark as MUA', 
            button_style='warning',
            layout=widgets.Layout(width='150px', height='35px')  # Match ◀mua + mua▶ = 75+75
        )
        self.classify_nonsomatic_btn = widgets.Button(
            description='mark as non-somatic', 
            button_style='primary',
            layout=widgets.Layout(width='300px', height='35px')  # Match ◀non-somatic + non-somatic▶ = 150+150
        )
        self.classify_noise_btn = widgets.Button(
            description='mark as noise', 
            button_style='danger',
            layout=widgets.Layout(width='170px', height='35px')  # Match ◀noise + noise▶ = 85+85
        )
        
        # Navigation helper button
        self.goto_next_unclassified_btn = widgets.Button(
            description='▶ next unclassified', 
            button_style='info',
            layout=widgets.Layout(width='180px', height='32px')
        )
        
        # Output widget for plots
        self.plot_output = widgets.Output()
        
        # Connect callbacks
        self.unit_slider.observe(self.on_unit_change, names='value')
        self.prev_btn.on_click(self.prev_unit)
        self.next_btn.on_click(self.next_unit)
        self.goto_unit_btn.on_click(self.goto_unit_number)
        
        # Unit type navigation callbacks
        self.goto_prev_good_btn.on_click(self.goto_prev_good)
        self.goto_good_btn.on_click(self.goto_next_good)
        self.goto_prev_mua_btn.on_click(self.goto_prev_mua)
        self.goto_mua_btn.on_click(self.goto_next_mua)
        self.goto_prev_noise_btn.on_click(self.goto_prev_noise)
        self.goto_noise_btn.on_click(self.goto_next_noise)
        self.goto_prev_nonsomatic_btn.on_click(self.goto_prev_nonsomatic)
        self.goto_nonsomatic_btn.on_click(self.goto_next_nonsomatic)
        self.classify_good_btn.on_click(lambda b: self.classify_unit(1))
        self.classify_mua_btn.on_click(lambda b: self.classify_unit(2))
        self.classify_nonsomatic_btn.on_click(lambda b: self.classify_unit(3))
        self.classify_noise_btn.on_click(lambda b: self.classify_unit(0))
        self.goto_next_unclassified_btn.on_click(self.goto_next_unclassified)

        self.compile_class_variables()
        
    def display_gui(self):
        """Display the GUI"""
        # Basic navigation section (prev/next unit)
        basic_nav_text = widgets.HTML("<b>Go to next/prev. unit:</b>", layout=widgets.Layout(text_align='center'))
        basic_nav_buttons = widgets.HBox([
            self.prev_btn, self.next_btn
        ], layout=widgets.Layout(justify_content='center'))
        basic_nav_section = widgets.VBox([basic_nav_text, basic_nav_buttons])
        
        # Unit type navigation section
        type_nav_text = widgets.HTML("<b>Go to next/prev. good, MUA, non-somatic or noise unit:</b>", layout=widgets.Layout(text_align='center'))
        type_nav_buttons = widgets.HBox([
            self.goto_prev_good_btn, self.goto_good_btn,
            self.goto_prev_mua_btn, self.goto_mua_btn,
            self.goto_prev_nonsomatic_btn, self.goto_nonsomatic_btn,
            self.goto_prev_noise_btn, self.goto_noise_btn
        ], layout=widgets.Layout(justify_content='center'))
        type_nav_section = widgets.VBox([type_nav_text, type_nav_buttons])
        
        # Combined navigation controls
        nav_controls = widgets.HBox([
            basic_nav_section,
            widgets.Label('  |  '),
            type_nav_section
        ], layout=widgets.Layout(justify_content='center'))
        
        # Slider and unit input controls combined
        slider_and_input = widgets.HBox([
            self.unit_slider,
            widgets.Label('    '),
            self.unit_input, 
            self.goto_unit_btn
        ])
        
        # Classification section - match the structure of nav_controls above
        classify_left_section = widgets.VBox([
            widgets.HTML("<b> _ </b>", layout=widgets.Layout(text_align='center')),  # Text above navigation button
            widgets.HBox([self.goto_next_unclassified_btn], layout=widgets.Layout(justify_content='center'))
        ])
        
        classify_right_section = widgets.VBox([
            widgets.HTML("<b>manual classification (optional):</b>", layout=widgets.Layout(text_align='center')),  # Text above buttons
            widgets.HBox([
                self.classify_good_btn, self.classify_mua_btn, 
                self.classify_nonsomatic_btn, self.classify_noise_btn
            ], layout=widgets.Layout(justify_content='center'))
        ])
        
        # Combined classification controls - match nav_controls structure exactly
        classify_controls = widgets.HBox([
            classify_left_section,
            widgets.Label('  |  '),
            classify_right_section
        ], layout=widgets.Layout(justify_content='center'))
        
        classify_section = classify_controls
        
        # Full interface
        interface = widgets.VBox([
            slider_and_input,
            self.unit_info,
            nav_controls,
            classify_section,  # Manual classification buttons
            self.plot_output
        ])
        
        display(interface)
        
        # Initial plot
        self.update_display()
        
    def on_unit_change(self, change):
        """Handle unit slider change"""
        self.current_unit_idx = change['new']
        self.update_display()

        self.compile_class_variables()
        
    def prev_unit(self, b=None):
        """Go to previous unit"""
        if self.current_unit_idx > 0:
            self.current_unit_idx -= 1
            self.unit_slider.value = self.current_unit_idx
        
        self.compile_class_variables()
            
    def next_unit(self, b=None):
        """Go to next unit"""
        if self.current_unit_idx < self.n_units - 1:
            self.current_unit_idx += 1
            self.unit_slider.value = self.current_unit_idx
        
        self.compile_class_variables()
            
    def goto_unit_number(self, b=None):
        """Go to specific unit number"""
        unit_num = self.unit_input.value
        if 0 <= unit_num < self.n_units:
            self.current_unit_idx = unit_num
            self.unit_slider.value = self.current_unit_idx
            self.update_display()

        self.compile_class_variables()
            
    def goto_next_good(self, b=None):
        """Go to next BombCell-classified good unit"""
        if self.bombcell_unit_types is not None:
            for i in range(self.current_unit_idx + 1, self.n_units):
                if self.bombcell_unit_types[i] == 1:  # Good unit
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
        self.compile_class_variables()
                    
    def goto_prev_good(self, b=None):
        """Go to previous BombCell-classified good unit"""
        if self.bombcell_unit_types is not None:
            for i in range(self.current_unit_idx - 1, -1, -1):
                if self.bombcell_unit_types[i] == 1:  # Good unit
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
        
        self.compile_class_variables()
                    
    def goto_next_mua(self, b=None):
        """Go to next BombCell-classified MUA unit"""
        if self.bombcell_unit_types is not None:
            for i in range(self.current_unit_idx + 1, self.n_units):
                if self.bombcell_unit_types[i] == 2:  # MUA unit
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
        
        self.compile_class_variables()
                    
    def goto_prev_mua(self, b=None):
        """Go to previous BombCell-classified MUA unit"""
        if self.bombcell_unit_types is not None:
            for i in range(self.current_unit_idx - 1, -1, -1):
                if self.bombcell_unit_types[i] == 2:  # MUA unit
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
        
        self.compile_class_variables()
                    
    def goto_next_noise(self, b=None):
        """Go to next BombCell-classified noise unit"""
        if self.bombcell_unit_types is not None:
            for i in range(self.current_unit_idx + 1, self.n_units):
                if self.bombcell_unit_types[i] == 0:  # Noise unit
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
        
        self.compile_class_variables()
                    
    def goto_prev_noise(self, b=None):
        """Go to previous BombCell-classified noise unit"""
        if self.bombcell_unit_types is not None:
            for i in range(self.current_unit_idx - 1, -1, -1):
                if self.bombcell_unit_types[i] == 0:  # Noise unit
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
        
        self.compile_class_variables()
                    
    def goto_next_nonsomatic(self, b=None):
        """Go to next BombCell-classified non-somatic unit"""
        if self.bombcell_unit_types is not None:
            for i in range(self.current_unit_idx + 1, self.n_units):
                if self.bombcell_unit_types[i] in [3, 4]:  # Non-somatic good or non-somatic MUA
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
        
        self.compile_class_variables()
                    
    def goto_prev_nonsomatic(self, b=None):
        """Go to previous BombCell-classified non-somatic unit"""
        if self.bombcell_unit_types is not None:
            for i in range(self.current_unit_idx - 1, -1, -1):
                if self.bombcell_unit_types[i] in [3, 4]:  # Non-somatic good or non-somatic MUA
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
        
        self.compile_class_variables()
                    
    def classify_unit(self, classification):
        """
        Classify current unit manually (separate from bombcell classification)
        
        Parameters:
        -----------
        classification : int
            0 = Noise, 1 = Good, 2 = MUA, 3 = Non-somatic
        """
        # Update manual classification
        old_manual_class = self.manual_unit_types[self.current_unit_idx]
        self.manual_unit_types[self.current_unit_idx] = classification
        
        # Update display unit_types (prioritize manual over bombcell)
        self.unit_types[self.current_unit_idx] = classification
        
        # Map classification numbers to names for user feedback
        class_names = {-1: 'Unclassified', 0: 'Noise', 1: 'Good', 2: 'MUA', 3: 'Non-somatic'}
        unit_id = self.unique_units[self.current_unit_idx]
        
        # Show both manual and bombcell classifications for comparison
        bombcell_class = self.bombcell_unit_types[self.current_unit_idx] if self.bombcell_unit_types is not None else -1
        bombcell_name = class_names.get(bombcell_class, 'Unknown')
        
        print(f"✓ Unit {unit_id} manually classified as: {class_names.get(classification, 'Unknown')}")
        print(f"  (BombCell auto-classification: {bombcell_name})")
        
        # Show progress
        n_classified = np.sum(self.manual_unit_types != -1)
        progress = n_classified / self.n_units * 100
        print(f"  Progress: {n_classified}/{self.n_units} units manually classified ({progress:.1f}%)")
        
        # Save classifications if save_path is provided
        if self.save_path:
            self._save_manual_classifications()
        
        self.update_unit_info()
        
        # Automatically advance to next unit
        self._auto_advance_to_next_unit()
    
    def _auto_advance_to_next_unit(self):
        """Automatically advance to the next unit after manual classification"""
        if not self.auto_advance:
            return
            
        if self.current_unit_idx < self.n_units - 1:
            self.current_unit_idx += 1
            self.unit_slider.value = self.current_unit_idx
            print(f"   → Advanced to Unit {self.unique_units[self.current_unit_idx]} (#{self.current_unit_idx+1}/{self.n_units})")
        else:
            print(f"   → Reached last unit ({self.n_units}/{self.n_units})")
        
        self.compile_class_variables()
    
    def goto_next_unclassified(self, b=None):
        """Navigate to the next unclassified unit"""
        start_idx = self.current_unit_idx
        
        # Look for next unclassified unit (manual_classification == -1)
        for i in range(self.current_unit_idx + 1, self.n_units):
            if self.manual_unit_types[i] == -1:
                self.current_unit_idx = i
                self.unit_slider.value = self.current_unit_idx
                unit_id = self.unique_units[self.current_unit_idx]
                print(f"→ Next unclassified unit: {unit_id} (#{self.current_unit_idx+1}/{self.n_units})")
                return
        
        # If no unclassified units found after current position, wrap around
        for i in range(0, start_idx):
            if self.manual_unit_types[i] == -1:
                self.current_unit_idx = i
                self.unit_slider.value = self.current_unit_idx
                unit_id = self.unique_units[self.current_unit_idx]
                print(f"→ Next unclassified unit (wrapped): {unit_id} (#{self.current_unit_idx+1}/{self.n_units})")
                return
        
        print("→ All units have been manually classified!")
    
    def get_classification_progress(self):
        """Get summary of manual classification progress"""
        n_classified = np.sum(self.manual_unit_types != -1)
        progress = n_classified / self.n_units * 100
        
        # Count by category
        class_names = {0: "Noise", 1: "Good", 2: "MUA", 3: "Non-somatic"}
        counts = {}
        for class_id, class_name in class_names.items():
            counts[class_name] = np.sum(self.manual_unit_types == class_id)
        
        unclassified = np.sum(self.manual_unit_types == -1)
        
        print("📊 Manual Classification Progress:")
        print(f"   Classified: {n_classified}/{self.n_units} ({progress:.1f}%)")
        print(f"   Unclassified: {unclassified}")
        print("   Breakdown:")
        for class_name, count in counts.items():
            if count > 0:
                print(f"     {class_name}: {count}")
        
        return {
            'total': self.n_units,
            'classified': n_classified,
            'unclassified': unclassified,
            'progress_percent': progress,
            'counts': counts
        }
    
    def _save_manual_classifications(self):
        """Save manual classifications to file (separate from bombcell classifications)"""
        try:
            import os
            import pandas as pd
            
            # Create save directory if it doesn't exist
            os.makedirs(self.save_path, exist_ok=True)
            
            # Create DataFrame with detailed comparison data
            bombcell_types = self.bombcell_unit_types if self.bombcell_unit_types is not None else np.full(self.n_units, -1)
            
            df = pd.DataFrame({
                'unit_id': self.unique_units,
                'manual_classification': self.manual_unit_types,
                'bombcell_classification': bombcell_types,
                'classification_source': ['manual' if self.manual_unit_types[i] != -1 else 'bombcell' for i in range(self.n_units)]
            })
            
            # Save manual classifications only (no bombcell data mixed in)
            manual_only_df = pd.DataFrame({
                'unit_id': self.unique_units,
                'manual_classification': self.manual_unit_types
            })
            
            # Save files with clear naming
            manual_csv = os.path.join(self.save_path, 'manual_unit_classifications.csv')
            manual_tsv = os.path.join(self.save_path, 'manual_unit_classifications.tsv')
            comparison_csv = os.path.join(self.save_path, 'manual_vs_bombcell_classifications.csv')
            
            manual_only_df.to_csv(manual_csv, index=False)
            manual_only_df.to_csv(manual_tsv, sep='\t', index=False)
            df.to_csv(comparison_csv, index=False)
            
            print(f"💾 Manual classifications saved to: {manual_csv}")
            print(f"💾 Manual vs BombCell comparison saved to: {comparison_csv}")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not save manual classifications: {str(e)}")
    
    def _load_manual_classifications(self):
        """Load previously saved manual classifications if they exist"""
        if not self.save_path:
            return None
            
        try:
            import os
            import pandas as pd
            
            # Try to load manual classifications specifically
            possible_files = [
                os.path.join(self.save_path, 'manual_unit_classifications.csv'),
                os.path.join(self.save_path, 'manual_unit_classifications.tsv')
            ]
            
            for file_path in possible_files:
                if os.path.exists(file_path):
                    if file_path.endswith('.csv'):
                        df = pd.read_csv(file_path)
                    else:
                        df = pd.read_csv(file_path, sep='\t')
                    
                    # Create mapping from unit_id to classification
                    classification_map = dict(zip(df['unit_id'], df['manual_classification']))
                    
                    # Apply to manual types array
                    manual_types = np.full(self.n_units, -1, dtype=int)  # -1 = unclassified
                    for i, unit_id in enumerate(self.unique_units):
                        if unit_id in classification_map:
                            manual_types[i] = classification_map[unit_id]
                    
                    print(f"📂 Loaded manual classifications from: {file_path}")
                    return manual_types
                    
        except Exception as e:
            print(f"⚠️  Warning: Could not load manual classifications: {str(e)}")
            
        return None

    @staticmethod
    def compile_variables(ephys_data, quality_metrics, ephys_properties=None, raw_waveforms=None, param=None, unit_types=None, gui_data=None, bombcell_unit_types=None, unit_slider=None) -> dict:

        unique_units = np.unique(ephys_data['spike_clusters'])
        n_units = len(unique_units)
        
        compiled_variables = {
            "ephys_data": ephys_data,
            "quality_metrics": quality_metrics,
            "ephys_properties": ephys_properties,
            "raw_waveforms": raw_waveforms,
            "param": param,
            "unit_types": unit_types,
            "gui_data": gui_data,
            "n_units": n_units,
            "unique_units": unique_units,
            "bombcell_unit_types": bombcell_unit_types,
            "unit_slider": unit_slider,
        }

        return compiled_variables

    def compile_class_variables(self):
        
        ephys_data = self.ephys_data if hasattr(self, "ephys_data") else None
        quality_metrics = self.quality_metrics if hasattr(self, "quality_metrics") else None
        ephys_properties = self.ephys_properties if hasattr(self, "ephys_properties") else None
        raw_waveforms = self.raw_waveforms if hasattr(self, "raw_waveforms") else None
        param = self.param if hasattr(self, "param") else None
        unit_types = self.unit_types if hasattr(self, "unit_types") else None
        gui_data = self.gui_data if hasattr(self, "gui_data") else None
        bombcell_unit_types = self.bombcell_unit_types if hasattr(self, "bombcell_unit_types") else None
        unit_slider = self.unit_slider if hasattr(self, "unit_slider") else None
        
        self.compiled_variables = InteractiveUnitQualityGUI.compile_variables(ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, bombcell_unit_types, unit_slider)

    @staticmethod
    def unpack_variables(compiled_variables: dict) -> list:
        ephys_data = compiled_variables["ephys_data"]
        quality_metrics = compiled_variables["quality_metrics"]
        ephys_properties = compiled_variables["ephys_properties"]
        raw_waveforms = compiled_variables["raw_waveforms"]
        param = compiled_variables["param"]
        unit_types = compiled_variables["unit_types"]
        gui_data  = compiled_variables["gui_data"]
        n_units = compiled_variables["n_units"]
        unique_units = compiled_variables["unique_units"]
        bombcell_unit_types = compiled_variables["bombcell_unit_types"]
        unit_slider = compiled_variables["unit_slider"]

        return ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider
    
    @staticmethod
    def get_unit_data(unit_idx, compiled_variables):
        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)

        """Get data for a specific unit"""
        if unit_idx >= n_units:
            return None
            
        unit_id = unique_units[unit_idx]
        
        # Get spike times for this unit
        spike_mask = ephys_data['spike_clusters'] == unit_id
        spike_times = ephys_data['spike_times'][spike_mask]
        
        # Get template waveform
        if unit_idx < len(ephys_data['template_waveforms']):
            template = ephys_data['template_waveforms'][unit_idx]
        else:
            template = np.zeros((82, 1))
            
        # Get quality metrics for this unit
        unit_metrics = {}
        
        # Handle different quality_metrics formats
        if isinstance(quality_metrics, list):
            # List of dicts format: [{'phy_clusterID': 0, 'metric1': val, ...}, ...]
            unit_found = False
            for unit_dict in quality_metrics:
                if unit_dict.get('phy_clusterID') == unit_id:
                    unit_metrics = unit_dict.copy()
                    unit_found = True
                    break
            if not unit_found:
                unit_metrics = {'phy_clusterID': unit_id}
                
        elif isinstance(quality_metrics, dict):
            if unit_id in quality_metrics:
                # Dict with unit_id keys: {0: {'metric1': val, ...}, 1: {...}}
                unit_metrics = quality_metrics[unit_id].copy()
            else:
                # Dict with metric keys: {'metric1': [val0, val1, ...], 'metric2': [...]}
                for key, values in quality_metrics.items():
                    if hasattr(values, '__len__') and len(values) > unit_idx:
                        unit_metrics[key] = values[unit_idx]
                    else:
                        unit_metrics[key] = np.nan
        elif hasattr(quality_metrics, 'iloc'):
            # DataFrame format from bc.load_bc_results()
            if unit_idx < len(quality_metrics):
                unit_metrics = quality_metrics.iloc[unit_idx].to_dict()
            else:
                unit_metrics = {}
        else:
            unit_metrics = {}
                
        return {
            'unit_id': unit_id,
            'unit_idx': unit_idx,
            'spike_times': spike_times,
            'template': template,
            'metrics': unit_metrics
        }
        
    def update_unit_info(self):
        """Update unit info display"""
        unit_data = self.get_unit_data(self.current_unit_idx, self.compiled_variables)
        if unit_data is None:
            return
            
        # Get manual and bombcell classifications
        type_map = {-1: "Unclassified", 0: "Noise", 1: "Good", 2: "MUA", 3: "Non-somatic", 4: "Non-somatic MUA"}
        
        manual_type = self.manual_unit_types[self.current_unit_idx] if self.current_unit_idx < len(self.manual_unit_types) else -1
        manual_type_str = type_map.get(manual_type, "Unknown")
        
        bombcell_type = self.bombcell_unit_types[self.current_unit_idx] if (self.bombcell_unit_types is not None and self.current_unit_idx < len(self.bombcell_unit_types)) else -1
        bombcell_type_str = type_map.get(bombcell_type, "Unknown")
        
        # Use manual classification for display color if available, otherwise bombcell
        display_type_str = manual_type_str if manual_type != -1 else bombcell_type_str
        
        # Get title color based on BOMBCELL classification (not user classification)
        title_colors = {
            "Unclassified": "gray",
            "Noise": "red",
            "Good": "green", 
            "MUA": "orange",
            "Non-somatic": "blue",
            "Non-somatic MUA": "blue",
            "Unknown": "black"
        }
        title_color = title_colors.get(bombcell_type_str, "black")
        
        # Create title showing BombCell classification first, then user classification
        if manual_type != -1:
            # Show BombCell classification | User classification
            classification_text = f"{bombcell_type_str} | User classification: {manual_type_str}"
        else:
            # Show only BombCell classification
            classification_text = f"{bombcell_type_str}"
        
        info_html = f"""
        <h1 style="color: {title_color}; text-align: center; font-size: 24px; margin: 10px 0;">Unit {unit_data['unit_id']} (phy ID = {self.current_unit_idx}, unit # {self.current_unit_idx+1}/{self.n_units}) - {classification_text}</h1>
        """
        
        self.unit_info.value = info_html

        self.compile_class_variables()
        
    def plot_unit(self, unit_idx):
        """Plot data for a specific unit with adaptive layout"""
        unit_data = self.get_unit_data(unit_idx, self.compiled_variables)
        if unit_data is None:
            return
            
        with self.plot_output:
            clear_output(wait=True)
            plt.close('all')
            
            # Always use landscape layout
            self._plot_unit_landscape(unit_data)
    
    def _plot_unit_landscape(self, unit_data):
        """Plot unit data in landscape mode (side-by-side layout)"""
        # CENTRALIZED FONT SIZE CONFIGURATION FOR LANDSCAPE MODE
        AXIS_LABEL_FONTSIZE = 20
        TICK_LABEL_FONTSIZE = 14
        LEGEND_FONTSIZE = 16
        TEXT_FONTSIZE = 16
        PLOT_TITLE_FONTSIZE = 22
        QUALITY_METRIC_TEXT_FONTSIZE = 15
        
        # Create figure with extended width and height for histograms
        fig = plt.figure(figsize=(30, 25))  # Taller figure
        fig.patch.set_facecolor('white')
        
        # LEFT HALF - Original GUI (columns 0-14) - MUCH LARGER GRID
        # 1. Unit location plot (left column)
        ax_location = plt.subplot2grid((100, 30), (0, 0), rowspan=100, colspan=1)
        self.plot_unit_location(ax_location, unit_data, self.compiled_variables)
        
        # 2. Template waveforms - scale to 100-row grid
        ax_template = plt.subplot2grid((100, 30), (0, 2), rowspan=20, colspan=6)
        self.plot_template_waveform(ax_template, unit_data, self.compiled_variables)
        
        # 3. Raw waveforms - scale to 100-row grid
        ax_raw = plt.subplot2grid((100, 30), (0, 9), rowspan=20, colspan=6)
        self.plot_raw_waveforms(ax_raw, unit_data, self.compiled_variables)
        
        # 4. Spatial decay - scale to 100-row grid
        ax_spatial = plt.subplot2grid((100, 30), (30, 2), rowspan=20, colspan=6)
        self.plot_spatial_decay(ax_spatial, unit_data, self.compiled_variables)
        
        # 5. ACG - scale to 100-row grid
        ax_acg = plt.subplot2grid((100, 30), (30, 9), rowspan=20, colspan=6)
        self.plot_autocorrelogram(ax_acg, unit_data, self.compiled_variables)
        
        # 6. Amplitudes over time - scale to 100-row grid
        ax_amplitude = plt.subplot2grid((100, 30), (60, 2), rowspan=20, colspan=10)
        self.plot_amplitudes_over_time(ax_amplitude, unit_data, self.compiled_variables)
        
        # 6b. Time bin metrics - scale to 100-row grid
        ax_bin_metrics = plt.subplot2grid((100, 30), (85, 2), rowspan=10, colspan=10, sharex=ax_amplitude)
        self.plot_time_bin_metrics(ax_bin_metrics, unit_data, self.compiled_variables)
        
        # 7. Amplitude fit - scale to 100-row grid
        ax_amp_fit = plt.subplot2grid((100, 30), (60, 13), rowspan=20, colspan=2)
        self.plot_amplitude_fit(ax_amp_fit, unit_data, self.compiled_variables)
        
        # RIGHT HALF - Histogram panel (columns 16-29)
        self.plot_histograms_panel(fig, unit_data, self.compiled_variables)
        
        # Adjust subplot margins to eliminate gap with title/buttons - seamless layout
        plt.subplots_adjust(left=0.03, right=0.98, top=0.99, bottom=0.08, hspace=0.4, wspace=0.4)
        
        # FORCE CONSISTENT FONTS ACROSS ALL PLOTS - OVERRIDE EVERYTHING (LANDSCAPE)
        for i, ax in enumerate(fig.get_axes()):
            # Skip axes that might be unit title or toggle buttons
            if hasattr(ax, 'get_position') and ax.get_position().height < 0.05:
                continue  # Skip very small axes (likely buttons)
            
            # Check which plots should have no ticks/labels
            is_template_waveform = (i == 1)  # Template waveforms
            is_raw_waveform = (i == 2)      # Raw waveforms
            is_spatial_decay = (i == 3)     # Spatial decay
            is_amplitude_plot = (i == 5)    # Scaling factor over time (amplitude over time)
            is_amplitude_fit = (i == 6)     # Scaling factor distribution (amplitude fit)
            
            # FIRST: FORCE ALL TITLES TO CONSISTENT SIZE (override any previous settings)
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=PLOT_TITLE_FONTSIZE, fontweight='bold')
            
            # SECOND: FORCE ALL AXIS LABELS TO CONSISTENT SIZE
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=AXIS_LABEL_FONTSIZE, labelpad=1)
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), fontsize=AXIS_LABEL_FONTSIZE, labelpad=1)
            
            # THIRD: FORCE ALL TICK LABELS TO CONSISTENT SIZE
            ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
            
            # NOW apply plot-specific rules for ticks/labels
            if (is_template_waveform or is_raw_waveform or is_spatial_decay or 
                is_amplitude_plot or is_amplitude_fit):
                # NO ticks or labels for these specific plots
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('')
                ax.set_ylabel('')
                # But keep the title if it exists
            else:
                # For plots that keep ticks, set min/max only with consistent formatting
                if len(ax.get_xlim()) == 2:
                    xlim = ax.get_xlim()
                    ax.set_xticks([xlim[0], xlim[1]])
                    xlabels = []
                    for x in xlim:
                        if abs(x) < 1000 and abs(x) > 0.01:
                            xlabels.append(f'{x:.2f}')
                        else:
                            xlabels.append(f'{x:.0f}')
                    ax.set_xticklabels(xlabels, fontsize=TICK_LABEL_FONTSIZE)
                
                if len(ax.get_ylim()) == 2:
                    ylim = ax.get_ylim()
                    ax.set_yticks([ylim[0], ylim[1]])
                    ylabels = []
                    for y in ylim:
                        if abs(y) < 1000 and abs(y) > 0.01:
                            ylabels.append(f'{y:.2f}')
                        else:
                            ylabels.append(f'{y:.0f}')
                    ax.set_yticklabels(ylabels, fontsize=TICK_LABEL_FONTSIZE)
            
            # FORCE ALL LEGENDS TO CONSISTENT SIZE
            legend = ax.get_legend()
            if legend:
                for text in legend.get_texts():
                    text.set_fontsize(LEGEND_FONTSIZE)
        
        plt.show()
    
    @staticmethod
    def plot_amplitude_histogram(ax, unit_data, compiled_variables, metric_name):
        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)

        """Plot amplitude histogram with current unit highlighted"""
        metric_data = quality_metrics[metric_name]
        metric_data = metric_data[~np.isnan(metric_data)]
        
        if len(metric_data) > 0:
            # Plot histogram with probability normalization (MATLAB style)
            bins = 40
            color = [1.0, 0.5469, 0]  # Orange color for amplitude
            
            n, bins_out, patches = ax.hist(metric_data, bins=bins, density=True, 
                                         color=color, alpha=0.7)
            
            # Convert to probability (like MATLAB's 'Normalization', 'probability')
            bin_width = bins_out[1] - bins_out[0]
            for patch in patches:
                patch.set_height(patch.get_height() * bin_width)
            
            # Add current unit highlighting
            current_unit_idx = unit_data["unit_idx"]
            if current_unit_idx < len(quality_metrics[metric_name]):
                current_value = quality_metrics[metric_name][current_unit_idx]
                if not np.isnan(current_value):
                    # NO red bin highlighting - removed for cleaner look
                    
                    # Add triangle above histogram (revert to original position)
                    bin_idx = np.digitize(current_value, bins_out) - 1
                    bin_height = patches[bin_idx].get_height() if 0 <= bin_idx < len(patches) else 0.5
                    triangle_y = bin_height + 0.15  # Above the histogram bars
                    
                    # Large black triangle pointing down with white contour for visibility
                    ax.scatter(current_value, triangle_y, marker='v', s=500, color='black', 
                              alpha=1.0, zorder=15, edgecolors='white', linewidths=4)
            
            # Formatting
            ax.set_xlabel('amplitude', fontsize=12)
            ax.set_ylabel('frac. units', fontsize=12)
            ax.set_ylim([0, 1.1])
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['0', '1'])
            
    @staticmethod
    def plot_template_waveform(ax, unit_data, compiled_variables):
        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)
        current_unit_idx = unit_data["unit_idx"]
        
        """Plot template waveform using BombCell MATLAB spatial arrangement"""
        template = unit_data['template']
        metrics = unit_data['metrics']
        
        if template.size > 0 and len(template.shape) > 1:
            # Get peak channel from quality metrics
            if 'maxChannels' in quality_metrics and current_unit_idx < len(quality_metrics['maxChannels']):
                max_ch = int(quality_metrics['maxChannels'][current_unit_idx])
            else:
                max_ch = int(metrics.get('maxChannels', 0))
            
            n_channels = template.shape[1]
            
            # Find channels within 100μm of max channel (like MATLAB BombCell)
            if 'channel_positions' in ephys_data and max_ch < len(ephys_data['channel_positions']):
                positions = ephys_data['channel_positions']
                max_pos = positions[max_ch]
                
                # Calculate distances and find nearby channels
                channels_to_plot = []
                for ch in range(n_channels):
                    if ch < len(positions):
                        distance = np.sqrt(np.sum((positions[ch] - max_pos)**2))
                        if distance < 100:  # Within 100μm like MATLAB
                            channels_to_plot.append(ch)
                
                # Limit to 20 channels max like MATLAB
                if len(channels_to_plot) > 20:
                    # Sort by distance and take closest 20
                    distances = [(ch, np.sqrt(np.sum((positions[ch] - max_pos)**2))) for ch in channels_to_plot]
                    distances.sort(key=lambda x: x[1])
                    channels_to_plot = [ch for ch, _ in distances[:20]]
                
                if len(channels_to_plot) > 0:
                    # Calculate scaling factor like MATLAB
                    max_ch_waveform = template[:, max_ch]
                    scaling_factor = np.ptp(max_ch_waveform) * 2.5
                    
                    # Create time axis
                    time_axis = np.arange(template.shape[0])
                    
                    # Plot each channel at its spatial position
                    for ch in channels_to_plot:
                        if ch < n_channels:
                            waveform = template[:, ch]
                            ch_pos = positions[ch]
                            
                            # Calculate X offset - use probe geometry with reasonable separation
                            waveform_width = template.shape[0]  # Usually 82 samples
                            x_offset = (ch_pos[0] - max_pos[0]) * waveform_width * 0.04  # Slightly increased from original 0.025
                            
                            # Calculate Y offset based on channel Y position 
                            # We want channel 0 at bottom, so higher Y values go up
                            y_offset = ch_pos[1] / 100 * scaling_factor  # Use absolute position
                            
                            # Plot waveform (not flipped)
                            x_vals = time_axis + x_offset
                            y_vals = waveform + y_offset
                            
                            if ch == max_ch:
                                ax.plot(x_vals, y_vals, 'k-', linewidth=3)  # Max channel thicker black
                            else:
                                ax.plot(x_vals, y_vals, 'k-', linewidth=1, alpha=0.7)
                            
                            # Add channel number closer to waveform
                            ax.text(x_offset - waveform_width * 0.05, y_offset, f'{ch}', fontsize=13, ha='right', va='center', fontfamily="DejaVu Sans", zorder=20)
                    
                    # Mark peaks and troughs on max channel
                    max_ch_waveform = template[:, max_ch]
                    # Max channel position matches its actual plotted position
                    max_ch_x_offset = 0  
                    max_ch_y_offset = max_pos[1] / 100 * scaling_factor  # Same as max channel waveform
                    
                    # Detect and mark peaks/troughs (waveform not inverted now)
                    InteractiveUnitQualityGUI.mark_peaks_and_troughs(ax, unit_data, compiled_variables, max_ch_waveform, max_ch_x_offset, max_ch_y_offset, metrics, scaling_factor)
                    
                    # Don't invert - channel 0 (Y=0) naturally at bottom
                    
            else:
                # Fallback: simple single channel display
                ax.plot(template[:, max_ch], 'k-', linewidth=2)
                ax.text(0.5, 0.5, f'Template\n(channel {max_ch})', 
                       ha='center', va='center', transform=ax.transAxes, fontfamily="DejaVu Sans")
                    
        ax.set_title('Template waveforms', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Extend x-axis limits to accommodate channel number text (right side only)
        x_min, x_max = ax.get_xlim()
        text_padding = 100  # Padding for text labels on right side
        ax.set_xlim(x_min, x_max + text_padding)
        
        # Add quality metrics text
        InteractiveUnitQualityGUI.add_metrics_text(ax, unit_data, compiled_variables, 'template')
        
    @staticmethod
    def plot_raw_waveforms(ax, unit_data, compiled_variables):
        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)
        current_unit_idx = unit_data["unit_idx"]

        """Plot raw waveforms with 16 nearest channels like MATLAB"""
        metrics = unit_data['metrics']
        
        # Check if raw extraction is enabled
        extract_raw = param.get('extractRaw', 0)
        if extract_raw != 1:
            ax.text(0.5, 0.5, 'Mean raw waveforms\n(extractRaw disabled)', 
                    ha='center', va='center', transform=ax.transAxes, fontsize = 15, fontfamily="DejaVu Sans")
            ax.set_title('Mean raw waveforms', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
            return
        
        if raw_waveforms is not None:
            raw_wf = raw_waveforms.get('average', None)
            if raw_wf is not None:
                try:
                    if hasattr(raw_wf, '__len__') and current_unit_idx < len(raw_wf):
                        waveforms = raw_wf[current_unit_idx]
                        
                        if hasattr(waveforms, 'shape') and len(waveforms.shape) > 1:
                            # waveforms shape is (channels, time), need to transpose for plotting
                            waveforms = waveforms.T  # Now (time, channels)
                            # Multi-channel raw waveforms - use MATLAB spatial arrangement
                            if 'maxChannels' in quality_metrics and current_unit_idx < len(quality_metrics['maxChannels']):
                                max_ch = int(quality_metrics['maxChannels'][current_unit_idx])
                            else:
                                max_ch = int(metrics.get('maxChannels', 0))
                            n_channels = waveforms.shape[1]  # Number of channels (after ensuring time x channels)
                            
                            # Find channels within 100μm of max channel (like MATLAB BombCell)
                            if 'channel_positions' in ephys_data and len(ephys_data['channel_positions']) > 0 and max_ch < len(ephys_data['channel_positions']):
                                positions = ephys_data['channel_positions']
                                max_pos = positions[max_ch]
                                
                                # Calculate distances and find nearby channels
                                channels_to_plot = []
                                for ch in range(n_channels):
                                    if ch < len(positions):
                                        distance = np.sqrt(np.sum((positions[ch] - max_pos)**2))
                                        if distance < 100:  # Within 100μm like MATLAB
                                            channels_to_plot.append(ch)
                                
                                # Limit to 20 channels max like MATLAB
                                if len(channels_to_plot) > 20:
                                    # Sort by distance and take closest 20
                                    distances = [(ch, np.sqrt(np.sum((positions[ch] - max_pos)**2))) for ch in channels_to_plot]
                                    distances.sort(key=lambda x: x[1])
                                    channels_to_plot = [ch for ch, _ in distances[:20]]
                                
                                if len(channels_to_plot) > 0:
                                    # Calculate scaling factor like MATLAB
                                    max_ch_waveform = waveforms[:, max_ch] if max_ch < waveforms.shape[1] else waveforms[:, 0]
                                    scaling_factor = np.ptp(max_ch_waveform) * 2.5
                                    
                                    # Create time axis
                                    time_axis = np.arange(waveforms.shape[0])
                                    # Adjust time axis to center spike at 0 (assuming spike is at sample 20 for KS4)
                                    if waveforms.shape[0] == 61:  # Kilosort 4
                                        time_axis = time_axis - 20
                                    elif waveforms.shape[0] == 82:  # Kilosort < 4
                                        time_axis = time_axis - 41
                                    
                                    # Plot each channel at its spatial position
                                    for ch in channels_to_plot:
                                        if ch < n_channels:
                                            waveform = waveforms[:, ch]
                                            ch_pos = positions[ch]
                                            
                                            # Calculate X offset - use probe geometry with reasonable separation
                                            waveform_width = waveforms.shape[0]  # Usually 82 samples
                                            x_offset = (ch_pos[0] - max_pos[0]) * waveform_width * 0.06  # Slightly increased from original 0.05
                                            
                                            # Calculate Y offset based on channel Y position 
                                            # We want channel 0 at bottom, so higher Y values go up
                                            y_offset = ch_pos[1] / 100 * scaling_factor  # Use absolute position
                                            
                                            # Plot waveform (not flipped)
                                            x_vals = time_axis + x_offset
                                            y_vals = waveform + y_offset
                                            
                                            if ch == max_ch:
                                                ax.plot(x_vals, y_vals, 'k-', linewidth=3)  # Max channel thicker black
                                            else:
                                                ax.plot(x_vals, y_vals, 'gray', linewidth=1, alpha=0.7)
                                            
                                            # Add channel number to the left of waveform
                                            # Position based on the leftmost point of the waveform (time_axis[0])
                                            ax.text(time_axis[0] + x_offset - 5, y_offset, f'{ch}', fontsize=13, ha='right', va='center', fontfamily="DejaVu Sans", zorder=20)
                                    
                                    # Don't mark peaks and troughs on this spatial plot
                                    
                                    # Don't invert - channel 0 (Y=0) naturally at bottom
                                    ax.axis('tight')
                                    ax.set_aspect('auto')
                            else:
                                # Fallback: plot multiple channels overlaid
                                time_axis = np.arange(waveforms.shape[0])
                                
                                # Plot nearby channels
                                channels_to_show = min(10, waveforms.shape[1])  # Show up to 10 channels
                                for i in range(channels_to_show):
                                    if i == max_ch and max_ch < waveforms.shape[1]:
                                        ax.plot(time_axis, waveforms[:, i], 'k-', linewidth=2, alpha=1.0, label=f'Ch {i} (max)')
                                    else:
                                        ax.plot(time_axis, waveforms[:, i], 'gray', linewidth=1, alpha=0.5)
                                
                                ax.legend(loc='best', fontsize=8)
                                ax.set_xlabel('Time (samples)')
                                ax.set_ylabel('Amplitude')
                        else:
                            # Single channel
                            ax.plot(waveforms, 'b-', alpha=0.7)
                            
                except (TypeError, IndexError, AttributeError):
                    ax.text(0.5, 0.5, 'Raw waveforms\n(data format issue)', 
                            ha='center', va='center', transform=ax.transAxes, fontfamily="DejaVu Sans")
        else:
            ax.text(0.5, 0.5, 'Mean raw waveforms\n(not available)', 
                    ha='center', va='center', transform=ax.transAxes, fontfamily="DejaVu Sans")
                    
        ax.set_title('Mean raw waveforms', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove aspect ratio constraint to prevent squishing
        
        # Extend x-axis limits to accommodate channel number text (right side only)
        x_min, x_max = ax.get_xlim()
        text_padding = 100  # Padding for text labels on right side
        ax.set_xlim(x_min, x_max + text_padding)
        
        # Add quality metrics text
        InteractiveUnitQualityGUI.add_metrics_text(ax, unit_data, compiled_variables, 'raw')

    @staticmethod        
    def plot_autocorrelogram(ax, unit_data, compiled_variables, cbin=0.5, cwin=100):
        """Plot autocorrelogram with tauR and firing rate lines"""
        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)

        spike_times = unit_data['spike_times']
        metrics = unit_data['metrics']
        filtered_spike_times = spike_times.copy()
        
        # Filter spike times
        if param and param.get('computeTimeChunks', False):

            good_start_times = metrics.get('useTheseTimesStart', None)
            good_stop_times = metrics.get('useTheseTimesStop', None)
            
            if good_start_times is not None and good_stop_times is not None:
                # Ensure they are arrays
                if np.isscalar(good_start_times):
                    good_start_times = [good_start_times]
                if np.isscalar(good_stop_times):
                    good_stop_times = [good_stop_times]
                
                # Create mask for spikes in good time chunks
                good_spike_mask = np.zeros(len(spike_times), dtype=bool)
                for g_start, g_stop in zip(good_start_times, good_stop_times):
                    if not (np.isnan(g_start) or np.isnan(g_stop)):
                        good_spike_mask |= (spike_times >= g_start) & (spike_times <= g_stop)
                
                # Filter spike times to only good time chunks
                filtered_spike_times = spike_times[good_spike_mask]
        
        # Compute autocorrelogram
        bins = np.arange(-cwin / 2, cwin / 2 + cbin, cbin)
        bin_centers = bins[:-1] + cbin/2
        filtered_spike_times_samples = np.round(filtered_spike_times * param['ephys_sample_rate']).astype(np.uint64)
        autocorr = acg(filtered_spike_times_samples,
                       cbin,
                       cwin,
                       normalize='hertz') # built-in caching
            
        if len(filtered_spike_times) <= 1:
            return
        
        
        # Plot with wider bars like MATLAB
        ax.plot(bins, autocorr, color='grey', alpha=1, linewidth=2)
        
        # Get mean firing rate from metrics or calculate if not available
        mean_fr = 0
        # Use filtered spike times for firing rate calculation when computeTimeChunks is enabled
        spikes_for_fr = filtered_spike_times if 'filtered_spike_times' in locals() else spike_times
        
        if len(spikes_for_fr) > 1:
            # Calculate firing rate from filtered spike times
            edge = int(len(autocorr) * 1/5)
            mean_fr = np.mean(np.append(autocorr[:edge], autocorr[-edge:]))
            # Set limits to accommodate both data and mean firing rate line
            ax.set_xlim(- cwin / 2, cwin / 2)
            if len(autocorr) > 0:
                data_max = np.max(autocorr)
                # More conservative y-limits - just ensure mean firing rate is visible
                y_max = max(data_max * 1.05, mean_fr * 1.1)
                ax.set_ylim(0, y_max)
            else:
                y_max = mean_fr * 1.1 if mean_fr > 0 else 10
                ax.set_ylim(0, y_max)
            
            # Calculate correct tauR using RPV_window_index
            tau_r = None
            if (param and 'tauR_valuesMin' in param and 'tauR_valuesMax' in param and 
                'tauR_valuesStep' in param and 'RPV_window_index' in metrics):
                try:
                    tau_r_min = param['tauR_valuesMin'] * 1000  # Convert to ms
                    tau_r_max = param['tauR_valuesMax'] * 1000  # Convert to ms
                    tau_r_step = param['tauR_valuesStep'] * 1000  # Convert to ms
                    rpv_index = int(metrics['RPV_window_index'])
                    
                    # Calculate tauR array and get the correct value
                    tau_r_array = np.arange(tau_r_min, tau_r_max + tau_r_step, tau_r_step)
                    if 0 <= rpv_index < len(tau_r_array):
                        tau_r = tau_r_array[rpv_index]
                except (KeyError, ValueError, IndexError):
                    pass
            
            # Fallback to estimatedTauR if RPV_window_index method fails
            if tau_r is None:
                tau_r = metrics.get('tauR_estimated', None)
                if tau_r is None:
                    tau_r = metrics.get('estimatedTauR', None)
                if tau_r is None:
                    tau_r = 2.0  # 2ms default refractory period
                    
            if tau_r is not None:
                ax.axvline(tau_r, color='red', linewidth=3, linestyle='--', alpha=1.0, 
                          label=f'τR = {tau_r:.1f}ms', zorder=10)
            
            # Add mean firing rate horizontal line
            if mean_fr > 0:
                ax.axhline(mean_fr, color='magenta', linewidth=3, linestyle='--', alpha=1.0, 
                          label=f'Mean FR = {mean_fr:.1f} sp/s', zorder=5)  # Lower zorder so legend appears above
                
        ax.set_title('Auto-correlogram', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        ax.set_xlabel('Time (ms)', fontsize=13, fontfamily="DejaVu Sans")
        ax.set_ylabel('Firing rate (sp/s)', fontsize=13, fontfamily="DejaVu Sans")
        ax.tick_params(labelsize=13)
        
        # Add tauR range visualization if min/max tauR are different
        range_text_for_legend = None
        if param and 'tauR_valuesMin' in param and 'tauR_valuesMax' in param:
            tau_r_min_ms = param['tauR_valuesMin'] * 1000  # Convert to ms
            tau_r_max_ms = param['tauR_valuesMax'] * 1000  # Convert to ms
            
            # Check if min and max are different (indicating a range)
            if tau_r_min_ms != tau_r_max_ms:
                # Add grey horizontal arrow and dotted vertical lines for range
                y_pos = ax.get_ylim()[1] * 0.95  # Position near top
                
                # Add dotted vertical lines at range boundaries
                ax.axvline(tau_r_min_ms, color='grey', linewidth=1, linestyle=':', alpha=0.7, zorder=5)
                ax.axvline(tau_r_max_ms, color='grey', linewidth=1, linestyle=':', alpha=0.7, zorder=5)
                
                # Add horizontal arrow between the lines
                ax.annotate('', xy=(tau_r_max_ms, y_pos), xytext=(tau_r_min_ms, y_pos),
                           arrowprops=dict(arrowstyle='<->', color='grey', lw=2, alpha=0.7),
                           zorder=5)
                
                # Prepare range text for legend
                range_text_for_legend = f"τR range: {tau_r_min_ms:.1f}-{tau_r_max_ms:.1f} ms"
        
        # Add legend in top right corner with range information
        handles, labels = ax.get_legend_handles_labels()
        if range_text_for_legend:
            # Add range text to legend labels
            labels.append(range_text_for_legend)
            # Create a dummy handle for the range text (invisible line)
            import matplotlib.lines as mlines
            range_handle = mlines.Line2D([], [], color='grey', alpha=0.7, linestyle='-')
            handles.append(range_handle)
        
        if handles:
            legend = ax.legend(handles, labels, loc='upper right', fontsize=11, framealpha=0.9,
                             prop={'family': 'DejaVu Sans'})
            legend.set_zorder(15)  # Set zorder after creation to appear above all lines
        
        # Add quality metrics text
        InteractiveUnitQualityGUI.add_metrics_text(ax, unit_data, compiled_variables, 'acg')
        
    @staticmethod
    def plot_spatial_decay(ax, unit_data, compiled_variables):
        """Plot spatial decay like MATLAB - only nearby channels"""
        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)

        metrics = unit_data['metrics']
        
        # Check if spatial decay metrics are available
        if 'spatialDecaySlope' in metrics and not pd.isna(metrics['spatialDecaySlope']):
            max_ch = int(metrics.get('maxChannels', 0))
            template = unit_data['template']
            
            if template.size > 0 and len(template.shape) > 1:
                # Get only nearby channels for spatial decay (like MATLAB)
                nearby_channels = InteractiveUnitQualityGUI.get_nearby_channels_for_spatial_decay(max_ch, template.shape[1], compiled_variables)
                
                if 'channel_positions' in ephys_data and len(ephys_data['channel_positions']) > max_ch:
                    positions = ephys_data['channel_positions']
                    peak_pos = positions[max_ch]
                    
                    distances = []
                    amplitudes = []
                    
                    # Calculate amplitude and distance for nearby channels only
                    for ch in nearby_channels:
                        if ch < len(positions) and ch < template.shape[1]:
                            dist = np.sqrt(np.sum((positions[ch] - peak_pos)**2))
                            amp = np.max(np.abs(template[:, ch]))
                            distances.append(dist)
                            amplitudes.append(amp)
                    
                    if len(distances) > 0:
                        distances = np.array(distances)
                        amplitudes = np.array(amplitudes)
                        
                        # Normalize amplitudes
                        max_amp = np.max(amplitudes)
                        if max_amp > 0:
                            amplitudes = amplitudes / max_amp
                            
                            # Plot spatial decay points
                            ax.scatter(distances, amplitudes, s=30, alpha=0.8, color='darkcyan', edgecolor='black')
                            
                            # Fit line (linear in log space for exponential)
                            valid_idx = (distances > 0) & (amplitudes > 0.05)
                            if np.sum(valid_idx) > 1:
                                x_fit = distances[valid_idx]
                                y_fit = amplitudes[valid_idx]
                                
                                # Log-linear fit for exponential decay
                                log_y = np.log(y_fit + 1e-10)  # Avoid log(0)
                                
                                # Linear fit in log space
                                coeffs = np.polyfit(x_fit, log_y, 1)
                                
                                # Plot fitted line - extend beyond data range for visibility
                                x_max = np.max(distances)
                                x_smooth = np.linspace(-x_max*0.1, x_max*1.1, 100)  # Extend slightly beyond data
                                y_smooth = np.exp(np.polyval(coeffs, x_smooth))
                                # Only plot where y values are reasonable
                                valid_y = (y_smooth > 0) & (y_smooth < 10)  # Avoid extreme values
                                ax.plot(x_smooth[valid_y], y_smooth[valid_y], color='darkslateblue', linewidth=2, alpha=0.8)
                            
                            ax.set_xlabel('Distance (μm)', fontsize=13, fontfamily="DejaVu Sans")
                            ax.set_ylabel('Scaling factor', fontsize=13, fontfamily="DejaVu Sans")
                            ax.tick_params(labelsize=13)
                            
                            # Set y-limits to show all data AND fitted line with generous padding
                            data_y_min = np.min(amplitudes)
                            data_y_max = np.max(amplitudes)
                            
                            # Consider fitted line values if they exist
                            if 'y_smooth' in locals() and 'valid_y' in locals():
                                line_y_min = np.min(y_smooth[valid_y]) if np.any(valid_y) else data_y_min
                                line_y_max = np.max(y_smooth[valid_y]) if np.any(valid_y) else data_y_max
                                
                                combined_y_min = min(data_y_min, line_y_min)
                                combined_y_max = max(data_y_max, line_y_max)
                            else:
                                combined_y_min = data_y_min
                                combined_y_max = data_y_max
                            
                            # Add generous padding (20% of range)
                            y_range = combined_y_max - combined_y_min
                            padding = max(0.2 * y_range, 0.1)  # At least 10% padding
                            
                            y_min = combined_y_min - padding
                            y_max = combined_y_max + padding
                            
                            # Ensure minimum range for visibility
                            if y_max - y_min < 0.5:
                                center = (y_max + y_min) / 2
                                y_min = center - 0.25
                                y_max = center + 0.25
                            
                            ax.set_ylim([y_min, y_max])
                        else:
                            ax.text(0.5, 0.5, 'Spatial decay\n(no signal)', 
                                    ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.text(0.5, 0.5, 'Spatial decay\n(no nearby channels)', 
                                ha='center', va='center', transform=ax.transAxes)
                else:
                    ax.text(0.5, 0.5, 'Spatial decay\n(no channel positions)', 
                            ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'Spatial decay\n(no template data)', 
                        ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Spatial decay\n(not computed)', 
                    ha='center', va='center', transform=ax.transAxes, fontfamily="DejaVu Sans")
                    
        ax.set_title('Spatial decay', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        
        # Add quality metrics text
        InteractiveUnitQualityGUI.add_metrics_text(ax, unit_data, compiled_variables, 'spatial')
    
    @staticmethod
    def plot_amplitudes_over_time(ax, unit_data, compiled_variables):
        """Plot amplitudes over time with firing rate below and presence ratio indicators"""
        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)

        spike_times = unit_data['spike_times']
        metrics = unit_data['metrics']
        
        if len(spike_times) > 0:
            # Get amplitudes if available
            unit_id = unit_data['unit_id']
            spike_mask = ephys_data['spike_clusters'] == unit_id
            
            # Calculate time bins for presence ratio and firing rate
            total_duration = np.max(spike_times) - np.min(spike_times)
            n_bins = max(20, int(total_duration / 60))  # ~1 minute bins, minimum 20 bins
            time_bins = np.linspace(np.min(spike_times), np.max(spike_times), n_bins + 1)
            bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
            bin_width = time_bins[1] - time_bins[0]
            
            # Calculate firing rate per bin
            bin_counts, _ = np.histogram(spike_times, bins=time_bins)
            firing_rates = bin_counts / bin_width
            
            
            if 'template_amplitudes' in ephys_data:
                amplitudes = ephys_data['template_amplitudes'][spike_mask]
                
                # Color spikes based on goodTimeChunks if computeTimeChunks is enabled
                spike_colors = np.full(len(spike_times), 'darkorange')  # Default: bad chunks (orange)
                
                if param and param.get('computeTimeChunks', False):
                    # Use useTheseTimesStart and useTheseTimesStop from metrics
                    good_start_times = metrics.get('useTheseTimesStart', None)
                    good_stop_times = metrics.get('useTheseTimesStop', None)
                    
                    if good_start_times is not None and good_stop_times is not None:
                        # Ensure they are arrays
                        if np.isscalar(good_start_times):
                            good_start_times = [good_start_times]
                        if np.isscalar(good_stop_times):
                            good_stop_times = [good_stop_times]
                        
                        # Color spikes green if they fall within any good time chunk
                        for start_time, stop_time in zip(good_start_times, good_stop_times):
                            if not (np.isnan(start_time) or np.isnan(stop_time)):
                                good_spike_mask = (spike_times >= start_time) & (spike_times <= stop_time)
                                spike_colors[good_spike_mask] = 'green'  # Good chunks: green
                
                # Plot amplitudes with colored dots based on good/bad time chunks
                unique_colors = np.unique(spike_colors)
                for color in unique_colors:
                    color_mask = spike_colors == color
                    ax.scatter(spike_times[color_mask], amplitudes[color_mask], 
                              s=3, alpha=0.6, c=color, edgecolors='none')
                ax.set_ylabel('Scaling factor', color='blue', fontsize=13, fontfamily="DejaVu Sans")
                ax.tick_params(labelsize=13)
                
                # Create twin axis for firing rate
                ax2 = ax.twinx()
                
                # Plot firing rate as step plot (outline only)
                ax2.step(bin_centers, firing_rates, where='mid', color='magenta', 
                        linewidth=2.5, alpha=0.8, label='Firing rate')
                ax2.set_ylabel('Firing rate (sp/s)', color='magenta', fontsize=13, fontfamily="DejaVu Sans")
                ax2.tick_params(labelsize=13)
                ax2.tick_params(axis='y', labelcolor='magenta')
                
                # Add drift plot if computeDrift is enabled
                if (param and param.get('computeDrift', False) and 
                    gui_data is not None):
                    
                    unit_idx = unit_data["unit_idx"]
                    
                    # Check if per_bin_metrics exists and contains drift data
                    if 'per_bin_metrics' in gui_data:
                        per_bin_metrics = gui_data['per_bin_metrics']
                        
                        # Check if unit has per-bin metrics data
                        if unit_idx in per_bin_metrics:
                            unit_metrics = per_bin_metrics[unit_idx]
                            
                            # Look for drift data in unit metrics
                            if isinstance(unit_metrics, dict) and 'drift' in unit_metrics:
                                drift_data = unit_metrics['drift']
                                
                                # Use pre-computed drift data only - never compute in GUI
                                if (isinstance(drift_data, dict) and 
                                    'time_bins' in drift_data and 
                                    'median_spike_depth_per_bin' in drift_data):
                                    
                                    drift_time_bins = drift_data['time_bins']
                                    drift_values = drift_data['median_spike_depth_per_bin']
                                    
                                    # Check if pre-computed bins match expected bin size - only plot if they match
                                    expected_bin_size = param.get('driftBinSize', 60)  # Default 60 seconds
                                    plot_drift = False
                                    
                                    if len(drift_time_bins) > 1:
                                        actual_bin_size = drift_time_bins[1] - drift_time_bins[0]
                                        # Allow some tolerance for floating point differences
                                        if abs(actual_bin_size - expected_bin_size) <= 1.0:
                                            plot_drift = True
                                        else:
                                            print(f"Drift not plotted: Pre-computed bin size ({actual_bin_size:.1f}s) differs from param.driftBinSize ({expected_bin_size}s)")
                                    
                                    # Only plot if drift data matches expected bin size
                                    if plot_drift and len(drift_values) > 0:
                                        drift_bin_centers = (drift_time_bins[:-1] + drift_time_bins[1:]) / 2
                                        
                                        # Create third axis for drift
                                        ax3 = ax.twinx()
                                        ax3.spines['right'].set_position(('outward', 80))
                                        
                                        # Plot drift as step plot
                                        ax3.step(drift_bin_centers, drift_values, where='mid', 
                                                color='lightpink', linewidth=2.5, alpha=0.9, label='Drift')
                                        ax3.set_ylabel('Drift (μm)', color='lightpink', fontsize=13, fontfamily="DejaVu Sans")
                                        ax3.tick_params(axis='y', labelcolor='lightpink', labelsize=13)
                
                
            else:
                # Color spikes based on goodTimeChunks if computeTimeChunks is enabled (fallback - no amplitudes)
                y_pos = np.ones_like(spike_times)
                spike_colors = np.full(len(spike_times), 'darkorange')  # Default: bad chunks (orange)
                
                if param and param.get('computeTimeChunks', False):
                    # Use useTheseTimesStart and useTheseTimesStop from metrics
                    good_start_times = metrics.get('useTheseTimesStart', None)
                    good_stop_times = metrics.get('useTheseTimesStop', None)
                    
                    if good_start_times is not None and good_stop_times is not None:
                        # Ensure they are arrays
                        if np.isscalar(good_start_times):
                            good_start_times = [good_start_times]
                        if np.isscalar(good_stop_times):
                            good_stop_times = [good_stop_times]
                        
                        # Color spikes green if they fall within any good time chunk
                        for start_time, stop_time in zip(good_start_times, good_stop_times):
                            if not (np.isnan(start_time) or np.isnan(stop_time)):
                                good_spike_mask = (spike_times >= start_time) & (spike_times <= stop_time)
                                spike_colors[good_spike_mask] = 'green'  # Good chunks: green
                
                # Plot spike times as raster with colored dots based on good/bad time chunks
                unique_colors = np.unique(spike_colors)
                for color in unique_colors:
                    color_mask = spike_colors == color
                    ax.scatter(spike_times[color_mask], y_pos[color_mask], 
                              s=3, alpha=0.6, c=color, edgecolors='none')
                ax.set_ylabel('Spikes', color='blue', fontsize=13, fontfamily="DejaVu Sans")
                ax.tick_params(labelsize=13)
                
                # Create twin axis for firing rate  
                ax2 = ax.twinx()
                ax2.step(bin_centers, firing_rates, where='mid', color='magenta', 
                        linewidth=2.5, alpha=0.8, label='Firing rate')
                ax2.set_ylabel('Firing rate (sp/s)', color='magenta', fontsize=13, fontfamily="DejaVu Sans")
                ax2.tick_params(labelsize=13)
                ax2.tick_params(axis='y', labelcolor='magenta')
                
                # Add drift plot if computeDrift is enabled
                if (param and param.get('computeDrift', False) and 
                    gui_data is not None):
                    
                    unit_idx = unit_data["unit_idx"]
                    
                    # Check if per_bin_metrics exists and contains drift data
                    if 'per_bin_metrics' in gui_data:
                        per_bin_metrics = gui_data['per_bin_metrics']
                        
                        # Check if unit has per-bin metrics data
                        if unit_idx in per_bin_metrics:
                            unit_metrics = per_bin_metrics[unit_idx]
                            
                            # Look for drift data in unit metrics
                            if isinstance(unit_metrics, dict) and 'drift' in unit_metrics:
                                drift_data = unit_metrics['drift']
                                
                                # Use pre-computed drift data only - never compute in GUI
                                if (isinstance(drift_data, dict) and 
                                    'time_bins' in drift_data and 
                                    'median_spike_depth_per_bin' in drift_data):
                                    
                                    drift_time_bins = drift_data['time_bins']
                                    drift_values = drift_data['median_spike_depth_per_bin']
                                    
                                    # Check if pre-computed bins match expected bin size - only plot if they match
                                    expected_bin_size = param.get('driftBinSize', 60)  # Default 60 seconds
                                    plot_drift = False
                                    
                                    if len(drift_time_bins) > 1:
                                        actual_bin_size = drift_time_bins[1] - drift_time_bins[0]
                                        # Allow some tolerance for floating point differences
                                        if abs(actual_bin_size - expected_bin_size) <= 1.0:
                                            plot_drift = True
                                        else:
                                            print(f"Drift not plotted: Pre-computed bin size ({actual_bin_size:.1f}s) differs from param.driftBinSize ({expected_bin_size}s)")
                                    
                                    # Only plot if drift data matches expected bin size
                                    if plot_drift and len(drift_values) > 0:
                                        drift_bin_centers = (drift_time_bins[:-1] + drift_time_bins[1:]) / 2
                                        
                                        # Create third axis for drift
                                        ax3 = ax.twinx()
                                        ax3.spines['right'].set_position(('outward', 80))
                                        
                                        # Plot drift as step plot
                                        ax3.step(drift_bin_centers, drift_values, where='mid', 
                                                color='lightpink', linewidth=2.5, alpha=0.9, label='Drift')
                                        ax3.set_ylabel('Drift (μm)', color='lightpink', fontsize=13, fontfamily="DejaVu Sans")
                                        ax3.tick_params(axis='y', labelcolor='lightpink', labelsize=13)
                
                # Add subtle time bin indicators to amplitude plot
                for bin_edge in time_bins:
                    ax.axvline(bin_edge, color='gray', alpha=0.2, linewidth=0.3, linestyle='--', zorder=0)
                
        ax.set_title('Amplitude scaling factor over time', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        # Remove x-axis labels since time bin plot below will show them
        ax.set_xlabel('')
        ax.tick_params(labelsize=13, labelbottom=False)  # Hide x-axis labels
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Store y-limits for amplitude fit plot consistency
        _amplitude_ylim = ax.get_ylim()
        compiled_variables['_amplitude_ylim'] = _amplitude_ylim
        
        # Spike count is now displayed as part of add_metrics_text
        # self._add_spike_count_quality_test(ax, unit_data, time_bins, bin_counts)
        
        # Add legend for time chunk coloring and drift if enabled
        import matplotlib.lines as mlines
        legend_elements = []
        
        # Add time chunk legend elements if computeTimeChunks is enabled
        if param and param.get('computeTimeChunks', False):
            legend_elements.extend([
                mlines.Line2D([], [], color='green', marker='o', linestyle='None', 
                             markersize=6, label='Spikes in good time chunks'),
                mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None', 
                             markersize=6, label='Spikes in MUA time chunks')
            ])
        
        # Add drift and firing rate legend elements
        legend_elements.append(
            mlines.Line2D([], [], color='magenta', linestyle='-', linewidth=2.5, 
                         label='Firing rate')
        )
        
        current_unit_idx = unit_data["unit_idx"]

        # Add drift if computeDrift is enabled and drift data exists
        if (param and param.get('computeDrift', False) and 
            gui_data is not None and
            'per_bin_metrics' in gui_data and 
            current_unit_idx in gui_data['per_bin_metrics'] and
            'drift' in gui_data['per_bin_metrics'][current_unit_idx]):
            
            # Add clarification if both drift and time chunks are computed
            if param.get('computeTimeChunks', False):
                drift_label = 'Drift (good time chunks only)'
            else:
                drift_label = 'Drift'
                
            legend_elements.append(
                mlines.Line2D([], [], color='lightpink', linestyle='-', linewidth=2.5, 
                             label=drift_label)
            )
        
        # Add legend below the plot with visible background if we have elements
        if legend_elements:
            ncols = min(len(legend_elements), 4)  # Max 4 columns
            legend = ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.05), 
                             loc='upper center', ncol=ncols, fontsize=13,
                             framealpha=0.8, facecolor='white', edgecolor='black', 
                             prop={'family': 'DejaVu Sans'})
            legend.set_zorder(15)  # Ensure legend appears above plot elements
        
        # Add quality metrics text
        InteractiveUnitQualityGUI.add_metrics_text(ax, unit_data, compiled_variables, 'amplitude')
    
    def _add_spike_count_quality_test(self, ax, unit_data, time_bins, bin_counts):
        """Add spike count quality metric test overlay to amplitude scaling plot"""
        metrics = unit_data['metrics']
        
        # Get spike count threshold from parameters
        min_spikes = self.param.get('minNumSpikes', 300) if self.param else 300
        total_spikes = len(unit_data['spike_times'])
        
        # Determine if unit passes spike count test
        passes_spike_test = total_spikes >= min_spikes
        
        # Choose color based on classification
        if passes_spike_test:
            test_color = 'green'
        else:
            test_color = 'darkorange'  # MUA color for failing test
        
        # Simple text format without threshold comparison
        test_text = f"# spikes = {total_spikes}"
        
        # Add quality test text below presence ratio text (0.95 - 2*0.12 = 0.71)
        ax.text(0.98, 0.71, test_text, 
                transform=ax.transAxes, 
                fontsize=12, 
                fontweight='bold',
                color=test_color,
                ha='right', va='top',
                zorder=10)
    
    @staticmethod
    def plot_time_bin_metrics(ax, unit_data, compiled_variables):
        """Plot time bin metrics: presence ratio, RPV rate, and percentage spikes missing"""
        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)

        spike_times = unit_data['spike_times']
        metrics = unit_data['metrics']
        unit_id = unit_data['unit_id']
        
        if len(spike_times) > 0:
            # Try to get saved per-bin data from GUI precomputation
            per_bin_data = None
            if gui_data is not None:
                per_bin_data = gui_data.get('per_bin_metrics', {}).get(unit_id)
            
            if per_bin_data and param.get('computeTimeChunks', False):
                # Use actual per-bin quality metrics data 
                
                # Get RPV data
                rpv_data = per_bin_data.get('rpv')
                if rpv_data and 'time_bins' in rpv_data and 'fraction_RPVs_per_bin' in rpv_data:
                    time_bins = rpv_data['time_bins']
                    # Adjust time bins to start at first spike time
                    if len(spike_times) > 0:
                        first_spike_time = np.min(spike_times)
                        if time_bins[0] < first_spike_time:
                            # Find the first bin that includes/after the first spike
                            first_bin_idx = np.searchsorted(time_bins, first_spike_time) - 1
                            first_bin_idx = max(0, first_bin_idx)  # Ensure non-negative
                            time_bins = time_bins[first_bin_idx:]
                    
                    bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
                    # Use the estimated tau_R index for plotting (usually the middle value)
                    rpv_matrix = rpv_data['fraction_RPVs_per_bin']  # Shape: (n_time_chunks, n_tauR)
                    if rpv_matrix.shape[1] > 0:
                        # Use middle tau_R value or estimated index if available
                        tau_r_idx = rpv_matrix.shape[1] // 2
                        if param is not None and 'RPV_tauR_estimate' in metrics:
                            try:
                                tau_r_idx = int(metrics['RPV_window_index'])
                            except:
                                pass
                        rpv_rates = rpv_matrix[:, tau_r_idx]
                        # Adjust rpv_rates to match truncated time_bins if needed
                        if len(spike_times) > 0:
                            first_spike_time = np.min(spike_times)
                            if rpv_data['time_bins'][0] < first_spike_time:
                                first_bin_idx = np.searchsorted(rpv_data['time_bins'], first_spike_time) - 1
                                first_bin_idx = max(0, first_bin_idx)
                                rpv_rates = rpv_rates[first_bin_idx:len(time_bins)-1]  # Match bin_centers length
                    else:
                        rpv_rates = np.zeros(len(bin_centers))
                else:
                    rpv_rates = None
                
                # Get percentage missing data
                perc_missing_data = per_bin_data.get('perc_missing')
                if perc_missing_data and 'percent_missing_gaussian_per_bin' in perc_missing_data:
                    perc_missing = perc_missing_data['percent_missing_gaussian_per_bin']
                    # Adjust perc_missing to match truncated time_bins if needed
                    if len(spike_times) > 0:
                        first_spike_time = np.min(spike_times)
                        if rpv_data and 'time_bins' in rpv_data and rpv_data['time_bins'][0] < first_spike_time:
                            first_bin_idx = np.searchsorted(rpv_data['time_bins'], first_spike_time) - 1
                            first_bin_idx = max(0, first_bin_idx)
                            perc_missing = perc_missing[first_bin_idx:len(time_bins)-1]  # Match bin_centers length
                else:
                    perc_missing = None
                
                # Calculate presence ratio per time chunk using proper algorithm
                presence_ratio_bin_size = param.get('presenceRatioBinSize', 60)  # Default 60 seconds
                presence_ratios = []
                
                for i in range(len(time_bins) - 1):
                    chunk_start = time_bins[i]
                    chunk_end = time_bins[i + 1]
                    chunk_spike_times = spike_times[(spike_times >= chunk_start) & (spike_times < chunk_end)]
                    
                    if len(chunk_spike_times) > 0:
                        # Create presence ratio bins within this time chunk
                        presence_bins = np.arange(chunk_start, chunk_end, presence_ratio_bin_size)
                        if len(presence_bins) < 2:
                            # If chunk is smaller than presence ratio bin size, use the chunk itself
                            presence_bins = np.array([chunk_start, chunk_end])
                        
                        # Count spikes per presence ratio bin
                        spikes_per_bin = np.array([
                            np.sum((chunk_spike_times >= presence_bins[j]) & 
                                   (chunk_spike_times < presence_bins[j + 1]))
                            for j in range(len(presence_bins) - 1)
                        ])
                        
                        # Calculate presence ratio: fraction of bins with spikes >= 5% of 90th percentile
                        if len(spikes_per_bin) > 0 and np.max(spikes_per_bin) > 0:
                            threshold = 0.05 * np.percentile(spikes_per_bin, 90)
                            full_bins = (spikes_per_bin >= threshold).astype(int)
                            chunk_presence_ratio = full_bins.sum() / len(full_bins)
                        else:
                            chunk_presence_ratio = 0.0
                    else:
                        chunk_presence_ratio = 0.0
                    
                    presence_ratios.append(chunk_presence_ratio)
                
                presence_ratio = np.array(presence_ratios)
                
                # Use useTheseTimesStart and useTheseTimesStop to determine good time chunks
                good_start_times = metrics.get('useTheseTimesStart', None)
                good_stop_times = metrics.get('useTheseTimesStop', None)
                
                # Add background coloring: orange by default, green for good time chunks
                for i, (start_time, end_time) in enumerate(zip(time_bins[:-1], time_bins[1:])):
                    # Default: orange background for bad chunks
                    chunk_color = 'darkorange'
                    
                    # Check if this time bin overlaps with any good time chunk
                    if good_start_times is not None and good_stop_times is not None:
                        # Ensure they are arrays
                        if np.isscalar(good_start_times):
                            good_start_times = [good_start_times]
                        if np.isscalar(good_stop_times):
                            good_stop_times = [good_stop_times]
                        
                        # Check overlap with any good time chunk
                        for g_start, g_stop in zip(good_start_times, good_stop_times):
                            if not (np.isnan(g_start) or np.isnan(g_stop)):
                                # Check if time bin overlaps with good chunk
                                if start_time < g_stop and end_time > g_start:
                                    chunk_color = 'lightgreen'
                                    break
                    
                    ax.axvspan(start_time, end_time, color=chunk_color, alpha=0.3, zorder=0)
                
                # Plot actual quality metrics as step plots
                if rpv_rates is not None:
                    ax.step(bin_centers, rpv_rates, where='mid', color='purple', linewidth=2, label='RPV rate', alpha=0.8)
                
                if perc_missing is not None:
                    ax.step(bin_centers, perc_missing / 100, where='mid', color='darkorange', linewidth=2, label='% missing (1=100%)', alpha=0.8)
                
                # Plot presence ratio step line on top
                ax.step(bin_centers, presence_ratio, where='mid', color='forestgreen', linewidth=2, label='Presence ratio', alpha=0.8)
                
                # Add time bin indicators
                for i, bin_edge in enumerate(time_bins[:-1]):
                    ax.axvline(bin_edge, color='gray', alpha=0.3, linewidth=0.5, linestyle='--')
                
                # Add final bin edge
                ax.axvline(time_bins[-1], color='gray', alpha=0.3, linewidth=0.5, linestyle='--')
                
                # Add arrows and dotted lines for good chunk ranges using useTheseTimesStart/Stop
                if good_start_times is not None and good_stop_times is not None:
                    y_max = ax.get_ylim()[1]
                    arrow_y = y_max * 0.9
                    
                    # Ensure they are arrays
                    if np.isscalar(good_start_times):
                        good_start_times = [good_start_times]
                    if np.isscalar(good_stop_times):
                        good_stop_times = [good_stop_times]
                    
                    for g_start, g_stop in zip(good_start_times, good_stop_times):
                        if not (np.isnan(g_start) or np.isnan(g_stop)):
                            # Dotted vertical lines at start and end
                            ax.axvline(g_start, color='darkgreen', linewidth=2, linestyle=':', alpha=0.8, zorder=5)
                            ax.axvline(g_stop, color='darkgreen', linewidth=2, linestyle=':', alpha=0.8, zorder=5)
                            
                            # Arrow between them
                            ax.annotate('', xy=(g_stop, arrow_y), xytext=(g_start, arrow_y),
                                       arrowprops=dict(arrowstyle='<->', color='darkgreen', lw=2, alpha=0.8),
                                       zorder=5)
                            
                            # Label for good chunks
                            mid_time = (g_start + g_stop) / 2
                            ax.text(mid_time, arrow_y + y_max * 0.05, 'Good', 
                                   ha='center', va='bottom', color='darkgreen', fontweight='bold',
                                   fontsize=13, alpha=0.9, zorder=5)
                
            else:
                # Fallback: compute simplified metrics on the fly
                total_duration = np.max(spike_times) - np.min(spike_times)
                n_bins = max(20, int(total_duration / 60))  # ~1 minute bins, minimum 20 bins
                time_bins = np.linspace(np.min(spike_times), np.max(spike_times), n_bins + 1)
                bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
                bin_width = time_bins[1] - time_bins[0]
                
                # Calculate firing rate and presence ratio per bin
                bin_counts, _ = np.histogram(spike_times, bins=time_bins)
                
                # Calculate presence ratio per time bin using proper algorithm
                presence_ratio_bin_size = param.get('presenceRatioBinSize', 60)  # Default 60 seconds
                presence_ratios = []
                
                for i in range(len(time_bins) - 1):
                    bin_start = time_bins[i]
                    bin_end = time_bins[i + 1]
                    bin_spike_times = spike_times[(spike_times >= bin_start) & (spike_times < bin_end)]
                    
                    if len(bin_spike_times) > 0:
                        # Create presence ratio bins within this time bin
                        presence_bins = np.arange(bin_start, bin_end, presence_ratio_bin_size)
                        if len(presence_bins) < 2:
                            # If bin is smaller than presence ratio bin size, use the bin itself
                            presence_bins = np.array([bin_start, bin_end])
                        
                        # Count spikes per presence ratio bin
                        spikes_per_bin = np.array([
                            np.sum((bin_spike_times >= presence_bins[j]) & 
                                   (bin_spike_times < presence_bins[j + 1]))
                            for j in range(len(presence_bins) - 1)
                        ])
                        
                        # Calculate presence ratio: fraction of bins with spikes >= 5% of 90th percentile
                        if len(spikes_per_bin) > 0 and np.max(spikes_per_bin) > 0:
                            threshold = 0.05 * np.percentile(spikes_per_bin, 90)
                            full_bins = (spikes_per_bin >= threshold).astype(int)
                            bin_presence_ratio = full_bins.sum() / len(full_bins)
                        else:
                            bin_presence_ratio = 0.0
                    else:
                        bin_presence_ratio = 0.0
                    
                    presence_ratios.append(bin_presence_ratio)
                
                presence_ratio = np.array(presence_ratios)
                
                # Plot presence ratio as step plot
                ax.step(bin_centers, presence_ratio, where='mid', color='forestgreen', linewidth=2, label='Presence ratio', alpha=0.8)
                
                # Add time bin indicators (fallback case - no per-bin quality data)
                for bin_edge in time_bins:
                    ax.axvline(bin_edge, color='gray', alpha=0.3, linewidth=0.5, linestyle='--')
            
            # Formatting for tiny plot
            ax.set_xlabel('Time (s)', fontsize=13, fontfamily="DejaVu Sans")
            ax.set_ylabel('Metrics', fontsize=13, fontfamily="DejaVu Sans")
            ax.tick_params(labelsize=13)
            ax.set_ylim(0, 1.1)  # Standard scale for all metrics
            
            # Add legend at the top of the plot
            ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=3, fontsize=13,
                     framealpha=0.9, prop={'family': 'DejaVu Sans'})
            
            # Make plot as compact as possible
            ax.margins(y=0.05)
            ax.spines['top'].set_visible(False)  # Remove top border for cleaner look
        else:
            ax.text(0.5, 0.5, 'No spikes\nfor bin analysis', ha='center', va='center', 
                   transform=ax.transAxes, fontfamily="DejaVu Sans", fontsize=13)
            ax.set_xlabel('Time (s)', fontsize=13, fontfamily="DejaVu Sans")
            ax.set_ylabel('Metrics', fontsize=13, fontfamily="DejaVu Sans")
            ax.tick_params(labelsize=13)
    
    @staticmethod
    def plot_unit_location(ax, unit_data, compiled_variables):
        """Plot all units by depth vs log firing rate, colored by classification"""
        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)

        # Define classification colors
        classification_colors = {
            'good': [0, 0.7, 0],        # Green
            'mua': [1, 0.7, 0.2],         # dark Orange  
            'noise': [1, 0, 0],         # Red
            'non-somatic': [0, 0, 1]    # Blue
        }
        
        if 'channel_positions' in ephys_data and 'maxChannels' in quality_metrics:
            positions = ephys_data['channel_positions']
            max_channels = quality_metrics['maxChannels']
            
            # Get all unit classifications and firing rates
            all_units = []
            all_depths = []
            all_firing_rates = []
            all_colors = []
            
            for i, unit_id in enumerate(unique_units):
                # Get max channel for this unit
                if i < len(max_channels):
                    max_ch = int(max_channels[i])
                    if max_ch < len(positions):
                        # Use Y position as depth - deeper channels have higher index, should be at bottom
                        depth = positions[max_ch, 1]  # Keep original - deeper = lower y values
                        
                        # Calculate firing rate for this unit
                        unit_spike_mask = ephys_data['spike_clusters'] == unit_id
                        unit_spike_times = ephys_data['spike_times'][unit_spike_mask]
                        
                        if len(unit_spike_times) > 0:
                            duration = np.max(unit_spike_times) - np.min(unit_spike_times)
                            if duration > 0:
                                firing_rate = len(unit_spike_times) / duration
                                
                                # Get BombCell classification for dot colors
                                if bombcell_unit_types is not None and i < len(bombcell_unit_types):
                                    unit_type = bombcell_unit_types[i]
                                    # Map numeric codes to classification names
                                    type_map = {
                                        0: 'noise',
                                        1: 'good', 
                                        2: 'mua',
                                        3: 'non-somatic',
                                        4: 'non-somatic'
                                    }
                                    classification = type_map.get(unit_type, 'good')
                                else:
                                    classification = 'good'  # Default
                                
                                all_units.append(unit_id)
                                all_depths.append(depth)
                                all_firing_rates.append(max(firing_rate, 0.01))  # Avoid log(0)
                                all_colors.append(classification_colors.get(classification, [0, 0, 0]))
            
            if len(all_units) > 0:
                all_depths = np.array(all_depths)
                all_firing_rates = np.array(all_firing_rates)
                log_firing_rates = np.log10(all_firing_rates)
                
                # Plot all units
                for i, (depth, log_fr, color, unit_id) in enumerate(zip(all_depths, log_firing_rates, all_colors, all_units)):
                    is_current = unit_id == unit_data['unit_id']
                    
                    if is_current:
                        # Current unit: larger with black outline
                        ax.scatter(log_fr, depth, c=[color], s=80, edgecolors='black', 
                                 linewidths=2, zorder=10)
                    else:
                        # Other units: smaller, no outline
                        ax.scatter(log_fr, depth, c=[color], s=30, alpha=0.7, zorder=5)
                
                ax.set_xlabel('Log₁₀ firing rate (sp/s)', fontsize=13, fontfamily="DejaVu Sans")
                ax.set_ylabel('Depth from tip of probe (μm)', fontsize=13, fontfamily="DejaVu Sans")
                ax.tick_params(labelsize=13)
                # Y-axis is now flipped via data transformation (no invert_yaxis needed)
                
                # Add depth arrow on the left side
                ylim = ax.get_ylim()
                xlim = ax.get_xlim()
                x_range = xlim[1] - xlim[0]
                arrow_x = xlim[0] - x_range * 1.2  # Much more to the left
                
                # Draw arrow spanning most of the plot height (slightly shorter)
                y_range = ylim[1] - ylim[0]
                arrow_start_y = ylim[0] + y_range * 0.05  # Start slightly above bottom
                arrow_end_y = ylim[1] - y_range * 0.05    # End slightly below top
                
                ax.annotate('', xy=(arrow_x, arrow_end_y), xytext=(arrow_x, arrow_start_y),
                           arrowprops=dict(arrowstyle='<->', color='black', lw=2),
                           annotation_clip=False)
                
                # Add labels below the arrow and to the left
                label_x = arrow_x - x_range * 0.02  # To the left of arrow
                
                ax.text(label_x, arrow_start_y - y_range * 0.01, 'deepest = tip \n of the probe', 
                       ha='center', va='top', fontsize=16, fontfamily="DejaVu Sans",
                       rotation=0, clip_on=False, fontweight='bold')
                       
                ax.text(label_x, arrow_end_y + y_range * 0.02, 'most superficial', 
                       ha='center', va='bottom', fontsize=16, fontfamily="DejaVu Sans",
                       rotation=0, clip_on=False, fontweight='bold')
                
                # Add legend
                legend_elements = []
                for class_name, color in classification_colors.items():
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=color, markersize=8, 
                                                    label=class_name))
                ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=13)
                
                if unit_slider is not None:
                    # Add click interactivity to navigate to units
                    def on_location_click(event):
                        if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                            # Find the closest unit to the click
                            click_x, click_y = event.xdata, event.ydata
                            min_distance = float('inf')
                            closest_unit_idx = None
                            
                            # Get current axis limits for normalization
                            xlims = ax.get_xlim()
                            ylims = ax.get_ylim()
                            
                            for i, (unit_id, log_fr, depth) in enumerate(zip(all_units, all_firing_rates, all_depths)):
                                # Calculate distance in data coordinates (simpler approach)
                                dx = (log_fr - click_x) / (xlims[1] - xlims[0])  # Normalize by axis range
                                dy = (depth - click_y) / (ylims[1] - ylims[0])   # Normalize by axis range
                                
                                distance = np.sqrt(dx**2 + dy**2)
                                if distance < min_distance:
                                    min_distance = distance
                                    closest_unit_idx = list(unique_units).index(unit_id)
                            
                            # Navigate to the closest unit if click is close enough
                            if min_distance < 0.1 and closest_unit_idx is not None:  # 10% of normalized plot area
                                current_unit_idx = closest_unit_idx
                                unit_slider.value = closest_unit_idx
                                print(f"Clicked on unit {closest_unit_idx} (unit_id: {unique_units[closest_unit_idx]})")
                    
                    # Store the click handler and connect it
                    _location_click_handler = on_location_click
                    ax.figure.canvas.mpl_connect('button_press_event', _location_click_handler)
                
            else:
                ax.text(0.5, 0.5, 'No units with\nvalid locations', 
                        ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Unit locations\n(requires probe geometry\nand max channels)', 
                    ha='center', va='center', transform=ax.transAxes, fontfamily="DejaVu Sans")
        
        ax.set_title('Units by depth', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")

    @staticmethod
    def plot_amplitude_fit(ax, unit_data, compiled_variables):
        """Plot amplitude distribution with cutoff Gaussian fit like BombCell"""
        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)

        spike_times = unit_data['spike_times']
        metrics = unit_data['metrics']
        
        # Get y-limits from amplitude plot for consistency
        amp_ylim = None
        if '_amplitude_ylim' in compiled_variables:
            amp_ylim = compiled_variables['_amplitude_ylim']
        
        if len(spike_times) > 0:
            # Get amplitudes if available
            unit_id = unit_data['unit_id']
            spike_mask = ephys_data['spike_clusters'] == unit_id
            
            if 'template_amplitudes' in ephys_data:
                amplitudes = ephys_data['template_amplitudes'][spike_mask]
                
                # Filter to good time chunks if computeTimeChunks is enabled
                if param and param.get('computeTimeChunks', False):
                    good_start_times = metrics.get('useTheseTimesStart', None)
                    good_stop_times = metrics.get('useTheseTimesStop', None)
                    
                    if good_start_times is not None and good_stop_times is not None:
                        # Ensure they are arrays
                        if np.isscalar(good_start_times):
                            good_start_times = [good_start_times]
                        if np.isscalar(good_stop_times):
                            good_stop_times = [good_stop_times]
                        
                        # Create mask for spikes in good time chunks
                        good_spike_mask = np.zeros(len(spike_times), dtype=bool)
                        for g_start, g_stop in zip(good_start_times, good_stop_times):
                            if not (np.isnan(g_start) or np.isnan(g_stop)):
                                good_spike_mask |= (spike_times >= g_start) & (spike_times <= g_stop)
                        
                        # Filter amplitudes to only good time chunks
                        amplitudes = amplitudes[good_spike_mask]
                
                if len(amplitudes) > 10:  # Need sufficient data for fit
                    # Create histogram with count (not density) like BombCell
                    n_bins = min(50, int(len(amplitudes) / 10))  # Adaptive bin count
                    hist_counts, bin_edges = np.histogram(amplitudes, bins=n_bins)
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    bin_width = bin_edges[1] - bin_edges[0]
                    
                    # Plot horizontal histogram like BombCell
                    ax.barh(bin_centers, hist_counts, height=bin_width*0.8, 
                           facecolor='grey', edgecolor='black')
                    
                    # Fit cutoff Gaussian like BombCell
                    try:
                        from scipy.optimize import curve_fit
                        from scipy.stats import norm
                        
                        def gaussian_cut(x, a, x0, sigma, xcut):
                            """Cutoff Gaussian function from BombCell"""
                            g = a * np.exp(-(x - x0)**2 / (2 * sigma**2))
                            g[x < xcut] = 0
                            return g
                        
                        # Initial parameters like BombCell
                        p0 = [
                            np.max(hist_counts),  # Height
                            np.median(amplitudes),  # Center
                            np.std(amplitudes),   # Width  
                            np.min(amplitudes)    # Cutoff
                        ]
                        
                        # Bounds like BombCell
                        bounds = (
                            [0, np.min(amplitudes), 0, np.min(amplitudes)],  # Lower bounds
                            [np.inf, np.max(amplitudes), np.ptp(amplitudes), np.median(amplitudes)]  # Upper bounds
                        )
                        
                        # Fit the cutoff Gaussian
                        try:
                            popt, _ = curve_fit(gaussian_cut, bin_centers, hist_counts, 
                                              p0=p0, bounds=bounds, maxfev=5000)
                            
                            # Generate smooth curve for fit
                            y_smooth = np.linspace(np.min(amplitudes), np.max(amplitudes), 200)
                            x_smooth = gaussian_cut(y_smooth, *popt)
                            
                            # Plot fit in red like BombCell
                            ax.plot(x_smooth, y_smooth, 'r-', linewidth=2)
                            
                            # Calculate percentage missing using BombCell formula
                            # norm_area_ndtr = normcdf((center - cutoff)/width)
                            norm_area_ndtr = norm.cdf((popt[1] - popt[3]) / popt[2])
                            percent_missing = 100 * (1 - norm_area_ndtr)
                            
                            # Display percentage like BombCell
                            ax.text(0.5, 0.98, f'{percent_missing:.1f}', 
                                   transform=ax.transAxes, va='top', ha='center',
                                   color=[0.7, 0.7, 0.7], fontsize=13, weight='bold')
                            
                        except Exception as e:
                            # Fallback to simple stats
                            ax.text(0.5, 0.5, 'Fit failed', 
                                   ha='center', va='center', transform=ax.transAxes)
                            
                    except ImportError:
                        ax.text(0.5, 0.5, 'SciPy required\nfor fitting', 
                               ha='center', va='center', transform=ax.transAxes)
                    
                    ax.set_xlabel('count', fontsize=13, fontfamily="DejaVu Sans")
                    ax.set_ylabel('Scaling factor', fontsize=13, fontfamily="DejaVu Sans")
                    ax.tick_params(labelsize=13)
                    
                    # Set y-limits to match amplitude plot if available
                    if amp_ylim is not None:
                        ax.set_ylim(amp_ylim)
                        
                else:
                    ax.text(0.5, 0.5, 'Insufficient data\nfor scaling factor fit', 
                            ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No scaling factor data\navailable', 
                        ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No spike data\navailable', 
                    ha='center', va='center', transform=ax.transAxes, fontfamily="DejaVu Sans")
                    
        ax.set_title('Scaling factor \n distribution', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        
        # Add quality metrics text
        InteractiveUnitQualityGUI.add_metrics_text(ax, unit_data, compiled_variables, 'amplitude_fit')
        
    @staticmethod
    def add_metrics_text(ax, unit_data, compiled_variables, plot_type):
        """Add quality metrics text overlay to plots like MATLAB with color coding"""
        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)

        metrics = unit_data['metrics']
        
        def format_metric(value, decimals=2):
            if value is None or (hasattr(value, '__len__') and len(value) == 0):
                return 'N/A'
            try:
                if np.isnan(value):
                    return 'N/A'
                return f"{value:.{decimals}f}"
            except (TypeError, ValueError):
                return 'N/A'
        
        def get_metric_color(metric_name, value, param):
            """Determine color based on metric value and thresholds - simplified logic"""
            if value == 'N/A' or not param:
                return 'black'
            
            try:
                val = float(value.split()[0])  # Extract numeric value
            except (ValueError, AttributeError):
                return 'black'
            
            # Noise metrics: red if noise, black if not noise
            if metric_name in ['nPeaks', 'nTroughs', 'waveformDuration_peakTrough', 'waveformBaselineFlatness', 'spatialDecaySlope', 'scndPeakToTroughRatio']:
                if metric_name == 'nPeaks':
                    max_peaks = param.get('maxNPeaks', 2)
                    return 'red' if val > max_peaks else 'black'
                elif metric_name == 'nTroughs':
                    max_troughs = param.get('maxNTroughs', 2) 
                    return 'red' if val > max_troughs else 'black'
                elif metric_name == 'waveformDuration_peakTrough':
                    min_dur = param.get('minWvDuration', 0.3)
                    max_dur = param.get('maxWvDuration', 1.0)
                    return 'red' if (val < min_dur or val > max_dur) else 'black'
                elif metric_name == 'waveformBaselineFlatness':
                    max_baseline = param.get('maxWvBaselineFraction', 0.3)
                    return 'red' if val > max_baseline else 'black'
                elif metric_name == 'spatialDecaySlope':
                    min_slope = param.get('minSpatialDecaySlope', 0.001)
                    return 'red' if val < min_slope else 'black'
                elif metric_name == 'scndPeakToTroughRatio':
                    max_ratio = param.get('maxScndPeakToTroughRatio_noise', 0.5)
                    return 'red' if val > max_ratio else 'black'
            
            # Non-somatic metrics: blue if non-somatic, black if somatic
            elif metric_name in ['mainPeakToTroughRatio', 'peak1ToPeak2Ratio']:
                if metric_name == 'mainPeakToTroughRatio':
                    max_ratio = param.get('maxMainPeakToTroughRatio_nonSomatic', 2.0)
                    return 'blue' if val > max_ratio else 'black'
                elif metric_name == 'peak1ToPeak2Ratio':
                    max_ratio = param.get('maxPeak1ToPeak2Ratio_nonSomatic', 0.5)
                    return 'blue' if val > max_ratio else 'black'
            
            # MUA metrics: orange if MUA, green if good
            elif metric_name in ['rawAmplitude', 'signalToNoiseRatio', 'fractionRPVs_estimatedTauR', 'presenceRatio', 'maxDriftEstimate', 'percentageSpikesMissing_gaussian', 'numSpikes']:
                if metric_name == 'rawAmplitude':
                    min_amp = param.get('minAmplitude', 50)
                    return  'darkorange' if val < min_amp else 'green'
                elif metric_name == 'signalToNoiseRatio':
                    minSNR = param.get('minSNR', 3)
                    return  'darkorange' if val < minSNR else 'green'
                elif metric_name == 'fractionRPVs_estimatedTauR':
                    max_rpv = param.get('maxRPVviolations', 0.1)
                    return  'darkorange' if val > max_rpv else 'green'
                elif metric_name == 'presenceRatio':
                    min_presence = param.get('minPresenceRatio', 0.7)
                    return  'darkorange' if val < min_presence else 'green'
                elif metric_name == 'maxDriftEstimate':
                    max_drift = param.get('maxDrift', 100)
                    return  'darkorange' if val > max_drift else 'green'
                elif metric_name == 'percentageSpikesMissing_gaussian':
                    max_missing = param.get('maxPercSpikesMissing', 20)
                    return  'darkorange' if val > max_missing else 'green'
                elif metric_name == 'numSpikes':
                    min_spikes = param.get('minNumSpikes', 300)
                    return  'darkorange' if val < min_spikes else 'green'
            
            # Default for informational metrics
            else:
                return 'black'
        
        # Different metrics for different plot types
        metric_info = []
        if plot_type == 'template':
            metric_info = [
                ('nPeaks', f"nPeaks: {format_metric(metrics.get('nPeaks'), 0)}"),
                ('nTroughs', f"nTroughs: {format_metric(metrics.get('nTroughs'), 0)}"),
                ('waveformDuration_peakTrough', f"Duration: {format_metric(metrics.get('waveformDuration_peakTrough'), 1)} ms"),
                ('scndPeakToTroughRatio', f"Peak2/Trough: {format_metric(metrics.get('scndPeakToTroughRatio'), 2)}"),
                ('waveformBaselineFlatness', f"Baseline: {format_metric(metrics.get('waveformBaselineFlatness'), 3)}"),
                ('peak1ToPeak2Ratio', f"Peak1/Peak2: {format_metric(metrics.get('peak1ToPeak2Ratio'), 2)}"),
                ('mainPeakToTroughRatio', f"Main P/T: {format_metric(metrics.get('mainPeakToTroughRatio'), 2)}")
            ]
        elif plot_type == 'raw':
            metric_info = [
                ('rawAmplitude', f"Raw Ampl: {format_metric(metrics.get('rawAmplitude'), 1)} μV"),
                ('signalToNoiseRatio', f"SNR: {format_metric(metrics.get('signalToNoiseRatio'), 1)}")
            ]
        elif plot_type == 'spatial':
            metric_info = [
                ('spatialDecaySlope', f"Spatial decay: {format_metric(metrics.get('spatialDecaySlope'), 3)}")
            ]
        elif plot_type == 'acg':
            metric_info = [
                ('fractionRPVs_estimatedTauR', f"RPV rate: {format_metric(metrics.get('fractionRPVs_estimatedTauR'), 4)}")
            ]
        elif plot_type == 'amplitude':
            # Add spike count to metrics
            total_spikes = len(unit_data.get('spike_times', []))
            metric_info = [
                ('maxDriftEstimate', f"Max drift: {format_metric(metrics.get('maxDriftEstimate'), 1)} μm"),
                ('presenceRatio', f"Presence ratio: {format_metric(metrics.get('presenceRatio'), 3)}"),
                ('numSpikes', f"# spikes = {total_spikes}")
            ]
        elif plot_type == 'amplitude_fit':
            metric_info = [
                ('percentageSpikesMissing_gaussian', f"% missing: {format_metric(metrics.get('percentageSpikesMissing_gaussian'), 1)}%")
            ]
            
        # Add colored text to plot with proper spacing
        if metric_info:
            y_start = 0.15 if plot_type == 'acg' else 0.95  # Bottom for ACG, top for others
            line_height = 0.12  # Much larger spacing to prevent overlaps
            
            for i, (metric_name, text) in enumerate(metric_info):
                color = get_metric_color(metric_name, format_metric(metrics.get(metric_name)), param)
                y_pos = y_start - i * line_height
                
                # Skip if text would go below plot area
                if y_pos < 0.05:
                    break
                    
                ax.text(0.98, y_pos, text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right', fontsize=15, 
                       color=color, weight='bold', fontfamily="DejaVu Sans",
                       bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, 
                                edgecolor='lightgray', linewidth=1), zorder=20)
        
    def get_nearest_channels(self, peak_channel, n_channels, n_to_get=16):
        """Get nearest channels to peak channel like MATLAB - prioritize above/below"""
        if 'channel_positions' in self.ephys_data:
            positions = self.ephys_data['channel_positions']
            if len(positions) > peak_channel:
                peak_pos = positions[peak_channel]
                
                # For neural probes, prioritize channels above and below (y-direction)
                # Calculate distances with preference for vertical neighbors
                channels_with_scores = []
                
                for ch in range(min(n_channels, len(positions))):
                    pos_diff = positions[ch] - peak_pos
                    
                    # Calculate distance
                    dist = np.sqrt(np.sum(pos_diff**2))
                    
                    # Give preference to channels above/below (smaller x difference)
                    x_diff = abs(pos_diff[0]) if len(pos_diff) > 0 else 0
                    y_diff = abs(pos_diff[1]) if len(pos_diff) > 1 else 0
                    
                    # Priority score: prefer small x differences (same column) and small overall distance
                    if x_diff < 20:  # Same column or very close
                        priority_score = dist  # Use distance as tie-breaker
                    else:
                        priority_score = dist + x_diff  # Penalize horizontal distance
                    
                    channels_with_scores.append((priority_score, ch))
                
                # Sort by priority score and take nearest n_to_get
                channels_with_scores.sort()
                nearest_channels = [ch for _, ch in channels_with_scores[:n_to_get]]
                return nearest_channels
        
        # Fallback: take channels above and below peak (sequential channels)
        half_range = n_to_get // 2
        start = max(0, peak_channel - half_range)
        end = min(n_channels, peak_channel + half_range)
        channels = list(range(start, end))
        
        # Pad to get exactly n_to_get channels if possible
        while len(channels) < n_to_get and len(channels) < n_channels:
            if start > 0:
                start -= 1
                channels.insert(0, start)
            elif end < n_channels:
                channels.append(end)
                end += 1
            else:
                break
                
        return channels[:n_to_get]
    
    def arrange_channels_for_display(self, channels, peak_channel):
        """Arrange channels for display to reflect probe geometry - peak in center"""
        if not channels:
            return channels
            
        # Try to put peak channel in center of 4x4 grid (position 5 or 6, 9 or 10)
        # and arrange others by their relationship to peak
        
        if 'channel_positions' in self.ephys_data and len(self.ephys_data['channel_positions']) > peak_channel:
            positions = self.ephys_data['channel_positions']
            peak_pos = positions[peak_channel]
            
            # Separate channels into above, below, and sides relative to peak
            above_channels = []
            below_channels = []
            left_channels = []
            right_channels = []
            peak_found = False
            
            for ch in channels:
                if ch == peak_channel:
                    peak_found = True
                    continue
                    
                if ch < len(positions):
                    pos_diff = positions[ch] - peak_pos
                    x_diff = pos_diff[0] if len(pos_diff) > 0 else 0
                    y_diff = pos_diff[1] if len(pos_diff) > 1 else 0
                    
                    # Classify by primary direction
                    if abs(y_diff) > abs(x_diff):  # Primarily vertical
                        if y_diff > 0:
                            above_channels.append(ch)
                        else:
                            below_channels.append(ch)
                    else:  # Primarily horizontal
                        if x_diff > 0:
                            right_channels.append(ch)
                        else:
                            left_channels.append(ch)
            
            # Sort each group by distance from peak
            def sort_by_distance(ch_list):
                distances = []
                for ch in ch_list:
                    if ch < len(positions):
                        dist = np.sqrt(np.sum((positions[ch] - peak_pos)**2))
                        distances.append((dist, ch))
                distances.sort()
                return [ch for _, ch in distances]
            
            above_channels = sort_by_distance(above_channels)
            below_channels = sort_by_distance(below_channels)
            left_channels = sort_by_distance(left_channels)
            right_channels = sort_by_distance(right_channels)
            
            # Arrange in 4x4 grid with peak near center
            arranged = [None] * 16
            
            # Place peak channel at position 5 (row 1, col 1)
            if peak_found:
                arranged[5] = peak_channel
            
            # Fill above (row 0)
            for i, ch in enumerate(above_channels[:4]):
                arranged[i] = ch
                
            # Fill below (rows 2-3)
            below_positions = [8, 9, 10, 11, 12, 13, 14, 15]
            for i, ch in enumerate(below_channels[:8]):
                if i < len(below_positions):
                    arranged[below_positions[i]] = ch
            
            # Fill sides around peak
            side_positions = [4, 6, 7]  # Left and right of peak
            side_channels = left_channels + right_channels
            for i, ch in enumerate(side_channels[:3]):
                arranged[side_positions[i]] = ch
            
            # Fill any remaining positions with remaining channels
            remaining_channels = [ch for ch in channels if ch not in arranged]
            empty_positions = [i for i, ch in enumerate(arranged) if ch is None]
            
            for i, pos in enumerate(empty_positions):
                if i < len(remaining_channels):
                    arranged[pos] = remaining_channels[i]
            
            # Remove None values and return
            return [ch for ch in arranged if ch is not None]
        
        else:
            # Fallback: arrange sequentially with peak in center
            channels_sorted = sorted(channels)
            arranged = []
            
            # Try to put peak channel in center
            if peak_channel in channels_sorted:
                peak_idx = channels_sorted.index(peak_channel)
                # Reorder to put peak near position 5-6
                mid_point = len(channels_sorted) // 2
                if peak_idx != mid_point:
                    channels_sorted[peak_idx], channels_sorted[mid_point] = channels_sorted[mid_point], channels_sorted[peak_idx]
            
            return channels_sorted
    
    @staticmethod
    def mark_peaks_and_troughs(ax, unit_data, compiled_variables, waveform, x_offset, y_offset, metrics, amp_range):
        """Mark all peaks and troughs on waveform with duration line"""

        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)
        current_unit_idx = unit_data["unit_idx"]
        
        if (gui_data and 
            'peak_locations' in gui_data and 
            'trough_locations' in gui_data and
            current_unit_idx in gui_data['peak_locations']):
            
            peaks = list(gui_data['peak_locations'][current_unit_idx])
            troughs = list(gui_data['trough_locations'][current_unit_idx])
            
            
        else:
            # Fallback to real-time computation
            try:
                from scipy.signal import find_peaks
                
                # More stringent peak detection to match visual perception
                waveform_range = np.max(waveform) - np.min(waveform)
                
                # Find all peaks (positive deflections) - use higher threshold and prominence
                peak_height_threshold = np.max(waveform) * 0.5  # Increased from 0.2 to 0.5
                peak_prominence = waveform_range * 0.1  # Require 10% of waveform range prominence
                peaks, peak_properties = find_peaks(waveform, 
                                                   height=peak_height_threshold, 
                                                   distance=10,  # Increased from 5
                                                   prominence=peak_prominence)
                
                # Find all troughs (negative deflections) 
                trough_height_threshold = -np.min(waveform) * 0.5
                trough_prominence = waveform_range * 0.1
                troughs, trough_properties = find_peaks(-waveform, 
                                                       height=trough_height_threshold, 
                                                       distance=10,
                                                       prominence=trough_prominence)
            except ImportError:
                # Fallback without scipy - find multiple troughs using simple approach
                if len(waveform) > 0:
                    # Find all local minima as potential troughs
                    troughs = []
                    # Look for local minima (points lower than neighbors)
                    for i in range(1, len(waveform) - 1):
                        if (waveform[i] < waveform[i-1] and 
                            waveform[i] < waveform[i+1] and 
                            waveform[i] < np.min(waveform) * 0.8):  # Must be at least 80% of minimum
                            troughs.append(i)
                    
                    # If no troughs found, use the global minimum
                    if len(troughs) == 0:
                        troughs = [np.argmin(waveform)]
                    
                    # Find peaks similarly
                    peaks = []
                    for i in range(1, len(waveform) - 1):
                        if (waveform[i] > waveform[i-1] and 
                            waveform[i] > waveform[i+1] and 
                            waveform[i] > np.max(waveform) * 0.8):  # Must be at least 80% of maximum
                            peaks.append(i)
                    
                    # If no peaks found, use the global maximum
                    if len(peaks) == 0:
                        peaks = [np.argmax(waveform)]
                else:
                    peaks = []
                    troughs = []
        
        # Plot peaks and troughs (works for both pre-computed and real-time data)
        # Find main peak and trough for special highlighting and duration calculation
        main_peak_idx = None
        main_trough_idx = None
        
        
        # Use pre-computed duration indices from quality metrics if available
        if (gui_data and 
            'peak_loc_for_duration' in gui_data and 
            'trough_loc_for_duration' in gui_data and
            current_unit_idx in gui_data['peak_loc_for_duration']):
            
            main_peak_idx = gui_data['peak_loc_for_duration'][current_unit_idx]
            main_trough_idx = gui_data['trough_loc_for_duration'][current_unit_idx]
            
        else:
            # Fallback: Use largest absolute values among detected peaks/troughs
            # Main peak: peak with largest absolute value among detected peaks
            if len(peaks) > 0:
                peak_values = waveform[peaks]
                peak_abs_values = np.abs(peak_values)
                main_peak_idx = peaks[np.argmax(peak_abs_values)]
            else:
                # Fallback: absolute maximum
                main_peak_idx = np.argmax(np.abs(waveform))
            
            # Main trough: trough with largest absolute value among detected troughs
            if len(troughs) > 0:
                trough_values = waveform[troughs]  # These are the actual negative values
                trough_abs_values = np.abs(trough_values)  # Convert to absolute values
                main_trough_idx = troughs[np.argmax(trough_abs_values)]  # Largest absolute = deepest trough
            else:
                # Fallback: absolute minimum  
                min_idx = np.argmin(waveform)
                max_idx = np.argmax(waveform)
                # Choose the one with larger absolute value
                if abs(waveform[min_idx]) > abs(waveform[max_idx]):
                    main_trough_idx = min_idx
                else:
                    main_trough_idx = max_idx
        
        # Ensure indices are valid integers
        if main_peak_idx is not None:
            main_peak_idx = int(main_peak_idx)
        if main_trough_idx is not None:
            main_trough_idx = int(main_trough_idx)
        
        # Plot all peaks with red dots
        legend_elements = []
        for i, peak_idx in enumerate(peaks):
            if peak_idx == main_peak_idx:
                # Main peak: larger, darker marker
                ax.plot(peak_idx + x_offset, waveform[peak_idx] + y_offset, 'ro', 
                       markersize=12, markeredgecolor='darkred', markeredgewidth=2, 
                       markerfacecolor='red', zorder=11, label='Main peak' if i == 0 else "")
            else:
                # Regular peaks: smaller, lighter markers
                ax.plot(peak_idx + x_offset, waveform[peak_idx] + y_offset, 'ro', 
                       markersize=8, markeredgecolor='red', markeredgewidth=1, 
                       markerfacecolor='lightcoral', alpha=0.8, zorder=10, 
                       label='Peaks' if i == 0 and peak_idx != main_peak_idx else "")
        
        # Plot all troughs with blue dots (ensure all detected troughs are shown)
        for i, trough_idx in enumerate(troughs):
            if trough_idx == main_trough_idx:
                # Main trough: larger, darker marker
                ax.plot(trough_idx + x_offset, waveform[trough_idx] + y_offset, 'bo', 
                       markersize=12, markeredgecolor='darkblue', markeredgewidth=2, 
                       markerfacecolor='blue', zorder=11, label='Main trough' if i == 0 else "")
            else:
                # Regular troughs: smaller, lighter markers
                ax.plot(trough_idx + x_offset, waveform[trough_idx] + y_offset, 'bo', 
                       markersize=8, markeredgecolor='blue', markeredgewidth=1, 
                       markerfacecolor='lightblue', alpha=0.8, zorder=10,
                       label='Troughs' if i == 0 and trough_idx != main_trough_idx else "")
        
        # Draw horizontal duration line from main peak to main trough
        if main_peak_idx is not None and main_trough_idx is not None:
            # Since the y-axis is inverted, we need to position the duration line appropriately
            # The waveform data is inverted (negative of original), and the y-axis display is inverted
            # So we want the duration line to appear below the waveform when viewed
            
            waveform_min = np.min(waveform) + y_offset  # Most negative value (appears at top due to inversion)
            waveform_max = np.max(waveform) + y_offset  # Most positive value (appears at bottom due to inversion)
            waveform_range = waveform_max - waveform_min
            
            # Since y-axis is inverted, "below" the waveform means higher y-values
            # Position line above the maximum value (which appears below due to inversion)
            line_y = waveform_max + waveform_range * 0.2  # 20% "below" the visual waveform
            
            peak_x = main_peak_idx + x_offset
            trough_x = main_trough_idx + x_offset
            peak_y = waveform[main_peak_idx] + y_offset
            trough_y = waveform[main_trough_idx] + y_offset
            
            # Draw all three lines with distinct styles and proper visibility
            # Use purple color as requested
            duration_color = '#8A2BE2'  # Purple - highly visible
            
            # 1. Horizontal duration line (solid line) - add to legend
            ax.plot([peak_x, trough_x], [line_y, line_y], 
                   color=duration_color, linewidth=3, solid_capstyle='round', zorder=20, label='Duration')
            
            # 2. Vertical line from peak to duration line (dashed)
            ax.plot([peak_x, peak_x], [peak_y, line_y], 
                   color=duration_color, linewidth=2, linestyle='--', zorder=19)
            
            # 3. Vertical line from trough to duration line (dashed)
            ax.plot([trough_x, trough_x], [trough_y, line_y], 
                   color=duration_color, linewidth=2, linestyle='--', zorder=19)
        
        # Add legend for peaks/troughs at bottom of plot
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), 
                     ncol=len(handles), fontsize=13, frameon=True, framealpha=0.9,
                     prop={'family': 'DejaVu Sans'})
    
    @staticmethod
    def get_nearby_channels_for_spatial_decay(peak_channel, n_channels, compiled_variables):
        """Get nearby channels for spatial decay plot - fewer points like MATLAB"""
        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)

        if 'channel_positions' in ephys_data:
            positions = ephys_data['channel_positions']
            if len(positions) > peak_channel:
                peak_pos = positions[peak_channel]
                
                # Get channels within reasonable distance (like MATLAB)
                nearby_channels = []
                max_distance = 100  # μm - adjust based on probe geometry
                
                for ch in range(min(n_channels, len(positions))):
                    dist = np.sqrt(np.sum((positions[ch] - peak_pos)**2))
                    if dist <= max_distance:
                        nearby_channels.append(ch)
                
                return nearby_channels
        
        # Fallback: channels within ±5 of peak
        start = max(0, peak_channel - 5)
        end = min(n_channels, peak_channel + 6)
        return list(range(start, end))
    
    @staticmethod
    def plot_histograms_panel(fig, unit_data, compiled_variables):
        """Plot histogram distributions showing where current unit sits - exact copy of plot_functions.py"""
        ephys_data, quality_metrics, ephys_properties, raw_waveforms, param, unit_types, gui_data, n_units, unique_units, bombcell_unit_types, unit_slider = InteractiveUnitQualityGUI.unpack_variables(compiled_variables)

        # Preprocessing - handle inf values using shared utility
        from bombcell.helper_functions import clean_inf_values
        quality_metrics = clean_inf_values(quality_metrics)

        # Define MATLAB-style color matrices - exact copy
        red_colors = np.array([
            [0.8627, 0.0784, 0.2353],  # Crimson
            [1.0000, 0.1412, 0.0000],  # Scarlet
            [0.7255, 0.0000, 0.0000],  # Cherry
            [0.5020, 0.0000, 0.1255],  # Burgundy
            [0.5020, 0.0000, 0.0000],  # Maroon
            [0.8039, 0.3608, 0.3608],  # Indian Red
        ])

        blue_colors = np.array([
            [0.2549, 0.4118, 0.8824],  # Royal Blue
            [0.0000, 0.0000, 0.5020],  # Navy Blue
        ])

        darker_yellow_orange_colors = np.array([
            [0.7843, 0.7843, 0.0000],  # Dark Yellow
            [0.8235, 0.6863, 0.0000],  # Dark Golden Yellow
            [0.8235, 0.5294, 0.0000],  # Dark Orange
            [0.8039, 0.4118, 0.3647],  # Dark Coral
            [0.8235, 0.3176, 0.2275],  # Dark Tangerine
            [0.8235, 0.6157, 0.6510],  # Dark Salmon
            [0.7882, 0.7137, 0.5765],  # Dark Goldenrod
            [0.8235, 0.5137, 0.3922],  # Dark Light Coral
            [0.7569, 0.6196, 0.0000],  # Darker Goldenrod
            [0.8235, 0.4510, 0.0000],  # Darker Orange
        ])

        color_mtx = np.vstack([red_colors, blue_colors, darker_yellow_orange_colors])

        # Define metrics in MATLAB order - exact copy
        metric_names = ['nPeaks', 'nTroughs', 'waveformBaselineFlatness', 'waveformDuration_peakTrough', 
                       'scndPeakToTroughRatio', 'spatialDecaySlope', 'peak1ToPeak2Ratio', 'mainPeakToTroughRatio',
                       'rawAmplitude', 'signalToNoiseRatio', 'fractionRPVs_estimatedTauR', 'nSpikes', 
                       'presenceRatio', 'percentageSpikesMissing_gaussian', 'maxDriftEstimate', 
                       'isolationDistance', 'Lratio']

        metric_names_short = ['# peaks', '# troughs', 'baseline flatness', 'waveform duration',
                             'peak_2/trough', 'spatial decay', 'peak_1/peak_2', 'peak_{main}/trough',
                             'amplitude', 'signal/noise (SNR)', 'refractory period viol. (RPV)', '# spikes',
                             'presence ratio', '% spikes missing', 'maximum drift',
                             'isolation dist.', 'L-ratio']

        # Define thresholds - exact copy
        metric_thresh1 = [param.get('maxNPeaks'), param.get('maxNTroughs'), param.get('maxWvBaselineFraction'),
                         param.get('minWvDuration'), param.get('maxScndPeakToTroughRatio_noise'),
                         param.get('minSpatialDecaySlope') if param.get('spDecayLinFit') else param.get('minSpatialDecaySlopeExp'),
                         param.get('maxPeak1ToPeak2Ratio_nonSomatic'), param.get('maxMainPeakToTroughRatio_nonSomatic'),
                         param.get('minAmplitude'), None, param.get('maxRPVviolations'), None, None, param.get('maxPercSpikesMissing'),
                         param.get('maxDrift'), param.get('isoDmin'), None]

        metric_thresh2 = [None, None, None, param.get('maxWvDuration'), None,
                         None if param.get('spDecayLinFit') else param.get('maxSpatialDecaySlopeExp'),
                         None, None, None, param.get('minSNR'),
                         None, param.get('minNumSpikes'), param.get('minPresenceRatio'), None, None,
                         None, param.get('lratioMax')]

        # Define plot conditions
        plot_conditions = [True, True, True, True, True,
                          param.get('computeSpatialDecay', False),
                          True, True,
                          param.get('extractRaw', False) and 'rawAmplitude' in quality_metrics and np.any(~np.isnan(quality_metrics.get('rawAmplitude', [np.nan]))),
                          param.get('extractRaw', False) and 'signalToNoiseRatio' in quality_metrics and np.any(~np.isnan(quality_metrics.get('signalToNoiseRatio', [np.nan]))),
                          True, True, True, True,
                          param.get('computeDrift', False),
                          param.get('computeDistanceMetrics', False),
                          param.get('computeDistanceMetrics', False)]

        # Define line colors for thresholds (MATLAB style) - exact copy
        metric_line_cols = np.array([
            [0.2, 0.2, 0.2, 1, 0, 0, 0, 0, 0],  # nPeaks
            [0.2, 0.2, 0.2, 1, 0, 0, 0, 0, 0],  # nTroughs
            [0.2, 0.2, 0.2, 1, 0, 0, 1, 0, 0],  # baseline flatness
            [1, 0, 0, 0.2, 0.2, 0.2, 1, 0, 0],  # waveform duration
            [0.2, 0.2, 0.2, 1, 0, 0, 1, 0, 0],  # peak2/trough
            [1, 0, 0, 0.2, 0.2, 0.2, 1, 0, 0],  # spatial decay
            [0.2, 0.2, 0.2, 0.25, 0.41, 0.88, 0, 0, 0],  # peak1/peak2
            [0.2, 0.2, 0.2, 0.25, 0.41, 0.88, 0, 0, 0],  # peak_main/trough
            [1.0, 0.5469, 0, 0, 0.5, 0, 0, 0, 0],  # amplitude
            [1.0, 0.5469, 0, 0, 0.5, 0, 0, 0, 0],  # SNR
            [0, 0.5, 0, 1.0, 0.5469, 0, 0, 0, 0],  # frac RPVs
            [1.0, 0.5469, 0, 0, 0.5, 0, 0, 0, 0],  # nSpikes
            [1.0, 0.5469, 0, 0, 0.5, 0, 0, 0, 0],  # presence ratio
            [0, 0.5, 0, 1.0, 0.5469, 0, 0, 0, 0],  # % spikes missing
            [0, 0.5, 0, 1.0, 0.5469, 0, 0, 0, 0],  # max drift
            [1.0, 0.5469, 0, 0, 0.5, 0, 0, 0, 0],  # isolation dist
            [0, 0.5, 0, 1.0, 0.5469, 0, 0, 0, 0],  # L-ratio
        ])

        # Filter metrics that should be plotted - exact copy
        valid_metrics = []
        valid_colors = []
        valid_labels = []
        valid_thresh1 = []
        valid_thresh2 = []
        valid_line_cols = []
        
        for i, (metric_name, condition) in enumerate(zip(metric_names, plot_conditions)):
            if condition and metric_name in quality_metrics:
                valid_metrics.append(metric_name)
                valid_colors.append(color_mtx[i % len(color_mtx)])
                valid_labels.append(metric_names_short[i])
                valid_thresh1.append(metric_thresh1[i])
                valid_thresh2.append(metric_thresh2[i])
                valid_line_cols.append(metric_line_cols[i])

        # ADAPTIVE LAYOUT based on number of histograms
        num_subplots = len(valid_metrics)
        
        # Use fewer columns, more rows for better readability
        # Fixed large grid size for all cases
        grid_rows = 100  # Very large grid to accommodate tall histograms
        
        if num_subplots <= 4:
            cols = 2  # 2 columns for small number of histograms
            col_width = 6
            col_start_positions = [16, 24]  # 2 wide columns
        else:
            cols = 3  # Maximum 3 columns, even for many histograms
            col_width = 4
            col_start_positions = [16, 21, 26]  # 3 columns with good spacing
        
        # Calculate how many rows of plots we need
        rows_of_plots = (num_subplots + cols - 1) // cols
        
        # Simple fixed heights - use most of the 100-row grid
        spacing_gap = 5  # Fixed gap between histograms
        
        if rows_of_plots == 1:
            plot_height = 85  # Even taller
            plot_positions = [8]
        elif rows_of_plots == 2:
            plot_height = 40  # Taller histograms
            plot_positions = [3, 50]
        elif rows_of_plots == 3:
            plot_height = 28  # Taller
            plot_positions = [3, 35, 67]
        elif rows_of_plots == 4:
            plot_height = 20  # Taller
            plot_positions = [3, 26, 49, 72]
        else:
            # Many rows - distribute evenly
            plot_height = max(12, (grid_rows - 10) // (rows_of_plots + 1))  # Taller minimum
            plot_positions = []
            current_pos = 3
            for i in range(rows_of_plots):
                plot_positions.append(current_pos)
                current_pos += plot_height + spacing_gap
        
        # Create histogram subplots
        for i, metric_name in enumerate(valid_metrics):
            row_id = i // cols
            col_id = i % cols
            
            # Use the pre-calculated positions for even distribution
            if row_id < len(plot_positions):
                start_row = plot_positions[row_id]
                start_col = col_start_positions[col_id]
                
                # ALL plots same height - no extending last row
                actual_height = plot_height
                
                ax = plt.subplot2grid((grid_rows, 30), (start_row, start_col), rowspan=actual_height, colspan=col_width)
            else:
                continue
            
            metric_data = quality_metrics[metric_name]
            metric_data = metric_data[~np.isnan(metric_data)]
            
            if len(metric_data) > 0:
                # Plot histogram with probability normalization (MATLAB style) - exact copy
                if metric_name in ['nPeaks', 'nTroughs']:
                    # Use integer bins for discrete metrics
                    bins = np.arange(np.min(metric_data), np.max(metric_data) + 2) - 0.5
                elif metric_name == 'waveformDuration_peakTrough':
                    # Use fewer bins for waveform duration like MATLAB
                    bins = 20
                else:
                    bins = 40
                    
                n, bins_out, patches = ax.hist(metric_data, bins=bins, density=True, 
                                             color=valid_colors[i], alpha=0.7)
                
                # Convert to probability (like MATLAB's 'Normalization', 'probability')
                if metric_name not in ['nPeaks', 'nTroughs']:
                    bin_width = bins_out[1] - bins_out[0]
                    for patch in patches:
                        patch.set_height(patch.get_height() * bin_width)
                
                # Add current unit highlighting with ARROW instead of line
                current_unit_idx = unit_data["unit_idx"]
                if current_unit_idx < len(quality_metrics[metric_name]):
                    current_value = quality_metrics[metric_name][current_unit_idx]
                    if not np.isnan(current_value):
                        # NO red bin highlighting - removed for cleaner look
                        
                        # Add triangle above histogram (revert to original position)
                        bin_idx = np.digitize(current_value, bins_out) - 1
                        bin_height = patches[bin_idx].get_height() if 0 <= bin_idx < len(patches) else 0.5
                        triangle_y = bin_height + 0.15  # Above the histogram bars
                        
                        # Large black triangle pointing down with white contour for visibility
                        ax.scatter(current_value, triangle_y, marker='v', s=500, color='black', 
                                  alpha=1.0, zorder=15, edgecolors='white', linewidths=4)
                
                # Add threshold lines above histogram at 0.9 - MUCH MORE EXTENDED x-limits for text
                x_lim = ax.get_xlim()
                # Extend x-axis MUCH MORE for text labels
                x_range = x_lim[1] - x_lim[0]
                if metric_name in ['waveformDuration_peakTrough', 'spatialDecaySlope']:
                    # MASSIVE extra space for these metrics that need room for "Noise" text
                    ax.set_xlim([x_lim[0] - 0.6*x_range, x_lim[1] + 0.6*x_range])
                else:
                    # More space for all other metrics
                    ax.set_xlim([x_lim[0] - 0.3*x_range, x_lim[1] + 0.3*x_range])
                x_lim = ax.get_xlim()
                line_y = 0.9  # Position lines at 0.9
                
                thresh1 = valid_thresh1[i]
                thresh2 = valid_thresh2[i]
                line_colors = valid_line_cols[i].reshape(3, 3)
                
                # Calculate binsize offset for accurate threshold positioning - 0.5 * bin width
                if metric_name in ['nPeaks', 'nTroughs']:
                    binsize_offset = 0.5  # 0.5 * 1.0 (since bin width is 1 for integers)
                else:
                    binsize_offset = (bins_out[1] - bins_out[0]) / 2 if len(bins_out) > 1 else 0  # 0.5 * bin_width
                
                # Threshold logic - APPLY OFFSET for accurate positioning
                if thresh1 is not None or thresh2 is not None:
                    if thresh1 is not None and thresh2 is not None:
                        # Add vertical lines for thresholds at value + 0.5*bin_width
                        ax.axvline(thresh1 + binsize_offset, color='k', linewidth=2)
                        ax.axvline(thresh2 + binsize_offset, color='k', linewidth=2)
                        # Add horizontal colored lines at value + 0.5*bin_width
                        thresh1_offset = thresh1 + binsize_offset
                        thresh2_offset = thresh2 + binsize_offset
                        ax.plot([x_lim[0], thresh1_offset], 
                               [line_y, line_y], color=line_colors[0], linewidth=6)
                        ax.plot([thresh1_offset, thresh2_offset], 
                               [line_y, line_y], color=line_colors[1], linewidth=6)
                        ax.plot([thresh2_offset, x_lim[1]], 
                               [line_y, line_y], color=line_colors[2], linewidth=6)
                        
                        # Add classification labels with arrows - using OFFSET thresholds
                        midpoint1 = (x_lim[0] + thresh1_offset) / 2
                        midpoint2 = (thresh1_offset + thresh2_offset) / 2
                        midpoint3 = (thresh2_offset + x_lim[1]) / 2
                        text_y = 0.95  # Position text at 0.95
                        
                        # Determine metric type based on metric name
                        noise_metrics = ['nPeaks', 'nTroughs', 'waveformBaselineFlatness', 'waveformDuration_peakTrough', 'scndPeakToTroughRatio', 'spatialDecaySlope']
                        nonsomatic_metrics = ['peak1ToPeak2Ratio', 'mainPeakToTroughRatio']
                        
                        if metric_name in noise_metrics:
                            # Noise metrics: both thresholds -> Noise, Neuronal, Noise
                            ax.text(midpoint1, text_y, '  Noise', ha='center', fontsize=16, 
                                   color=line_colors[0], weight='bold')
                            ax.text(midpoint2, text_y, '  Neuronal', ha='center', fontsize=16, 
                                   color=line_colors[1], weight='bold')
                            ax.text(midpoint3, text_y, '  Noise', ha='center', fontsize=16, 
                                   color=line_colors[2], weight='bold')
                        elif metric_name in nonsomatic_metrics:
                            # Non-somatic metrics: both thresholds -> Non-somatic, Somatic, Non-somatic
                            ax.text(midpoint1, text_y, '  Non-somatic', ha='center', fontsize=16, 
                                   color=line_colors[0], weight='bold')
                            ax.text(midpoint2, text_y, '  Somatic', ha='center', fontsize=16, 
                                   color=line_colors[1], weight='bold')
                            ax.text(midpoint3, text_y, '  Non-somatic', ha='center', fontsize=16, 
                                   color=line_colors[2], weight='bold')
                        else:
                            # For metrics where higher values = better quality (like nSpikes, presenceRatio)
                            # Left should be MUA, middle Good, right MUA, but we need to check the metric
                            good_higher_metrics = ['nSpikes', 'presenceRatio', 'rawAmplitude', 'isolationDistance']
                            if metric_name in good_higher_metrics:
                                # Higher values = Good, so Good should be on the right
                                ax.text(midpoint1, text_y, '  MUA', ha='center', fontsize=16, 
                                       color=line_colors[0], weight='bold')
                                ax.text(midpoint2, text_y, '  Good', ha='center', fontsize=16, 
                                       color=line_colors[1], weight='bold')
                                ax.text(midpoint3, text_y, '  MUA', ha='center', fontsize=16, 
                                       color=line_colors[2], weight='bold')
                            else:
                                # Lower values = Good, so Good should be on the left  
                                ax.text(midpoint1, text_y, '  Good', ha='center', fontsize=16, 
                                       color=line_colors[0], weight='bold')
                                ax.text(midpoint2, text_y, '  MUA', ha='center', fontsize=16, 
                                       color=line_colors[1], weight='bold')
                                ax.text(midpoint3, text_y, '  Good', ha='center', fontsize=16, 
                                       color=line_colors[2], weight='bold')
                        
                    elif thresh1 is not None or thresh2 is not None:
                        # Single threshold logic - handle BOTH thresh1 and thresh2 cases
                        thresh = thresh1 if thresh1 is not None else thresh2
                        thresh_offset = thresh + binsize_offset
                        ax.axvline(thresh_offset, color='k', linewidth=2)
                        ax.plot([x_lim[0], thresh_offset], 
                               [line_y, line_y], color=line_colors[0], linewidth=6)
                        ax.plot([thresh_offset, x_lim[1]], 
                               [line_y, line_y], color=line_colors[1], linewidth=6)
                        
                        midpoint1 = (x_lim[0] + thresh_offset) / 2
                        midpoint2 = (thresh_offset + x_lim[1]) / 2
                        text_y = 0.95
                        
                        noise_metrics = ['nPeaks', 'nTroughs', 'waveformBaselineFlatness', 'waveformDuration_peakTrough', 'scndPeakToTroughRatio', 'spatialDecaySlope']
                        nonsomatic_metrics = ['peak1ToPeak2Ratio', 'mainPeakToTroughRatio']
                        
                        if metric_name in noise_metrics:
                            ax.text(midpoint1, text_y, '  Neuronal', ha='center', fontsize=16, 
                                   color=line_colors[0], weight='bold')
                            ax.text(midpoint2, text_y, '  Noise', ha='center', fontsize=16, 
                                   color=line_colors[1], weight='bold')
                        elif metric_name in nonsomatic_metrics:
                            ax.text(midpoint1, text_y, '  Somatic', ha='center', fontsize=16, 
                                   color=line_colors[0], weight='bold')
                            ax.text(midpoint2, text_y, '  Non-somatic', ha='center', fontsize=16, 
                                   color=line_colors[1], weight='bold')
                        else:
                            # MUA metrics: determine labels based on colors
                            # Orange colors (1.0, 0.5469, 0) = MUA, Green colors (0, 0.5, 0) = Good
                            if np.allclose(line_colors[0], [1.0, 0.5469, 0]):
                                # First color is orange (MUA), second should be green (Good)
                                ax.text(midpoint1, text_y, '  MUA', ha='center', fontsize=16, 
                                       color=line_colors[0], weight='bold')
                                ax.text(midpoint2, text_y, '  Good', ha='center', fontsize=16, 
                                       color=line_colors[1], weight='bold')
                            else:
                                # First color is green (Good), second should be orange (MUA)
                                ax.text(midpoint1, text_y, '  Good', ha='center', fontsize=16, 
                                       color=line_colors[0], weight='bold')
                                ax.text(midpoint2, text_y, '  MUA', ha='center', fontsize=16, 
                                       color=line_colors[1], weight='bold')

                # Set histogram limits from 0 to 1.1 to show classification lines and text
                ax.set_ylim([0, 1.1])
                
            ax.set_xlabel(valid_labels[i], fontsize=16, fontweight='bold')
            if i == 0:
                ax.set_ylabel('frac. units', fontsize=16, fontweight='bold')
            
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
            # REFINED Y-AXIS: Only show ticks and labels at 0 and 1
            ax.set_yticks([0, 1])
            ax.set_yticklabels(['0', '1'])
            ax.tick_params(labelsize=14)
            
            # Add legend ONLY to the rightmost histogram in top row - positioned OUTSIDE plots
            if col_id == cols - 1 and row_id == 0:  # Rightmost plot in first row
                from matplotlib.lines import Line2D
                
                # Create legend elements to match the actual markers
                triangle_marker = Line2D([0], [0], marker='v', color='w', markerfacecolor='black', 
                                       markersize=15, markeredgecolor='white', markeredgewidth=3)
                black_line = Line2D([0], [0], color='black', linewidth=4)
                
                # Position legend OUTSIDE the rightmost plot
                ax.legend([triangle_marker, black_line], 
                         ['Current unit location', 'Classification parameter'], 
                         bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=12, 
                         frameon=True, fancybox=True, shadow=True)

    def update_display(self):
        """Update the entire display"""
        self.update_unit_info()
        self.plot_unit(self.current_unit_idx)




def load_metrics_for_gui(ks_dir, quality_metrics, ephys_properties=None, param=None, save_path=None):
    """
    Load and prepare data for GUI - Python equivalent of loadMetricsForGUI.m
    
    Parameters
    ----------
    ks_dir : str
        Path to kilosort directory
    quality_metrics : dict
        Quality metrics from bombcell
    ephys_properties : list, optional
        Ephys properties from compute_all_ephys_properties
    param : dict, optional
        Parameters dictionary
    save_path : str, optional
        Path where bombcell data was saved. If None, defaults to ks_dir/bombcell
        
    Returns
    -------
    dict
        Dictionary containing all data needed for GUI
    """
    from . import loading_utils as bc_load
    
    # Load ephys data (returns tuple)
    ephys_data_tuple = bc_load.load_ephys_data(ks_dir)
    
    # Convert tuple to dictionary
    ephys_data = {
        'spike_times': ephys_data_tuple[0] / param['ephys_sample_rate'],  # Convert to seconds
        'spike_clusters': ephys_data_tuple[1],
        'template_waveforms': ephys_data_tuple[2],
        'template_amplitudes': ephys_data_tuple[3],
        'pc_features': ephys_data_tuple[4],
        'pc_features_idx': ephys_data_tuple[5],
        'channel_positions': ephys_data_tuple[6]
    }
    
    # Determine the save path for bombcell data
    if save_path is None:
        bombcell_path = Path(ks_dir) / "bombcell"
    else:
        bombcell_path = Path(save_path)
    
    # Load raw waveforms if available
    raw_waveforms = None
    raw_wf_path = bombcell_path / "templates._bc_rawWaveforms.npy"
    if raw_wf_path.exists():
        try:
            raw_waveforms = {
                'average': np.load(raw_wf_path, allow_pickle=True),
                'peak_channels': np.load(bombcell_path / "templates._bc_rawWaveformPeakChannels.npy", allow_pickle=True)
            }
        except FileNotFoundError:
            raw_waveforms = None
    
    # Ensure param has the ks_dir path for auto-loading GUI data
    if param is None:
        param = {}
    if 'ephysKilosortPath' not in param:
        param['ephysKilosortPath'] = ks_dir
    
    return {
        'ephys_data': ephys_data,
        'quality_metrics': quality_metrics,
        'ephys_properties': ephys_properties,
        'raw_waveforms': raw_waveforms,
        'param': param
    }


def load_unit_quality_gui(ephys_data_or_path=None, quality_metrics=None, ephys_properties=None, 
                     unit_types=None, param=None, ks_dir=None, save_path=None, layout='landscape', auto_advance=True):
    """
    Launch the Unit Quality GUI - Python equivalent of unitQualityGUI_synced
    
    Parameters
    ----------
    ephys_data_or_path : str or dict, optional
        Either path to kilosort directory or pre-loaded ephys_data dictionary
    quality_metrics : dict
        Quality metrics from bombcell
    ephys_properties : list, optional
        Ephys properties from compute_all_ephys_properties
    unit_types : array, optional
        Unit type classifications
    param : dict, optional
        Parameters dictionary
    ks_dir : str, optional
        Alternative parameter name for kilosort directory path (for backward compatibility)
    save_path : str, optional
        Path where precomputed gui data could have been saved
    auto_advance : bool, optional
        Whether to automatically advance to next unit after manual classification
        Default: True
        
    Returns
    -------
    UnitQualityGUI
        The GUI object
    """
    # Handle backward compatibility with ks_dir parameter
    if ks_dir is not None:
        ephys_data_or_path = ks_dir
    # Handle case where save_path is not provided
    if save_path is None:
        save_path = ks_dir
    # Check if input is a path or already loaded data
    if isinstance(ephys_data_or_path, dict):
        # Data is already loaded
        ephys_data = ephys_data_or_path
        raw_waveforms = None  # Would need to be passed separately if needed
        if param is None:
            param = {}
    else:
        # Load data from path
        gui_data = load_metrics_for_gui(ephys_data_or_path, quality_metrics, ephys_properties, param, save_path)
        ephys_data = gui_data['ephys_data']
        raw_waveforms = gui_data['raw_waveforms']
        param = gui_data['param']
    
    # Check if ipywidgets is available
    if widgets is None:
        raise ImportError(
            "❌ ipywidgets is required for the BombCell GUI!\n"
            "📦 Please install it with:\n"
            "   pip install ipywidgets\n"
            "   OR\n"
            "   conda install ipywidgets\n"
            "💡 Then restart your Jupyter kernel and try again."
        )
    
    # Create and return interactive GUI
    gui = InteractiveUnitQualityGUI(
        ephys_data=ephys_data,
        quality_metrics=quality_metrics,
        ephys_properties=ephys_properties,
        raw_waveforms=raw_waveforms,
        param=param,
        unit_types=unit_types,
        save_path=save_path,
        layout=layout,
        auto_advance=auto_advance,
    )
    return gui

# make static methods more easily accessible from outside the class
compile_variables = InteractiveUnitQualityGUI.compile_variables
unpack_variables = InteractiveUnitQualityGUI.unpack_variables
get_unit_data = InteractiveUnitQualityGUI.get_unit_data
plot_amplitude_histogram = InteractiveUnitQualityGUI.plot_amplitude_histogram
plot_template_waveform = InteractiveUnitQualityGUI.plot_template_waveform
plot_raw_waveforms = InteractiveUnitQualityGUI.plot_raw_waveforms
plot_autocorrelogram = InteractiveUnitQualityGUI.plot_autocorrelogram
plot_spatial_decay = InteractiveUnitQualityGUI.plot_spatial_decay
plot_amplitudes_over_time = InteractiveUnitQualityGUI.plot_amplitudes_over_time
plot_time_bin_metrics = InteractiveUnitQualityGUI.plot_time_bin_metrics
plot_unit_location = InteractiveUnitQualityGUI.plot_unit_location
plot_amplitude_fit = InteractiveUnitQualityGUI.plot_amplitude_fit
add_metrics_text = InteractiveUnitQualityGUI.add_metrics_text
mark_peaks_and_troughs = InteractiveUnitQualityGUI.mark_peaks_and_troughs
get_nearby_channels_for_spatial_decay = InteractiveUnitQualityGUI.get_nearby_channels_for_spatial_decay
plot_histograms_panel = InteractiveUnitQualityGUI.plot_histograms_panel
