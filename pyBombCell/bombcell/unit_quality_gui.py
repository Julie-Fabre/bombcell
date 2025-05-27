import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
from pathlib import Path
import pandas as pd
import pickle
import os

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False

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
        'peak_trough_labels': {},
        'duration_lines': {},
        'spatial_decay_fits': {},
        'amplitude_fits': {},
        'channel_arrangements': {},
        'waveform_scaling': {},
        'acg_data': {}  # Pre-computed autocorrelograms
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
            
        # Pre-compute peak/trough detection
        if template.size > 0 and len(template.shape) > 1 and max_ch < template.shape[1]:
            max_ch_waveform = template[:, max_ch]
            
            if SCIPY_AVAILABLE:
                # Peak detection parameters (same as GUI)
                waveform_range = np.max(max_ch_waveform) - np.min(max_ch_waveform)
                peak_height_threshold = np.max(max_ch_waveform) * 0.5
                peak_prominence = waveform_range * 0.1
                
                # Find peaks and troughs
                peaks, _ = find_peaks(max_ch_waveform, 
                                    height=peak_height_threshold, 
                                    distance=10,
                                    prominence=peak_prominence)
                                    
                troughs, _ = find_peaks(-max_ch_waveform, 
                                      height=-np.min(max_ch_waveform) * 0.5, 
                                      distance=10,
                                      prominence=waveform_range * 0.1)
                
                gui_data['peak_locations'][unit_idx] = peaks
                gui_data['trough_locations'][unit_idx] = troughs
                
                # Pre-compute duration lines
                if len(peaks) > 0 and len(troughs) > 0:
                    main_peak_idx = peaks[np.argmax(max_ch_waveform[peaks])]
                    main_trough_idx = troughs[np.argmin(max_ch_waveform[troughs])]
                    gui_data['duration_lines'][unit_idx] = {
                        'main_peak': main_peak_idx,
                        'main_trough': main_trough_idx
                    }
            
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
                'scaling_factor': np.ptp(max_ch_waveform) * 2.5 if template.size > 0 else 1.0
            }
        
        # Skip ACG pre-computation - will be computed lazily in GUI to avoid slowdown
        # Just store placeholder to indicate ACG needs computation
        gui_data['acg_data'][unit_idx] = None
    
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
                 raw_waveforms=None, param=None, unit_types=None, gui_data=None):
        """
        Initialize the interactive GUI
        
        Parameters:
        -----------
        gui_data : dict, optional
            Pre-computed GUI visualization data from precompute_gui_data()
            If provided, will use pre-computed results for faster display
        """
        self.ephys_data = ephys_data
        self.quality_metrics = quality_metrics
        self.ephys_properties = ephys_properties or []
        self.raw_waveforms = raw_waveforms
        self.param = param or {}
        self.unit_types = unit_types
        
        # Auto-load GUI data if not provided but param has path info
        if gui_data is None and param and 'ephysKilosortPath' in param:
            # Try to auto-load from standard bombcell location
            import os
            ks_path = param['ephysKilosortPath']
            possible_paths = [
                os.path.join(ks_path, 'bombcell', 'for_GUI', 'gui_data.pkl'),
                os.path.join(ks_path, 'for_GUI', 'gui_data.pkl'),
                os.path.join(ks_path, 'gui_data.pkl')
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
                    
            if 'acg_data' in gui_data:
                count = len(gui_data['acg_data'])
                data_status.append(f"ACG data: {count} units (computed on-demand)")
            
            for status in data_status:
                print(f"   {status}")
        else:
            print("No pre-computed GUI data found - will compute everything real-time")
        
        # Get unique units
        self.unique_units = np.unique(ephys_data['spike_clusters'])
        self.n_units = len(self.unique_units)
        print(f"Total units: {self.n_units}")
        self.current_unit_idx = 0
        
        # Setup widgets and display
        self.setup_widgets()
        self.display_gui()
        
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
        
        # Classification toggle buttons
        self.classify_good_btn = widgets.Button(description='Mark as Good', button_style='success')
        self.classify_mua_btn = widgets.Button(description='Mark as MUA', button_style='warning')
        self.classify_noise_btn = widgets.Button(description='Mark as Noise', button_style='danger')
        
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
        self.classify_noise_btn.on_click(lambda b: self.classify_unit(0))
        
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
            self.goto_prev_noise_btn, self.goto_noise_btn,
            self.goto_prev_nonsomatic_btn, self.goto_nonsomatic_btn
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
            widgets.Label('  Go to:  '),
            self.unit_input, 
            self.goto_unit_btn
        ])
        
        # Classification controls (hidden for now)
        # classify_controls = widgets.HBox([
        #     self.classify_good_btn, self.classify_mua_btn, self.classify_noise_btn
        # ])
        
        # Full interface
        interface = widgets.VBox([
            slider_and_input,
            self.unit_info,
            nav_controls,
            # classify_controls,  # Hidden for now
            self.plot_output
        ])
        
        display(interface)
        
        # Initial plot
        self.update_display()
        
    def on_unit_change(self, change):
        """Handle unit slider change"""
        self.current_unit_idx = change['new']
        self.update_display()
        
    def prev_unit(self, b=None):
        """Go to previous unit"""
        if self.current_unit_idx > 0:
            self.current_unit_idx -= 1
            self.unit_slider.value = self.current_unit_idx
            
    def next_unit(self, b=None):
        """Go to next unit"""
        if self.current_unit_idx < self.n_units - 1:
            self.current_unit_idx += 1
            self.unit_slider.value = self.current_unit_idx
            
    def goto_unit_number(self, b=None):
        """Go to specific unit number"""
        unit_num = self.unit_input.value
        if 0 <= unit_num < self.n_units:
            self.current_unit_idx = unit_num
            self.unit_slider.value = self.current_unit_idx
            self.update_display()
            
    def goto_next_good(self, b=None):
        """Go to next good unit"""
        if self.unit_types is not None:
            for i in range(self.current_unit_idx + 1, self.n_units):
                if self.unit_types[i] == 1:  # Good unit
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
                    
    def goto_prev_good(self, b=None):
        """Go to previous good unit"""
        if self.unit_types is not None:
            for i in range(self.current_unit_idx - 1, -1, -1):
                if self.unit_types[i] == 1:  # Good unit
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
                    
    def goto_next_mua(self, b=None):
        """Go to next MUA unit"""
        if self.unit_types is not None:
            for i in range(self.current_unit_idx + 1, self.n_units):
                if self.unit_types[i] == 2:  # MUA unit
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
                    
    def goto_prev_mua(self, b=None):
        """Go to previous MUA unit"""
        if self.unit_types is not None:
            for i in range(self.current_unit_idx - 1, -1, -1):
                if self.unit_types[i] == 2:  # MUA unit
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
                    
    def goto_next_noise(self, b=None):
        """Go to next noise unit"""
        if self.unit_types is not None:
            for i in range(self.current_unit_idx + 1, self.n_units):
                if self.unit_types[i] == 0:  # Noise unit
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
                    
    def goto_prev_noise(self, b=None):
        """Go to previous noise unit"""
        if self.unit_types is not None:
            for i in range(self.current_unit_idx - 1, -1, -1):
                if self.unit_types[i] == 0:  # Noise unit
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
                    
    def goto_next_nonsomatic(self, b=None):
        """Go to next non-somatic unit"""
        if self.unit_types is not None:
            for i in range(self.current_unit_idx + 1, self.n_units):
                if self.unit_types[i] in [3, 4]:  # Non-somatic good or non-somatic MUA
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
                    
    def goto_prev_nonsomatic(self, b=None):
        """Go to previous non-somatic unit"""
        if self.unit_types is not None:
            for i in range(self.current_unit_idx - 1, -1, -1):
                if self.unit_types[i] in [3, 4]:  # Non-somatic good or non-somatic MUA
                    self.current_unit_idx = i
                    self.unit_slider.value = self.current_unit_idx
                    break
                    
    def classify_unit(self, classification):
        """Classify current unit"""
        if self.unit_types is not None:
            self.unit_types[self.current_unit_idx] = classification
            self.update_unit_info()
            
    def get_unit_data(self, unit_idx):
        """Get data for a specific unit"""
        if unit_idx >= self.n_units:
            return None
            
        unit_id = self.unique_units[unit_idx]
        
        # Get spike times for this unit
        spike_mask = self.ephys_data['spike_clusters'] == unit_id
        spike_times = self.ephys_data['spike_times'][spike_mask]
        
        # Get template waveform
        if unit_idx < len(self.ephys_data['template_waveforms']):
            template = self.ephys_data['template_waveforms'][unit_idx]
        else:
            template = np.zeros((82, 1))
            
        # Get quality metrics for this unit
        unit_metrics = {}
        
        # Handle different quality_metrics formats
        if isinstance(self.quality_metrics, list):
            # List of dicts format: [{'phy_clusterID': 0, 'metric1': val, ...}, ...]
            unit_found = False
            for unit_dict in self.quality_metrics:
                if unit_dict.get('phy_clusterID') == unit_id:
                    unit_metrics = unit_dict.copy()
                    unit_found = True
                    break
            if not unit_found:
                unit_metrics = {'phy_clusterID': unit_id}
                
        elif isinstance(self.quality_metrics, dict):
            if unit_id in self.quality_metrics:
                # Dict with unit_id keys: {0: {'metric1': val, ...}, 1: {...}}
                unit_metrics = self.quality_metrics[unit_id].copy()
            else:
                # Dict with metric keys: {'metric1': [val0, val1, ...], 'metric2': [...]}
                for key, values in self.quality_metrics.items():
                    if hasattr(values, '__len__') and len(values) > unit_idx:
                        unit_metrics[key] = values[unit_idx]
                    else:
                        unit_metrics[key] = np.nan
        elif hasattr(self.quality_metrics, 'iloc'):
            # DataFrame format from bc.load_bc_results()
            if unit_idx < len(self.quality_metrics):
                unit_metrics = self.quality_metrics.iloc[unit_idx].to_dict()
            else:
                unit_metrics = {}
        else:
            unit_metrics = {}
                
        return {
            'unit_id': unit_id,
            'spike_times': spike_times,
            'template': template,
            'metrics': unit_metrics
        }
        
    def update_unit_info(self):
        """Update unit info display"""
        unit_data = self.get_unit_data(self.current_unit_idx)
        if unit_data is None:
            return
            
        unit_type_str = "Unknown"
        if self.unit_types is not None and self.current_unit_idx < len(self.unit_types):
            unit_type = self.unit_types[self.current_unit_idx]
            type_map = {0: "Noise", 1: "Good", 2: "MUA", 3: "Non-soma good", 4: "Non-soma MUA"}
            unit_type_str = type_map.get(unit_type, "Unknown")
            
        # Get title color based on unit type
        title_colors = {
            "Noise": "red",
            "Good": "green", 
            "MUA": "orange",
            "Non-soma good": "blue",
            "Non-soma MUA": "blue",
            "Unknown": "black"
        }
        title_color = title_colors.get(unit_type_str, "black")
        
        # Simple title with unit number, phy ID, and type, colored by classification (large and centered)
        info_html = f"""
        <h1 style="color: {title_color}; text-align: center; font-size: 24px; margin: 10px 0;">Unit {unit_data['unit_id']} (phy ID = {self.current_unit_idx}, {self.current_unit_idx+1}/{self.n_units}) - {unit_type_str}</h1>
        """
        
        self.unit_info.value = info_html
        
    def plot_unit(self, unit_idx):
        """Plot data for a specific unit"""
        unit_data = self.get_unit_data(unit_idx)
        if unit_data is None:
            return
            
        with self.plot_output:
            clear_output(wait=True)
            
            # Create figure with tighter grid spacing 
            fig = plt.figure(figsize=(20, 14))
            fig.patch.set_facecolor('white')
            
            # 1. Unit location plot (left column) - subplot(10, 15, spans all rows)
            ax_location = plt.subplot2grid((10, 15), (0, 0), rowspan=10, colspan=1)
            self.plot_unit_location(ax_location, unit_data)
            
            # 2. Template waveforms - rows 0-1
            ax_template = plt.subplot2grid((10, 15), (0, 2), rowspan=2, colspan=6)
            self.plot_template_waveform(ax_template, unit_data)
            
            # 3. Raw waveforms - rows 0-1
            ax_raw = plt.subplot2grid((10, 15), (0, 9), rowspan=2, colspan=6)
            self.plot_raw_waveforms(ax_raw, unit_data)
            
            # 4. Spatial decay - rows 3-4 (uniform gap after row 2)
            ax_spatial = plt.subplot2grid((10, 15), (3, 2), rowspan=2, colspan=6)
            self.plot_spatial_decay(ax_spatial, unit_data)
            
            # 5. ACG - rows 3-4 (uniform gap after row 2)
            ax_acg = plt.subplot2grid((10, 15), (3, 9), rowspan=2, colspan=6)
            self.plot_autocorrelogram(ax_acg, unit_data)
            
            # 6. Amplitudes over time - rows 6-7 (uniform gap after row 5)
            ax_amplitude = plt.subplot2grid((10, 15), (6, 2), rowspan=2, colspan=10)
            self.plot_amplitudes_over_time(ax_amplitude, unit_data)
            
            # 6b. Time bin metrics - row 9 (uniform gap after row 8)
            ax_bin_metrics = plt.subplot2grid((10, 15), (9, 2), rowspan=1, colspan=10, sharex=ax_amplitude)
            self.plot_time_bin_metrics(ax_bin_metrics, unit_data)
            
            # 7. Amplitude fit - positioned to match amplitude height (rows 6-7)
            ax_amp_fit = plt.subplot2grid((10, 15), (6, 13), rowspan=2, colspan=2)
            self.plot_amplitude_fit(ax_amp_fit, unit_data)
            
            # Adjust subplot margins manually with tighter spacing
            plt.subplots_adjust(left=0.06, right=0.83, top=0.95, bottom=0.08, hspace=0.15, wspace=0.15)
            plt.show()
            
    def plot_template_waveform(self, ax, unit_data):
        """Plot template waveform using BombCell MATLAB spatial arrangement"""
        template = unit_data['template']
        metrics = unit_data['metrics']
        
        if template.size > 0 and len(template.shape) > 1:
            # Get peak channel from quality metrics
            if 'maxChannels' in self.quality_metrics and self.current_unit_idx < len(self.quality_metrics['maxChannels']):
                max_ch = int(self.quality_metrics['maxChannels'][self.current_unit_idx])
            else:
                max_ch = int(metrics.get('maxChannels', 0))
            
            n_channels = template.shape[1]
            
            # Find channels within 100μm of max channel (like MATLAB BombCell)
            if 'channel_positions' in self.ephys_data and max_ch < len(self.ephys_data['channel_positions']):
                positions = self.ephys_data['channel_positions']
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
                            
                            # Calculate Y offset based on channel Y position (like MATLAB)
                            y_offset = (ch_pos[1] - max_pos[1]) / 100 * scaling_factor
                            
                            # Plot waveform
                            x_vals = time_axis + x_offset
                            y_vals = -waveform + y_offset  # Negative like MATLAB
                            
                            if ch == max_ch:
                                ax.plot(x_vals, y_vals, 'k-', linewidth=3)  # Max channel thicker black
                            else:
                                ax.plot(x_vals, y_vals, 'k-', linewidth=1, alpha=0.7)
                            
                            # Add channel number closer to waveform
                            ax.text(x_offset - waveform_width * 0.05, y_offset, f'{ch}', fontsize=13, ha='right', va='center', fontfamily="DejaVu Sans", zorder=20)
                    
                    # Mark peaks and troughs on max channel
                    max_ch_waveform = template[:, max_ch]
                    # Max channel is at center (0,0) since all offsets are relative to it
                    max_ch_x_offset = 0  
                    max_ch_y_offset = 0
                    
                    # Detect and mark peaks/troughs (account for inverted waveform)
                    self.mark_peaks_and_troughs(ax, -max_ch_waveform, max_ch_x_offset, max_ch_y_offset, metrics, scaling_factor)
                    
                    # Set axis properties like MATLAB
                    ax.invert_yaxis()  # Reverse Y direction like MATLAB
                    
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
        self.add_metrics_text(ax, unit_data, 'template')
        
    def plot_raw_waveforms(self, ax, unit_data):
        """Plot raw waveforms with 16 nearest channels like MATLAB"""
        metrics = unit_data['metrics']
        
        # Check if raw extraction is enabled
        extract_raw = self.param.get('extractRaw', 0)
        if extract_raw != 1:
            ax.text(0.5, 0.5, 'Raw waveforms\n(extractRaw disabled)', 
                    ha='center', va='center', transform=ax.transAxes, fontfamily="DejaVu Sans")
            ax.set_title('Raw waveforms', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
            return
        
        if self.raw_waveforms is not None:
            raw_wf = self.raw_waveforms.get('average', None)
            if raw_wf is not None:
                try:
                    if hasattr(raw_wf, '__len__') and self.current_unit_idx < len(raw_wf):
                        waveforms = raw_wf[self.current_unit_idx]
                        
                        if hasattr(waveforms, 'shape') and len(waveforms.shape) > 1:
                            # Multi-channel raw waveforms - use MATLAB spatial arrangement
                            if 'maxChannels' in self.quality_metrics and self.current_unit_idx < len(self.quality_metrics['maxChannels']):
                                max_ch = int(self.quality_metrics['maxChannels'][self.current_unit_idx])
                            else:
                                max_ch = int(metrics.get('maxChannels', 0))
                            n_channels = waveforms.shape[1]
                            
                            # Find channels within 100μm of max channel (like MATLAB BombCell)
                            if 'channel_positions' in self.ephys_data and max_ch < len(self.ephys_data['channel_positions']):
                                positions = self.ephys_data['channel_positions']
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
                                    max_ch_waveform = waveforms[:, max_ch]
                                    scaling_factor = np.ptp(max_ch_waveform) * 2.5
                                    
                                    # Create time axis
                                    time_axis = np.arange(waveforms.shape[0])
                                    
                                    # Plot each channel at its spatial position
                                    for ch in channels_to_plot:
                                        if ch < n_channels:
                                            waveform = waveforms[:, ch]
                                            ch_pos = positions[ch]
                                            
                                            # Calculate X offset - use probe geometry with reasonable separation
                                            waveform_width = waveforms.shape[0]  # Usually 82 samples
                                            x_offset = (ch_pos[0] - max_pos[0]) * waveform_width * 0.06  # Slightly increased from original 0.05
                                            
                                            # Calculate Y offset based on channel Y position (like MATLAB)
                                            y_offset = (ch_pos[1] - max_pos[1]) / 100 * scaling_factor
                                            
                                            # Plot waveform
                                            x_vals = time_axis + x_offset
                                            y_vals = -waveform + y_offset  # Negative like MATLAB
                                            
                                            if ch == max_ch:
                                                ax.plot(x_vals, y_vals, 'k-', linewidth=3)  # Max channel thicker black
                                            else:
                                                ax.plot(x_vals, y_vals, 'gray', linewidth=1, alpha=0.7)
                                            
                                            # Add channel number closer to waveform
                                            ax.text(x_offset - waveform_width * 0.02, y_offset, f'{ch}', fontsize=13, ha='right', va='center', fontfamily="DejaVu Sans", zorder=20)
                                    
                                    # Mark peaks and troughs on max channel
                                    max_ch_waveform = waveforms[:, max_ch]
                                    # Max channel is at center (0,0) since all offsets are relative to it
                                    max_ch_x_offset = 0
                                    max_ch_y_offset = 0
                                    
                                    # Detect and mark peaks/troughs (account for inverted waveform)
                                    self.mark_peaks_and_troughs(ax, -max_ch_waveform, max_ch_x_offset, max_ch_y_offset, metrics, scaling_factor)
                                    
                                    # Set axis properties like MATLAB
                                    ax.invert_yaxis()  # Reverse Y direction like MATLAB
                            else:
                                # Fallback: simple single channel display
                                ax.plot(waveforms[:, max_ch], 'b-', linewidth=2)
                                ax.text(0.5, 0.5, f'Raw waveforms\n(channel {max_ch})', 
                                       ha='center', va='center', transform=ax.transAxes, fontfamily="DejaVu Sans")
                        else:
                            # Single channel
                            ax.plot(waveforms, 'b-', alpha=0.7)
                            
                except (TypeError, IndexError, AttributeError):
                    ax.text(0.5, 0.5, 'Raw waveforms\n(data format issue)', 
                            ha='center', va='center', transform=ax.transAxes, fontfamily="DejaVu Sans")
        else:
            ax.text(0.5, 0.5, 'Raw waveforms\n(not available)', 
                    ha='center', va='center', transform=ax.transAxes, fontfamily="DejaVu Sans")
                    
        ax.set_title('Raw waveforms', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove aspect ratio constraint to prevent squishing
        
        # Extend x-axis limits to accommodate channel number text (right side only)
        x_min, x_max = ax.get_xlim()
        text_padding = 100  # Padding for text labels on right side
        ax.set_xlim(x_min, x_max + text_padding)
        
        # Add quality metrics text
        self.add_metrics_text(ax, unit_data, 'raw')
        
    def plot_autocorrelogram(self, ax, unit_data):
        """Plot autocorrelogram with tauR and firing rate lines"""
        spike_times = unit_data['spike_times']
        metrics = unit_data['metrics']
        
        # Check if we have pre-computed ACG data
        if (self.gui_data and 
            'acg_data' in self.gui_data and 
            self.current_unit_idx in self.gui_data['acg_data'] and
            self.gui_data['acg_data'][self.current_unit_idx] is not None):
            
            print(f"🚀 ACG: Using PRE-COMPUTED autocorrelogram for unit {self.current_unit_idx}")
            acg_data = self.gui_data['acg_data'][self.current_unit_idx]
            autocorr = acg_data['autocorr']
            bin_centers = acg_data['bin_centers']
            bin_size = acg_data['bin_size']
            
        elif len(spike_times) > 1:
            # Filter spike times to good time chunks if computeTimeChunks is enabled
            filtered_spike_times = spike_times.copy()
            if self.param and self.param.get('computeTimeChunks', False):
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
            
            # ACG calculation will proceed without status messages
            
            # ACG calculation with MATLAB-style parameters
            max_lag = 0.05  # 50ms
            bin_size = 0.001  # 1ms bins
            
            # Calculate proper autocorrelogram (not just ISIs)
            # Create bins centered around 0
            bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
            bin_centers = bins[:-1] + bin_size/2
            
            # Calculate autocorrelogram
            autocorr = np.zeros(len(bin_centers))
            
            # For efficiency, subsample spikes if there are too many
            if len(filtered_spike_times) > 10000:
                indices = np.random.choice(len(filtered_spike_times), 10000, replace=False)
                spike_subset = filtered_spike_times[indices]
            else:
                spike_subset = filtered_spike_times
            
            # Calculate cross-correlation with itself  
            for i, spike_time in enumerate(spike_subset[::10]):  # Subsample further for speed
                # Find spikes within max_lag of this spike
                time_diffs = filtered_spike_times - spike_time
                valid_diffs = time_diffs[(np.abs(time_diffs) <= max_lag) & (time_diffs != 0)]
                
                if len(valid_diffs) > 0:
                    hist, _ = np.histogram(valid_diffs, bins=bins)
                    autocorr += hist
            
            # Convert to firing rate (spikes/sec)
            if len(spike_subset) > 0:
                recording_duration = np.max(filtered_spike_times) - np.min(filtered_spike_times) if len(filtered_spike_times) > 0 else 1
                autocorr = autocorr / (len(spike_subset) * bin_size) if recording_duration > 0 else autocorr
                
            # Cache the computed ACG for future use
            if self.gui_data and 'acg_data' in self.gui_data:
                self.gui_data['acg_data'][self.current_unit_idx] = {
                    'autocorr': autocorr,
                    'bin_centers': bin_centers,
                    'bin_size': bin_size
                }
        else:
            return
        
        # Plot only positive lags (like MATLAB)
        positive_mask = bin_centers >= 0
        positive_centers = bin_centers[positive_mask]
        positive_autocorr = autocorr[positive_mask]
        
        # Plot with wider bars like MATLAB
        ax.bar(positive_centers * 1000, positive_autocorr, 
               width=bin_size*1000*0.9, color='grey', alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Get mean firing rate from metrics or calculate if not available
        mean_fr = 0
        # Use filtered spike times for firing rate calculation when computeTimeChunks is enabled
        spikes_for_fr = filtered_spike_times if 'filtered_spike_times' in locals() else spike_times
        
        if len(spikes_for_fr) > 1:
            # Use pre-computed firing rate if available (but may need to adjust for filtered data)
            if 'firing_rate_mean' in metrics and not np.isnan(metrics['firing_rate_mean']) and not (self.param and self.param.get('computeTimeChunks', False)):
                mean_fr = metrics['firing_rate_mean']
            else:
                # Calculate firing rate from filtered spike times
                recording_duration = np.max(spikes_for_fr) - np.min(spikes_for_fr)
                if recording_duration > 0:
                    mean_fr = len(spikes_for_fr) / recording_duration
            # Set limits to accommodate both data and mean firing rate line
            ax.set_xlim(0, max_lag * 1000)
            if len(positive_autocorr) > 0:
                data_max = np.max(positive_autocorr)
                # More conservative y-limits - just ensure mean firing rate is visible
                y_max = max(data_max * 1.05, mean_fr * 1.1)
                ax.set_ylim(0, y_max)
            else:
                y_max = mean_fr * 1.1 if mean_fr > 0 else 10
                ax.set_ylim(0, y_max)
            
            # Calculate correct tauR using RPV_window_index
            tau_r = None
            if (self.param and 'tauR_valuesMin' in self.param and 'tauR_valuesMax' in self.param and 
                'tauR_valuesStep' in self.param and 'RPV_window_index' in metrics):
                try:
                    tau_r_min = self.param['tauR_valuesMin'] * 1000  # Convert to ms
                    tau_r_max = self.param['tauR_valuesMax'] * 1000  # Convert to ms
                    tau_r_step = self.param['tauR_valuesStep'] * 1000  # Convert to ms
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
        if self.param and 'tauR_valuesMin' in self.param and 'tauR_valuesMax' in self.param:
            tau_r_min_ms = self.param['tauR_valuesMin'] * 1000  # Convert to ms
            tau_r_max_ms = self.param['tauR_valuesMax'] * 1000  # Convert to ms
            
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
        self.add_metrics_text(ax, unit_data, 'acg')
        
    def plot_spatial_decay(self, ax, unit_data):
        """Plot spatial decay like MATLAB - only nearby channels"""
        metrics = unit_data['metrics']
        
        # Check if spatial decay metrics are available
        if 'spatialDecaySlope' in metrics and not pd.isna(metrics['spatialDecaySlope']):
            max_ch = int(metrics.get('maxChannels', 0))
            template = unit_data['template']
            
            if template.size > 0 and len(template.shape) > 1:
                # Get only nearby channels for spatial decay (like MATLAB)
                nearby_channels = self.get_nearby_channels_for_spatial_decay(max_ch, template.shape[1])
                
                if 'channel_positions' in self.ephys_data and len(self.ephys_data['channel_positions']) > max_ch:
                    positions = self.ephys_data['channel_positions']
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
                            ax.set_ylabel('Normalized amplitude', fontsize=13, fontfamily="DejaVu Sans")
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
        self.add_metrics_text(ax, unit_data, 'spatial')
        
    def plot_amplitudes_over_time(self, ax, unit_data):
        """Plot amplitudes over time with firing rate below and presence ratio indicators"""
        spike_times = unit_data['spike_times']
        metrics = unit_data['metrics']
        
        if len(spike_times) > 0:
            # Get amplitudes if available
            unit_id = unit_data['unit_id']
            spike_mask = self.ephys_data['spike_clusters'] == unit_id
            
            # Calculate time bins for presence ratio and firing rate
            total_duration = np.max(spike_times) - np.min(spike_times)
            n_bins = max(20, int(total_duration / 60))  # ~1 minute bins, minimum 20 bins
            time_bins = np.linspace(np.min(spike_times), np.max(spike_times), n_bins + 1)
            bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
            bin_width = time_bins[1] - time_bins[0]
            
            # Calculate firing rate per bin
            bin_counts, _ = np.histogram(spike_times, bins=time_bins)
            firing_rates = bin_counts / bin_width
            
            
            if 'template_amplitudes' in self.ephys_data:
                amplitudes = self.ephys_data['template_amplitudes'][spike_mask]
                
                # Color spikes based on goodTimeChunks if computeTimeChunks is enabled
                spike_colors = np.full(len(spike_times), 'darkorange')  # Default: bad chunks (orange)
                
                if self.param and self.param.get('computeTimeChunks', False):
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
                ax.set_ylabel('Template scaling', color='blue', fontsize=13, fontfamily="DejaVu Sans")
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
                if (self.param and self.param.get('computeDrift', False) and 
                    hasattr(self, 'gui_data') and self.gui_data is not None):
                    
                    unit_idx = self.current_unit_idx
                    
                    # Check if per_bin_metrics exists and contains drift data
                    if 'per_bin_metrics' in self.gui_data:
                        per_bin_metrics = self.gui_data['per_bin_metrics']
                        
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
                                    expected_bin_size = self.param.get('driftBinSize', 60)  # Default 60 seconds
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
                
                if self.param and self.param.get('computeTimeChunks', False):
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
                if (self.param and self.param.get('computeDrift', False) and 
                    hasattr(self, 'gui_data') and self.gui_data is not None):
                    
                    unit_idx = self.current_unit_idx
                    
                    # Check if per_bin_metrics exists and contains drift data
                    if 'per_bin_metrics' in self.gui_data:
                        per_bin_metrics = self.gui_data['per_bin_metrics']
                        
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
                                    expected_bin_size = self.param.get('driftBinSize', 60)  # Default 60 seconds
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
                
        ax.set_title('Amplitude (template scaling factor) over time', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        # Remove x-axis labels since time bin plot below will show them
        ax.set_xlabel('')
        ax.tick_params(labelsize=13, labelbottom=False)  # Hide x-axis labels
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Store y-limits for amplitude fit plot consistency
        self._amplitude_ylim = ax.get_ylim()
        
        # Add legend for time chunk coloring and drift if enabled
        import matplotlib.lines as mlines
        legend_elements = []
        
        # Add time chunk legend elements if computeTimeChunks is enabled
        if self.param and self.param.get('computeTimeChunks', False):
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
        
        # Add drift if computeDrift is enabled and drift data exists
        if (self.param and self.param.get('computeDrift', False) and 
            hasattr(self, 'gui_data') and self.gui_data is not None and
            'per_bin_metrics' in self.gui_data and 
            self.current_unit_idx in self.gui_data['per_bin_metrics'] and
            'drift' in self.gui_data['per_bin_metrics'][self.current_unit_idx]):
            
            # Add clarification if both drift and time chunks are computed
            if self.param.get('computeTimeChunks', False):
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
                             loc='upper center', ncol=ncols, fontsize=10,
                             framealpha=0.8, facecolor='white', edgecolor='black', 
                             prop={'family': 'DejaVu Sans'})
            legend.set_zorder(15)  # Ensure legend appears above plot elements
        
        # Add quality metrics text
        self.add_metrics_text(ax, unit_data, 'amplitude')
    
    def plot_time_bin_metrics(self, ax, unit_data):
        """Plot time bin metrics: presence ratio, RPV rate, and percentage spikes missing"""
        spike_times = unit_data['spike_times']
        metrics = unit_data['metrics']
        unit_id = unit_data['unit_id']
        
        if len(spike_times) > 0:
            # Try to get saved per-bin data from GUI precomputation
            per_bin_data = None
            if hasattr(self, 'gui_data') and self.gui_data is not None:
                per_bin_data = self.gui_data.get('per_bin_metrics', {}).get(unit_id)
            
            if per_bin_data and self.param.get('computeTimeChunks', False):
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
                        if hasattr(self, 'param') and 'RPV_tauR_estimate' in metrics:
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
                
                # Calculate presence ratio per bin using spike counts
                bin_counts, _ = np.histogram(spike_times, bins=time_bins)
                bin_width = time_bins[1] - time_bins[0]
                presence_threshold = 0.1 * np.mean(bin_counts) if np.mean(bin_counts) > 0 else 0.1
                presence_ratio = np.minimum(bin_counts / presence_threshold, 1.0)  # Cap at 1.0
                
                # Use useTheseTimesStart and useTheseTimesStop to determine good time chunks
                good_start_times = metrics.get('useTheseTimesStart', None)
                good_stop_times = metrics.get('useTheseTimesStop', None)
                
                # Add background coloring: orange by default, green for good time chunks
                for i, (start_time, end_time) in enumerate(zip(time_bins[:-1], time_bins[1:])):
                    # Default: orange background for bad chunks
                    chunk_color = 'orange'
                    
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
                                   fontsize=10, alpha=0.9, zorder=5)
                
            else:
                # Fallback: compute simplified metrics on the fly
                total_duration = np.max(spike_times) - np.min(spike_times)
                n_bins = max(20, int(total_duration / 60))  # ~1 minute bins, minimum 20 bins
                time_bins = np.linspace(np.min(spike_times), np.max(spike_times), n_bins + 1)
                bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
                bin_width = time_bins[1] - time_bins[0]
                
                # Calculate firing rate and presence ratio per bin
                bin_counts, _ = np.histogram(spike_times, bins=time_bins)
                
                # Presence ratio per bin (0 to 1)
                presence_threshold = 0.1 * np.mean(bin_counts) if np.mean(bin_counts) > 0 else 0.1
                presence_ratio = np.minimum(bin_counts / presence_threshold, 1.0)  # Cap at 1.0
                
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
            ax.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=3, fontsize=11,
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
        
    def plot_unit_location(self, ax, unit_data):
        """Plot all units by depth vs log firing rate, colored by classification"""
        # Define classification colors
        classification_colors = {
            'good': [0, 0.7, 0],        # Green
            'mua': [1, 0.5, 0],         # Orange  
            'noise': [1, 0, 0],         # Red
            'non-somatic': [0, 0, 1]    # Blue
        }
        
        if 'channel_positions' in self.ephys_data and 'maxChannels' in self.quality_metrics:
            positions = self.ephys_data['channel_positions']
            max_channels = self.quality_metrics['maxChannels']
            
            # Get all unit classifications and firing rates
            all_units = []
            all_depths = []
            all_firing_rates = []
            all_colors = []
            
            for i, unit_id in enumerate(self.unique_units):
                # Get max channel for this unit
                if i < len(max_channels):
                    max_ch = int(max_channels[i])
                    if max_ch < len(positions):
                        # Use Y position as depth (inverted for probe coordinates)
                        depth = positions[max_ch, 1]
                        
                        # Calculate firing rate for this unit
                        unit_spike_mask = self.ephys_data['spike_clusters'] == unit_id
                        unit_spike_times = self.ephys_data['spike_times'][unit_spike_mask]
                        
                        if len(unit_spike_times) > 0:
                            duration = np.max(unit_spike_times) - np.min(unit_spike_times)
                            if duration > 0:
                                firing_rate = len(unit_spike_times) / duration
                                
                                # Get classification
                                if self.unit_types is not None and i < len(self.unit_types):
                                    unit_type = self.unit_types[i]
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
                ax.invert_yaxis()  # Deeper = higher values, but show at bottom
                
                # Add legend
                legend_elements = []
                for class_name, color in classification_colors.items():
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=color, markersize=8, 
                                                    label=class_name))
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
                
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
                                closest_unit_idx = list(self.unique_units).index(unit_id)
                        
                        # Navigate to the closest unit if click is close enough
                        if min_distance < 0.1 and closest_unit_idx is not None:  # 10% of normalized plot area
                            self.current_unit_idx = closest_unit_idx
                            self.unit_slider.value = closest_unit_idx
                            print(f"Clicked on unit {closest_unit_idx} (unit_id: {self.unique_units[closest_unit_idx]})")
                
                # Store the click handler and connect it
                self._location_click_handler = on_location_click
                ax.figure.canvas.mpl_connect('button_press_event', self._location_click_handler)
                
            else:
                ax.text(0.5, 0.5, 'No units with\nvalid locations', 
                        ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Unit locations\n(requires probe geometry\nand max channels)', 
                    ha='center', va='center', transform=ax.transAxes, fontfamily="DejaVu Sans")
        
        ax.set_title('Units by depth', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        
    def plot_amplitude_fit(self, ax, unit_data):
        """Plot amplitude distribution with cutoff Gaussian fit like BombCell"""
        spike_times = unit_data['spike_times']
        metrics = unit_data['metrics']
        
        # Get y-limits from amplitude plot for consistency
        amp_ylim = None
        if hasattr(self, '_amplitude_ylim'):
            amp_ylim = self._amplitude_ylim
        
        if len(spike_times) > 0:
            # Get amplitudes if available
            unit_id = unit_data['unit_id']
            spike_mask = self.ephys_data['spike_clusters'] == unit_id
            
            if 'template_amplitudes' in self.ephys_data:
                amplitudes = self.ephys_data['template_amplitudes'][spike_mask]
                
                # Filter to good time chunks if computeTimeChunks is enabled
                if self.param and self.param.get('computeTimeChunks', False):
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
                    ax.set_ylabel('amplitude', fontsize=13, fontfamily="DejaVu Sans")
                    ax.tick_params(labelsize=13)
                    
                    # Set y-limits to match amplitude plot if available
                    if amp_ylim is not None:
                        ax.set_ylim(amp_ylim)
                        
                else:
                    ax.text(0.5, 0.5, 'Insufficient data\nfor amplitude fit', 
                            ha='center', va='center', transform=ax.transAxes)
            else:
                ax.text(0.5, 0.5, 'No amplitude data\navailable', 
                        ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'No spike data\navailable', 
                    ha='center', va='center', transform=ax.transAxes, fontfamily="DejaVu Sans")
                    
        ax.set_title('Amplitude distribution', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        
        # Add quality metrics text
        self.add_metrics_text(ax, unit_data, 'amplitude_fit')
        
    def add_metrics_text(self, ax, unit_data, plot_type):
        """Add quality metrics text overlay to plots like MATLAB with color coding"""
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
            elif metric_name in ['rawAmplitude', 'signalToNoiseRatio', 'fractionRPVs_estimatedTauR', 'presenceRatio', 'maxDriftEstimate', 'percentageSpikesMissing_gaussian']:
                if metric_name == 'rawAmplitude':
                    min_amp = param.get('minAmplitude', 50)
                    return 'orange' if val < min_amp else 'green'
                elif metric_name == 'signalToNoiseRatio':
                    min_snr = param.get('min_SNR', 3)
                    return 'orange' if val < min_snr else 'green'
                elif metric_name == 'fractionRPVs_estimatedTauR':
                    max_rpv = param.get('maxRPVviolations', 0.1)
                    return 'orange' if val > max_rpv else 'green'
                elif metric_name == 'presenceRatio':
                    min_presence = param.get('minPresenceRatio', 0.9)
                    return 'orange' if val < min_presence else 'green'
                elif metric_name == 'maxDriftEstimate':
                    max_drift = param.get('maxDrift', 100)
                    return 'orange' if val > max_drift else 'green'
                elif metric_name == 'percentageSpikesMissing_gaussian':
                    max_missing = param.get('maxPercSpikesMissing', 20)
                    return 'orange' if val > max_missing else 'green'
            
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
            metric_info = [
                ('maxDriftEstimate', f"Max drift: {format_metric(metrics.get('maxDriftEstimate'), 1)} μm"),
                ('presenceRatio', f"Presence ratio: {format_metric(metrics.get('presenceRatio'), 3)}")
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
                color = get_metric_color(metric_name, format_metric(metrics.get(metric_name)), self.param)
                y_pos = y_start - i * line_height
                
                # Skip if text would go below plot area
                if y_pos < 0.05:
                    break
                    
                ax.text(0.98, y_pos, text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right', fontsize=13, 
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
    
    def mark_peaks_and_troughs(self, ax, waveform, x_offset, y_offset, metrics, amp_range):
        """Mark all peaks and troughs on waveform with duration line"""
        
        if (self.gui_data and 
            'peak_locations' in self.gui_data and 
            'trough_locations' in self.gui_data and
            self.current_unit_idx in self.gui_data['peak_locations']):
            
            peaks = self.gui_data['peak_locations'][self.current_unit_idx]
            troughs = self.gui_data['trough_locations'][self.current_unit_idx]
            
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
                
                # Find all troughs (negative deflections) - similar stringent criteria
                trough_height_threshold = -np.min(waveform) * 0.5  # Increased from 0.2 to 0.5
                trough_prominence = waveform_range * 0.1
                troughs, trough_properties = find_peaks(-waveform, 
                                                       height=trough_height_threshold, 
                                                       distance=10,  # Increased from 5
                                                       prominence=trough_prominence)
            except ImportError:
                # Fallback without scipy
                max_idx = np.argmax(waveform)
                min_idx = np.argmin(waveform)
                peaks = [max_idx] if len(waveform) > 0 else []
                troughs = [min_idx] if len(waveform) > 0 else []
        
        # Plot peaks and troughs (works for both pre-computed and real-time data)
        # Find main peak and trough for special highlighting and duration calculation
        main_peak_idx = None
        main_trough_idx = None
        
        
        # Use pre-computed duration indices from quality metrics if available
        if (self.gui_data and 
            'peak_loc_for_duration' in self.gui_data and 
            'trough_loc_for_duration' in self.gui_data and
            self.current_unit_idx in self.gui_data['peak_loc_for_duration']):
            
            main_peak_idx = self.gui_data['peak_loc_for_duration'][self.current_unit_idx]
            main_trough_idx = self.gui_data['trough_loc_for_duration'][self.current_unit_idx]
            
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
        
        # Plot all troughs with blue dots
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
    
    def get_nearby_channels_for_spatial_decay(self, peak_channel, n_channels):
        """Get nearby channels for spatial decay plot - fewer points like MATLAB"""
        if 'channel_positions' in self.ephys_data:
            positions = self.ephys_data['channel_positions']
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
    
    def update_display(self):
        """Update the entire display"""
        self.update_unit_info()
        self.plot_unit(self.current_unit_idx)


class UnitQualityGUI:
    """
    GUI that exactly matches the MATLAB unitQualityGUI_synced layout
    """
    
    def __init__(self, ephys_data, quality_metrics, ephys_properties=None, 
                 raw_waveforms=None, param=None, unit_types=None):
        """
        Initialize the MATLAB-style GUI
        """
        self.ephys_data = ephys_data
        self.quality_metrics = quality_metrics
        self.ephys_properties = ephys_properties or []
        self.raw_waveforms = raw_waveforms
        self.param = param or {}
        self.unit_types = unit_types
        
        # Get unique units
        self.unique_units = np.unique(ephys_data['spike_clusters'])
        self.n_units = len(self.unique_units)
        self.current_unit_idx = 0
        
        # Create the GUI with exact MATLAB layout
        self.setup_gui()
        self.update_unit_display()
        
    def setup_gui(self):
        """Set up the GUI layout exactly like MATLAB"""
        # Create main figure with exact MATLAB proportions
        self.fig = plt.figure(figsize=(18, 12))
        self.fig.patch.set_facecolor('white')
        
        # Use subplot with exact MATLAB grid (6x13)
        self.setup_matlab_layout()
        
        # Setup navigation
        self.setup_navigation()
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def setup_matlab_layout(self):
        """Setup the exact MATLAB subplot layout (6x13 grid)"""
        
        # 1. Unit location plot (left column) - subplot(6, 13, [1, 14, 27, 40, 53, 66])
        self.ax_location = plt.subplot2grid((6, 13), (0, 0), rowspan=6, colspan=1)
        self.setup_location_plot()
        
        # 2. Template waveforms - subplot(6, 13, [2:7, 15:20])
        self.ax_template = plt.subplot2grid((6, 13), (0, 1), rowspan=2, colspan=6)
        self.setup_template_plot()
        
        # 3. Raw waveforms - subplot(6, 13, [8:13, 21:26])
        self.ax_raw = plt.subplot2grid((6, 13), (0, 7), rowspan=2, colspan=6)
        self.setup_raw_plot()
        
        # 4. Spatial decay - subplot(6, 13, 29:33)
        self.ax_spatial = plt.subplot2grid((6, 13), (2, 1), rowspan=2, colspan=6)
        self.setup_spatial_plot()
        
        # 5. ACG - subplot(6, 13, 35:39)
        self.ax_acg = plt.subplot2grid((6, 13), (2, 7), rowspan=2, colspan=6)
        self.setup_acg_plot()
        
        # 6. Amplitudes over time - subplot(6, 13, [55:57, 68:70, 74:76])
        self.ax_amplitude = plt.subplot2grid((6, 13), (4, 1), rowspan=2, colspan=9)
        self.setup_amplitude_plot()
        
        # 7. Amplitude fit - subplot(6, 13, [78])
        self.ax_amp_fit = plt.subplot2grid((6, 13), (4, 11), rowspan=2, colspan=2)
        self.setup_amplitude_fit_plot()
        
    def setup_location_plot(self):
        """Setup unit location on probe plot"""
        self.ax_location.set_title('Location on probe', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        self.ax_location.set_xlabel('Norm. log rate', fontsize=13, fontfamily="DejaVu Sans")
        self.ax_location.set_ylabel('Depth from tip (μm)', fontsize=13, fontfamily="DejaVu Sans")
        self.ax_location.tick_params(labelsize=13)
        self.ax_location.invert_yaxis()  # MATLAB uses 'YDir', 'reverse'
        
    def setup_template_plot(self):
        """Setup template waveform plot"""
        self.ax_template.set_title('Template waveforms', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        self.ax_template.set_xticks([])
        self.ax_template.set_yticks([])
        self.ax_template.invert_yaxis()
        
    def setup_raw_plot(self):
        """Setup raw waveform plot"""
        self.ax_raw.set_title('Raw waveforms', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        self.ax_raw.set_xticks([])
        self.ax_raw.set_yticks([])
        self.ax_raw.invert_yaxis()
        
    def setup_spatial_plot(self):
        """Setup spatial decay plot"""
        self.ax_spatial.set_title('Spatial decay', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        self.ax_spatial.set_ylabel('Ampli. (a.u.)', fontsize=13, fontfamily="DejaVu Sans")
        self.ax_spatial.set_xlabel('Distance', fontsize=13, fontfamily="DejaVu Sans")
        self.ax_spatial.tick_params(labelsize=13)
        
    def setup_acg_plot(self):
        """Setup auto-correlogram plot"""
        self.ax_acg.set_title('Auto-correlogram', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        self.ax_acg.set_xlabel('Time (ms)', fontsize=13, fontfamily="DejaVu Sans")
        self.ax_acg.set_ylabel('sp/s', fontsize=13, fontfamily="DejaVu Sans")
        self.ax_acg.tick_params(labelsize=13)
        
    def setup_amplitude_plot(self):
        """Setup amplitude over time plot"""
        self.ax_amplitude.set_title('Amplitudes over time', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        self.ax_amplitude.set_xlabel('Experiment time (s)', fontsize=13, fontfamily="DejaVu Sans")
        self.ax_amplitude.tick_params(labelsize=13)
        # Dual y-axis like MATLAB
        self.ax_amplitude_right = self.ax_amplitude.twinx()
        self.ax_amplitude.set_ylabel('Template scaling', color='k', fontsize=13, fontfamily="DejaVu Sans")
        self.ax_amplitude_right.set_ylabel('Firing rate (sp/sec)', color='magenta', fontsize=13, fontfamily="DejaVu Sans")
        self.ax_amplitude_right.tick_params(labelsize=13)
        
    def setup_amplitude_fit_plot(self):
        """Setup amplitude fit plot"""
        self.ax_amp_fit.set_title('Amplitude fit', fontsize=15, fontweight='bold', fontfamily="DejaVu Sans")
        
    def setup_navigation(self):
        """Setup navigation buttons at bottom"""
        # Create navigation area at bottom
        nav_height = 0.05
        button_width = 0.08
        button_height = 0.03
        y_pos = 0.01
        
        # Navigation buttons
        self.btn_prev = Button(plt.axes([0.1, y_pos, button_width, button_height]), 
                              '← Previous', color='lightblue')
        self.btn_next = Button(plt.axes([0.2, y_pos, button_width, button_height]), 
                              'Next →', color='lightblue')
        self.btn_good = Button(plt.axes([0.35, y_pos, button_width, button_height]), 
                              'Good Units', color='lightgreen')
        self.btn_mua = Button(plt.axes([0.45, y_pos, button_width, button_height]), 
                             'MUA Units', color='orange')
        self.btn_noise = Button(plt.axes([0.55, y_pos, button_width, button_height]), 
                               'Noise Units', color='lightcoral')
        
        # Unit number display
        self.unit_text = plt.figtext(0.7, y_pos + 0.015, '', fontsize=12, weight='bold')
        
        # Connect callbacks
        self.btn_prev.on_clicked(self.prev_unit)
        self.btn_next.on_clicked(self.next_unit)
        self.btn_good.on_clicked(self.goto_next_good)
        self.btn_mua.on_clicked(self.goto_next_mua)
        self.btn_noise.on_clicked(self.goto_next_noise)
        
    def get_unit_data(self, unit_idx):
        """Get data for a specific unit"""
        if unit_idx >= self.n_units:
            return None
            
        unit_id = self.unique_units[unit_idx]
        
        # Get spike times for this unit
        spike_mask = self.ephys_data['spike_clusters'] == unit_id
        spike_times = self.ephys_data['spike_times'][spike_mask]
        
        # Get template waveform
        if unit_idx < len(self.ephys_data['template_waveforms']):
            template = self.ephys_data['template_waveforms'][unit_idx]
        else:
            template = np.zeros((82, 1))
            
        # Get quality metrics for this unit
        unit_metrics = {}
        for key, values in self.quality_metrics.items():
            if hasattr(values, '__len__') and len(values) > unit_idx:
                unit_metrics[key] = values[unit_idx]
            else:
                unit_metrics[key] = np.nan
                
        # Get ephys properties if available
        unit_ephys = {}
        if self.ephys_properties and unit_idx < len(self.ephys_properties):
            unit_ephys = self.ephys_properties[unit_idx]
            
        return {
            'unit_id': unit_id,
            'spike_times': spike_times,
            'template': template,
            'metrics': unit_metrics,
            'ephys': unit_ephys
        }
        
    def update_unit_display(self):
        """Update all plots for current unit"""
        if self.current_unit_idx >= self.n_units:
            return
            
        unit_data = self.get_unit_data(self.current_unit_idx)
        if unit_data is None:
            return
            
        # Clear all axes
        for ax in [self.ax_location, self.ax_template, self.ax_raw, self.ax_spatial, 
                  self.ax_acg, self.ax_amplitude, self.ax_amp_fit]:
            ax.clear()
            
        # Re-setup axes after clearing
        self.setup_location_plot()
        self.setup_template_plot()
        self.setup_raw_plot()
        self.setup_spatial_plot()
        self.setup_acg_plot()
        self.setup_amplitude_plot()
        self.setup_amplitude_fit_plot()
        
        # Update plots
        self.plot_unit_location(unit_data)
        self.plot_template_waveform(unit_data)
        self.plot_raw_waveform(unit_data)
        self.plot_spatial_decay(unit_data)
        self.plot_acg(unit_data)
        self.plot_amplitudes(unit_data)
        self.plot_amplitude_fit(unit_data)
        
        # Update main title
        unit_type_str = self.get_unit_type_string(self.current_unit_idx)
        main_title = f'Unit {unit_data["unit_id"]} ({unit_type_str}) - {self.current_unit_idx+1}/{self.n_units}'
        self.fig.suptitle(main_title, fontsize=16, weight='bold')
        
        # Update unit text
        self.unit_text.set_text(f'Unit: {self.current_unit_idx+1}/{self.n_units}')
        
        # Refresh display
        self.fig.canvas.draw()
        
    def plot_unit_location(self, unit_data):
        """Plot unit location on probe (exact MATLAB style)"""
        # Get all units for context
        all_spike_counts = []
        all_depths = []
        all_colors = []
        
        # Color mapping like MATLAB
        color_map = {0: [1, 0, 0],      # red for noise
                    1: [0, 0.5, 0],     # green for good
                    2: [1, 0.55, 0],    # orange for MUA
                    3: [0.25, 0.41, 0.88]}  # blue for non-soma
        
        for i, unit_id in enumerate(self.unique_units):
            spike_mask = self.ephys_data['spike_clusters'] == unit_id
            spike_count = np.sum(spike_mask)
            
            # Get depth (channel position)
            if 'maxChannels' in self.quality_metrics and i < len(self.quality_metrics['maxChannels']):
                max_chan = int(self.quality_metrics['maxChannels'][i])
                if max_chan < len(self.ephys_data['channel_positions']):
                    depth = self.ephys_data['channel_positions'][max_chan, 1]
                else:
                    depth = 0
            else:
                depth = i * 20  # fallback
                
            all_spike_counts.append(spike_count)
            all_depths.append(depth)
            
            # Get color
            if self.unit_types is not None and i < len(self.unit_types):
                unit_type = self.unit_types[i]
                color = color_map.get(unit_type, [0.5, 0.5, 0.5])
            else:
                color = [0.5, 0.5, 0.5]
            all_colors.append(color)
        
        # Normalize spike counts like MATLAB
        norm_spike_counts = np.array(all_spike_counts)
        if len(norm_spike_counts) > 0:
            norm_spike_counts = (np.log10(norm_spike_counts + 1) - 
                               np.min(np.log10(norm_spike_counts + 1))) / \
                              (np.max(np.log10(norm_spike_counts + 1)) - 
                               np.min(np.log10(norm_spike_counts + 1)))
        
        # Plot all units
        self.ax_location.scatter(norm_spike_counts, all_depths, c=all_colors, s=20, alpha=0.7)
        
        # Highlight current unit
        if self.current_unit_idx < len(norm_spike_counts):
            current_color = all_colors[self.current_unit_idx]
            self.ax_location.scatter(norm_spike_counts[self.current_unit_idx], 
                                   all_depths[self.current_unit_idx],
                                   c=[current_color], s=100, edgecolors='black', linewidth=3)
        
        self.ax_location.set_xlim([-0.1, 1.1])
        if all_depths:
            self.ax_location.set_ylim([min(all_depths) - 50, max(all_depths) + 50])
        
    def plot_template_waveform(self, unit_data):
        """Plot template waveform (exact MATLAB style)"""
        template = unit_data['template']
        if template.size > 0:
            n_channels = min(template.shape[1], 20)  # Max 20 channels like MATLAB
            
            # Plot multiple channels with offset
            for ch in range(n_channels):
                waveform = template[:, ch]
                time_axis = np.arange(len(waveform))
                
                # Add vertical offset for each channel
                offset = ch * 50
                self.ax_template.plot(time_axis, waveform + offset, 'k-', linewidth=1, alpha=0.7)
            
            # Highlight peak channel
            peak_chan = np.argmin(np.min(template, axis=0))
            if peak_chan < n_channels:
                waveform = template[:, peak_chan]
                time_axis = np.arange(len(waveform))
                offset = peak_chan * 50
                self.ax_template.plot(time_axis, waveform + offset, 'b-', linewidth=2)
        
    def plot_raw_waveform(self, unit_data):
        """Plot raw waveform if available"""
        # Placeholder for raw waveforms
        self.ax_raw.text(0.5, 0.5, 'Raw waveforms\n(Not available)', 
                        ha='center', va='center', transform=self.ax_raw.transAxes,
                        fontsize=12, alpha=0.6)
        
    def plot_spatial_decay(self, unit_data):
        """Plot spatial decay"""
        # Placeholder implementation
        x = np.linspace(0, 100, 10)
        y = np.exp(-x/50) + np.random.normal(0, 0.1, len(x))
        self.ax_spatial.scatter(x, y, c='black', s=20)
        
        # Add exponential fit
        fit_x = np.linspace(0, 100, 100)
        fit_y = np.exp(-fit_x/50)
        self.ax_spatial.plot(fit_x, fit_y, 'r-', linewidth=2)
        
    def plot_acg(self, unit_data):
        """Plot auto-correlogram (exact MATLAB style)"""
        spike_times = unit_data['spike_times']
        
        if len(spike_times) > 10:
            # Compute ACG like MATLAB
            max_lag = 50  # ms
            bin_size = 1  # ms
            
            # Convert to milliseconds
            spike_times_ms = spike_times * 1000
            
            # Compute ISIs
            isis = np.diff(spike_times_ms)
            
            # Create histogram
            bins = np.arange(0, max_lag + bin_size, bin_size)
            hist, _ = np.histogram(isis[isis <= max_lag], bins=bins)
            
            # Plot as bar chart like MATLAB
            bin_centers = bins[:-1] + bin_size/2
            self.ax_acg.bar(bin_centers, hist, width=bin_size*0.8, color='grey', alpha=0.7)
            
            # Add refractory period line
            self.ax_acg.axvline(x=2, color='r', linestyle='--', linewidth=2, alpha=0.8)
            
            # Set limits
            self.ax_acg.set_xlim([0, max_lag])
        
    def plot_amplitudes(self, unit_data):
        """Plot amplitudes over time (exact MATLAB style with dual y-axis)"""
        spike_times = unit_data['spike_times']
        
        if len(spike_times) > 10:
            # Compute firing rate over time
            bin_size = 60  # seconds
            if len(spike_times) > 0:
                time_bins = np.arange(spike_times.min(), spike_times.max() + bin_size, bin_size)
                hist, _ = np.histogram(spike_times, bins=time_bins)
                firing_rates = hist / bin_size
                bin_centers = time_bins[:-1] + bin_size/2
                
                # Plot on right y-axis (magenta, avoiding classification colors)
                self.ax_amplitude_right.stairs(firing_rates, time_bins, color='magenta', linewidth=2)
                
            # Simulate template amplitudes (left y-axis)
            n_points = min(len(spike_times), 1000)
            indices = np.linspace(0, len(spike_times)-1, n_points, dtype=int)
            sim_times = spike_times[indices]
            sim_amplitudes = 1.0 + np.random.normal(0, 0.1, len(sim_times))
            
            self.ax_amplitude.scatter(sim_times, sim_amplitudes, c='black', s=1, alpha=0.6)
            
            # Add trend line
            if len(sim_times) > 1:
                z = np.polyfit(sim_times, sim_amplitudes, 1)
                p = np.poly1d(z)
                self.ax_amplitude.plot(sim_times, p(sim_times), 'g-', linewidth=2)
        
    def plot_amplitude_fit(self, unit_data):
        """Plot amplitude distribution fit"""
        # Simulate amplitude distribution
        amplitudes = np.random.normal(1.0, 0.2, 1000)
        
        # Create histogram
        hist, bins = np.histogram(amplitudes, bins=30)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Plot as horizontal bar chart
        self.ax_amp_fit.barh(bin_centers, hist, height=np.diff(bins)[0]*0.8, 
                            color='blue', alpha=0.5)
        
        # Add Gaussian fit
        from scipy import stats
        mu, sigma = stats.norm.fit(amplitudes)
        fit_y = np.linspace(amplitudes.min(), amplitudes.max(), 100)
        fit_x = len(amplitudes) * np.diff(bins)[0] * stats.norm.pdf(fit_y, mu, sigma)
        self.ax_amp_fit.plot(fit_x, fit_y, color='gold', linewidth=2)
        
    def get_unit_type_string(self, unit_idx):
        """Get unit type string"""
        if self.unit_types is not None and unit_idx < len(self.unit_types):
            unit_type = self.unit_types[unit_idx]
            type_map = {0: 'NOISE', 1: 'GOOD', 2: 'MUA', 3: 'NON-SOMA'}
            return type_map.get(unit_type, 'UNKNOWN')
        return 'UNKNOWN'
        
    # Navigation callbacks (fixed to properly update display)
    def prev_unit(self, event):
        """Go to previous unit"""
        if self.current_unit_idx > 0:
            self.current_unit_idx -= 1
            self.update_unit_display()
            
    def next_unit(self, event):
        """Go to next unit"""
        if self.current_unit_idx < self.n_units - 1:
            self.current_unit_idx += 1
            self.update_unit_display()
            
    def goto_next_good(self, event):
        """Go to next good unit"""
        if self.unit_types is not None:
            good_indices = np.where(self.unit_types == 1)[0]
            next_good = good_indices[good_indices > self.current_unit_idx]
            if len(next_good) > 0:
                self.current_unit_idx = next_good[0]
                self.update_unit_display()
                
    def goto_next_mua(self, event):
        """Go to next MUA unit"""
        if self.unit_types is not None:
            mua_indices = np.where(self.unit_types == 2)[0]
            next_mua = mua_indices[mua_indices > self.current_unit_idx]
            if len(next_mua) > 0:
                self.current_unit_idx = next_mua[0]
                self.update_unit_display()
                
    def goto_next_noise(self, event):
        """Go to next noise unit"""
        if self.unit_types is not None:
            noise_indices = np.where(self.unit_types == 0)[0]
            next_noise = noise_indices[noise_indices > self.current_unit_idx]
            if len(next_noise) > 0:
                self.current_unit_idx = next_noise[0]
                self.update_unit_display()
                
    def on_key_press(self, event):
        """Handle keyboard shortcuts"""
        if event.key == 'right':
            self.next_unit(None)
        elif event.key == 'left':
            self.prev_unit(None)
        elif event.key == 'g':
            self.goto_next_good(None)
        elif event.key == 'm':
            self.goto_next_mua(None)
        elif event.key == 'n':
            self.goto_next_noise(None)


def load_metrics_for_gui(ks_dir, quality_metrics, ephys_properties=None, param=None):
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
        'spike_times': ephys_data_tuple[0] / 30000.0,  # Convert to seconds
        'spike_clusters': ephys_data_tuple[1],
        'template_waveforms': ephys_data_tuple[2],
        'template_amplitudes': ephys_data_tuple[3],
        'pc_features': ephys_data_tuple[4],
        'pc_features_idx': ephys_data_tuple[5],
        'channel_positions': ephys_data_tuple[6]
    }
    
    # Load raw waveforms if available
    raw_waveforms = None
    raw_wf_path = Path(ks_dir) / "bombcell" / "templates._bc_rawWaveforms.npy"
    if raw_wf_path.exists():
        try:
            raw_waveforms = {
                'average': np.load(raw_wf_path, allow_pickle=True),
                'peak_channels': np.load(Path(ks_dir) / "bombcell" / "templates._bc_rawWaveformPeakChannels.npy", allow_pickle=True)
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


def unit_quality_gui(ephys_data_or_path=None, quality_metrics=None, ephys_properties=None, 
                     unit_types=None, param=None, ks_dir=None):
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
        
    Returns
    -------
    UnitQualityGUI
        The GUI object
    """
    # Handle backward compatibility with ks_dir parameter
    if ks_dir is not None:
        ephys_data_or_path = ks_dir
    
    # Check if input is a path or already loaded data
    if isinstance(ephys_data_or_path, dict):
        # Data is already loaded
        ephys_data = ephys_data_or_path
        raw_waveforms = None  # Would need to be passed separately if needed
        if param is None:
            param = {}
    else:
        # Load data from path
        gui_data = load_metrics_for_gui(ephys_data_or_path, quality_metrics, ephys_properties, param)
        ephys_data = gui_data['ephys_data']
        raw_waveforms = gui_data['raw_waveforms']
        param = gui_data['param']
    
    # Create and return GUI - use interactive version if ipywidgets is available
    if IPYWIDGETS_AVAILABLE:
        gui = InteractiveUnitQualityGUI(
            ephys_data=ephys_data,
            quality_metrics=quality_metrics,
            ephys_properties=ephys_properties,
            raw_waveforms=raw_waveforms,
            param=param,
            unit_types=unit_types
        )
    else:
        gui = UnitQualityGUI(
            ephys_data=ephys_data,
            quality_metrics=quality_metrics,
            ephys_properties=ephys_properties,
            raw_waveforms=raw_waveforms,
            param=param,
            unit_types=unit_types
        )
    
    return gui