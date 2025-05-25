import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
from pathlib import Path
import pandas as pd

try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False


class InteractiveUnitQualityGUI:
    """
    Interactive GUI using ipywidgets for Jupyter notebooks
    """
    
    def __init__(self, ephys_data, quality_metrics, ephys_properties=None, 
                 raw_waveforms=None, param=None, unit_types=None):
        """
        Initialize the interactive GUI
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
        
        # Navigation buttons
        self.prev_btn = widgets.Button(description='← unit', button_style='info')
        self.next_btn = widgets.Button(description='unit →', button_style='info')
        
        # Unit type navigation - both directions
        self.goto_prev_good_btn = widgets.Button(description='← good', button_style='success')
        self.goto_good_btn = widgets.Button(description='good →', button_style='success')
        self.goto_prev_mua_btn = widgets.Button(description='← mua', button_style='warning')
        self.goto_mua_btn = widgets.Button(description='mua →', button_style='warning')
        self.goto_prev_noise_btn = widgets.Button(description='← noise', button_style='danger')
        self.goto_noise_btn = widgets.Button(description='noise →', button_style='danger')
        self.goto_prev_nonsomatic_btn = widgets.Button(description='← non-soma', button_style='primary')
        self.goto_nonsomatic_btn = widgets.Button(description='non-soma →', button_style='primary')
        
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
        # Navigation controls
        nav_controls = widgets.HBox([
            self.prev_btn, self.next_btn, 
            widgets.Label('  |  '),
            self.goto_prev_good_btn, self.goto_good_btn,
            self.goto_prev_mua_btn, self.goto_mua_btn,
            self.goto_prev_noise_btn, self.goto_noise_btn,
            self.goto_prev_nonsomatic_btn, self.goto_nonsomatic_btn
        ])
        
        # Unit input controls
        unit_input_controls = widgets.HBox([
            self.unit_input, self.goto_unit_btn
        ])
        
        # Classification controls (hidden for now)
        # classify_controls = widgets.HBox([
        #     self.classify_good_btn, self.classify_mua_btn, self.classify_noise_btn
        # ])
        
        # Full interface
        interface = widgets.VBox([
            self.unit_slider,
            unit_input_controls,
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
        for key, values in self.quality_metrics.items():
            if hasattr(values, '__len__') and len(values) > unit_idx:
                unit_metrics[key] = values[unit_idx]
            else:
                unit_metrics[key] = np.nan
                
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
        
        # Simple title with unit number, phy ID, and type, colored by classification
        info_html = f"""
        <h3 style="color: {title_color};">Unit {unit_data['unit_id']} (phy ID = {self.current_unit_idx}, {self.current_unit_idx+1}/{self.n_units}) - {unit_type_str}</h3>
        """
        
        self.unit_info.value = info_html
        
    def plot_unit(self, unit_idx):
        """Plot data for a specific unit"""
        unit_data = self.get_unit_data(unit_idx)
        if unit_data is None:
            return
            
        with self.plot_output:
            clear_output(wait=True)
            
            # Create figure with exact MATLAB layout (6x13 grid)
            fig = plt.figure(figsize=(18, 12))
            fig.patch.set_facecolor('white')
            
            # 1. Unit location plot (left column) - subplot(6, 13, [1, 14, 27, 40, 53, 66])
            ax_location = plt.subplot2grid((6, 13), (0, 0), rowspan=6, colspan=1)
            self.plot_unit_location(ax_location, unit_data)
            
            # 2. Template waveforms - subplot(6, 13, [2:7, 15:20])
            ax_template = plt.subplot2grid((6, 13), (0, 1), rowspan=2, colspan=6)
            self.plot_template_waveform(ax_template, unit_data)
            
            # 3. Raw waveforms - subplot(6, 13, [8:13, 21:26])
            ax_raw = plt.subplot2grid((6, 13), (0, 7), rowspan=2, colspan=6)
            self.plot_raw_waveforms(ax_raw, unit_data)
            
            # 4. Spatial decay - subplot(6, 13, 29:33)
            ax_spatial = plt.subplot2grid((6, 13), (2, 1), rowspan=2, colspan=6)
            self.plot_spatial_decay(ax_spatial, unit_data)
            
            # 5. ACG - subplot(6, 13, 35:39)
            ax_acg = plt.subplot2grid((6, 13), (2, 7), rowspan=2, colspan=6)
            self.plot_autocorrelogram(ax_acg, unit_data)
            
            # 6. Amplitudes over time - subplot(6, 13, [42:44, 55:57, 68:70])
            ax_amplitude = plt.subplot2grid((6, 13), (4, 1), rowspan=2, colspan=9)
            self.plot_amplitudes_over_time(ax_amplitude, unit_data)
            
            # 7. Amplitude fit - subplot(6, 13, [45:46, 58:59, 71:72])
            ax_amp_fit = plt.subplot2grid((6, 13), (4, 11), rowspan=2, colspan=2)
            self.plot_amplitude_fit(ax_amp_fit, unit_data)
            
            plt.tight_layout()
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
                            
                            # Calculate X offset - use waveform width for proper side-by-side spacing
                            waveform_width = template.shape[0]  # Usually 82 samples
                            x_offset = (ch_pos[0] - max_pos[0]) * waveform_width * 0.05  # Reduced spacing
                            
                            # Calculate Y offset based on channel Y position (like MATLAB)
                            y_offset = (ch_pos[1] - max_pos[1]) / 100 * scaling_factor
                            
                            # Plot waveform
                            x_vals = time_axis + x_offset
                            y_vals = -waveform + y_offset  # Negative like MATLAB
                            
                            if ch == max_ch:
                                ax.plot(x_vals, y_vals, 'k-', linewidth=3)  # Max channel thicker black
                            else:
                                ax.plot(x_vals, y_vals, 'k-', linewidth=1, alpha=0.7)
                            
                            # Add channel number
                            ax.text(x_offset - 2, y_offset, f'{ch}', fontsize=8, ha='right', va='center')
                    
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
                       ha='center', va='center', transform=ax.transAxes)
                    
        ax.set_title('Template waveforms')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add quality metrics text
        self.add_metrics_text(ax, unit_data, 'template')
        
    def plot_raw_waveforms(self, ax, unit_data):
        """Plot raw waveforms with 16 nearest channels like MATLAB"""
        metrics = unit_data['metrics']
        
        # Check if raw extraction is enabled
        extract_raw = self.param.get('extractRaw', 0)
        if extract_raw != 1:
            ax.text(0.5, 0.5, 'Raw waveforms\n(extractRaw disabled)', 
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Raw waveforms')
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
                                            
                                            # Calculate X offset - use waveform width for proper side-by-side spacing
                                            waveform_width = waveforms.shape[0]  # Usually 82 samples
                                            x_offset = (ch_pos[0] - max_pos[0]) * waveform_width * 0.05  # Reduced spacing
                                            
                                            # Calculate Y offset based on channel Y position (like MATLAB)
                                            y_offset = (ch_pos[1] - max_pos[1]) / 100 * scaling_factor
                                            
                                            # Plot waveform
                                            x_vals = time_axis + x_offset
                                            y_vals = -waveform + y_offset  # Negative like MATLAB
                                            
                                            if ch == max_ch:
                                                ax.plot(x_vals, y_vals, 'k-', linewidth=3)  # Max channel thicker black
                                            else:
                                                ax.plot(x_vals, y_vals, 'gray', linewidth=1, alpha=0.7)
                                            
                                            # Add channel number
                                            ax.text(x_offset - 2, y_offset, f'{ch}', fontsize=8, ha='right', va='center')
                                    
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
                                       ha='center', va='center', transform=ax.transAxes)
                        else:
                            # Single channel
                            ax.plot(waveforms, 'b-', alpha=0.7)
                            
                except (TypeError, IndexError, AttributeError):
                    ax.text(0.5, 0.5, 'Raw waveforms\n(data format issue)', 
                            ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Raw waveforms\n(not available)', 
                    ha='center', va='center', transform=ax.transAxes)
                    
        ax.set_title('Raw waveforms')
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove aspect ratio constraint to prevent squishing
        
        # Add quality metrics text
        self.add_metrics_text(ax, unit_data, 'raw')
        
    def plot_autocorrelogram(self, ax, unit_data):
        """Plot autocorrelogram with tauR and firing rate lines"""
        spike_times = unit_data['spike_times']
        metrics = unit_data['metrics']
        
        if len(spike_times) > 1:
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
            if len(spike_times) > 10000:
                indices = np.random.choice(len(spike_times), 10000, replace=False)
                spike_subset = spike_times[indices]
            else:
                spike_subset = spike_times
            
            # Calculate cross-correlation with itself
            for i, spike_time in enumerate(spike_subset[::10]):  # Subsample further for speed
                # Find spikes within max_lag of this spike
                time_diffs = spike_times - spike_time
                valid_diffs = time_diffs[(np.abs(time_diffs) <= max_lag) & (time_diffs != 0)]
                
                if len(valid_diffs) > 0:
                    hist, _ = np.histogram(valid_diffs, bins=bins)
                    autocorr += hist
            
            # Convert to firing rate (spikes/sec)
            if len(spike_subset) > 0:
                recording_duration = np.max(spike_times) - np.min(spike_times)
                autocorr = autocorr / (len(spike_subset) * bin_size) if recording_duration > 0 else autocorr
            
            # Plot only positive lags (like MATLAB)
            positive_mask = bin_centers >= 0
            positive_centers = bin_centers[positive_mask]
            positive_autocorr = autocorr[positive_mask]
            
            # Plot with wider bars like MATLAB
            ax.bar(positive_centers * 1000, positive_autocorr, 
                   width=bin_size*1000*0.9, color='grey', alpha=0.7, edgecolor='black', linewidth=0.5)
            
            # Calculate mean firing rate first to determine proper y-limits
            mean_fr = 0
            if len(spike_times) > 1:
                recording_duration = np.max(spike_times) - np.min(spike_times)
                if recording_duration > 0:
                    mean_fr = len(spike_times) / recording_duration
            
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
            
            # Add tauR vertical line - use milliseconds and ensure it's visible
            tau_r = metrics.get('tauR_estimated', None)
            if tau_r is None:
                # Try alternative parameter names
                tau_r = metrics.get('estimatedTauR', None)
            if tau_r is None:
                # Use a default value if not available
                tau_r = 2.0  # 2ms default refractory period
                
            if tau_r is not None:
                ax.axvline(tau_r, color='red', linewidth=3, linestyle='--', alpha=1.0, 
                          label=f'τR = {tau_r:.1f}ms', zorder=10)
            
            # Add mean firing rate horizontal line
            if mean_fr > 0:
                ax.axhline(mean_fr, color='orange', linewidth=3, linestyle='--', alpha=1.0, 
                          label=f'Mean FR = {mean_fr:.1f} sp/s', zorder=10)
                
        ax.set_title('Auto-correlogram')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Firing rate (sp/s)')
        
        # Add legend in bottom right corner
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc='lower right', fontsize=8, framealpha=0.9)
        
        # Add quality metrics text
        self.add_metrics_text(ax, unit_data, 'acg')
        
    def plot_spatial_decay(self, ax, unit_data):
        """Plot spatial decay like MATLAB - only nearby channels"""
        metrics = unit_data['metrics']
        
        # Check if spatial decay metrics are available
        if 'spatialDecaySlope' in metrics and not np.isnan(metrics['spatialDecaySlope']):
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
                            ax.scatter(distances, amplitudes, s=30, alpha=0.8, color='blue', edgecolor='black')
                            
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
                                ax.plot(x_smooth[valid_y], y_smooth[valid_y], 'r-', linewidth=2, alpha=0.8)
                            
                            ax.set_xlabel('Distance (μm)')
                            ax.set_ylabel('Normalized amplitude')
                            
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
                    ha='center', va='center', transform=ax.transAxes)
                    
        ax.set_title('Spatial decay')
        
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
            
            # Calculate presence ratio per bin (assuming threshold of >0 spikes for presence)
            presence_threshold = 0.1 * np.mean(bin_counts)  # 10% of mean rate
            good_presence = bin_counts > presence_threshold
            
            if 'template_amplitudes' in self.ephys_data:
                amplitudes = self.ephys_data['template_amplitudes'][spike_mask]
                
                # Plot amplitudes with slightly bigger dots
                ax.scatter(spike_times, amplitudes, s=3, alpha=0.6, color='black', edgecolors='none')
                ax.set_ylabel('Template scaling', color='blue')
                
                # Create twin axis for firing rate
                ax2 = ax.twinx()
                
                # Plot firing rate as step plot (outline only)
                ax2.step(bin_centers, firing_rates, where='mid', color='orange', 
                        linewidth=2.5, alpha=0.8, label='Firing rate')
                ax2.set_ylabel('Firing rate (sp/s)', color='orange')
                ax2.tick_params(axis='y', labelcolor='orange')
                
                # Highlight time bins with good presence ratio
                for i, (center, good) in enumerate(zip(bin_centers, good_presence)):
                    if good:
                        # Add subtle green background for good presence bins
                        y_min, y_max = ax.get_ylim()
                        ax.axvspan(center - bin_width/2, center + bin_width/2, 
                                  alpha=0.1, color='green', zorder=0)
                
            else:
                # Just plot spike times as raster with bigger dots
                y_pos = np.ones_like(spike_times)
                ax.scatter(spike_times, y_pos, s=3, alpha=0.6, color='black', edgecolors='none')
                ax.set_ylabel('Spikes', color='blue')
                
                # Create twin axis for firing rate  
                ax2 = ax.twinx()
                ax2.step(bin_centers, firing_rates, where='mid', color='orange', 
                        linewidth=2.5, alpha=0.8, label='Firing rate')
                ax2.set_ylabel('Firing rate (sp/s)', color='orange')
                ax2.tick_params(axis='y', labelcolor='orange')
                
                # Highlight good presence bins
                for i, (center, good) in enumerate(zip(bin_centers, good_presence)):
                    if good:
                        y_min, y_max = ax.get_ylim()
                        ax.axvspan(center - bin_width/2, center + bin_width/2, 
                                  alpha=0.1, color='green', zorder=0)
                
        ax.set_title('Amplitudes over time')
        ax.set_xlabel('Time (s)')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Store y-limits for amplitude fit plot consistency
        self._amplitude_ylim = ax.get_ylim()
        
        # Add quality metrics text
        self.add_metrics_text(ax, unit_data, 'amplitude')
        
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
                
                ax.set_xlabel('Log₁₀ firing rate (sp/s)')
                ax.set_ylabel('Depth (μm)')
                ax.invert_yaxis()  # Deeper = higher values, but show at bottom
                
                # Add legend
                legend_elements = []
                for class_name, color in classification_colors.items():
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                    markerfacecolor=color, markersize=8, 
                                                    label=class_name))
                ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
                
            else:
                ax.text(0.5, 0.5, 'No units with\nvalid locations', 
                        ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, 'Unit locations\n(requires probe geometry\nand max channels)', 
                    ha='center', va='center', transform=ax.transAxes)
        
        ax.set_title('Units by depth')
        
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
                                   color=[0.7, 0.7, 0.7], fontsize=10, weight='bold')
                            
                        except Exception as e:
                            # Fallback to simple stats
                            ax.text(0.5, 0.5, 'Fit failed', 
                                   ha='center', va='center', transform=ax.transAxes)
                            
                    except ImportError:
                        ax.text(0.5, 0.5, 'SciPy required\nfor fitting', 
                               ha='center', va='center', transform=ax.transAxes)
                    
                    ax.set_xlabel('count')
                    ax.set_ylabel('amplitude')
                    
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
                    ha='center', va='center', transform=ax.transAxes)
                    
        ax.set_title('Amplitude distribution')
        
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
                ('presenceRatio', f"Presence: {format_metric(metrics.get('presenceRatio'), 3)}")
            ]
        elif plot_type == 'amplitude_fit':
            metric_info = [
                ('percentageSpikesMissing_gaussian', f"% missing: {format_metric(metrics.get('percentageSpikesMissing_gaussian'), 1)}%")
            ]
            
        # Add colored text to plot with proper spacing
        if metric_info:
            y_start = 0.95
            line_height = 0.08  # Increased spacing to prevent overlaps
            
            for i, (metric_name, text) in enumerate(metric_info):
                color = get_metric_color(metric_name, format_metric(metrics.get(metric_name)), self.param)
                y_pos = y_start - i * line_height
                
                # Skip if text would go below plot area
                if y_pos < 0.05:
                    break
                    
                ax.text(0.98, y_pos, text, transform=ax.transAxes, 
                       verticalalignment='top', horizontalalignment='right', fontsize=8, 
                       color=color, weight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor=color))
        
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
            
            # Mark all peaks with red circles (on top)
            for i, peak_idx in enumerate(peaks):
                ax.plot(peak_idx + x_offset, waveform[peak_idx] + y_offset, 'ro', markersize=6, 
                       markeredgecolor='darkred', markeredgewidth=1, zorder=10)
                # Label peak number
                ax.text(peak_idx + x_offset, waveform[peak_idx] + y_offset + amp_range*0.1, f'peak {i+1}', 
                       ha='center', va='bottom', fontsize=8, color='red', weight='bold', zorder=10)
            
            # Mark all troughs with blue circles (on top)
            for i, trough_idx in enumerate(troughs):
                ax.plot(trough_idx + x_offset, waveform[trough_idx] + y_offset, 'bo', markersize=6,
                       markeredgecolor='darkblue', markeredgewidth=1, zorder=10)
                # Label trough number
                ax.text(trough_idx + x_offset, waveform[trough_idx] + y_offset - amp_range*0.1, f'trough {i+1}', 
                       ha='center', va='top', fontsize=8, color='blue', weight='bold', zorder=10)
            
            # Draw horizontal duration line from main peak to main trough
            if len(peaks) > 0 and len(troughs) > 0:
                # Find main peak (highest amplitude)
                main_peak_idx = peaks[np.argmax(waveform[peaks])]
                # Find main trough (most negative)
                main_trough_idx = troughs[np.argmin(waveform[troughs])]
                
                # Draw horizontal line at a fixed y-position below the waveform (on top)
                line_y = y_offset - amp_range * 0.3  # Position line below waveform
                ax.plot([main_peak_idx + x_offset, main_trough_idx + x_offset], 
                       [line_y, line_y], 
                       'g-', linewidth=2, alpha=0.7, zorder=10)
                
                # Add vertical lines to connect to peak and trough (on top)
                ax.plot([main_peak_idx + x_offset, main_peak_idx + x_offset], 
                       [waveform[main_peak_idx] + y_offset, line_y], 
                       'g--', linewidth=1, alpha=0.5, zorder=10)
                ax.plot([main_trough_idx + x_offset, main_trough_idx + x_offset], 
                       [waveform[main_trough_idx] + y_offset, line_y], 
                       'g--', linewidth=1, alpha=0.5, zorder=10)
                
                # Mark main peak and trough with larger markers (on top)
                ax.plot(main_peak_idx + x_offset, waveform[main_peak_idx] + y_offset, 'ro', 
                       markersize=8, markeredgecolor='darkred', markeredgewidth=2, zorder=11)
                ax.plot(main_trough_idx + x_offset, waveform[main_trough_idx] + y_offset, 'bo', 
                       markersize=8, markeredgecolor='darkblue', markeredgewidth=2, zorder=11)
            
        except ImportError:
            # Fallback: simple peak/trough detection without scipy
            max_idx = np.argmax(waveform)
            min_idx = np.argmin(waveform)
            
            # Mark main peak and trough
            ax.plot(max_idx + x_offset, waveform[max_idx] + y_offset, 'ro', markersize=6)
            ax.plot(min_idx + x_offset, waveform[min_idx] + y_offset, 'bo', markersize=6)
            
            # Draw horizontal duration line
            line_y = y_offset - amp_range * 0.3
            ax.plot([max_idx + x_offset, min_idx + x_offset], 
                   [line_y, line_y], 
                   'g-', linewidth=2, alpha=0.7)
            
            # Add vertical lines to connect to peak and trough
            ax.plot([max_idx + x_offset, max_idx + x_offset], 
                   [waveform[max_idx] + y_offset, line_y], 
                   'g--', linewidth=1, alpha=0.5)
            ax.plot([min_idx + x_offset, min_idx + x_offset], 
                   [waveform[min_idx] + y_offset, line_y], 
                   'g--', linewidth=1, alpha=0.5)
    
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
        self.ax_location.set_title('Location on probe')
        self.ax_location.set_xlabel('Norm. log rate')
        self.ax_location.set_ylabel('Depth from tip (μm)')
        self.ax_location.invert_yaxis()  # MATLAB uses 'YDir', 'reverse'
        
    def setup_template_plot(self):
        """Setup template waveform plot"""
        self.ax_template.set_title('Template waveforms')
        self.ax_template.set_xticks([])
        self.ax_template.set_yticks([])
        self.ax_template.invert_yaxis()
        
    def setup_raw_plot(self):
        """Setup raw waveform plot"""
        self.ax_raw.set_title('Raw waveforms')
        self.ax_raw.set_xticks([])
        self.ax_raw.set_yticks([])
        self.ax_raw.invert_yaxis()
        
    def setup_spatial_plot(self):
        """Setup spatial decay plot"""
        self.ax_spatial.set_title('Spatial decay')
        self.ax_spatial.set_ylabel('Ampli. (a.u.)')
        self.ax_spatial.set_xlabel('Distance')
        
    def setup_acg_plot(self):
        """Setup auto-correlogram plot"""
        self.ax_acg.set_title('Auto-correlogram')
        self.ax_acg.set_xlabel('Time (ms)')
        self.ax_acg.set_ylabel('sp/s')
        
    def setup_amplitude_plot(self):
        """Setup amplitude over time plot"""
        self.ax_amplitude.set_title('Amplitudes over time')
        self.ax_amplitude.set_xlabel('Experiment time (s)')
        # Dual y-axis like MATLAB
        self.ax_amplitude_right = self.ax_amplitude.twinx()
        self.ax_amplitude.set_ylabel('Template scaling', color='k')
        self.ax_amplitude_right.set_ylabel('Firing rate (sp/sec)', color='orange')
        
    def setup_amplitude_fit_plot(self):
        """Setup amplitude fit plot"""
        self.ax_amp_fit.set_title('Amplitude fit')
        
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
                
                # Plot on right y-axis (orange, like MATLAB)
                self.ax_amplitude_right.stairs(firing_rates, time_bins, color='orange', linewidth=2)
                
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
        self.ax_amp_fit.plot(fit_x, fit_y, 'orange', linewidth=2)
        
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
    
    return {
        'ephys_data': ephys_data,
        'quality_metrics': quality_metrics,
        'ephys_properties': ephys_properties,
        'raw_waveforms': raw_waveforms,
        'param': param or {}
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