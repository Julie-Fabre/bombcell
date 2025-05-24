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
        
        # Navigation buttons
        self.prev_btn = widgets.Button(description='← Previous', button_style='info')
        self.next_btn = widgets.Button(description='Next →', button_style='info')
        self.goto_good_btn = widgets.Button(description='Next Good', button_style='success')
        self.goto_mua_btn = widgets.Button(description='Next MUA', button_style='warning')
        self.goto_noise_btn = widgets.Button(description='Next Noise', button_style='danger')
        
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
        self.goto_good_btn.on_click(self.goto_next_good)
        self.goto_mua_btn.on_click(self.goto_next_mua)
        self.goto_noise_btn.on_click(self.goto_next_noise)
        self.classify_good_btn.on_click(lambda b: self.classify_unit(1))
        self.classify_mua_btn.on_click(lambda b: self.classify_unit(2))
        self.classify_noise_btn.on_click(lambda b: self.classify_unit(0))
        
    def display_gui(self):
        """Display the GUI"""
        # Navigation controls
        nav_controls = widgets.HBox([
            self.prev_btn, self.next_btn, 
            widgets.Label('  |  '),
            self.goto_good_btn, self.goto_mua_btn, self.goto_noise_btn
        ])
        
        # Classification controls
        classify_controls = widgets.HBox([
            self.classify_good_btn, self.classify_mua_btn, self.classify_noise_btn
        ])
        
        # Full interface
        interface = widgets.VBox([
            self.unit_slider,
            self.unit_info,
            nav_controls,
            classify_controls,
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
            
    def goto_next_good(self, b=None):
        """Go to next good unit"""
        if self.unit_types is not None:
            for i in range(self.current_unit_idx + 1, self.n_units):
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
                    
    def goto_next_noise(self, b=None):
        """Go to next noise unit"""
        if self.unit_types is not None:
            for i in range(self.current_unit_idx + 1, self.n_units):
                if self.unit_types[i] == 0:  # Noise unit
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
            
        # Simple title with just unit number and type
        info_html = f"""
        <h3>Unit {unit_data['unit_id']} ({self.current_unit_idx+1}/{self.n_units}) - {unit_type_str}</h3>
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
            ax_spatial = plt.subplot2grid((6, 13), (2, 1), rowspan=1, colspan=5)
            self.plot_spatial_decay(ax_spatial, unit_data)
            
            # 5. ACG - subplot(6, 13, 35:39)
            ax_acg = plt.subplot2grid((6, 13), (2, 7), rowspan=1, colspan=5)
            self.plot_autocorrelogram(ax_acg, unit_data)
            
            # 6. Amplitudes over time - subplot(6, 13, [42:44, 55:57, 68:70])
            ax_amplitude = plt.subplot2grid((6, 13), (3, 1), rowspan=3, colspan=9)
            self.plot_amplitudes_over_time(ax_amplitude, unit_data)
            
            # 7. Amplitude fit - subplot(6, 13, [45:46, 58:59, 71:72])
            ax_amp_fit = plt.subplot2grid((6, 13), (3, 11), rowspan=3, colspan=2)
            self.plot_amplitude_fit(ax_amp_fit, unit_data)
            
            plt.tight_layout()
            plt.show()
            
    def plot_template_waveform(self, ax, unit_data):
        """Plot template waveform with 16 nearest channels like MATLAB"""
        template = unit_data['template']
        metrics = unit_data['metrics']
        
        if template.size > 0 and len(template.shape) > 1:
            # Get peak channel
            max_ch = int(metrics.get('maxChannels', 0))
            n_channels = template.shape[1]
            
            # Find 16 nearest channels to peak channel
            channels_to_plot = self.get_nearest_channels(max_ch, n_channels, 16)
            
            # Calculate layout for channels (4x4 grid)
            n_cols = 4
            n_rows = 4
            
            # Get amplitude scaling
            all_amps = []
            for ch in channels_to_plot:
                if ch < n_channels:
                    all_amps.extend(template[:, ch])
            amp_range = np.max(all_amps) - np.min(all_amps) if all_amps else 1
            
            for i, ch in enumerate(channels_to_plot):
                if ch < n_channels and i < 16:
                    row = i // n_cols
                    col = i % n_cols
                    
                    # Calculate position offset
                    x_offset = col * 100
                    y_offset = row * amp_range * 1.2
                    
                    waveform = template[:, ch]
                    x_vals = np.arange(len(waveform)) + x_offset
                    
                    # Plot waveform
                    if ch == max_ch:
                        ax.plot(x_vals, waveform + y_offset, 'k-', linewidth=2)
                    else:
                        ax.plot(x_vals, waveform + y_offset, 'gray', linewidth=1, alpha=0.7)
                    
                    # Add channel number
                    ax.text(x_offset - 5, y_offset, f'{ch}', fontsize=8, ha='right', va='center')
                    
            # Mark peaks and troughs on peak channel only
            if 'nPeaks' in metrics and 'nTroughs' in metrics:
                peak_waveform = template[:, max_ch]
                # Find peak channel position in grid
                peak_idx = channels_to_plot.index(max_ch) if max_ch in channels_to_plot else 0
                peak_row = peak_idx // n_cols
                peak_col = peak_idx % n_cols
                peak_x_offset = peak_col * 100
                peak_y_offset = peak_row * amp_range * 1.2
                
                try:
                    from scipy.signal import find_peaks
                    peaks, _ = find_peaks(peak_waveform, height=np.max(peak_waveform)*0.3)
                    troughs, _ = find_peaks(-peak_waveform, height=-np.min(peak_waveform)*0.3)
                    
                    for peak in peaks:
                        ax.plot(peak + peak_x_offset, peak_waveform[peak] + peak_y_offset, 'ro', markersize=4)
                    for trough in troughs:
                        ax.plot(trough + peak_x_offset, peak_waveform[trough] + peak_y_offset, 'bo', markersize=4)
                except ImportError:
                    pass
                    
        ax.set_title('Template waveforms')
        ax.set_xticks([])
        ax.set_yticks([])
        # Remove aspect ratio constraint to prevent squishing
        
        # Add quality metrics text
        self.add_metrics_text(ax, unit_data, 'template')
        
    def plot_raw_waveforms(self, ax, unit_data):
        """Plot raw waveforms with 16 nearest channels like MATLAB"""
        metrics = unit_data['metrics']
        
        if self.raw_waveforms is not None:
            raw_wf = self.raw_waveforms.get('average', None)
            if raw_wf is not None:
                try:
                    if hasattr(raw_wf, '__len__') and self.current_unit_idx < len(raw_wf):
                        waveforms = raw_wf[self.current_unit_idx]
                        
                        if hasattr(waveforms, 'shape') and len(waveforms.shape) > 1:
                            # Multi-channel raw waveforms
                            max_ch = int(metrics.get('maxChannels', 0))
                            n_channels = waveforms.shape[1]
                            
                            # Find 16 nearest channels
                            channels_to_plot = self.get_nearest_channels(max_ch, n_channels, 16)
                            
                            # Calculate layout for channels (4x4 grid)
                            n_cols = 4
                            n_rows = 4
                            
                            # Get amplitude scaling
                            all_amps = []
                            for ch in channels_to_plot:
                                if ch < n_channels:
                                    all_amps.extend(waveforms[:, ch])
                            amp_range = np.max(all_amps) - np.min(all_amps) if all_amps else 1
                            
                            for i, ch in enumerate(channels_to_plot):
                                if ch < n_channels and i < 16:
                                    row = i // n_cols
                                    col = i % n_cols
                                    
                                    # Calculate position offset
                                    x_offset = col * 150
                                    y_offset = row * amp_range * 1.2
                                    
                                    waveform = waveforms[:, ch]
                                    x_vals = np.arange(len(waveform)) + x_offset
                                    
                                    # Plot waveform
                                    if ch == max_ch:
                                        ax.plot(x_vals, waveform + y_offset, 'b-', linewidth=1.5, alpha=0.8)
                                    else:
                                        ax.plot(x_vals, waveform + y_offset, 'lightblue', linewidth=1, alpha=0.6)
                                    
                                    # Add channel number
                                    ax.text(x_offset - 5, y_offset, f'{ch}', fontsize=8, ha='right', va='center')
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
        """Plot autocorrelogram"""
        spike_times = unit_data['spike_times']
        if len(spike_times) > 1:
            # Simple autocorrelogram calculation
            max_lag = 0.05  # 50ms
            bin_size = 0.001  # 1ms bins
            
            # Calculate ISIs
            isis = np.diff(spike_times)
            isis = isis[isis <= max_lag]
            
            if len(isis) > 0:
                bins = np.arange(0, max_lag + bin_size, bin_size)
                hist, _ = np.histogram(isis, bins=bins)
                bin_centers = bins[:-1] + bin_size/2
                ax.bar(bin_centers * 1000, hist, width=bin_size*1000*0.8, color='blue', alpha=0.7)
                
        ax.set_title('Auto-correlogram')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Count')
        
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
                                
                                # Plot fitted line
                                x_smooth = np.linspace(0, np.max(distances), 50)
                                y_smooth = np.exp(np.polyval(coeffs, x_smooth))
                                ax.plot(x_smooth, y_smooth, 'r-', linewidth=2, alpha=0.8)
                            
                            ax.set_xlabel('Distance (μm)')
                            ax.set_ylabel('Normalized amplitude')
                            ax.set_ylim([0, 1.1])
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
        """Plot amplitudes over time"""
        spike_times = unit_data['spike_times']
        if len(spike_times) > 0:
            # Get amplitudes if available
            unit_id = unit_data['unit_id']
            spike_mask = self.ephys_data['spike_clusters'] == unit_id
            
            if 'template_amplitudes' in self.ephys_data:
                amplitudes = self.ephys_data['template_amplitudes'][spike_mask]
                ax.scatter(spike_times, amplitudes, s=1, alpha=0.5, color='blue')
                ax.set_ylabel('Amplitude')
            else:
                # Just plot spike times as raster
                y_pos = np.ones_like(spike_times)
                ax.scatter(spike_times, y_pos, s=1, alpha=0.5, color='blue')
                ax.set_ylabel('Spikes')
                
        ax.set_title('Amplitudes over time')
        ax.set_xlabel('Time (s)')
        
        # Add quality metrics text
        self.add_metrics_text(ax, unit_data, 'amplitude')
        
    def plot_unit_location(self, ax, unit_data):
        """Plot unit location on probe"""
        # Get channel positions if available
        if 'channel_positions' in self.ephys_data:
            positions = self.ephys_data['channel_positions']
            # Plot probe outline
            if len(positions) > 0:
                ax.scatter(positions[:, 0], positions[:, 1], c='lightgray', s=20, alpha=0.5)
                
                # Highlight current unit's channel
                if 'maxChannels' in self.quality_metrics:
                    max_ch = self.quality_metrics['maxChannels'][self.current_unit_idx]
                    if max_ch < len(positions):
                        ax.scatter(positions[int(max_ch), 0], positions[int(max_ch), 1], 
                                 c='red', s=50, marker='o')
        else:
            ax.text(0.5, 0.5, 'Unit location\n(requires probe geometry)', 
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Location on probe')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.invert_yaxis()  # MATLAB style
        
    def plot_amplitude_fit(self, ax, unit_data):
        """Plot amplitude fit"""
        # Placeholder for amplitude fit analysis
        ax.text(0.5, 0.5, 'Amplitude fit\n(analysis needed)', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Amplitude fit')
        
        # Add quality metrics text
        self.add_metrics_text(ax, unit_data, 'amplitude')
        
    def add_metrics_text(self, ax, unit_data, plot_type):
        """Add quality metrics text overlay to plots like MATLAB"""
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
        
        # Different metrics for different plot types
        if plot_type == 'template':
            text_lines = [
                f"nPeaks: {format_metric(metrics.get('nPeaks'), 0)}",
                f"nTroughs: {format_metric(metrics.get('nTroughs'), 0)}",
                f"Duration: {format_metric(metrics.get('waveformDuration_peakTrough'), 1)} ms",
                f"Main P/T: {format_metric(metrics.get('mainPeakToTroughRatio'), 2)}"
            ]
        elif plot_type == 'raw':
            text_lines = [
                f"Raw Ampl: {format_metric(metrics.get('rawAmplitude'), 1)} μV",
                f"SNR: {format_metric(metrics.get('signalToNoiseRatio'), 1)}",
                f"Baseline: {format_metric(metrics.get('waveformBaselineFlatness'), 3)}"
            ]
        elif plot_type == 'spatial':
            text_lines = [
                f"Spatial decay: {format_metric(metrics.get('spatialDecaySlope'), 3)}",
                f"Max channel: {format_metric(metrics.get('maxChannels'), 0)}"
            ]
        elif plot_type == 'acg':
            text_lines = [
                f"Frac RPVs: {format_metric(metrics.get('fractionRPVs_estimatedTauR'), 4)}",
                f"Presence: {format_metric(metrics.get('presenceRatio'), 3)}"
            ]
        elif plot_type == 'amplitude':
            text_lines = [
                f"Max drift: {format_metric(metrics.get('maxDriftEstimate'), 1)} μm",
                f"% missing: {format_metric(metrics.get('percentageSpikesMissing_gaussian'), 1)}%"
            ]
        else:
            text_lines = []
            
        # Add text to plot
        if text_lines:
            text_str = '\n'.join(text_lines)
            ax.text(0.02, 0.98, text_str, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=8, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    def get_nearest_channels(self, peak_channel, n_channels, n_to_get=16):
        """Get nearest channels to peak channel like MATLAB"""
        if 'channel_positions' in self.ephys_data:
            positions = self.ephys_data['channel_positions']
            if len(positions) > peak_channel:
                peak_pos = positions[peak_channel]
                
                # Calculate distances to all channels
                distances = []
                for ch in range(min(n_channels, len(positions))):
                    dist = np.sqrt(np.sum((positions[ch] - peak_pos)**2))
                    distances.append((dist, ch))
                
                # Sort by distance and take nearest n_to_get
                distances.sort()
                nearest_channels = [ch for _, ch in distances[:n_to_get]]
                return nearest_channels
        
        # Fallback: just take channels around peak
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
        self.ax_spatial = plt.subplot2grid((6, 13), (2, 1), rowspan=1, colspan=5)
        self.setup_spatial_plot()
        
        # 5. ACG - subplot(6, 13, 35:39)
        self.ax_acg = plt.subplot2grid((6, 13), (2, 7), rowspan=1, colspan=5)
        self.setup_acg_plot()
        
        # 6. Amplitudes over time - subplot(6, 13, [55:57, 68:70, 74:76])
        self.ax_amplitude = plt.subplot2grid((6, 13), (3, 1), rowspan=3, colspan=9)
        self.setup_amplitude_plot()
        
        # 7. Amplitude fit - subplot(6, 13, [78])
        self.ax_amp_fit = plt.subplot2grid((6, 13), (3, 11), rowspan=3, colspan=2)
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
            self.ax_acg.bar(bin_centers, hist, width=bin_size*0.8, color='blue', alpha=0.7)
            
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