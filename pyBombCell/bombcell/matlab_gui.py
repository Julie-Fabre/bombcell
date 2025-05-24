import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import matplotlib.gridspec as gridspec
from pathlib import Path
import pandas as pd


class MatlabStyleGUI:
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


def matlab_style_gui(ks_dir, quality_metrics, ephys_properties=None, 
                     unit_types=None, param=None):
    """
    Launch MATLAB-style GUI
    
    Parameters
    ----------
    ks_dir : str
        Path to kilosort directory
    quality_metrics : dict
        Quality metrics from bombcell
    ephys_properties : list, optional
        Ephys properties from compute_all_ephys_properties
    unit_types : array, optional
        Unit type classifications
    param : dict, optional
        Parameters dictionary
        
    Returns
    -------
    MatlabStyleGUI
        The GUI object
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
    
    # Create and return GUI
    gui = MatlabStyleGUI(
        ephys_data=ephys_data,
        quality_metrics=quality_metrics,
        ephys_properties=ephys_properties,
        raw_waveforms=None,
        param=param or {},
        unit_types=unit_types
    )
    
    return gui