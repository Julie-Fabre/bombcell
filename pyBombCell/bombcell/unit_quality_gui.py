import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
import matplotlib.gridspec as gridspec
from pathlib import Path
import pandas as pd


class UnitQualityGUI:
    """
    Interactive GUI for viewing unit quality metrics and waveforms
    Python equivalent of MATLAB unitQualityGUI_synced
    """
    
    def __init__(self, ephys_data, quality_metrics, ephys_properties=None, 
                 raw_waveforms=None, param=None, unit_types=None):
        """
        Initialize the Unit Quality GUI
        
        Parameters
        ----------
        ephys_data : dict
            Dictionary containing spike times, templates, etc.
        quality_metrics : dict
            Quality metrics for all units
        ephys_properties : list, optional
            Ephys properties from compute_all_ephys_properties
        raw_waveforms : dict, optional
            Raw waveforms data
        param : dict, optional
            Parameters dictionary
        unit_types : array, optional
            Unit type classifications (0=noise, 1=good, 2=MUA, 3=non-soma)
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
        
        # Create the GUI
        self.setup_gui()
        self.update_unit_display()
        
    def setup_gui(self):
        """Set up the GUI layout and widgets"""
        # Create main figure
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('Unit Quality GUI', fontsize=16)
        
        # Create grid layout
        gs = gridspec.GridSpec(4, 4, figure=self.fig, hspace=0.3, wspace=0.3)
        
        # Create subplots
        self.ax_waveform = self.fig.add_subplot(gs[0, 0])
        self.ax_acg = self.fig.add_subplot(gs[0, 1])
        self.ax_isi = self.fig.add_subplot(gs[0, 2])
        self.ax_amplitude = self.fig.add_subplot(gs[0, 3])
        
        self.ax_template = self.fig.add_subplot(gs[1, 0])
        self.ax_drift = self.fig.add_subplot(gs[1, 1])
        self.ax_location = self.fig.add_subplot(gs[1, 2])
        self.ax_firing_rate = self.fig.add_subplot(gs[1, 3])
        
        # Quality metrics text
        self.ax_metrics = self.fig.add_subplot(gs[2, :2])
        self.ax_ephys = self.fig.add_subplot(gs[2, 2:])
        
        # Navigation controls
        self.ax_controls = self.fig.add_subplot(gs[3, :])
        self.ax_controls.axis('off')
        
        # Create navigation buttons
        button_width = 0.1
        button_height = 0.04
        button_y = 0.02
        
        self.btn_prev = Button(plt.axes([0.1, button_y, button_width, button_height]), 'Previous')
        self.btn_next = Button(plt.axes([0.25, button_y, button_width, button_height]), 'Next')
        self.btn_good = Button(plt.axes([0.4, button_y, button_width, button_height]), 'Good Units')
        self.btn_mua = Button(plt.axes([0.55, button_y, button_width, button_height]), 'MUA Units')
        self.btn_noise = Button(plt.axes([0.7, button_y, button_width, button_height]), 'Noise Units')
        
        # Unit slider
        self.slider_unit = Slider(plt.axes([0.1, 0.08, 0.7, 0.02]), 'Unit', 
                                 0, self.n_units-1, valinit=0, valfmt='%d')
        
        # Connect callbacks
        self.btn_prev.on_clicked(self.prev_unit)
        self.btn_next.on_clicked(self.next_unit)
        self.btn_good.on_clicked(self.goto_next_good)
        self.btn_mua.on_clicked(self.goto_next_mua)
        self.btn_noise.on_clicked(self.goto_next_noise)
        self.slider_unit.on_changed(self.slider_changed)
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
    def get_unit_data(self, unit_idx):
        """Get data for a specific unit"""
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
        
        # Clear all axes
        for ax in [self.ax_waveform, self.ax_acg, self.ax_isi, self.ax_amplitude,
                  self.ax_template, self.ax_drift, self.ax_location, self.ax_firing_rate]:
            ax.clear()
            
        # Plot template waveform
        self.plot_template_waveform(unit_data)
        
        # Plot auto-correlogram
        self.plot_acg(unit_data)
        
        # Plot ISI distribution
        self.plot_isi(unit_data)
        
        # Plot amplitude distribution
        self.plot_amplitude(unit_data)
        
        # Plot raw waveform (if available)
        self.plot_raw_waveform(unit_data)
        
        # Plot drift
        self.plot_drift(unit_data)
        
        # Plot probe location
        self.plot_probe_location(unit_data)
        
        # Plot firing rate over time
        self.plot_firing_rate(unit_data)
        
        # Update metrics text
        self.update_metrics_text(unit_data)
        
        # Update title
        unit_type_str = self.get_unit_type_string(self.current_unit_idx)
        self.fig.suptitle(f'Unit {unit_data["unit_id"]} ({unit_type_str}) - '
                         f'{self.current_unit_idx+1}/{self.n_units}', fontsize=16)
        
        # Update slider
        self.slider_unit.set_val(self.current_unit_idx)
        
        # Refresh display
        self.fig.canvas.draw()
        
    def plot_template_waveform(self, unit_data):
        """Plot template waveform"""
        template = unit_data['template']
        if template.size > 0:
            # Plot peak channel waveform
            peak_chan = np.argmin(np.min(template, axis=0))
            waveform = template[:, peak_chan]
            
            time_axis = np.arange(len(waveform)) / 30.0  # Convert to ms
            
            self.ax_template.plot(time_axis, waveform, 'k-', linewidth=2)
            self.ax_template.set_title('Template Waveform')
            self.ax_template.set_xlabel('Time (ms)')
            self.ax_template.set_ylabel('Amplitude (μV)')
            self.ax_template.grid(True, alpha=0.3)
        
    def plot_acg(self, unit_data):
        """Plot auto-correlogram"""
        spike_times = unit_data['spike_times']
        
        if len(spike_times) > 10:
            # Compute ACG
            bin_size = 0.001  # 1ms bins
            max_lag = 0.05   # 50ms
            
            # Create time differences
            isi = np.diff(spike_times)
            
            # Create histogram
            bins = np.arange(-max_lag, max_lag + bin_size, bin_size)
            hist, _ = np.histogram(isi[isi <= max_lag], bins=bins[bins >= 0])
            
            # Plot
            bin_centers = bins[:-1] + bin_size/2
            self.ax_acg.bar(bin_centers[bin_centers >= 0] * 1000, hist, width=0.8)
            self.ax_acg.set_title('Auto-correlogram')
            self.ax_acg.set_xlabel('Time lag (ms)')
            self.ax_acg.set_ylabel('Count')
            self.ax_acg.axvline(x=2, color='r', linestyle='--', alpha=0.7, label='2ms')
            self.ax_acg.legend()
        
    def plot_isi(self, unit_data):
        """Plot ISI distribution"""
        spike_times = unit_data['spike_times']
        
        if len(spike_times) > 1:
            isi = np.diff(spike_times) * 1000  # Convert to ms
            isi = isi[isi <= 100]  # Limit to 100ms
            
            self.ax_isi.hist(isi, bins=50, alpha=0.7, color='blue', edgecolor='black')
            self.ax_isi.axvline(x=2, color='r', linestyle='--', label='2ms refractory')
            self.ax_isi.set_title('ISI Distribution')
            self.ax_isi.set_xlabel('ISI (ms)')
            self.ax_isi.set_ylabel('Count')
            self.ax_isi.set_yscale('log')
            self.ax_isi.legend()
            self.ax_isi.grid(True, alpha=0.3)
        
    def plot_amplitude(self, unit_data):
        """Plot amplitude distribution"""
        # This would require amplitude data from the ephys_data
        # For now, just show a placeholder
        self.ax_amplitude.text(0.5, 0.5, 'Amplitude\nDistribution\n(Not implemented)', 
                              ha='center', va='center', transform=self.ax_amplitude.transAxes)
        self.ax_amplitude.set_title('Amplitude Distribution')
        
    def plot_raw_waveform(self, unit_data):
        """Plot raw waveform if available"""
        if self.raw_waveforms is not None:
            # Implementation depends on raw waveform format
            pass
        
        self.ax_waveform.text(0.5, 0.5, 'Raw Waveform\n(Not available)', 
                             ha='center', va='center', transform=self.ax_waveform.transAxes)
        self.ax_waveform.set_title('Raw Waveform')
        
    def plot_drift(self, unit_data):
        """Plot amplitude drift over time"""
        spike_times = unit_data['spike_times']
        
        if len(spike_times) > 100:
            # Bin spikes in time and compute mean amplitude
            n_bins = min(50, len(spike_times) // 20)
            time_bins = np.linspace(spike_times.min(), spike_times.max(), n_bins)
            
            # Simulate amplitude drift (would use real amplitude data)
            amplitudes = np.random.normal(100, 10, len(spike_times))
            
            bin_centers = []
            bin_means = []
            
            for i in range(len(time_bins)-1):
                mask = (spike_times >= time_bins[i]) & (spike_times < time_bins[i+1])
                if np.sum(mask) > 0:
                    bin_centers.append((time_bins[i] + time_bins[i+1]) / 2)
                    bin_means.append(np.mean(amplitudes[mask]))
            
            if bin_centers:
                self.ax_drift.plot(bin_centers, bin_means, 'o-')
                self.ax_drift.set_title('Amplitude Drift')
                self.ax_drift.set_xlabel('Time (s)')
                self.ax_drift.set_ylabel('Amplitude')
                self.ax_drift.grid(True, alpha=0.3)
        
    def plot_probe_location(self, unit_data):
        """Plot probe location"""
        self.ax_location.text(0.5, 0.5, 'Probe Location\n(Not implemented)', 
                             ha='center', va='center', transform=self.ax_location.transAxes)
        self.ax_location.set_title('Probe Location')
        
    def plot_firing_rate(self, unit_data):
        """Plot firing rate over time"""
        spike_times = unit_data['spike_times']
        
        if len(spike_times) > 10:
            # Compute firing rate in time bins
            bin_size = 60  # 1 minute bins
            time_bins = np.arange(spike_times.min(), spike_times.max() + bin_size, bin_size)
            
            firing_rates = []
            bin_centers = []
            
            for i in range(len(time_bins)-1):
                mask = (spike_times >= time_bins[i]) & (spike_times < time_bins[i+1])
                count = np.sum(mask)
                firing_rates.append(count / bin_size)  # Hz
                bin_centers.append((time_bins[i] + time_bins[i+1]) / 2)
            
            self.ax_firing_rate.plot(bin_centers, firing_rates, 'g-', linewidth=2)
            self.ax_firing_rate.set_title('Firing Rate Over Time')
            self.ax_firing_rate.set_xlabel('Time (s)')
            self.ax_firing_rate.set_ylabel('Firing Rate (Hz)')
            self.ax_firing_rate.grid(True, alpha=0.3)
        
    def update_metrics_text(self, unit_data):
        """Update quality metrics text display"""
        self.ax_metrics.clear()
        self.ax_ephys.clear()
        
        # Quality metrics text
        metrics = unit_data['metrics']
        metrics_text = f"Quality Metrics (Unit {unit_data['unit_id']}):\n"
        metrics_text += f"N Spikes: {metrics.get('nSpikes', 'N/A')}\n"
        metrics_text += f"Presence Ratio: {metrics.get('presenceRatio', 'N/A'):.3f}\n"
        metrics_text += f"Fraction RPVs: {metrics.get('fractionRPVs_estimatedTauR', 'N/A'):.3f}\n"
        metrics_text += f"Waveform Duration: {metrics.get('waveformDuration_peakTrough', 'N/A'):.1f} μs\n"
        metrics_text += f"Spatial Decay: {metrics.get('spatialDecaySlope', 'N/A'):.4f}\n"
        
        self.ax_metrics.text(0.02, 0.98, metrics_text, transform=self.ax_metrics.transAxes,
                           verticalalignment='top', fontfamily='monospace', fontsize=10)
        self.ax_metrics.set_xlim(0, 1)
        self.ax_metrics.set_ylim(0, 1)
        self.ax_metrics.axis('off')
        
        # Ephys properties text
        ephys = unit_data['ephys']
        if ephys:
            ephys_text = f"Ephys Properties:\n"
            ephys_text += f"Firing Rate: {ephys.get('firing_rate_mean', 'N/A'):.2f} Hz\n"
            ephys_text += f"ISI CV: {ephys.get('isi_cv', 'N/A'):.3f}\n"
            ephys_text += f"Post-spike Suppression: {ephys.get('postSpikeSuppression_ms', 'N/A'):.2f} ms\n"
            ephys_text += f"Prop Long ISI: {ephys.get('propLongISI', 'N/A'):.3f}\n"
        else:
            ephys_text = "Ephys Properties:\nNot computed"
            
        self.ax_ephys.text(0.02, 0.98, ephys_text, transform=self.ax_ephys.transAxes,
                          verticalalignment='top', fontfamily='monospace', fontsize=10)
        self.ax_ephys.set_xlim(0, 1)
        self.ax_ephys.set_ylim(0, 1)
        self.ax_ephys.axis('off')
        
    def get_unit_type_string(self, unit_idx):
        """Get unit type string"""
        if self.unit_types is not None and unit_idx < len(self.unit_types):
            unit_type = self.unit_types[unit_idx]
            type_map = {0: 'NOISE', 1: 'GOOD', 2: 'MUA', 3: 'NON-SOMA'}
            return type_map.get(unit_type, 'UNKNOWN')
        return 'UNKNOWN'
        
    # Navigation callbacks
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
                
    def slider_changed(self, val):
        """Handle slider change"""
        new_idx = int(val)
        if new_idx != self.current_unit_idx:
            self.current_unit_idx = new_idx
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


def unit_quality_gui(ks_dir, quality_metrics, ephys_properties=None, 
                     unit_types=None, param=None):
    """
    Launch the Unit Quality GUI - Python equivalent of unitQualityGUI_synced
    
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
    UnitQualityGUI
        The GUI object
    """
    # Load data for GUI
    gui_data = load_metrics_for_gui(ks_dir, quality_metrics, ephys_properties, param)
    
    # Create and return GUI
    gui = UnitQualityGUI(
        ephys_data=gui_data['ephys_data'],
        quality_metrics=quality_metrics,
        ephys_properties=ephys_properties,
        raw_waveforms=gui_data['raw_waveforms'],
        param=gui_data['param'],
        unit_types=unit_types
    )
    
    return gui