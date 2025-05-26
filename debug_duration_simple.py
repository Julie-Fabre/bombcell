#!/usr/bin/env python3
"""
Simple debug script to test duration line with mock data
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/jf5479/Dropbox/Python/bombcell/pyBombCell')

def test_duration_line_logic():
    """Test the duration line logic directly"""
    
    from bombcell.unit_quality_gui import InteractiveUnitQualityGUI
    
    print("🔧 Testing duration line logic with mock data...")
    
    # Create simple mock data that should work
    n_units = 3
    n_samples = 82
    n_channels = 10
    
    # Create mock ephys data
    spike_clusters = np.random.randint(0, n_units, 1000)
    ephys_data = {
        'spike_times': np.random.rand(1000) * 1000,
        'spike_templates': np.random.randint(0, n_units, 1000),
        'spike_clusters': spike_clusters,
        'template_waveforms': np.random.randn(n_units, n_samples, n_channels) * 100,
        'template_amplitudes': np.random.rand(1000) * 200,
        'pc_features': np.random.randn(1000, 3, 12),
        'pc_features_idx': np.random.randint(0, n_channels, (n_units, 12)),
        'channel_positions': np.column_stack([np.arange(n_channels) * 20, np.arange(n_channels) * 30])
    }
    
    # Create a realistic waveform for unit 0
    waveform = np.zeros(n_samples)
    # Add a clear peak at sample 30 and trough at sample 45
    waveform[28:33] = [0.2, 0.5, 1.0, 0.5, 0.2]  # Peak around sample 30
    waveform[43:48] = [-0.3, -0.8, -1.5, -0.8, -0.3]  # Trough around sample 45
    # Add some noise
    waveform += np.random.normal(0, 0.1, n_samples)
    
    # Set this waveform for unit 0, channel 5
    ephys_data['template_waveforms'][0, :, 5] = waveform
    
    # Create mock quality metrics
    quality_metrics = {
        'nPeaks': [2, 1, 3],
        'nTroughs': [1, 2, 1], 
        'waveformDuration_peakTrough': [500, 800, 300],
        'spatialDecaySlope': [0.01, 0.02, 0.005],
        'rawAmplitude': [150, 120, 180],
        'signalToNoiseRatio': [5.2, 4.1, 6.8],
        'maxChannels': [5, 3, 7]
    }
    
    # Create simple param dict
    param = {
        'ephysKilosortPath': '/tmp',
        'minThreshDetectPeaksTroughs': 0.2
    }
    
    print("📊 Creating GUI with mock data...")
    
    try:
        # Create GUI instance
        gui = InteractiveUnitQualityGUI(ephys_data, quality_metrics, param=param)
        
        print("✅ GUI created successfully!")
        
        # Test plotting unit 0
        print("\n🔍 Testing duration line for unit 0...")
        print(f"Waveform shape: {waveform.shape}")
        print(f"Waveform min: {np.min(waveform):.3f}, max: {np.max(waveform):.3f}")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Get unit data for unit 0
        unit_data = gui.get_unit_data(0)
        
        print("🎨 Plotting template waveform...")
        gui.plot_template_waveform(ax, unit_data)
        
        plt.title('Duration Line Debug Test - Unit 0 (Mock Data)', fontsize=14)
        plt.tight_layout()
        plt.savefig('/home/jf5479/Dropbox/Python/bombcell/debug_duration_mock.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("📸 Debug plot saved as debug_duration_mock.png")
        print("\n📋 Check console output above for DEBUG messages!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Simple Duration Line Debug Test")
    print("=" * 40)
    
    success = test_duration_line_logic()
    
    if success:
        print("\n✅ Debug test completed!")
    else:
        print("\n❌ Debug test failed!")