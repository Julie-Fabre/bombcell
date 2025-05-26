#!/usr/bin/env python3
"""
Debug script specifically for the duration line issue
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('/home/jf5479/Dropbox/Python/bombcell/pyBombCell')

def test_duration_display():
    """Test the GUI duration display with actual data"""
    
    # Load some real data to test with
    from bombcell.loading_utils import load_ephys_data, load_bc_results
    
    data_dir = "/home/jf5479/Dropbox/Python/bombcell/pyBombCell/Demos/toy_data"
    save_dir = "/home/jf5479/Dropbox/Python/bombcell/pyBombCell/Demos/toy_data/bombcell"
    
    print("📊 Loading real ephys data...")
    try:
        # Load data
        ephys_data_tuple = load_ephys_data(data_dir)
        param, quality_metrics, fractions_RPVs_all_taur = load_bc_results(save_dir)
        
        # Convert to dict format expected by GUI
        ephys_data = {
            'spike_times': ephys_data_tuple[0],
            'spike_templates': ephys_data_tuple[1], 
            'spike_clusters': ephys_data_tuple[0],  # Use spike times as placeholder
            'template_waveforms': ephys_data_tuple[2],
            'template_amplitudes': ephys_data_tuple[3],
            'pc_features': ephys_data_tuple[4],
            'pc_features_idx': ephys_data_tuple[5],
            'channel_positions': ephys_data_tuple[6]
        }
        
        print("✅ Data loaded successfully!")
        
        # Create GUI instance
        from bombcell.unit_quality_gui import UnitQualityGUI
        
        print("🚀 Creating GUI instance...")
        gui = UnitQualityGUI(ephys_data, quality_metrics, param=param)
        
        print("✅ GUI created successfully!")
        
        # Test plotting unit 0 with debug output
        print("\n🔍 Testing duration line for unit 0...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # Get unit data
        unit_data = gui.get_unit_data(0)
        
        print(f"Unit 0 data keys: {list(unit_data.keys())}")
        print(f"Template shape: {unit_data['template'].shape}")
        
        # Test the template waveform plotting (which includes duration line)
        gui.plot_template_waveform(ax, unit_data)
        
        plt.title('Duration Line Debug Test - Unit 0')
        plt.tight_layout()
        plt.savefig('/home/jf5479/Dropbox/Python/bombcell/debug_duration_unit0.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("📸 Debug plot saved as debug_duration_unit0.png")
        
        # Test a few more units
        for unit_id in [1, 2, 7]:
            print(f"\n🔍 Testing unit {unit_id}...")
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            unit_data = gui.get_unit_data(unit_id)
            gui.plot_template_waveform(ax, unit_data)
            plt.title(f'Duration Line Debug Test - Unit {unit_id}')
            plt.tight_layout()
            plt.savefig(f'/home/jf5479/Dropbox/Python/bombcell/debug_duration_unit{unit_id}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"📸 Debug plot saved as debug_duration_unit{unit_id}.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 Debugging Duration Line Display Issue")
    print("=" * 50)
    
    success = test_duration_display()
    
    if success:
        print("\n✅ Debug test completed!")
        print("📁 Check the debug_duration_unit*.png files")
        print("📋 Look at console output for DEBUG messages")
    else:
        print("\n❌ Debug test failed!")