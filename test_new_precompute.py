#!/usr/bin/env python3
"""
Test the new streamlined pre-computation to see what data is actually generated
"""

import sys
sys.path.append('/home/jf5479/Dropbox/Python/bombcell/pyBombCell')

import bombcell as bc
from pathlib import Path
import time

def test_new_precompute():
    print("🧪 Testing NEW streamlined pre-computation...")
    
    # Use toy data
    ks_dir = "/home/jf5479/Dropbox/Python/bombcell/pyBombCell/Demos/toy_data"
    save_path = Path(ks_dir) / "bombcell_new_test"
    
    # Clean up any existing files
    if save_path.exists():
        import shutil
        shutil.rmtree(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 KS dir: {ks_dir}")
    print(f"💾 Save dir: {save_path}")
    
    # Get parameters with verbose enabled
    param = bc.get_default_parameters(ks_dir)
    param["verbose"] = True
    
    # Disable slow computations for testing
    param["computeDistanceMetrics"] = False
    param["computeDrift"] = False
    param["extractRaw"] = False
    
    print("\n🚀 Running BombCell with NEW streamlined pre-computation...")
    
    start_time = time.time()
    
    # Run BombCell - should create comprehensive GUI data
    (
        quality_metrics,
        param_out,
        unit_type,
        unit_type_string,
    ) = bc.run_bombcell(ks_dir, save_path, param)
    
    runtime = time.time() - start_time
    
    print(f"\n✅ BombCell completed in {runtime:.2f} seconds!")
    
    # Check what GUI data was actually created
    gui_data_path = save_path / "for_GUI" / "gui_data.pkl"
    
    if gui_data_path.exists():
        print(f"\n📊 GUI data created! Analyzing contents...")
        
        gui_data = bc.load_gui_data(str(save_path))
        if gui_data:
            print(f"\n📈 Detailed analysis:")
            
            for key, data in gui_data.items():
                if isinstance(data, dict):
                    non_none_count = sum(1 for v in data.values() if v is not None)
                    print(f"   📊 {key}: {len(data)} total, {non_none_count} with data")
                    
                    # Show example for spatial decay and amplitude fits
                    if key in ['spatial_decay_fits', 'amplitude_fits'] and non_none_count > 0:
                        sample_unit = None
                        for unit, value in data.items():
                            if value is not None:
                                sample_unit = unit
                                break
                        if sample_unit is not None:
                            sample_data = data[sample_unit]
                            if isinstance(sample_data, dict):
                                print(f"      - Unit {sample_unit} example keys: {list(sample_data.keys())}")
                else:
                    print(f"   📊 {key}: {type(data).__name__}")
            
            return True
        else:
            print("❌ Failed to load created GUI data")
            return False
    else:
        print(f"❌ No GUI data created at: {gui_data_path}")
        return False

if __name__ == "__main__":
    success = test_new_precompute()
    
    if success:
        print("\n🎉 New streamlined pre-computation analysis complete!")
    else:
        print("\n💥 Issues with new pre-computation")