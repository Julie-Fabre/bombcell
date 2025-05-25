#!/usr/bin/env python3
"""
Check the peak/trough distribution from the latest results
"""
import pandas as pd
import numpy as np
import os
from collections import Counter

def check_results():
    # Load the parquet file with quality metrics
    results_dir = "/home/jf5479/Dropbox/Python/bombcell/pyBombCell/Demos/toy_data/bombcell_new_test"
    
    # Check what files are available
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        print(f"Available files in {results_dir}:")
        for file in files:
            print(f"  - {file}")
        
        # Look for GUI data file
        gui_dir = os.path.join(results_dir, "for_GUI")
        if os.path.exists(gui_dir):
            gui_files = os.listdir(gui_dir)
            print(f"\nGUI files in {gui_dir}:")
            for file in gui_files:
                print(f"  - {file}")
            
            # Check if GUI data was generated
            gui_data_file = os.path.join(gui_dir, "gui_data.pkl")
            if os.path.exists(gui_data_file):
                import pickle
                print(f"\n📊 Loading GUI data from {gui_data_file}")
                try:
                    with open(gui_data_file, 'rb') as f:
                        gui_data = pickle.load(f)
                    
                    print(f"GUI data keys: {list(gui_data.keys())}")
                    
                    if 'n_peaks' in gui_data and 'n_troughs' in gui_data:
                        n_peaks = gui_data['n_peaks']
                        n_troughs = gui_data['n_troughs']
                        
                        print(f"\n🔍 Peak/Trough Analysis:")
                        print(f"Number of units: {len(n_peaks)}")
                        print(f"Peak counts: {list(n_peaks)}")
                        print(f"Trough counts: {list(n_troughs)}")
                        
                        # Check for variation
                        peak_counts = Counter(n_peaks)
                        trough_counts = Counter(n_troughs)
                        
                        print(f"\n📊 Peak/Trough Distribution:")
                        print(f"   Peaks:")
                        for count, freq in sorted(peak_counts.items()):
                            print(f"     {count} peaks: {freq} units ({freq/len(n_peaks)*100:.1f}%)")
                        
                        print(f"   Troughs:")
                        for count, freq in sorted(trough_counts.items()):
                            print(f"     {count} troughs: {freq} units ({freq/len(n_troughs)*100:.1f}%)")
                        
                        # Check if all units have the same counts
                        if len(set(n_peaks)) == 1:
                            print(f"\n⚠️  WARNING: All units have exactly {n_peaks[0]} peaks!")
                        else:
                            print(f"\n✅ Peak counts show natural variation: {set(n_peaks)}")
                            
                        if len(set(n_troughs)) == 1:
                            print(f"⚠️  WARNING: All units have exactly {n_troughs[0]} troughs!")
                        else:
                            print(f"✅ Trough counts show natural variation: {set(n_troughs)}")
                    else:
                        print("❌ GUI data doesn't contain peak/trough information")
                        
                except Exception as e:
                    print(f"❌ Error loading GUI data: {e}")
            else:
                print(f"❌ GUI data file not found: {gui_data_file}")
        else:
            print("❌ GUI directory not found")
    else:
        print(f"❌ Results directory not found: {results_dir}")

if __name__ == "__main__":
    check_results()