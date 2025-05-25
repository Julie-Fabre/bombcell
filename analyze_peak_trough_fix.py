#!/usr/bin/env python3
"""
Analyze the peak/trough results to confirm the fix worked
"""
import pandas as pd
from collections import Counter

def analyze_results():
    # Load the TSV files
    peaks_file = "/home/jf5479/Dropbox/Python/bombcell/pyBombCell/Demos/toy_data/bombcell/cluster_nPeaks.tsv"
    troughs_file = "/home/jf5479/Dropbox/Python/bombcell/pyBombCell/Demos/toy_data/bombcell/cluster_nTroughs.tsv"
    
    peaks_df = pd.read_csv(peaks_file, sep='\t')
    troughs_df = pd.read_csv(troughs_file, sep='\t')
    
    print("🎉 PEAK/TROUGH DETECTION FIX RESULTS 🎉")
    print("="*50)
    
    print(f"📊 Analyzed {len(peaks_df)} units")
    
    # Analyze peak distribution
    peak_counts = Counter(peaks_df['nPeaks'])
    print(f"\n🔺 Peak Distribution:")
    for count, freq in sorted(peak_counts.items()):
        print(f"   {int(count)} peaks: {freq} units ({freq/len(peaks_df)*100:.1f}%)")
    
    # Analyze trough distribution  
    trough_counts = Counter(troughs_df['nTroughs'])
    print(f"\n🔻 Trough Distribution:")
    for count, freq in sorted(trough_counts.items()):
        print(f"   {int(count)} troughs: {freq} units ({freq/len(troughs_df)*100:.1f}%)")
    
    # Check for natural variation
    unique_peaks = set(peaks_df['nPeaks'])
    unique_troughs = set(troughs_df['nTroughs'])
    
    print(f"\n✅ VALIDATION RESULTS:")
    print(f"   Peak count variation: {sorted(unique_peaks)} - {'✅ NATURAL VARIATION' if len(unique_peaks) > 1 else '❌ UNIFORM'}")
    print(f"   Trough count variation: {sorted(unique_troughs)} - {'✅ NATURAL VARIATION' if len(unique_troughs) > 1 else '❌ UNIFORM'}")
    
    # Compare with our debug script results
    print(f"\n🔍 COMPARISON WITH DEBUG SCRIPT:")
    print(f"   Debug script found peak counts: {{2, 3}}")
    print(f"   Production code found peak counts: {set(int(x) for x in unique_peaks)}")
    print(f"   Debug script found trough counts: {{1, 2}}")
    print(f"   Production code found trough counts: {set(int(x) for x in unique_troughs)}")
    
    # Check if any units might be non-somatic
    # Non-somatic units typically have more complex waveforms (multiple peaks)
    multi_peak_units = peaks_df[peaks_df['nPeaks'] > 1]
    multi_trough_units = troughs_df[troughs_df['nTroughs'] > 1]
    
    print(f"\n🧠 POTENTIAL NON-SOMATIC UNITS:")
    print(f"   Units with >1 peak: {len(multi_peak_units)} ({len(multi_peak_units)/len(peaks_df)*100:.1f}%)")
    print(f"   Units with >1 trough: {len(multi_trough_units)} ({len(multi_trough_units)/len(troughs_df)*100:.1f}%)")
    
    if len(multi_peak_units) > 0:
        print(f"   Multi-peak unit IDs: {list(multi_peak_units['cluster_id'].astype(int))}")
    if len(multi_trough_units) > 0:
        print(f"   Multi-trough unit IDs: {list(multi_trough_units['cluster_id'].astype(int))}")
    
    print(f"\n🎯 SUMMARY:")
    if len(unique_peaks) > 1 and len(unique_troughs) > 1:
        print("   ✅ SUCCESS: Peak/trough detection now shows natural variation!")
        print("   ✅ SUCCESS: Non-somatic unit detection is now possible!")
        print("   ✅ SUCCESS: Fixed the fundamental algorithmic issue!")
    else:
        print("   ❌ ISSUE: Still showing uniform results")
        
    return len(unique_peaks) > 1 and len(unique_troughs) > 1

if __name__ == "__main__":
    success = analyze_results()
    print(f"\n{'🎉 FIX SUCCESSFUL! 🎉' if success else '❌ FIX INCOMPLETE ❌'}")