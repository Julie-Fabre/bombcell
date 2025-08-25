"""
Example demonstrating the updated run_bombcell that always returns figures
"""

from bombcell.helper_functions import run_bombcell
from bombcell.default_parameters import get_default_parameters
import matplotlib.pyplot as plt

# Example usage:
def example_usage():
    # Define paths
    ks_dir = "/path/to/kilosort/output"  # Replace with actual path
    save_path = "/path/to/save/results"  # Replace with actual path
    
    # Get default parameters
    param = get_default_parameters(
        kilosort_path=ks_dir,
        raw_file="/path/to/raw/data.bin",  # Optional
        kilosort_version=4,
        meta_file="/path/to/meta.meta",  # Optional
    )
    
    # Example 1: Run bombcell - figures are ALWAYS returned now
    quality_metrics, param, unit_type, unit_type_string, figures = run_bombcell(
        ks_dir, save_path, param
    )
    
    # Access individual figures
    waveforms_fig = figures['waveforms_overlay']
    upset_figs = figures['upset_plots']  # List of figures
    histograms_fig = figures['histograms']
    
    # Example 2: Run bombcell and save figures to disk
    quality_metrics, param, unit_type, unit_type_string, figures = run_bombcell(
        ks_dir, save_path, param, save_figures=True
    )
    # Figures will be saved in save_path/bombcell_plots/ AND returned
    
    # You can always manipulate the returned figures
    # For example, save with different settings:
    waveforms_fig.savefig("custom_waveforms.pdf", dpi=600)
    
    # Or modify the figures:
    ax = waveforms_fig.axes[0]
    ax.set_title("Modified Title")
    
    # Save individual upset plots
    for i, fig in enumerate(upset_figs):
        fig.savefig(f"upset_plot_{i}.svg", format='svg')
    
    # Close figures when done to free memory
    plt.close('all')
    
    return quality_metrics, figures


if __name__ == "__main__":
    print("This is an example script showing how run_bombcell now ALWAYS returns figures.")
    print("Replace the paths with actual data paths before running.")
    # Uncomment the line below to run the example:
    # quality_metrics, figures = example_usage()