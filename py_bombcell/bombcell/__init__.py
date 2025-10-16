from . import (
    default_parameters,
    extract_raw_waveforms,
    quality_metrics,
    ephys_properties,
    save_utils,
    loading_utils,
    helper_functions,
    plot_functions,
    classification,
    unit_quality_gui,
    manual_analysis
)

from .default_parameters import *
from .extract_raw_waveforms import *
from .helper_functions import *
from .helper_functions import run_bombcell_unit_match
from .quality_metrics import *
from .ephys_properties import *
from .save_utils import *
from .loading_utils import *
from .plot_functions import *
from .ccg_fast import acg, ccg

# Explicitly expose key functions
from .ephys_properties import get_ephys_parameters
from .classification import classify_and_plot_brain_region
from .default_parameters import get_unit_match_parameters
from .unit_quality_gui import load_unit_quality_gui, InteractiveUnitQualityGUI, precompute_gui_data, load_gui_data
# CCG functions are in ephys_properties.py (fast_acg, compute_acg)
from .manual_analysis import (
    load_manual_classifications, 
    analyze_classification_concordance, 
    suggest_parameter_adjustments, 
    plot_classification_comparison,
    analyze_manual_vs_bombcell,
    compare_manual_vs_bombcell
)

# __version__ attribute exposition
try:
    from importlib.metadata import version
    __version__ = version(__name__)
except ImportError:
    # For Python < 3.8
    try:
        from importlib_metadata import version
        __version__ = version(__name__)
    except ImportError:
        __version__ = "unknown"