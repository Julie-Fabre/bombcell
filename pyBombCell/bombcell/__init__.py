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
    unit_quality_gui
)

from .default_parameters import *
from .extract_raw_waveforms import *
from .helper_functions import *
from .quality_metrics import *
from .ephys_properties import *
from .save_utils import *
from .loading_utils import *
from .plot_functions import *

# Explicitly expose key functions
from .ephys_properties import get_ephys_parameters
from .classification import classify_and_plot_brain_region
from .unit_quality_gui import unit_quality_gui, UnitQualityGUI, InteractiveUnitQualityGUI

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