# BombCell

Python port of BombCell. Automated quality control, curation and neuron classification of spike-sorted electrophysiology data.

## Installation

```bash
pip install bombcell
```

For full functionality including optional features:

```bash
pip install bombcell[all]
```

## Quick Start

```python
import bombcell as bc

# Set up parameters
param = bc.get_default_parameters("path/to/kilosort_output")

# Run quality metrics
quality_metrics, param, unit_type, unit_type_string = bc.run_bombcell(
    "path/to/kilosort_output", 
    "path/to/bombcell_output", 
    param
)

# Launch GUI for manual inspection
gui = bc.unit_quality_gui(
    ks_dir="path/to/kilosort_output",
    quality_metrics=quality_metrics,
    unit_types=unit_type,
    param=param,
    save_path="path/to/bombcell_output"
)
```

## Features

- Automated quality control for spike-sorted data
- Interactive GUI for manual curation
- Cell type classification (cortical and striatal neurons)
- Comprehensive quality metrics computation
- Parameter optimization tools

## Documentation

See the demo notebooks in the `Demos/` directory for detailed examples.

## License

GPL-3.0 License