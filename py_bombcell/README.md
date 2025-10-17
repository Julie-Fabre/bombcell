# Bombcell

Python port of BombCell. Automated quality control, curation and neuron classification of spike-sorted electrophysiology data.

## Installation

To install the latest stable version:
```bash
conda create -n bombcell python=3.11
conda activate bombcell
pip install uv
uv pip install bombcell # you could do `pip install .`, but uv is much quicker!
```
To install the dev version (with the latest updates): 
```bash
conda create -n bombcell python=3.11
conda activate bombcell
git clone https://github.com/Julie-Fabre/bombcell.git
cd bombcell/pyBombCell
pip install uv
uv pip install -e .
```

## Quick Start

See the [demo script](https://github.com/Julie-Fabre/bombcell/blob/main/py_bombcell/demos/BC_demo.ipynb) 

## Features

- Automated quality control for spike-sorted data
- Interactive GUI for manual curation
- Cell type classification (cortical and striatal neurons)
- Comprehensive quality metrics computation
- Parameter optimization tools

## Documentation

See the demo notebooks in the `Demos/` directory for detailed examples, and see the [wiki](https://github.com/Julie-Fabre/bombcell/wiki) for more information

## License

GPL-3.0 License
