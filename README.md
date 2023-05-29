# BombCell: find bombshell cells!

<img align="left" src="https://github.com/Julie-Fabre/bombcell/blob/master/images/bombcell_logo_crop_small_flame.png" width=50% height=50%>

Bombcell is a powerful toolbox designed to evaluate the quality of recorded units and extract essential electrophysiological properties. It is specifically tailored for units recorded with Neuropixel probes (3A, 1.0, and 2.0) using SpikeGLX or OpenEphys and spike-sorted with Kilosort.

Please note that Bombcell is currently unpublished. If you find Bombcell useful in your work, we kindly request that you acknowledge its contribution by citing xxx.

### Bombcell wiki
A bombcell wiki can be found [here](https://github.com/Julie-Fabre/bombcell/wiki)

### Quick start guide

#### Overview

Bombcell extracts relevant quality metrics to categorize units into three categories: single somatic units, multi-units, noise units and non-somatic units.

The script `bc_qualityMetrics_pipeline` provides an example workflow to get started.

#### Installation

To begin using Bombcell, clone the [repository](https://github.com/Julie-Fabre/bombcell/bombcell) and the [dependancies](#Dependancies).

If you want to compute ephys properties, change your working directory to `bombcell\ephysProperties\helpers` in matlab and run `mex -O CCGHeart.c` to able to compute fast ACGs.

If you have z-lib compressed ephys data, compressed with [mtscomp](https://github.com/int-brain-lab/mtscomp), you will additionally need the [zmat toolbox](https://uk.mathworks.com/matlabcentral/fileexchange/71434-zmat).

#### Dependancies

- https://github.com/kwikteam/npy-matlab (to load .npy data in)

### Contact us

If you run into any issues or if you have any suggestions, please raise a github issue or alternatively email [us](mailto:julie.mfabre@gmail.com).
