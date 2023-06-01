# 💣 BombCell: find bombshell cells! 💣

<img align="left" src="https://github.com/Julie-Fabre/bombcell/blob/master/images/bombcell_logo_crop_small_flame.png" width=50% height=50%>

Bombcell is a powerful toolbox designed to evaluate the quality of recorded units and extract essential electrophysiological properties. It is specifically tailored for units recorded with Neuropixel probes (3A, 1.0, and 2.0) using SpikeGLX or OpenEphys and spike-sorted with Kilosort.

Please note that Bombcell is currently unpublished. If you find Bombcell useful in your work, we kindly request that you acknowledge its contribution by citing xxx.

### 📖 Bombcell wiki

Thorough documentation and guides to using bombcell can be found on the dedicated [wiki](https://github.com/Julie-Fabre/bombcell/wiki)

### 🏁 Quick start guide

#### Overview

Bombcell extracts relevant quality metrics to categorize units into three categories: single somatic units, multi-units, noise units and non-somatic units.

The script [`bc_qualityMetrics_pipeline`](https://github.com/Julie-Fabre/bombcell/blob/master/bc_qualityMetrics_pipeline.m) provides an example workflow to get started.

#### Installation

To begin using Bombcell, 
- [clone](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository) the [repository](https://github.com/Julie-Fabre/bombcell/bombcell) and the [dependancies](#Dependancies).
- add bombcell's and the dependancies' folders to [MATLAB's path](https://uk.mathworks.com/help/matlab/ref/pathtool.html)
- in addition, if you want to compute ephys properties, change your working directory to `bombcell\ephysProperties\helpers` in matlab and run `mex -O CCGHeart.c` to able to compute fast ACGs.

#### Dependancies

- https://github.com/kwikteam/npy-matlab (to load .npy data in)
- If you have z-lib compressed ephys data, compressed with [mtscomp](https://github.com/int-brain-lab/mtscomp), you will additionally need the [zmat toolbox](https://uk.mathworks.com/matlabcentral/fileexchange/71434-zmat). More information about compressing ephys data [here](https://www.biorxiv.org/content/biorxiv/early/2023/05/24/2023.05.22.541700.full.pdf?%3Fcollection=). 

### 📬 Contact us

If you run into any issues or if you have any suggestions, please raise a [github issue](https://github.com/Julie-Fabre/bombcell/issues) or alternatively email [us](mailto:julie.mfabre@gmail.com).
