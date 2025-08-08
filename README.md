[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8172821.svg)](https://doi.org/10.5281/zenodo.8172821)
[![License](https://img.shields.io/badge/license-GPLv3-yellow)](https://github.com/Julie-Fabre/bombcell/blob/master/LICENSE)
[![View bombcell on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://uk.mathworks.com/matlabcentral/fileexchange/136619-bombcell)

# 💣 Bombcell: find bombshell cells! 💣 
<picture>
  <source media="(prefers-color-scheme: light)" srcset="./docs/images/bombcell_logo_crop.svg"  width="40%" title="bombcell" alt="bombcell" align="right" vspace = "20">
  <source media="(prefers-color-scheme: dark)" srcset="./docs/images/bombcell_logo_crop_DARK.svg" width="40%" title="bombcell" alt="bombcell" align="right" vspace = "20">
  <img alt="Shows a black logo in light color mode and a white one in dark color mode." width="25%" title="bombcell" alt="bombcell" align="right" vspace = "20">
</picture>

Manual curation of electrophysiology spike sorted units is slow, laborious, and hard to standardize and reproduce. Bombcell is a powerful toolbox that addresses this problem, evaluating the quality of recorded units and extracting essential electrophysiological properties. Bombcell can replace manual curation or can be used as a tool to aid manual curation. See [this talk](https://youtu.be/CvXUtGzkXIY?si=lHkLN885OCb4WJEv) at the annual Neuropixels course about quality control.

📢 After many requests, we now have a Python of version of bombcell! See the installation instructions below to get started! 📢

Please star the project to support us, using the top-right "⭐ Star" button.

### 📔 Bombcell wiki

Documentation and guides to using and troubleshooting bombcell can be found on the dedicated [wiki](https://github.com/Julie-Fabre/bombcell/wiki).

### 🔍️ How bombcell works

Below is a flowchart of how bombcell evaluates and classifies each unit:
<img align="center" src="./docs/images/bombcell_flowchart.png" width=100% height=100%>

### 🏁 Quick start guide

#### Overview

Bombcell extracts relevant quality metrics to categorize units into four categories: single somatic units, multi-units, noise units and non-somatic units.

Take a look at:
- the MATLAB live script [`gettingStarted`](https://github.com/Julie-Fabre/bombcell/blob/main/gettingStarted.mlx) to see an example workflow and play around with our small toy dataset.
- the Python Jupyter notebook [`BC_demo`](https://github.com/Julie-Fabre/bombcell/blob/main/py_bombcell/demos/BC_demo.ipynb)
- You can also take a look at the exercise we prepared for the 2024 Neuropixels course [here](https://github.com/BombCell/Neuropixels_course_2024). 

#### Installation
#### Python 

##### Heys lab version (hyunwoo branch): 
```bash
# Navigate to your server code folder
cd ~/codes/
# Create a conda environment
conda create -n bombcell python=3.11
conda activate bombcell
# Clone bombcell repository from github
git clone -b hyunwoo --single-branch git@github.com:heyslab/bombcell.git
cd bombcell/py_bombcell/
# Install bombcell
pip install uv
uv pip install -e .
```

##### Original Bombcell:
```bash
# Create a conda environment
conda create -n bombcell python=3.11
conda activate bombcell
# Install bombcell
pip install uv
uv pip install bombcell # you could do `pip install .`, but uv is much quicker!
```


### 🤗 Support and citing

If you find Bombcell useful in your work, we kindly request that you cite:

> Julie M.J. Fabre, Enny H. van Beest, Andrew J. Peters, Matteo Carandini, & Kenneth D. Harris. (2023). Bombcell: automated curation and cell classification of spike-sorted electrophysiology data. Zenodo. https://doi.org/10.5281/zenodo.8172821

### :page_facing_up: License

Bombcell is under the open-source [copyleft](https://www.gnu.org/licenses/copyleft.en.html) [GNU General Public License 3](https://www.gnu.org/licenses/gpl-3.0.html). You can run, study, share, and modify the software under the condition that you keep and do not modify the license.

### 📬 Contact us

If you run into any issues or if you have any suggestions, please raise a [github issue](https://github.com/Julie-Fabre/bombcell/issues) or create a [pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request). You can also use the [Neuropixels slack workgroup](https://join.slack.com/t/neuropixelsgroup/shared_invite/zt-2h3sp1nfr-JZrpKWxeVptI0EPbHAoxKA).
Please star the project to support us, using the top-right "⭐ Star" button.

