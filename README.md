# 🪄 prettify matlab
My one-stop shop to instantly make your MATLAB scripts and plots beautiful, publication-ready and colorblind friendly.

## 🏁 Installation

To use prettify_matlab:

- clone the repository
- add the repository folder to MATLAB's path.

prettify_matlab doesn't use any of MATLAB's add-on toolboxes and doesn't have any dependancies. 

## :triangular_flag_on_post: Features     
### Prettify plots

Run `prettify_plot;` to prettify your current figure (includes all subplots). Includes options to modify the background color, text size and homogenize x and y limits across plots. 
<img src="./images/prettify_plot.png" width="100%">

### Prettify code

Run `prettify_current_code;` to prettify your current script open in the MATLAB editor. All code editing rules are stored in the `formatRules.xml` file, where they can easily be changed.
<img src="./images/prettify_code.png" width="100%">

### Prettify colors
#### Colorblind simulator 
Run `prettify_colorblind_simulator;` to plot your current figure as it would seen with different types of color blindness.
Uses the matrices from:  
> Gustavo M. Machado, Manuel M. Oliveira, and Leandro A. F. Fernandes "A Physiologically-based Model for Simulation of Color Vision Deficiency". IEEE Transactions on Visualization and Computer Graphics. Volume 15 (2009), Number 6, November/December 2009. pp. 1291-1298.

![cb](https://github.com/Julie-Fabre/prettify_matlab/assets/29582008/6ca9b2c6-5560-45f3-8f1b-767f9ce37965)

#### Colormaps
[Coming soon] Perceptually-uniform, colorblind friendly colormaps. 
